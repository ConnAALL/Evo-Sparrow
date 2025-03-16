"""
Main script for training the LSTM model using CMA-ES.
"""

import os, datetime, time, cma, torch, jax, multiprocessing
import numpy as np
from tqdm import tqdm

from data_management import create_log_file, log_run_parameters, setup_database, insert_game_result
from env_management import init_env, reset_random_seed, state_to_input, shuffled_to_normal, actions_to_hand, choose_next_move, tile_to_action
from lstm_structure import SparrowMahjongLSTM

# HYPERPARAMETERS
INPUT_SIZE = 37
OUTPUT_SIZE = 6
BATCH_SIZE = 200
ENV_ID = "sparrow_mahjong"
NUM_GENERATIONS = 50
HIDDEN_LAYER_SIZE = 32
LSTM_LAYER_COUNT = 3
FC_LAYER_COUNT = 8
HIDDEN_FC_SIZE = 32
USE_DIAGONAL = False
SIGMA = 1
CHECKPOINT_FREQ = 1
LOG_FILE = create_log_file()

# Logging string of the run parameters. 
LOG_STR = (f"RUN_START: unified_run_id - BATCH_SIZE: {BATCH_SIZE} - "
           f"HIDDEN_LAYER_SIZE: {HIDDEN_LAYER_SIZE} - "
           f"LSTM_LAYER_COUNT: {LSTM_LAYER_COUNT} - "
           f"FC_LAYER_COUNT: {FC_LAYER_COUNT} - "
           f"HIDDEN_FC_SIZE: {HIDDEN_FC_SIZE} - "
           f"USE_DIAGONAL: {USE_DIAGONAL} - "
           f"SIGMA: {SIGMA}")


def get_valid_actions(state, idx, player_idx):
    """Get the valid actions for a player, representing red Dora tiles as negative indices."""
    shuffled_hands = state._hands[idx]
    shuffled_reds = state._n_red_in_hands[idx]
    original_hands = shuffled_to_normal(state, shuffled_hands, idx)
    original_reds = shuffled_to_normal(state, shuffled_reds, idx)

    legal_action_list = state.legal_action_mask[idx].tolist()
    valid_indices = [i for i, val in enumerate(legal_action_list) if val]

    player_hand = original_hands[player_idx].tolist()
    red_tile_counts = original_reds[player_idx].tolist()
    complete_actions = []

    for action in valid_indices:
        tile_count = player_hand[action]
        red_count = red_tile_counts[action]

        # Add red Dora actions as negative indices
        for _ in range(red_count):
            complete_actions.append(-action)

        # Add regular tile actions as positive indices
        for _ in range(tile_count - red_count):
            complete_actions.append(action)

    return complete_actions


def handle_batch_player_move(model, state, key, hidden):
    """Handle moves for all games in the batch."""
    key, subkey = jax.random.split(key)
    picked_actions = np.zeros(BATCH_SIZE, dtype=int)
    active_games = ~(state.terminated | state.truncated)
    
    # Determine which games are player 1's turns (our LSTM agent)
    player1_turns = (state.current_player == 0) & active_games

    # Process player 1's moves
    indices_player1 = np.where(player1_turns)[0]
    if len(indices_player1) > 0:
        input_tensors = []
        valid_moves_list = []
        for idx in indices_player1:
            valid_moves = get_valid_actions(state, idx, 0)
            valid_moves_list.append(valid_moves)
            input_array = state_to_input(state, idx)
            input_tensor = torch.tensor(input_array, dtype=torch.float32)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensors.append(input_tensor)
        input_batch = torch.stack(input_tensors, dim=0)
        h0 = hidden[0][:, indices_player1, :]
        c0 = hidden[1][:, indices_player1, :]
        device = next(model.parameters()).device
        input_batch = input_batch.to(device)
        h0 = h0.to(device)
        c0 = c0.to(device)
        output, (h_n, c_n) = model(input_batch, (h0, c0))
        max_indices = output.argmax(dim=1)
        for i, idx in enumerate(indices_player1):
            chosen = max_indices[i].item()
            valid_moves = valid_moves_list[i]
            picked_action = abs(valid_moves[chosen])
            picked_actions[idx] = picked_action
            # Update hidden state
            hidden[0][:, idx, :] = h_n[:, i, :]
            hidden[1][:, idx, :] = c_n[:, i, :]

    # Process other players' moves (rule-based agents)
    indices_other = np.where(~player1_turns & active_games)[0]
    for idx in indices_other:
        current_player = state.current_player[idx]
        actions = get_valid_actions(state, idx, current_player)
        hand = actions_to_hand(actions)
        current_dora = state._dora[idx].tolist() + 1
        picked_tile = choose_next_move(hand, current_dora)
        picked_action = tile_to_action(picked_tile)
        picked_actions[idx] = picked_action
    return picked_actions, key, hidden


def get_game_id(idx, run_id, current_generation, current_solution):
    """Creates a game_id using the unified run_id, current generation, solution, and trial index."""
    return f"{run_id}_{current_generation}_{current_solution}_{idx + 1}"


def play_games(model, run_id, current_generation, current_solution):
    """
    Plays games using the provided model.
    Instead of writing results to the database, it returns a tuple:
      (fitness_value, game_results)
    where game_results is a list of (game_id, rewards) tuples.
    """
    init, step = init_env(ENV_ID)
    KEY = reset_random_seed()
    KEY, subkey = jax.random.split(KEY)
    keys = jax.random.split(subkey, BATCH_SIZE)
    state = init(keys)
    hidden = model.init_hidden(BATCH_SIZE)
    processed_games = set()
    all_rewards = []
    game_results = []  # collect (game_id, rewards) tuples

    while not (state.terminated | state.truncated).all():
        picked_actions, KEY, hidden = handle_batch_player_move(model, state, KEY, hidden)
        state = step(state, picked_actions)

        # Check for games that have just terminated
        for i in range(BATCH_SIZE):
            if (state.terminated[i] or state.truncated[i]) and (i not in processed_games):
                game_id = get_game_id(i, run_id, current_generation, current_solution)
                rewards_list = state.rewards[i].tolist()
                all_rewards.append(rewards_list)
                game_results.append((game_id, rewards_list))
                processed_games.add(i)
    fitness_value = rewards_to_fitness(all_rewards)
    return fitness_value, game_results


def rewards_to_fitness(rewards):
    """Converts a list of rewards to a fitness value."""
    return sum(reward[0] for reward in rewards)


def evaluate_candidate(candidate_weights, candidate_index, current_generation, run_id):
    """
    Evaluates one candidate solution:
      - Creates a new LSTM model instance.
      - Sets its weights from candidate_weights.
      - Runs the game simulation (without doing database writes).
      - Returns a tuple: (candidate_index, fitness, elapsed_time, game_results)
    """
    current_solution = candidate_index + 1
    start_time = time.time()
    lstm_model = SparrowMahjongLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_LAYER_SIZE,
        output_size=OUTPUT_SIZE,
        num_lstm_layers=LSTM_LAYER_COUNT,
        num_fc_layers=FC_LAYER_COUNT,
        fc_hidden_size=HIDDEN_FC_SIZE
    )
    lstm_model.set_model_weights(candidate_weights)

    fitness_value, game_results = play_games(lstm_model, run_id, current_generation, current_solution)
    elapsed = time.time() - start_time
    return candidate_index, fitness_value, elapsed, game_results


def create_dir(path, folder_name):
    """Creates a directory if it does not exist."""
    full_path = os.path.join(path, folder_name)
    if not os.path.exists(full_path):
        os.makedirs(full_path)


def main():
    print("Setting up the Training Environment...")
    create_dir("..", "DATA")
    run_id = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    create_dir("../DATA", f"{run_id}")
    log_run_parameters(LOG_FILE, f"RUN_START: {run_id} - BATCH_SIZE: {BATCH_SIZE} - "
                                   f"HIDDEN_LAYER_SIZE: {HIDDEN_LAYER_SIZE} - "
                                   f"LSTM_LAYER_COUNT: {LSTM_LAYER_COUNT} - "
                                   f"FC_LAYER_COUNT: {FC_LAYER_COUNT} - "
                                   f"HIDDEN_FC_SIZE: {HIDDEN_FC_SIZE} - "
                                   f"USE_DIAGONAL: {USE_DIAGONAL} - "
                                   f"SIGMA: {SIGMA}")

    conn, cursor = setup_database()

    lstm_model = SparrowMahjongLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_LAYER_SIZE,
        output_size=OUTPUT_SIZE,
        num_lstm_layers=LSTM_LAYER_COUNT,
        num_fc_layers=FC_LAYER_COUNT,
        fc_hidden_size=HIDDEN_FC_SIZE
    )
    lstm_model.set_random_weights_from_model()
    initial_weights = lstm_model.get_model_weights()
    es = cma.CMAEvolutionStrategy(initial_weights, SIGMA, {'CMA_diagonal': USE_DIAGONAL})
    print("Setup Complete! Starting the training...")

    avg_time = 0.0

    # Training loop
    with tqdm(total=NUM_GENERATIONS, desc="Training Progress", unit="gen") as pbar:
        for generation in range(NUM_GENERATIONS):
            current_generation = generation + 1
            start_gen_time = time.time()

            solutions = es.ask()
            generation_fitness = [None] * len(solutions)
            generation_game_results = []

            pool = multiprocessing.Pool()
            async_results = []
            for idx, candidate in enumerate(solutions):
                async_results.append(pool.apply_async(evaluate_candidate, args=(candidate, idx, current_generation, run_id)))
            pool.close()
            pool.join()

            for async_result in async_results:
                candidate_index, fitness, elapsed, game_results = async_result.get()
                generation_fitness[candidate_index] = -fitness  # CMA-ES minimizes so use negative fitness
                generation_game_results.extend(game_results)
            
            for game_id, rewards in generation_game_results:
                insert_game_result(cursor, rewards, game_id)
            conn.commit()

            es.tell(solutions, generation_fitness)

            generation_time = time.time() - start_gen_time
            avg_time = ((avg_time * generation) + generation_time) / (generation + 1)
            pbar.set_postfix(avg_time=f"{avg_time:.2f} sec")
            pbar.update(1)

            if CHECKPOINT_FREQ > 0 and current_generation != NUM_GENERATIONS and current_generation % CHECKPOINT_FREQ == 0:
                best_weights = es.result.xbest
                lstm_model.set_model_weights(best_weights)
                checkpoint_path = f"../DATA/{run_id}/{run_id}_{current_generation}.pth"
                torch.save(lstm_model.state_dict(), checkpoint_path)

    best_weights = es.result.xbest
    lstm_model.set_model_weights(best_weights)
    torch.save(lstm_model.state_dict(), f"../DATA/{run_id}/{run_id}_FINAL_BEST.pth")
    conn.close()

if __name__ == "__main__":
    main()
