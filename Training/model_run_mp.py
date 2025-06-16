"""
Script for testing the speed of the LSTM model if it is ran in parallel. Not used for training in the paper. 
"""

import os, datetime, time, cma, torch, jax, multiprocessing
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count

from env_management import init_env, reset_random_seed, state_to_input, shuffled_to_normal, actions_to_hand, choose_next_move, tile_to_action
from lstm_structure import SparrowMahjongLSTM

# Use spawn to start subprocesses
multiprocessing.set_start_method('spawn', force=True)

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


def play_games(model):
    """
    Plays games using the provided model and returns fitness value.
    """
    init, step = init_env(ENV_ID)
    KEY = reset_random_seed()
    KEY, subkey = jax.random.split(KEY)
    keys = jax.random.split(subkey, BATCH_SIZE)
    state = init(keys)
    hidden = model.init_hidden(BATCH_SIZE)
    processed_games = set()
    all_rewards = []

    while not (state.terminated | state.truncated).all():
        picked_actions, KEY, hidden = handle_batch_player_move(model, state, KEY, hidden)
        state = step(state, picked_actions)

        # Check for games that have just terminated
        for i in range(BATCH_SIZE):
            if (state.terminated[i] or state.truncated[i]) and (i not in processed_games):
                rewards_list = state.rewards[i].tolist()
                all_rewards.append(rewards_list)
                processed_games.add(i)
    
    # Fitness is sum of player 1 rewards
    fitness_value = sum(reward[0] for reward in all_rewards)
    return fitness_value


def evaluate_candidate(candidate_weights, candidate_index, core_id):
    """
    Evaluates one candidate solution on a specific core.
    Returns: (candidate_index, fitness, elapsed_time)
    """
    start_time = time.time()
    
    # Create LSTM model
    lstm_model = SparrowMahjongLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_LAYER_SIZE,
        output_size=OUTPUT_SIZE,
        num_lstm_layers=LSTM_LAYER_COUNT,
        num_fc_layers=FC_LAYER_COUNT,
        fc_hidden_size=HIDDEN_FC_SIZE
    )
    lstm_model.set_model_weights(candidate_weights)

    # Evaluate fitness
    fitness_value = play_games(lstm_model)
    elapsed = time.time() - start_time
    
    return candidate_index, fitness_value, elapsed


def main():
    print("Setting up Multi-Processing CMA-ES Training Environment...")
    
    # Get number of CPU cores
    num_cores = cpu_count()
    print(f"Detected {num_cores} CPU cores for parallel evaluation")
    
    # Initialize LSTM model to get initial weights
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
    
    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(initial_weights, SIGMA, {'CMA_diagonal': USE_DIAGONAL})
    print(f"CMA-ES initialized with population size: {es.popsize}")
    print("Setup Complete! Starting multi-processing training...")

    # Training statistics
    generation_times = []
    best_fitness_history = []
    
    # Training loop with progress bar
    with tqdm(total=NUM_GENERATIONS, desc="CMA-ES Training", unit="gen") as pbar:
        for generation in range(NUM_GENERATIONS):
            start_gen_time = time.time()
            
            # Ask CMA-ES for candidate solutions
            solutions = es.ask()
            generation_fitness = [None] * len(solutions)
            
            # Prepare arguments for multiprocessing
            pool_args = [(candidate, idx, idx % num_cores) 
                        for idx, candidate in enumerate(solutions)]
            
            # Evaluate candidates in parallel
            with multiprocessing.Pool(processes=num_cores) as pool:
                results = pool.starmap(evaluate_candidate, pool_args)
            
            # Collect results
            total_eval_time = 0
            for candidate_index, fitness, elapsed in results:
                generation_fitness[candidate_index] = -fitness  # CMA-ES minimizes, so negate
                total_eval_time += elapsed
            
            # Tell CMA-ES the fitness values
            es.tell(solutions, generation_fitness)
            
            # Calculate statistics
            generation_time = time.time() - start_gen_time
            generation_times.append(generation_time)
            best_fitness = -min(generation_fitness)  # Convert back to positive
            best_fitness_history.append(best_fitness)
            
            # Update progress bar
            avg_gen_time = np.mean(generation_times)
            pbar.set_postfix({
                'Best Fitness': f'{best_fitness:.1f}',
                'Gen Time': f'{generation_time:.1f}s',
                'Avg Time': f'{avg_gen_time:.1f}s',
                'Eval Time': f'{total_eval_time:.1f}s'
            })
            pbar.update(1)
    
    # Final statistics
    total_time = sum(generation_times)
    avg_time = np.mean(generation_times)
    final_best_fitness = max(best_fitness_history)
    
    print(f"\n=== Training Complete ===")
    print(f"Total generations: {NUM_GENERATIONS}")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per generation: {avg_time:.2f} seconds")
    print(f"Final best fitness: {final_best_fitness:.1f}")
    print(f"Best fitness improvement: {final_best_fitness - best_fitness_history[0]:.1f}")
    print(f"CPU cores used: {num_cores}")
    print(f"Population size: {es.popsize}")
    print(f"Games per generation: {es.popsize * BATCH_SIZE}")


if __name__ == "__main__":
    main() 