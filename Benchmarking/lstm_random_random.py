import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import numpy as np
import jax
import random

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import cpu_count
from tqdm import tqdm

from env_management import *
from lstm_structure import *


def lstm_random_random_benchmark_core(MODEL_PATH: str,
                                      GAME_COUNT: int,
                                      HIDDEN_LAYER_SIZE: int,
                                      LSTM_LAYER_COUNT: int,
                                      FC_LAYER_COUNT: int,
                                      HIDDEN_FC_SIZE: int,
                                      core_id: int):
    """
    Run the LSTM-Random-Random benchmark for a subset of games on one core.
    """
    INPUT_SIZE = 37
    OUTPUT_SIZE = 6
    ENV_ID = "sparrow_mahjong"

    def get_valid_actions(state, idx, player_idx):
        """Get valid actions for a player (with red Dora as negative indices)."""
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
        """
        Handle moves for all games in the batch:
         - Player 0 uses the LSTM agent.
         - Players 1 and 2 choose a random valid move.
        """
        key, subkey = jax.random.split(key)
        picked_actions = np.zeros(GAME_COUNT, dtype=int)
        active_games = ~(state.terminated | state.truncated)
        # Identify games where the current player is 0 (LSTM agent)
        player1_turns = (state.current_player == 0) & active_games

        # Process LSTM agent moves for player 0
        indices_player1 = np.where(player1_turns)[0]
        if len(indices_player1) > 0:
            input_tensors = []
            valid_moves_list = []
            for idx in indices_player1:
                valid_moves = get_valid_actions(state, idx, 0)
                valid_moves_list.append(valid_moves)
                input_array = state_to_input(state, idx)
                input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0)
                input_tensors.append(input_tensor)
            input_batch = torch.stack(input_tensors, dim=0)
            h0 = hidden[0][:, indices_player1, :]
            c0 = hidden[1][:, indices_player1, :]
            device = next(model.parameters()).device
            input_batch = input_batch.to(device)
            h0 = h0.to(device)
            c0 = c0.to(device)
            with torch.no_grad():
                output, (h_n, c_n) = model(input_batch, (h0, c0))
            max_indices = output.argmax(dim=1)
            for i, idx in enumerate(indices_player1):
                chosen = max_indices[i].item()
                valid_moves = valid_moves_list[i]
                picked_action = abs(valid_moves[chosen])
                picked_actions[idx] = picked_action
                # Update hidden state for the LSTM agent.
                hidden[0][:, idx, :] = h_n[:, i, :]
                hidden[1][:, idx, :] = c_n[:, i, :]

        # Process random moves for players 1 and 2
        indices_other = np.where(~player1_turns & active_games)[0]
        for idx in indices_other:
            legal_action_list = state.legal_action_mask[idx].tolist()
            valid_indices = [i for i, val in enumerate(legal_action_list) if val]
            picked_action = random.choice(valid_indices)
            picked_actions[idx] = picked_action

        return picked_actions, key, hidden

    def play_games(model):
        init, step = init_env(ENV_ID)
        KEY = reset_random_seed()
        KEY, subkey = jax.random.split(KEY)
        keys = jax.random.split(subkey, GAME_COUNT)
        state = init(keys)
        hidden = model.init_hidden(GAME_COUNT)
        processed_games = set()
        all_rewards = []
        # Set up a fixed-width tqdm progress bar for this core.
        progress_bar = tqdm(
            total=GAME_COUNT,
            desc=f"Core {core_id}".ljust(8),  # Fixed width for the core label
            position=core_id,
            leave=True,
            ncols=100,  # Fixed total width of the progress bar
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:05d}/{total:05d} [{elapsed:>8s}<{remaining:>8s}, {rate_fmt:^10}]"
        )

        while not (state.terminated | state.truncated).all():
            picked_actions, KEY, hidden = handle_batch_player_move(model, state, KEY, hidden)
            state = step(state, picked_actions)
            prev_count = len(processed_games)
            for i in range(GAME_COUNT):
                if (state.terminated[i] or state.truncated[i]) and (i not in processed_games):
                    game_reward = state.rewards[i].tolist()
                    all_rewards.append(game_reward)
                    processed_games.add(i)
            # Update progress bar with the number of newly finished games.
            progress_bar.update(len(processed_games) - prev_count)
        progress_bar.close()
        return all_rewards

    # Initialize and load the LSTM model.
    lstm_model = SparrowMahjongLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_LAYER_SIZE,
        output_size=OUTPUT_SIZE,
        num_lstm_layers=LSTM_LAYER_COUNT,
        num_fc_layers=FC_LAYER_COUNT,
        fc_hidden_size=HIDDEN_FC_SIZE
    )
    lstm_model.load_state_dict(torch.load(MODEL_PATH))
    lstm_model.eval()
    all_rewards = play_games(lstm_model)
    return all_rewards


def merge_results(results_list):
    merged = []
    for result in results_list:
        merged.extend(result)
    return merged


def game_result(game, player_idx):
    player_score = game[player_idx]
    other_scores = [score for idx, score in enumerate(game) if idx != player_idx]
    if player_score > 0:
        return 1
    elif player_score == 0 and all(score == 0 for score in other_scores):
        return 0
    elif player_score == 0 and any(score != 0 for score in other_scores):
        return -1
    elif player_score < 0:
        return -2


if __name__ == '__main__':
    START_DT = "241104_234416_40"
    MODEL_PATH = f'../{START_DT}.pth'
    TOTAL_GAME_COUNT = 1000000
    HIDDEN_LAYER_SIZE = 32
    LSTM_LAYER_COUNT = 3
    FC_LAYER_COUNT = 8
    HIDDEN_FC_SIZE = 32

    # Determine available CPU cores and distribute games among them.
    num_cores = cpu_count()
    print(f"Detected {num_cores} CPU cores.")
    games_per_core = TOTAL_GAME_COUNT // num_cores
    remaining_games = TOTAL_GAME_COUNT % num_cores
    game_counts = [games_per_core] * num_cores
    for i in range(remaining_games):
        game_counts[i] += 1

    pool_args = [(MODEL_PATH,
                  game_counts[i],
                  HIDDEN_LAYER_SIZE,
                  LSTM_LAYER_COUNT,
                  FC_LAYER_COUNT,
                  HIDDEN_FC_SIZE,
                  i) for i in range(num_cores)]

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.starmap(lstm_random_random_benchmark_core, pool_args)

    merged_results = merge_results(results)

    output_filename = f"lstm_random_random_mp_result_{START_DT}.txt"
    with open(output_filename, mode="a") as f:
        f.write("P1 LSTM  P2 Random  P3 Random (Multi-Processing)\n")
        f.write("Games Played: " + str(len(merged_results)) + "\n")

        # --- Player 1 (LSTM) Results ---
        player_1_total_score = sum(game[0] for game in merged_results)
        player_1_results = [game_result(game, 0) for game in merged_results]
        total_player_1_wins = player_1_results.count(1)
        total_player_1_draws = player_1_results.count(0)
        total_player_1_loss = player_1_results.count(-1)
        total_player_1_deal_in = player_1_results.count(-2)
        player_1_average_wins = total_player_1_wins / len(merged_results)

        f.write("Player 1 Total Score: " + str(player_1_total_score) + "\n")
        f.write("Player 1 Total Wins: " + str(total_player_1_wins) + "\n")
        f.write("Player 1 Total Draws: " + str(total_player_1_draws) + "\n")
        f.write("Player 1 Total Loss (No Point Loss Someone Else Won): " + str(total_player_1_loss) + "\n")
        f.write("Player 1 Deal In (Loss Point): " + str(total_player_1_deal_in) + "\n")
        f.write("Player 1 Wins / Game Count: " + str(player_1_average_wins) + "\n\n")

        # --- Player 2 (Random) Results ---
        player_2_total_score = sum(game[1] for game in merged_results)
        player_2_results = [game_result(game, 1) for game in merged_results]
        total_player_2_wins = player_2_results.count(1)
        total_player_2_draws = player_2_results.count(0)
        total_player_2_loss = player_2_results.count(-1)
        total_player_2_deal_in = player_2_results.count(-2)
        player_2_average_wins = total_player_2_wins / len(merged_results)

        f.write("Player 2 Total Score: " + str(player_2_total_score) + "\n")
        f.write("Player 2 Total Wins: " + str(total_player_2_wins) + "\n")
        f.write("Player 2 Total Draws: " + str(total_player_2_draws) + "\n")
        f.write("Player 2 Total Loss (No Point Loss Someone Else Won): " + str(total_player_2_loss) + "\n")
        f.write("Player 2 Deal In (Loss Point): " + str(total_player_2_deal_in) + "\n")
        f.write("Player 2 Wins / Game Count: " + str(player_2_average_wins) + "\n\n")

        # --- Player 3 (Random) Results ---
        player_3_total_score = sum(game[2] for game in merged_results)
        player_3_results = [game_result(game, 2) for game in merged_results]
        total_player_3_wins = player_3_results.count(1)
        total_player_3_draws = player_3_results.count(0)
        total_player_3_loss = player_3_results.count(-1)
        total_player_3_deal_in = player_3_results.count(-2)
        player_3_average_wins = total_player_3_wins / len(merged_results)

        f.write("Player 3 Total Score: " + str(player_3_total_score) + "\n")
        f.write("Player 3 Total Wins: " + str(total_player_3_wins) + "\n")
        f.write("Player 3 Total Draws: " + str(total_player_3_draws) + "\n")
        f.write("Player 3 Total Loss (No Point Loss Someone Else Won): " + str(total_player_3_loss) + "\n")
        f.write("Player 3 Deal In (Loss Point): " + str(total_player_3_deal_in) + "\n")
        f.write("Player 3 Wins / Game Count: " + str(player_3_average_wins) + "\n")
    
    print(f"Benchmark completed. Merged results written to {output_filename}.")
