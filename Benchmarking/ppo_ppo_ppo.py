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
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_rl_agent import ActorCriticLSTM


def ppo_ppo_ppo_benchmark_core(PPO_MODEL_PATH: str,
                               GAME_COUNT: int,
                               HIDDEN_LAYER_SIZE: int,
                               LSTM_LAYER_COUNT: int,
                               FC_LAYER_COUNT: int,
                               HIDDEN_FC_SIZE: int,
                               core_id: int):
    """
    Run a subset of games on one core with a progress bar.
    This version is for the PPO vs. PPO vs. PPO benchmark.
    """
    INPUT_SIZE = 37
    OUTPUT_SIZE = 6
    ENV_ID = "sparrow_mahjong"

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

    def handle_batch_player_move(ppo_model1, ppo_model2, ppo_model3, state, key, ppo_hidden1, ppo_hidden2, ppo_hidden3):
        """Handle moves for all games in the batch."""
        key, subkey = jax.random.split(key)
        picked_actions = np.zeros(GAME_COUNT, dtype=int)
        active_games = ~(state.terminated | state.truncated)

        # --- Process PPO agent for Player 0 ---
        player0_turns = (state.current_player == 0) & active_games
        indices_p0 = np.where(player0_turns)[0]
        if len(indices_p0) > 0:
            input_tensors = []
            valid_moves_list = []
            for idx in indices_p0:
                valid_moves = get_valid_actions(state, idx, 0)
                valid_moves_list.append(valid_moves)
                input_array = state_to_input(state, idx, 0)
                input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                input_tensors.append(input_tensor)
            input_batch = torch.cat(input_tensors, dim=0)
            h0 = ppo_hidden1[0][:, indices_p0, :]
            c0 = ppo_hidden1[1][:, indices_p0, :]
            device = next(ppo_model1.parameters()).device
            input_batch = input_batch.to(device)
            h0 = h0.to(device)
            c0 = c0.to(device)
            
            with torch.no_grad():
                actions_list = []
                for i in range(len(indices_p0)):
                    single_input = input_batch[i:i+1]
                    single_hidden = (h0[:, i:i+1, :], c0[:, i:i+1, :])
                    action, _, _, new_hidden = ppo_model1.act(single_input, single_hidden)
                    actions_list.append(action.item())
                    idx = indices_p0[i]
                    ppo_hidden1[0][:, idx, :] = new_hidden[0][:, 0, :]
                    ppo_hidden1[1][:, idx, :] = new_hidden[1][:, 0, :]
            
            for i, idx in enumerate(indices_p0):
                chosen = actions_list[i]
                valid_moves = valid_moves_list[i]
                picked_action = abs(valid_moves[chosen])
                picked_actions[idx] = picked_action

        # --- Process PPO agent for Player 1 ---
        player1_turns = (state.current_player == 1) & active_games
        indices_p1 = np.where(player1_turns)[0]
        if len(indices_p1) > 0:
            input_tensors1 = []
            valid_moves_list1 = []
            for idx in indices_p1:
                valid_moves = get_valid_actions(state, idx, 1)
                valid_moves_list1.append(valid_moves)
                input_array = state_to_input(state, idx, 1)
                input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                input_tensors1.append(input_tensor)
            input_batch1 = torch.cat(input_tensors1, dim=0)
            h0_1 = ppo_hidden2[0][:, indices_p1, :]
            c0_1 = ppo_hidden2[1][:, indices_p1, :]
            device1 = next(ppo_model2.parameters()).device
            input_batch1 = input_batch1.to(device1)
            h0_1 = h0_1.to(device1)
            c0_1 = c0_1.to(device1)
            
            with torch.no_grad():
                actions_list1 = []
                for i in range(len(indices_p1)):
                    single_input = input_batch1[i:i+1]
                    single_hidden = (h0_1[:, i:i+1, :], c0_1[:, i:i+1, :])
                    action, _, _, new_hidden = ppo_model2.act(single_input, single_hidden)
                    actions_list1.append(action.item())
                    idx = indices_p1[i]
                    ppo_hidden2[0][:, idx, :] = new_hidden[0][:, 0, :]
                    ppo_hidden2[1][:, idx, :] = new_hidden[1][:, 0, :]
            
            for i, idx in enumerate(indices_p1):
                chosen = actions_list1[i]
                valid_moves = valid_moves_list1[i]
                picked_action = abs(valid_moves[chosen])
                picked_actions[idx] = picked_action

        # --- Process PPO agent for Player 2 ---
        player2_turns = (state.current_player == 2) & active_games
        indices_p2 = np.where(player2_turns)[0]
        if len(indices_p2) > 0:
            input_tensors2 = []
            valid_moves_list2 = []
            for idx in indices_p2:
                valid_moves = get_valid_actions(state, idx, 2)
                valid_moves_list2.append(valid_moves)
                input_array = state_to_input(state, idx, 2)
                input_tensor = torch.tensor(input_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                input_tensors2.append(input_tensor)
            input_batch2 = torch.cat(input_tensors2, dim=0)
            h0_2 = ppo_hidden3[0][:, indices_p2, :]
            c0_2 = ppo_hidden3[1][:, indices_p2, :]
            device2 = next(ppo_model3.parameters()).device
            input_batch2 = input_batch2.to(device2)
            h0_2 = h0_2.to(device2)
            c0_2 = c0_2.to(device2)
            
            with torch.no_grad():
                actions_list2 = []
                for i in range(len(indices_p2)):
                    single_input = input_batch2[i:i+1]
                    single_hidden = (h0_2[:, i:i+1, :], c0_2[:, i:i+1, :])
                    action, _, _, new_hidden = ppo_model3.act(single_input, single_hidden)
                    actions_list2.append(action.item())
                    idx = indices_p2[i]
                    ppo_hidden3[0][:, idx, :] = new_hidden[0][:, 0, :]
                    ppo_hidden3[1][:, idx, :] = new_hidden[1][:, 0, :]
            
            for i, idx in enumerate(indices_p2):
                chosen = actions_list2[i]
                valid_moves = valid_moves_list2[i]
                picked_action = abs(valid_moves[chosen])
                picked_actions[idx] = picked_action

        return picked_actions, key, ppo_hidden1, ppo_hidden2, ppo_hidden3

    def play_games(ppo_model1, ppo_model2, ppo_model3):
        init, step = init_env(ENV_ID)
        KEY = reset_random_seed()
        KEY, subkey = jax.random.split(KEY)
        keys = jax.random.split(subkey, GAME_COUNT)
        state = init(keys)
        ppo_hidden1 = ppo_model1.init_hidden(GAME_COUNT)
        ppo_hidden2 = ppo_model2.init_hidden(GAME_COUNT)
        ppo_hidden3 = ppo_model3.init_hidden(GAME_COUNT)

        processed_games = set()
        all_rewards = []
        progress_bar = tqdm(
            total=GAME_COUNT,
            desc=f"Core {core_id}".ljust(8),
            position=core_id,
            leave=True,
            ncols=100,
            bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:05d}/{total:05d} [{elapsed:>8s}<{remaining:>8s}, {rate_fmt:^10}]"
        )
        while not (state.terminated | state.truncated).all():
            picked_actions, KEY, ppo_hidden1, ppo_hidden2, ppo_hidden3 = handle_batch_player_move(
                ppo_model1, ppo_model2, ppo_model3, state, KEY, ppo_hidden1, ppo_hidden2, ppo_hidden3)
            state = step(state, picked_actions)
            prev_count = len(processed_games)
            for i in range(GAME_COUNT):
                if (state.terminated[i] or state.truncated[i]) and (i not in processed_games):
                    game_reward = state.rewards[i].tolist()
                    all_rewards.append(game_reward)
                    processed_games.add(i)
            progress_bar.update(len(processed_games) - prev_count)
        progress_bar.close()
        return all_rewards

    # Initialize the PPO models
    ppo_model1 = ActorCriticLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_LAYER_SIZE,
        num_actions=OUTPUT_SIZE,
        num_lstm_layers=LSTM_LAYER_COUNT,
        num_fc_layers=FC_LAYER_COUNT,
        fc_hidden_size=HIDDEN_FC_SIZE
    )

    ppo_model2 = ActorCriticLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_LAYER_SIZE,
        num_actions=OUTPUT_SIZE,
        num_lstm_layers=LSTM_LAYER_COUNT,
        num_fc_layers=FC_LAYER_COUNT,
        fc_hidden_size=HIDDEN_FC_SIZE
    )

    ppo_model3 = ActorCriticLSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_LAYER_SIZE,
        num_actions=OUTPUT_SIZE,
        num_lstm_layers=LSTM_LAYER_COUNT,
        num_fc_layers=FC_LAYER_COUNT,
        fc_hidden_size=HIDDEN_FC_SIZE
    )
    
    ppo_model1.load_state_dict(torch.load(PPO_MODEL_PATH))
    ppo_model2.load_state_dict(torch.load(PPO_MODEL_PATH))
    ppo_model3.load_state_dict(torch.load(PPO_MODEL_PATH))
    ppo_model1.eval()
    ppo_model2.eval()
    ppo_model3.eval()

    return play_games(ppo_model1, ppo_model2, ppo_model3)


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


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    PPO_MODEL_PATH = "agents/ppo_agent_update_1750.pth"
    TOTAL_GAME_COUNT = 1000000
    SPLIT = 10
    HIDDEN_LAYER_SIZE = 32
    LSTM_LAYER_COUNT = 3
    FC_LAYER_COUNT = 8
    HIDDEN_FC_SIZE = 32

    games_per_round = TOTAL_GAME_COUNT // SPLIT
    remaining_total_games = TOTAL_GAME_COUNT % SPLIT
    
    num_cores = cpu_count() - 2
    print(f"Using {num_cores} CPU cores (reserved 2 cores).")
    print(f"Running {TOTAL_GAME_COUNT} total games in {SPLIT} rounds of ~{games_per_round} games each.")
    
    all_round_results = []
    
    for round_num in range(SPLIT):
        current_round_games = games_per_round
        if round_num == SPLIT - 1:
            current_round_games += remaining_total_games
            
        print(f"\n--- Round {round_num + 1}/{SPLIT}: Running {current_round_games} games ---")
        
        games_per_core = current_round_games // num_cores
        remaining_games = current_round_games % num_cores
        game_counts = [games_per_core] * num_cores
        for i in range(remaining_games):
            game_counts[i] += 1

        pool_args = [
            (PPO_MODEL_PATH,
             game_counts[i],
             HIDDEN_LAYER_SIZE,
             LSTM_LAYER_COUNT,
             FC_LAYER_COUNT,
             HIDDEN_FC_SIZE,
             i)
            for i in range(num_cores)
        ]

        with multiprocessing.Pool(processes=num_cores) as pool:
            round_results = pool.starmap(ppo_ppo_ppo_benchmark_core, pool_args)
        
        round_merged = merge_results(round_results)
        all_round_results.extend(round_merged)
        
        print(f"Round {round_num + 1} completed: {len(round_merged)} games finished")
    
    merged_results = all_round_results
    print(f"\nAll rounds completed! Total games played: {len(merged_results)}")

    output_filename = f"ppo_ppo_ppo_mp_result_1750_1M.txt"
    with open(output_filename, mode="a") as f:
        f.write("P1 PPO  P2 PPO  P3 PPO (Multi-Processing)\n")
        f.write("Games Played: " + str(len(merged_results)) + "\n")

        ### Player 1 Results (PPO) ###
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

        ### Player 2 Results (PPO) ###
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

        ### Player 3 Results (PPO) ###
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
        f.write("Player 3 Wins / Game Count: " + str(player_3_average_wins) + "\n\n")

    print(f"Results saved to {output_filename}")
    print(f"Total games played: {len(merged_results)}")
    print(f"PPO Agent (Player 1) win rate: {player_1_average_wins:.4f}")
    print(f"PPO Agent (Player 2) win rate: {total_player_2_wins / len(merged_results):.4f}")
    print(f"PPO Agent (Player 3) win rate: {total_player_3_wins / len(merged_results):.4f}") 