import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import multiprocessing
import os
import random
from datetime import datetime
from multiprocessing import cpu_count
from typing import List, Tuple

import jax
import numpy as np
import torch
from tqdm import tqdm
import sqlite3

# Project-local imports
from env_management import (
    init_env,
    reset_random_seed,
    shuffled_to_normal,
    state_to_input,
    choose_next_move,
    tile_to_action,
    actions_to_hand,
)
from lstm_structure import SparrowMahjongLSTM
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_rl_agent import ActorCriticLSTM

# ---------------------------------------------------------------------------
# Helper functions shared with specialised benchmark scripts
# ---------------------------------------------------------------------------

def get_valid_actions(state, idx: int, player_idx: int) -> List[int]:
    """Return the complete list of valid action indices for the given player.

    Red Dora tiles are represented with negative indices (following existing
    convention in earlier benchmark scripts). The returned list is *not*
    truncated/padded – it contains all legal moves for that player in that
    position.
    """
    shuffled_hands = state._hands[idx]
    shuffled_reds = state._n_red_in_hands[idx]
    original_hands = shuffled_to_normal(state, shuffled_hands, idx)
    original_reds = shuffled_to_normal(state, shuffled_reds, idx)

    legal_action_list = state.legal_action_mask[idx].tolist()
    valid_indices = [i for i, val in enumerate(legal_action_list) if val]

    player_hand = original_hands[player_idx].tolist()
    red_tile_counts = original_reds[player_idx].tolist()
    complete_actions: List[int] = []
    for action in valid_indices:
        tile_count = player_hand[action]
        red_count = red_tile_counts[action]
        # Add red Dora tiles first (represented with negative indices).
        for _ in range(red_count):
            complete_actions.append(-action)
        # Then the regular tiles.
        for _ in range(tile_count - red_count):
            complete_actions.append(action)
    return complete_actions


def game_result(game: List[int], player_idx: int) -> int:
    """Convert raw reward triplet into a discrete outcome label (same rules)."""
    player_score = game[player_idx]
    other_scores = [score for i, score in enumerate(game) if i != player_idx]
    if player_score > 0:
        return 1
    if player_score == 0 and all(score == 0 for score in other_scores):
        return 0
    if player_score == 0 and any(score != 0 for score in other_scores):
        return -1
    if player_score < 0:
        return -2
    return 0  # Fallback (should not happen)


# ---------------------------------------------------------------------------
# Core benchmark routine executed on each worker process
# ---------------------------------------------------------------------------

def benchmark_core(
    p_types: Tuple[str, str, str],
    game_count: int,
    spare_cores: int,
    core_id: int,
    dl_model_path: str,
    ppo_model_path: str,
):
    """Run `game_count` games on a single CPU core.

    Parameters
    ----------
    p_types: Tuple of three strings – each of `"rnd"`, `"rb"`, `"dl"`, `"ppo"`.
    game_count: Number of games this worker should simulate.
    spare_cores: Unused, kept for signature consistency.
    core_id: Numerical identifier for tqdm positioning.
    dl_model_path / ppo_model_path: Paths to pretrained checkpoints.
    """
    # Model hyper-parameters (consistent with earlier scripts)
    INPUT_SIZE = 37
    OUTPUT_SIZE = 6
    ENV_ID = "sparrow_mahjong"

    HIDDEN_LAYER_SIZE = 32
    LSTM_LAYER_COUNT = 3
    FC_LAYER_COUNT = 8
    HIDDEN_FC_SIZE = 32

    # ---------------------------------------------------------------------
    # Build per-player agent objects (load models only if required)
    # ---------------------------------------------------------------------
    models = [None, None, None]  # Holds torch.nn.Module or None per player
    hidden_states = [None, None, None]  # Tuple[Tensor, Tensor] per player

    # Cache DL and PPO models if multiple players happen to use same model
    cached_dl_model = None
    cached_ppo_model = None

    def _load_dl():
        nonlocal cached_dl_model
        if cached_dl_model is None:
            model = SparrowMahjongLSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_LAYER_SIZE,
                output_size=OUTPUT_SIZE,
                num_lstm_layers=LSTM_LAYER_COUNT,
                num_fc_layers=FC_LAYER_COUNT,
                fc_hidden_size=HIDDEN_FC_SIZE,
            )
            model.load_state_dict(torch.load(dl_model_path, map_location="cpu"))
            model.eval()
            cached_dl_model = model
        return cached_dl_model

    def _load_ppo():
        nonlocal cached_ppo_model
        if cached_ppo_model is None:
            model = ActorCriticLSTM(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_LAYER_SIZE,
                num_actions=OUTPUT_SIZE,
                num_lstm_layers=LSTM_LAYER_COUNT,
                num_fc_layers=FC_LAYER_COUNT,
                fc_hidden_size=HIDDEN_FC_SIZE,
            )
            model.load_state_dict(torch.load(ppo_model_path, map_location="cpu"))
            model.eval()
            cached_ppo_model = model
        return cached_ppo_model

    for idx, p_type in enumerate(p_types):
        if p_type == "dl":
            models[idx] = _load_dl()
            hidden_states[idx] = models[idx].init_hidden(game_count)
        elif p_type == "ppo":
            models[idx] = _load_ppo()
            hidden_states[idx] = models[idx].init_hidden(game_count)
        else:
            # rnd / rb do not need a model or hidden state
            models[idx] = None
            hidden_states[idx] = None

    # ---------------------------------------------------------------------
    # Environment initialisation
    # ---------------------------------------------------------------------
    init, step = init_env(ENV_ID)
    KEY = reset_random_seed()
    KEY, subkey = jax.random.split(KEY)
    keys = jax.random.split(subkey, game_count)
    state = init(keys)

    processed_games = set()
    all_rewards: List[List[int]] = []

    progress_bar = tqdm(
        total=game_count,
        desc=f"Core {core_id}".ljust(8),
        position=core_id,
        leave=True,
        ncols=100,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n:05d}/{total:05d} [" "{elapsed:>8s}<{remaining:>8s}, {rate_fmt:^10}]",
    )

    # ---------------------------------------------------------------------
    # Main simulation loop – continues until all games finished
    # ---------------------------------------------------------------------
    while not (state.terminated | state.truncated).all():
        picked_actions = np.zeros(game_count, dtype=int)
        active_games = ~(state.terminated | state.truncated)

        # Iterate over players 0‥2 once – handle only games where it's that
        # player's turn and the game is still active.
        for player_idx in range(3):
            player_turns = (state.current_player == player_idx) & active_games
            if not player_turns.any():
                continue
            indices = np.where(player_turns)[0]
            p_type = p_types[player_idx]

            if p_type == "rnd":
                # Pure random – sample any legal action index.
                for g_idx in indices:
                    legal = state.legal_action_mask[g_idx].tolist()
                    valid_indices = [i for i, val in enumerate(legal) if val]
                    picked_actions[g_idx] = random.choice(valid_indices)

            elif p_type == "rb":
                # Rule-based discard heuristic (identical to existing scripts).
                for g_idx in indices:
                    actions = get_valid_actions(state, g_idx, player_idx)
                    hand = actions_to_hand(actions)
                    current_dora = state._dora[g_idx].tolist() + 1
                    picked_tile = choose_next_move(hand, current_dora)
                    picked_actions[g_idx] = tile_to_action(picked_tile)

            elif p_type == "dl":
                model = models[player_idx]
                input_tensors = []
                valid_moves_lists = []
                for g_idx in indices:
                    valid_moves = get_valid_actions(state, g_idx, player_idx)
                    valid_moves_lists.append(valid_moves)
                    input_arr = state_to_input(state, g_idx, player_idx)
                    t = torch.tensor(input_arr, dtype=torch.float32).unsqueeze(0)  # seq len 1
                    input_tensors.append(t)
                input_batch = torch.stack(input_tensors, dim=0)
                h0 = hidden_states[player_idx][0][:, indices, :]
                c0 = hidden_states[player_idx][1][:, indices, :]
                with torch.no_grad():
                    out, (h_n, c_n) = model(input_batch, (h0, c0))
                max_indices = out.argmax(dim=1)
                for i, g_idx in enumerate(indices):
                    chosen = max_indices[i].item()
                    picked_actions[g_idx] = abs(valid_moves_lists[i][chosen])
                    # Update hidden state
                    hidden_states[player_idx][0][:, g_idx, :] = h_n[:, i, :]
                    hidden_states[player_idx][1][:, g_idx, :] = c_n[:, i, :]

            elif p_type == "ppo":
                model = models[player_idx]
                input_tensors = []
                valid_moves_lists = []
                for g_idx in indices:
                    vals = get_valid_actions(state, g_idx, player_idx)
                    valid_moves_lists.append(vals)
                    arr = state_to_input(state, g_idx, player_idx)
                    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (batch, seq, feat)
                    input_tensors.append(t)
                input_batch = torch.cat(input_tensors, dim=0)
                h0 = hidden_states[player_idx][0][:, indices, :]
                c0 = hidden_states[player_idx][1][:, indices, :]
                actions_list = []
                with torch.no_grad():
                    for i in range(len(indices)):
                        single_inp = input_batch[i : i + 1]  # Keep batch dim
                        single_hidden = (h0[:, i : i + 1, :], c0[:, i : i + 1, :])
                        act, _, _, new_hidden = model.act(single_inp, single_hidden)
                        actions_list.append(act.item())
                        # Store updated hidden.
                        g_idx = indices[i]
                        hidden_states[player_idx][0][:, g_idx, :] = new_hidden[0][:, 0, :]
                        hidden_states[player_idx][1][:, g_idx, :] = new_hidden[1][:, 0, :]
                for i, g_idx in enumerate(indices):
                    picked_actions[g_idx] = abs(valid_moves_lists[i][actions_list[i]])
            else:
                raise ValueError(f"Unsupported player type: {p_type}")

        # Environment step
        state = step(state, picked_actions)

        # Collect and record completed games for progress tracking.
        prev_cnt = len(processed_games)
        for i in range(game_count):
            if (state.terminated[i] or state.truncated[i]) and (i not in processed_games):
                all_rewards.append(state.rewards[i].tolist())
                processed_games.add(i)
        progress_bar.update(len(processed_games) - prev_cnt)
    progress_bar.close()
    return all_rewards


# ---------------------------------------------------------------------------
# Utility for merging results from worker processes
# ---------------------------------------------------------------------------

def merge_results(results_list: List[List[List[int]]]) -> List[List[int]]:
    merged: List[List[int]] = []
    for res in results_list:
        merged.extend(res)
    return merged


# ---------------------------------------------------------------------------
# Result summarisation identical to existing scripts
# ---------------------------------------------------------------------------

def summarise_and_save(results: List[List[int]], p_types: Tuple[str, str, str]):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    fname = f"{p_types[0]}_{p_types[1]}_{p_types[2]}_mp_result_{timestamp}.txt"
    results_dir = os.path.join(os.path.dirname(__file__), "Results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, fname), "w") as f:
        title = f"P1 {p_types[0].upper()}  P2 {p_types[1].upper()}  P3 {p_types[2].upper()} (Multi-Processing)\n"
        f.write(title)
        f.write("Games Played: " + str(len(results)) + "\n")

        for p_idx in range(3):
            total_score = sum(game[p_idx] for game in results)
            outcomes = [game_result(game, p_idx) for game in results]
            total_wins = outcomes.count(1)
            total_draws = outcomes.count(0)
            total_loss = outcomes.count(-1)
            total_deal_in = outcomes.count(-2)
            avg_wins = total_wins / len(results)

            f.write(f"\n--- Player {p_idx + 1} ({p_types[p_idx].upper()}) Results ---\n")
            f.write(f"Player {p_idx + 1} Total Score: {total_score}\n")
            f.write(f"Player {p_idx + 1} Total Wins: {total_wins}\n")
            f.write(f"Player {p_idx + 1} Total Draws: {total_draws}\n")
            f.write(f"Player {p_idx + 1} Total Loss (No Point Loss Someone Else Won): {total_loss}\n")
            f.write(f"Player {p_idx + 1} Deal In (Loss Point): {total_deal_in}\n")
            f.write(f"Player {p_idx + 1} Wins / Game Count: {avg_wins}\n")

    # ------------------------------------------------------------------
    # Save raw per-game scores to SQLite database for further analysis
    # ------------------------------------------------------------------
    db_path = os.path.join(os.path.dirname(__file__), "run_results.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_name = f"{p_types[0]}_{p_types[1]}_{p_types[2]}_{timestamp}"
    # Ensure the table exists – names are quoted to allow digits / underscores
    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            p1_score INTEGER,
            p2_score INTEGER,
            p3_score INTEGER
        );
    """
    cursor.execute(create_table_sql)

    # Insert all game results
    cursor.executemany(
        f'INSERT INTO "{table_name}" (p1_score, p2_score, p3_score) VALUES (?, ?, ?);',
        [tuple(game) for game in results],
    )
    conn.commit()
    conn.close()

    full_path = os.path.join(results_dir, fname)
    print("Benchmark completed.\n  • Text summary :", full_path,
          "\n  • Scores saved :", db_path, "->", table_name)

    # Print statistics to console
    print("\nBenchmark Statistics:")
    print("=" * 50)
    print(f"Total games played: {len(results)}")
    print("=" * 50)

    for p_idx in range(3):
        total_score = sum(game[p_idx] for game in results)
        outcomes = [game_result(game, p_idx) for game in results]
        total_wins = outcomes.count(1)
        total_draws = outcomes.count(0)
        total_loss = outcomes.count(-1)
        total_deal_in = outcomes.count(-2)
        avg_wins = total_wins / len(results)

        print(f"\nPlayer {p_idx + 1} ({p_types[p_idx].upper()}):")
        print("-" * 30)
        print(f"Total Score      : {total_score}")
        print(f"Total Wins       : {total_wins}")
        print(f"Total Draws      : {total_draws}")
        print(f"Total Losses     : {total_loss}")
        print(f"Total Deal-ins   : {total_deal_in}")
        print(f"Win Rate         : {avg_wins:.4f}")


# ---------------------------------------------------------------------------
# Main entry point – parses CLI arguments and orchestrates multiprocessing
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Sparrow Mahjong benchmarks for arbitrary agent combinations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    accepted = "{rnd | rb | dl | ppo}"
    parser.add_argument("--p1", required=True, choices=["rnd", "rb", "dl", "ppo"],
                        help=f"Agent type for player 1. Acceptable: {accepted}.")
    parser.add_argument("--p2", required=True, choices=["rnd", "rb", "dl", "ppo"],
                        help=f"Agent type for player 2. Acceptable: {accepted}.")
    parser.add_argument("--p3", required=True, choices=["rnd", "rb", "dl", "ppo"],
                        help=f"Agent type for player 3. Acceptable: {accepted}.")
    parser.add_argument(
        "--spare",
        type=int,
        default=2,
        help="Number of CPU cores to leave idle (not used by the benchmark).",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1_000_000,
        help="Total number of games to simulate.",
    )
    parser.add_argument(
        "--dl_model",
        default="agents/241104_234416_40.pth",
        help="Path to pretrained LSTM (dl) checkpoint.",
    )
    parser.add_argument(
        "--ppo_model",
        default="agents/ppo_agent_update_1750.pth",
        help="Path to pretrained PPO checkpoint.",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=1,
        help="Number of rounds to split the simulation into (to control memory usage).",
    )

    args = parser.parse_args()

    p_types = (args.p1, args.p2, args.p3)

    # Determine cores to use
    total_cores = max(1, cpu_count() - args.spare)
    print(f"Detected {cpu_count()} CPU cores – using {total_cores} cores with split={args.split}.")

    # Calculate games per round
    games_per_round = args.games // args.split
    remaining_total_games = args.games % args.split

    all_round_results = []

    for round_idx in range(args.split):
        current_round_games = games_per_round
        if round_idx == args.split - 1:  # Add remainder to last round
            current_round_games += remaining_total_games

        print(f"\n--- Round {round_idx + 1}/{args.split}: Running {current_round_games} games ---")

        # Distribute games among cores this round
        games_per_core = current_round_games // total_cores
        remaining_games = current_round_games % total_cores
        game_counts = [games_per_core] * total_cores
        for i in range(remaining_games):
            game_counts[i] += 1

        pool_args = [
            (
                p_types,
                game_counts[i],
                args.spare,
                i,
                args.dl_model,
                args.ppo_model,
            )
            for i in range(total_cores)
        ]

        multiprocessing.set_start_method("spawn", force=True)
        with multiprocessing.Pool(processes=total_cores) as pool:
            round_results = pool.starmap(benchmark_core, pool_args)

        round_merged = merge_results(round_results)
        all_round_results.extend(round_merged)
        print(f"Round {round_idx + 1} completed: {len(round_merged)} games finished")

    merged = all_round_results
    summarise_and_save(merged, p_types)


if __name__ == "__main__":
    main() 