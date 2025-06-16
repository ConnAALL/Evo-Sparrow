import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import time
import csv
import numpy as np
import jax
import random
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from multiprocessing import cpu_count
from tqdm import tqdm
from datetime import datetime

from env_management import *

def rule_random_experiment_core(
    population_size: int,
    trial_number: int,
    core_id: int
):
    """
    Run the experiment for different population sizes (number of trials)
    to assess duration and variation.
    
    Args:
        population_size: Number of games to run (1-300)
        trial_number: Which trial run this is (1-100)
        core_id: CPU core ID for progress tracking
    
    Returns:
        Dictionary containing experiment results
    """
    ENV_ID = "sparrow_mahjong"
    GAME_COUNT = population_size  # Number of games to run is the population size we're testing
    
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
            for _ in range(red_count):
                complete_actions.append(-action)
            for _ in range(tile_count - red_count):
                complete_actions.append(action)
        return complete_actions

    def handle_batch_player_move(state, key):
        """
        Handle moves for a batch of games:
         - Player 0: Rule-based agent
         - Player 1: Random agent
         - Player 2: Random agent
        """
        key, subkey = jax.random.split(key)
        picked_actions = np.zeros(GAME_COUNT, dtype=int)
        active_games = ~(state.terminated | state.truncated)

        # Identify players by current_player index
        player_rule = (state.current_player == 0) & active_games
        player_random1 = (state.current_player == 1) & active_games
        player_random2 = (state.current_player == 2) & active_games

        # Process Rule-based agent moves (Player 0)
        indices_rule = np.where(player_rule)[0]
        for idx in indices_rule:
            actions = get_valid_actions(state, idx, 0)
            hand = actions_to_hand(actions)
            current_dora = state._dora[idx].tolist() + 1
            picked_tile = choose_next_move(hand, current_dora)
            picked_action = tile_to_action(picked_tile)
            picked_actions[idx] = picked_action

        # Process first Random agent moves (Player 1)
        indices_random1 = np.where(player_random1)[0]
        for idx in indices_random1:
            legal_action_list = state.legal_action_mask[idx].tolist()
            valid_indices = [i for i, val in enumerate(legal_action_list) if val]
            picked_action = random.choice(valid_indices)
            picked_actions[idx] = picked_action
            
        # Process second Random agent moves (Player 2)
        indices_random2 = np.where(player_random2)[0]
        for idx in indices_random2:
            legal_action_list = state.legal_action_mask[idx].tolist()
            valid_indices = [i for i, val in enumerate(legal_action_list) if val]
            picked_action = random.choice(valid_indices)
            picked_actions[idx] = picked_action

        return picked_actions, key

    def play_games():
        start_time = time.time()
        
        init, step = init_env(ENV_ID)
        KEY = reset_random_seed()
        KEY, subkey = jax.random.split(KEY)
        keys = jax.random.split(subkey, GAME_COUNT)
        state = init(keys)

        processed_games = set()
        all_rewards = []

        while not (state.terminated | state.truncated).all():
            picked_actions, KEY = handle_batch_player_move(state, KEY)
            state = step(state, picked_actions)
            for i in range(GAME_COUNT):
                if (state.terminated[i] or state.truncated[i]) and (i not in processed_games):
                    game_reward = state.rewards[i].tolist()
                    all_rewards.append(game_reward)
                    processed_games.add(i)
        
        duration = time.time() - start_time
        
        # Calculate rule-based agent score (player 0)
        rule_scores = [reward[0] for reward in all_rewards]
        rule_total_score = sum(rule_scores)
        
        # Return only the required results
        return {
            "population_size": population_size,
            "trial_number": trial_number,
            "duration": duration,
            "rule_total_score": rule_total_score
        }

    return play_games()


def ensure_directory_exists(directory_path):
    """Ensure the specified directory exists"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def worker_function(job_queue, result_queue, core_id):
    """Worker function for processing jobs from the queue"""
    while True:
        job = job_queue.get()
        if job is None:  # Sentinel value to stop
            break
        
        pop_size, trial = job
        result = rule_random_experiment_core(pop_size, trial, core_id)
        result_queue.put(result)
        job_queue.task_done()


if __name__ == '__main__':
    # Create results directory 
    RESULTS_DIR = "../Data Processing & Plotting/trial_size_experiment"
    ensure_directory_exists(RESULTS_DIR)
    
    # Experiment parameters
    MIN_POPULATION_SIZE = 1
    MAX_POPULATION_SIZE = 1
    TRIALS_PER_SIZE = 100
    
    # Timestamp for files
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # CSV file to store results
    csv_filename = f"{RESULTS_DIR}/trial_experiment_{timestamp}.csv"
    
    # Create a CSV file for results
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'population_size', 
            'trial_number', 
            'duration', 
            'rule_total_score'
        ])

    # Determine available CPU cores and use all of them
    num_cores = cpu_count()
    print(f"Using all {num_cores} CPU cores.")
    
    # Create job queue with all tasks
    job_queue = multiprocessing.JoinableQueue()
    result_queue = multiprocessing.Queue()
    
    # Create all jobs upfront
    total_jobs = 0
    for pop_size in range(MIN_POPULATION_SIZE, MAX_POPULATION_SIZE + 1):
        for trial in range(1, TRIALS_PER_SIZE + 1):
            job_queue.put((pop_size, trial))
            total_jobs += 1
    
    # Add sentinel values to stop workers
    for _ in range(num_cores):
        job_queue.put(None)
    
    # Start worker processes
    workers = []
    for i in range(num_cores):
        process = multiprocessing.Process(
            target=worker_function,
            args=(job_queue, result_queue, i)
        )
        process.daemon = True
        process.start()
        workers.append(process)
    
    # Open CSV file for writing results
    csv_file = open(csv_filename, 'a', newline='')
    writer = csv.writer(csv_file)
    
    # Process results as they come in
    completed = 0
    print(f"\nStarting experiment with {total_jobs} total trials...")
    
    with tqdm(total=total_jobs, desc="Overall Progress", unit="trial", ncols=100) as progress_bar:
        while completed < total_jobs:
            # This will block until a result is available
            if not result_queue.empty():
                result = result_queue.get()
                writer.writerow([
                    result['population_size'],
                    result['trial_number'],
                    result['duration'],
                    result['rule_total_score']
                ])
                csv_file.flush()  # Make sure data is written to disk
                completed += 1
                progress_bar.update(1)
            else:
                time.sleep(0.1)  # Small sleep to prevent CPU hogging
    
    # Close CSV file
    csv_file.close()
    
    # Wait for all processes to complete
    for process in workers:
        process.join()
    
    print(f"\nExperiment completed. Results saved to {csv_filename}")
    print(f"To visualize results, run: python ../Data\\ Processing\\ \\&\\ Plotting/plot_trial_size_experiment.py") 