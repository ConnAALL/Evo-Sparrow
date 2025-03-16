"""
Script to save processed data from the SQLite database to a NumPy file.
The data saved here is used by the plot_data.py script to generate plots.
"""

import os, sqlite3
import numpy as np

MAX_GEN = 50  # Maximum number of generations to consider
# List of runs to exclude
EXCLUDE_DATES = ["250308_031232", "250308_204025", "250308_191003", "250309_033129", "250309_133534",
                 "250308_023354", "250309_064115", "250310_062458", "250309_040935", "250310_190014", "250308_094056"]
SAVE_PATH = "processed_data.npz"

def get_run_date(game_id):
    return "_".join(game_id.split("_")[:2])

def get_run_generation(game_id):
    return game_id.split("_")[2]

def get_run_solution(game_id):
    return game_id.split("_")[3]

def get_sql_data():
    """Load all rows from the DB."""
    conn = sqlite3.connect('game_results.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM results")
    rows = cursor.fetchall()
    conn.close()
    return rows

def organize_data_by_date_generation_solution(rows):
    """Organize data in a nested dictionary format."""
    data_dict = {}
    for row in rows:
        game_id = row[0]
        fitness = row[1]
        run_date = get_run_date(game_id)
        run_generation = get_run_generation(game_id)
        run_solution = get_run_solution(game_id)

        if run_date not in data_dict:
            data_dict[run_date] = {}
        if run_generation not in data_dict[run_date]:
            data_dict[run_date][run_generation] = {}
        if run_solution not in data_dict[run_date][run_generation]:
            data_dict[run_date][run_generation][run_solution] = []

        data_dict[run_date][run_generation][run_solution].append(fitness)
    return data_dict

def get_total_fitness(generation_data):
    """Return the SUM of fitness across solutions for a single generation."""
    return sum(sum(fitness_list) for fitness_list in generation_data.values())

def get_run_fitness_list(run_data, max_gen=MAX_GEN):
    """Build a list of total fitness for each generation in this run."""
    fitness_list = []
    for gen in range(1, max_gen + 1):
        gen_key = str(gen)
        if gen_key in run_data:
            total_fitness = get_total_fitness(run_data[gen_key])
            fitness_list.append(total_fitness)
        else:
            break
    return fitness_list

rows = get_sql_data()
data_dict = organize_data_by_date_generation_solution(rows)

# Store fitness values per generation across runs
fitness_per_generation = {gen: [] for gen in range(1, MAX_GEN + 1)}

for run_date, run_data in data_dict.items():
    if run_date not in EXCLUDE_DATES:
        run_fitness_list = get_run_fitness_list(run_data, MAX_GEN)

        if len(run_fitness_list) < 45:
            print(f"Skipping run {run_date} because it only has {len(run_fitness_list)} generations (< 45).")
            continue

        for gen, fitness in enumerate(run_fitness_list, start=1):
            fitness_per_generation[gen].append(fitness)

# Convert to NumPy format and save
all_generations = np.array(list(fitness_per_generation.keys()))
fitness_values = np.array([fitness_per_generation[gen] for gen in all_generations], dtype=object)

# Compute global max and min across all solutions for each generation
max_fitness_values = np.array([max(fitness_per_generation[gen]) for gen in all_generations])
min_fitness_values = np.array([min(fitness_per_generation[gen]) for gen in all_generations])

np.savez(SAVE_PATH,
         all_generations=all_generations,
         fitness_values=fitness_values, 
         max_fitness_values=max_fitness_values,
         min_fitness_values=min_fitness_values)

