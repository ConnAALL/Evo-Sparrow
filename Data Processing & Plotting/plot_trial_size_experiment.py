#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

# Use SciencePlots styles
plt.style.use(['science', 'no-latex'])

# --- CONFIGURATION ---
CSV_FILE = 'trial_experiment_250520_071640.csv'

# --- LOAD DATA ---
df = pd.read_csv(CSV_FILE)

# --- AGGREGATE METRICS ---
grouped = df.groupby('population_size')

pop_sizes = grouped['duration'].mean().index.values
avg_total_duration = grouped['duration'].mean().values
avg_duration_per_game = avg_total_duration / pop_sizes

var_total_score = grouped['rule_total_score'].var(ddof=1).values
var_score_per_game = var_total_score / (pop_sizes**2)

# Efficiency = variance per game / duration per game
efficiency = (avg_duration_per_game * var_score_per_game)

# --- PLOTTING HELPERS ---
def save_plot(x, y, xlabel, ylabel, title, fname):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, linestyle='-', color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()

# --- PLOTS ---
save_plot(
    pop_sizes,
    avg_duration_per_game,
    'Population Size',
    'Average Duration per Game (s)',
    'Population Size vs. Avg. Duration per Game',
    'pop_vs_avg_duration_per_game.png'
)

save_plot(
    pop_sizes,
    var_score_per_game,
    'Population Size',
    'Variance of Score per Game',
    'Population Size vs. Variance of Score per Game',
    'pop_vs_var_score_per_game.png'
)

save_plot(
    pop_sizes,
    efficiency,
    'Population Size',
    'Score Variance x Second',
    'Run Efficiency: Variance x Time',
    'pop_vs_variance_rate.png'
)

# Optional: display interactively
plt.show()
