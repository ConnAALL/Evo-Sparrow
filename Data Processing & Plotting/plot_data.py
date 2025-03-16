import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots

# Hyperparameters for the plotting
DPI = 800
RESULT_FOLDER = "RESULT_PLOTS"
SAVE_PATH = "processed_data.npz"

def create_result_folder(folder_name=RESULT_FOLDER):
    """Ensure the result folder exists."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def save_plot_to_folder(file_name, folder_name=RESULT_FOLDER):
    """Save the plot to a folder."""
    full_path = os.path.join(folder_name, file_name + ".png")
    plt.savefig(full_path, dpi=DPI, bbox_inches='tight')
    plt.close()


if not os.path.exists(SAVE_PATH):
    print(f"Processed data file {SAVE_PATH} not found. Run data_processing.py first.")
    exit()

# Load the processed data
data = np.load(SAVE_PATH, allow_pickle=True)

all_generations = data["all_generations"]
fitness_values = data["fitness_values"]
max_fitness_values = data["max_fitness_values"]
min_fitness_values = data["min_fitness_values"]

# Compute percentiles
percentile_25 = np.percentile(fitness_values.tolist(), 25, axis=1)
percentile_50 = np.percentile(fitness_values.tolist(), 50, axis=1)
percentile_75 = np.percentile(fitness_values.tolist(), 75, axis=1)

# This is the plotting style that is used.
# To use the IEEE style, you also need LaTeX and several installed.
plt.style.use(['science', 'ieee'])
plt.figure(dpi=DPI)

plt.plot(all_generations,
         percentile_50,
         color='black',
         label="Median Fitness")

plt.fill_between(all_generations,
                 percentile_25,
                 percentile_75,
                 color='gray',
                 alpha=0.4,
                 label="25th-75th Percentile")

# Plot max and min fitness values
plt.plot(all_generations,
         max_fitness_values,
         color='black',
         linestyle='dashed',
         label="Max Total Fitness")

plt.plot(all_generations,
         min_fitness_values,
         color='black',
         linestyle='dotted',
         label="Min Total Fitness")

# Adjusting the plot styling
plt.xlabel("Generation")
plt.ylabel("Total Fitness")
plt.yticks(np.arange(-11000, 4000, 1000))  # Adjust Y-axis 
plt.ylim(-12000, 4000)

major_ticks = [1, 10, 20, 30, 40, 50]
minor_ticks = [1] + list(range(2, 51, 2))

ax = plt.gca()  # Get current axis
ax.set_xticks(major_ticks)  # Set major ticks
plt.xlim(0, 51)
ax.set_xticks(minor_ticks, minor=True)  # Set minor ticks explicitly

plt.legend(loc='lower right')

# Save and close
create_result_folder()
save_plot_to_folder("fitness_graph")
