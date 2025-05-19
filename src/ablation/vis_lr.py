"""
Visualization script for learning rate ablation results.

This script loads and visualizes the results from learning rate ablation studies,
creating plots to compare the performance of different model configurations.
"""

import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

from read_tensorboard import read_tensorboard_data

# Get project root directory
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.join(cwd, "../")
dataset = "MOVI_F"  # Dataset name
save_path_base = os.path.join(cwd, f"../outputs/ablation/lr/{dataset}")

# Create directory for saving figures
output_dir = os.path.join(save_path_base, "figures")
os.makedirs(output_dir, exist_ok=True)

# Set up plot style and size
plt.figure(figsize=(12, 8))
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration for different ablation settings
# Each setting is a list of values for different loss terms:
# [reconstruction, flow, flow_smoothness, reconstruction_flow, point_smoothness]
lr_multi = {
    "ALL":      [0.1,    1.0, 1000.0,  0.01,    0.01],  # All losses enabled
    "NSFP":     [0.0,    1.0,    0.0,   0.0,     0.0],  # Only flow loss (baseline)
    "NO_OGC1":  [0.0,    1.0, 1000.0,  0.01,    0.01],  # Without reconstruction
    "NO_GWM":   [0.1,    1.0,    0.0,  0.01,    0.01],  # Without flow smoothness
    "NO_OGC2":  [0.1,    1.0, 1000.0,  0.01,     0.0],  # Without point smoothness
    "NO_rec":   [0.0,    1.0, 1000.0,   0.0,    0.01],  # Without reconstruction losses
}

# Dictionary to store EPE results
epe_results = {}
run_times = 3  # Number of runs for each configuration

# Collect results for each configuration
for key in lr_multi.keys():
    for i in range(run_times):
        savepath = os.path.join(save_path_base, key, f"run_{i}")
        savepath = os.path.join(cwd, savepath)
        if not os.path.exists(savepath):
            print(f"Path does not exist: {savepath}")
            continue
        _, values = read_tensorboard_data(savepath, "epe")
        if epe_results.get(key) is None:
            epe_results[key] = []
        epe_results[key].append(values)

# Color mapping for different configurations
colors = {
    "ALL": "#1f77b4",     # Blue
    "NSFP": "#ff7f0e",    # Orange
    "NO_OGC1": "#2ca02c", # Green
    "NO_GWM": "#d62728",  # Red
    "NO_OGC2": "#9467bd", # Purple
    "NO_rec": "#8c564b",  # Brown
    "Ideal": "#7f7f7f",   # Gray
}

# Ensure all results have the same length
max_length = 0
for key in epe_results:
    for run in epe_results[key]:
        max_length = max(max_length, len(run))

# Process data for each configuration
for key in epe_results:
    all_runs = []
    for run in epe_results[key]:
        # Pad shorter runs with their last value
        if len(run) < max_length:
            run = run + [run[-1]] * (max_length - len(run))
        all_runs.append(run[:max_length])
    
    # Convert to numpy array for calculations
    all_runs = np.array(all_runs)
    run_times = 3

    # Calculate mean and bounds
    mean_values = np.mean(all_runs, axis=0)
    min_values = np.min(all_runs, axis=0)
    max_values = np.max(all_runs, axis=0)
    
    # X-axis represents training steps
    x = np.arange(len(mean_values))
    
    # Plot mean line and confidence region
    plt.plot(x, mean_values, label=key, color=colors.get(key), linewidth=2)
    plt.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))

# Add chart elements
plt.title('Comparison of EPE Metrics for Different Ablation Configurations', fontsize=16)
plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('EPE Value (Lower is Better)', fontsize=14)
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.legend(fontsize=12)

# Add explanatory text
plt.text(0.5, -0.1, f'For memory efficiency, only the [dynamic objects] from the {dataset} dataset were used. Unlike other literature, the numerical values are not directly comparable.', fontsize=12, ha='center', transform=plt.gca().transAxes)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.savefig(f"{output_dir}/ablation_comparison.png", dpi=300)
print(f"Plot saved to {output_dir}/ablation_comparison.png")

# Display the plot
plt.show()