"""
Learning rate ablation study script.

This script performs ablation studies by running training with different combinations
of loss terms and analyzing their impact on model performance.
"""

import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

from read_tensorboard import read_tensorboard_data

# Configuration for different ablation settings
# Each setting is a list of boolean flags for different loss terms:
# [reconstruction, flow, flow_smoothness, reconstruction_flow, point_smoothness]
lr_multi = {  
    "ALL":      [True,  True,  True,    True,     True],   # All losses enabled
    "NSFP":     [False, True,  False,   False,    False],  # Only flow loss (baseline)
    "NO_OGC1":  [False, True,  True,    True,     True],   # Without reconstruction
    "NO_GWM":   [True,  True,  False,   True,     True],   # Without flow smoothness
    "NO_OGC2":  [True,  True,  True,    True,     False],  # Without point smoothness
    "NO_rec":   [False, True,  True,    False,    True],   # Without reconstruction losses
}

run_times = 4  # Number of runs for each configuration
dataset_list = ["MOVI_F"]  # Dataset to use for ablation study

# Get project root directory
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.join(cwd, "../")

for dataset in dataset_list:
    save_path_base = os.path.join(cwd, f"../outputs/ablation/lr/{dataset}")
    
    for key in lr_multi.keys():
        # Run multiple times for each configuration
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            if os.path.exists(savepath):
                continue
            os.makedirs(savepath, exist_ok=True)
            
            # Set up configuration path
            config_path = os.path.dirname(os.path.abspath(__file__))
            if dataset == "MOVI_F":
                config_path = os.path.join(config_path, "../config/movi_f_prescene.yaml")
            else:
                config_path = os.path.join(config_path, "../config/baseconfig.yaml")
                
            # Build command with appropriate loss flags
            command_list = [
                "python", "main.py",
                f"--config {config_path}",
                f"log.dir={savepath}",
                f"dataset.name={dataset}",
            ]
            
            # Add loss term settings based on configuration
            if not lr_multi[key][0]:
                command_list.append("lr_multi.rec_loss=0.0")
            if not lr_multi[key][1]:
                command_list.append("lr_multi.flow_loss=0.0")
            if not lr_multi[key][2]:
                command_list.append("lr_multi.scene_flow_smoothness=0.0")
            if not lr_multi[key][3]:
                command_list.append("lr_multi.rec_flow_loss=0.0")
            if not lr_multi[key][4]:
                command_list.append("lr_multi.point_smooth_loss=0.0")
                
            # Execute training command
            command = " ".join(command_list)
            print(command)
            result = subprocess.run(command, cwd=cwd, shell=True)

    # Dictionary to store EPE results
    epe_results = {}

    # Collect results for each configuration
    for key in lr_multi.keys():
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            savepath = os.path.join(cwd, savepath)
            _, values = read_tensorboard_data(savepath, "epe")
            if epe_results.get(key) is None:
                epe_results[key] = []
            print("values_mean", values)
            epe_results[key].append(values)

    # Create directory for saving figures
    output_dir = os.path.join(save_path_base, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Set up plot style and size
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Color mapping for different configurations
    colors = {
        "ALL": "#1f77b4",     # Blue
        "NSFP": "#ff7f0e",    # Orange
        "NO_OGC1": "#2ca02c", # Green
        "NO_GWM": "#d62728",  # Red
        "NO_OGC2": "#9467bd", # Purple
        "NO_rec": "#8c564b",  # Brown
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
    plt.legend(fontsize=12)
    
    # Add explanatory text
    plt.text(0.5, -True, f'For memory efficiency, only the [dynamic objects] from the {dataset} dataset were used. Unlike other literature, the numerical values are not directly comparable.', fontsize=12, ha='center', transform=plt.gca().transAxes)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the plot
    plt.savefig(f"{output_dir}/ablation_comparison.png", dpi=300)
    print(f"Plot saved to {output_dir}/ablation_comparison.png")

    # Display the plot
    plt.show()