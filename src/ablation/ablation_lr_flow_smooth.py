"""
Flow smoothness ablation study script.

This script performs ablation studies by running training with different weights
for the flow smoothness loss term, analyzing its impact on model performance.
"""

import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

from read_tensorboard import read_tensorboard_data

ab_name = "lr_flow_smooth_ego"  # Name for the ablation study
# Configuration for different ablation settings
# Each setting is a list of values for different loss terms:
# [reconstruction, flow, flow_smoothness, reconstruction_flow, point_smoothness]
lr_multi = {
    "NSFP":     [   0.0,    1.0,     0.0,    0.0,    0.0],  # Baseline without smoothness
    "LR0.001":  [   0.0,    1.0,     0.001,  0.0,    0.001],
    "LR0.01":   [   0.0,    1.0,     0.01,   0.0,    0.001], # Flow smoothness weight = 0.01
    "LR0.1":    [   0.0,    1.0,     0.1,    0.0,    0.001], # Flow smoothness weight = 0.1
    "LR1":      [   0.0,    1.0,     1.0,    0.0,    0.001], # Flow smoothness weight = 1
    # "LR10":     [   0.0,    1.0,    10.0,    0.0,    0.001], # Flow smoothness weight = 10
    # "LR100":    [   0.0,    1.0,   100.0,    0.0,    0.001], # Flow smoothness weight = 100
    # "LR1000":   [   0.0,    1.0,  1000.0,    0.0,    0.001], # Flow smoothness weight = 1000
}
# Color mapping for different configurations
colors = {
    "NSFP": "#ff7f0e",    # Orange
    "LR0.001": "#1f77b4", # Blue
    "LR0.01": "#2ca02c",  # Green
    "LR0.1": "#d62728",   # Red
    "LR1": "#9467bd",     # Purple
    "LR10": "#8c564b",    # Brown
    "LR100": "#e377c2",   # Pink
    "LR1000": "#7f7f7f",  # Gray
}
run_times = 5  # Number of runs for each configuration
dataset_list = ["AV2"]  # Dataset to use for ablation study

# Get project root directory
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.join(cwd, "../")

for dataset in dataset_list:
    save_path_base = os.path.join(cwd, f"../outputs/ablation/{ab_name}/{dataset}")
    
    for key in lr_multi.keys():
        # Run multiple times for each configuration
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            if os.path.exists(savepath):
                continue
            os.makedirs(savepath, exist_ok=True)
            
            # Build command with appropriate loss weights
            command_list = [
                "python", "main.py",
                f"--config", "config/config_lr_smooth_ab.yaml",
                f"log.dir={savepath}",
                f"dataset.name={dataset}",
                f"lr_multi.rec_loss={lr_multi[key][0]}",
                f"lr_multi.flow_loss={lr_multi[key][1]}",
                f"lr_multi.scene_flow_smoothness={lr_multi[key][2]}",
                f"lr_multi.rec_flow_loss={lr_multi[key][3]}",
                f"lr_multi.point_smooth_loss={lr_multi[key][4]}",
            ]
            
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
        smoothed_runs = []
        # Apply a moving average with a window size of 5
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            # Handle edge cases by padding with edge values
            smoothed_run[:2] = run[3]
            smoothed_run[-2:] = run[-3]
            smoothed_runs.append(smoothed_run)
        # Calculate mean and bounds
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        
        # X-axis represents training steps
        x = np.arange(len(mean_values))
        
        # Plot mean line and confidence region for EPE
        plt.plot(x, mean_values, label=f"{key} (EPE)", color=colors.get(key), linewidth=2)
        plt.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))

    # Add chart elements for EPE
    plt.title('Comparison of EPE Metrics for Different Ablation Configurations', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('EPE Value (Lower is Better)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the EPE plot
    epe_output_path = f"{output_dir}/epe_comparison.png"
    plt.savefig(epe_output_path, dpi=300)
    print(f"EPE plot saved to {epe_output_path}")

    # Display the EPE plot
    plt.show()

    # Dictionary to store mIoU results
    miou_results = {}

    # Collect mIoU results for each configuration
    for key in lr_multi.keys():
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            savepath = os.path.join(cwd, savepath)
            _, values = read_tensorboard_data(savepath, "miou")
            if miou_results.get(key) is None:
                miou_results[key] = []
            miou_results[key].append(values)

    # Ensure all mIoU results have the same length
    max_length_miou = 0
    for key in miou_results:
        for run in miou_results[key]:
            max_length_miou = max(max_length_miou, len(run))

    # Set up a new plot for mIoU
    plt.figure(figsize=(12, 8))

    # Process mIoU data for each configuration
    for key in miou_results:
        all_runs = []
        for run in miou_results[key]:
            # Pad shorter runs with their last value
            if len(run) < max_length_miou:
                run = run + [run[-1]] * (max_length_miou - len(run))
            all_runs.append(run[:max_length_miou])
        
        # Convert to numpy array for calculations
        all_runs = np.array(all_runs)
        
        # Calculate mean and bounds
        mean_values = np.mean(all_runs, axis=0)
        min_values = np.min(all_runs, axis=0)
        # Apply a moving average with a window size of 5
        smoothed_runs = []
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            # Handle edge cases by padding with edge values
            smoothed_run[:2] = run[3]
            smoothed_run[-2:] = run[-3]
            smoothed_runs.append(smoothed_run)
        
        # Convert to numpy array for calculations
        smoothed_runs = np.array(smoothed_runs)
        
        # Calculate mean and bounds
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        
        # X-axis represents training steps
        x = np.arange(len(mean_values))
        
        # Plot mean line and confidence region for mIoU
        plt.plot(x, mean_values, label=f"{key} (mIoU)", linestyle='--', color=colors.get(key), linewidth=2)
        plt.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))

    # Add chart elements for mIoU
    plt.title('Comparison of mIoU Metrics for Different Ablation Configurations', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('mIoU Value (Higher is Better)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the mIoU plot
    miou_output_path = f"{output_dir}/miou_comparison.png"
    plt.savefig(miou_output_path, dpi=300)
    print(f"mIoU plot saved to {miou_output_path}")

    # Display the mIoU plot
    plt.show()