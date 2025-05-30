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

ab_name = "alter_trainings"  # Name for the ablation study
# Configuration for different ablation settings
# Each setting is a list of values for different loss terms:
# [reconstruction, flow, flow_smoothness, reconstruction_flow, point_smoothness]
alters = {
    "train_both":[
        [[100,True]],
        [[100,True]]
        ],
    "alter_1":[
        [[1,True],[1,False]],
        [[1,False],[1,True]]
        ],
    "alter_10":[
        [[10,True],[10,False]],
        [[10,False],[10,True]]
        ],
    "alter_100":[
        [[100,True],[100,False]],
        [[100,False],[100,True]]
        ],
    "alter_1000":[
        [[1000,True],[1000,False]],
        [[1000,False],[1000,True]]
        ],
}
# Color mapping for different configurations
colors = {
    "train_both": "#ff7f0e",    # Orange
    "alter_1": "#1f77b4", # Blue
    "alter_10": "#2ca02c",  # Green
    "alter_100": "#d62728",   # Red
    "alter_1000": "#9467bd",     # Purple
    # "LR10": "#8c564b",    # Brown
    # "LR100": "#e377c2",   # Pink
    # "LR1000": "#7f7f7f",  # Gray
}
scene_id_size = 15
scene_ids = [ str(i).zfill(6) for i in range(scene_id_size)]  # Scene IDs for the dataset
# scene_ids = ["000008"]
dataset_list = ["KITTISF"]  # Dataset to use for ablation study

# Get project root directory
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.join(cwd, "../")

for dataset in dataset_list:
    save_path_base = os.path.join(cwd, f"../outputs/ablation/{ab_name}/{dataset}")
    
    for key in alters.keys():
        # Run multiple times for each configuration
        for scene_id in scene_ids:
            savepath = os.path.join(save_path_base, key, f"run_{scene_id}")
            if os.path.exists(savepath):
                continue
            os.makedirs(savepath, exist_ok=True)
            
            # Build command with appropriate loss weights
            flow_str = str(alters[key][0])
            mask_str = str(alters[key][1])
            print("flow_str", flow_str)
            command_list = [
                "python", "main.py",
                f"--config", "config/ablation_alter.yaml",
                f"log.dir={savepath}",
                f"dataset.name={dataset}",
                f"dataset.KITTISF.fixed_scene_id='{scene_id}'",
                f"alternate.flow=\"{flow_str}\"",
                f"alternate.mask=\"{mask_str}\"",
            ]
            
            # Execute training command
            command = " ".join(command_list)
            print(command)
            result = subprocess.run(command, cwd=cwd, shell=True)

    # Dictionary to store EPE results
    epe_results = {}

    # Collect results for each configuration
    for key in alters.keys():
        # Run multiple times for each configuration
        for scene_id in scene_ids:
            savepath = os.path.join(save_path_base, key, f"run_{scene_id}")
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
    for key in alters.keys():
        # Run multiple times for each configuration
        for scene_id in scene_ids:
            savepath = os.path.join(save_path_base, key, f"run_{scene_id}")
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