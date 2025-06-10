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
import concurrent.futures
from read_tensorboard import read_tensorboard_data

ab_name = "slot_size"  # Name for the ablation study
# Configuration for different ablation settings
# Each setting is a list of values for different loss terms:
# [reconstruction, flow, flow_smoothness, reconstruction_flow, point_smoothness]
slots = {
    "slot_5":5,
    "slot_10":10,
    "slot_20":20,
    "slot_25":25,
    "slot_30":30,
    "slot_35":35,
    # "slot_50":50,
    # "slot_100":100,
}
smooth_size = 7
# Color mapping for different configurations
colors = {
    "slot_5": "#ff7f0e",    # Orange
    "slot_10": "#1f77b4", # Blue
    "slot_20": "#2ca02c",  # Green
    "slot_25": "#d62728",   # Red
    "slot_30": "#9467bd",     # Purple
    "slot_35": "#8c564b",    # Brown
    # "LR100": "#e377c2",   # Pink
    # "LR1000": "#7f7f7f",  # Gray
}
scene_id_size = 5
scene_ids = [ str(i).zfill(6) for i in range(scene_id_size)]  # Scene IDs for the dataset
# scene_ids = ["000008"]
dataset_list = ["KITTISF"]  # Dataset to use for ablation study

# Get project root directory
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.join(cwd, "../")

max_workers = 5  # 并发数量，可根据需要调整

def run_command(args):
    command, cwd, savepath = args
    os.makedirs(savepath, exist_ok=True)
    print(command)
    result = subprocess.run(command, cwd=cwd, shell=True)
    return result.returncode

for dataset in dataset_list:
    save_path_base = os.path.join(cwd, f"../outputs/ablation/{ab_name}/{dataset}")
    commands = []
    for key in slots.keys():
        for scene_id in scene_ids:
            savepath = os.path.join(save_path_base, key, f"run_{scene_id}")
            if os.path.exists(savepath):
                continue
            slot_num = slots[key]
            command_list = [
                "python", "main.py",
                f"--config", "config/ablation_slot.yaml",
                f"log.dir={savepath}",
                f"dataset.name={dataset}",
                f"dataset.KITTISF.fixed_scene_id='{scene_id}'",
                f"model.mask.slot_num={slot_num}",
            ]
            command = " ".join(command_list)
            commands.append((command, cwd, savepath))

    # 并发执行命令
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(run_command, commands))

    # Dictionary to store EPE results
    epe_results = {}

    # Collect results for each configuration
    for key in slots.keys():
        # Run multiple times for each configuration
        for scene_id in scene_ids:
            savepath = os.path.join(save_path_base, key, f"run_{scene_id}")
            savepath = os.path.join(cwd, savepath)
            _, values = read_tensorboard_data(savepath, "epe")
            if epe_results.get(key) is None:
                epe_results[key] = []
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
            smoothed_run = np.convolve(run, np.ones(smooth_size*2+1)/(smooth_size*2+1), mode='same')
            # Handle edge cases by padding with edge values
            smoothed_run[:smooth_size] = run[smooth_size+1]
            smoothed_run[-smooth_size:] = run[-smooth_size-1]
            smoothed_runs.append(smoothed_run)
        # Calculate mean and bounds
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        min_value_6000 = np.min(mean_values[:6000])
        min_value_1500 = np.min(mean_values[:1500])
        print(f"Minimum EPE for first 6000 steps in '{key}': {min_value_6000}")
        print(f"Minimum EPE for first 1500 steps in '{key}': {min_value_1500}")
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
    for key in slots.keys():
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
            smoothed_run = np.convolve(run, np.ones(smooth_size*2+1)/(smooth_size*2+1), mode='same')
            # Handle edge cases by padding with edge values
            smoothed_run[:smooth_size] = run[smooth_size+1]
            smoothed_run[-smooth_size:] = run[-smooth_size-1]
            smoothed_runs.append(smoothed_run)
        
        # Convert to numpy array for calculations
        smoothed_runs = np.array(smoothed_runs)
        
        # Calculate mean and bounds
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        max_value_6000 = np.max(mean_values[:6000])
        max_value_1500 = np.max(mean_values[:1500])
        print(f"Max miou for first 6000 steps in '{key}': {max_value_6000}")
        print(f"Max miou for first 1500 steps in '{key}': {max_value_1500}")
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