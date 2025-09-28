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
skip_training = False
ab_name = "flow_smooth"  # Name for the ablation study
# Configuration for different ablation settings
# Each setting is a list of values for different loss terms:
# [reconstruction, flow, flow_smoothness, reconstruction_flow, point_smoothness]
lr_multi = {
    "ours_0.01":     {
        "scene_flow_smoothness": 0.01,
        "l1_regularization": 0.0,
        "begin_train_smooth": 4000,
        "alternate_flow": [[200,True],[200,True]],
        "alternate_mask": [[4000,False],[200000,True]],
    },
    "ours_0.1":     {
        "scene_flow_smoothness": 0.1,
        "l1_regularization": 0.0,
        "begin_train_smooth": 4000,
        "alternate_flow": [[200,True],[200,True]],
        "alternate_mask": [[4000,False],[200000,True]],
    },

    "eularflow":     {
        "scene_flow_smoothness": 0.0,
        "l1_regularization": 0.0,
        "begin_train_smooth": 0,
        "alternate_flow": [[200,True],[200,True]],
        "alternate_mask": [[4000,False],[200000,False]],
    },
    # "ours_1":     {
    #     "scene_flow_smoothness": 1.0,
    #     "l1_regularization": 0.01,
    #     "begin_train_smooth": 4000,
    #     "alternate_flow": [[200,True],[200,True]],
    #     "alternate_mask": [[4000,False],[200000,True]],
    # },
}  # Baseline without smoothness

# Color mapping for different configurations
colors = {
    "eularflow": "#ff6b6b",    # Red
    "eularflow_l1": "#4ecdc4", # Teal
    "ours_0.0001": "#45b7d1",   # Blue
    "ours_0.001": "#45b7d1",   # Blue
    "ours_0.01": "#f39c12",   # Orange
    "ours_0.01_alter": "#9b59b6",   # Purple
    "ours_0.1": "#e74c3c",   # Dark Red
    # "ours_0.01": "#9b59b6",   # Purple
    # "ours_0.01": "#2ecc71",   # Green
    # "ours_0.01": "#f1c40f",     # Yellow
    # "ours_1": "#8e44ad",    # Dark Purple
}
run_times = 1  # Number of runs for each configuration
dataset_list = ["AV2Sequence"]  # Dataset to use for ablation study

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
            
            # Build command with appropriate loss weights
            command_list = [
                "python", "main.py",
                f"--config", "config/altereular.yaml",
                f"log.dir={savepath}",
                f"dataset.name={dataset}",
                f"lr_multi.scene_flow_smoothness={lr_multi[key]['scene_flow_smoothness']}",
                f"lr_multi.l1_regularization={lr_multi[key]['l1_regularization']}",
                f"training.begin_train_smooth={lr_multi[key]['begin_train_smooth']}",
                f"alternate.flow=\"{lr_multi[key]['alternate_flow']}\"",
                f"alternate.mask=\"{lr_multi[key]['alternate_mask']}\"",
            ]
            
            # Execute training command
            command = " ".join(command_list)
            print(command)
            if skip_training:
                continue
            os.makedirs(savepath, exist_ok=True)
            result = subprocess.run(command, cwd=cwd, shell=True)

    # Dictionary to store EPE results
    epe_results = {}

    # Collect results for each configuration
    for key in lr_multi.keys():
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            savepath = os.path.join(cwd, savepath)
            if not os.path.exists(savepath):
                continue
            _, values = read_tensorboard_data(savepath, "epe")
            if epe_results.get(key) is None:
                epe_results[key] = []
            epe_results[key].append(values)

    val_threeway_mean_results = {}
    for key in lr_multi.keys():
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            savepath = os.path.join(cwd, savepath)
            if not os.path.exists(savepath):
                continue
            _, values = read_tensorboard_data(savepath, "val_threeway_mean")
            if val_threeway_mean_results.get(key) is None:
                val_threeway_mean_results[key] = []
            val_threeway_mean_results[key].append(values)
    val_miou_results = {}
    for key in lr_multi.keys():
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            savepath = os.path.join(cwd, savepath)
            if not os.path.exists(savepath):
                continue
            _, values = read_tensorboard_data(savepath, "val_miou")
            if val_miou_results.get(key) is None:
                val_miou_results[key] = []
            val_miou_results[key].append(values)
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
            if len(run) > 0:
                max_length = max(max_length, len(run))
    
    # If no data found, use default length
    if max_length == 0:
        max_length = 100

    # Process data for each configuration
    for key in epe_results:
        all_runs = []
        for run in epe_results[key]:
            if len(run) == 0:
                continue
            # Pad shorter runs with their last value
            if len(run) < max_length:
                run = run + [run[-1]] * (max_length - len(run))
            all_runs.append(run[:max_length])
        
        # Skip if no valid runs
        if len(all_runs) == 0:
            print(f"No valid EPE data for {key}, skipping...")
            continue
            
        # Convert to numpy array for calculations
        all_runs = np.array(all_runs)
        smoothed_runs = []
        # Apply a moving average with a window size of 5
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            # Handle edge cases by padding with edge values
            if len(run) > 2:
                smoothed_run[:2] = run[2]
                smoothed_run[-2:] = run[-3]
            smoothed_runs.append(smoothed_run)
        
        # Skip if no smoothed runs
        if len(smoothed_runs) == 0:
            continue
            
        # Calculate mean and bounds
        smoothed_runs = np.array(smoothed_runs)
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
            if not os.path.exists(savepath):
                continue
            _, values = read_tensorboard_data(savepath, "miou")
            if miou_results.get(key) is None:
                miou_results[key] = []
            miou_results[key].append(values)

    # Ensure all mIoU results have the same length
    max_length_miou = 0
    for key in miou_results:
        for run in miou_results[key]:
            if len(run) > 0:
                max_length_miou = max(max_length_miou, len(run))
    
    # If no data found, use default length
    if max_length_miou == 0:
        max_length_miou = 100

    # Set up a new plot for mIoU
    plt.figure(figsize=(12, 8))

    # Process mIoU data for each configuration
    for key in miou_results:
        all_runs = []
        for run in miou_results[key]:
            if len(run) == 0:
                continue
            # Pad shorter runs with their last value
            if len(run) < max_length_miou:
                run = run + [run[-1]] * (max_length_miou - len(run))
            all_runs.append(run[:max_length_miou])
        
        # Skip if no valid runs
        if len(all_runs) == 0:
            print(f"No valid mIoU data for {key}, skipping...")
            continue
            
        # Convert to numpy array for calculations
        all_runs = np.array(all_runs)
        
        # Apply a moving average with a window size of 5
        smoothed_runs = []
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            # Handle edge cases by padding with edge values
            if len(run) > 2:
                smoothed_run[:2] = run[2]
                smoothed_run[-2:] = run[-3]
            smoothed_runs.append(smoothed_run)
        
        # Skip if no smoothed runs
        if len(smoothed_runs) == 0:
            continue
            
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

    # Create three-way EPE visualization
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Ensure all three-way EPE results have the same length
    max_length_threeway = 0
    for key in val_threeway_mean_results:
        for run in val_threeway_mean_results[key]:
            if len(run) > 0:  # Check if data exists
                max_length_threeway = max(max_length_threeway, len(run))

    # Process three-way EPE data for each configuration
    for key in val_threeway_mean_results:
        if not val_threeway_mean_results[key] or all(len(run) == 0 for run in val_threeway_mean_results[key]):
            print(f"No three-way EPE data found for {key}, skipping...")
            continue
            
        all_runs = []
        for run in val_threeway_mean_results[key]:
            if len(run) == 0:
                continue
            # Pad shorter runs with their last value
            if len(run) < max_length_threeway:
                run = run + [run[-1]] * (max_length_threeway - len(run))
            all_runs.append(run[:max_length_threeway])
        
        if not all_runs:
            continue
            
        # Convert to numpy array for calculations
        all_runs = np.array(all_runs)
        smoothed_runs = []
        # Apply a moving average with a window size of 5
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            # Handle edge cases by padding with edge values
            if len(run) > 2:
                smoothed_run[:2] = run[2] if len(run) > 2 else run[0]
                smoothed_run[-2:] = run[-3] if len(run) > 2 else run[-1]
            smoothed_runs.append(smoothed_run)
        
        # Calculate mean and bounds
        smoothed_runs = np.array(smoothed_runs)
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        
        # X-axis represents training steps
        x = np.arange(len(mean_values))
        
        # Plot mean line and confidence region for three-way EPE
        plt.plot(x, mean_values, label=f"{key} (3-way EPE)", color=colors.get(key), linewidth=2, linestyle='-.')
        plt.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))

    # Add chart elements for three-way EPE
    plt.title('Comparison of Three-way EPE Metrics for Different Ablation Configurations', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Three-way EPE Value (Lower is Better)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the three-way EPE plot
    threeway_epe_output_path = f"{output_dir}/threeway_epe_comparison.png"
    plt.savefig(threeway_epe_output_path, dpi=300)
    print(f"Three-way EPE plot saved to {threeway_epe_output_path}")

    # Display the three-way EPE plot
    plt.show()

    # Create validation mIoU visualization
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')

    # Ensure all validation mIoU results have the same length
    max_length_val_miou = 0
    for key in val_miou_results:
        for run in val_miou_results[key]:
            if len(run) > 0:  # Check if data exists
                max_length_val_miou = max(max_length_val_miou, len(run))

    # Process validation mIoU data for each configuration
    for key in val_miou_results:
        if not val_miou_results[key] or all(len(run) == 0 for run in val_miou_results[key]):
            print(f"No validation mIoU data found for {key}, skipping...")
            continue
            
        all_runs = []
        for run in val_miou_results[key]:
            if len(run) == 0:
                continue
            # Pad shorter runs with their last value
            if len(run) < max_length_val_miou:
                run = run + [run[-1]] * (max_length_val_miou - len(run))
            all_runs.append(run[:max_length_val_miou])
        
        if not all_runs:
            continue
            
        # Convert to numpy array for calculations
        all_runs = np.array(all_runs)
        smoothed_runs = []
        # Apply a moving average with a window size of 5
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            # Handle edge cases by padding with edge values
            if len(run) > 2:
                smoothed_run[:2] = run[2] if len(run) > 2 else run[0]
                smoothed_run[-2:] = run[-3] if len(run) > 2 else run[-1]
            smoothed_runs.append(smoothed_run)
        
        # Calculate mean and bounds
        smoothed_runs = np.array(smoothed_runs)
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        
        # X-axis represents training steps
        x = np.arange(len(mean_values))
        
        # Plot mean line and confidence region for validation mIoU
        plt.plot(x, mean_values, label=f"{key} (val mIoU)", color=colors.get(key), linewidth=2, linestyle=':')
        plt.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))

    # Add chart elements for validation mIoU
    plt.title('Comparison of Validation mIoU Metrics for Different Ablation Configurations', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Validation mIoU Value (Higher is Better)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Save the validation mIoU plot
    val_miou_output_path = f"{output_dir}/val_miou_comparison.png"
    plt.savefig(val_miou_output_path, dpi=300)
    print(f"Validation mIoU plot saved to {val_miou_output_path}")

    # Display the validation mIoU plot
    plt.show()

    # Create combined plot with all metrics
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Comprehensive Ablation Study Results', fontsize=20)

    # Plot 1: EPE
    ax1 = axes[0, 0]
    for key in epe_results:
        all_runs = []
        for run in epe_results[key]:
            if len(run) < max_length:
                run = run + [run[-1]] * (max_length - len(run))
            all_runs.append(run[:max_length])
        
        all_runs = np.array(all_runs)
        smoothed_runs = []
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            smoothed_run[:2] = run[2] if len(run) > 2 else run[0]
            smoothed_run[-2:] = run[-3] if len(run) > 2 else run[-1]
            smoothed_runs.append(smoothed_run)
        
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        x = np.arange(len(mean_values))
        
        ax1.plot(x, mean_values, label=f"{key}", color=colors.get(key), linewidth=2)
        ax1.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))
    
    ax1.set_title('EPE Comparison', fontsize=14)
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('EPE (Lower is Better)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot 2: Three-way EPE
    ax2 = axes[0, 1]
    for key in val_threeway_mean_results:
        if not val_threeway_mean_results[key] or all(len(run) == 0 for run in val_threeway_mean_results[key]):
            continue
            
        all_runs = []
        for run in val_threeway_mean_results[key]:
            if len(run) == 0:
                continue
            if len(run) < max_length_threeway:
                run = run + [run[-1]] * (max_length_threeway - len(run))
            all_runs.append(run[:max_length_threeway])
        
        if not all_runs:
            continue
            
        all_runs = np.array(all_runs)
        smoothed_runs = []
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            if len(run) > 2:
                smoothed_run[:2] = run[2]
                smoothed_run[-2:] = run[-3]
            smoothed_runs.append(smoothed_run)
        
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        x = np.arange(len(mean_values))
        
        ax2.plot(x, mean_values, label=f"{key}", color=colors.get(key), linewidth=2)
        ax2.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))
    
    ax2.set_title('Three-way EPE Comparison', fontsize=14)
    ax2.set_xlabel('Training Steps', fontsize=12)
    ax2.set_ylabel('3-way EPE (Lower is Better)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)

    # Plot 3: Training mIoU
    ax3 = axes[1, 0]
    for key in miou_results:
        if not miou_results[key]:
            continue
            
        all_runs = []
        for run in miou_results[key]:
            if len(run) < max_length_miou:
                run = run + [run[-1]] * (max_length_miou - len(run))
            all_runs.append(run[:max_length_miou])
        
        all_runs = np.array(all_runs)
        smoothed_runs = []
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            smoothed_run[:2] = run[2] if len(run) > 2 else run[0]
            smoothed_run[-2:] = run[-3] if len(run) > 2 else run[-1]
            smoothed_runs.append(smoothed_run)
        
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        x = np.arange(len(mean_values))
        
        ax3.plot(x, mean_values, label=f"{key}", color=colors.get(key), linewidth=2)
        ax3.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))
    
    ax3.set_title('Training mIoU Comparison', fontsize=14)
    ax3.set_xlabel('Training Steps', fontsize=12)
    ax3.set_ylabel('mIoU (Higher is Better)', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, linestyle='--', alpha=0.7)

    # Plot 4: Validation mIoU
    ax4 = axes[1, 1]
    for key in val_miou_results:
        if not val_miou_results[key] or all(len(run) == 0 for run in val_miou_results[key]):
            continue
            
        all_runs = []
        for run in val_miou_results[key]:
            if len(run) == 0:
                continue
            if len(run) < max_length_val_miou:
                run = run + [run[-1]] * (max_length_val_miou - len(run))
            all_runs.append(run[:max_length_val_miou])
        
        if not all_runs:
            continue
            
        all_runs = np.array(all_runs)
        smoothed_runs = []
        for run in all_runs:
            smoothed_run = np.convolve(run, np.ones(5)/5, mode='same')
            if len(run) > 2:
                smoothed_run[:2] = run[2]
                smoothed_run[-2:] = run[-3]
            smoothed_runs.append(smoothed_run)
        
        mean_values = np.mean(smoothed_runs, axis=0)
        min_values = np.min(smoothed_runs, axis=0)
        max_values = np.max(smoothed_runs, axis=0)
        x = np.arange(len(mean_values))
        
        ax4.plot(x, mean_values, label=f"{key}", color=colors.get(key), linewidth=2)
        ax4.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))
    
    ax4.set_title('Validation mIoU Comparison', fontsize=14)
    ax4.set_xlabel('Training Steps', fontsize=12)
    ax4.set_ylabel('Val mIoU (Higher is Better)', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, linestyle='--', alpha=0.7)

    # Adjust layout and save combined plot
    plt.tight_layout()
    combined_output_path = f"{output_dir}/combined_metrics_comparison.png"
    plt.savefig(combined_output_path, dpi=300, bbox_inches='tight')
    print(f"Combined metrics plot saved to {combined_output_path}")

    # Display the combined plot
    plt.show()