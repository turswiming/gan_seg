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
}
run_times = 3
# Color mapping for different configurations
colors = {
    "train_both": "#ff7f0e",    # Orange
    "alter_1": "#1f77b4", # Blue
    "alter_10": "#2ca02c",  # Green
    "alter_100": "#d62728",   # Red
    "alter_100": "#9467bd",     # Purple
    # "LR10": "#8c564b",    # Brown
    # "LR100": "#e377c2",   # Pink
    # "LR1000": "#7f7f7f",  # Gray
}

# Get project root directory
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.join(cwd, "../")

save_path_base = os.path.join(cwd, f"../outputs/ablation/{ab_name}/")

for i in range(run_times):
    for key in alters.keys():
        savepath = os.path.join(save_path_base, f"run_{i}", key)
        if os.path.exists(savepath):
            print(f"savepath {savepath} already exists")
            continue
        os.makedirs(savepath, exist_ok=True)
        
        # Build command with appropriate loss weights
        flow_str = str(alters[key][0])
        mask_str = str(alters[key][1])

        command_list = [
            "python", "main_general.py",
            f"--config", "config/ablation_alter.yaml",
            f"log.dir={savepath}",
            f"alternate.flow=\"{flow_str}\"",
            f"alternate.mask=\"{mask_str}\"",
            f"seed={i}",
        ]
        
        # Execute training command
        command = " ".join(command_list)
        print(command)
        result = subprocess.run(command, cwd=cwd, shell=True)
