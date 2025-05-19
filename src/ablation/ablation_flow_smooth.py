import subprocess
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

from read_tensorboard import read_tensorboard_data


lr_multi = { #rec   flow   flow_s rec_flow point_s
    "NSFP":[    0.0,    1.0,  0.0,    0.0,    0.0],
    "LR1":[    0.1,    1.0, 1.0,  0.0,    0.01],
    "LR10":[    0.1,    1.0, 10.0,  0.0,    0.01],
    "LR100":[    0.1,    1.0, 100.0,  0.0,    0.01],
    "LR1000":[    0.1,    1.0, 1000.0,  0.0,    0.01],
    "LR10000":[    0.1,    1.0, 10000.0,  0.0,    0.01],
    "LR100000":[    0.1,    1.0, 100000.0,  0.0,    0.01],
}
run_times = 2
dataset_list = ["AV2"]
cwd = os.path.dirname(os.path.abspath(__file__))
cwd = os.path.join(cwd, "../")
for dataset in dataset_list:
    save_path_base = os.path.join(cwd, f"../outputs/ablation/lr_flow_smooth/{dataset}")
    for key in lr_multi.keys():
        # Create a new directory for the current key
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            if os.path.exists(savepath):
                continue
            os.makedirs(savepath, exist_ok=True)
            # Create a new JSON file for the current key
            command_list = [
                "python","main.py",
                f"log.dir={savepath}",
                f"dataset.name={dataset}",
                f"lr_multi.rec_loss={lr_multi[key][0]}",
                f"lr_multi.flow_loss={lr_multi[key][1]}",
                f"lr_multi.flow_s_loss={lr_multi[key][2]}",
                f"lr_multi.rec_flow_loss={lr_multi[key][3]}",
                f"lr_multi.point_s_loss={lr_multi[key][4]}",
            ]
            command_list.append("model.mask.slot_num=30")
            # Run the command
            command = " ".join(command_list)
            print(command)


            result = subprocess.run(command, cwd=cwd, shell=True)


    epe_results = {

    }

    for key in lr_multi.keys():
        # Create a new directory for the current key
        for i in range(run_times):
            savepath = os.path.join(save_path_base, key, f"run_{i}")
            savepath = os.path.join(cwd, savepath)
            _, values = read_tensorboard_data(savepath, "epe")
            if epe_results.get(key) is None:
                epe_results[key] = []
            print("values_mean", values)
            epe_results[key].append(values)


    # 保存图表的目录
    output_dir = os.path.join(save_path_base, "figures")
    os.makedirs(output_dir, exist_ok=True)


    # 设置图表大小和样式
    plt.figure(figsize=(12, 8))
    plt.style.use('seaborn-v0_8-darkgrid')

    # 颜色映射
    colors = {
        "NSFP": "#ff7f0e",    # 橙色
        "LR1": "#2ca02c", # 绿色
        "LR10": "#d62728",  # 红色
        "LR100": "#9467bd", # 紫色
        "LR1000": "#8c564b",  # 棕色
        "LR10000": "#7f7f7f",   # 灰色
        "LR100000": "#1f77b4",     # 蓝色
    }

    # 确保所有运行结果长度相同
    max_length = 0
    for key in epe_results:
        for run in epe_results[key]:
            max_length = max(max_length, len(run))

    # 处理每个配置的数据
    for key in epe_results:
        all_runs = []
        for run in epe_results[key]:
            # 如果运行结果长度不够，用最后一个值填充
            if len(run) < max_length:
                run = run + [run[-1]] * (max_length - len(run))
            all_runs.append(run[:max_length])
        
        # 转换为numpy数组便于计算
        all_runs = np.array(all_runs)
        
        # 计算均值和上下界
        mean_values = np.mean(all_runs, axis=0)
        min_values = np.min(all_runs, axis=0)
        max_values = np.max(all_runs, axis=0)
        
        # X轴为训练步骤
        x = np.arange(len(mean_values))
        
        # 绘制均值线和上下界区域
        plt.plot(x, mean_values, label=key, color=colors.get(key), linewidth=2)
        plt.fill_between(x, min_values, max_values, alpha=0.2, color=colors.get(key))

    # Add chart elements
    plt.title('Comparison of EPE Metrics for Different Ablation Configurations', fontsize=16)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('EPE Value (Lower is Better)', fontsize=14)
    plt.legend(fontsize=12)
    # Add a line of explanatory text
    plt.text(0.5, -0.1, f'For memory efficiency, only the [dynamic objects] from the {dataset} dataset were used. Unlike other literature, the numerical values are not directly comparable.', fontsize=12, ha='center', transform=plt.gca().transAxes)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)



    # 保存图表
    plt.savefig(f"{output_dir}/ablation_comparison.png", dpi=300)
    print(f"图表已保存到 {output_dir}/ablation_comparison.png")

    # 显示图表
    plt.show()