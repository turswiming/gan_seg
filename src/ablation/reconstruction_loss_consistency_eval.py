# filepath: /workspace/gan_seg/src/ablation/reconstruction_loss_consistency_eval.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import functools
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from losses.ReconstructionLoss import ReconstructionLoss
from losses.ReconstructionLoss_optimized import ReconstructionLossOptimized
from utils.config_utils import load_config_with_inheritance
from utils.dataloader_utils import create_dataloaders
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from Predictor import get_scene_flow_predictor, get_mask_predictor
from model.eulerflow_raw_mlp import QueryDirection

def remap_instance_labels(labels):
    """
    将任意整数标签重映射为连续的标签编号，从0开始
    例如: [0,1,8,1] -> [0,1,2,1]
    
    Args:
        labels: 输入标签张量
    
    Returns:
        重映射后的标签张量
    """
    unique_labels = torch.unique(labels)
    mapping = {label.item(): idx for idx, label in enumerate(sorted(unique_labels))}
    print(f"remap {mapping}")
    # 创建新的标签张量
    remapped = torch.zeros_like(labels)
    for old_label, new_label in mapping.items():
        remapped[labels == old_label] = new_label
        
    return remapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config_obj = load_config_with_inheritance("/workspace/gan_seg/src/config/baseeular.yaml")
config_obj.dataloader.batchsize = 4
dataloader, infinite_loader, val_dataloader, batch_size, N = create_dataloaders(config_obj)

# Initialize both reconstruction loss implementations
reconstruction_loss_ref = ReconstructionLoss(device)
reconstruction_loss_opt = ReconstructionLossOptimized(device, use_checkpointing=False, chunk_size=2048)

# Initialize models
flow_model = get_scene_flow_predictor(config_obj.model.flow, N)
flow_model.to(device)
mask_model = get_mask_predictor(config_obj.model.mask, N)
mask_model.to(device)

bucket_size = 10
max_sample = 10

# Initialize accumulators for average losses and timing
all_losses_ref = []
all_losses_opt = []
all_loss_diffs = []
all_times_ref = []
all_times_opt = []
sample_count = 0

# Start overall timing
overall_start_time = time.time()

print("Processing validation samples...")

# Iterate through all validation samples
for val_sample in val_dataloader:
    sample_count += 1
    if sample_count > max_sample:
        break
    print(f"Processing sample {sample_count}")

    # Get ground truth data
    mask = val_sample["dynamic_instance_mask"][0].to(device)
    mask = remap_instance_labels(mask)
    one_hot_mask = F.one_hot(mask).permute(1, 0)
    one_hot_mask = one_hot_mask.float()
    
    # Get predicted data
    mask_noise = mask_model(val_sample["point_cloud_first"][0].to(device), val_sample["idx"][0], val_sample["total_frames"][0])
    mask_noise = mask_noise.permute(1, 0)[:one_hot_mask.shape[0],:].to(device)

    N = val_sample["point_cloud_first"][0].shape[0]
    flow_noise = flow_model(val_sample["point_cloud_first"][0].to(device), val_sample["idx"][0], val_sample["total_frames"][0], QueryDirection.FORWARD)
    flow_noise = flow_noise.to(device)
    
    # Get point clouds
    point_cloud_first = val_sample["point_cloud_first"][0].to(device)
    point_cloud_second = val_sample["self"][0].get_item(val_sample["idx"][0]+1)["point_cloud_first"].to(device)
    gt_flow = val_sample["flow"][0].to(device).squeeze(0)

    # Calculate losses for this sample across different noise levels
    sample_losses_ref = []
    sample_losses_opt = []
    sample_loss_diffs = []
    sample_times_ref = []
    sample_times_opt = []
    
    for i in range(bucket_size):
        loss_ref_row = []
        loss_opt_row = []
        loss_diff_row = []
        time_ref_row = []
        time_opt_row = []
        
        for j in range(bucket_size):
            # Scale flow noise
            flow_scaled = flow_noise * i / bucket_size * 0.01
            flow = flow_scaled + gt_flow * (1 - i / bucket_size)
            
            # Scale mask noise
            mask_noise_scaled = mask_noise * (j / bucket_size + 1e-4)
            mask_current = mask_noise_scaled + one_hot_mask * (1 - j / bucket_size)
            
            # Convert to list format for loss functions
            pc1_list = [point_cloud_first]
            pc2_list = [point_cloud_second]
            mask_list = [mask_current]
            flow_list = [flow]
            
            # Calculate both loss implementations with timing
            try:
                # Time reference implementation
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time_ref = time.time()
                loss_ref_val, rec_ref = reconstruction_loss_ref(pc1_list, pc2_list, mask_list, flow_list)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time_ref = time.time()
                time_ref = end_time_ref - start_time_ref
                
                # Time optimized implementation
                torch.cuda.synchronize() if device.type == 'cuda' else None
                start_time_opt = time.time()
                loss_opt_val, rec_opt = reconstruction_loss_opt(pc1_list, pc2_list, mask_list, flow_list)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time_opt = time.time()
                time_opt = end_time_opt - start_time_opt
                
                loss_ref_row.append(loss_ref_val.item())
                loss_opt_row.append(loss_opt_val.item())
                loss_diff_row.append(abs(loss_ref_val.item() - loss_opt_val.item()))
                time_ref_row.append(time_ref)
                time_opt_row.append(time_opt)
                
            except Exception as e:
                print(f"Error at flow_scale={i}, mask_scale={j}: {e}")
                loss_ref_row.append(0.0)
                loss_opt_row.append(0.0)
                loss_diff_row.append(0.0)
                time_ref_row.append(0.0)
                time_opt_row.append(0.0)
        
        sample_losses_ref.append(loss_ref_row)
        sample_losses_opt.append(loss_opt_row)
        sample_loss_diffs.append(loss_diff_row)
        sample_times_ref.append(time_ref_row)
        sample_times_opt.append(time_opt_row)
    
    print(f"Sample {sample_count} - Mean ref loss: {np.mean(sample_losses_ref):.6f}, Mean opt loss: {np.mean(sample_losses_opt):.6f}")
    print(f"Sample {sample_count} - Mean ref time: {np.mean(sample_times_ref):.6f}s, Mean opt time: {np.mean(sample_times_opt):.6f}s")
    all_losses_ref.append(sample_losses_ref)
    all_losses_opt.append(sample_losses_opt)
    all_loss_diffs.append(sample_loss_diffs)
    all_times_ref.append(sample_times_ref)
    all_times_opt.append(sample_times_opt)

print(f"Processed {sample_count} validation samples")

# Calculate average losses and times across all samples
losses_ref = np.mean(np.array(all_losses_ref), axis=0)
losses_opt = np.mean(np.array(all_losses_opt), axis=0)
loss_diffs = np.mean(np.array(all_loss_diffs), axis=0)
times_ref = np.mean(np.array(all_times_ref), axis=0)
times_opt = np.mean(np.array(all_times_opt), axis=0)

# Calculate total runtime
overall_end_time = time.time()
total_runtime = overall_end_time - overall_start_time

# Reshape losses and times into 2D arrays for plotting
losses_ref = np.array(losses_ref).reshape(bucket_size, bucket_size)
losses_opt = np.array(losses_opt).reshape(bucket_size, bucket_size)
loss_diffs = np.array(loss_diffs).reshape(bucket_size, bucket_size)
times_ref = np.array(times_ref).reshape(bucket_size, bucket_size)
times_opt = np.array(times_opt).reshape(bucket_size, bucket_size)

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(20, 15))

# 1. Reference Implementation 3D Surface
ax1 = fig.add_subplot(3, 4, 1, projection='3d')
x = np.linspace(0, 1, bucket_size)
y = np.linspace(0, 1, bucket_size)
X, Y = np.meshgrid(x, y)
surf1 = ax1.plot_surface(X, Y, losses_ref, cmap='viridis', edgecolor='none')
ax1.set_xlabel('Mask Noise Scale')
ax1.set_ylabel('Flow Noise Scale')
ax1.set_zlabel('Loss Value')
ax1.set_title(f'Reference Implementation\n(Avg over {sample_count} samples)')

# 2. Optimized Implementation 3D Surface
ax2 = fig.add_subplot(3, 4, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, losses_opt, cmap='plasma', edgecolor='none')
ax2.set_xlabel('Mask Noise Scale')
ax2.set_ylabel('Flow Noise Scale')
ax2.set_zlabel('Loss Value')
ax2.set_title(f'Optimized Implementation\n(Avg over {sample_count} samples)')

# 3. Difference 3D Surface
ax3 = fig.add_subplot(3, 4, 3, projection='3d')
surf3 = ax3.plot_surface(X, Y, loss_diffs, cmap='coolwarm', edgecolor='none')
ax3.set_xlabel('Mask Noise Scale')
ax3.set_ylabel('Flow Noise Scale')
ax3.set_zlabel('Absolute Difference')
ax3.set_title(f'Implementation Difference\n|Ref - Opt|')

# 4. Reference Implementation Heatmap
ax4 = fig.add_subplot(3, 4, 4)
heatmap_ref = ax4.imshow(losses_ref, cmap='viridis', origin='lower', 
                        extent=[0, 1, 0, 1], aspect='auto')
ax4.set_xlabel('Mask Noise Scale')
ax4.set_ylabel('Flow Noise Scale')
ax4.set_title('Reference Implementation Heatmap')
plt.colorbar(heatmap_ref, ax=ax4, label='Loss Value')

# 5. Optimized Implementation Heatmap
ax5 = fig.add_subplot(3, 4, 5)
heatmap_opt = ax5.imshow(losses_opt, cmap='plasma', origin='lower', 
                        extent=[0, 1, 0, 1], aspect='auto')
ax5.set_xlabel('Mask Noise Scale')
ax5.set_ylabel('Flow Noise Scale')
ax5.set_title('Optimized Implementation Heatmap')
plt.colorbar(heatmap_opt, ax=ax5, label='Loss Value')

# 6. Difference Heatmap
ax6 = fig.add_subplot(3, 4, 6)
heatmap_diff = ax6.imshow(loss_diffs, cmap='coolwarm', origin='lower', 
                         extent=[0, 1, 0, 1], aspect='auto')
ax6.set_xlabel('Mask Noise Scale')
ax6.set_ylabel('Flow Noise Scale')
ax6.set_title('Implementation Difference Heatmap')
plt.colorbar(heatmap_diff, ax=ax6, label='Absolute Difference')

# 7. Reference Implementation Contour
ax7 = fig.add_subplot(3, 4, 7)
contour_ref = ax7.contour(X, Y, losses_ref, levels=20, colors='black', linewidths=0.5)
ax7.contourf(X, Y, losses_ref, levels=20, cmap='viridis', alpha=0.8)
ax7.set_xlabel('Mask Noise Scale')
ax7.set_ylabel('Flow Noise Scale')
ax7.set_title('Reference Implementation Contour')
ax7.clabel(contour_ref, inline=True, fontsize=8, fmt='%.4f')

# 8. Optimized Implementation Contour
ax8 = fig.add_subplot(3, 4, 8)
contour_opt = ax8.contour(X, Y, losses_opt, levels=20, colors='black', linewidths=0.5)
ax8.contourf(X, Y, losses_opt, levels=20, cmap='plasma', alpha=0.8)
ax8.set_xlabel('Mask Noise Scale')
ax8.set_ylabel('Flow Noise Scale')
ax8.set_title('Optimized Implementation Contour')
ax8.clabel(contour_opt, inline=True, fontsize=8, fmt='%.4f')

# 9. Difference Contour
ax9 = fig.add_subplot(3, 4, 9)
contour_diff = ax9.contour(X, Y, loss_diffs, levels=20, colors='black', linewidths=0.5)
ax9.contourf(X, Y, loss_diffs, levels=20, cmap='coolwarm', alpha=0.8)
ax9.set_xlabel('Mask Noise Scale')
ax9.set_ylabel('Flow Noise Scale')
ax9.set_title('Implementation Difference Contour')
ax9.clabel(contour_diff, inline=True, fontsize=8, fmt='%.6f')

# 10. Statistical Analysis
ax10 = fig.add_subplot(3, 4, 10)
ax10.axis('off')

# Calculate statistics
max_diff = np.max(loss_diffs)
mean_diff = np.mean(loss_diffs)
std_diff = np.std(loss_diffs)
max_diff_idx = np.unravel_index(np.argmax(loss_diffs), loss_diffs.shape)

# Calculate timing statistics
total_time_ref = np.sum(times_ref)
total_time_opt = np.sum(times_opt)
mean_time_ref = np.mean(times_ref)
mean_time_opt = np.mean(times_opt)
speedup_ratio = total_time_ref / total_time_opt if total_time_opt > 0 else float('inf')

stats_text = f"""
STATISTICAL ANALYSIS
===================
Max Difference: {max_diff:.6f}
Mean Difference: {mean_diff:.6f}
Std Difference: {std_diff:.6f}

Max Diff Location:
  Flow Scale: {max_diff_idx[1]/bucket_size:.2f}
  Mask Scale: {max_diff_idx[0]/bucket_size:.2f}

TIMING ANALYSIS
==============
Total Runtime: {total_runtime:.2f}s
Reference Total Time: {total_time_ref:.4f}s
Optimized Total Time: {total_time_opt:.4f}s
Speedup Ratio: {speedup_ratio:.2f}x

Mean Time per Call:
  Reference: {mean_time_ref:.6f}s
  Optimized: {mean_time_opt:.6f}s

Reference Loss:
  Min: {np.min(losses_ref):.6f}
  Max: {np.max(losses_ref):.6f}
  Mean: {np.mean(losses_ref):.6f}

Optimized Loss:
  Min: {np.min(losses_opt):.6f}
  Max: {np.max(losses_opt):.6f}
  Mean: {np.mean(losses_opt):.6f}

Consistency Check:
  Max Diff < 1e-5: {max_diff < 1e-5}
  Mean Diff < 1e-6: {mean_diff < 1e-6}
"""

ax10.text(0.05, 0.95, stats_text, transform=ax10.transAxes, fontsize=9,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))

# 11. Line plots for specific mask noise levels
ax11 = fig.add_subplot(3, 4, 11)
for mask_scale in [0, bucket_size//2, bucket_size-1]:
    mask_idx = mask_scale
    ax11.plot(x, losses_ref[mask_idx, :], 'o-', label=f'Ref (mask={mask_scale/bucket_size:.1f})', linewidth=2)
    ax11.plot(x, losses_opt[mask_idx, :], 's-', label=f'Opt (mask={mask_scale/bucket_size:.1f})', linewidth=2)
ax11.set_xlabel('Flow Noise Scale')
ax11.set_ylabel('Loss Value')
ax11.set_title('Loss vs Flow Noise (Fixed Mask)')
ax11.legend()
ax11.grid(True, alpha=0.3)

# 12. Line plots for specific flow noise levels
ax12 = fig.add_subplot(3, 4, 12)
for flow_scale in [0, bucket_size//2, bucket_size-1]:
    flow_idx = flow_scale
    ax12.plot(y, losses_ref[:, flow_idx], 'o-', label=f'Ref (flow={flow_scale/bucket_size:.1f})', linewidth=2)
    ax12.plot(y, losses_opt[:, flow_idx], 's-', label=f'Opt (flow={flow_scale/bucket_size:.1f})', linewidth=2)
ax12.set_xlabel('Mask Noise Scale')
ax12.set_ylabel('Loss Value')
ax12.set_title('Loss vs Mask Noise (Fixed Flow)')
ax12.legend()
ax12.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'reconstruction_loss_consistency_eval_{sample_count}_samples.png', dpi=300, bbox_inches='tight')
print(f"Visualization saved as 'reconstruction_loss_consistency_eval_{sample_count}_samples.png'")

# Print summary
print(f"\nSUMMARY:")
print(f"Processed {sample_count} validation samples")
print(f"Total runtime: {total_runtime:.2f} seconds")
print(f"")
print(f"LOSS CONSISTENCY:")
print(f"Maximum difference between implementations: {max_diff:.8f}")
print(f"Mean difference: {mean_diff:.8f}")
print(f"Standard deviation of differences: {std_diff:.8f}")
print(f"Implementations are consistent within 1e-5: {max_diff < 1e-5}")
print(f"Implementations are consistent within 1e-6: {max_diff < 1e-6}")
print(f"")
print(f"TIMING PERFORMANCE:")
print(f"Reference implementation total time: {total_time_ref:.4f}s")
print(f"Optimized implementation total time: {total_time_opt:.4f}s")
print(f"Speedup ratio: {speedup_ratio:.2f}x")
print(f"Mean time per call - Reference: {mean_time_ref:.6f}s")
print(f"Mean time per call - Optimized: {mean_time_opt:.6f}s")
print(f"Time efficiency: Optimized is {speedup_ratio:.2f}x faster than Reference")