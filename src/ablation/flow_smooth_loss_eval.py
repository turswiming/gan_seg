# filepath: /home/lzq/workspace/gan_seg/src/ablation/flow_smooth_loss_eval.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import functools
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from losses.FlowSmoothLoss import FlowSmoothLoss
from utils.config_utils import load_config_with_inheritance
from utils.dataloader_utils import create_dataloaders
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from losses.KNNDistanceLoss import KNNDistanceLoss
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
config_obj = load_config_with_inheritance("/workspace/gan_seg/src/config/altereular.yaml")
config_obj.dataloader.batchsize = 4
dataloader, infinite_loader, val_dataloader, batch_size, N = create_dataloaders(config_obj)

flow_smooth_loss = FlowSmoothLoss(device, config_obj.loss.scene_flow_smoothness)
knn_loss = KNNDistanceLoss()
from Predictor import get_scene_flow_predictor, get_mask_predictor
from model.eulerflow_raw_mlp import QueryDirection
flow_model = get_scene_flow_predictor(config_obj.model.flow, N)
flow_model.to(device)
mask_model = get_mask_predictor(config_obj.model.mask, N)
mask_model.to(device)
bucket_size = 10
max_sample = 10
# Initialize accumulator for average losses
all_losses = []
sample_count = 0

print("Processing validation samples...")

# Iterate through all validation samples
for val_sample in val_dataloader:
    sample_count += 1
    if sample_count > max_sample:
        break
    print(f"Processing sample {sample_count}")

    
    mask = val_sample["dynamic_instance_mask"][0].to(device)
    mask = remap_instance_labels(mask)
    one_hot_mask = F.one_hot(mask).permute(1, 0)
    one_hot_mask = one_hot_mask.float()
    mask_noise = mask_model(val_sample["point_cloud_first"][0].to(device), val_sample["idx"][0], val_sample["total_frames"][0])
    mask_noise = mask_noise.permute(1, 0)[:one_hot_mask.shape[0],:].to(device)

    N = val_sample["point_cloud_first"][0].shape[0]
    flow_noise = flow_model(val_sample["point_cloud_first"][0].to(device), val_sample["idx"][0], val_sample["total_frames"][0],QueryDirection.FORWARD)
    flow_noise = flow_noise.to(device)
    flow_noise = flow_noise.to(device)

    # Calculate losses for this sample
    sample_losses = []
    for i in range(bucket_size):
        loss_same_flow_noise = []
        for j in range(bucket_size):
            flow_Scaled = flow_noise * i/bucket_size *0.01
            mask_noise_scaled = mask_noise * (j/bucket_size+1e-4)
            mask_current = mask_noise_scaled + one_hot_mask*(1-j/bucket_size)
            # mask = mask/pow(mask.std(),0.5)
            flow = flow_Scaled + val_sample["flow"][0].to(device).squeeze(0) * (1-i/bucket_size)
            # flow = flow/pow(flow.std(),0.5)
            # flow = flow/pow(flow.std(),1.5)
            point_position = val_sample["point_cloud_first"][0].to(device)
            idx = val_sample["idx"][0]
            point_position_next = val_sample["self"][0].get_item(idx+1)["point_cloud_first"].to(device)
            loss = flow_smooth_loss(point_position,[mask_current], [flow])
            # loss += knn_loss(point_position+flow, point_position_next)*4
            loss_same_flow_noise.append(loss.item())
        sample_losses.append(loss_same_flow_noise)
    print("mean loss", np.mean(sample_losses))
    all_losses.append(sample_losses)

print(f"Processed {sample_count} validation samples")

# Calculate average losses across all samples
losses = np.mean(np.array(all_losses), axis=0)

# Reshape losses into a 2D array for plotting
losses = np.array(losses).reshape(bucket_size, bucket_size)

# losses = np.where(losses == -1, np.mean(losses), losses)

# Create a figure with multiple subplots
plt.figure(figsize=(18, 6))

# 1. 3D Surface Plot
ax1 = plt.subplot(1, 3, 1, projection='3d')
x = np.linspace(0, 1, bucket_size)
y = np.linspace(0, 1, bucket_size)
X, Y = np.meshgrid(x, y)
surf = ax1.plot_surface(X, Y, losses, cmap='viridis', edgecolor='none')
ax1.set_xlabel('Mask Noise Scale')
ax1.set_ylabel('Flow Noise Scale')
ax1.set_zlabel('Average Loss')
ax1.set_title(f'3D Surface Plot (Avg over {sample_count} samples)')

# 2. Contour Plot (2D with contour lines) with very fine intervals
ax2 = plt.subplot(1, 3, 2)
# Calculate levels with non-uniform intervals - denser near max values
min_loss = np.min(losses)
max_loss = np.max(losses)

# Create non-uniform levels using logarithmic or exponential spacing
# This will create more levels near the maximum loss
range_loss = max_loss - min_loss
# Using power function to create non-uniform spacing
# Higher exponent = more concentration near max value
power = 1  # Adjust this value to control density distribution
num_levels = 100  # Total number of contour levels

# Generate normalized values between 0 and 1, more concentrated toward 1
t = np.linspace(0, 1, num_levels)
normalized_levels = 1 - (1 - t)**power

# Map normalized values back to the loss range
levels = min_loss + normalized_levels * range_loss

# Draw fine contour lines
contour = ax2.contour(X, Y, losses, levels=levels, colors='black', linewidths=0.5)

# Only label some contours to avoid overcrowding
label_indices = np.linspace(0, len(levels)-1, 10).astype(int)
labeled_levels = levels[label_indices]
labeled_contour = ax2.contour(X, Y, losses, levels=labeled_levels, colors='black', linewidths=1.0)
ax2.clabel(labeled_contour, inline=True, fontsize=8, fmt='%.3f')

# Fill with colors as before
filled_c = ax2.contourf(X, Y, losses, 20, cmap='viridis')

# 3. Heatmap with Color Terrain
ax3 = plt.subplot(1, 3, 3)
heatmap = ax3.imshow(losses, cmap='terrain', origin='lower', 
                     extent=[0, 1, 0, 1], aspect='auto')
ax3.set_xlabel('Mask Noise Scale')
ax3.set_ylabel('Flow Noise Scale')
ax3.set_title(f'Color Terrain Heatmap (Avg over {sample_count} samples)')
plt.colorbar(heatmap, ax=ax3, label='Loss Value')

# Add markers for min and max values
min_idx = np.unravel_index(np.argmin(losses), losses.shape)
max_idx = np.unravel_index(np.argmax(losses), losses.shape)
min_x, min_y = x[min_idx[1]], y[min_idx[0]]
max_x, max_y = x[max_idx[1]], y[max_idx[0]]

# Add min/max markers to each plot
ax2.plot(min_x, min_y, 'ro', markersize=8, label=f'Min: {losses.min():.4f}')
ax2.plot(max_x, max_y, 'go', markersize=8, label=f'Max: {losses.max():.4f}')
ax3.plot(min_idx[1]/bucket_size, min_idx[0]/bucket_size, 'ro', markersize=8)
ax3.plot(max_idx[1]/bucket_size, max_idx[0]/bucket_size, 'go', markersize=8)
ax2.legend()

plt.tight_layout()
plt.savefig(f'loss_visualization_avg_{sample_count}_samples.png', dpi=300, bbox_inches='tight')
print(f"Visualization saved as 'loss_visualization_avg_{sample_count}_samples.png'")
