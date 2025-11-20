# filepath: /workspace/gan_seg/src/ablation/loss_eval.py
"""
Unified loss evaluation script for various loss functions.

Usage:
    python loss_eval.py --loss flow_smooth
    python loss_eval.py --loss knn
    python loss_eval.py --loss chamfer
    python loss_eval.py --loss point_smooth
    python loss_eval.py --loss dynamic
    python loss_eval.py --loss invariance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import functools
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.dataloader_utils import create_dataloaders_general

from losses.FlowSmoothLoss import FlowSmoothLoss
from losses.KNNDistanceLoss import KNNDistanceLoss
from losses.ChamferDistanceLoss import ChamferDistanceLoss
from losses.PointSmoothLoss import PointSmoothLoss
from losses.ReconstructionLoss import ReconstructionLoss
from losses.InvarianceLoss import InvarianceLoss
from utils.config_utils import load_config_with_inheritance
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils.visualization_utils import remap_instance_labels
from utils.model_utils import load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Evaluate different loss functions")
parser.add_argument("--loss", type=str, default="flow_smooth", 
                    choices=["flow_smooth", "knn", "chamfer", "point_smooth", "dynamic", "invariance"],
                    help="Loss function to evaluate")
parser.add_argument("--config", type=str, default="/workspace/gan_seg/src/config/general_tryfastincrease.yaml",
                    help="Path to config file")
parser.add_argument("--bucket_size", type=int, default=30,
                    help="Number of buckets for noise scale")
parser.add_argument("--max_sample", type=int, default=3,
                    help="Maximum number of samples to process")
args = parser.parse_args()

config_obj = load_config_with_inheritance(args.config)
config_obj.dataloader.batchsize = 4
config_obj.model.mask.slot_num = 10

# Get loss_name first to determine which dataset to use
loss_name = args.loss

(dataset, dataloader, val_flow_dataset, val_flow_dataloader, val_mask_dataset, val_mask_dataloader) = (
    create_dataloaders_general(config_obj)
)

# Select appropriate dataset based on loss type
# For flow_smooth loss, use dataset (training set with flow data) - matching flow_smooth_loss_eval.py
# For other losses that need mask, use val_mask_dataset (has boxes/mask)
if loss_name == "flow_smooth":
    # Use training dataset (matching flow_smooth_loss_eval.py which uses dataset)
    eval_dataset = dataset
    print(f"Using training dataset with {len(dataset)} samples (has flow data)")
else:
    # Use val_mask_dataset if available (has mask with boxes loaded), otherwise fallback to training dataset
    # val_mask_dataset loads boxes=True which provides mask data (instance_ids)
    if val_mask_dataset is not None and len(val_mask_dataset) > 0:
        eval_dataset = val_mask_dataset
        print(f"Using val_mask_dataset with {len(val_mask_dataset)} samples (has boxes/mask)")
    else:
        eval_dataset = dataset
        print(f"Using training dataset with {len(dataset)} samples (may not have mask)")

# Initialize loss function based on argument
loss_name = args.loss
use_robust_flow_component = False
if loss_name == "flow_smooth":
    loss_fn = FlowSmoothLoss(device, config_obj.loss.scene_flow_smoothness)
    if config_obj.loss.scene_flow_smoothness.each_mask_item.relative_gradient > 0:
        loss_display_name = "Rigid Loss"
        use_robust_flow_component = True
    else:
        loss_display_name = "Coverage Loss"
        use_robust_flow_component = False
elif loss_name == "knn":
    loss_fn = KNNDistanceLoss(
        k=config_obj.loss.knn.k,
        reduction=config_obj.loss.knn.reduction,
        distance_max_threshold=config_obj.loss.knn.distance_max_threshold,
        distance_min_threshold=config_obj.loss.knn.distance_min_threshold,
    )
    loss_display_name = "KNN Loss"
elif loss_name == "chamfer":
    loss_fn = ChamferDistanceLoss(reduction='mean')
    loss_display_name = "Chamfer Loss"
elif loss_name == "point_smooth":
    loss_fn = PointSmoothLoss(
        w_knn=3,
        w_ball_q=1,
        knn_loss_params=config_obj.loss.point_smooth_loss.knn_loss_params,
        ball_q_loss_params=config_obj.loss.point_smooth_loss.ball_q_loss_params,
    )
    loss_display_name = "Point Smooth Loss"
elif loss_name == "dynamic":
    loss_fn = ReconstructionLoss(config_obj, device)
    loss_display_name = "Dynamic Loss"
elif loss_name == "invariance":
    loss_fn = InvarianceLoss(cross_entropy=False, loss_norm=2)
    loss_display_name = "Invariance Loss"
else:
    raise ValueError(f"Unknown loss function: {loss_name}")

# Note: We don't need to load models since we're using random noise for flow and mask
# Point clouds remain unchanged, only flow and mask have noise added

bucket_size = args.bucket_size
max_sample = args.max_sample

# Initialize accumulator for average losses
all_losses = []
sample_count = 0

print(f"Evaluating {loss_display_name}...")
print(f"Processing validation samples...")

# Iterate through all validation samples
for val_sample in eval_dataset:
    sample_count += 1
    if sample_count > max_sample:
        break
    print(f"Processing sample {sample_count}")

    # Handle mask - may not be present in all datasets
    if "mask" in val_sample and val_sample["mask"] is not None:
        mask = val_sample["mask"].to(device)
        mask = remap_instance_labels(mask)
        one_hot_mask = F.one_hot(mask).permute(1, 0)
        one_hot_mask = one_hot_mask.float()
        # Filter mask less than 50 points
        one_hot_mask = one_hot_mask[one_hot_mask.sum(dim=1) > 50]
    else:
        # Create a dummy mask if not available (for losses that don't need mask, we'll skip mask-dependent ones)
        # For now, create a simple mask with one class
        N = val_sample["point_cloud_first"].shape[0]
        mask = torch.zeros(N, dtype=torch.long, device=device)
        one_hot_mask = F.one_hot(mask).permute(1, 0).float()
        # Add a small random component to make it more realistic
        one_hot_mask = one_hot_mask + torch.randn_like(one_hot_mask) * 0.01
        one_hot_mask = torch.softmax(one_hot_mask, dim=0)
    
    N = val_sample["point_cloud_first"].shape[0]
    point_cloud_first = val_sample["point_cloud_first"].to(device).float()
    point_cloud_next = val_sample["point_cloud_next"].to(device).float()
    center = point_cloud_first.mean(dim=0)
    point_cloud_first = point_cloud_first - center
    point_cloud_next = point_cloud_next - center
    flow_gt = val_sample["flow"].to(device).float().squeeze(0)
    
    # Prepare noise for flow and mask (point clouds remain unchanged)
    # Generate mask noise with same shape as one_hot_mask
    mask_noise = torch.randn(one_hot_mask.shape[0], one_hot_mask.shape[1], device=device, dtype=one_hot_mask.dtype)
    
    # Generate flow noise with same shape as flow
    # Use a fixed seed per sample to ensure reproducibility, but different noise per sample
    torch.manual_seed(sample_count)
    flow_noise = torch.randn_like(flow_gt)
    
    # Calculate losses for this sample
    # All losses now use flow noise (i) and mask noise (j), point clouds remain unchanged
    sample_losses = []
    for i in range(bucket_size):
        loss_row = []
        for j in range(bucket_size):
            # Scale flow noise: i/bucket_size controls flow noise level
            # Follow the same approach as flow_smooth_loss_eval.py
            flow_noise_scale = i / bucket_size
            if loss_name in ["dynamic"]:
                flow_noise_scale = flow_noise_scale*0.01
            flow_std = flow_gt.std()
            flow_noise_std = flow_noise.std()
            
            # Match original flow_smooth_loss_eval.py logic exactly:
            # flow_Scaled = flow_noise * i / bucket_size * flow_gt.std() / flow_noise.std()
            # flow = flow_Scaled + flow_gt * (1 - i / bucket_size)
            # If flow_gt.std() is 0, flow_Scaled will be 0 (matching original behavior)
            if flow_noise_std > 1e-8:
                # Exact match to original: flow_Scaled = flow_noise * i/bucket_size * flow_gt.std() / flow_noise.std()
                flow_scaled = flow_noise * flow_noise_scale * flow_std / flow_noise_std
                # if loss_name == "dynamic":
                #     flow_scaled = flow_scaled*0.01
            else:
                # If flow_noise_std is too small, flow_scaled = 0 (matching original when flow_gt.std() = 0)
                flow_scaled = torch.zeros_like(flow_noise)
            
            # Mix: flow = flow_scaled + flow_gt * (1 - flow_noise_scale)
            # Exact match to original: flow = flow_Scaled + flow_gt * (1 - i / bucket_size)
            flow_current = flow_scaled + flow_gt * (1 - flow_noise_scale)
            flow_current = flow_current.float()  # Match original
            
            # Scale mask noise: j/bucket_size controls mask noise level
            mask_noise_scale = j / bucket_size + 1e-1
            mask_current = mask_noise * mask_noise_scale + one_hot_mask * (1 - mask_noise_scale)
            mask_current = torch.softmax(mask_current, dim=0).float()
            if loss_name == "flow_smooth":
                # Flow smooth loss: uses point cloud (unchanged), mask (with noise), flow (with noise)
                # Use flow directly without extra normalization (matching original script)
                print(f"mask_current: {mask_current.shape}")
                print(f"flow_current: {flow_current.shape}")
                print(f"point_cloud_first: {point_cloud_first.shape}")
                loss = loss_fn(point_cloud_first, [mask_current], [flow_current])
                
            elif loss_name == "knn":
                # KNN loss: uses point clouds generated from flow (point cloud unchanged, flow with noise)
                # Generate second point cloud using noisy flow
                # flow_current changes with i (flow noise scale), so pc_next_from_flow should change
                pc_next_from_flow = point_cloud_first + flow_current
                pc_next_batch = pc_next_from_flow.unsqueeze(0)
                
                # Compare predicted next point cloud (from noisy flow) with ground truth next point cloud
                # Note: KNN loss measures distance from points in pc_next_from_flow to nearest points in point_cloud_next
                # As flow noise increases, pc_next_from_flow deviates more from point_cloud_next, so loss should increase
                loss = loss_fn(pc_next_batch, point_cloud_next.unsqueeze(0), bidirectional=False)
                
            elif loss_name == "chamfer":
                # Chamfer loss: uses point clouds generated from flow (point cloud unchanged, flow with noise)
                # Generate second point cloud using noisy flow
                pc_next_from_flow = point_cloud_first + flow_current
                pc_first_batch = point_cloud_first.unsqueeze(0)
                pc_next_batch = pc_next_from_flow.unsqueeze(0)
                loss = loss_fn(pc_first_batch, pc_next_batch)
                
            elif loss_name == "point_smooth":
                # Point smooth loss: uses point cloud (unchanged) and mask (with noise)
                loss = loss_fn([point_cloud_first], [mask_current])
                
            elif loss_name == "dynamic":
                # Dynamic loss: uses point clouds (unchanged), mask (with noise), flow (with noise)
                # Generate second point cloud using noisy flow
                pc_next_from_flow = point_cloud_first + flow_current
                # flow_current = flow_current/flow_current.std()
                loss, _ = loss_fn([point_cloud_first], [point_cloud_next], [mask_current], [flow_current])
                
            elif loss_name == "invariance":
                # Invariance loss: uses two masks with different noise levels
                # Mask 1: noise level i
                mask_noise_scale_1 = i / bucket_size + 1e-1
                mask_current_1 = mask_noise * mask_noise_scale_1 + one_hot_mask * (1 - mask_noise_scale_1)
                mask_current_1 = torch.softmax(mask_current_1, dim=0).float()
                
                # Mask 2: noise level j
                mask_noise_scale_2 = j / bucket_size + 1e-1
                mask_noise_2 = torch.randn(one_hot_mask.shape[0], one_hot_mask.shape[1], device=device, dtype=one_hot_mask.dtype)
                mask_current_2 = mask_noise_2 * mask_noise_scale_2 + one_hot_mask * (1 - mask_noise_scale_2)
                mask_current_2 = torch.softmax(mask_current_2, dim=0).float()
                
                mask_1_batch = mask_current_1.permute(1, 0).unsqueeze(0)
                mask_2_batch = mask_current_2.permute(1, 0).unsqueeze(0)
                loss = loss_fn(mask_1_batch, mask_2_batch)
            
            # Handle tuple outputs (some losses may return auxiliary values)
            if isinstance(loss, tuple):
                if loss_name == "flow_smooth" and use_robust_flow_component:
                    loss = loss[1]
                else:
                    loss = loss[0]
            if torch.is_tensor(loss):
                loss_value = loss.item()
            else:
                loss_value = float(loss)
            loss_row.append(loss_value)
            
            if j == 3:
                display_loss = loss_value
                print(f"Noise step {i}, {j}: loss {display_loss:.4f}")
        sample_losses.append(loss_row)
    print("mean loss", np.mean(sample_losses))
    all_losses.append(sample_losses)

print(f"Processed {sample_count} validation samples")

# Calculate average losses across all samples
losses = np.mean(np.array(all_losses), axis=0)

# Reshape losses into a 2D array for plotting
losses = np.array(losses).reshape(bucket_size, bucket_size)

# All losses now use flow noise (x-axis) and mask noise (y-axis)
ylabel, xlabel = "Flow Noise Scale", "Mask Noise Scale"
if loss_name == "invariance":
    ylabel, xlabel = "Mask Noise Scale 1", "Mask Noise Scale 2"
# 准备坐标（x -> flow noise, y -> mask noise）
x = np.linspace(0, 1, bucket_size)  # flow noise axis (columns)
y = np.linspace(0, 1, bucket_size)  # mask noise axis (rows)
X, Y = np.meshgrid(x, y)

# 1. 3D Surface Plot for Loss
plt.figure(figsize=(10, 8))
ax1 = plt.subplot(1, 1, 1, projection="3d")
surf = ax1.plot_surface(X, Y, losses, cmap="viridis", edgecolor="none")
ax1.set_xlabel(xlabel)
ax1.set_ylabel(ylabel)
ax1.set_zlabel("Average Loss")
ax1.set_title(f"{loss_display_name} 3D Surface Plot (Avg over {max_sample} samples)")
plt.colorbar(surf, ax=ax1, label="Loss Value", shrink=0.8)
plt.tight_layout()
filename_3d = f"{loss_name}_loss_3d_surface_avg_{max_sample}_samples.png"
plt.savefig(filename_3d, dpi=300, bbox_inches="tight")
print(f"3D Surface Plot saved as '{filename_3d}'")

# 2. Contour Plot (2D with contour lines) with very fine intervals
plt.figure(figsize=(10, 8))
ax2 = plt.subplot(1, 1, 1)
# Calculate levels with non-uniform intervals - denser near max values
min_loss = np.min(losses)
max_loss = np.max(losses)

# Handle case where all losses are the same (e.g., all zeros)
if max_loss - min_loss < 1e-10:
    # If all losses are the same, create a simple contour plot
    filled_c = ax2.contourf(X, Y, losses, 1, cmap="viridis")
    ax2.text(0.5, 0.5, f"Constant Loss: {min_loss:.6f}", 
             transform=ax2.transAxes, ha="center", va="center", fontsize=14)
else:
    # Create non-uniform levels using logarithmic or exponential spacing
    range_loss = max_loss - min_loss
    power = 1
    num_levels = 100

    # Generate normalized values between 0 and 1, more concentrated toward 1
    t = np.linspace(0, 1, num_levels)
    normalized_levels = 1 - (1 - t) ** power

    # Map normalized values back to the loss range
    levels = min_loss + normalized_levels * range_loss

    # Draw fine contour lines
    contour = ax2.contour(X, Y, losses, levels=levels, colors="black", linewidths=0.5)

    # Only label some contours to avoid overcrowding
    label_indices = np.linspace(0, len(levels) - 1, 10).astype(int)
    labeled_levels = levels[label_indices]
    labeled_contour = ax2.contour(X, Y, losses, levels=labeled_levels, colors="black", linewidths=1.0)
    ax2.clabel(labeled_contour, inline=True, fontsize=8, fmt="%.3f")

# Fill with colors as before (only if not already done)
if max_loss - min_loss >= 1e-10:
    filled_c = ax2.contourf(X, Y, losses, 20, cmap="viridis")
else:
    filled_c = ax2.contourf(X, Y, losses, 1, cmap="viridis")

# Add markers for min and max values
min_idx = np.unravel_index(np.argmin(losses), losses.shape)
max_idx = np.unravel_index(np.argmax(losses), losses.shape)
min_x, min_y = x[min_idx[1]], y[min_idx[0]]
max_x, max_y = x[max_idx[1]], y[max_idx[0]]

# Add min/max markers
ax2.plot(min_x, min_y, "ro", markersize=8, label=f"Min: {losses.min():.4f}")
ax2.plot(max_x, max_y, "go", markersize=8, label=f"Max: {losses.max():.4f}")
ax2.set_xlabel(xlabel)
ax2.set_ylabel(ylabel)
ax2.set_title(f"{loss_display_name} Contour Plot (Avg over {max_sample} samples)")
ax2.legend()
plt.colorbar(filled_c, ax=ax2, label="Loss Value")
plt.tight_layout()
filename_contour = f"{loss_name}_loss_contour_avg_{max_sample}_samples.png"
plt.savefig(filename_contour, dpi=300, bbox_inches="tight")
print(f"Contour Plot saved as '{filename_contour}'")

# 3. Heatmap with Color Terrain
plt.figure(figsize=(10, 8))
ax3 = plt.subplot(1, 1, 1)
heatmap = ax3.imshow(losses, cmap="terrain", origin="lower", extent=[0, 1, 0, 1], aspect="auto")
ax3.set_xlabel(xlabel)
ax3.set_ylabel(ylabel)
ax3.set_title(f"{loss_display_name} Color Terrain Heatmap (Avg over {max_sample} samples)")
plt.colorbar(heatmap, ax=ax3, label="Loss Value")

# Add markers for min and max values
ax3.plot(min_x, min_y, "ro", markersize=8, label=f"Min: {losses.min():.4f}")
ax3.plot(max_x, max_y, "go", markersize=8, label=f"Max: {losses.max():.4f}")
ax3.legend()
plt.tight_layout()
filename_heatmap = f"{loss_name}_loss_heatmap_avg_{max_sample}_samples.png"
plt.savefig(filename_heatmap, dpi=100, bbox_inches="tight")
print(f"Heatmap saved as '{filename_heatmap}'")

print(f"\nAll visualizations for {loss_display_name} completed!")


