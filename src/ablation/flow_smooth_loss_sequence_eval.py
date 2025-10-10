# filepath: /workspace/gan_seg/src/ablation/flow_smooth_loss_sequence_eval.py
"""
序列帧中flow_smooth_loss变化分析
分析当输入为0.9倍真值和0.1倍噪声时，flow_smooth_loss在整个序列中的变化
"""

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

def analyze_sequence_flow_smooth_loss(sequence_data, flow_model, mask_model, flow_smooth_loss, device):
    """
    分析序列中flow_smooth_loss的变化，使用模型推理结果
    
    Args:
        sequence_data: 序列数据
        flow_model: 流预测模型
        mask_model: 掩码预测模型
        flow_smooth_loss: 流平滑损失函数
        device: 设备
    
    Returns:
        sequence_losses: 每个时间步的loss值列表
        frame_info: 每帧的详细信息
    """
    sequence_losses = []
    frame_info = []
    
    total_frames = sequence_data["total_frames"][0]
    print(f"分析序列，总帧数: {total_frames}")
    
    for frame_idx in range(total_frames):
        print(f"处理第 {frame_idx} 帧...")
        
        try:
            # 获取当前帧数据
            if frame_idx == 0:
                current_frame = sequence_data
            else:
                current_frame = sequence_data["self"][0].get_item(frame_idx)
            
            # 提取当前帧信息
            point_cloud_first = current_frame["point_cloud_first"].to(device)
            flow_gt = current_frame["flow"].to(device)
            mask_gt = current_frame["dynamic_instance_mask"].to(device)
            
            # 重映射标签
            mask_gt = remap_instance_labels(mask_gt)
            one_hot_mask = F.one_hot(mask_gt).permute(1, 0).float()
            
            # 计算GT mask size - 不同实例的数量
            unique_instances = torch.unique(mask_gt)
            gt_mask_size = len(unique_instances)
            
            # 计算GT flow的方差和标准差
            flow_variance = torch.var(flow_gt, dim=0).sum().item()  # 计算所有维度的方差之和
            flow_std = torch.std(flow_gt, dim=0).sum().item()  # 计算所有维度的标准差之和
            
            # 获取模型预测结果
            pred_mask = mask_model(point_cloud_first, frame_idx, total_frames)
            pred_mask = pred_mask.permute(1, 0)[:one_hot_mask.shape[0], :].to(device)
            
            pred_flow = flow_model(point_cloud_first, frame_idx, total_frames, 
                                  QueryDirection.FORWARD).to(device)
            
            # 使用模型预测结果，不进行混合
            final_mask = pred_mask
            final_flow = pred_flow
            
            # 计算flow_smooth_loss
            loss = flow_smooth_loss(point_cloud_first, [final_mask], [final_flow])
            
            sequence_losses.append(loss.item())
            frame_info.append({
                'frame_idx': frame_idx,
                'loss': loss.item(),
                'point_cloud_shape': point_cloud_first.shape,
                'mask_shape': final_mask.shape,
                'flow_shape': final_flow.shape,
                'gt_mask_size': gt_mask_size,  # 不同实例的数量
                'gt_flow_variance': flow_variance,  # GT flow的方差
                'gt_flow_std': flow_std  # GT flow的标准差
            })
            
        except Exception as e:
            print(f"处理第 {frame_idx} 帧时出错: {e}")
            sequence_losses.append(np.nan)
            frame_info.append({
                'frame_idx': frame_idx,
                'loss': np.nan,
                'error': str(e)
            })
    
    return sequence_losses, frame_info

def create_sequence_visualization(sequence_losses, frame_info, save_path="flow_smooth_loss_sequence.png"):
    """
    Create visualization for sequence loss changes
    
    Args:
        sequence_losses: List of sequence loss values
        frame_info: Frame information list
        save_path: Save path
    """
    # Filter out NaN values
    valid_losses = [loss for loss in sequence_losses if not np.isnan(loss)]
    valid_frames = [i for i, loss in enumerate(sequence_losses) if not np.isnan(loss)]
    
    if len(valid_losses) == 0:
        print("No valid loss data for visualization")
        return
    
    # Extract GT mask sizes, flow variances, and flow stds for correlation analysis
    mask_sizes = []
    flow_variances = []
    flow_stds = []
    for info in frame_info:
        if 'gt_mask_size' in info and not np.isnan(info.get('loss', np.nan)):
            # GT mask size is the number of unique instances
            mask_size = info['gt_mask_size']
            mask_sizes.append(mask_size)
        else:
            mask_sizes.append(np.nan)
            
        if 'gt_flow_variance' in info and not np.isnan(info.get('loss', np.nan)):
            # GT flow variance
            flow_var = info['gt_flow_variance']
            flow_variances.append(flow_var)
        else:
            flow_variances.append(np.nan)
            
        if 'gt_flow_std' in info and not np.isnan(info.get('loss', np.nan)):
            # GT flow std
            flow_std = info['gt_flow_std']
            flow_stds.append(flow_std)
        else:
            flow_stds.append(np.nan)
    
    # Filter valid data
    valid_mask_sizes = [size for size in mask_sizes if not np.isnan(size)]
    valid_flow_variances = [var for var in flow_variances if not np.isnan(var)]
    valid_flow_stds = [std for std in flow_stds if not np.isnan(std)]
    
    # Create figure with 5x2 subplots to include flow std analysis
    fig, axes = plt.subplots(5, 2, figsize=(15, 30))
    fig.suptitle('Flow Smooth Loss Sequence Analysis (Model Inference)', fontsize=16)
    
    # 1. Time series plot
    ax1 = axes[0, 0]
    ax1.plot(valid_frames, valid_losses, 'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Flow Smooth Loss')
    ax1.set_title('Loss Over Time')
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss distribution histogram
    ax2 = axes[0, 1]
    ax2.hist(valid_losses, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('Loss Value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Loss Value Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative statistics
    ax3 = axes[1, 0]
    cumulative_loss = np.cumsum(valid_losses)
    ax3.plot(valid_frames, cumulative_loss, 'g-s', linewidth=2, markersize=3)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Cumulative Loss')
    ax3.set_title('Cumulative Loss Change')
    ax3.grid(True, alpha=0.3)
    
    # 4. Loss vs Mask Size correlation
    ax4 = axes[1, 1]
    if len(valid_mask_sizes) > 1 and len(valid_losses) > 1:
        # Ensure same length
        min_len = min(len(valid_losses), len(valid_mask_sizes))
        corr_losses = valid_losses[:min_len]
        corr_sizes = valid_mask_sizes[:min_len]
        
        ax4.scatter(corr_sizes, corr_losses, alpha=0.6, color='red', s=50)
        ax4.set_xlabel('GT Mask Size (Number of Unique Instances)')
        ax4.set_ylabel('Flow Smooth Loss')
        ax4.set_title('Loss vs GT Mask Size Correlation')
        ax4.grid(True, alpha=0.3)
        
        # Calculate and display correlation
        if len(corr_losses) > 1:
            correlation = np.corrcoef(corr_sizes, corr_losses)[0, 1]
            ax4.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                    transform=ax4.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'Insufficient data for correlation', 
                transform=ax4.transAxes, ha='center', va='center')
        ax4.set_title('Loss vs GT Mask Size Correlation')
    
    # 5. Loss vs GT Flow Variance correlation
    ax5 = axes[2, 0]
    if len(valid_flow_variances) > 1 and len(valid_losses) > 1:
        # Ensure same length
        min_len = min(len(valid_losses), len(valid_flow_variances))
        corr_losses = valid_losses[:min_len]
        corr_variances = valid_flow_variances[:min_len]
        
        ax5.scatter(corr_variances, corr_losses, alpha=0.6, color='purple', s=50)
        ax5.set_xlabel('GT Flow Variance')
        ax5.set_ylabel('Flow Smooth Loss')
        ax5.set_title('Loss vs GT Flow Variance Correlation')
        ax5.grid(True, alpha=0.3)
        
        # Calculate and display correlation
        if len(corr_losses) > 1:
            correlation = np.corrcoef(corr_variances, corr_losses)[0, 1]
            ax5.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                    transform=ax5.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for correlation', 
                transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('Loss vs GT Flow Variance Correlation')
    
    # 6. GT Flow Variance over time
    ax6 = axes[2, 1]
    if len(valid_flow_variances) > 0:
        ax6.plot(valid_frames[:len(valid_flow_variances)], valid_flow_variances, 'purple', linewidth=2, markersize=3)
        ax6.set_xlabel('Frame Index')
        ax6.set_ylabel('GT Flow Variance')
        ax6.set_title('GT Flow Variance Over Time')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No flow variance data', 
                transform=ax6.transAxes, ha='center', va='center')
        ax6.set_title('GT Flow Variance Over Time')
    
    # 7. Loss vs GT Flow Std correlation
    ax7 = axes[3, 0]
    if len(valid_flow_stds) > 1 and len(valid_losses) > 1:
        # Ensure same length
        min_len = min(len(valid_losses), len(valid_flow_stds))
        corr_losses = valid_losses[:min_len]
        corr_stds = valid_flow_stds[:min_len]
        
        ax7.scatter(corr_stds, corr_losses, alpha=0.6, color='green', s=50)
        ax7.set_xlabel('GT Flow Std')
        ax7.set_ylabel('Flow Smooth Loss')
        ax7.set_title('Loss vs GT Flow Std Correlation')
        ax7.grid(True, alpha=0.3)
        
        # Calculate and display correlation
        if len(corr_losses) > 1:
            correlation = np.corrcoef(corr_stds, corr_losses)[0, 1]
            ax7.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                    transform=ax7.transAxes, fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    else:
        ax7.text(0.5, 0.5, 'Insufficient data for correlation', 
                transform=ax7.transAxes, ha='center', va='center')
        ax7.set_title('Loss vs GT Flow Std Correlation')
    
    # 8. GT Flow Std over time
    ax8 = axes[3, 1]
    if len(valid_flow_stds) > 0:
        ax8.plot(valid_frames[:len(valid_flow_stds)], valid_flow_stds, 'green', linewidth=2, markersize=3)
        ax8.set_xlabel('Frame Index')
        ax8.set_ylabel('GT Flow Std')
        ax8.set_title('GT Flow Std Over Time')
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, 'No flow std data', 
                transform=ax8.transAxes, ha='center', va='center')
        ax8.set_title('GT Flow Std Over Time')
    
    # 9. GT Mask Size over time
    ax9 = axes[4, 0]
    if len(valid_mask_sizes) > 0:
        ax9.plot(valid_frames[:len(valid_mask_sizes)], valid_mask_sizes, 'r-s', linewidth=2, markersize=3)
        ax9.set_xlabel('Frame Index')
        ax9.set_ylabel('GT Mask Size (Unique Instances)')
        ax9.set_title('GT Mask Size Over Time')
        ax9.grid(True, alpha=0.3)
    else:
        ax9.text(0.5, 0.5, 'No mask size data', 
                transform=ax9.transAxes, ha='center', va='center')
        ax9.set_title('GT Mask Size Over Time')
    
    # 10. Statistical information
    ax10 = axes[4, 1]
    ax10.axis('off')
    
    # Calculate statistics
    mean_loss = np.mean(valid_losses)
    std_loss = np.std(valid_losses)
    min_loss = np.min(valid_losses)
    max_loss = np.max(valid_losses)
    min_frame = valid_frames[np.argmin(valid_losses)]
    max_frame = valid_frames[np.argmax(valid_losses)]
    
    # Calculate correlations if possible
    correlation_text = ""
    
    # Loss-Mask correlation
    if len(valid_mask_sizes) > 1 and len(valid_losses) > 1:
        min_len = min(len(valid_losses), len(valid_mask_sizes))
        corr_losses = valid_losses[:min_len]
        corr_sizes = valid_mask_sizes[:min_len]
        if len(corr_losses) > 1:
            correlation = np.corrcoef(corr_sizes, corr_losses)[0, 1]
            correlation_text += f"Loss-Mask Correlation: {correlation:.4f}\n"
    
    # Loss-Flow Variance correlation
    if len(valid_flow_variances) > 1 and len(valid_losses) > 1:
        min_len = min(len(valid_losses), len(valid_flow_variances))
        corr_losses = valid_losses[:min_len]
        corr_variances = valid_flow_variances[:min_len]
        if len(corr_losses) > 1:
            correlation = np.corrcoef(corr_variances, corr_losses)[0, 1]
            correlation_text += f"Loss-Flow Variance Correlation: {correlation:.4f}\n"
    
    # Loss-Flow Std correlation
    if len(valid_flow_stds) > 1 and len(valid_losses) > 1:
        min_len = min(len(valid_losses), len(valid_flow_stds))
        corr_losses = valid_losses[:min_len]
        corr_stds = valid_flow_stds[:min_len]
        if len(corr_losses) > 1:
            correlation = np.corrcoef(corr_stds, corr_losses)[0, 1]
            correlation_text += f"Loss-Flow Std Correlation: {correlation:.4f}\n"
    
    # Flow variance and std statistics
    flow_var_stats = ""
    if len(valid_flow_variances) > 0:
        mean_flow_var = np.mean(valid_flow_variances)
        std_flow_var = np.std(valid_flow_variances)
        flow_var_stats += f"""
    GT Flow Variance:
    Mean: {mean_flow_var:.6f}
    Std: {std_flow_var:.6f}
    """
    
    if len(valid_flow_stds) > 0:
        mean_flow_std = np.mean(valid_flow_stds)
        std_flow_std = np.std(valid_flow_stds)
        flow_var_stats += f"""
    GT Flow Std:
    Mean: {mean_flow_std:.6f}
    Std: {std_flow_std:.6f}
    """
    
    stats_text = f"""
    Statistical Information:
    
    Total Frames: {len(sequence_losses)}
    Valid Frames: {len(valid_losses)}
    
    Mean Loss: {mean_loss:.6f}
    Std Loss: {std_loss:.6f}
    Min Loss: {min_loss:.6f} (Frame {min_frame})
    Max Loss: {max_loss:.6f} (Frame {max_frame})
    
    Loss Range: {max_loss - min_loss:.6f}
    Coefficient of Variation: {std_loss/mean_loss:.4f}
    
    {correlation_text}
    {flow_var_stats}
    Model: Trained Model Inference
    """
    
    ax10.text(0.1, 0.9, stats_text, transform=ax10.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {save_path}")
    
    return fig

def main():
    """Main function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config_obj = load_config_with_inheritance("/workspace/gan_seg/src/config/baseeular.yaml")
    config_obj.dataloader.batchsize = 1  # Use batch_size=1 for sequence analysis
    
    # Create data loaders
    dataloader, infinite_loader, val_dataloader, batch_size, N = create_dataloaders(config_obj)
    
    # Initialize models and loss function
    flow_smooth_loss = FlowSmoothLoss(device, config_obj.loss.scene_flow_smoothness)
    
    from Predictor import get_scene_flow_predictor, get_mask_predictor
    
    flow_model = get_scene_flow_predictor(config_obj.model.flow, N)
    flow_model.to(device)
    mask_model = get_mask_predictor(config_obj.model.mask, N)
    mask_model.to(device)
    
    # Load model checkpoints
    checkpoint_path = "/workspace/gan_seg/outputs/exp/20251009_174907/checkpoints/step_10000.pt"
    print(f"Loading model checkpoint from: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load flow model state
        if 'flow_model_state_dict' in checkpoint:
            flow_model.load_state_dict(checkpoint['flow_model_state_dict'])
            print("Flow model loaded successfully")
        else:
            print("Warning: flow_model_state_dict not found in checkpoint")
        
        # Load mask model state
        if 'mask_model_state_dict' in checkpoint:
            mask_model.load_state_dict(checkpoint['mask_model_state_dict'])
            print("Mask model loaded successfully")
        else:
            print("Warning: mask_model_state_dict not found in checkpoint")
            
        # Set models to evaluation mode
        flow_model.eval()
        mask_model.eval()
        print("Models set to evaluation mode")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using randomly initialized models")
    
    # Analysis parameters
    
    print(f"Starting sequence analysis using trained model inference")
    
    all_sequence_results = []
    
    # Process multiple sequences
    sequence_count = 0
    for val_sample in val_dataloader:
        try:
            # Analyze current sequence
            sequence_losses, frame_info = analyze_sequence_flow_smooth_loss(
                val_sample, flow_model, mask_model, flow_smooth_loss, device
            )
            
            # Save results
            all_sequence_results.append({
                'sequence_idx': sequence_count,
                'losses': sequence_losses,
                'frame_info': frame_info
            })
            
            # Create visualization for single sequence
            save_path = f"flow_smooth_loss_sequence_{sequence_count + 1}.png"
            create_sequence_visualization(sequence_losses, frame_info, save_path)
            
            sequence_count += 1
            
        except Exception as e:
            print(f"Error analyzing sequence {sequence_count + 1}: {e}")
            continue
        break
    
    # Create summary analysis for all sequences
    if all_sequence_results:
        print("\n=== Summary Analysis ===")
        
        # Collect all valid losses, mask sizes, flow variances, and flow stds
        all_losses = []
        all_mask_sizes = []
        all_flow_variances = []
        all_flow_stds = []
        for result in all_sequence_results:
            valid_losses = [loss for loss in result['losses'] if not np.isnan(loss)]
            all_losses.extend(valid_losses)
            
            # Extract GT mask sizes, flow variances, and flow stds from frame info
            for frame_info in result['frame_info']:
                if 'gt_mask_size' in frame_info and not np.isnan(frame_info.get('loss', np.nan)):
                    mask_size = frame_info['gt_mask_size']
                    all_mask_sizes.append(mask_size)
                    
                if 'gt_flow_variance' in frame_info and not np.isnan(frame_info.get('loss', np.nan)):
                    flow_var = frame_info['gt_flow_variance']
                    all_flow_variances.append(flow_var)
                    
                if 'gt_flow_std' in frame_info and not np.isnan(frame_info.get('loss', np.nan)):
                    flow_std = frame_info['gt_flow_std']
                    all_flow_stds.append(flow_std)
        
        if all_losses:
            # Create summary visualization with more subplots for flow std analysis
            fig, axes = plt.subplots(4, 3, figsize=(24, 24))
            fig.suptitle(f'All Sequences Flow Smooth Loss Summary Analysis (Model Inference)', fontsize=16)
            
            # 1. All sequences loss distribution
            ax1 = axes[0, 0]
            ax1.hist(all_losses, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
            ax1.set_xlabel('Loss Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('All Sequences Loss Distribution')
            ax1.grid(True, alpha=0.3)
            
            # 2. Average loss per sequence
            ax2 = axes[0, 1]
            sequence_means = []
            for result in all_sequence_results:
                valid_losses = [loss for loss in result['losses'] if not np.isnan(loss)]
                if valid_losses:
                    sequence_means.append(np.mean(valid_losses))
                else:
                    sequence_means.append(np.nan)
            
            ax2.bar(range(len(sequence_means)), sequence_means, alpha=0.7, color='lightgreen')
            ax2.set_xlabel('Sequence Index')
            ax2.set_ylabel('Average Loss')
            ax2.set_title('Average Loss per Sequence')
            ax2.grid(True, alpha=0.3)
            
            # 3. Loss vs Mask Size correlation across all sequences
            ax3 = axes[0, 2]
            if len(all_mask_sizes) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_mask_sizes))
                corr_losses = all_losses[:min_len]
                corr_sizes = all_mask_sizes[:min_len]
                
                ax3.scatter(corr_sizes, corr_losses, alpha=0.6, color='red', s=30)
                ax3.set_xlabel('GT Mask Size (Number of Unique Instances)')
                ax3.set_ylabel('Flow Smooth Loss')
                ax3.set_title('Overall Loss vs GT Mask Size Correlation')
                ax3.grid(True, alpha=0.3)
                
                # Calculate and display correlation
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_sizes, corr_losses)[0, 1]
                    ax3.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                            transform=ax3.transAxes, fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            else:
                ax3.text(0.5, 0.5, 'Insufficient data for correlation', 
                        transform=ax3.transAxes, ha='center', va='center')
                ax3.set_title('Overall Loss vs GT Mask Size Correlation')
            
            # 4. Loss trend over time
            ax4 = axes[1, 0]
            for i, result in enumerate(all_sequence_results):
                valid_losses = [loss for loss in result['losses'] if not np.isnan(loss)]
                if valid_losses:
                    ax4.plot(valid_losses, alpha=0.6, label=f'Sequence {i+1}')
            ax4.set_xlabel('Frame Index')
            ax4.set_ylabel('Loss Value')
            ax4.set_title('Loss Trend Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. Loss vs GT Flow Variance correlation
            ax5 = axes[1, 1]
            if len(all_flow_variances) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_flow_variances))
                corr_losses = all_losses[:min_len]
                corr_variances = all_flow_variances[:min_len]
                
                ax5.scatter(corr_variances, corr_losses, alpha=0.6, color='purple', s=30)
                ax5.set_xlabel('GT Flow Variance')
                ax5.set_ylabel('Flow Smooth Loss')
                ax5.set_title('Overall Loss vs GT Flow Variance Correlation')
                ax5.grid(True, alpha=0.3)
                
                # Calculate and display correlation
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_variances, corr_losses)[0, 1]
                    ax5.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                            transform=ax5.transAxes, fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            else:
                ax5.text(0.5, 0.5, 'Insufficient data for correlation', 
                        transform=ax5.transAxes, ha='center', va='center')
                ax5.set_title('Overall Loss vs GT Flow Variance Correlation')
            
            # 6. GT Flow Variance distribution
            ax6 = axes[1, 2]
            if len(all_flow_variances) > 0:
                ax6.hist(all_flow_variances, bins=20, alpha=0.7, color='purple', edgecolor='black')
                ax6.set_xlabel('GT Flow Variance')
                ax6.set_ylabel('Frequency')
                ax6.set_title('GT Flow Variance Distribution')
                ax6.grid(True, alpha=0.3)
            else:
                ax6.text(0.5, 0.5, 'No flow variance data', 
                        transform=ax6.transAxes, ha='center', va='center')
                ax6.set_title('GT Flow Variance Distribution')
            
            # 7. Loss vs GT Flow Std correlation
            ax7 = axes[2, 0]
            if len(all_flow_stds) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_flow_stds))
                corr_losses = all_losses[:min_len]
                corr_stds = all_flow_stds[:min_len]
                
                ax7.scatter(corr_stds, corr_losses, alpha=0.6, color='green', s=30)
                ax7.set_xlabel('GT Flow Std')
                ax7.set_ylabel('Flow Smooth Loss')
                ax7.set_title('Overall Loss vs GT Flow Std Correlation')
                ax7.grid(True, alpha=0.3)
                
                # Calculate and display correlation
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_stds, corr_losses)[0, 1]
                    ax7.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
                            transform=ax7.transAxes, fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
            else:
                ax7.text(0.5, 0.5, 'Insufficient data for correlation', 
                        transform=ax7.transAxes, ha='center', va='center')
                ax7.set_title('Overall Loss vs GT Flow Std Correlation')
            
            # 8. GT Flow Std distribution
            ax8 = axes[2, 1]
            if len(all_flow_stds) > 0:
                ax8.hist(all_flow_stds, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax8.set_xlabel('GT Flow Std')
                ax8.set_ylabel('Frequency')
                ax8.set_title('GT Flow Std Distribution')
                ax8.grid(True, alpha=0.3)
            else:
                ax8.text(0.5, 0.5, 'No flow std data', 
                        transform=ax8.transAxes, ha='center', va='center')
                ax8.set_title('GT Flow Std Distribution')
            
            # 9. GT Mask Size distribution
            ax9 = axes[2, 2]
            if len(all_mask_sizes) > 0:
                ax9.hist(all_mask_sizes, bins=20, alpha=0.7, color='orange', edgecolor='black')
                ax9.set_xlabel('GT Mask Size (Unique Instances)')
                ax9.set_ylabel('Frequency')
                ax9.set_title('GT Mask Size Distribution')
                ax9.grid(True, alpha=0.3)
            else:
                ax9.text(0.5, 0.5, 'No mask size data', 
                        transform=ax9.transAxes, ha='center', va='center')
                ax9.set_title('GT Mask Size Distribution')
            
            # 10. Statistical information
            ax10 = axes[3, 0]
            ax10.axis('off')
            
            overall_mean = np.mean(all_losses)
            overall_std = np.std(all_losses)
            overall_min = np.min(all_losses)
            overall_max = np.max(all_losses)
            
            # Calculate overall correlations
            correlation_text = ""
            
            # Loss-Mask correlation
            if len(all_mask_sizes) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_mask_sizes))
                corr_losses = all_losses[:min_len]
                corr_sizes = all_mask_sizes[:min_len]
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_sizes, corr_losses)[0, 1]
                    correlation_text += f"Loss-Mask Correlation: {correlation:.4f}\n"
            
            # Loss-Flow Variance correlation
            if len(all_flow_variances) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_flow_variances))
                corr_losses = all_losses[:min_len]
                corr_variances = all_flow_variances[:min_len]
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_variances, corr_losses)[0, 1]
                    correlation_text += f"Loss-Flow Variance Correlation: {correlation:.4f}\n"
            
            # Loss-Flow Std correlation
            if len(all_flow_stds) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_flow_stds))
                corr_losses = all_losses[:min_len]
                corr_stds = all_flow_stds[:min_len]
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_stds, corr_losses)[0, 1]
                    correlation_text += f"Loss-Flow Std Correlation: {correlation:.4f}\n"
            
            # Flow variance and std statistics
            flow_var_stats = ""
            if len(all_flow_variances) > 0:
                mean_flow_var = np.mean(all_flow_variances)
                std_flow_var = np.std(all_flow_variances)
                flow_var_stats += f"""
            GT Flow Variance:
            Mean: {mean_flow_var:.6f}
            Std: {std_flow_var:.6f}
            """
            
            if len(all_flow_stds) > 0:
                mean_flow_std = np.mean(all_flow_stds)
                std_flow_std = np.std(all_flow_stds)
                flow_var_stats += f"""
            GT Flow Std:
            Mean: {mean_flow_std:.6f}
            Std: {std_flow_std:.6f}
            """
            
            summary_text = f"""
            Summary Statistics:
            
            Analyzed Sequences: {len(all_sequence_results)}
            Total Valid Frames: {len(all_losses)}
            
            Overall Mean Loss: {overall_mean:.6f}
            Overall Std Loss: {overall_std:.6f}
            Min Loss: {overall_min:.6f}
            Max Loss: {overall_max:.6f}
            
            Loss Range: {overall_max - overall_min:.6f}
            Coefficient of Variation: {overall_std/overall_mean:.4f}
            
            {correlation_text}
            {flow_var_stats}
            Model: Trained Model Inference
            """
            
            ax10.text(0.1, 0.9, summary_text, transform=ax10.transAxes, fontsize=10,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig("flow_smooth_loss_all_sequences_summary.png", dpi=300, bbox_inches='tight')
            print("Summary analysis saved as: flow_smooth_loss_all_sequences_summary.png")
            
            # Print key conclusions
            print(f"\n=== Key Conclusions ===")
            print(f"Under trained model inference setting:")
            print(f"- Average Loss: {overall_mean:.6f}")
            print(f"- Loss Range: {overall_max - overall_min:.6f}")
            print(f"- Coefficient of Variation: {overall_std/overall_mean:.4f}")
            
            if len(all_mask_sizes) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_mask_sizes))
                corr_losses = all_losses[:min_len]
                corr_sizes = all_mask_sizes[:min_len]
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_sizes, corr_losses)[0, 1]
                    print(f"- Loss-Mask Correlation: {correlation:.4f}")
            
            if len(all_flow_variances) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_flow_variances))
                corr_losses = all_losses[:min_len]
                corr_variances = all_flow_variances[:min_len]
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_variances, corr_losses)[0, 1]
                    print(f"- Loss-Flow Variance Correlation: {correlation:.4f}")
            
            if len(all_flow_stds) > 1 and len(all_losses) > 1:
                min_len = min(len(all_losses), len(all_flow_stds))
                corr_losses = all_losses[:min_len]
                corr_stds = all_flow_stds[:min_len]
                if len(corr_losses) > 1:
                    correlation = np.corrcoef(corr_stds, corr_losses)[0, 1]
                    print(f"- Loss-Flow Std Correlation: {correlation:.4f}")
            
            if overall_std/overall_mean < 0.1:
                print("- Conclusion: Loss is relatively stable across sequences")
            elif overall_std/overall_mean < 0.3:
                print("- Conclusion: Loss shows moderate variation across sequences")
            else:
                print("- Conclusion: Loss shows significant variation across sequences")
    
    print(f"\nAnalysis completed! Processed {sequence_count} sequences")

if __name__ == "__main__":
    main()

