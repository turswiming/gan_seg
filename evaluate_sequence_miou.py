#!/usr/bin/env python3
"""
Script to evaluate mask predictor model on multi-frame sequences and calculate mIoU.

This script loads a trained mask predictor model and evaluates it on the entire
AV2Sequence dataset to compute mIoU metrics across multiple frames.

Usage:
    python evaluate_sequence_miou.py --checkpoint path/to/checkpoint.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append('/workspace/gan_seg/src')

from dataset.av2_dataset import AV2SequenceDataset
from utils.metrics import calculate_miou, calculate_epe
from utils.visualization_utils import remap_instance_labels
from Predictor import get_mask_predictor, get_scene_flow_predictor
from config.config import print_config
from utils.config_utils import load_config_with_inheritance, save_config_and_code
from model.eulerflow_raw_mlp import QueryDirection

def load_checkpoint(checkpoint_path, device):
    """
    Load model checkpoint from file.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load checkpoint on
        
    Returns:
        dict: Checkpoint data
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def create_mask_predictor_and_flow_predictor_from_checkpoint(checkpoint, config, device, point_length=65536):
    """
    Create mask predictor model from checkpoint.
    
    Args:
        checkpoint (dict): Checkpoint data
        config (Config): Configuration object
        device (torch.device): Device to create model on
        point_length (int): Number of points in point cloud
        
    Returns:
        torch.nn.Module: Loaded mask predictor model
    """
    # Extract mask predictor state from checkpoint
    mask_state = None
    if "mask_predictor" in checkpoint:
        mask_state = checkpoint["mask_predictor"]
    elif "model" in checkpoint:
        mask_state = checkpoint["model"]
    else:
        mask_state = checkpoint
    if "scene_flow_predictor" in checkpoint:
        flow_state = checkpoint["scene_flow_predictor"]
    elif "model" in checkpoint:
        flow_state = checkpoint["model"]
    else:
        flow_state = checkpoint
    
    # Create model using configuration
    mask_predictor = get_mask_predictor(config.model.mask, point_length)
    flow_predictor = get_scene_flow_predictor(config.model.flow, point_length)
    
    # Load state dict
    try:
        mask_predictor.load_state_dict(mask_state)
        flow_predictor.load_state_dict(flow_state)
        print(f"Successfully loaded mask predictor: {type(mask_predictor).__name__}")
        print(f"Successfully loaded flow predictor: {type(flow_predictor).__name__}")
    except Exception as e:
        print(f"Failed to load mask predictor state dict: {e}")
        print("Available keys in checkpoint:", list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Not a dict")
        raise e
    
    return mask_predictor.to(device), flow_predictor.to(device)

def evaluate_sequence_miou(mask_predictor,flow_predictor, dataset, device, max_frames=None):
    """
    Evaluate mask predictor on entire sequence and calculate mIoU.
    
    Args:
        mask_predictor (torch.nn.Module): Trained mask predictor model
        dataset (AV2SequenceDataset): Dataset containing sequence data
        device (torch.device): Device to run evaluation on
        max_frames (int, optional): Maximum number of frames to evaluate
        
    Returns:
        dict: Evaluation results containing mIoU statistics
    """
    mask_predictor.eval()
    flow_predictor.eval()
    miou_list = []
    frame_results = []
    
    print(f"Evaluating on {len(dataset)} frames...")
    
    with torch.no_grad():
        for frame_idx in range(len(dataset)):
            if max_frames and frame_idx >= max_frames:
                break
                
            try:
                # Get data for current frame
                sample = dataset.get_item(frame_idx)
                
                # Extract point clouds, ground truth masks, and flows
                point_cloud_first = sample["point_cloud_first"].to(device)
                sample_idx = sample["idx"]
                dataset = sample["self"]
                gt_mask = sample["dynamic_instance_mask"].to(device)
                gt_flow = sample["flow"].to(device)
                
                # Get point cloud dimensions
                N = point_cloud_first.shape[0]
                
                
                # Predict masks for both frames
                from model.mask_predict_model import EulerMaskMLP ,EulerMaskMLPResidual, EulerMaskMLPRoutine
                if isinstance(mask_predictor, EulerMaskMLP) or isinstance(mask_predictor, EulerMaskMLPResidual) or isinstance(mask_predictor, EulerMaskMLPRoutine):
                    pred_mask_first = mask_predictor(point_cloud_first, sample_idx, sample["total_frames"])
                else:
                    pred_mask_first = mask_predictor(point_cloud_first)
                
                # Calculate predicted flow
                pred_flow = flow_predictor(point_cloud_first, sample_idx, sample["total_frames"], QueryDirection.FORWARD)
                
                # Calculate mIoU and EPE for first frame
                miou_value = 0.0
                epe_value = 0.0
                num_pred_instances = 0
                num_gt_instances = 0
                
                if pred_mask_first.shape[0] > 0 and gt_mask.shape[0] > 0:
                    # Remap ground truth labels
                    gt_mask_remapped = remap_instance_labels(gt_mask)
                    
                    # Convert to one-hot format
                    gt_mask_onehot = F.one_hot(gt_mask_remapped.to(torch.long)).permute(1, 0).to(device=device)
                    
                    # Calculate mIoU
                    pred_mask_first = pred_mask_first.permute(1, 0)
                    miou_first = calculate_miou(pred_mask_first, gt_mask_onehot)
                    miou_value = miou_first.item()
                    miou_list.append(miou_value)
                    
                    num_pred_instances = pred_mask_first.shape[0]
                    num_gt_instances = gt_mask_onehot.shape[0]
                
                # Calculate EPE if flow data is available
                if pred_flow is not None and gt_flow is not None:
                    try:
                        epe_first = calculate_epe([pred_flow], [gt_flow])
                        epe_value = epe_first.item()
                    except Exception as e:
                        print(f"EPE calculation error for frame {frame_idx}: {e}")
                        epe_value = 0.0
                else:
                    epe_value = 0.0
                
                frame_results.append({
                    'frame_idx': frame_idx,
                    'miou': miou_value,
                    'epe': epe_value,
                    'num_pred_instances': num_pred_instances,
                    'num_gt_instances': num_gt_instances
                })
                
                print(f"Frame {frame_idx:3d}: mIoU = {miou_value:.4f}, EPE = {epe_value:.4f}, "
                      f"Pred instances: {num_pred_instances}, GT instances: {num_gt_instances}")
                    
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                import traceback
                traceback.print_exc()
                frame_results.append({
                    'frame_idx': frame_idx,
                    'miou': 0.0,
                    'epe': 0.0,
                    'num_pred_instances': 0,
                    'num_gt_instances': 0,
                    'error': str(e)
                })
                continue
    
    # Calculate overall statistics
    valid_mious = [r['miou'] for r in frame_results if r['miou'] > 0]
    valid_epes = [r['epe'] for r in frame_results if r['epe'] > 0]
    
    results = {
        'frame_results': frame_results,
        'miou_list': miou_list,
        'overall_miou': np.mean(valid_mious) if valid_mious else 0.0,
        'miou_std': np.std(valid_mious) if valid_mious else 0.0,
        'miou_min': np.min(valid_mious) if valid_mious else 0.0,
        'miou_max': np.max(valid_mious) if valid_mious else 0.0,
        'overall_epe': np.mean(valid_epes) if valid_epes else 0.0,
        'epe_std': np.std(valid_epes) if valid_epes else 0.0,
        'epe_min': np.min(valid_epes) if valid_epes else 0.0,
        'epe_max': np.max(valid_epes) if valid_epes else 0.0,
        'num_valid_frames': len(valid_mious),
        'num_valid_epe_frames': len(valid_epes),
        'total_frames': len(frame_results)
    }
    
    return results

def create_visualizations(results, save_plots=True, show_plots=True):
    """
    Create comprehensive visualizations of the evaluation results.
    
    Args:
        results (dict): Evaluation results
        save_plots (bool): Whether to save plots to files
        show_plots (bool): Whether to display plots interactively
    """
    frame_results = results['frame_results']
    valid_results = [r for r in frame_results if r['miou'] > 0 and 'error' not in r]
    
    if not valid_results:
        print("No valid results to visualize")
        return
    
    # Extract data for plotting
    frame_indices = [r['frame_idx'] for r in valid_results]
    miou_values = [r['miou'] for r in valid_results]
    epe_values = [r['epe'] for r in valid_results]
    pred_counts = [r['num_pred_instances'] for r in valid_results]
    gt_counts = [r['num_gt_instances'] for r in valid_results]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: mIoU and EPE Timeline (top row, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1_twin = ax1.twinx()
    
    # Plot mIoU
    line1 = ax1.plot(frame_indices, miou_values, 'b-', linewidth=2, marker='o', markersize=4, label='mIoU')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('mIoU', color='b')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot EPE
    line2 = ax1_twin.plot(frame_indices, epe_values, 'r-', linewidth=2, marker='s', markersize=4, label='EPE')
    ax1_twin.set_ylabel('EPE', color='r')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    ax1.set_title('mIoU and EPE Performance Across Sequence Frames')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"mIoU: {results['overall_miou']:.4f}±{results['miou_std']:.4f}\nEPE: {results['overall_epe']:.4f}±{results['epe_std']:.4f}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    # Plot 2: EPE Distribution (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(epe_values, bins=15, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('EPE')
    ax2.set_ylabel('Frequency')
    ax2.set_title('EPE Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: EPE vs Frame Index (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(frame_indices, epe_values, alpha=0.6, c='red', s=50)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('EPE')
    ax3.set_title('EPE vs Frame Index')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_epe_frame = np.corrcoef(frame_indices, epe_values)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {corr_epe_frame:.3f}', 
             transform=ax3.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Plot 4: EPE vs Ground Truth Instances (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter(gt_counts, epe_values, alpha=0.6, c='orange', s=50)
    ax4.set_xlabel('Ground Truth Instances')
    ax4.set_ylabel('EPE')
    ax4.set_title('EPE vs Ground Truth Instances')
    ax4.grid(True, alpha=0.3)
    
    corr_epe_gt = np.corrcoef(gt_counts, epe_values)[0, 1]
    ax4.text(0.05, 0.95, f'Correlation: {corr_epe_gt:.3f}', 
             transform=ax4.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 5: mIoU vs EPE Correlation (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.scatter(miou_values, epe_values, alpha=0.6, c='purple', s=50)
    ax5.set_xlabel('mIoU')
    ax5.set_ylabel('EPE')
    ax5.set_title('mIoU vs EPE Correlation')
    ax5.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr_miou_epe = np.corrcoef(miou_values, epe_values)[0, 1]
    ax5.text(0.05, 0.95, f'Correlation: {corr_miou_epe:.3f}', 
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
    
    # Plot 6: EPE vs GT Instances Scatter (bottom left)
    ax6 = fig.add_subplot(gs[2, 0])
    scatter = ax6.scatter(gt_counts, epe_values, alpha=0.6, c=frame_indices, 
                         cmap='viridis', s=50)
    ax6.set_xlabel('Ground Truth Instances')
    ax6.set_ylabel('EPE')
    ax6.set_title('EPE vs GT Instances (colored by frame)')
    ax6.grid(True, alpha=0.3)
    
    # Add colorbar for frame index
    cbar = plt.colorbar(scatter, ax=ax6)
    cbar.set_label('Frame Index')
    
    # Plot 7: EPE Performance by Frame Segments (bottom center)
    ax7 = fig.add_subplot(gs[2, 1])
    # Divide frames into segments for analysis
    num_segments = min(8, len(frame_indices) // 5)
    if num_segments > 1:
        segment_size = len(frame_indices) // num_segments
        segment_centers = []
        segment_means = []
        segment_stds = []
        
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else len(frame_indices)
            segment_frames = frame_indices[start_idx:end_idx]
            segment_epes = epe_values[start_idx:end_idx]
            
            if segment_epes:
                segment_centers.append(np.mean(segment_frames))
                segment_means.append(np.mean(segment_epes))
                segment_stds.append(np.std(segment_epes))
        
        if segment_centers:
            ax7.errorbar(segment_centers, segment_means, yerr=segment_stds, 
                        marker='o', capsize=5, capthick=2, linewidth=2)
            ax7.set_xlabel('Frame Index (Segment Center)')
            ax7.set_ylabel('Mean EPE')
            ax7.set_title('EPE Performance by Frame Segments')
            ax7.grid(True, alpha=0.3)
    
    # Plot 8: Statistics Summary (bottom right)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    # Create comprehensive statistics text
    stats_text = f"""
    EVALUATION SUMMARY
    ==================
    Total Frames: {results['total_frames']}
    Valid mIoU Frames: {results['num_valid_frames']}
    Valid EPE Frames: {results['num_valid_epe_frames']}
    
    mIoU: {results['overall_miou']:.4f} ± {results['miou_std']:.4f}
    Range: [{results['miou_min']:.4f}, {results['miou_max']:.4f}]
    
    EPE: {results['overall_epe']:.4f} ± {results['epe_std']:.4f}
    Range: [{results['epe_min']:.4f}, {results['epe_max']:.4f}]
    
    CORRELATION ANALYSIS
    ====================
    mIoU vs Frame: {np.corrcoef(frame_indices, miou_values)[0,1]:.3f}
    EPE vs Frame: {corr_epe_frame:.3f}
    mIoU vs EPE: {corr_miou_epe:.3f}
    
    INSTANCE STATISTICS
    ===================
    Mean GT Instances: {np.mean(gt_counts):.2f} ± {np.std(gt_counts):.2f}
    GT vs EPE: {corr_epe_gt:.3f}
    """
    
    ax8.text(0.1, 0.9, stats_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.suptitle('mIoU and EPE Evaluation Results - Performance and Correlation Analysis', 
                 fontsize=16, fontweight='bold')
    
    if save_plots:
        plt.savefig('miou_evaluation_plots.png', dpi=300, bbox_inches='tight')
        print("Plots saved to: miou_evaluation_plots.png")
    
    if show_plots:
        plt.show()
    else:
        plt.close()

def print_results(results):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results (dict): Evaluation results
    """
    print("\n" + "="*60)
    print("SEQUENCE EVALUATION RESULTS")
    print("="*60)
    print(f"Total frames evaluated: {results['total_frames']}")
    print(f"Valid mIoU frames: {results['num_valid_frames']}")
    print(f"Valid EPE frames: {results['num_valid_epe_frames']}")
    print(f"Overall mIoU: {results['overall_miou']:.4f}")
    print(f"mIoU std: {results['miou_std']:.4f}")
    print(f"mIoU min: {results['miou_min']:.4f}")
    print(f"mIoU max: {results['miou_max']:.4f}")
    print(f"Overall EPE: {results['overall_epe']:.4f}")
    print(f"EPE std: {results['epe_std']:.4f}")
    print(f"EPE min: {results['epe_min']:.4f}")
    print(f"EPE max: {results['epe_max']:.4f}")
    print("="*60)
    
    # Print frame-by-frame results
    print("\nFrame-by-frame results:")
    print("Frame | mIoU   | EPE    | GT   | Status")
    print("-" * 45)
    for result in results['frame_results']:
        status = "OK"
        if 'error' in result:
            status = "ERROR"
        elif result['miou'] == 0.0:
            status = "SKIP"
            
        print(f"{result['frame_idx']:5d} | {result['miou']:6.4f} | "
              f"{result['epe']:6.4f} | {result['num_gt_instances']:4d} | {status}")
    
    # Calculate and print correlations
    valid_results = [r for r in results['frame_results'] if r['miou'] > 0 and 'error' not in r]
    if len(valid_results) > 1:
        frame_indices = [r['frame_idx'] for r in valid_results]
        miou_values = [r['miou'] for r in valid_results]
        epe_values = [r['epe'] for r in valid_results]
        gt_counts = [r['num_gt_instances'] for r in valid_results]
        
        corr_miou_frame = np.corrcoef(frame_indices, miou_values)[0, 1]
        corr_epe_frame = np.corrcoef(frame_indices, epe_values)[0, 1]
        corr_miou_epe = np.corrcoef(miou_values, epe_values)[0, 1]
        corr_epe_gt = np.corrcoef(gt_counts, epe_values)[0, 1]
        
        print(f"\nCorrelation Analysis:")
        print(f"Frame Index vs mIoU: {corr_miou_frame:.4f}")
        print(f"Frame Index vs EPE:  {corr_epe_frame:.4f}")
        print(f"mIoU vs EPE:         {corr_miou_epe:.4f}")
        print(f"GT Instances vs EPE: {corr_epe_gt:.4f}")

def main():
    """Main function to run sequence evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate mask predictor on sequence data')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint file')
    parser.add_argument('--config', type=str, default='/workspace/gan_seg/src/config/baseconfig.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to run evaluation on')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to evaluate (None for all)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results (optional)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create and display visualization plots')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files')
    parser.add_argument('--no-show-plots', action='store_true',
                       help='Do not display plots interactively')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # 1. Load configuration
        print("Loading configuration...")
        config = load_config_with_inheritance(args.config)
        
        # 2. Load checkpoint
        print("Loading checkpoint...")
        checkpoint = load_checkpoint(args.checkpoint, device)
        
        # 3. Create mask predictor
        print("Creating mask predictor...")
        mask_predictor, flow_predictor = create_mask_predictor_and_flow_predictor_from_checkpoint(
            checkpoint, config, device, point_length=65536
        )
        
        # 4. Load dataset
        print("Loading AV2Sequence dataset...")
        dataset = AV2SequenceDataset(
            fix_ego_motion=True,
            max_k=1,
            apply_ego_motion=True
        )
        print(f"Dataset length: {len(dataset)}")
        
        # 5. Evaluate sequence
        print("Evaluating sequence...")
        results = evaluate_sequence_miou(
            mask_predictor, flow_predictor, dataset, device, max_frames=args.max_frames
        )
        
        # 6. Print results
        print_results(results)
        
        # 7. Create visualizations if requested
        if args.visualize:
            print("\nCreating visualizations...")
            create_visualizations(
                results, 
                save_plots=args.save_plots,
                show_plots=not args.no_show_plots
            )
        
        # 8. Save results if requested
        if args.output:
            print(f"Saving results to: {args.output}")
            torch.save(results, args.output)
        
        print("\nEvaluation completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
