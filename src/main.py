"""
Main training script for scene flow and mask prediction.

This script handles the complete training pipeline including:
- Configuration loading and setup
- Dataset and model initialization
- Training loop with loss computation
- Visualization and logging
"""

# Standard library imports
import datetime
import argparse
import os
import shutil

# Third party imports
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from config.config import print_config
from tqdm import tqdm

# Local imports
import open3d as o3d
from dataset.av2_dataset import AV2Dataset
from dataset.per_scene_dataset import PerSceneDataset
from utils.config_utils import load_config_with_inheritance, save_config_and_code
from utils.dataloader_utils import create_dataloaders
from model.scene_flow_predict_model import FLowPredictor, Neural_Prior, SceneFlowPredictor
from model.mask_predict_model import MaskPredictor
from losses.ChamferDistanceLoss import ChamferDistanceLoss
from losses.ReconstructionLoss import ReconstructionLoss
from losses.PointSmoothLoss import PointSmoothLoss
from losses.FlowSmoothLoss import FlowSmoothLoss
from visualize.open3d_func import visualize_vectors, update_vector_visualization
from visualize.pca import pca
from Predictor import get_mask_predictor, get_scene_flow_predictor

def main(config, writer):
    """
    Main training function.
    
    Args:
        config: Configuration object containing all training parameters
        writer: TensorBoard SummaryWriter for logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataloaders
    dataloader, infinite_loader, batch_size, N = create_dataloaders(config)
    
    # Initialize models
    mask_predictor = get_mask_predictor(config.model.mask, N)
    scene_flow_predictor = get_scene_flow_predictor(config.model.flow, N)
    scene_flow_predictor.to(device)

    # Initialize optimizers
    optimizer = torch.optim.AdamW(scene_flow_predictor.parameters(), lr=config.model.flow.lr)
    optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=config.model.mask.lr)

    # Initialize loss functions
    reconstructionLoss = ReconstructionLoss(device)
    chamferLoss = ChamferDistanceLoss()
    flowSmoothLoss = FlowSmoothLoss(device)
    pointsmoothloss = PointSmoothLoss()
    flowRecLoss = nn.MSELoss()

    # Initialize visualization if enabled
    if config.vis.show_window:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        gt_pcd = o3d.geometry.PointCloud()
        reconstructed_pcd = o3d.geometry.PointCloud()
    
    first_iteration = True
    step = 0
    epe = None

    # Main training loop
    for sample in tqdm(infinite_loader, total=config.training.max_iter):
        if epe is not None:
            tqdm.write(f"epe: {epe.mean().item():.4f}")
        
        step += 1
        if step > config.training.max_iter:
            break

        # Forward pass
        pred_flow = scene_flow_predictor(sample["point_cloud_first"].to(device))  # Shape: [B, N, 3]
        gt_flow = sample["flow"].to(device)  # Shape: [B, N, 3]
        pred_mask = mask_predictor(sample)  # Shape: [B, K, N]

        # Compute losses
        if config.lr_multi.rec_loss > 0 or config.lr_multi.rec_flow_loss > 0:
            rec_loss, reconstructed_points = reconstructionLoss(sample, pred_mask, pred_flow)
            rec_loss = rec_loss * config.lr_multi.rec_loss
        else:
            rec_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.scene_flow_smoothness > 0:
            scene_flow_smooth_loss = flowSmoothLoss(sample, pred_mask, pred_flow)
            scene_flow_smooth_loss = scene_flow_smooth_loss * config.lr_multi.scene_flow_smoothness
        else:
            scene_flow_smooth_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.rec_flow_loss > 0:
            point_cloud_first = sample["point_cloud_first"].to(device)
            pred_second_points = point_cloud_first + pred_flow
            rec_flow_loss = flowRecLoss(pred_second_points, reconstructed_points)
            rec_flow_loss = rec_flow_loss * config.lr_multi.rec_flow_loss
        else:
            rec_flow_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.flow_loss > 0:
            point_cloud_first = sample["point_cloud_first"].to(device)
            pred_second_points = point_cloud_first + pred_flow
            flow_loss = chamferLoss(pred_second_points, sample["point_cloud_second"].to(device))
            flow_loss = flow_loss * config.lr_multi.flow_loss
        else:
            flow_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.point_smooth_loss > 0:
            point_smooth_loss = pointsmoothloss(sample["point_cloud_first"].to(device), pred_mask)
            point_smooth_loss = point_smooth_loss * config.lr_multi.point_smooth_loss
        else:
            point_smooth_loss = torch.tensor(0.0, device=device)

        # Combine losses
        loss = rec_loss + flow_loss + scene_flow_smooth_loss + rec_flow_loss + point_smooth_loss

        # Log losses
        print("rec_loss", rec_loss.item())
        print("flow_loss", flow_loss.item())
        print("scene_flow_smooth_loss", scene_flow_smooth_loss.item())
        print("rec_flow_loss", rec_flow_loss.item())
        print("point_smooth_loss", point_smooth_loss.item())
        print("iteration", step)

        # Log to tensorboard
        writer.add_scalars("losses", {
            "rec_loss": rec_loss.item(),
            "flow_loss": flow_loss.item(),
            "scene_flow_smooth_loss": scene_flow_smooth_loss.item(),
            "rec_flow_loss": rec_flow_loss.item(),
            "point_smooth_loss": point_smooth_loss.item(),
            "total_loss": loss.item(),
        }, step)
        
        # Backward pass
        optimizer.zero_grad()
        optimizer_mask.zero_grad()
        loss.backward()
        
        # Log gradients if needed
        if hasattr(pred_flow, 'grad') and pred_flow.grad is not None:
            print("pred_flow.grad", pred_flow.grad.std().item())
        if hasattr(pred_mask, 'grad') and pred_mask.grad is not None:
            print("pred_mask.grad", pred_mask.grad.std().item())
            
        optimizer.step()
        optimizer_mask.step()
        
        # Compute EPE (End Point Error)
        epe = torch.norm(pred_flow - gt_flow, dim=2, p=2)  # Shape: [B, N]
        epe_mean = epe.mean()
        print("epe", epe_mean.item())
        writer.add_scalar("epe", epe_mean.item(), step)

        # Visualization
        if config.vis.show_window and sample["point_cloud_first"].shape[0] > 0:
            batch_idx = 0
            
            # Get data for visualization
            point_cloud_first = sample["point_cloud_first"][batch_idx].cpu().numpy()
            point_cloud_second = sample["point_cloud_second"][batch_idx].cpu().numpy()
            current_pred_flow = pred_flow[batch_idx].cpu().detach().numpy()
            current_pred_mask = pred_mask[batch_idx].cpu().detach()
            
            # PCA for coloring
            current_pred_mask = current_pred_mask.permute(1, 0)  # Change to [N, K] for PCA
            color = pca(current_pred_mask)
            
            # Update point clouds
            pred_point = point_cloud_first + current_pred_flow
            pcd.points = o3d.utility.Vector3dVector(pred_point)
            pcd.colors = o3d.utility.Vector3dVector(color.numpy())
            
            gt_pcd.points = o3d.utility.Vector3dVector(point_cloud_second)
            gt_pcd.paint_uniform_color([0, 1, 0])
            
            if "reconstructed_points" in locals():
                current_reconstructed = reconstructed_points[batch_idx].cpu().detach().numpy()
                reconstructed_pcd.points = o3d.utility.Vector3dVector(current_reconstructed)
                reconstructed_pcd.paint_uniform_color([0, 0, 1])
                
            if first_iteration:
                # vis.add_geometry(pcd)
                vis.add_geometry(gt_pcd)
                if "reconstructed_points" in locals():
                    vis.add_geometry(reconstructed_pcd)
                vis, lineset = visualize_vectors(
                    point_cloud_first,
                    current_pred_flow,
                    vis=vis,
                    color=color.numpy(),
                )
                first_iteration = False
            else:
                # vis.update_geometry(pcd)
                lineset = update_vector_visualization(
                    lineset,
                    point_cloud_first,
                    current_pred_flow,
                    color=color.numpy(),
                )
                vis.update_geometry(lineset)
                vis.update_geometry(gt_pcd)
                if "reconstructed_points" in locals():
                    vis.update_geometry(reconstructed_pcd)
            vis.poll_events()
            vis.update_renderer()

    # Cleanup
    if config.vis.show_window:
        vis.destroy_window()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scene Flow and Mask Prediction")
    parser.add_argument("--config", type=str, default="config/baseconfig.yaml", 
                       help="Path to the config file")
    
    args, unknown = parser.parse_known_args()
    config_obj = load_config_with_inheritance(args.config)

    print_config(config_obj)
    cli_opts = OmegaConf.from_cli()
    print_config(cli_opts)
    
    # Merge configs
    config = OmegaConf.merge(config_obj, cli_opts)
    
    # Setup logging directory
    if config.log.dir == "":
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.log.dir = f"../outputs/exp/{time_str}"
    
    writer = SummaryWriter(log_dir=config.log.dir)
    
    # Save config and code
    save_config_and_code(config, config.log.dir)

    # Start training
    main(config, writer)