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
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from config.config import print_config
from tqdm import tqdm

# Local imports
import open3d as o3d
from dataset.av2_dataset import AV2PerSceneDataset
from dataset.movi_per_scene_dataset import MOVIPerSceneDataset
from utils.config_utils import load_config_with_inheritance, save_config_and_code
from utils.dataloader_utils import create_dataloaders
from utils.visualization_utils import remap_instance_labels, create_label_colormap, color_mask
from utils.metrics import calculate_miou, calculate_epe
from utils.optimizer_utils import ProximalOptimizer
from model.scene_flow_predict_model import OptimizedFLowPredictor, Neural_Prior, SceneFlowPredictor
from model.mask_predict_model import OptimizedMaskPredictor
from losses.ChamferDistanceLoss import ChamferDistanceLoss
from losses.ReconstructionLoss import ReconstructionLoss
from losses.PointSmoothLoss import PointSmoothLoss
from losses.FlowSmoothLoss import FlowSmoothLoss
from visualize.open3d_func import visualize_vectors, update_vector_visualization
from visualize.pca import pca
from Predictor import get_mask_predictor, get_scene_flow_predictor

def evaluate_predictions(pred_flows, gt_flows, pred_masks, gt_masks, device, writer, step):
    """
    Evaluate model predictions by computing EPE and mIoU metrics.
    
    Args:
        pred_flows (list[torch.Tensor]): Predicted scene flows
        gt_flows (list[torch.Tensor]): Ground truth scene flows
        pred_masks (list[torch.Tensor]): Predicted instance masks
        gt_masks (list[torch.Tensor]): Ground truth instance masks
        device (torch.device): Device to run computations on
        writer (SummaryWriter): TensorBoard writer for logging
        step (int): Current training step
        
    Returns:
        tuple: (epe_mean, miou_mean) containing the computed metrics
    """
    # Compute EPE
    epe_mean = calculate_epe(pred_flows, gt_flows)
    tqdm.write(f"\rEPE: {epe_mean.item()}", end="")
    writer.add_scalar("epe", epe_mean.item(), step)
    
    # Compute mIoU
    miou_list = []
    for i in range(len(pred_masks)):
        gt_mask = remap_instance_labels(gt_masks[i])
        # tqdm.write(f"gt_mask size {max(gt_mask)}")
        miou_list.append(
            calculate_miou(
                pred_masks[i], 
                F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(device=device)
            )
        )
    miou_mean = torch.mean(torch.stack(miou_list))
    # tqdm.write(f"miou {miou_mean.item()}")
    writer.add_scalar("miou", miou_mean.item(), step)
    
    return epe_mean, miou_mean

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
    optimizer_flow = torch.optim.AdamW(scene_flow_predictor.parameters(), lr=config.model.flow.lr)
    optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=config.model.mask.lr)
    
    loop = 100
    scene_flow_scheduler = lambda iter: iter%loop/loop
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
            tqdm.write(f"epe: {epe.mean().item()}")
        
        step += 1
        if step > config.training.max_iter:
            break
        if step // loop%2 == 0:
            train_flow = True
            train_mask = False
        else:
            train_flow = False
            train_mask = True
        train_flow = True
        train_mask = True
        # Forward pass
        point_cloud_firsts = [item.to(device) for item in sample["point_cloud_first"]]
        if train_flow:
            pred_flow = []
            for i in range(len(point_cloud_firsts)):
                pred_flow.append(scene_flow_predictor(point_cloud_firsts[i]))  # Shape: [B, N, 3]
            gt_flow = [flow.to(device) for flow in sample["flow"]]  # Shape: [B, N, 3]
        else:
            with torch.no_grad():
                pred_flow = []
                for i in range(len(point_cloud_firsts)):
                    pred_flow.append(scene_flow_predictor(point_cloud_firsts[i]))
                gt_flow = [flow.to(device) for flow in sample["flow"]]  # Shape: [B, N, 3]
        if train_mask:
            pred_mask = []
            for i in range(len(point_cloud_firsts)):
                pred_mask.append(mask_predictor(point_cloud_firsts[i]))
        else:
            with torch.no_grad():
                pred_mask =[]
                for i in range(len(point_cloud_firsts)):
                    pred_mask.append(mask_predictor(point_cloud_firsts[i]))
        # Compute losses
        if config.lr_multi.rec_loss > 0 or config.lr_multi.rec_flow_loss > 0:
            rec_loss, reconstructed_points = reconstructionLoss(sample, pred_mask, pred_flow)
            rec_loss = rec_loss * config.lr_multi.rec_loss
        else:
            rec_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.scene_flow_smoothness > 0:
            scene_flow_smooth_loss = flowSmoothLoss(sample, pred_mask, pred_flow)
            scene_flow_smooth_loss = scene_flow_smooth_loss * config.lr_multi.scene_flow_smoothness
            # scene_flow_smooth_loss = scene_flow_smooth_loss * scene_flow_scheduler(step)
        else:
            scene_flow_smooth_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.rec_flow_loss > 0:
            rec_flow_loss = torch.tensor(0.0, device=device)
            for i in range(len(point_cloud_firsts)):
                pred_second_point = point_cloud_firsts[i] + pred_flow[i]
                rec_flow_loss += flowRecLoss(pred_second_point, reconstructed_points[i])
            rec_flow_loss = rec_flow_loss * config.lr_multi.rec_flow_loss
        else:
            rec_flow_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.flow_loss > 0:
            flow_loss = torch.tensor(0.0, device=device)
            for i in range(len(point_cloud_firsts)):
                pred_second_points = point_cloud_firsts[i] + pred_flow[i]
                flow_loss += chamferLoss(pred_second_points.unsqueeze(0), sample["point_cloud_second"][i].to(device).unsqueeze(0))
            flow_loss = flow_loss * config.lr_multi.flow_loss
        else:
            flow_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.point_smooth_loss > 0:
            point_smooth_loss = pointsmoothloss(point_cloud_firsts, pred_mask)
            point_smooth_loss = point_smooth_loss * config.lr_multi.point_smooth_loss
        else:
            point_smooth_loss = torch.tensor(0.0, device=device)

        # Combine losses
        loss = rec_loss + flow_loss + scene_flow_smooth_loss + rec_flow_loss + point_smooth_loss

        # Log losses
        tqdm.write(f"rec_loss: {rec_loss.item()}")
        tqdm.write(f"flow_loss: {flow_loss.item()}")
        tqdm.write(f"scene_flow_smooth_loss: {scene_flow_smooth_loss.item()}")
        tqdm.write(f"rec_flow_loss: {rec_flow_loss.item()}")
        tqdm.write(f"point_smooth_loss: {point_smooth_loss.item()}")
        tqdm.write(f"iteration: {step}")

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
        optimizer_flow.zero_grad()
        optimizer_mask.zero_grad()
        loss.backward()
        
        # Log gradients if needed
        if hasattr(pred_flow, 'grad') and pred_flow.grad is not None:
            tqdm.write(f"pred_flow.grad {pred_flow.grad.std().item()}")
        if hasattr(pred_mask, 'grad') and pred_mask.grad is not None:
            tqdm.write(f"pred_mask.grad {pred_mask.grad.std().item()}")
            
        optimizer_flow.step()
        optimizer_mask.step()
        
        # Evaluate predictions
        epe, miou = evaluate_predictions(
            pred_flow, 
            gt_flow, 
            pred_mask, 
            sample["dynamic_instance_mask"], 
            device, 
            writer, 
            step
        )

        # Visualization
        if config.vis.show_window and point_cloud_firsts[0].shape[0] > 0:
            batch_idx = 0
            
            # Get data for visualization
            point_cloud_first = point_cloud_firsts[batch_idx].cpu().numpy()
            point_cloud_second = sample["point_cloud_second"][batch_idx].cpu().numpy()
            current_pred_flow = pred_flow[batch_idx].cpu().detach().numpy()
            current_pred_mask = pred_mask[batch_idx].cpu().detach()
            
            # PCA for coloring
            # current_pred_mask = current_pred_mask.permute(1, 0)  # Change to [N, K] for PCA
            # pred_color = pca(current_pred_mask)

            pred_color = color_mask(current_pred_mask)
            gt_mask = sample["dynamic_instance_mask"][0]
            gt_mask = remap_instance_labels(gt_mask)
            gt_color = color_mask(F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(torch.float32))
            writer.add_histogram("pred_color", pred_color, step)
            writer.add_histogram("gt_color", gt_color, step)
            # Update point clouds
            pred_point = point_cloud_first + current_pred_flow
            pcd.points = o3d.utility.Vector3dVector(point_cloud_first)
            pcd.colors = o3d.utility.Vector3dVector(gt_color.numpy())
            
            gt_pcd.points = o3d.utility.Vector3dVector(point_cloud_second)
            gt_pcd.paint_uniform_color([0, 1, 0])
            
            if "reconstructed_points" in locals():
                current_reconstructed = reconstructed_points[batch_idx].cpu().detach().numpy().squeeze(0)
                reconstructed_pcd.points = o3d.utility.Vector3dVector(current_reconstructed)
                reconstructed_pcd.paint_uniform_color([0, 0, 1])
                
            if first_iteration:
                vis.add_geometry(pcd)
                vis.add_geometry(gt_pcd)
                if "reconstructed_points" in locals():
                    vis.add_geometry(reconstructed_pcd)
                vis, lineset = visualize_vectors(
                    point_cloud_first,
                    current_pred_flow,
                    vis=vis,
                    color=pred_color.numpy(),
                )
                first_iteration = False
            else:
                vis.update_geometry(pcd)
                vis.update_geometry(gt_pcd)
                if "reconstructed_points" in locals():
                    vis.update_geometry(reconstructed_pcd)
                lineset = update_vector_visualization(
                    lineset,
                    point_cloud_first,
                    current_pred_flow,
                    color=pred_color.numpy(),
                )
                vis.update_geometry(lineset)
                
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