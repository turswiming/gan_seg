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
from model.scene_flow_predict_model import OptimizedFLowPredictor, Neural_Prior, SceneFlowPredictor
from model.mask_predict_model import OptimizedMaskPredictor
from losses.ChamferDistanceLoss import ChamferDistanceLoss
from losses.ReconstructionLoss import ReconstructionLoss
from losses.PointSmoothLoss import PointSmoothLoss
from losses.FlowSmoothLoss import FlowSmoothLoss
from visualize.open3d_func import visualize_vectors, update_vector_visualization
from visualize.pca import pca
from Predictor import get_mask_predictor, get_scene_flow_predictor

def calculate_miou(pred_mask, gt_mask):
    """
    Calculate Mean Intersection over Union (mIoU) for the predicted masks.
    
    Args:
        pred_mask (torch.Tensor): Predicted masks [K, N]
        gt_mask (torch.Tensor): Ground truth masks [K, N]
        
    Returns:
        float: Mean IoU value
    """
    pred_mask = torch.softmax(pred_mask, dim=0)
    pred_mask = torch.argmax(pred_mask, dim=0)
    pred_mask = F.one_hot(pred_mask).permute(1, 0).to(device=gt_mask.device)
    pred_mask = pred_mask>0.49
    pred_mask = pred_mask.to(dtype=torch.float32)
    max_iou_list = []
    gt_mask = gt_mask>0.5
    #pre calculate the size of each mask
    gt_mask_size = torch.sum(gt_mask, dim=1)
    pred_mask_size = torch.sum(pred_mask, dim=1)
    for j in range(gt_mask.shape[0]):
        if j == 0:
            continue#skip the background
        max_iou = 0
        print(f"gt_mask {j} size {gt_mask_size[j]}")
        for i in range(pred_mask.shape[0]):
        
            intersection = torch.sum(pred_mask[i] * gt_mask[j])
            union = pred_mask_size[i] + gt_mask_size[j] - intersection
            iou = float(intersection) / float(union) if union != 0 else 0
            if iou > max_iou:
                max_iou = iou
        print(f"max_iou {max_iou}")
        max_iou_list.append(max_iou)
    print(f"max_iou_list {max_iou_list}")
    mean_iou = torch.mean(torch.tensor(max_iou_list).to(dtype=torch.float32))
    return mean_iou

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

def create_label_colormap():
    """Creates a label colormap used in CITYSCAPES segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
    colormap = np.zeros((256, 3), dtype=np.int64)
    colormap[0] = [100, 134, 102]
    colormap[1] = [166, 206, 227]
    colormap[2] = [31, 120, 180]
    colormap[3] = [178, 223, 138]
    colormap[4] = [51, 160, 44]
    colormap[5] = [251, 154, 153]
    colormap[6] = [227, 26, 28]
    colormap[7] = [253, 191, 111]
    colormap[8] = [255, 127, 0]
    colormap[9] = [202, 178, 214]
    colormap[10] = [106, 61, 154]
    colormap[11] = [255, 255, 153]
    colormap[12] = [177, 89, 40]
    colormap[13] = [0, 0, 142]
    colormap[14] = [0, 0, 70]
    colormap[15] = [0, 60, 100]
    colormap[16] = [0, 80, 100]
    colormap[17] = [0, 0, 230]
    colormap[18] = [119, 11, 32]
    colormap[19] = [100, 134, 102]
    colormap[20] = [166, 206, 227]
    colormap[21] = [31, 120, 180]
    colormap[22] = [178, 223, 138]
    colormap[23] = [51, 160, 44]
    colormap[24] = [251, 154, 153]
    colormap[25] = [227, 26, 28]
    colormap[26] = [253, 191, 111]
    colormap[27] = [255, 127, 0]
    colormap[28] = [202, 178, 214]
    colormap[29] = [106, 61, 154]
    colormap[30] = [255, 255, 153]
    colormap[31] = [177, 89, 40]
    colormap[32] = [0, 0, 142]
    colormap[33] = [0, 0, 70]
    colormap[34] = [0, 60, 100]
    colormap[35] = [0, 80, 100]
    colormap[36] = [0, 0, 230]
    colormap[37] = [119, 11, 32]
    colormap[38] = [100, 134, 102]
    colormap[39] = [166, 206, 227]
    colormap[40] = [31, 120, 180]
    colormap[41] = [178, 223, 138]
    colormap[42] = [51, 160, 44]
    colormap[43] = [251, 154, 153]
    colormap[44] = [227, 26, 28]
    colormap[45] = [253, 191, 111]
    colormap[46] = [255, 127, 0]
    colormap[47] = [202, 178, 214]
    colormap[48] = [106, 61, 154]
    colormap[49] = [255, 255, 153]
    colormap[50] = [177, 89, 40]
    colormap[51] = [0, 0, 142]
    colormap[52] = [0, 0, 70]
    colormap[53] = [0, 60, 100]
    colormap[54] = [0, 80, 100]
    colormap[55] = [0, 0, 230]
    colormap[56] = [119, 11, 32]
    colormap[57] = [100, 134, 102]
    colormap[58] = [166, 206, 227]
    colormap[59] = [31, 120, 180]
    colormap[60] = [178, 223, 138]
    colormap[61] = [51, 160, 44]
    colormap[62] = [251, 154, 153]
    colormap[63] = [227, 26, 28]
    colormap[64] = [253, 191, 111]
    colormap[65] = [255, 127, 0]
    colormap[66] = [202, 178, 214]
    colormap[67] = [106, 61, 154]
    colormap[68] = [255, 255, 153]
    colormap[69] = [177, 89, 40]
    colormap[70] = [0, 0, 142]

    return torch.from_numpy(colormap).long()

def color_mask(mask):
    """
    Color the mask using PCA for visualization.
    
    Args:
        mask (torch.Tensor): Input mask [K, N]
        
    Returns:
        torch.Tensor: Colored mask [N, 3]
    """
    color_label = create_label_colormap()
    #get all different values in mask
    mask_argmax = torch.argmax(mask, dim=0)
    unique_values = torch.unique(mask_argmax)
    color_result = torch.zeros((mask_argmax.shape[0], 3), dtype=torch.float32)
    for i in range(len(unique_values)):
        # Convert color_label to float before assignment
        color_result[mask_argmax == unique_values[i]] = color_label[unique_values[i]].to(torch.float32)
    color_result = color_result / 255.0
    return color_result

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
        print(f"pred_mask shape {pred_mask[0].shape}")
        # Compute losses
        if config.lr_multi.rec_loss > 0 or config.lr_multi.rec_flow_loss > 0:
            rec_loss, reconstructed_points = reconstructionLoss(sample, pred_mask, pred_flow)
            rec_loss = rec_loss * config.lr_multi.rec_loss
        else:
            rec_loss = torch.tensor(0.0, device=device)

        if config.lr_multi.scene_flow_smoothness > 0:
            scene_flow_smooth_loss = flowSmoothLoss(sample, pred_mask, pred_flow)
            scene_flow_smooth_loss = scene_flow_smooth_loss * config.lr_multi.scene_flow_smoothness
            scene_flow_smooth_loss = scene_flow_smooth_loss * scene_flow_scheduler(step)
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
        optimizer.zero_grad()
        optimizer_mask.zero_grad()
        loss.backward()
        
        # Log gradients if needed
        if hasattr(pred_flow, 'grad') and pred_flow.grad is not None:
            tqdm.write(f"pred_flow.grad {pred_flow.grad.std().item()}")
        if hasattr(pred_mask, 'grad') and pred_mask.grad is not None:
            tqdm.write(f"pred_mask.grad {pred_mask.grad.std().item()}")
            
        optimizer.step()
        optimizer_mask.step()
        
        # Compute EPE (End Point Error)
        epe_list = []
        for i in range(len(point_cloud_firsts)):
            epe_list.append(torch.norm(pred_flow[i] - gt_flow[i], dim=1, p=2))  # Shape: [B, N]
        epe_mean = [torch.mean(epe) for epe in epe_list]
        epe_mean = torch.mean(torch.stack(epe_mean))
        tqdm.write(f"epe {epe_mean.item()}")
        writer.add_scalar("epe", epe_mean.item(), step)
        #calculate miou using pred_mask and sample["dynamic_instance_mask"]
        miou_list = []
        for i in range(len(point_cloud_firsts)):
            gt_mask = remap_instance_labels(sample["dynamic_instance_mask"][i])
            tqdm.write(f"gt_mask size {max(gt_mask)}")
            miou_list.append(
                calculate_miou(
                    pred_mask[i], 
                    F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(device=device)  # Shape: [K, N]
                    )
                )
        miou_mean = torch.mean(torch.stack(miou_list))
        tqdm.write(f"miou {miou_mean.item()}")
        writer.add_scalar("miou", miou_mean.item(), step)

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