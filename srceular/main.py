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
from eval import evaluate_predictions, eval_dict
from dataset.av2_dataset import AV2PerSceneDataset
from dataset.movi_per_scene_dataset import MOVIPerSceneDataset
from utils.config_utils import load_config_with_inheritance, save_config_and_code
from utils.dataloader_utils import create_dataloaders
from utils.visualization_utils import remap_instance_labels, create_label_colormap, color_mask
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
from alter_scheduler import AlterScheduler
from config.config import correct_datatype


def main(config, writer):
    """
    Main training function.
    
    Args:
        config: Configuration object containing all training parameters
        writer: TensorBoard SummaryWriter for logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)
    # Create dataloaders
    dataloader, infinite_loader, batch_size, N = create_dataloaders(config)
    
    # Initialize models
    mask_predictor = get_mask_predictor(config.model.mask, N)
    scene_flow_predictor = get_scene_flow_predictor(config.model.flow, N)
    scene_flow_predictor.to(device)

    # Initialize optimizers
    optimizer_flow = torch.optim.AdamW(scene_flow_predictor.parameters(), lr=config.model.flow.lr)
    optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=config.model.mask.lr)
    
    alter_scheduler = AlterScheduler(config.alternate)
    # Initialize loss functions
    reconstructionLoss = ReconstructionLoss(device)
    chamferLoss = ChamferDistanceLoss()
    flowSmoothLoss = FlowSmoothLoss(device, config.loss.scene_flow_smoothness)
    pointsmoothloss = PointSmoothLoss()
    flowRecLoss = nn.MSELoss()

    # Initialize visualization if enabled
    if config.vis.show_window:
        
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        gt_pcd = o3d.geometry.PointCloud()
        reconstructed_pcd = o3d.geometry.PointCloud()
    
    first_iteration = True
    step = 0

    # Main training loop
    with tqdm(infinite_loader, desc="Training", total=config.training.max_iter) as infinite_loader:
        tqdm.write("Starting training...")
        for sample in infinite_loader:


            
            step += 1
            if step > config.training.max_iter:
                tqdm.write("Reached maximum training iterations, stopping.")
                break
            train_flow = alter_scheduler.flow_train()
            train_mask = alter_scheduler.mask_train()
            if train_flow:
                scene_flow_predictor.train()
            else:
                scene_flow_predictor.eval()
            if train_mask:
                mask_predictor.train()
            else:
                mask_predictor.eval()
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
            # tqdm.write(f"rec_loss: {rec_loss.item()}")
            # tqdm.write(f"flow_loss: {flow_loss.item()}")
            # tqdm.write(f"scene_flow_smooth_loss: {scene_flow_smooth_loss.item()}")
            # tqdm.write(f"rec_flow_loss: {rec_flow_loss.item()}")
            # tqdm.write(f"point_smooth_loss: {point_smooth_loss.item()}")
            # tqdm.write(f"iteration: {step}")

            # Log to tensorboard
            writer.add_scalars("losses", {
                "rec_loss": rec_loss.item(),
                "flow_loss": flow_loss.item(),
                "scene_flow_smooth_loss": scene_flow_smooth_loss.item(),
                "rec_flow_loss": rec_flow_loss.item(),
                "point_smooth_loss": point_smooth_loss.item(),
                "total_loss": loss.item(),
            }, step)
            writer.add_histogram("pred_mask",
                torch.stack(pred_mask).cpu().detach().numpy(), step)
            def compute_individual_gradients(loss_dict, model, retain_graph=False):
                grad_contributions = {}
                
                for name, loss in loss_dict.items():
                    # 确保loss是计算图的一部分
                    if not loss.requires_grad:
                        loss = loss.clone().requires_grad_(True)
                        
                    model.zero_grad()
                    
                    # 添加梯度检查
                    try:
                        loss.backward(retain_graph=retain_graph)
                    except RuntimeError as e:
                        print(f"Error computing gradient for {name}: {str(e)}")
                        print(f"Loss value: {loss.item()}")
                        continue
                        
                    grad_norms = {}
                    for param_name, param in model.named_parameters():
                        if param.grad is not None:
                            grad_norms[param_name] = torch.norm(param.grad).item()
                            param.grad = None  # 立即清除梯度
                    
                    grad_contributions[name] = grad_norms
                
                return grad_contributions
            
            loss_dict = {
                "rec_loss": rec_loss,
                "flow_loss": flow_loss,
                "scene_flow_smooth_loss": scene_flow_smooth_loss,
                "rec_flow_loss": rec_flow_loss,
                "point_smooth_loss": point_smooth_loss,
            }

            # --- 步骤1：单独计算并记录每个loss的梯度 ---
            if config.vis.debug_grad:
                with torch.no_grad():  # 避免影响实际梯度计算
                    flow_grads = compute_individual_gradients(
                        loss_dict, scene_flow_predictor, retain_graph=True
                    )
                    mask_grads = compute_individual_gradients(
                        loss_dict, mask_predictor, retain_graph=True
                    )

                # 记录到TensorBoard
                for loss_name in loss_dict:
                    if loss_dict[loss_name].item() == 0:
                        continue
                    writer.add_scalar(
                        f"grad_norm/flow_{loss_name}", 
                        sum(flow_grads[loss_name].values()),  # 所有层梯度范数和
                        step
                    )
                    writer.add_scalar(
                        f"grad_norm/mask_{loss_name}", 
                        sum(mask_grads[loss_name].values()), 
                        step
                    )

            # --- 步骤2：正常训练（统一计算总loss并更新） ---
            total_loss = sum(loss_dict.values())
            optimizer_flow.zero_grad()
            optimizer_mask.zero_grad()
            total_loss.backward()  # 实际参数更新只用总loss的梯度
            if not train_flow:
                optimizer_flow.zero_grad()
            if not train_mask:
                optimizer_mask.zero_grad()
            optimizer_flow.step()
            optimizer_mask.step()


            alter_scheduler.step()
            # Evaluate predictions
            global eval_dict
            epe, miou = evaluate_predictions(
                pred_flow, 
                gt_flow, 
                pred_mask, 
                sample["dynamic_instance_mask"], 
                device, 
                writer, 
                step,
                timestamp=timestamp,
                argoverse2=config.dataset.name=="AV2",
                background_static_mask=sample['background_static_mask'],
                foreground_static_mask=sample['foreground_static_mask'],
                foreground_dynamic_mask=sample['foreground_dynamic_mask']
            )
            postfix = {
                "EPE": f"{epe.mean().item():.4f}",
                "mIoU": f"{miou.item():.4f}",
                "Loss": f"{loss.item():.4f}",
                "t_flow": train_flow,
                "t_mask": train_mask,
            }
            infinite_loader.set_postfix(postfix)
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
            pass #end loop 

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

    # print_config(config_obj)
    
    cli_opts = OmegaConf.from_cli()
    
    # Merge configs
    config = OmegaConf.merge(config_obj, cli_opts)
    config = correct_datatype(config)
    print_config(config)

    # Setup logging directory
    if config.log.dir == "":
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.log.dir = f"../outputs/exp/{time_str}"
    
    writer = SummaryWriter(log_dir=config.log.dir)
    
    # Save config and code
    save_config_and_code(config, config.log.dir)

    # Start training
    main(config, writer)