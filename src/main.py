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
import gc
import time
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
from eval import evaluate_predictions, eval_model
from utils.config_utils import load_config_with_inheritance, save_config_and_code
from utils.dataloader_utils import create_dataloaders
from utils.visualization_utils import remap_instance_labels, color_mask
# from losses.ChamferDistanceLoss import ChamferDistanceLoss
# from losses.ReconstructionLoss import ReconstructionLoss
# from losses.PointSmoothLoss import PointSmoothLoss
# from losses.FlowSmoothLoss import FlowSmoothLoss
from visualize.open3d_func import visualize_vectors, update_vector_visualization
from Predictor import get_mask_predictor, get_scene_flow_predictor
from alter_scheduler import AlterScheduler
from config.config import correct_datatype
from model.eulerflow_raw_mlp import QueryDirection

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
    dataloader, infinite_loader, val_dataloader, batch_size, N = create_dataloaders(config)
    
    # Initialize models
    mask_predictor = get_mask_predictor(config.model.mask, N)
    scene_flow_predictor = get_scene_flow_predictor(config.model.flow, N)
    scene_flow_predictor.to(device)

    # Initialize optimizers
    optimizer_flow = torch.optim.AdamW(scene_flow_predictor.parameters(), lr=config.model.flow.lr)
    optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=config.model.mask.lr)
    
    alter_scheduler = AlterScheduler(config.alternate)
    # Initialize loss functions
    if config.lr_multi.rec_loss > 0:
        from losses.ReconstructionLoss import ReconstructionLoss
        from losses.ReconstructionLoss_optimized import ReconstructionLossOptimized
        reconstructionLoss = ReconstructionLossOptimized(device)
    else:
        reconstructionLoss = None
    if config.lr_multi.flow_loss > 0:
        from losses.ChamferDistanceLoss import ChamferDistanceLoss
        chamferLoss = ChamferDistanceLoss()
    else:
        chamferLoss = None
    if config.lr_multi.scene_flow_smoothness > 0:
        from losses.FlowSmoothLoss import FlowSmoothLoss
        flowSmoothLoss = FlowSmoothLoss(device, config.loss.scene_flow_smoothness)
    else:
        flowSmoothLoss = None
    if config.lr_multi.rec_flow_loss > 0:
        flowRecLoss = nn.MSELoss()
    else:
        flowRecLoss = None
    if config.lr_multi.point_smooth_loss > 0:
        from losses.PointSmoothLoss import PointSmoothLoss
        pointsmoothloss = PointSmoothLoss()
    else:
        pointsmoothloss = None
    if config.lr_multi.KDTree_loss > 0:
        from losses.KDTreeDistanceLoss import KDTreeDistanceLoss
        kdtree_loss = KDTreeDistanceLoss(max_distance=1.0, reduction="mean")
        kdtree_loss.to(device)

    if config.lr_multi.KNN_loss > 0:
        from losses.KNNDistanceLoss import TruncatedKNNDistanceLoss
        knn_loss = TruncatedKNNDistanceLoss(k=1, reduction="mean")

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

    # =====================
    # Checkpointing setup
    # =====================
    checkpoint_dir = os.path.join(config.log.dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume logic
    # Config options supported:
    # - config.checkpoint.resume (bool)
    # - config.checkpoint.resume_path (str)
    # - config.checkpoint.save_every_iters (int, default 1000)
    save_every_iters = 1000
    resume = False
    resume_path = None
    if hasattr(config, "checkpoint"):
        save_every_iters = getattr(config.checkpoint, "save_every_iters", 1000)
        resume = getattr(config.checkpoint, "resume", False)
        resume_path = getattr(config.checkpoint, "resume_path", None)

    if resume:
        candidate_path = resume_path if resume_path else os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(candidate_path):
            ckpt = torch.load(candidate_path, map_location=device)
            scene_flow_predictor.load_state_dict(ckpt["scene_flow_predictor"])  # required
            mask_predictor.load_state_dict(ckpt["mask_predictor"])            # required
            optimizer_flow.load_state_dict(ckpt.get("optimizer_flow", optimizer_flow.state_dict()))
            optimizer_mask.load_state_dict(ckpt.get("optimizer_mask", optimizer_mask.state_dict()))
            if "alter_scheduler" in ckpt:
                try:
                    alter_scheduler.load_state_dict(ckpt["alter_scheduler"])  # optional
                except Exception:
                    pass
            step = int(ckpt.get("step", 0))
            tqdm.write(f"Resumed from checkpoint: {candidate_path} (step={step})")
        else:
            tqdm.write(f"No checkpoint found at {candidate_path}, starting fresh.")

    def save_checkpoint(path_latest: str, step_value: int):
        state = {
            "step": step_value,
            "scene_flow_predictor": scene_flow_predictor.state_dict(),
            "mask_predictor": mask_predictor.state_dict(),
            "optimizer_flow": optimizer_flow.state_dict(),
            "optimizer_mask": optimizer_mask.state_dict(),
            "alter_scheduler": getattr(alter_scheduler, 'state_dict', lambda: {})(),
            "config": OmegaConf.to_container(config, resolve=True),
        }
        torch.save(state, path_latest)
        # Also keep a step-suffixed snapshot
        step_path = os.path.join(checkpoint_dir, f"step_{step_value}.pt")
        try:
            torch.save(state, step_path)
        except Exception:
            pass

    # Main training loop
    with tqdm(infinite_loader, desc="Training", total=config.training.max_iter-step) as infinite_loader:
        tqdm.write("Starting training...")
        for sample in infinite_loader:
            step += 1

            if len(sample["idx"]) == 0:
                continue
            
            if step > config.training.max_iter:
                tqdm.write("Reached maximum training iterations, stopping.")
                break
            train_flow = alter_scheduler.flow_train()
            train_mask = alter_scheduler.mask_train()
            if step < config.training.begin_train_mask:
                train_mask = False
                train_flow = True
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
            idxs=sample.get("idx")
            nextbatchs= [sample["self"][0].get_item(idx+1) for idx in idxs]
            point_cloud_nexts = [item["point_cloud_first"].to(device) for item in nextbatchs]
            pred_flow = []
            reverse_pred_flow = []
            longterm_pred_flow = {}
            if train_flow:                    
                for i in range(len(point_cloud_firsts)):
                    if config.model.flow.name == "EulerFlowMLP":
                        if sample["k"][i] ==1:
                            pred_flow.append(scene_flow_predictor(point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i],QueryDirection.FORWARD))  # Shape: [N, 3]
                            
                            reverse_pred_flow.append(scene_flow_predictor(point_cloud_nexts[i], sample["idx"][i]+1, sample["total_frames"][i],QueryDirection.REVERSE))  # Shape: [N, 3]
                        else:
                            pred_pc = point_cloud_firsts[i].clone()
                            for k in range(0, sample["k"][i]):
                                idx = sample["idx"][i]
                                pred_flow_temp = scene_flow_predictor(pred_pc, sample["idx"][i]+k, sample["total_frames"][i],QueryDirection.FORWARD)
                                pred_pc = pred_pc + pred_flow_temp
                                longterm_pred_flow[sample["idx"][i]+k+1] = pred_pc.clone()
                                if k == 0:
                                    pred_flow.append(pred_flow_temp)
                                    
                            pred_pc = point_cloud_nexts[i].clone()
                            for k in range(0, sample["k"][i]):
                                idx = sample["idx"][i]
                                pred_flow_temp = scene_flow_predictor(pred_pc, sample["idx"][i]-k+1, sample["total_frames"][i],QueryDirection.REVERSE)
                                pred_pc = pred_pc + pred_flow_temp
                                longterm_pred_flow[sample["idx"][i]-k] = pred_pc.clone()
                                if k == 0:
                                    reverse_pred_flow.append(pred_flow_temp)
                                    
                    else:
                        pred_flow.append(scene_flow_predictor(point_cloud_firsts[i]))  # Shape: [N, 3]
            else:
                with torch.no_grad():
                    for i in range(len(point_cloud_firsts)):
                        if config.model.flow.name == "EulerFlowMLP":
                            pred_flow.append(scene_flow_predictor(point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i],QueryDirection.FORWARD))
                        else:
                            pred_flow.append(scene_flow_predictor(point_cloud_firsts[i]))
            if train_mask:
                pred_mask = []
                for i in range(len(point_cloud_firsts)):
                    if config.model.mask.name in ["EulerMaskMLP", "EulerMaskMLPResidual"]:
                        mask = mask_predictor(point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i])
                        mask = mask.permute(1, 0)
                        
                        pred_mask.append(mask)
                    else:
                        pred_mask.append(mask_predictor(point_cloud_firsts[i]))
            else:
                with torch.no_grad():
                    pred_mask =[]
                    for i in range(len(point_cloud_firsts)):
                        if config.model.mask.name in ["EulerMaskMLP", "EulerMaskMLPResidual"]:
                            mask = mask_predictor(point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i])
                            mask = mask.permute(1, 0)
                            pred_mask.append(mask)
                        else:
                            pred_mask.append(mask_predictor(point_cloud_firsts[i]))
            # Compute losses
            if (config.lr_multi.rec_loss > 0 or config.lr_multi.rec_flow_loss > 0) and train_mask:
                pred_flow_detach = [flow.detach() for flow in pred_flow]
                rec_loss, reconstructed_points = reconstructionLoss(point_cloud_firsts, point_cloud_nexts, pred_mask, pred_flow_detach)
                rec_loss = rec_loss * config.lr_multi.rec_loss
            else:
                rec_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if config.lr_multi.scene_flow_smoothness > 0 and step > config.training.begin_train_smooth:
                scene_flow_smooth_loss = flowSmoothLoss(point_cloud_firsts, pred_mask, pred_flow)
                scene_flow_smooth_loss = scene_flow_smooth_loss * config.lr_multi.scene_flow_smoothness
            else:
                scene_flow_smooth_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if config.lr_multi.rec_flow_loss > 0 and train_mask:
                rec_flow_loss = 0
                for i in range(len(point_cloud_firsts)):
                    pred_second_point = point_cloud_firsts[i][:, :3] + pred_flow[i]
                    rec_flow_loss += flowRecLoss(pred_second_point, reconstructed_points[i])
                rec_flow_loss = rec_flow_loss * config.lr_multi.rec_flow_loss
            else:
                rec_flow_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if config.lr_multi.flow_loss > 0:
                flow_loss = 0
                for i in range(len(point_cloud_firsts)):
                    pred_second_points = point_cloud_firsts[i][:, :3] + pred_flow[i]
                    flow_loss += chamferLoss(pred_second_points.unsqueeze(0), point_cloud_nexts[i][:, :3].to(device).unsqueeze(0))
                    if len(reverse_pred_flow) > 0:
                        pred_first_point = point_cloud_nexts[i][:, :3] + reverse_pred_flow[i]
                        flow_loss += chamferLoss(pred_first_point.unsqueeze(0), point_cloud_firsts[i][:, :3].to(device).unsqueeze(0))
                        flow_loss = flow_loss / 2
                if len(longterm_pred_flow) > 0:
                    for idx in longterm_pred_flow:
                        pred_points = longterm_pred_flow[idx][:, :3]
                        real_points = sample["self"][i].get_item(idx)["point_cloud_first"][:, :3].to(device)
                        flow_loss += chamferLoss(pred_points.unsqueeze(0), real_points.to(device).unsqueeze(0))
                flow_loss = flow_loss * config.lr_multi.flow_loss
            else:
                flow_loss = torch.tensor(0.0, device=device)

            if config.lr_multi.point_smooth_loss > 0:
                point_smooth_loss = pointsmoothloss(point_cloud_firsts, pred_mask)
                point_smooth_loss = point_smooth_loss * config.lr_multi.point_smooth_loss
            else:
                point_smooth_loss = torch.tensor(0.0, device=device, requires_grad=True)

            if config.lr_multi.eular_flow_loss > 0 and config.model.flow.name == "EulerFlowMLP" and train_flow:
                eular_flow_loss = 0
                for i in range(len(point_cloud_firsts)):
                    point_cloud_first_forward = point_cloud_firsts[i][:, :3] + pred_flow[i]
                    forward_reverse = scene_flow_predictor(point_cloud_first_forward, sample["idx"][i]+1, sample["total_frames"][i], QueryDirection.REVERSE)
                    l2_error = torch.norm(pred_flow[i] + forward_reverse, dim=1)
                    eular_flow_loss += torch.sigmoid(l2_error).mean()
                    if len(reverse_pred_flow) == 0:
                        continue
                    point_cloud_next_reverse = point_cloud_nexts[i][:, :3] + reverse_pred_flow[i]
                    reverse_forward = scene_flow_predictor(point_cloud_next_reverse, sample["idx"][i], sample["total_frames"][i], QueryDirection.FORWARD)
                    l2_error = torch.norm(reverse_pred_flow[i] + reverse_forward, dim=1)
                    eular_flow_loss += torch.sigmoid(l2_error).mean()
                eular_flow_loss = eular_flow_loss * config.lr_multi.eular_flow_loss
            else:
                eular_flow_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if config.lr_multi.eular_mask_loss > 0 and config.model.mask.name in ["EulerMaskMLP", "EulerMaskMLPResidual"] and train_mask:
                eular_mask_loss = 0
                for i in range(len(point_cloud_firsts)):
                    point_cloud_first_forward = point_cloud_firsts[i][:, :3] + pred_flow[i]
                    forward_mask = mask_predictor(point_cloud_first_forward, sample["idx"][i]+1, sample["total_frames"][i])
                    #计算forward_mask和pred_mask的相对熵
                    forward_mask = F.log_softmax(forward_mask, dim=1)
                    target_mask = F.log_softmax(pred_mask[i].clone().permute(1, 0), dim=1)
                    relative_entropy = torch.nn.functional.kl_div(forward_mask, target_mask, reduction='batchmean',log_target=True)
                    eular_mask_loss += relative_entropy.mean()
                eular_mask_loss = eular_mask_loss * config.lr_multi.eular_mask_loss
            else:
                eular_mask_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if config.lr_multi.KDTree_loss > 0 and train_flow:
                kdtree_dist_loss = 0
                for i in range(len(point_cloud_firsts)):
                    pred_second_point = point_cloud_firsts[i][:, :3] + pred_flow[i]
                    kdtree_dist_loss += kdtree_loss(pred_second_point, sample["idx"][i]+1, point_cloud_nexts[i][:, :3].to(device))
                if len(longterm_pred_flow) > 0:
                    for idx in longterm_pred_flow:
                        pred_points = longterm_pred_flow[idx][:, :3]
                        data_object = sample["self"][i].get_item(idx)
                        real_points = data_object["point_cloud_first"][:, :3].to(device)
                        kdtree_dist_loss += kdtree_loss(pred_points, data_object["idx"], real_points)
                kdtree_dist_loss = kdtree_dist_loss * config.lr_multi.KDTree_loss
            else:
                kdtree_dist_loss = torch.tensor(0.0, device=device)

            if config.lr_multi.KNN_loss > 0 and train_flow:
                knn_dist_loss = 0
                for i in range(len(point_cloud_firsts)):
                    pred_second_point = point_cloud_firsts[i][:, :3] + pred_flow[i]
                    knn_dist_loss += knn_loss(point_cloud_nexts[i][:, :3].to(device),pred_second_point)
                if len(longterm_pred_flow) > 0:
                    for idx in longterm_pred_flow:
                        pred_points = longterm_pred_flow[idx][:, :3]
                        real_points = sample["self"][i].get_item(idx)["point_cloud_first"][:, :3].to(device)
                        knn_dist_loss += knn_loss(real_points, pred_points)
                knn_dist_loss = knn_dist_loss * config.lr_multi.KNN_loss
            else:
                knn_dist_loss = torch.tensor(0.0, device=device)
            if config.lr_multi.l1_regularization > 0 and train_flow:
                
                l1_regularization_loss = 0
                for flow in pred_flow:
                    thres = 0.25
                    dist = torch.norm(flow, dim=1)
                    #for dist > 0.005, set to 0
                    dist = torch.where(dist > thres, torch.zeros_like(dist), dist)
                    l1_regularization_loss += dist.mean()   
                for flow in reverse_pred_flow:
                    dist = torch.norm(flow, dim=1)
                    dist = torch.where(dist > thres, torch.zeros_like(dist), dist)
                    l1_regularization_loss += dist.mean()
                for idx in longterm_pred_flow:
                    dist = torch.norm(longterm_pred_flow[idx], dim=1)
                    dist = torch.where(dist > thres, torch.zeros_like(dist), dist)
                    l1_regularization_loss += dist.mean()
                l1_regularization_loss = l1_regularization_loss * config.lr_multi.l1_regularization
            else:
                l1_regularization_loss = torch.tensor(0.0, device=device)
            # Combine losses
            loss = rec_loss + flow_loss + scene_flow_smooth_loss + rec_flow_loss + point_smooth_loss + eular_flow_loss + kdtree_dist_loss + knn_dist_loss + l1_regularization_loss + eular_mask_loss

                
            # Log to tensorboard
            if step % 7 == 0:
                writer.add_scalars("losses", {
                    "rec_loss": rec_loss.item(),
                    "flow_loss": flow_loss.item(),
                    "scene_flow_smooth_loss": scene_flow_smooth_loss.item(),
                    "rec_flow_loss": rec_flow_loss.item(),
                    "point_smooth_loss": point_smooth_loss.item(),
                    "eular_flow_loss": eular_flow_loss.item(),
                    "kdtree_dist_loss": kdtree_dist_loss.item(),
                    "knn_dist_loss": knn_dist_loss.item(),
                    "l1_regularization_loss": l1_regularization_loss.item(),
                    "eular_mask_loss": eular_mask_loss.item(),
                    "total_loss": loss.item(),
                }, step)
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
                "eular_flow_loss": eular_flow_loss,
                "kdtree_dist_loss": kdtree_dist_loss,
                "knn_dist_loss": knn_dist_loss,
                "l1_regularization_loss": l1_regularization_loss,
                "eular_mask_loss": eular_mask_loss,
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
            try:
                total_loss.backward()  # 实际参数更新只用总loss的梯度
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                print("Error computing gradient for total loss")
                print(f"Total loss value: {total_loss.item()}")
                import traceback
                traceback.print_exc()
                continue
            if not train_flow:
                optimizer_flow.zero_grad()
            if not train_mask:
                optimizer_mask.zero_grad()
            if hasattr(config.training, "flow_model_grad_clip"):
                torch.nn.utils.clip_grad_norm_(scene_flow_predictor.parameters(), config.training.flow_model_grad_clip)
            if hasattr(config.training, "mask_model_grad_clip"):
                torch.nn.utils.clip_grad_norm_(mask_predictor.parameters(), config.training.mask_model_grad_clip)
            if train_flow:
                optimizer_flow.step()
            if train_mask:
                optimizer_mask.step()


            alter_scheduler.step()
            # Iteration-based checkpoint save
            if save_every_iters > 0 and step % save_every_iters == 0:
                latest_path = os.path.join(checkpoint_dir, "latest.pt")
                try:
                    save_checkpoint(latest_path, step)
                    tqdm.write(f"Saved checkpoint at step {step} -> {latest_path}")
                except Exception as e:
                    tqdm.write(f"Failed to save checkpoint: {e}")
            # Evaluate predictions
            if step % config.training.eval_loop == 1:
                epe, miou, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean = eval_model(scene_flow_predictor, mask_predictor, dataloader, config, device, writer, step)
                writer.add_scalar(
                    "val_epe",
                    epe.mean().item(),
                    step
                )
                writer.add_scalar(
                    "val_miou",
                    miou.item(),
                    step
                )
                if bg_epe is not None:
                    writer.add_scalar(
                        "val_bg_epe",
                        bg_epe.mean().item(),
                        step
                    )
                if fg_static_epe is not None:
                    writer.add_scalar(
                        "val_fg_static_epe",
                        fg_static_epe.mean().item(),
                        step
                    )
                if fg_dynamic_epe is not None:
                    writer.add_scalar(
                        "val_fg_dynamic_epe",
                        fg_dynamic_epe.mean().item(),
                        step
                    )
                if threeway_mean is not None:
                    writer.add_scalar(
                        "val_threeway_mean",
                        threeway_mean.mean().item(),
                        step
                    )
            #clear the cache
            torch.cuda.empty_cache()
            gc.collect()
            if config.vis.log_histogram:
                #log the first flow prediction
                #if requires grad, detach it
                if pred_flow[0].requires_grad:
                    flow_log = pred_flow[0].detach()
                else:
                    flow_log = pred_flow[0]
                if pred_mask[0].requires_grad:
                    mask_log = pred_mask[0].detach()
                else:
                    mask_log = pred_mask[0]
                flow_log = flow_log.cpu().numpy()
                mask_log = mask_log.cpu().numpy()
                processed_mask = mask_log.copy()
                processed_mask = processed_mask / pow(processed_mask.std(),0.5)
                #numpy softmax
                min_value = np.min(processed_mask)*10
                softmaxed_mask = np.exp(processed_mask*10 - min_value) / np.sum(np.exp(processed_mask*10 - min_value), axis=0)
                writer.add_histogram(f"prediction_flow", flow_log, step)
                writer.add_histogram(f"prediction_mask", mask_log, step)
                writer.add_histogram(f"prediction_mask_processed", processed_mask, step)
                writer.add_histogram(f"prediction_mask_softmaxed", softmaxed_mask, step)
                writer.add_scalar(f"prediction_flow_mean", np.mean(np.linalg.norm(flow_log, axis=1)), step)
                writer.add_scalar(f"prediction_mask_mean", np.mean(np.linalg.norm(mask_log, axis=1)), step)
                writer.add_scalar(f"prediction_flow_std", np.std(np.linalg.norm(flow_log, axis=1)), step)
                writer.add_scalar(f"prediction_mask_processed_mean", np.mean(np.linalg.norm(processed_mask, axis=1)), step)
                writer.add_scalar(f"prediction_mask_softmaxed_mean", np.mean(np.linalg.norm(softmaxed_mask, axis=1)), step)
                writer.add_scalar(f"prediction_mask_std", np.std(np.linalg.norm(mask_log, axis=1)), step)
                writer.add_scalar(f"prediction_mask_processed_std", np.std(np.linalg.norm(processed_mask, axis=1)), step)
                writer.add_scalar(f"prediction_mask_softmaxed_std", np.std(np.linalg.norm(softmaxed_mask, axis=1)), step)
            # Visualization
            if config.vis.show_window and point_cloud_firsts[0].shape[0] > 0:
                batch_idx = 0
                
                # Get data for visualization
                point_cloud_first = point_cloud_firsts[batch_idx].cpu().numpy()
                point_cloud_second = point_cloud_nexts[batch_idx].cpu().numpy()
                current_pred_flow = pred_flow[batch_idx].cpu().detach().numpy()
                current_pred_mask = pred_mask[batch_idx].cpu().detach()
                gt_flow = sample["flow"][batch_idx].cpu().detach().numpy()
                print("mean length of pred_flow",np.mean(np.linalg.norm(current_pred_flow)))
                # PCA for coloring
                # current_pred_mask = current_pred_mask.permute(1, 0)  # Change to [N, K] for PCA
                # pred_color = pca(current_pred_mask)

                pred_color = color_mask(current_pred_mask)
                gt_mask = sample["dynamic_instance_mask"][0]
                gt_mask = remap_instance_labels(gt_mask)
                gt_color = color_mask(F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(torch.float32))
                # Update point clouds
                pcd.points = o3d.utility.Vector3dVector(point_cloud_first)
                pcd.colors = o3d.utility.Vector3dVector(gt_color.numpy())
                
                gt_pcd.points = o3d.utility.Vector3dVector(point_cloud_second)
                gt_pcd.paint_uniform_color([0, 1, 0])
                
                    
                if first_iteration:
                    # vis.add_geometry(pcd)
                    # vis.add_geometry(gt_pcd)
                    vis, lineset = visualize_vectors(
                        point_cloud_first,
                        current_pred_flow,
                        vis=vis,
                        color=pred_color.numpy(),
                    )
                    vis, lineset_gt = visualize_vectors(
                        point_cloud_first,
                        gt_flow,
                        vis=vis,
                        color=gt_color.numpy(),
                    )
                    first_iteration = False
                else:
                    # vis.update_geometry(pcd)
                    # vis.update_geometry(gt_pcd)
                    lineset = update_vector_visualization(
                        lineset,
                        point_cloud_first,
                        current_pred_flow,
                        color=pred_color.numpy(),
                    )
                    lineset_gt = update_vector_visualization(
                        lineset_gt,
                        point_cloud_first,
                        gt_flow,
                        color=gt_color.numpy(),
                    )
                    vis.update_geometry(lineset)
                    vis.update_geometry(lineset_gt)
                    
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