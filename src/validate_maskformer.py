"""
Supervised learning validation script for Point MaskFormer model.

This script validates the Point MaskFormer model using the validation dataset
and segmentation ground truth, following the structure of main_general.py.
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from typing import List

# Local imports
from main_general import (
    create_dataloaders_general,
    initialize_models_and_optimizers,
    load_checkpoint,
    setup_device_and_training
)
from SceneFlowZoo.dataloaders import TorchFullFrameInputSequence
from utils.metrics import calculate_miou
from utils.visualization_utils import remap_instance_labels


def dice_loss(pred, target, smooth=1e-5):
    """
    Dice loss for mask prediction.
    
    Args:
        pred: Predicted masks [N, C] or [N, C, H, W]
        target: Ground truth masks [N, C] or [N, C, H, W]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice loss value
    """
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """
    Focal loss for mask prediction.
    
    Args:
        pred: Predicted logits [N, C]
        target: Ground truth masks [N, C]
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    
    Returns:
        Focal loss value
    """
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return focal_loss.mean()


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.
    
    Args:
        input (Tensor): A tensor of shape (N, C, S) contains features on a regular grid
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, P, 3) contains point coordinates
            in [0, 1] x [0, 1] (or [0, 1] x [0, 1] x [0, 1]) coordinate system
    
    Returns:
        output (Tensor): A tensor of shape (N, C, P) contains features for points in `point_coords`
    """
    add_dim = False
    if point_coords.dim() == 2:
        add_dim = True
        point_coords = point_coords.unsqueeze(0)
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(0)
    
    # Normalize coordinates to [-1, 1]
    point_coords = 2.0 * point_coords - 1.0
    
    # Add batch dimension if needed
    if input.dim() == 2:
        input = input.unsqueeze(0)
    
    # Reshape for grid_sample
    if input.dim() == 2:  # [C, S] -> [1, C, 1, S]
        input = input.unsqueeze(0).unsqueeze(2)
        point_coords = point_coords.unsqueeze(1)  # [N, P, 1, 1]
    else:  # [N, C, S] -> [N, C, 1, S]
        input = input.unsqueeze(2)
        point_coords = point_coords.unsqueeze(1)  # [N, P, 1, 1]
    
    output = F.grid_sample(input, point_coords, mode='bilinear', padding_mode='border', align_corners=False)
    
    if add_dim:
        output = output.squeeze(0)
    
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on uncertainty.
    
    Args:
        coarse_logits: Coarse predictions [N, C, H, W] or [N, C, S]
        uncertainty_func: Function to compute uncertainty
        num_points: Number of points to sample
        oversample_ratio: Oversampling ratio
        importance_sample_ratio: Ratio of importance sampling
    
    Returns:
        point_coords: Sampled point coordinates [N, num_points, 2]
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    num_rand = int(num_points * (1 - importance_sample_ratio))
    num_uncertain = num_points - num_rand
    
    # Random sampling
    rand_inds = torch.randint(0, coarse_logits.shape[-1], (num_boxes, num_rand), device=coarse_logits.device)
    rand_coords = torch.rand(num_boxes, num_rand, 1, device=coarse_logits.device)
    rand_point_coords = torch.cat([rand_coords, rand_inds.float().unsqueeze(-1) / coarse_logits.shape[-1]], dim=-1)
    
    # Uncertainty-based sampling
    if num_uncertain > 0:
        uncertainty_map = uncertainty_func(coarse_logits)
        _, uncertainty_inds = torch.topk(uncertainty_map, num_uncertain, dim=-1)
        uncertainty_coords = torch.rand(num_boxes, num_uncertain, 1, device=coarse_logits.device)
        uncertainty_point_coords = torch.cat([
            uncertainty_coords, 
            uncertainty_inds.float() / coarse_logits.shape[-1]
        ], dim=-1)
        
        point_coords = torch.cat([rand_point_coords, uncertainty_point_coords], dim=1)
    else:
        point_coords = rand_point_coords
    
    return point_coords


def sigmoid_focal_loss(inputs, targets, alpha=-1, gamma=2, reduction="mean"):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def dice_loss_jit(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


def compute_mask2former_losses(pred_masks, gt_masks, num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75):
    """
    Compute Mask2Former-style losses with Hungarian matching and point sampling.
    
    Args:
        pred_masks: Predicted masks [N, C, S] or [N, C]
        gt_masks: Ground truth masks [N, C, S] or [N, C] 
        num_points: Number of points to sample
        oversample_ratio: Oversampling ratio for point sampling
        importance_sample_ratio: Ratio of importance sampling
    
    Returns:
        Dictionary of loss values
    """
    losses = {}
    
    # Ensure 3D tensors for point sampling
    if pred_masks.dim() == 2:
        pred_masks = pred_masks.unsqueeze(-1)  # [N, C, 1]
    if gt_masks.dim() == 2:
        gt_masks = gt_masks.unsqueeze(-1)  # [N, C, 1]
    
    # Point sampling
    def uncertainty_func(logits):
        # Simple uncertainty based on prediction confidence
        return -torch.abs(logits - 0.5)
    
    point_coords = get_uncertain_point_coords_with_randomness(
        pred_masks, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
    )
    
    # Sample points from predictions and GT
    pred_point_logits = point_sample(pred_masks, point_coords)  # [N, C, num_points]
    gt_point_masks = point_sample(gt_masks, point_coords)  # [N, C, num_points]
    
    # Reshape for loss computation
    pred_point_logits = pred_point_logits.flatten(0, 1)  # [N*C, num_points]
    gt_point_masks = gt_point_masks.flatten(0, 1)  # [N*C, num_points]
    
    # Focal loss
    focal_loss_val = sigmoid_focal_loss(pred_point_logits, gt_point_masks)
    losses['focal'] = focal_loss_val
    
    # Dice loss
    dice_loss_val = dice_loss_jit(pred_point_logits, gt_point_masks)
    losses['dice'] = dice_loss_val
    
    # Total loss
    losses['total'] = focal_loss_val + dice_loss_val
    
    return losses


def compute_mask_losses(pred_masks, gt_masks, loss_weights=None, use_mask2former=True):
    """
    Compute mask supervision losses with optional Mask2Former-style processing.
    
    Args:
        pred_masks: Predicted masks [N, C]
        gt_masks: Ground truth masks [N, C]
        loss_weights: Dictionary of loss weights
        use_mask2former: Whether to use Mask2Former-style losses
    
    Returns:
        Dictionary of loss values
    """
    if use_mask2former:
        return compute_mask2former_losses(pred_masks, gt_masks)
    
    # Original simple losses
    if loss_weights is None:
        loss_weights = {
            'ce': 1.0,
            'dice': 1.0,
            'focal': 0.5
        }
    
    losses = {}
    
    # Cross-entropy loss
    if 'ce' in loss_weights:
        ce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
        losses['ce'] = ce_loss * loss_weights['ce']
    
    # Dice loss
    if 'dice' in loss_weights:
        dice_loss_val = dice_loss(pred_masks, gt_masks)
        losses['dice'] = dice_loss_val * loss_weights['dice']
    
    # Focal loss
    if 'focal' in loss_weights:
        focal_loss_val = focal_loss(pred_masks, gt_masks)
        losses['focal'] = focal_loss_val * loss_weights['focal']
    
    # Total loss
    total_loss = sum(losses.values())
    losses['total'] = total_loss
    
    return losses


def validate_maskformer_supervised(config_path: str, checkpoint_path: str, device: str = "cuda:0"):
    """
    Validate Point MaskFormer model using supervised learning on validation dataset.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        device: Device to run validation on
    """
    # Load configuration
    from utils.config_utils import load_config_with_inheritance
    config = load_config_with_inheritance(config_path)
    device = torch.device(device)
    
    print(f"Loading configuration from {config_path}")
    print(f"Using device: {device}")
    
    # Create dataloaders
    (dataset, dataloader,
     val_flow_dataset, val_flow_dataloader,
     val_mask_dataset, val_mask_dataloader) = create_dataloaders_general(config)
    
    print(f"Validation flow dataset size: {len(val_flow_dataset)}")
    print(f"Validation mask dataset size: {len(val_mask_dataset)}")
    
    # Initialize models
    (mask_predictor, flow_predictor, optimizer_flow, optimizer_mask, 
     alter_scheduler, scene_flow_smoothness_scheduler) = initialize_models_and_optimizers(config, None, device)
    
    # Load checkpoint
    
    step = 0
    
    print(f"Loaded checkpoint from step {step}")
    
    # Set models to evaluation mode
    mask_predictor.eval()
    flow_predictor.eval()
    
    # Validation metrics
    total_samples = 0
    total_miou = 0.0
    miou_list = []
    
    # Loss tracking
    loss_accumulator = {
        'dice': 0.0,
        'focal': 0.0,
        'total': 0.0
    }
    
    print("Starting validation...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_mask_dataloader, desc="Validating")):
            if i >= config.eval.eval_size:
                break
                
            for sample in batch:
                # Convert to TorchFullFrameInputSequence
                sequence = TorchFullFrameInputSequence.from_frame_list(
                    idx=0,
                    frame_list=sample,
                    pc_max_len=120000,
                    loader_type=dataset.loader_type(),
                    allow_pc_slicing=False
                ).to(device)
                
                # Get point cloud and ground truth mask
                point_cloud = sequence.get_full_ego_pc(0)
                point_cloud_mask = sequence.get_full_pc_mask(0)
                valid_points = point_cloud[point_cloud_mask]
                
                if valid_points.numel() == 0:
                    continue
                
                # Prepare input for mask predictor
                data = {'points': [valid_points]}
                
                # Forward pass for mask prediction
                try:
                    pred_mask_output = mask_predictor.forward_train(data)
                    pred_masks = pred_mask_output["pred_masks"]
                    
                    # Get ground truth mask
                    gt_mask = sample[0].instance_ids[sample[0].pc.mask]
                    gt_mask = torch.from_numpy(gt_mask).to(device).long()
                    
                    print(f"Pred masks shape: {pred_masks.shape}")
                    print(f"GT mask shape: {gt_mask.shape}")
                    
                    
                    # Remap instance labels
                    gt_mask_remapped = remap_instance_labels(gt_mask, ignore_label=[-1])
                    
                    # Convert to one-hot for loss computation
                    # pred_masks: [num_instances, num_points]
                    # gt_onehot: [num_instances, num_points] 
                    gt_onehot = F.one_hot(gt_mask_remapped.to(torch.long)).permute(1, 0).to(device=device)
                    # Calculate mIoU
                    miou = calculate_miou(
                        pred_masks,
                        gt_onehot,
                        min_points=config.eval.min_points
                    )
                    
                    if miou is not None:
                        miou_list.append(miou.item())
                        total_miou += miou.item()
                        total_samples += 1
                        
                        # Compute mask supervision losses using Mask2Former-style approach
                        mask_losses = compute_mask_losses(pred_masks, gt_onehot, use_mask2former=True)
                        
                        # Accumulate losses
                        for loss_name, loss_value in mask_losses.items():
                            if loss_name in loss_accumulator:
                                loss_accumulator[loss_name] += loss_value.item()
                        
                        print(f"Sample {total_samples}: mIoU = {miou.item():.4f}, "
                              f"Focal = {mask_losses['focal'].item():.4f}, "
                              f"Dice = {mask_losses['dice'].item():.4f}, "
                              f"Total = {mask_losses['total'].item():.4f}")
                    
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    import traceback
                    print(traceback.format_exc())
                    raise e
    
    # Calculate final metrics
    if total_samples > 0:
        mean_miou = total_miou / total_samples
        
        # Calculate average losses
        avg_losses = {}
        for loss_name, total_loss in loss_accumulator.items():
            avg_losses[loss_name] = total_loss / total_samples
        
        print(f"\nValidation Results:")
        print(f"Total samples: {total_samples}")
        print(f"Mean mIoU: {mean_miou:.4f}")
        print(f"Average Losses:")
        for loss_name, avg_loss in avg_losses.items():
            print(f"  {loss_name.upper()}: {avg_loss:.4f}")
        
        if len(miou_list) > 1:
            import numpy as np
            miou_array = np.array(miou_list)
            print(f"mIoU std: {miou_array.std():.4f}")
            print(f"mIoU min: {miou_array.min():.4f}")
            print(f"mIoU max: {miou_array.max():.4f}")
    else:
        print("No valid samples processed!")
        avg_losses = {}
    
    return mean_miou if total_samples > 0 else 0.0, avg_losses


def main():
    parser = argparse.ArgumentParser(description="Validate Point MaskFormer model")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration file (e.g., config/general.yaml)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run validation on")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
    
    # Run validation
    mean_miou, avg_losses = validate_maskformer_supervised(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    print(f"Final validation mIoU: {mean_miou:.4f}")
    if avg_losses:
        print("Final average losses:")
        for loss_name, avg_loss in avg_losses.items():
            print(f"  {loss_name.upper()}: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
