"""
Model initialization and setup utilities.

This module contains model-related utility functions that were
previously in main.py, organized for better modularity.
"""

import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm

from Predictor import get_mask_predictor, get_scene_flow_predictor
from alter_scheduler import AlterScheduler, SceneFlowSmoothnessScheduler, MaskLRScheduler


def setup_device_and_training():
    """Setup device and training configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)
    return device


from torch.optim.lr_scheduler import LambdaLR


def lr_curve_mask(it, config):  # write a closure function for the lambda scheduler
    return max(
        config.model.mask.lr_decay ** (int(it * config.dataloader.batchsize / config.model.mask.lr_decay_step)),
        config.model.mask.lr_clip / config.model.mask.lr,
    )


def initialize_models_and_optimizers(config, N, device):
    """Initialize models and optimizers.

    Args:
        config: Configuration object
        N: Number of points
        device: Training device

    Returns:
        tuple: (mask_predictor, flow_predictor, optimizer_flow, optimizer_mask, schedulers)
    """
    # Initialize models
    mask_predictor = get_mask_predictor(config.model.mask, N)
    flow_predictor = get_scene_flow_predictor(config.model.flow, N)
    flow_predictor.to(device)

    # Initialize optimizers
    optimizer_flow = torch.optim.Adam(flow_predictor.parameters(), lr=config.model.flow.lr)
    optimizer_mask = torch.optim.Adam(
        mask_predictor.parameters(), lr=config.model.mask.lr, weight_decay=config.model.mask.weight_decay
    )

    # Initialize schedulers
    alter_scheduler = AlterScheduler(config.alternate)
    scene_flow_smoothness_scheduler = SceneFlowSmoothnessScheduler(config.lr_multi.scene_flow_smoothness_scheduler)

    # Initialize mask scheduler if configured
    mask_scheduler = None
    mask_scheduler = LambdaLR(optimizer_mask, lr_lambda=lambda it, config=config: lr_curve_mask(it, config))

    return (
        mask_predictor,
        flow_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
        scene_flow_smoothness_scheduler,
        mask_scheduler,
    )


def initialize_loss_functions(config, device):
    """Initialize all loss functions based on configuration.

    Args:
        config: Configuration object
        device: Training device

    Returns:
        dict: Dictionary containing all loss functions
    """
    loss_functions = {}

    # Reconstruction Loss
    if config.lr_multi.rec_loss > 0:
        from losses.ReconstructionLoss import ReconstructionLoss

        loss_functions["reconstruction"] = ReconstructionLoss(config,device)
    else:
        loss_functions["reconstruction"] = None

    # Chamfer Distance Loss
    if config.lr_multi.flow_loss > 0:
        from losses.ChamferDistanceLoss import ChamferDistanceLoss

        loss_functions["chamfer"] = ChamferDistanceLoss()
    else:
        loss_functions["chamfer"] = None

    # Flow Smooth Loss
    if config.lr_multi.scene_flow_smoothness > 0:
        from losses.FlowSmoothLoss import FlowSmoothLoss

        loss_functions["flow_smooth"] = FlowSmoothLoss(device, config.loss.scene_flow_smoothness)
    else:
        loss_functions["flow_smooth"] = None

    # Flow Reconstruction Loss
    if config.lr_multi.rec_flow_loss > 0:
        loss_functions["flow_rec"] = torch.nn.MSELoss()
    else:
        loss_functions["flow_rec"] = None

    # Point Smooth Loss
    if config.lr_multi.point_smooth_loss > 0:
        from losses.PointSmoothLoss import PointSmoothLoss

        loss_functions["point_smooth"] = PointSmoothLoss(
            knn_loss_params=config.loss.point_smooth_loss.knn_loss_params,
            ball_q_loss_params=config.loss.point_smooth_loss.ball_q_loss_params,
        )
    else:
        loss_functions["point_smooth"] = None

    # KDTree Loss
    if config.lr_multi.KDTree_loss > 0:
        from losses.KDTreeDistanceLoss import KDTreeDistanceLoss

        kdtree_loss = KDTreeDistanceLoss(
            max_distance=config.loss.kdtree.max_distance, reduction=config.loss.kdtree.reduction
        )
        kdtree_loss.to(device)
        loss_functions["kdtree"] = kdtree_loss
    else:
        loss_functions["kdtree"] = None

    # KNN Loss
    if config.lr_multi.KNN_loss > 0:
        from losses.KNNDistanceLoss import TruncatedKNNDistanceLoss

        loss_functions["knn"] = TruncatedKNNDistanceLoss(
            k=config.loss.knn.k,
            distance_max_threshold=config.loss.knn.distance_max_threshold,
            distance_min_threshold=config.loss.knn.distance_min_threshold,
            reduction=config.loss.knn.reduction,
        )
    else:
        loss_functions["knn"] = None

    # Invariance Loss
    if config.lr_multi.invariance_loss > 0:
        from losses.InvarianceLoss import InvarianceLoss

        loss_functions["invariance"] = InvarianceLoss()
    else:
        loss_functions["invariance"] = None

    return loss_functions


def initialize_visualization(config):
    """Initialize visualization components if enabled.

    Args:
        config: Configuration object

    Returns:
        tuple: (vis, pcd, gt_pcd, reconstructed_pcd) or (None, None, None, None)
    """
    if config.vis.show_window:
        import open3d as o3d

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        gt_pcd = o3d.geometry.PointCloud()
        reconstructed_pcd = o3d.geometry.PointCloud()
        return vis, pcd, gt_pcd, reconstructed_pcd
    return None, None, None, None


def setup_checkpointing(config, device):
    """Setup checkpointing configuration and resume logic.

    Args:
        config: Configuration object
        device: Training device

    Returns:
        tuple: (checkpoint_dir, save_every_iters, step)
    """
    checkpoint_dir = os.path.join(config.log.dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume logic
    save_every_iters = config.checkpoint.save_every_iters
    resume = config.checkpoint.resume
    resume_path = config.checkpoint.resume_path
    step = 0

    return checkpoint_dir, save_every_iters, step, resume, resume_path


def load_checkpoint(
    config, flow_predictor, mask_predictor, optimizer_flow, optimizer_mask, alter_scheduler, mask_scheduler=None
):
    """Load checkpoint if resume is enabled.

    Args:
        config: Configuration object
        flow_predictor: Scene flow model
        mask_predictor: Mask prediction model
        optimizer_flow: Flow optimizer
        optimizer_mask: Mask optimizer
        alter_scheduler: Alternating scheduler

    Returns:
        int: Starting step number
    """
    step = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.checkpoint.resume:
        resume_path = config.checkpoint.resume_path
        checkpoint_dir = config.log.dir
        candidate_path = resume_path if resume_path else os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(candidate_path):
            ckpt = torch.load(candidate_path, map_location=device)
            if "scene_flow_predictor" in ckpt:
                flow_predictor.load_state_dict(ckpt["scene_flow_predictor"])
            else:
                flow_predictor.load_state_dict(ckpt["flow_predictor"])
            mask_predictor.load_state_dict(ckpt["mask_predictor"])
            optimizer_flow.load_state_dict(ckpt.get("optimizer_flow", optimizer_flow.state_dict()))
            optimizer_mask.load_state_dict(ckpt.get("optimizer_mask", optimizer_mask.state_dict()))
            if "alter_scheduler" in ckpt:
                try:
                    alter_scheduler.load_state_dict(ckpt["alter_scheduler"])
                except Exception:
                    pass
            if "mask_scheduler" in ckpt and mask_scheduler is not None:
                try:
                    mask_scheduler.load_state_dict(ckpt["mask_scheduler"])
                except Exception:
                    pass
            step = int(ckpt.get("step", 0))
            tqdm.write(f"Resumed from checkpoint: {candidate_path} (step={step})")
        else:
            tqdm.write(f"No checkpoint found at {candidate_path}, starting fresh.")
    if config.checkpoint.overwrite_flow_predictor:
        print("Overwriting flow predictor with checkpoint: ", config.checkpoint.overwrite_flow_path)
        flow_predictor = load_flow_predictor_from_checkpoint(
            flow_predictor, config.checkpoint.overwrite_flow_path, device
        )
        optimizer_flow = torch.optim.Adam(
            flow_predictor.parameters(), lr=config.model.flow.lr
        )  # reload the optimizer to make it work
    # set learning rate to the value in the config
    optimizer_flow.param_groups[0]["lr"] = config.model.flow.lr
    optimizer_mask.param_groups[0]["lr"] = config.model.mask.lr
    mask_scheduler.step()
    return step


def load_flow_predictor_from_checkpoint(flow_predictor, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "scene_flow_predictor" in ckpt:
        flow_predictor.load_state_dict(ckpt["scene_flow_predictor"])
    elif "flow_predictor" in ckpt:
        flow_predictor.load_state_dict(ckpt["flow_predictor"])
    else:
        state_dict = ckpt["state_dict"]
        # remove the "model." prefix from the state_dict
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        flow_predictor.load_state_dict(state_dict)
    return flow_predictor


def create_checkpoint_saver(
    checkpoint_dir,
    flow_predictor,
    mask_predictor,
    optimizer_flow,
    optimizer_mask,
    alter_scheduler,
    config,
    mask_scheduler=None,
):
    """Create checkpoint saving function.

    Returns:
        function: Checkpoint saving function
    """

    def save_checkpoint(path_latest: str, step_value: int):
        state = {
            "step": step_value,
            "flow_predictor": flow_predictor.state_dict(),
            "mask_predictor": mask_predictor.state_dict(),
            "optimizer_flow": optimizer_flow.state_dict(),
            "optimizer_mask": optimizer_mask.state_dict(),
            "alter_scheduler": getattr(alter_scheduler, "state_dict", lambda: {})(),
            "mask_scheduler": getattr(mask_scheduler, "state_dict", lambda: {})() if mask_scheduler is not None else {},
            "config": OmegaConf.to_container(config, resolve=True),
        }
        torch.save(state, path_latest)
        # Also keep a step-suffixed snapshot
        step_path = os.path.join(checkpoint_dir, f"step_{step_value}.pt")
        try:
            torch.save(state, step_path)
        except Exception:
            pass

    return save_checkpoint
