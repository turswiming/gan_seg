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
from config.config import correct_datatype

# Import refactored modules
from utils.model_utils import (
    setup_device_and_training,
    initialize_models_and_optimizers,
    initialize_loss_functions,
    initialize_visualization,
    setup_checkpointing,
    load_checkpoint,
    create_checkpoint_saver,
)
from utils.forward_utils import forward_scene_flow, forward_mask_prediction
from utils.training_utils import (
    determine_training_modes,
    set_model_training_modes,
    log_gradient_debug_info,
    perform_optimization_step,
    handle_checkpoint_saving,
    handle_evaluation,
    log_prediction_histograms,
    handle_visualization,
    cleanup_memory,
)
from losses.loss_functions import compute_all_losses


def main(config, writer):
    """
    Main training function.

    Args:
        config: Configuration object containing all training parameters
        writer: TensorBoard SummaryWriter for logging
    """
    # Setup device and basic configurations
    device = setup_device_and_training()

    # Create dataloaders
    dataloader, infinite_loader, val_dataloader, batch_size, N = create_dataloaders(config)

    # Initialize models, optimizers and schedulers
    (
        mask_predictor,
        flow_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
        scene_flow_smoothness_scheduler,
    ) = initialize_models_and_optimizers(config, N, device)

    # Initialize loss functions
    loss_functions = initialize_loss_functions(config, device)

    # Initialize visualization
    if config.vis.show_window:
        vis, pcd, gt_pcd, reconstructed_pcd = initialize_visualization(config)

    # Setup checkpointing
    checkpoint_dir, save_every_iters, step, resume, resume_path = setup_checkpointing(config, device)

    # Load checkpoint if resuming
    step = load_checkpoint(
        resume,
        resume_path,
        checkpoint_dir,
        device,
        flow_predictor,
        mask_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
    )

    # Create checkpoint saver
    save_checkpoint = create_checkpoint_saver(
        checkpoint_dir, flow_predictor, mask_predictor, optimizer_flow, optimizer_mask, alter_scheduler, config
    )

    first_iteration = True

    # Main training loop
    with tqdm(infinite_loader, desc="Training", total=config.training.max_iter - step) as infinite_loader:
        tqdm.write("Starting training...")
        for sample in infinite_loader:
            step += 1
                        # Clear memory cache
            if step % config.hardware.clear_cache_interval == 0:
                cleanup_memory()
            if len(sample["idx"]) == 0:
                continue

            if step > config.training.max_iter:
                tqdm.write("Reached maximum training iterations, stopping.")
                break

            # Determine training modes and set model states
            train_flow, train_mask = determine_training_modes(step, config, alter_scheduler)
            set_model_training_modes(flow_predictor, mask_predictor, train_flow, train_mask)

            # Prepare input data
            point_cloud_firsts = [item.to(device) for item in sample["point_cloud_first"]]
            idxs = sample.get("idx")
            nextbatchs = [sample["self"][0].get_item(idx + 1) for idx in idxs]
            point_cloud_nexts = [item["point_cloud_first"].to(device) for item in nextbatchs]

            # Forward pass for scene flow
            pred_flow, reverse_pred_flow, longterm_pred_flow = forward_scene_flow(
                point_cloud_firsts, point_cloud_nexts, sample, flow_predictor, config, train_flow, device
            )

            # Forward pass for mask prediction
            pred_mask = forward_mask_prediction(point_cloud_firsts, sample, mask_predictor, config, train_mask)

            # Compute all losses
            loss_dict, total_loss, reconstructed_points = compute_all_losses(
                config,
                loss_functions,
                flow_predictor,
                mask_predictor,
                point_cloud_firsts,
                point_cloud_nexts,
                pred_flow,
                reverse_pred_flow,
                longterm_pred_flow,
                pred_mask,
                sample,
                step,
                scene_flow_smoothness_scheduler,
                train_flow,
                train_mask,
                device,
            )

            # Log to tensorboard
            if step % config.log.tensorboard_log_interval == 0:
                loss_log_dict = {name: loss.item() for name, loss in loss_dict.items()}
                loss_log_dict["total_loss"] = total_loss.item()
                writer.add_scalars("losses", loss_log_dict, step)

            # Log gradient debugging information
            log_gradient_debug_info(config, writer, loss_dict, flow_predictor, mask_predictor, step)

            # Perform optimization step
            optimization_success = perform_optimization_step(
                config,
                total_loss,
                optimizer_flow,
                optimizer_mask,
                flow_predictor,
                mask_predictor,
                train_flow,
                train_mask,
                step,
            )

            if not optimization_success:
                continue

            alter_scheduler.step()

            # Handle checkpoint saving
            handle_checkpoint_saving(save_every_iters, step, checkpoint_dir, save_checkpoint)

            # Handle evaluation
            handle_evaluation(config, step, flow_predictor, mask_predictor, dataloader, device, writer)



            # Log prediction histograms
            log_prediction_histograms(config, writer, pred_flow, pred_mask, step)

            # Handle visualization
            first_iteration, lineset, lineset_gt = handle_visualization(
                config,
                vis,
                pcd,
                gt_pcd,
                point_cloud_firsts,
                point_cloud_nexts,
                pred_flow,
                pred_mask,
                sample,
                first_iteration,
            )

            pass  # end loop

    # Cleanup
    if config.vis.show_window:
        vis.destroy_window()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scene Flow and Mask Prediction")
    parser.add_argument("--config", type=str, default="config/baseconfig.yaml", help="Path to the config file")

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
