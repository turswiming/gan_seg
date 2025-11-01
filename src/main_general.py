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


# Third party imports
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from config.config import print_config
from tqdm import tqdm
import random

# Local imports
from utils.config_utils import load_config_with_inheritance, save_config_and_code
from utils.dataloader_utils import create_dataloaders_general, infinite_dataloader
from config.config import correct_datatype
from utils.forward_utils import inference_models_general

# Import refactored modules
from utils.model_utils import (
    setup_device_and_training,
    initialize_models_and_optimizers,
    initialize_loss_functions,
    setup_checkpointing,
    load_checkpoint,
    create_checkpoint_saver,
)
from utils.training_utils import (
    determine_training_modes,
    set_model_training_modes,
    log_gradient_debug_info,
    perform_optimization_step,
    handle_checkpoint_saving,
    handle_evaluation_general,
    log_prediction_histograms,
    cleanup_memory,
)
from losses.loss_functions import compute_all_losses_general


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(config, writer):
    """
    Main training function.

    Args:
        config: Configuration object containing all training parameters
        writer: TensorBoard SummaryWriter for logging
    """
    # Setup device and basic configurations
    device = setup_device_and_training()
    set_seed(config.seed)
    # Create dataloaders
    (dataset, dataloader, val_flow_dataset, val_flow_dataloader, val_mask_dataset, val_mask_dataloader) = (
        create_dataloaders_general(config)
    )

    # # Initialize models, optimizers and schedulers
    (
        mask_predictor,
        flow_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
        scene_flow_smoothness_scheduler,
        mask_scheduler,
    ) = initialize_models_and_optimizers(config, None, device)
    # infinite dataloader
    inf_dataloader = infinite_dataloader(dataloader)
    # # Initialize loss functions
    loss_functions = initialize_loss_functions(config, device)

    # # Initialize visualization
    # if config.vis.show_window:
    #     vis, pcd, gt_pcd, reconstructed_pcd = initialize_visualization(config)

    # # Setup checkpointing
    checkpoint_dir, save_every_iters, step, resume, resume_path = setup_checkpointing(config, device)

    # # Load checkpoint if resuming
    step = load_checkpoint(
        config, flow_predictor, mask_predictor, optimizer_flow, optimizer_mask, alter_scheduler, mask_scheduler
    )

    # Create checkpoint saver
    save_checkpoint = create_checkpoint_saver(
        checkpoint_dir,
        flow_predictor,
        mask_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
        config,
        mask_scheduler,
    )

    first_iteration = True
    loss_dict_move_average = []
    # Main training loop
    with tqdm(inf_dataloader, desc="Training", total=config.training.max_iter - step) as inf_dataloader:
        tqdm.write("Starting training...")
        for sample in inf_dataloader:
            step += 1

            handle_evaluation_general(
                config=config,
                step=step,
                flow_predictor=flow_predictor,
                mask_predictor=mask_predictor,
                val_flow_dataloader=val_flow_dataloader,
                val_mask_dataloader=val_mask_dataloader,
                train_dataloader=dataloader,
                device=device,
                writer=writer,
            )

            if step > config.training.max_iter:
                tqdm.write("Reached maximum training iterations, stopping.")
                break
            # Determine training modes and set model states
            train_flow, train_mask = determine_training_modes(step, config, alter_scheduler)
            set_model_training_modes(flow_predictor, mask_predictor, train_flow, train_mask)
            flow_predictor.to(device)
            mask_predictor.to(device)

            try:
                (pred_flow, pred_mask, point_cloud_firsts, point_cloud_nexts, cascade_flow_outs) = (
                    inference_models_general(
                        flow_predictor,
                        mask_predictor,
                        sample,
                        config.dataset.name,
                        train_flow=train_flow,
                        downsample=config.training.mask_downsample_factor,
                        augment_params=config.training.augment_params,
                    )
                )
            except Exception as e:
                print(e)
                import traceback

                print(traceback.format_exc())
                continue

            # Compute all losses
            loss_dict, total_loss, reconstructed_points = compute_all_losses_general(
                config=config,
                loss_functions=loss_functions,
                flow_predictor=flow_predictor,
                mask_predictor=mask_predictor,
                point_cloud_firsts=point_cloud_firsts,
                point_cloud_nexts=point_cloud_nexts,
                pred_flow=pred_flow,
                pred_mask=pred_mask,
                step=step,
                scene_flow_smoothness_scheduler=scene_flow_smoothness_scheduler,
                train_flow=train_flow,
                train_mask=train_mask,
                device=device,
                cascade_flow_outs=cascade_flow_outs,
            )

            # Log to tensorboard
            loss_dict_move_average.append(loss_dict)
            if step % config.log.tensorboard_log_interval == 0:
                loss_mean_dict = {
                    name: np.mean([loss_dict[name].cpu().item() for loss_dict in loss_dict_move_average])
                    for name in loss_dict_move_average[0].keys()
                }
                loss_mean_dict["total_loss"] = np.sum([loss_mean_dict[key] for key in loss_mean_dict.keys()])
                writer.add_scalars("losses", loss_mean_dict, step)
                loss_dict_move_average = []

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
            )

            if not optimization_success:
                continue

            alter_scheduler.step()
            # Step mask scheduler if available
            if mask_scheduler is not None:
                mask_scheduler.step()

            # Handle checkpoint saving
            handle_checkpoint_saving(save_every_iters, step, checkpoint_dir, save_checkpoint)

            # Handle evaluation

            # Clear memory cache
            if step % config.hardware.clear_cache_interval == 0:
                cleanup_memory()

            # Log prediction histograms
            log_prediction_histograms(config, writer, pred_flow, pred_mask, step)

            # Handle visualization
            # first_iteration, lineset, lineset_gt = handle_visualization(
            #     config, vis, pcd, gt_pcd, point_cloud_firsts, point_cloud_nexts,
            #     pred_flow, pred_mask, sample, first_iteration)

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
