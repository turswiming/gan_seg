import argparse
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from utils.config_utils import load_config_with_inheritance
from SceneFlowZoo.dataloaders import TorchFullFrameInputSequence
from SceneFlowZoo.models.base_models import BaseTorchModel

from main_general import (
    create_dataloaders_general,
    initialize_models_and_optimizers,
    load_checkpoint,
    forward_scene_flow_general,
    forward_mask_prediction_general,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Infer one frame and visualize results")
    parser.add_argument("--config", type=str, required=True, help="Path to general.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to saved checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g., cuda:0 or cpu")
    return parser.parse_args()


def visualize_pointclouds(pc0, pc1, title: str):
    plt.figure(figsize=(10, 5))
    ax = plt.subplot(1, 2, 1, projection="3d")
    ax.scatter(pc0[:, 0].cpu(), pc0[:, 1].cpu(), pc0[:, 2].cpu(), s=0.2, c="b")
    ax.set_title("pc0")

    ax = plt.subplot(1, 2, 2, projection="3d")
    ax.scatter(pc1[:, 0].cpu(), pc1[:, 1].cpu(), pc1[:, 2].cpu(), s=0.2, c="r")
    ax.set_title("pc1")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def visualize_warp(pc0, pc1, flow, title: str):
    warped = pc0 + flow
    plt.figure(figsize=(15, 5))

    ax = plt.subplot(1, 3, 1, projection="3d")
    ax.scatter(pc0[:, 0].cpu(), pc0[:, 1].cpu(), pc0[:, 2].cpu(), s=0.2, c="b")
    ax.set_title("pc0")

    ax = plt.subplot(1, 3, 2, projection="3d")
    ax.scatter(pc1[:, 0].cpu(), pc1[:, 1].cpu(), pc1[:, 2].cpu(), s=0.2, c="r")
    ax.set_title("pc1")

    ax = plt.subplot(1, 3, 3, projection="3d")
    ax.scatter(warped[:, 0].cpu(), warped[:, 1].cpu(), warped[:, 2].cpu(), s=0.2, c="g")
    ax.set_title("pc0 + flow (warped)")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    config = load_config_with_inheritance(args.config)


    device = torch.device(args.device)

    (
        dataset,
        dataloader,
        val_flow_dataset,
        val_flow_dataloader,
        val_mask_dataset,
        val_mask_dataloader,
    ) = create_dataloaders_general(config)

    (
        mask_predictor,
        flow_predictor,
        _optimizer_flow,
        _optimizer_mask,
        alter_scheduler,
        _scene_flow_smoothness_scheduler,
    ) = initialize_models_and_optimizers(config, None, device)

    _ = load_checkpoint(
        resume=True,
        resume_path=Path(args.checkpoint),
        checkpoint_dir=Path(args.checkpoint).parent,
        device=device,
        flow_predictor=flow_predictor,
        mask_predictor=mask_predictor,
        optimizer_flow=_optimizer_flow,
        optimizer_mask=_optimizer_mask,
        alter_scheduler=alter_scheduler,
    )

    flow_predictor.to(device).eval()
    mask_predictor.to(device).eval()

    framelists = next(iter(dataloader))
    sequences = [
        TorchFullFrameInputSequence.from_frame_list(
            idx=0,
            frame_list=framelist,
            pc_max_len=120000,
            loader_type=dataset.loader_type(),
            allow_pc_slicing=False,
        ).to(device)
        for framelist in framelists
    ]

    flow_out = forward_scene_flow_general(sequences, flow_predictor)
    mask_out = forward_mask_prediction_general(sequences, mask_predictor)

    seq = sequences[0]
    ego_pc0 = seq.get_full_ego_pc(0)
    ego_pc1 = seq.get_full_ego_pc(1)
    mask0 = seq.get_full_pc_mask(0)
    mask1 = seq.get_full_pc_mask(1)

    pc0 = ego_pc0[mask0]
    pc1 = ego_pc1[mask1]

    flow_full = flow_out[0].ego_flows.squeeze(0)
    flow = flow_full[mask0]

    visualize_pointclouds(pc0, pc1, title="Input Point Clouds (ego frame)")
    visualize_warp(pc0, pc1, flow, title="Warped pc0 vs pc1 (ego frame)")

    # Optionally visualize mask logits/predictions if needed
    _ = mask_out


if __name__ == "__main__":
    main()


