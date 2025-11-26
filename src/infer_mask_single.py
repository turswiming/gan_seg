"""
Single-sample Argoverse2 mask inference script.

基于 `infer_av2_mask_quadrants.py` 的逻辑，本脚本：
- 从配置文件构建 AV2SceneFlowZoo 数据集（与训练/验证一致）；
- 一次只对数据集中一个样本（单个 quadrant）做 mask 推理；
- 不做 idx*4 聚合，也不在不同 quadrant 之间拼接或合并 cluster；
- 使用与多 quadrant 版本一致的 mask 预测、class_ids 过滤、zero-mask 过滤和可视化方式。
"""

import argparse
import os
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

from config.config import correct_datatype
from utils.config_utils import load_config_with_inheritance
from utils.model_utils import (
    initialize_models_and_optimizers,
    load_checkpoint,
    setup_device_and_training,
)
from utils.forward_utils import forward_mask_prediction_general

# 复用多 quadrant 版本中的数据集构建和可视化工具 / reuse helpers from quadrants script
from infer_av2_mask_quadrants import (
    build_av2_val_flow_dataset,
    save_scene_visualization,
)


def run_inference_single(
    config_path: str,
    output_dir: str,
    scene_idx: int,
):
    """
    对单个 AV2SceneFlowZoo 样本（一个 quadrant）做 mask 推理和可视化。

    Run mask inference and visualization for a single AV2 sample (one quadrant).
    """
    # 加载配置 / load config
    config_obj = load_config_with_inheritance(config_path)
    cli_opts = OmegaConf.from_cli()
    config = OmegaConf.merge(config_obj, cli_opts)
    config = correct_datatype(config)

    device = setup_device_and_training()

    # 根据配置构建 AV2 验证数据集 / build AV2 val dataset from config
    dataset = build_av2_val_flow_dataset(config)

    # 初始化模型并加载 checkpoint（只使用 mask_predictor） / init model and load checkpoint
    point_size = config.dataset.AV2_SceneFlowZoo.point_size
    (
        mask_predictor,
        flow_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
        scene_flow_smoothness_scheduler,
        mask_scheduler,
    ) = initialize_models_and_optimizers(config, point_size, device)

    mask_predictor.to(device)
    flow_predictor.to(device)
    mask_predictor.eval()
    flow_predictor.eval()

    if hasattr(config, "checkpoint"):
        config.checkpoint.resume = True
    _ = load_checkpoint(
        config,
        flow_predictor,
        mask_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
        mask_scheduler,
    )

    os.makedirs(output_dir, exist_ok=True)

    if scene_idx < 0 or scene_idx >= len(dataset):
        raise ValueError(
            f"scene_idx={scene_idx} is out of range for dataset of length {len(dataset)} "
            f"(this script treats each dataset index as a single sample)."
        )

    with torch.no_grad():
        sample = dataset[scene_idx]
        pc = sample["point_cloud_first"].to(device).float()  # (N, 3)
        valid_mask_first = sample["valid_mask_first"].to(device).bool()  # (N,)
        class_ids = torch.from_numpy(sample["class_ids"]).to(device)

        # 中心化以和多 quadrant 版本保持一致 / center normalization (same as quadrants script)
        center = pc.mean(dim=0)
        pc_centered = pc - center

        # 只在 valid_mask 区域上做预测 / predict only on valid points
        pred_masks_list = forward_mask_prediction_general(
            [pc_centered[valid_mask_first]], mask_predictor
        )
        pred_masks = pred_masks_list[0]  # (K, N_valid)

        # 回填到原始点数目 / scatter back to full N
        pred_mask_origin_shape = torch.zeros(
            pred_masks.shape[0], pc.shape[0], device=device
        )
        pred_mask_origin_shape[:, valid_mask_first] = pred_masks

        # 按 class_ids 过滤背景和无效类 / zero-out background & invalid classes
        pred_mask_origin_shape[:, class_ids == 0] = 0
        pred_mask_origin_shape[:, class_ids == -1] = 0

        # 计算 labels：argmax + zero-mask 过滤 / compute labels via argmax with zero-mask filtering
        mask_sums = pred_mask_origin_shape.abs().sum(dim=0)  # (N,)
        labels = pred_mask_origin_shape.argmax(dim=0)  # (N,)
        labels[mask_sums == 0] = -1  # 所有 slot 为 0 的点设为 -1

        # 还原坐标并转 numpy / restore coordinates and convert to numpy
        points_all = (pc_centered + center).cpu().numpy().astype(np.float32)
        labels_all = labels.cpu().numpy().astype(np.int64)

        sequence_id = sample.get("sequence_id", f"seq_{scene_idx}")

    # 保存结果 / save npz
    out_name = f"{sequence_id}_single_{scene_idx:06d}.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez_compressed(
        out_path,
        points=points_all,
        labels=labels_all,
        sequence_id=str(sequence_id),
        scene_idx=scene_idx,
    )

    # 可视化：与多 quadrant 版本保持一致的视角和配色 / visualize with same style
    save_scene_visualization(points_all, labels_all, sequence_id, scene_idx, output_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Argoverse2 single-sample mask inference (no quadrants merging)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (e.g., src/config/general_av2_ptv3.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference results (.npz and .png)",
    )
    parser.add_argument(
        "--scene_idx",
        type=int,
        required=True,
        help="Index into AV2SceneFlowZoo dataset (treated as a single sample)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference_single(
        config_path=args.config,
        output_dir=args.output_dir,
        scene_idx=args.scene_idx,
    )

"""
Single-quadrant Argoverse2 mask inference script.

基于 `infer_av2_mask_quadrants.py` 的逻辑，但本脚本：
- 一次只推理一个 AV2SceneFlowZoo 样本（一个 quadrant）；
- 不做 idx*4 聚合，也不在不同 quadrant 之间做拼接或 cluster 合并；
- 仍然使用相同的 mask 预测、class_ids 过滤、zero-mask 过滤和可视化方式。
"""

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf

from config.config import correct_datatype
from utils.config_utils import load_config_with_inheritance
from utils.model_utils import (
    initialize_models_and_optimizers,
    load_checkpoint,
    setup_device_and_training,
)
from utils.forward_utils import forward_mask_prediction_general
from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo

# 复用多 quadrant 版本中的数据集构建和可视化工具 / reuse helpers
from infer_av2_mask_quadrants import (
    build_av2_val_flow_dataset,
    save_scene_visualization,
)


def run_inference_single(
    config_path: str,
    output_dir: str,
    scene_idx: int,
):
    # 加载配置 / load config
    config_obj = load_config_with_inheritance(config_path)
    cli_opts = OmegaConf.from_cli()
    config = OmegaConf.merge(config_obj, cli_opts)
    config = correct_datatype(config)

    device = setup_device_and_training()
    from utils.dataloader_utils import create_dataloaders_general

    (dataset, dataloader, val_flow_dataset, val_flow_dataloader, val_mask_dataset, val_mask_dataloader) = (
        create_dataloaders_general(config)
    )

    # 初始化模型并加载 checkpoint（只使用 mask_predictor） / init model and load checkpoint
    point_size = config.dataset.AV2_SceneFlowZoo.point_size
    (
        mask_predictor,
        flow_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
        scene_flow_smoothness_scheduler,
        mask_scheduler,
    ) = initialize_models_and_optimizers(config, point_size, device)

    mask_predictor.to(device)
    flow_predictor.to(device)
    mask_predictor.eval()
    flow_predictor.eval()

    if hasattr(config, "checkpoint"):
        config.checkpoint.resume = True
    _ = load_checkpoint(
        config,
        flow_predictor,
        mask_predictor,
        optimizer_flow,
        optimizer_mask,
        alter_scheduler,
        mask_scheduler,
    )

    os.makedirs(output_dir, exist_ok=True)

    if scene_idx < 0 or scene_idx >= len(dataset):
        raise ValueError(
            f"scene_idx={scene_idx} is out of range for dataset of length {len(dataset)} "
            f"(this script treats each dataset index as a single quadrant)."
        )
    print(f"begin inference: {scene_idx}")
    with torch.no_grad():
        sample = val_flow_dataset[scene_idx]
        pc = sample["point_cloud_first"].to(device).float()  # (N, 3)

        # 中心化以匹配多 quadrant 版本的处理 / center normalization (same as quadrants script)
        center = pc.mean(dim=0)
        center[1] =0
        pc_centered = pc - center

        # 只在 valid_mask 区域上做预测 / predict only on valid points
        pred_masks_list = forward_mask_prediction_general(
            [pc_centered], mask_predictor
        )
        pred_masks = pred_masks_list[0]  # (K, N_valid)

        # 回填到原始点数目 / scatter back to full N
        pred_mask_origin_shape = torch.zeros(
            pred_masks.shape[0], pc.shape[0], device=device
        )
        pred_mask_origin_shape = pred_masks


        # 计算 labels：argmax + zero-mask 过滤 / compute labels via argmax with zero-mask filtering
        mask_sums = pred_mask_origin_shape.abs().sum(dim=0)  # (N,)
        labels = pred_mask_origin_shape.argmax(dim=0)  # (N,)
        labels[mask_sums == 0] = -1  # 所有 slot 为 0 的点设为 -1

        # 还原坐标并转 numpy / restore coordinates and convert to numpy
        points_all = (pc_centered + center).cpu().numpy().astype(np.float32)
        labels_all = labels.cpu().numpy().astype(np.int64)
        #select masks that greater than 4000,set to -1
        unique_labels = np.unique(labels_all)
        for label in unique_labels:
            if (label ==labels_all).sum() > 4000:
                labels_all[labels_all == label] = -1
        sequence_id = sample.get("sequence_id", f"seq_{scene_idx}")

    # 保存结果 / save npz
    out_name = f"{sequence_id}_single_{scene_idx:06d}.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez_compressed(
        out_path,
        points=points_all,
        labels=labels_all,
        sequence_id=str(sequence_id),
        scene_idx=scene_idx,
    )

    # 可视化：与多 quadrant 版本保持一致的视角和配色 / visualize with same style
    save_scene_visualization(
        points_all[:,[0,2,1]],
        labels_all,
        sequence_id,
        scene_idx,
        output_dir,
        zoom_factor=1.5,
        point_pixel=3,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Argoverse2 single-quadrant mask inference (no quadrants merging)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (e.g., src/config/general_av2_ptv3.yaml)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save inference results (.npz and .png)",
    )
    parser.add_argument(
        "--scene_idx",
        type=int,
        required=True,
        help="Index into AV2SceneFlowZoo dataset (treated as a single quadrant)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference_single(
        config_path=args.config,
        output_dir=args.output_dir,
        scene_idx=args.scene_idx,
    )
