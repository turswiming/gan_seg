"""
Argoverse2 mask inference script with quadrant merging.

功能 / Functionality
--------------------
- 从 config 读取 Argoverse2 AV2SceneFlowZoo 数据配置与模型配置；
- 加载训练好的 mask predictor（例如 Sonata / PTV3）权重；
- 对同一帧被 AV2SceneFlowZoo 切成的 4 份点云分别做 mask 预测；
- 将 4 份点云在全局坐标系下拼接回同一个点云；
- 如果来自不同子点云的两个实例 mask 在空间上相邻，则将它们合并为同一种颜色（同一个实例 ID）；
- 将每一帧的合并结果保存为 `.npz` 文件：`points` (N,3) 和 `labels` (N,)。

用法 / Usage
------------
示例：

    python infer_av2_mask_quadrants.py \\
        --config src/config/general_av2_ptv3.yaml \\
        --output_dir ./av2_infer_results \\
        --max_frames 50 \\
        --merge_radius 1.0

说明：
- 默认使用 config 中的 checkpoint.resume_path 作为权重路径；
- `merge_radius` 控制跨子点云实例合并的空间距离阈值（单位与点云坐标一致）；
- 没有在代码中加入额外的测试或可视化逻辑，仅做推理与保存。
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

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


def build_av2_val_flow_dataset(config) -> AV2SceneFlowZoo:
    """
    根据 config 构建 AV2SceneFlowZoo 验证集（用于 mask 推理）。

    Build AV2SceneFlowZoo validation dataset for mask inference.
    """
    base_cfg = config.dataset.AV2_SceneFlowZoo
    val_cfg = config.dataset.AV2_SceneFlowZoo_val_flow

    dataset = AV2SceneFlowZoo(
        point_size=base_cfg.point_size,
        root_dir=Path(val_cfg.root_dir),
        subsequence_length=base_cfg.subsequence_length,
        sliding_window_step_size=base_cfg.sliding_window_step_size,
        with_ground=val_cfg.with_ground,
        use_gt_flow=val_cfg.use_gt_flow,
        eval_type=val_cfg.eval_type,
        expected_camera_shape=val_cfg.expected_camera_shape,
        eval_args=dict(output_path=getattr(val_cfg, "eval_args_output_path", "")),
        with_rgb=val_cfg.with_rgb,
        flow_data_path=Path(val_cfg.flow_data_path),
        range_crop_type="ego",
        load_flow=True,
        load_boxes=False,
        cache_root=(
            Path(config.dataset.AV2_SceneFlowZoo.cache_root)
            if hasattr(config.dataset.AV2_SceneFlowZoo, "cache_root")
            else Path("/tmp/val_mask_cache/")
        ),
        min_instance_size=base_cfg.min_instance_size,
    )
    return dataset


class UnionFind:
    """简单并查集 / Simple union-find for merging instance clusters."""

    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


def merge_quadrant_instances(
    points_list: List[torch.Tensor],
    masks_list: List[torch.Tensor],
    valid_mask_list: List[torch.Tensor],
    merge_radius: float,
    do_merge_clusters: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将四个子点云及其 mask 预测结果合并为一个点云和统一的实例标签。

    Merge four quadrant point clouds and mask predictions into one point cloud
    with unified instance labels across quadrants when they are spatially adjacent.

    Args:
        points_list: list of (N_q, 3) tensors, each on CPU.
        masks_list: list of (K, N_q) tensors (slot-first) on CPU.
        merge_radius: distance threshold to merge instances across quadrants.

    Returns:
        points_all: (N_total, 3) numpy array
        labels_all: (N_total,) numpy array of merged instance ids
    """
    # 拼接点云 / concatenate all points
    points_all = torch.cat(points_list, dim=0)  # (N, 3)

    # 为每个 (quadrant, slot) 创建一个 cluster
    # 每个 cluster 保存：点索引、所属 quadrant、以及该 cluster 的 bounding box
    clusters = []  # list of dict: { "indices": Tensor, "qid": int, "bbox_min": Tensor, "bbox_max": Tensor }
    start = 0
    for qid, (pts, masks) in enumerate(zip(points_list, masks_list)):
        n_q = pts.shape[0]
        # mask shape: (K, N_q)
        # 先计算哪些点在所有 slot 上都为 0，这些点在 merge 时完全忽略
        # First, compute points whose all slots are zero and ignore them in clustering
        mask_sums_q = masks.abs().sum(dim=0)  # (N_q,)
        nonzero_mask_q = mask_sums_q > 0  # True where at least one slot is non-zero

        if nonzero_mask_q.sum() == 0:
            start += n_q
            continue

        labels_q = masks.argmax(dim=0)  # (N_q,)
        for slot_id in labels_q.unique().tolist():
            slot_id_int = int(slot_id)
            # 只在 nonzero_mask_q 为 True 的点上考虑该 slot
            idx_local = (
                (labels_q == slot_id_int) & nonzero_mask_q
            ).nonzero(as_tuple=False).squeeze(-1)
            if idx_local.numel() == 0:
                continue
            idx_global = idx_local + start
            pts_cluster = points_all[idx_global]
            bbox_min = pts_cluster.min(dim=0).values  # (3,)
            bbox_max = pts_cluster.max(dim=0).values  # (3,)
            clusters.append(
                {
                    "indices": idx_global,
                    "qid": qid,
                    "bbox_min": bbox_min,
                    "bbox_max": bbox_max,
                }
            )
        start += n_q

    if not clusters:
        labels_all = torch.full((points_all.shape[0],), -1, dtype=torch.long)
        return points_all.cpu().numpy(), labels_all.cpu().numpy()

    num_clusters = len(clusters)
    uf = UnionFind(num_clusters)

    if do_merge_clusters:
        # 计算 cluster 之间的 bounding box 距离，仅在不同 quadrant 之间考虑合并
        # Compute distance between AABBs (axis-aligned bounding boxes) and merge
        # clusters whose bbox distance is <= merge_radius.
        with torch.no_grad():
            for i in range(num_clusters):
                for j in range(i + 1, num_clusters):

                    bbox_min_i = clusters[i]["bbox_min"]
                    bbox_max_i = clusters[i]["bbox_max"]
                    bbox_min_j = clusters[j]["bbox_min"]
                    bbox_max_j = clusters[j]["bbox_max"]

                    # 逐维计算两个 AABB 之间的间距（若重叠则该维距离为 0）
                    # Compute per-axis distances between two AABBs (0 if they overlap on that axis)
                    dx = torch.maximum(
                        bbox_min_j[0] - bbox_max_i[0],
                        bbox_min_i[0] - bbox_max_j[0],
                    )
                    dy = torch.maximum(
                        bbox_min_j[1] - bbox_max_i[1],
                        bbox_min_i[1] - bbox_max_j[1],
                    )
                    dz = torch.maximum(
                        bbox_min_j[2] - bbox_max_i[2],
                        bbox_min_i[2] - bbox_max_j[2],
                    )

                    # 如果在某一维度上重叠，则该维距离为 <= 0，取 max(., 0) 得到非负分量
                    dx = torch.clamp(dx, min=0.0)
                    dy = torch.clamp(dy, min=0.0)
                    dz = torch.clamp(dz, min=0.0)

                    dist = torch.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist.item() <= merge_radius:
                        uf.union(i, j)

    # 为每个 cluster 分配最终实例 ID（如果 do_merge_clusters=False，则每个 cluster 保持独立 ID）
    root_to_label = {}
    next_label = 0
    cluster_labels = [0] * num_clusters
    for cid in range(num_clusters):
        root = uf.find(cid)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        cluster_labels[cid] = root_to_label[root]

    # 为所有点构建 label 向量
    labels_all = torch.full((points_all.shape[0],), -1, dtype=torch.long)
    for cid, cluster in enumerate(clusters):
        labels_all[cluster["indices"]] = cluster_labels[cid]

    # 未被任何 cluster 覆盖的点设为 -1
    labels_all[labels_all < 0] = -1

    # 计算哪些点在所有 slot 上都是 0：即 masks_list 所有 slot 全为 0 的点
    # Compute mask_zero: points where all slots are exactly zero across masks_list
    with torch.no_grad():
        mask_sums_list = []
        for masks in masks_list:  # masks: (K, N_q)
            # 对每个点在所有 slot 上取绝对值和，形状 (N_q,)
            mask_sums_list.append(masks.abs().sum(dim=0))
        mask_sums_all = torch.cat(mask_sums_list, dim=0)  # (N,)
        zero_mask = mask_sums_all == 0
    # 对这些点强制设为 -1
    labels_all[zero_mask] = -1

    return points_all.cpu().numpy(), labels_all.cpu().numpy()


def run_inference(
    config_path: str,
    output_dir: str,
    scene_idx: int,
    merge_radius: float = 1.0,
    disable_merge_clusters: bool = False,
):
    """
    运行单个 AV2 场景（由 4 个切片构成）的推理、合并与可视化。

    Run inference, merging and visualization for a single AV2 scene
    (composed of 4 quadrants in AV2SceneFlowZoo).
    """
    # 加载配置 / load config
    config_obj = load_config_with_inheritance(config_path)
    cli_opts = OmegaConf.from_cli()
    config = OmegaConf.merge(config_obj, cli_opts)
    config = correct_datatype(config)

    device = setup_device_and_training()

    # 构建数据集（仅使用 AV2_SceneFlowZoo_val_mask） / build dataset
    # dataset = build_av2_val_mask_dataset(config)
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

    # 只需要 mask_predictor，flow_predictor 保持在 eval 但不使用
    mask_predictor.to(device)
    flow_predictor.to(device)
    mask_predictor.eval()
    flow_predictor.eval()

    # 确保从 config.checkpoint.resume_path 加载权重
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

    # 每 4 个 index 对应同一个原始场景 / frame in this dataset design
    num_quadrants = 4
    base_index = scene_idx * num_quadrants
    if base_index + num_quadrants - 1 >= len(dataset):
        raise ValueError(
            f"scene_idx={scene_idx} is out of range for dataset of length {len(dataset)} "
            f"(each scene corresponds to indices [scene_idx*4 + 0..3])."
        )

    sub_indices = [base_index + i for i in range(num_quadrants)]

    points_list: List[torch.Tensor] = []
    valid_mask_list: List[torch.Tensor] = []
    masks_list: List[torch.Tensor] = []
    class_ids_list: List[torch.Tensor] = []
    sequence_id = None
    
    with torch.no_grad():
        for si in sub_indices:
            sample = dataset[si]
            pc = sample["point_cloud_first"].to(device).float()
            points_list.append(pc)  # keep CPU copy for merging
            valid_mask_first = sample["valid_mask_first"].to(device)
            valid_mask_list.append(valid_mask_first)
            class_ids = torch.from_numpy(sample["class_ids"]).to(device)
            class_ids_list.append(class_ids)
        centers = []
        for i in range(len(points_list)):
            centers.append(torch.mean(points_list[i], dim=0))
            points_list[i] = points_list[i] - centers[i]

        for i, pc in enumerate(points_list):
            # 使用通用的前向接口做 mask 预测 / use shared forward for mask
            pred_masks = forward_mask_prediction_general([pc[valid_mask_list[i]]], mask_predictor)
            pred_mask_origin_shape = torch.zeros(
                pred_masks[0].shape[0], pc.shape[0], device=device
            )
            pred_mask_origin_shape[:, valid_mask_list[i]] = pred_masks[0]
            pred_mask_origin_shape[:,class_ids_list[i]==0] = 0
            pred_mask_origin_shape[:,class_ids_list[i]==-1] = 0
            masks_list.append(pred_mask_origin_shape.cpu())

            if sequence_id is None:
                sequence_id = dataset[sub_indices[i]].get("sequence_id", f"seq_{scene_idx}")

        for i in range(len(points_list)):
            points_list[i] = (points_list[i] + centers[i]).cpu()
            
    points_all, labels_all = merge_quadrant_instances(
        points_list,
        masks_list,
        valid_mask_list,
        merge_radius=merge_radius,
        do_merge_clusters=not disable_merge_clusters,
    )

    # print(f"Points all shape: {points_all.shape}")
    # print(f"Labels all shape: {labels_all.shape}")
    # points_all, labels_all = points_list[0].numpy(), masks_list[0].argmax(
    #     dim=0
    # ).numpy().astype(np.int64)
    # 保存为 npz，文件名包含 sequence_id 和 scene_idx
    out_name = f"{sequence_id}_scene_{scene_idx:06d}.npz"
    out_path = os.path.join(output_dir, out_name)
    np.savez_compressed(
        out_path,
        points=points_all.astype(np.float32),
        labels=labels_all.astype(np.int64),
        sequence_id=str(sequence_id),
        scene_idx=scene_idx,
    )

    # 可视化：在无显示环境下用 Matplotlib 直接渲染到 PNG /
    # Visualization: render directly to PNG with Matplotlib (headless-safe)
    save_scene_visualization(points_all, labels_all, sequence_id, scene_idx, output_dir)


def save_scene_visualization(
    points_all: np.ndarray,
    labels_all: np.ndarray,
    sequence_id: str,
    scene_idx: int,
    output_dir: str,
    zoom_factor: float = 10.0,
    point_pixel: int = 5,
):
    """
    使用固定 45 度鸟瞰视角，将点云渲染到 PNG 文件（无窗口）。

    Render merged point cloud to a PNG file with a fixed 45-degree bird's-eye view
    without creating any GUI windows (headless-friendly).
    """
    # 计算颜色 / compute colors
    max_label = labels_all.max() if labels_all.size > 0 else -1
    colors = np.zeros((labels_all.shape[0], 3), dtype=np.float32)
    if max_label >= 0:
        rng = np.random.default_rng(42)
        palette = rng.uniform(0.1, 0.9, size=(max_label + 1, 3)).astype(np.float32)
        for i, lab in enumerate(labels_all):
            if lab >= 0:
                colors[i] = palette[int(lab)]
            else:
                colors[i] = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    else:
        colors[:] = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    # 使用 Matplotlib 的 Agg 后端进行离屏渲染 / use Agg backend for offscreen rendering
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, E402

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 设置等比例坐标轴 / enforce equal scaling on xyz
    x, y, z = points_all[:, 0], points_all[:, 1], points_all[:, 2]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) * 0.5
    if max_range <= 0:
        max_range = 1.0

    # 只看中心：把视野缩小为原来的 1/3，相当于放大 3 倍 / zoom 3x by shrinking range to 1/3
    center_range = max_range / zoom_factor

    ax.set_xlim(x_mid - center_range, x_mid + center_range)
    ax.set_ylim(y_mid - center_range, y_mid + center_range)
    ax.set_zlim(z_mid - center_range, z_mid + center_range)

    # 固定摄像机：位置在中心 + (48,48,48)，看向中心。
    # 在 Matplotlib 中通过等价的球坐标角度设置视角：
    # 方向向量为 (1,1,1) -> azim=45°, elev≈35.26°
    ax.view_init(elev=35, azim=45)

    ax.scatter(
        points_all[:, 0],
        points_all[:, 1],
        points_all[:, 2],
        c=colors,
        s=point_pixel,
        linewidths=0,
    )
    ax.set_axis_off()

    # 让图像紧凑一些 / tighter layout
    plt.tight_layout(pad=0)

    png_path = os.path.join(output_dir, f"{sequence_id}_scene_{scene_idx:06d}.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Argoverse2 mask inference with quadrant merging"
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
        help="Directory to save merged inference results (.npz)",
    )
    parser.add_argument(
        "--scene_idx",
        type=int,
        required=True,
        help="Scene index s.t. indices [scene_idx*4 + 0,1,2,3] are inferred",
    )
    parser.add_argument(
        "--merge_radius",
        type=float,
        default=1.0,
        help="Distance threshold to merge instances across quadrants",
    )
    parser.add_argument(
        "--dm",
        
        action="store_true",
        help="If set, do NOT merge clusters across quadrants (keep them separate)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        config_path=args.config,
        output_dir=args.output_dir,
        scene_idx=args.scene_idx,
        merge_radius=args.merge_radius,
        disable_merge_clusters=args.dm,
    )
