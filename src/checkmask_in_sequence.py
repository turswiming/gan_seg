"""
Evaluate a list of checkpoints sequentially, compute mIoU / per-class IoU / speed
statistics, and store the results into a JSON file for later visualization.
"""

from pathlib import Path
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import CATEGORY_MAP
from utils.forward_utils import augment_transform, forward_mask_prediction_general
from utils.metrics import calculate_miou
from Predictor import get_mask_predictor


MIN_INSTANCE_SIZE = 50
MAX_EVAL_BATCHES = 10
OUTPUT_JSON_PATH = Path("/workspace/gan_seg/outputs/mask_sequence_metrics.json")

# ---------------------------------------------------------------------------
# Dataset and dataloaders
# ---------------------------------------------------------------------------
av2_sceneflow_zoo_flow = AV2SceneFlowZoo(
    root_dir=Path("/workspace/av2data/val"),
    expected_camera_shape=(194, 256, 3),
    eval_args={},
    with_rgb=False,
    flow_data_path=Path("/workspace/av2flow/val"),
    range_crop_type="ego",
    point_size=8192,
    load_flow=True,
    load_boxes=False,
    min_instance_size=MIN_INSTANCE_SIZE,
    cache_root=Path("/tmp/val_flow_cache/"),
)

flow_generator = torch.Generator().manual_seed(42)
flow_dataloader = DataLoader(av2_sceneflow_zoo_flow, batch_size=10, shuffle=True, generator=flow_generator)

av2_sceneflow_zoo_mask = AV2SceneFlowZoo(
    root_dir=Path("/workspace/av2data/val"),
    expected_camera_shape=(194, 256, 3),
    eval_args={},
    with_rgb=False,
    flow_data_path=Path("/workspace/av2flow/val"),
    range_crop_type="ego",
    point_size=8192,
    load_flow=False,
    load_boxes=True,
    min_instance_size=MIN_INSTANCE_SIZE,
    cache_root=Path("/tmp/val_mask_cache/"),
)

mask_generator = torch.Generator().manual_seed(42)
mask_dataloader = DataLoader(av2_sceneflow_zoo_mask, batch_size=10, shuffle=True, generator=mask_generator)

# ---------------------------------------------------------------------------
# Config / checkpoints
# ---------------------------------------------------------------------------
config_path = "//workspace/gan_seg/outputs/exp/20251111_132106run3/config.yaml"
config = OmegaConf.load(config_path)
config.model.mask.slot_num = 20

checkpoint_lists = [
    "/workspace/gan_seg/outputs/exp/20251111_080942run2/checkpoints/step_6000.pt",
    "/workspace/gan_seg/outputs/exp/20251111_080942run2/checkpoints/step_9900.pt",
    "/workspace/gan_seg/outputs/exp/20251111_132106run3/checkpoints/step_20100.pt",
    "/workspace/gan_seg/outputs/exp/20251111_132106run3/checkpoints/step_30000.pt",
    "/workspace/gan_seg/outputs/exp/20251112_142843run4/checkpoints/step_40000.pt",
    "/workspace/gan_seg/outputs/exp/20251112_142843run4/checkpoints/step_50000.pt",
    "/workspace/gan_seg/outputs/exp/20251112_142843run4/checkpoints/step_60000.pt",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_mask_predictor(ckpt_path: str):
    predictor = get_mask_predictor(config.model.mask, 10)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    predictor.load_state_dict(ckpt["mask_predictor"], strict=False)
    predictor = predictor.to(device)
    predictor.eval()
    return predictor


def evaluate_checkpoint(ckpt_path: str):
    print(f"[Eval] {ckpt_path}")
    mask_predictor = load_mask_predictor(ckpt_path)

    miou_list = []
    filtered_miou_list = []
    valid_mask_counts = []
    iou_dict = defaultdict(list)
    speed_dict = defaultdict(list)
    static_count = 0
    processed_batches = 0
    processed_samples = 0

    start_time = time.time()
    with torch.no_grad():
        for sample_flow, sample_mask in zip(flow_dataloader, mask_dataloader):
            if processed_batches >= MAX_EVAL_BATCHES:
                break

            flow_first = sample_flow["flow"]
            class_ids_gt = sample_flow["class_ids"]
            pc_first_mask = sample_mask["point_cloud_first"]
            instance_mask_gt = sample_mask.get("mask")

            for flow in flow_first:
                if flow.std(dim=0).max() < 0.01:
                    static_count += 1

            pc_first_aug = []
            for j in range(len(pc_first_mask)):
                pc_first_mask_j = pc_first_mask[j]
                aug_params = config.training.augment_params
                aug_params.scale = [1, 1]
                aug_params.rotation = [0, 0]
                aug_params.translation_range = [0, 0, 0]
                aug_params.mirror_x = False
                aug_params.mirror_z = False
                pc_first_aug_j, _, _, _ = augment_transform(
                    pc_first_mask_j.clone().to(device),
                    pc_first_mask_j.clone().to(device),
                    torch.zeros(pc_first_mask_j.shape[0], 3).to(device),
                    None,
                    config.training.augment_params,
                )
                pc_first_aug.append(pc_first_aug_j)

            pc_first_aug = torch.stack(pc_first_aug)
            pred_masks_logits = forward_mask_prediction_general(pc_first_aug, mask_predictor)

            for j in range(len(pred_masks_logits)):
                pred_masks_logits_j = pred_masks_logits[j]
                instance_mask_gt_j = instance_mask_gt[j]
                instance_mask_gt_onehot_j = F.one_hot(instance_mask_gt_j).to(device=pred_masks_logits_j.device)

                miou_value = calculate_miou(
                    pred_masks_logits_j,
                    instance_mask_gt_onehot_j.permute(1, 0),
                    min_points=MIN_INSTANCE_SIZE,
                )
                miou_list.append(miou_value)
                valid_mask_counts.append(instance_mask_gt_onehot_j.shape[1])

                pred_mask_processed = torch.softmax(pred_masks_logits_j, dim=0)
                pred_mask_processed = torch.argmax(pred_mask_processed, dim=0)
                pred_mask_processed = F.one_hot(pred_mask_processed).permute(1, 0).to(
                    device=pred_masks_logits_j.device
                )
                pred_mask_processed = (pred_mask_processed > 0.49).to(dtype=torch.float32)
                gt_mask_processed = (instance_mask_gt_onehot_j > 0.5).to(dtype=torch.float32)

                gt_mask_size = torch.sum(gt_mask_processed, dim=0)
                filtered_iou_local = []
                for k in range(instance_mask_gt_onehot_j.shape[1]):
                    if gt_mask_size[k] <= MIN_INSTANCE_SIZE:
                        continue

                    mask_bool_cpu = instance_mask_gt_onehot_j[:, k].bool().cpu()
                    class_k = class_ids_gt[j][mask_bool_cpu]
                    class_mode = torch.mode(class_k, dim=0)[0].item() if len(class_k) > 0 else -1

                    gt_mask_k = gt_mask_processed[:, k]
                    max_iou = 0.0
                    for i in range(pred_mask_processed.shape[0]):
                        pred_mask_i = pred_mask_processed[i]
                        intersection = torch.sum(pred_mask_i * gt_mask_k)
                        union = torch.sum(pred_mask_i) + torch.sum(gt_mask_k) - intersection
                        iou = float(intersection) / float(union) if union != 0 else 0.0
                        if iou > max_iou:
                            max_iou = iou

                    if class_mode not in [16, 26, 27, 28, 29]:
                        filtered_iou_local.append(max_iou)
                    iou_dict[class_mode].append(max_iou)

                    speed = flow_first[j][mask_bool_cpu]
                    speed_in_gt_mask = speed.norm(dim=1).mean(dim=0)
                    speed_dict[class_mode].append(speed_in_gt_mask.item())

                if filtered_iou_local:
                    filtered_miou_list.append(np.mean(filtered_iou_local))

            processed_batches += 1
            processed_samples += len(pred_masks_logits)

    elapsed = time.time() - start_time

    result = {
        "checkpoint": ckpt_path,
        "num_batches": processed_batches,
        "num_samples": processed_samples,
        "static_ratio": float(static_count / max(1, processed_samples)),
        "miou_mean": float(np.mean(miou_list)) if miou_list else 0.0,
        "miou_std": float(np.std(miou_list)) if miou_list else 0.0,
        "miou_min": float(np.min(miou_list)) if miou_list else 0.0,
        "miou_max": float(np.max(miou_list)) if miou_list else 0.0,
        "miou_median": float(np.median(miou_list)) if miou_list else 0.0,
        "filtered_miou_mean": float(np.mean(filtered_miou_list)) if filtered_miou_list else 0.0,
        "valid_mask_mean": float(np.mean(valid_mask_counts)) if valid_mask_counts else 0.0,
        "valid_mask_std": float(np.std(valid_mask_counts)) if valid_mask_counts else 0.0,
        "inference_time_sec": float(elapsed),
        "samples_per_sec": float(processed_samples / elapsed) if elapsed > 0 else 0.0,
        "class_stats": [],
    }

    for class_id, iou_values in iou_dict.items():
        class_name = CATEGORY_MAP.get(class_id, f"CLASS_{class_id}")
        speed_values = speed_dict.get(class_id, [])
        result["class_stats"].append(
            {
                "class_id": int(class_id),
                "class_name": class_name,
                "num_instances": len(iou_values),
                "iou_mean": float(np.mean(iou_values)) if iou_values else 0.0,
                "speed_mean": float(np.mean(speed_values)) if speed_values else 0.0,
            }
        )

    return result


def main():
    dataset_size = len(av2_sceneflow_zoo_flow)
    print(f"Dataset size: {dataset_size}. Sequentially evaluating {len(checkpoint_lists)} checkpoints...")

    all_results = []
    for ckpt_path in checkpoint_lists:
        if not Path(ckpt_path).exists():
            print(f"[Warning] Checkpoint not found, skipping: {ckpt_path}")
            continue
        result = evaluate_checkpoint(ckpt_path)
        all_results.append(result)

    OUTPUT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Saved evaluation results to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()

