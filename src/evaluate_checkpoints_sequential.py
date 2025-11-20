#!/usr/bin/env python3
"""
Sequential evaluation of multiple checkpoints.
Evaluate each checkpoint in order and save metrics to JSON file.

Metrics saved:
- mIoU (mean Intersection over Union)
- Per-category IoU (CAR, WHEELED_VRU, OTHER_VEHICLES, PEDESTRIAN, BACKGROUND)
- Inference speed (seconds per sample)
"""

import torch
import numpy as np
import json
from pathlib import Path
import sys
import time
from collections import defaultdict
from omegaconf import OmegaConf
from tqdm import tqdm

# Add src directory to path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import CATEGORY_MAP
from bucketed_scene_flow_eval.datasets.argoverse2.av2_metacategories import BUCKETED_METACATAGORIES
from utils.forward_utils import augment_transform
from utils.metrics import calculate_miou
from utils.visualization_utils import remap_instance_labels
import torch.nn.functional as F


# Meta category mapping
META_CATEGORY_NAMES = {
    0: "BACKGROUND",
    1: "CAR",
    2: "WHEELED_VRU",
    3: "OTHER_VEHICLES",
    4: "PEDESTRIAN"
}


def get_meta_category_id(class_id):
    """Map class_id to meta category ID"""
    if class_id == -1:
        return 0  # BACKGROUND
    class_name = CATEGORY_MAP.get(class_id, "BACKGROUND")

    if class_name in BUCKETED_METACATAGORIES["CAR"]:
        return 1  # CAR
    elif class_name in BUCKETED_METACATAGORIES["WHEELED_VRU"]:
        return 2  # WHEELED_VRU
    elif class_name in BUCKETED_METACATAGORIES["OTHER_VEHICLES"]:
        return 3  # OTHER_VEHICLES
    elif class_name in BUCKETED_METACATAGORIES["PEDESTRIAN"]:
        return 4  # PEDESTRIAN
    else:
        return 0  # BACKGROUND


def load_model(network_config, checkpoint_path):
    """Load model from checkpoint"""
    print(f"  Loading model from {checkpoint_path}...")

    # Import model class
    sys.path.append(str(src_dir / "models"))
    from KinematicModel import KinematicModel

    model = KinematicModel(network_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.cuda()

    print(f"  Model loaded successfully!")
    return model


def evaluate_single_checkpoint(model, dataset, network_config, num_samples=100):
    """Evaluate a single checkpoint"""

    results = {
        "iou_per_sample": [],
        "miou": 0.0,
        "per_category_iou": {},
        "speed": 0.0,  # seconds per sample
        "num_samples": 0
    }

    # Category tracking
    category_iou_sum = defaultdict(float)
    category_count = defaultdict(int)

    # Speed measurement
    total_time = 0.0

    # Evaluate samples
    num_eval = min(num_samples, len(dataset))
    print(f"  Evaluating {num_eval} samples...")

    for i in tqdm(range(num_eval), desc="  Processing", ncols=80):
        sample = dataset[i]

        # Prepare input data
        pc1 = sample["point_cloud_first"].unsqueeze(0).cuda()  # [1, N, 3]
        pc2 = sample["point_cloud_next"].unsqueeze(0).cuda()   # [1, N, 3]
        class_ids = sample["class_ids"]  # [N]

        # Convert class_ids to meta categories
        meta_labels = torch.tensor([get_meta_category_id(int(cid)) for cid in class_ids]).cuda()  # [N]

        # Inference
        start_time = time.time()

        with torch.no_grad():
            # Forward pass
            est_flow, confidence = model(pc1, pc2, class_ids.unsqueeze(0).cuda(), torch.ones(1, pc1.shape[1]).cuda())

            # Move data to CPU for metrics calculation
            est_flow = est_flow.cpu()
            pc1_cpu = pc1.cpu()
            meta_labels_cpu = meta_labels.cpu()

            # Warp points
            warped_pc = pc1_cpu + est_flow

            # Calculate IoU using the model's built-in forward_and_compute_mask_iou
            if hasattr(model, "forward_and_compute_mask_iou"):
                model.cpu()
                try:
                    iou_dict = model.forward_and_compute_mask_iou(
                        pc1_cpu, pc2.cpu(), meta_labels_cpu, est_flow
                    )
                    miou = iou_dict.get("miou", 0.0)
                    per_category_iou = iou_dict.get("per_category_iou", {})
                except:
                    # Fallback to direct calculation
                    miou = calculate_miou(warped_pc, pc2.cpu(), meta_labels_cpu, threshold=0.5)
                    per_category_iou = {}
                model.cuda()
            else:
                # Fallback
                miou = calculate_miou(warped_pc, pc2.cpu(), meta_labels_cpu, threshold=0.5)
                per_category_iou = {}

        end_time = time.time()
        inference_time = end_time - start_time
        total_time += inference_time

        # Store results
        results["iou_per_sample"].append({
            "sample_idx": i,
            "miou": float(miou),
            "per_category_iou": per_category_iou,
            "inference_time": inference_time
        })

        # Accumulate category IoUs
        for cat_name, iou_val in per_category_iou.items():
            category_iou_sum[cat_name] += iou_val
            category_count[cat_name] += 1

    # Calculate average mIoU
    results["miou"] = float(np.mean([r["miou"] for r in results["iou_per_sample"]]))
    results["num_samples"] = num_eval

    # Calculate per-category IoU
    for cat_name in category_iou_sum:
        results["per_category_iou"][cat_name] = float(category_iou_sum[cat_name] / category_count[cat_name])

    # Calculate average speed
    results["speed"] = float(total_time / num_eval)

    return results


def find_checkpoints(checkpoints_dir):
    """Find all checkpoint files in directory"""
    checkpoints = []

    # Common checkpoint patterns
    patterns = [
        "*.pt",
        "*.pth",
        "*.pkl",
        "checkpoint*",
    ]

    for pattern in patterns:
        checkpoints.extend(checkpoints_dir.glob(pattern))

    # Sort by modification time (newest first)
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # Also look in subdirectories
    for subdir in checkpoints_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("checkpoint"):
            checkpoints.extend(subdir.glob("*.pt"))

    return sorted(list(set(checkpoints)), key=lambda x: x.name)


def save_results_to_json(results, output_path):
    """Save evaluation results to JSON file"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {output_path}")


def main():
    """Main evaluation function"""
    print("=" * 80)
    print("CHECKPOINT SEQUENTIAL EVALUATION")
    print("=" * 80)

    # Configuration
    # =============
    # Dataset paths
    root_dir = Path("/workspace/av2data/val")
    flow_data_path = Path("/workspace/av2flow/val")

    # Model configuration
    network_config_path = Path("/workspace/gan_seg/src/config/network_config.yaml")

    # Checkpoints directory
    checkpoints_dir = Path("/workspace/gan_seg/outputs/exp/20251117_121131/checkpoints")

    # Output directory
    output_dir = Path("/workspace/gan_seg/src/evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Evaluation settings
    num_samples = 100  # Number of samples to evaluate per checkpoint

    # Load network configuration
    print(f"\nLoading network config from {network_config_path}...")
    network_config = OmegaConf.load(network_config_path)

    # Load dataset
    print(f"\nLoading validation dataset...")
    dataset = AV2SceneFlowZoo(
        root_dir=root_dir,
        expected_camera_shape=(194, 256, 3),
        eval_args={},
        with_rgb=False,
        flow_data_path=flow_data_path,
        range_crop_type="ego",
        subsequence_length=2,
        sliding_window_step_size=1,
        with_ground=False,
        use_gt_flow=True,
        eval_type="bucketed_epe",
        load_flow=True,
        load_boxes=False,
        cache_root=Path("/workspace/av2data/cache"),
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Find checkpoints
    print(f"\nSearching for checkpoints in {checkpoints_dir}...")
    checkpoints = find_checkpoints(checkpoints_dir)

    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Found {len(checkpoints)} checkpoints:")
    for i, ckpt in enumerate(checkpoints, 1):
        print(f"  {i}. {ckpt.name}")

    # Evaluate each checkpoint
    all_results = {}

    print(f"\n{'=' * 80}")
    print("STARTING EVALUATION")
    print(f"{'=' * 80}\n")

    for checkpoint_path in checkpoints:
        checkpoint_name = checkpoint_path.stem

        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {checkpoint_name}")
        print(f"{'=' * 80}")

        try:
            # Load model
            model = load_model(network_config, checkpoint_path)

            # Evaluate
            results = evaluate_single_checkpoint(model, dataset, network_config, num_samples)

            # Store results
            all_results[checkpoint_name] = results

            # Print summary
            print(f"\n  Results Summary:")
            print(f"    mIoU: {results['miou']:.4f}")
            print(f"    Speed: {results['speed']:.4f} sec/sample")
            print(f"    Per-category IoU:")
            for cat, iou in results["per_category_iou"].items():
                print(f"      {cat}: {iou:.4f}")

            # Save intermediate results
            output_path = output_dir / f"{checkpoint_name}_results.json"
            save_results_to_json(results, output_path)

        except Exception as e:
            print(f"  Error evaluating {checkpoint_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Save all results
    print(f"\n{'=' * 80}")
    print("SAVING ALL RESULTS")
    print(f"{'=' * 80}")

    final_output_path = output_dir / "all_checkpoints_evaluation.json"
    with open(final_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {final_output_path}")

    # Print summary table
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLE")
    print(f"{'=' * 80}\n")

    print(f"{'Checkpoint':<30} {'mIoU':<8} {'Speed (sec/sample)':<18} {'Samples'}")
    print("-" * 80)

    for checkpoint_name, results in all_results.items():
        print(f"{checkpoint_name:<30} {results['miou']:<8.4f} {results['speed']:<18.4f} {results['num_samples']}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
