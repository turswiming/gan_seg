#!/usr/bin/env python3
"""
Sequential checkpoint evaluation for checkmask.
Evaluate multiple checkpoints and save mIoU, per-category IoU, and speed to JSON.
"""

import torch
import json
import time
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict
from tqdm import tqdm

src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import CATEGORY_MAP
from bucketed_scene_flow_eval.datasets.argoverse2.av2_metacategories import BUCKETED_METACATAGORIES
from utils.metrics import calculate_miou
from omegaconf import OmegaConf

META_CATEGORY_NAMES = {
    0: "BACKGROUND", 1: "CAR", 2: "WHEELED_VRU",
    3: "OTHER_VEHICLES", 4: "PEDESTRIAN"
}

def get_meta_category_id(class_id):
    if class_id == -1:
        return 0
    class_name = CATEGORY_MAP.get(class_id, "BACKGROUND")
    if class_name in BUCKETED_METACATAGORIES["CAR"]:
        return 1
    elif class_name in BUCKETED_METACATAGORIES["WHEELED_VRU"]:
        return 2
    elif class_name in BUCKETED_METACATAGORIES["OTHER_VEHICLES"]:
        return 3
    elif class_name in BUCKETED_METACATAGORIES["PEDESTRIAN"]:
        return 4
    return 0

def load_model(config, ckpt_path):
    sys.path.append(str(src_dir / "models"))
    from KinematicModel import KinematicModel

    model = KinematicModel(config)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()
    return model

def evaluate_checkpoint(model, dataset, num_samples=100):
    category_iou = defaultdict(list)
    miou_values = []
    total_time = 0

    for i in tqdm(range(min(num_samples, len(dataset))), desc="Processing", ncols=80):
        sample = dataset[i]
        pc1 = sample["point_cloud_first"].unsqueeze(0).cuda()
        pc2 = sample["point_cloud_next"].unsqueeze(0).cuda()
        class_ids = sample["class_ids"]

        meta_labels = torch.tensor([get_meta_category_id(int(cid)) for cid in class_ids]).cuda()

        start = time.time()
        with torch.no_grad():
            est_flow, _ = model(pc1, pc2, class_ids.unsqueeze(0).cuda(),
                              torch.ones(1, pc1.shape[1]).cuda())
            warped = pc1.cpu() + est_flow.cpu()
            miou = calculate_miou(warped, pc2.cpu(), meta_labels.cpu(), threshold=0.5)
        total_time += time.time() - start

        miou_values.append(float(miou))

        for cat_id, cat_name in META_CATEGORY_NAMES.items():
            mask = (meta_labels == cat_id)
            if mask.sum() > 0:
                category_iou[cat_name].append(float(miou))

    results = {
        "miou": float(np.mean(miou_values)),
        "per_category_iou": {cat: float(np.mean(vals)) for cat, vals in category_iou.items()},
        "speed": total_time / len(miou_values),
        "num_samples": len(miou_values)
    }

    return results

def main():
    print("="*80)
    print("CHECKPOINT EVALUATION")
    print("="*80)

    # Configuration
    config_path = Path("/workspace/gan_seg/src/config/network_config.yaml")
    checkpoints_dir = Path("/workspace/gan_seg/outputs/exp/20251117_121131/checkpoints")
    output_dir = Path("/workspace/gan_seg/src/evaluation_results")
    output_dir.mkdir(exist_ok=True)

    # Define checkpoints to evaluate
    checkpoint_list = [
        checkpoints_dir / f"step_{step}.pt"
        for step in [12000, 14000, 16000, 17400]
    ]
    checkpoint_list = [ckpt for ckpt in checkpoint_list if ckpt.exists()]

    if not checkpoint_list:
        print("No checkpoints found, searching directory...")
        checkpoint_list = sorted(checkpoints_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime)

    # Load config and dataset
    config = OmegaConf.load(config_path)
    dataset = AV2SceneFlowZoo(
        root_dir=Path("/workspace/av2data/val"),
        flow_data_path=Path("/workspace/av2flow/val"),
        range_crop_type="ego",
        cache_root=Path("/workspace/av2data/cache"),
    )

    print(f"Found {len(checkpoint_list)} checkpoints\n")

    all_results = {}

    for ckpt_path in checkpoint_list:
        name = ckpt_path.stem
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        model = load_model(config, ckpt_path)
        results = evaluate_checkpoint(model, dataset, num_samples=100)

        all_results[name] = results

        print(f"\nmIoU: {results['miou']:.4f}")
        print(f"Speed: {results['speed']:.4f} sec/sample")
        print("\nPer-category IoU:")
        for cat, iou in results['per_category_iou'].items():
            print(f"  {cat}: {iou:.4f}")

    # Save results
    output_file = output_dir / "checkpoints_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}\n")

    # Print summary
    print("Summary:")
    print(f"{'Checkpoint':<20} {'mIoU':<8} {'Speed (s/sample)':<16}")
    print("-"*80)

    for name, res in all_results.items():
        print(f"{name:<20} {res['miou']:<8.4f} {res['speed']:<16.4f}")

if __name__ == "__main__":
    main()
