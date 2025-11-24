# This code is write by ai, I haven`t check it, the results can only used for reference
# dont put any results in your paper!

from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf
from utils.forward_utils import augment_transform, forward_scene_flow
from utils.metrics import calculate_epe, calculate_recall_at_threshold
from utils.visualization_utils import remap_instance_labels
import torch.nn.functional as F
from dataset.av2_dataset import AV2SequenceDataset

min_instance_size = 50
collect_fn = lambda x: x

av2sequence_dataset = AV2SequenceDataset(
    max_k=1,
    fix_ego_motion=True,
    apply_ego_motion=True,
    motion_threshold=0.05,
)
from torch.utils.data import DataLoader

generator = torch.Generator().manual_seed(42)
flow_dataloader = DataLoader(av2sequence_dataset, batch_size=1, shuffle=True, generator=generator, collate_fn=collect_fn)
# 加载配置文件
config_path = "/workspace/gan_seg/src/config/ablation_loss.yaml"
from utils.config_utils import load_config_with_inheritance
config = load_config_with_inheritance(config_path)
worked_path = "/workspace/gan_seg/outputs/ablation/fix_loss_contribution/without_dynamic/run_0/checkpoints/step_400.pt"
ckpt_path = worked_path
from Predictor import get_scene_flow_predictor

flow_predictor = get_scene_flow_predictor(config.model.flow, None)
ckpt = torch.load(ckpt_path, map_location="cpu")
# 使用 strict=False 允许缺少某些键（模型结构可能发生变化）
missing_keys, unexpected_keys = flow_predictor.load_state_dict(
    ckpt.get("flow_predictor", ckpt.get("scene_flow_predictor", {})), strict=False
)
if missing_keys:
    print(f"Warning: {len(missing_keys)} keys are missing from checkpoint:")
    if len(missing_keys) <= 20:
        print("Missing keys:", missing_keys)
    else:
        print("First 20 missing keys:", missing_keys[:20])
        print(f"... and {len(missing_keys) - 20} more")
if unexpected_keys:
    print(f"Warning: {len(unexpected_keys)} unexpected keys in checkpoint:")
    if len(unexpected_keys) <= 10:
        print("Unexpected keys:", unexpected_keys)
    else:
        print("First 10 unexpected keys:", unexpected_keys[:10])
flow_predictor.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
flow_predictor = flow_predictor.to(device)

# Flow evaluation metrics
epe_list = []  # 存储每个样本的EPE值
recall_5cm_list = []  # 存储每个样本的5cm recall
recall_10cm_list = []  # 存储每个样本的10cm recall
max_count = 10
count = 0
static_count = 0
with torch.no_grad():
    for sample in flow_dataloader:
        # 获取flow数据（包含class_id）
        print(f"processing sample {count}")
        pc_first = [s["point_cloud_first"] for s in sample]
        pc_next = [s["point_cloud_next"] for s in sample]  # (N, 3)
        flow_first = [s["flow"] for s in sample]
        # class_ids_gt = sample["class_ids"]  # (N,)
        valid_mask_first = [s["valid_mask_first"] for s in sample]  # (N,)
        valid_mask_next = [s["valid_mask_next"] for s in sample]  # (N,)
        instance_mask_gt = [s.get("mask") for s in sample]
        pc_first_aug = []
        pc_next_aug = []
        flow_dummy = []
        for j in range(len(pc_first)):
            pc_first_j = pc_first[j]
            pc_next_j = pc_next[j]
            valid_mask_first_j = valid_mask_first[j]
            valid_mask_next_j = valid_mask_next[j]
            instance_mask_gt_j = instance_mask_gt[j]
            aug_params = config.training.augment_params
            # no scale
            aug_params.scale = [1, 1]
            aug_params.rotation = [0, 0]
            aug_params.translation_range = [0, 0, 0]
            aug_params.mirror_x = False
            aug_params.mirror_z = False
            pc_first_aug_j, pc_next_aug_j, flow_dummy_j, _ = augment_transform(
                pc_first_j.clone().to(device),
                pc_next_j.clone().to(device),  # 使用相同的点云作为pc2（因为只需要augment pc_first）
                torch.zeros(pc_first_j.shape[0], 3).to(device),  # dummy flow
                None,  # cascade_flow_outs
                config.training.augment_params,
            )
            pc_first_aug.append(pc_first_aug_j)
            pc_next_aug.append(pc_next_aug_j)
            flow_dummy.append(flow_dummy_j)
        # Convert to lists (forward_scene_flow expects lists)
        pc_first_aug_list = [pc for pc in pc_first_aug]
        pc_next_aug_list = [pc for pc in pc_next_aug]

        # Predict scene flow
        from utils.forward_utils import forward_scene_flow_general
        valid_mask_first_list = [valid_mask.to(device) for valid_mask in valid_mask_first]
        valid_mask_next_list = [valid_mask.to(device) for valid_mask in valid_mask_next]
        pred_flows = forward_scene_flow_general(
            pc_first_aug_list, pc_next_aug_list, flow_predictor, "AV2Sequence", train_flow=False
        )

        # forward_scene_flow returns a tuple, extract the first element (pred_flow)
        if isinstance(pred_flows, tuple):
            pred_flows = pred_flows[0]

        # Calculate flow metrics for each sample
        for j in range(len(pred_flows)):
            pred_flow_j = pred_flows[j].to(device)
            gt_flow_j = flow_first[j].to(device)

            # Ensure same device and shape
            if pred_flow_j.shape != gt_flow_j.shape:
                min_len = min(pred_flow_j.shape[0], gt_flow_j.shape[0])
                pred_flow_j = pred_flow_j[:min_len]
                gt_flow_j = gt_flow_j[:min_len]

            # Calculate EPE for this sample
            epe = torch.norm(pred_flow_j - gt_flow_j, dim=1, p=2)
            mean_epe = torch.mean(epe)
            epe_list.append(mean_epe.item())

            # Calculate recall at 1cm (0.01m)
            # recall_1cm = calculate_recall_at_threshold([pred_flow_j], [gt_flow_j], threshold=0.01)
            # recall_1cm_list.append(recall_1cm)

            # Calculate recall at 5cm (0.05m)
            recall_5cm = calculate_recall_at_threshold([pred_flow_j], [gt_flow_j], threshold=0.05)
            recall_5cm_list.append(recall_5cm)
            recall_10cm = calculate_recall_at_threshold([pred_flow_j], [gt_flow_j], threshold=0.1)
            recall_10cm_list.append(recall_10cm)

        count += 1
        if count > max_count:
            break
print(f"static count: {static_count}")
print(f"static count ratio: {static_count / count}")
print(f"\n=== Flow Evaluation Metrics ===")
print(f"Mean EPE: {np.mean(epe_list):.6f} m")
print(f"Std EPE: {np.std(epe_list):.6f} m")
print(f"Max EPE: {np.max(epe_list):.6f} m")
print(f"Min EPE: {np.min(epe_list):.6f} m")
print(f"Median EPE: {np.median(epe_list):.6f} m")
print(f"Recall@5cm: {np.mean(recall_5cm_list):.4f}")
print(f"Recall@5cm std: {np.std(recall_5cm_list):.4f}")
print(f"Recall@10cm: {np.mean(recall_10cm_list):.4f}")
print(f"Recall@10cm std: {np.std(recall_10cm_list):.4f}")