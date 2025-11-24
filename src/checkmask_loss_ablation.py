# This code is write by ai, I haven`t check it, the results can only used for reference
# dont put any results in your paper!

from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf
from utils.forward_utils import augment_transform
from utils.metrics import calculate_miou, accumulate_eval_results, calculate_AP, calculate_PQ_F1
from utils.visualization_utils import remap_instance_labels
import torch.nn.functional as F
from dataset.av2_dataset import AV2SequenceDataset
min_instance_size = 50
av2sequence_dataset = AV2SequenceDataset(
    max_k=1,
    fix_ego_motion=True,
    apply_ego_motion=True,
    motion_threshold=0.05,
)
from dataset.movi_f_sequence_dataset import MOVIFSequenceDataset
movifsequence_dataset = MOVIFSequenceDataset(
    dataset_path="/workspace/movi/0",
    max_k=1,
    motion_threshold=0.05,
)
collect_fn = lambda x: x

def compute_rand_index(pred_labels: torch.Tensor, gt_labels: torch.Tensor, min_points: int = 0):
    """
    Compute Rand Index between predicted and ground-truth instance labels.
    Points belonging to GT instances smaller than min_points are ignored.
    """
    pred_labels = pred_labels.detach().cpu()
    gt_labels = gt_labels.detach().cpu()

    if min_points > 0:
        unique_ids, counts = torch.unique(gt_labels, return_counts=True)
        small_ids = unique_ids[counts < min_points]
        if small_ids.numel() > 0:
            for sid in small_ids:
                gt_labels[gt_labels == sid.item()] = -1

    valid_mask = gt_labels >= 0
    if valid_mask.sum() < 2:
        return None

    pred_valid = pred_labels[valid_mask]
    gt_valid = gt_labels[valid_mask]

    same_pred = pred_valid.unsqueeze(0) == pred_valid.unsqueeze(1)
    same_gt = gt_valid.unsqueeze(0) == gt_valid.unsqueeze(1)

    ri = (same_pred == same_gt).float().mean()
    return ri.item()

from torch.utils.data import DataLoader
generator = torch.Generator().manual_seed(42)
flow_dataloader = DataLoader(movifsequence_dataset, batch_size=1, shuffle=True, generator=generator, collate_fn=collect_fn)
# 加载配置文件
config_path = "/workspace/gan_seg/src/config/ablation_coord.yaml"
from utils.config_utils import load_config_with_inheritance
config = load_config_with_inheritance(config_path)
worked_path = "/workspace/gan_seg/output/ablation/coord/projection/checkpoints/step_400.pt"
from Predictor import get_mask_predictor
mask_predictor = get_mask_predictor(config.model.mask,10)
ckpt = torch.load(worked_path, map_location="cpu")
# 使用 strict=False 允许缺少某些键（模型结构可能发生变化）
missing_keys, unexpected_keys = mask_predictor.load_state_dict(ckpt["mask_predictor"], strict=False)
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
mask_predictor.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_predictor = mask_predictor.to(device)

# 统计不同class_id的mIoU
class_iou_stats = defaultdict(lambda: {'intersection': 0, 'union': 0, 'total_points': 0})
# 收集每个mask的IoU和flow幅度数据用于绘图
mask_iou_flow_data = []  # [(iou, flow_magnitude), ...]
# 统计有多个background instance的场景
scenes_with_multiple_bg = []  # [(sequence_id, index, bg_count, bg_details), ...]
# 使用calculate_miou计算的mIoU列表（与eval.py一致）
miou_list_calculate = []  # 存储每个样本的mIoU值
ri_values = []  # 存储Rand Index

valid_mask_num = []
# 间隔100采样，因为相邻150条数据在一个场景内，避免重复采样同一场景
dataset_size = len(av2sequence_dataset)

max_count = 100
count = 0
iou_dict = {}
speed_dict = {}
static_count = 0
filtered_miou_list = []

# AP metrics accumulation
ap_eval_meter = {'Pred_IoU': [], 'Pred_Matched': [], 'Confidence': [], 'N_GT_Inst': []}
with torch.no_grad():
    for sample in flow_dataloader:
        # 获取flow数据（包含class_id）
        print(f'processing sample {count}')
        pc_first = [s["point_cloud_first"] for s in sample]
        pc_next = [s["point_cloud_next"] for s in sample]  # (N, 3)
        flow_first = [s["flow"] for s in sample]
        # class_ids_gt = sample["class_ids"]  # (N,)
        valid_mask_first = [s["valid_mask_first"] for s in sample]  # (N,)
        valid_mask_next = [s["valid_mask_next"] for s in sample]  # (N,)
        instance_mask_gt = [s.get("mask") for s in sample]
        # for flow in sample["flow"]:
        #     flow_std = flow.std(dim=0).max()
        #     if flow_std < 0.01:
        #         static_count += 1
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
            #no scale 
            aug_params.scale = [1,1]
            aug_params.rotation = [0,0]
            aug_params.translation_range = [0,0,0]
            aug_params.mirror_x = False
            aug_params.mirror_z = False
            # pc_first_aug_j, pc_next_aug_j, flow_dummy_j, _ = augment_transform(
            #     pc_first_j.clone().to(device),
            #     pc_next_j.clone().to(device),  # 使用相同的点云作为pc2（因为只需要augment pc_first）
            #     torch.zeros(pc_first_j.shape[0], 3).to(device),  # dummy flow
            #     None,  # cascade_flow_outs
            #     config.training.augment_params
            # )
            pc_first_aug.append(pc_first_j)
            pc_next_aug.append(pc_next_j)
            flow_dummy.append(flow_first[j])
        # pc_first_aug = torch.stack(pc_first_aug)
        # pc_next_aug = torch.stack(pc_next_aug)
        # flow_dummy = torch.stack(flow_dummy)
        from utils.forward_utils import forward_mask_prediction_general
        pred_masks_logits = []
        for i in range(len(pc_first_aug)):
            pc_first_aug_i = pc_first_aug[i].unsqueeze(0)
            pc_next_aug_i = pc_next_aug[i].unsqueeze(0)
            flow_dummy_i = flow_dummy[i].unsqueeze(0)
            pred_masks_logits_i = forward_mask_prediction_general(pc_first_aug_i.clone().to(device), mask_predictor)
            pred_masks_logits.append([p.squeeze(0) for p in pred_masks_logits_i])
        
        # Prepare data for AP metrics
        batch_segm = []
        batch_mask = []
        
        for j in range(len(pred_masks_logits)):
            pred_masks_logits_j = pred_masks_logits[j][0]
            pred_masks_logits_argmax_j = torch.argmax(pred_masks_logits_j, dim=0)
            pred_mask_j = F.one_hot(pred_masks_logits_argmax_j).permute(1, 0).float()
            instance_mask_gt_j = instance_mask_gt[j].long()
            instance_mask_gt_onehot_j = F.one_hot(instance_mask_gt_j).to(device=pred_masks_logits_j.device)
            
            # Calculate mIoU
            miou_value = calculate_miou(pred_masks_logits_j, instance_mask_gt_onehot_j.permute(1, 0),min_points=min_instance_size)
            miou_list_calculate.append(miou_value)
            
            # Prepare for AP metrics: convert to instance label format
            # segm: (N,) instance labels starting from 0
            # mask: (K, N) -> (N, K) for accumulate_eval_results
            # Apply softmax to convert logits to probabilities for confidence calculation
            segm_j = instance_mask_gt_j.cpu()
            mask_j = F.softmax(pred_masks_logits_j, dim=0).permute(1, 0).cpu()  # (N, K) probabilities
            batch_segm.append(segm_j)
            batch_mask.append(mask_j)

            ri_score = compute_rand_index(pred_masks_logits_argmax_j.cpu(), segm_j, min_instance_size)
            if ri_score is not None:
                ri_values.append(ri_score)
        
        # Accumulate AP metrics for this batch
        if len(batch_segm) > 0:
            batch_segm_tensor = torch.stack(batch_segm)  # (B, N)
            batch_mask_tensor = torch.stack(batch_mask)  # (B, N, K)
            Pred_IoU, Pred_Matched, Confidence, N_GT_Inst = accumulate_eval_results(
                batch_segm_tensor, batch_mask_tensor, ignore_npoint_thresh=min_instance_size
            )
            ap_eval_meter['Pred_IoU'].append(Pred_IoU)
            ap_eval_meter['Pred_Matched'].append(Pred_Matched)
            ap_eval_meter['Confidence'].append(Confidence)
            ap_eval_meter['N_GT_Inst'].append(N_GT_Inst)
            
        count += 1
        if count > max_count:
            break
print(f"static count: {static_count}")
print(f"static count ratio: {static_count / count}")
print(f"mean miou: {np.mean(miou_list_calculate)}")
print(f"std miou: {np.std(miou_list_calculate)}")
print(f"max miou: {np.max(miou_list_calculate)}")
print(f"min miou: {np.min(miou_list_calculate)}")
print(f"median miou: {np.median(miou_list_calculate)}")
if len(ri_values) > 0:
    print(f"mean RI: {np.mean(ri_values)}")


# Calculate and print AP metrics
if len(ap_eval_meter['Pred_IoU']) > 0:
    Pred_IoU = np.concatenate(ap_eval_meter['Pred_IoU'])
    Pred_Matched = np.concatenate(ap_eval_meter['Pred_Matched'])
    Confidence = np.concatenate(ap_eval_meter['Confidence'])
    N_GT_Inst = np.sum(ap_eval_meter['N_GT_Inst'])
    
    AP = calculate_AP(Pred_Matched, Confidence, N_GT_Inst, plot=False)
    print(f'AveragePrecision@50: {AP:.4f}')
    
    PQ, F1, Pre, Rec = calculate_PQ_F1(Pred_IoU, Pred_Matched, N_GT_Inst)
    print(f'PanopticQuality@50: {PQ:.4f}')
    print(f'F1-score@50: {F1:.4f}')
    print(f'Precision@50: {Pre:.4f}')
    print(f'Recall@50: {Rec:.4f}')
else:
    print("No AP metrics collected (no valid predictions)")
