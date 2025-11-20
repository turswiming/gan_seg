# This code is write by ai, I haven`t check it, the results can only used for reference
# dont put any results in your paper!

from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
from pathlib import Path
import torch
import numpy as np
from collections import defaultdict
from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import CATEGORY_MAP
from bucketed_scene_flow_eval.datasets.argoverse2.av2_metacategories import BUCKETED_METACATAGORIES
from omegaconf import OmegaConf
from utils.forward_utils import augment_transform
from utils.metrics import calculate_miou
from utils.visualization_utils import remap_instance_labels
import torch.nn.functional as F

# 建立class_id到元类别的映射
def get_meta_category_id(class_id):
    """将class_id映射到元类别ID"""
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
        return 0  # BACKGROUND (包括ROAD_SIGNS等)

# 元类别名称映射
META_CATEGORY_NAMES = {
    0: "BACKGROUND",
    1: "CAR",
    2: "WHEELED_VRU",
    3: "OTHER_VEHICLES",
    4: "PEDESTRIAN"
}
min_instance_size = 50
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
    min_instance_size=min_instance_size,
    cache_root=Path("/tmp/val_flow_cache/"),
)
from torch.utils.data import DataLoader
generator = torch.Generator().manual_seed(42)
flow_dataloader = DataLoader(av2_sceneflow_zoo_flow, batch_size=10, shuffle=True, generator=generator)
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
    min_instance_size=min_instance_size,
    cache_root=Path("/tmp/val_mask_cache/"),

)
generator = torch.Generator().manual_seed(42)   
mask_dataloader = DataLoader(av2_sceneflow_zoo_mask, batch_size=10, shuffle=True, generator=generator)
# 加载配置文件
config_path = "//workspace/gan_seg/outputs/exp/20251111_132106run3/config.yaml"
config = OmegaConf.load(config_path)
config.model.mask.slot_num = 20
checkpoint_lists = [
    "/workspace/gan_seg/outputs/exp/20251111_080942run2/checkpoints/step_6000.pt",
    "/workspace/gan_seg/outputs/exp/20251111_080942run2/checkpoints/step_9900.pt",
    "/workspace/gan_seg/outputs/exp/20251111_132106run3/checkpoints/step_20000.pt",
    "/workspace/gan_seg/outputs/exp/20251111_132106run3/checkpoints/step_30000.pt",
    "/workspace/gan_seg/outputs/exp/20251112_142843run4/checkpoints/step_40000.pt",
    "/workspace/gan_seg/outputs/exp/20251112_142843run4/checkpoints/step_50000.pt",
    "/workspace/gan_seg/outputs/exp/20251112_142843run4/checkpoints/step_60000.pt"
]
worked_path = "/workspace/gan_seg/outputs/exp/20251112_142843run4/checkpoints/step_63000.pt"
ckpt_path = "/workspace/gan_seg/outputs/exp/20251119_201006/checkpoints/step_2000.pt"
ckpt_path = worked_path
from OGCModel.segnet_av2 import MaskFormer3D
from model.ptv3_mask_predictor import PTV3MaskPredictor
from Predictor import get_mask_predictor
mask_predictor = get_mask_predictor(config.model.mask,10)
ckpt = torch.load(ckpt_path, map_location="cpu")
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

valid_mask_num = []
# 间隔100采样，因为相邻150条数据在一个场景内，避免重复采样同一场景
dataset_size = len(av2_sceneflow_zoo_flow)

max_count = 10
count = 0
print(f"推理 {max_count} 条数据（间隔100采样，共{dataset_size}条数据），计算不同class_id的mIoU...")
print("="*80)
iou_dict = {}
speed_dict = {}
static_count = 0
filtered_miou_list = []
with torch.no_grad():
    for sample_flow, sample_mask in zip(flow_dataloader, mask_dataloader):
        # 获取flow数据（包含class_id）
        print(f'processing sample {count}')
        pc_first_flow = sample_flow["point_cloud_first"]  # (N, 3)
        flow_first = sample_flow["flow"]
        class_ids_gt = sample_flow["class_ids"]  # (N,)
        pc_first_mask = sample_mask["point_cloud_first"]  # (N, 3)
        instance_mask_gt = sample_mask.get("mask")
        for flow in sample_flow["flow"]:
            flow_std = flow.std(dim=0).max()
            if flow_std < 0.01:
                static_count += 1
        pc_first_aug = []
        pc_next_aug = []
        flow_dummy = []
        for j in range(len(pc_first_mask)):
            pc_first_mask_j = pc_first_mask[j]
            instance_mask_gt_j = instance_mask_gt[j]
            aug_params = config.training.augment_params
            #no scale 
            aug_params.scale = [1,1]
            aug_params.rotation = [0,0]
            aug_params.translation_range = [0,0,0]
            aug_params.mirror_x = False
            aug_params.mirror_z = False
            pc_first_aug_j, pc_next_aug_j, flow_dummy_j, _ = augment_transform(
                pc_first_mask_j.clone().to(device),
                pc_first_mask_j.clone().to(device),  # 使用相同的点云作为pc2（因为只需要augment pc_first）
                torch.zeros(pc_first_mask_j.shape[0], 3).to(device),  # dummy flow
                None,  # cascade_flow_outs
                config.training.augment_params
            )
            pc_first_aug.append(pc_first_aug_j)
            pc_next_aug.append(pc_next_aug_j)
            flow_dummy.append(flow_dummy_j)
        pc_first_aug = torch.stack(pc_first_aug)
        pc_next_aug = torch.stack(pc_next_aug)
        flow_dummy = torch.stack(flow_dummy)

        from utils.forward_utils import forward_mask_prediction_general
        pred_masks_logits = forward_mask_prediction_general(pc_first_aug, mask_predictor)
        for j in range(len(pred_masks_logits)):
            pred_masks_logits_j = pred_masks_logits[j]
            pred_masks_logits_argmax_j = torch.argmax(pred_masks_logits_j, dim=0)
            pred_mask_j = F.one_hot(pred_masks_logits_argmax_j).permute(1, 0).float()
            instance_mask_gt_j = instance_mask_gt[j]
            instance_mask_gt_onehot_j = F.one_hot(instance_mask_gt_j).to(device=pred_masks_logits_j.device)
            miou_value = calculate_miou(pred_masks_logits_j, instance_mask_gt_onehot_j.permute(1, 0),min_points=min_instance_size)
            miou_list_calculate.append(miou_value)
            valid_mask_num.append(instance_mask_gt_onehot_j.shape[1])
            # 计算每个类别的IoU，使用与calculate_miou相同的方式
            instance_mask_gt_onehot_j = instance_mask_gt_onehot_j.to(device=pred_masks_logits_j.device)
            # 使用与calculate_miou相同的处理方式
            pred_mask_processed = torch.softmax(pred_masks_logits_j, dim=0)
            pred_mask_processed = torch.argmax(pred_mask_processed, dim=0)
            pred_mask_processed = F.one_hot(pred_mask_processed).permute(1, 0).to(device=instance_mask_gt_onehot_j.device)
            pred_mask_processed = (pred_mask_processed > 0.49).to(dtype=torch.float32)
            gt_mask_processed = (instance_mask_gt_onehot_j > 0.5).to(dtype=torch.float32)
            filtered_iou_list = []
            # 计算每个GT instance的大小，过滤小instance（与calculate_miou一致）
            gt_mask_size = torch.sum(gt_mask_processed, dim=0)
            for k in range(instance_mask_gt_onehot_j.shape[1]):
                # 跳过小于min_instance_size的instance（与calculate_miou一致）
                if gt_mask_size[k] <= min_instance_size:
                    continue
                    
                instance_mask_gt_onehot_j = instance_mask_gt_onehot_j.to(device=class_ids_gt.device)
                class_k = class_ids_gt[j][instance_mask_gt_onehot_j[:, k].bool()]
                # 众数：使用 torch.mode() 求众数，返回 (values, indices)
                if len(class_k) > 0:
                    class_mode = torch.mode(class_k, dim=0)[0].item()  # 取 values 并转为 Python int
                else:
                    class_mode = -1  # 如果 instance 为空，返回 -1
                
                # 计算该GT instance与所有预测mask的最大IoU（与calculate_miou一致）
                max_iou = 0.0
                gt_mask_k = gt_mask_processed[:, k]  # (N,)
                for i in range(pred_mask_processed.shape[0]):
                    pred_mask_i = pred_mask_processed[i]  # (N,)
                    intersection = torch.sum(pred_mask_i * gt_mask_k)
                    union = torch.sum(pred_mask_i) + torch.sum(gt_mask_k) - intersection
                    iou = float(intersection) / float(union) if union != 0 else 0.0
                    if iou > max_iou:
                        max_iou = iou
                if class_mode not in [16,26,27,28,29]:
                    filtered_iou_list.append(max_iou)
                if class_mode not in iou_dict:
                    iou_dict[class_mode] = []
                speed_in_gt_mask = flow_first[j][instance_mask_gt_onehot_j[:, k].bool()].norm(dim=0).mean(dim=0)
                if class_mode not in speed_dict:
                    speed_dict[class_mode] = []
                speed_dict[class_mode].append(speed_in_gt_mask)
                iou_dict[class_mode].append(max_iou)
            if len(filtered_iou_list) > 0:
                filtered_miou_list.append(np.mean(filtered_iou_list))
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
print(f"valid mask num: {np.mean(valid_mask_num)}")
print(f"valid mask num std: {np.std(valid_mask_num)}")
print(f"valid mask num max: {np.max(valid_mask_num)}")
print(f"valid mask num min: {np.min(valid_mask_num)}")
print(f"valid mask num median: {np.median(valid_mask_num)}")
print(f"filtered miou: {np.mean(filtered_miou_list)}")
for key in iou_dict:
    classname = CATEGORY_MAP[key]
    print(f"class {classname},{len(iou_dict[key])} iou: {np.mean(iou_dict[key])}, speed: {np.mean(speed_dict[key])}")
