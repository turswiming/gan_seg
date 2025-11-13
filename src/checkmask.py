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

# 加载配置文件
config_path = "/workspace/gan_seg/src/config/general_av2.yaml"
config = OmegaConf.load(config_path)

ckpt_path = "/workspace/gan_seg/outputs/exp/20251112_142843/checkpoints/step_42300.pt"
from OGCModel.segnet_av2 import MaskFormer3D
mask_predictor = MaskFormer3D(
    n_slot=20,
    transformer_embed_dim=256,
    transformer_input_pos_enc=False,
    n_transformer_layer=2,
    n_point=8192,
)
mask_predictor.load_state_dict(torch.load(ckpt_path)["mask_predictor"])
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

# 间隔100采样，因为相邻150条数据在一个场景内，避免重复采样同一场景
dataset_size = len(av2_sceneflow_zoo_flow)
sample_indices = list(range(0, min(dataset_size, 50000), 100))  # 间隔100采样，最多500个场景
total_samples = len(sample_indices)

print(f"推理 {total_samples} 条数据（间隔100采样，共{dataset_size}条数据），计算不同class_id的mIoU...")
print("="*80)

with torch.no_grad():
    for idx, i in enumerate(sample_indices):
        try:
            # 清理GPU缓存，避免占用过多内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 获取flow数据（包含class_id）
            sample_flow = av2_sceneflow_zoo_flow[i]
            pc_first_flow = sample_flow["point_cloud_first"]  # (N, 3)
            class_ids_gt = sample_flow["class_ids"]  # (N,)
            
            # 获取mask数据（包含instance_id）
            sample_mask = av2_sceneflow_zoo_mask[i]
            pc_first_mask = sample_mask["point_cloud_first"]  # (N, 3)
            instance_mask_gt = sample_mask.get("mask")  # (N,) instance_id
            
            # 确保点云位置一致（因为下采样固定了）
            # 通过点云位置匹配来建立instance_id到class_id的映射
            if instance_mask_gt is not None:
                # 由于下采样固定，点云应该完全一致，可以直接使用索引
                # 验证点云是否一致
                if pc_first_flow.shape[0] == pc_first_mask.shape[0]:
                    # 点云数量一致，直接使用（因为下采样固定了）
                    instance_mask_matched = instance_mask_gt
                else:
                    # 如果数量不一致，使用最近邻匹配
                    pc_flow_np = pc_first_flow.cpu().numpy()
                    pc_mask_np = pc_first_mask.cpu().numpy()
                    from scipy.spatial.distance import cdist
                    distances = cdist(pc_flow_np, pc_mask_np)
                    matched_indices = np.argmin(distances, axis=1)
                    instance_mask_matched = instance_mask_gt[matched_indices]  # (N,)
                
                # 确保instance_mask_matched是tensor
                if isinstance(instance_mask_matched, np.ndarray):
                    instance_mask_matched = torch.from_numpy(instance_mask_matched)
                from utils.visualization_utils import remap_instance_labels
                instance_mask_matched = remap_instance_labels(instance_mask_matched)

                # 建立instance_id到class_id的映射
                unique_instances = torch.unique(instance_mask_matched)
                instance_to_class = {}
                for inst_id in unique_instances:
                    inst_mask = instance_mask_matched == inst_id
                    if inst_mask.sum() > 0:
                        # 获取该instance对应的class_id（取众数）
                        inst_class_ids = class_ids_gt[inst_mask]
                        if len(inst_class_ids) > min_instance_size:
                            # 转换为numpy取众数
                            inst_class_ids_np = inst_class_ids.cpu().numpy() if isinstance(inst_class_ids, torch.Tensor) else inst_class_ids
                            from scipy.stats import mode
                            most_common_class, _ = mode(inst_class_ids_np, keepdims=True)
                            instance_to_class[inst_id.item()] = most_common_class[0]
                
                # 应用augment_transform（使用配置文件中的参数）
                # config.training.augment_params["angle_range"] = [0,0]
                # config.training.augment_params["translation_range"] = [0,0,0]
                # config.training.augment_params["scale_range"] = [1,1]
                # config.training.augment_params["mirror_x"] = False
                # config.training.augment_params["mirror_z"] = False

                pc_first_aug, pc_next_aug, flow_dummy, _ = augment_transform(
                    pc_first_mask.to(device),
                    pc_first_mask.to(device),  # 使用相同的点云作为pc2（因为只需要augment pc_first）
                    torch.zeros(pc_first_mask.shape[0], 3).to(device),  # dummy flow
                    None,  # cascade_flow_outs
                    config.training.augment_params
                )
                
                # 推理预测mask（使用batch size=1，避免占用过多GPU内存）
                pc_first_tensor = pc_first_aug.unsqueeze(0)  # (1, N, 3)
                pc_first_tensor = pc_first_tensor.repeat(8, 1, 1)
                from utils.forward_utils import forward_mask_prediction_general
                pred_masks_logits = forward_mask_prediction_general(pc_first_tensor, mask_predictor)
                pred_masks_logits = pred_masks_logits[0].permute(1, 0)  # (N, K) - logits格式
                
                # 将预测mask转换为slot分配（用于后续的class映射和统计）
                pred_masks_softmax = torch.softmax(pred_masks_logits, dim=1)
                pred_slot = torch.argmax(pred_masks_softmax, dim=1)  # (N,)
                
                # 确保instance_mask_matched在CPU上（因为数据在CPU）
                instance_mask_matched_cpu = instance_mask_matched.cpu() if instance_mask_matched.is_cuda else instance_mask_matched
                pred_slot_cpu = pred_slot.cpu() if pred_slot.is_cuda else pred_slot
                pred_masks_logits_cpu = pred_masks_logits.cpu() if pred_masks_logits.is_cuda else pred_masks_logits
                
                # 确保class_ids_gt是tensor并在CPU上
                if isinstance(class_ids_gt, np.ndarray):
                    class_ids_gt = torch.from_numpy(class_ids_gt)
                class_ids_gt = class_ids_gt.cpu() if class_ids_gt.is_cuda else class_ids_gt
                
                # 使用KNN查询为pc_first_mask计算新的class_ids_gt
                # 找到pc_first_mask中每个点在pc_first_flow中的最近邻，使用最近邻的class_id
                from pointnet2.pointnet2 import knn
                
                # 确保点云在相同设备上
                device_knn = 'cuda' if torch.cuda.is_available() else 'cpu'
                pc_first_mask_knn = pc_first_mask.to(device_knn).contiguous()
                pc_first_flow_knn = pc_first_flow.to(device_knn).contiguous()
                class_ids_gt_knn = class_ids_gt.to(device_knn)
                
                # KNN需要批次维度: (B, N, 3)
                pc_first_mask_batch = pc_first_mask_knn.unsqueeze(0)  # [1, N_mask, 3]
                pc_first_flow_batch = pc_first_flow_knn.unsqueeze(0)  # [1, N_flow, 3]
                
                # 使用KNN查找最近邻 (k=1)
                k = 1
                dists, knn_indices = knn(k, pc_first_mask_batch, pc_first_flow_batch)  # dists: [1, N_mask, 1], idx: [1, N_mask, 1]
                knn_indices = knn_indices.squeeze(0).squeeze(-1).long()  # [N_mask]
                
                # 使用最近邻索引获取对应的class_ids
                class_ids_gt_mask = class_ids_gt_knn[knn_indices]  # [N_mask]
                
                # 移回CPU
                class_ids_gt_mask = class_ids_gt_mask.cpu()
                
                # 更新class_ids_gt为新的基于KNN的值
                class_ids_gt = class_ids_gt_mask
                
                #

                # 获取flow数据用于计算flow幅度
                flow_gt = sample_flow["flow"]  # (N, 3)
                if isinstance(flow_gt, torch.Tensor):
                    flow_gt = flow_gt.cpu() if flow_gt.is_cuda else flow_gt
                else:
                    flow_gt = torch.from_numpy(flow_gt) if isinstance(flow_gt, np.ndarray) else flow_gt
                flow_magnitude = torch.norm(flow_gt, dim=1)  # (N,)
                
                # 使用calculate_miou计算整体mIoU（与eval.py一致）
                # 准备GT mask为one-hot格式 [K_gt, N]，其中K_gt是GT instance数量
                # instance_mask_matched_cpu已经在前面通过remap_instance_labels处理过了
                gt_mask_onehot = F.one_hot(instance_mask_matched_cpu.to(torch.long)).permute(1, 0).to(device=device)  # [K_gt, N]
                
                # 准备pred mask为logits格式 [K_pred, N]，转置为[K_pred, N]
                pred_masks_for_miou = pred_masks_logits_cpu.permute(1, 0).to(device=device)  # [K_pred, N] - logits格式
                
                # 使用calculate_miou计算整体mIoU
                min_points = config.eval.min_points if hasattr(config.eval, 'min_points') else min_instance_size
                miou_value = calculate_miou(pred_masks_for_miou, gt_mask_onehot, min_points=min_points)
                
                # 如果mIoU计算成功，记录到列表（用于后续统计）
                if miou_value is not None:
                    miou_list_calculate.append(miou_value.item())
                
                # 同时保留自定义的每个instance IoU计算，用于flow magnitude分析和绘图
                # 将pred_slot转换为one-hot格式 [K, N]，其中K是slot数量
                num_slots = config.model.mask.slot_num  # 从配置文件读取slot数量
                pred_mask_onehot = torch.nn.functional.one_hot(pred_slot_cpu, num_classes=num_slots).permute(1, 0).float()  # [K, N]
                pred_mask_onehot = (pred_mask_onehot > 0.49).float()
                
                # 获取所有GT instance
                unique_gt_instances = torch.unique(instance_mask_matched_cpu)
                
                # 统计background instance信息（用于调试）
                background_instances = []
                foreground_instances = []
                for gt_inst_id in unique_gt_instances:
                    gt_inst_id = gt_inst_id.item()
                    gt_instance_mask = (instance_mask_matched_cpu == gt_inst_id).float()  # [N]
                    gt_size = gt_instance_mask.sum().item()
                    
                    # 跳过太小的instance（min_points=50）
                    if gt_size < min_instance_size:
                        continue
                    
                    # 获取该GT instance的class_id
                    if gt_inst_id in instance_to_class:
                        inst_class_id = instance_to_class[gt_inst_id]
                    else:
                        inst_class_id = -1  # 背景类
                    
                    if inst_class_id == -1:
                        background_instances.append({
                            'instance_id': gt_inst_id,
                            'size': gt_size,
                            'class_id': inst_class_id
                        })
                    else:
                        foreground_instances.append({
                            'instance_id': gt_inst_id,
                            'size': gt_size,
                            'class_id': inst_class_id
                        })
                
                # 先计算所有GT instance的flow magnitude
                gt_flow_magnitudes = {}
                for gt_inst_id in unique_gt_instances:
                    gt_inst_id = gt_inst_id.item()
                    gt_instance_mask = (instance_mask_matched_cpu == gt_inst_id).float()  # [N]
                    gt_size = gt_instance_mask.sum().item()
                    
                    # 跳过太小的instance（min_points=50）
                    if gt_size < min_instance_size:
                        continue
                    
                    # 计算该GT instance区域内gt flow的平均幅度
                    gt_flow_magnitude = flow_magnitude[gt_instance_mask.bool()].mean().item() if gt_instance_mask.sum() > 0 else 0.0
                    gt_flow_magnitudes[gt_inst_id] = gt_flow_magnitude
                
                # 从GT instance出发：对每个GT instance，找到与所有pred slot的最大IoU，并建立匹配关系
                # gt_instance_to_best_slot: {gt_inst_id: best_slot_id}
                # slot_to_class: {slot_id: class_id} - 基于GT instance的匹配结果
                gt_instance_to_best_slot = {}  # {gt_inst_id: (best_slot_id, max_iou)}
                slot_to_best_gt_iou = {}  # {slot_id: (gt_inst_id, iou)} - 用于确定每个slot匹配的最佳GT instance
                
                for gt_inst_id in unique_gt_instances:
                    gt_inst_id = gt_inst_id.item()
                    gt_instance_mask = (instance_mask_matched_cpu == gt_inst_id).float()  # [N]
                    gt_size = gt_instance_mask.sum().item()
                    
                    # 跳过太小的instance（min_points=50）
                    if gt_size < min_instance_size:
                        continue
                    
                    # 获取该GT instance的class_id
                    if gt_inst_id in instance_to_class:
                        inst_class_id = instance_to_class[gt_inst_id]
                    else:
                        inst_class_id = -1  # 背景类
                    
                    # 计算该GT instance与所有pred slot的IoU
                    max_iou = 0.0
                    best_slot = -1
                    
                    for slot_id in range(num_slots):
                        pred_slot_mask = pred_mask_onehot[slot_id]  # [N]
                        intersection = (pred_slot_mask * gt_instance_mask).sum().item()
                        union = pred_slot_mask.sum().item() + gt_instance_mask.sum().item() - intersection
                        iou = intersection / union if union > 0 else 0.0
                        
                        # 记录每个pred slot匹配到的最佳GT instance（选择IoU最大的）
                        if slot_id not in slot_to_best_gt_iou or iou > slot_to_best_gt_iou[slot_id][1]:
                            slot_to_best_gt_iou[slot_id] = (gt_inst_id, iou)
                        
                        if iou > max_iou:
                            max_iou = iou
                            best_slot = slot_id
                    
                    # 记录GT instance到最佳pred slot的匹配
                    if best_slot >= 0:
                        gt_instance_to_best_slot[gt_inst_id] = (best_slot, max_iou)
                    
                    # 记录所有类别的数据（不包括背景）- 用于mIoU统计和绘图
                    meta_class_id = get_meta_category_id(inst_class_id)
                    mask_iou_flow_data.append((max_iou, gt_flow_magnitudes[gt_inst_id], meta_class_id,sample_flow["sequence_id"]))
                
                # 基于GT instance的匹配结果，建立slot_to_class映射
                # 对于每个slot，找到匹配到它的最佳GT instance（IoU最大），使用该GT instance的class_id
                slot_to_class = {}
                for slot_id in range(num_slots):
                    if slot_id in slot_to_best_gt_iou:
                        best_gt_inst_id, best_iou = slot_to_best_gt_iou[slot_id]
                        if best_gt_inst_id in instance_to_class:
                            slot_to_class[slot_id] = instance_to_class[best_gt_inst_id]
                        else:
                            slot_to_class[slot_id] = -1  # 背景类
                    else:
                        slot_to_class[slot_id] = -1  # 背景类（没有匹配到任何GT instance）
                
                # 将pred_slot映射到class_id（基于GT instance的匹配结果）
                pred_class = torch.zeros_like(pred_slot_cpu)
                for slot_id, class_id in slot_to_class.items():
                    pred_class[pred_slot_cpu == slot_id] = class_id
                
                # 将class_id映射到元类别ID
                pred_meta_class = torch.zeros_like(pred_class)
                gt_meta_class = torch.zeros_like(class_ids_gt)
                
                for i in range(len(pred_class)):
                    pred_meta_class[i] = get_meta_category_id(pred_class[i].item())
                for i in range(len(class_ids_gt)):
                    gt_meta_class[i] = get_meta_category_id(class_ids_gt[i].item())
                
                # 计算每个元类别的IoU
                unique_meta_classes = torch.unique(torch.cat([gt_meta_class, pred_meta_class]))
                for meta_class_id in unique_meta_classes:
                    meta_id = meta_class_id.item()
                    gt_mask = (gt_meta_class == meta_id)
                    pred_mask = (pred_meta_class == meta_id)
                    intersection = (gt_mask & pred_mask).sum().item()
                    union = (gt_mask | pred_mask).sum().item()
                    if union > 0:
                        class_iou_stats[meta_id]['intersection'] += intersection
                        class_iou_stats[meta_id]['union'] += union
                    class_iou_stats[meta_id]['total_points'] += gt_mask.sum().item()
            
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{total_samples} 条数据 (索引: {i})")
        except Exception as e:
            print(f"处理第 {i} 条数据时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

print("\n" + "="*80)
print("使用calculate_miou计算的mIoU统计（与eval.py一致）")
print("="*80)
if len(miou_list_calculate) > 0:
    miou_mean_calculate = np.mean(miou_list_calculate)
    print(f"样本数量: {len(miou_list_calculate)}")
    print(f"平均mIoU (calculate_miou): {miou_mean_calculate:.4f}")
    print(f"mIoU标准差: {np.std(miou_list_calculate):.4f}")
    print(f"mIoU最小值: {np.min(miou_list_calculate):.4f}")
    print(f"mIoU最大值: {np.max(miou_list_calculate):.4f}")
else:
    print("⚠️  没有成功计算mIoU的样本")

print("\n" + "="*80)
print("不同class_id的mIoU统计（基于元类别点级别IoU）")
print("="*80)

# 计算每个class_id的IoU
class_iou_results = {}
for class_id, stats in class_iou_stats.items():
    if stats['union'] > 0:
        iou = stats['intersection'] / stats['union']
        class_iou_results[class_id] = {
            'iou': iou,
            'intersection': stats['intersection'],
            'union': stats['union'],
            'total_points': stats['total_points']
        }

# 按IoU排序
sorted_iou = sorted(class_iou_results.items(), key=lambda x: x[1]['iou'], reverse=True)

print(f"\n{'Meta Class ID':<15} {'Meta Class Name':<20} {'IoU':<12} {'Intersection':<15} {'Union':<15} {'Total Points':<15}")
print("-" * 110)
for meta_class_id, stats in sorted_iou:
    meta_class_name = META_CATEGORY_NAMES.get(meta_class_id, "Unknown")
    print(f"{meta_class_id:<15} {meta_class_name:<20} {stats['iou']:<12.4f} {stats['intersection']:<15} {stats['union']:<15} {stats['total_points']:<15}")

# 计算平均mIoU（排除背景类）
foreground_ious = [stats['iou'] for meta_class_id, stats in sorted_iou if meta_class_id != 0]
if foreground_ious:
    mean_iou = np.mean(foreground_ious)
    print(f"\n平均mIoU（前景类）: {mean_iou:.4f}")
    print(f"前景类数量: {len(foreground_ious)}")
    print(f"前景类包括: {[META_CATEGORY_NAMES.get(mid, 'Unknown') for mid, _ in sorted_iou if mid != 0]}")

# 绘制IoU和flow幅度的散点图
if len(mask_iou_flow_data) > 0:
    print("\n" + "="*80)
    print("绘制IoU和flow幅度的散点图...")
    print("="*80)
    
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端，不需要屏幕
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    
    # 提取数据
    ious = [d[0] for d in mask_iou_flow_data]
    flow_magnitudes = [d[1] for d in mask_iou_flow_data]
    meta_classes = [d[2] for d in mask_iou_flow_data]
    sequence_ids = [d[3] for d in mask_iou_flow_data]
    # 过滤掉flow幅度小于0.01的点（用于线性回归）
    filter_mask = np.array(flow_magnitudes) >= 0.01
    ious_filtered = np.array(ious)[filter_mask]
    flow_magnitudes_filtered = np.array(flow_magnitudes)[filter_mask]
    meta_classes_filtered = np.array(meta_classes)[filter_mask]
    
    print(f"原始数据点: {len(mask_iou_flow_data)}")
    print(f"过滤后数据点（flow >= 0.01）: {filter_mask.sum()}")
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 按元类别分组绘制散点图，使用不同颜色
    colors = {
        0: '#808080',  # BACKGROUND - 灰色
        1: '#1f77b4',  # CAR - 蓝色
        2: '#2ca02c',  # WHEELED_VRU - 绿色
        3: '#ff7f0e',  # OTHER_VEHICLES - 橙色
        4: '#d62728'   # PEDESTRIAN - 红色
    }
    
    labels = {
        0: 'BACKGROUND',
        1: 'CAR',
        2: 'WHEELED_VRU',
        3: 'OTHER_VEHICLES',
        4: 'PEDESTRIAN'
    }
    
    # 为每个类别绘制散点（使用所有数据，包括背景）
    for meta_id in [0, 1, 2, 3, 4]:
        mask = np.array(meta_classes) == meta_id
        if mask.sum() > 0:
            class_ious = np.array(ious)[mask]
            class_flows = np.array(flow_magnitudes)[mask]
            ax.scatter(class_flows, class_ious, c=colors[meta_id], label=labels[meta_id], 
                      alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # 对过滤后的数据点进行线性回归（flow >= 0.01）
    X = flow_magnitudes_filtered.reshape(-1, 1)
    y = ious_filtered
    
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    r2 = r2_score(y, y_pred)
    
    # 绘制回归线（使用过滤后的数据范围）
    x_line = np.linspace(min(flow_magnitudes_filtered), max(flow_magnitudes_filtered), 100).reshape(-1, 1)
    y_line = reg.predict(x_line)
    ax.plot(x_line, y_line, 'k--', linewidth=2, label=f'Linear Regression (R²={r2:.3f}, flow≥0.01)')
    
    ax.set_xlabel('GT Flow Magnitude', fontsize=14)
    ax.set_ylabel('Mask IoU', fontsize=14)
    ax.set_title('Mask IoU vs GT Flow Magnitude (by Meta Category)', fontsize=16)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # 添加回归方程
    equation = f'y = {reg.coef_[0]:.4f}x + {reg.intercept_:.4f}'
    ax.text(0.05, 0.95, equation, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    output_path = "/workspace/gan_seg/src/iou_vs_flow_magnitude.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # 关闭图形，释放内存
    print(f"\n图表已保存到: {output_path}")
    
    # 打印统计信息
    print(f"\n数据点总数: {len(mask_iou_flow_data)}")
    print(f"用于线性回归的数据点（flow >= 0.01）: {len(ious_filtered)}")
    print(f"线性回归系数: {reg.coef_[0]:.6f}")
    print(f"截距: {reg.intercept_:.6f}")
    print(f"R²得分: {r2:.4f}")
    print(f"平均IoU（所有数据，包括背景）: {np.mean(ious):.4f}")
    print(f"平均Flow幅度（所有数据）: {np.mean(flow_magnitudes):.4f}")
    print(f"平均IoU（过滤后）: {np.mean(ious_filtered):.4f}")
    print(f"平均Flow幅度（过滤后）: {np.mean(flow_magnitudes_filtered):.4f}")
    
    # 按场景统计平均IoU
    miou_perscene = []
    iou_thisScene = []
    current_sequence_id = None
    
    for i, sequence_id in enumerate(sequence_ids):
        # 如果遇到新场景（且当前场景有数据），先计算前一个场景的平均IoU
        if current_sequence_id is not None and sequence_id != current_sequence_id:
            if len(iou_thisScene) > 0:
                miou_perscene.append(np.mean(iou_thisScene))
            iou_thisScene = []
        
        # 添加当前数据点到当前场景
        iou_thisScene.append(ious[i])
        current_sequence_id = sequence_id
    
    # 处理最后一个场景
    if len(iou_thisScene) > 0:
        miou_perscene.append(np.mean(iou_thisScene))
    
    if len(miou_perscene) > 0:
        print(f"每个场景的平均IoU: {np.mean(miou_perscene):.4f}")
        print(f"场景数量: {len(miou_perscene)}")
    else:
        print(f"每个场景的平均IoU: 无数据")

    # 按类别统计（包括背景）
    print(f"\n按类别统计:")
    for meta_id in [0, 1, 2, 3, 4]:
        mask = np.array(meta_classes) == meta_id
        if mask.sum() > 0:
            class_ious = np.array(ious)[mask]
            class_flows = np.array(flow_magnitudes)[mask]
            print(f"  {labels[meta_id]}: {mask.sum()}个mask, 平均IoU={np.mean(class_ious):.4f}, 平均Flow={np.mean(class_flows):.4f}")
else:
    print("\n⚠️  没有收集到mask IoU和flow幅度数据")

# 注释掉flow幅度统计部分，只运行mIoU计算
# 统计不同class_id的flow幅度
# class_flow_magnitudes = defaultdict(list)  # {class_id: [flow_magnitudes]}
# total_samples = 100

# print(f"读取 {total_samples} 条数据，检查不同class_id内的flow幅度...")
# print("="*80)

# for i in range(min(total_samples, len(av2_sceneflow_zoo_flow))):
#     try:
#         sample = av2_sceneflow_zoo_flow[i]
#         flow = sample["flow"]  # (N, 3)
#         class_ids = sample["class_ids"]  # (N,)
#         
#         # 确保class_ids是tensor
#         if isinstance(class_ids, np.ndarray):
#             class_ids = torch.from_numpy(class_ids)
#         
#         # 计算flow幅度
#         flow_magnitude = torch.norm(flow, dim=1)  # (N,)
#         
#         # 按class_id分组统计
#         unique_class_ids = torch.unique(class_ids)
#         for class_id in unique_class_ids:
#             mask = class_ids == class_id
#             if mask.sum() > 0:
#                 class_flow_magnitudes[class_id.item()].extend(flow_magnitude[mask].tolist())
#         
#         if (i + 1) % 10 == 0:
#             print(f"已处理 {i + 1}/{total_samples} 条数据")
#     except Exception as e:
#         print(f"处理第 {i} 条数据时出错: {e}")
#         continue

# print("\n" + "="*80)
# print("不同class_id的flow幅度统计（平均值）")
# print("="*80)

# # # 计算每个class_id的平均flow幅度和top 10%
# # class_avg_magnitudes = {}
# # for class_id, magnitudes in class_flow_magnitudes.items():
#     if len(magnitudes) > 0:
#         magnitudes_array = np.array(magnitudes)
#         # 计算top 10%的索引
#         top_10_percent_count = max(1, int(len(magnitudes) * 0.1))  # 至少1个点
#         top_10_percent_indices = np.argsort(magnitudes_array)[-top_10_percent_count:]
#         top_10_percent_magnitudes = magnitudes_array[top_10_percent_indices]
#         
#         class_avg_magnitudes[class_id] = {
#             'mean': np.mean(magnitudes),
#             'std': np.std(magnitudes),
#             'min': np.min(magnitudes),
#             'max': np.max(magnitudes),
#             'count': len(magnitudes),
#             'top10_mean': np.mean(top_10_percent_magnitudes),
#             'top10_min': np.min(top_10_percent_magnitudes),
#             'top10_max': np.max(top_10_percent_magnitudes),
#             'top10_count': len(top_10_percent_magnitudes)
#         }
# 
# # 按平均flow幅度排序
# sorted_classes = sorted(class_avg_magnitudes.items(), key=lambda x: x[1]['mean'], reverse=True)
# 
# print(f"\n{'Class ID':<10} {'Class Name':<25} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'Count':<10}")
# print("-" * 110)
# for class_id, stats in sorted_classes:
#     class_name = CATEGORY_MAP.get(class_id, "Unknown")
#     print(f"{class_id:<10} {class_name:<25} {stats['mean']:<12.4f} {stats['std']:<12.4f} {stats['min']:<12.4f} {stats['max']:<12.4f} {stats['count']:<10}")
# 
# print("\n" + "="*80)
# print("每个类别Top 10%的flow幅度统计")
# print("="*80)
# 
# # 按top 10%平均值排序
# sorted_top10 = sorted(class_avg_magnitudes.items(), key=lambda x: x[1]['top10_mean'], reverse=True)
# 
# print(f"\n{'Class ID':<10} {'Class Name':<25} {'Top10% Mean':<15} {'Top10% Min':<15} {'Top10% Max':<15} {'Top10% Count':<15} {'All Mean':<12}")
# print("-" * 120)
# for class_id, stats in sorted_top10:
#     class_name = CATEGORY_MAP.get(class_id, "Unknown")
#     print(f"{class_id:<10} {class_name:<25} {stats['top10_mean']:<15.4f} {stats['top10_min']:<15.4f} {stats['top10_max']:<15.4f} {stats['top10_count']:<15} {stats['mean']:<12.4f}")
# 
# print("\n" + "="*80)
# print("总结（排除背景类）")
# print("="*80)
# # 排除背景类（class_id == -1）
# foreground_classes = {k: v for k, v in class_avg_magnitudes.items() if k != -1}
# if foreground_classes:
#     sorted_foreground = sorted(foreground_classes.items(), key=lambda x: x[1]['mean'], reverse=True)
#     print(f"总共处理了 {len(class_avg_magnitudes)} 个不同的class_id（其中 {len(foreground_classes)} 个前景类）")
#     print(f"\n前景类flow幅度排名（全部平均）:")
#     for i, (class_id, stats) in enumerate(sorted_foreground[:5], 1):
#         class_name = CATEGORY_MAP.get(class_id, "Unknown")
#         print(f"  {i}. {class_name} (ID={class_id}): 平均 {stats['mean']:.4f}, 点数 {stats['count']}")
#     
#     # Top 10%排名
#     foreground_top10 = {k: v for k, v in class_avg_magnitudes.items() if k != -1}
#     sorted_foreground_top10 = sorted(foreground_top10.items(), key=lambda x: x[1]['top10_mean'], reverse=True)
#     
#     print(f"\n前景类Top 10% flow幅度排名:")
#     for i, (class_id, stats) in enumerate(sorted_foreground_top10[:5], 1):
#         class_name = CATEGORY_MAP.get(class_id, "Unknown")
#         print(f"  {i}. {class_name} (ID={class_id}): Top10%平均 {stats['top10_mean']:.4f}, 全部平均 {stats['mean']:.4f}, Top10%点数 {stats['top10_count']}")
#     
#     print(f"\nflow幅度最大的前景类（全部平均）: {CATEGORY_MAP.get(sorted_foreground[0][0], 'Unknown')} (ID={sorted_foreground[0][0]}, 平均 {sorted_foreground[0][1]['mean']:.4f})")
#     print(f"flow幅度最小的前景类（全部平均）: {CATEGORY_MAP.get(sorted_foreground[-1][0], 'Unknown')} (ID={sorted_foreground[-1][0]}, 平均 {sorted_foreground[-1][1]['mean']:.4f})")
#     print(f"所有前景类的平均flow幅度: {np.mean([s['mean'] for s in foreground_classes.values()]):.4f}")
#     
#     print(f"\nTop 10% flow幅度最大的前景类: {CATEGORY_MAP.get(sorted_foreground_top10[0][0], 'Unknown')} (ID={sorted_foreground_top10[0][0]}, Top10%平均 {sorted_foreground_top10[0][1]['top10_mean']:.4f})")
#     print(f"Top 10% flow幅度最小的前景类: {CATEGORY_MAP.get(sorted_foreground_top10[-1][0], 'Unknown')} (ID={sorted_foreground_top10[-1][0]}, Top10%平均 {sorted_foreground_top10[-1][1]['top10_mean']:.4f})")
#     print(f"所有前景类的Top 10%平均flow幅度: {np.mean([s['top10_mean'] for s in foreground_top10.values()]):.4f}")
#     
#     # 背景类统计
#     if -1 in class_avg_magnitudes:
#         bg_stats = class_avg_magnitudes[-1]
#         print(f"\n背景类 (BACKGROUND): 平均flow幅度 {bg_stats['mean']:.4f}, 点数 {bg_stats['count']}")
# else:
#     print(f"总共处理了 {len(class_avg_magnitudes)} 个不同的class_id")
#     if sorted_classes:
#         print(f"flow幅度最大的class_id: {sorted_classes[0][0]} (平均 {sorted_classes[0][1]['mean']:.4f})")
#         print(f"flow幅度最小的class_id: {sorted_classes[-1][0]} (平均 {sorted_classes[-1][1]['mean']:.4f})")
#         print(f"所有class_id的平均flow幅度: {np.mean([s['mean'] for s in class_avg_magnitudes.values()]):.4f}")
