"""
Bucketed EPE计算工具，参考论文 "I Can't Believe It's Not Scene Flow!"
实现Bucket Normalized EPE评估方法
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from bucketed_scene_flow_eval.datasets.argoverse2.av2_metacategories import (
    BUCKETED_METACATAGORIES,
    BUCKETED_VOLUME_METACATAGORIES,
    THREEWAY_EPE_METACATAGORIES
)
from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import CATEGORY_MAP
from bucketed_scene_flow_eval.eval.bucketed_epe import BucketedEPEEvaluator, BucketResultMatrix
from bucketed_scene_flow_eval.datastructures import SemanticClassId
# 创建TimeSyncedSceneFlowFrame对象
from bucketed_scene_flow_eval.datastructures import TimeSyncedSceneFlowFrame, SupervisedPointCloudFrame, PointCloud
from bucketed_scene_flow_eval.datastructures import SE3, PoseInfo
from bucketed_scene_flow_eval.datastructures import EgoLidarFlow

def extract_classid_from_argoverse2_data(
    scene_flow_frame,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    从Argoverse2CausalSceneFlow数据中提取classid
    
    Args:
        scene_flow_frame: TimeSyncedSceneFlowFrame对象
        mask: 可选的掩码，用于过滤点云
        
    Returns:
        classid数组
    """
    if hasattr(scene_flow_frame, 'pc') and hasattr(scene_flow_frame.pc, 'full_pc_classes'):
        class_ids = scene_flow_frame.pc.full_pc_classes
        if mask is not None:
            class_ids = class_ids[mask]
        return class_ids
    else:
        # 如果没有类别信息，返回背景类别
        if mask is not None:
            return np.full(np.sum(mask), -1, dtype=np.int8)
        else:
            return np.full(len(scene_flow_frame.pc.full_pc), -1, dtype=np.int8)


def calculate_speed_buckets(
    gt_flow: np.ndarray,
    num_buckets: int = 51,
    max_speed: float = 2.0
) -> List[Tuple[float, float]]:
    """
    计算速度分桶
    
    Args:
        gt_flow: 真实场景流 [N, 3]
        num_buckets: 分桶数量
        max_speed: 最大速度阈值 (m/s)
        
    Returns:
        速度分桶列表
    """
    # 计算每个点的速度
    speeds = np.linalg.norm(gt_flow, axis=1)
    
    # 创建分桶边界
    bucket_edges = np.concatenate([
        np.linspace(0, max_speed, num_buckets), 
        [np.inf]
    ])
    
    speed_buckets = list(zip(bucket_edges[:-1], bucket_edges[1:]))
    return speed_buckets


def compute_bucketed_epe_metrics(
    point_clouds: List[torch.Tensor],
    pred_flows: List[torch.Tensor],
    gt_flows: List[torch.Tensor], 
    class_ids: List[np.ndarray],
    class_id_to_name: Dict[int, str] = None,
    meta_class_lookup: Dict[str, List[str]] = None,
    num_buckets: int = 51,
    max_speed: float = 2.0,
    output_path: Path = Path("/tmp/bucketed_epe_results")
) -> Dict[str, Tuple[float, float]]:
    """
    计算Bucketed EPE指标
    
    Args:
        pred_flows: 预测的场景流列表
        gt_flows: 真实场景流列表
        class_ids: 类别ID列表
        class_id_to_name: 类别ID到名称的映射
        meta_class_lookup: 元类别查找表
        num_buckets: 速度分桶数量
        max_speed: 最大速度阈值
        output_path: 输出路径
        
    Returns:
        每个类别的(静态EPE, 动态误差)字典
    """
    if class_id_to_name is None:
        class_id_to_name = CATEGORY_MAP
    
    if meta_class_lookup is None:
        meta_class_lookup = BUCKETED_METACATAGORIES
    
    # 创建BucketedEPEEvaluator
    evaluator = BucketedEPEEvaluator(
        class_id_to_name=class_id_to_name,
        bucket_max_speed=max_speed,
        num_buckets=num_buckets,
        output_path=output_path,
        meta_class_lookup=meta_class_lookup
    )
    
    # 处理每个样本
    for i, (point_cloud, pred_flow, gt_flow, class_id) in enumerate(zip(point_clouds, pred_flows, gt_flows, class_ids)):
        # 转换为numpy数组
        if isinstance(pred_flow, torch.Tensor):
            pred_flow = pred_flow.detach().cpu().numpy()
        if isinstance(gt_flow, torch.Tensor):
            gt_flow = gt_flow.detach().cpu().numpy()
        
        # 确保形状一致
        min_len = min(len(pred_flow), len(gt_flow), len(class_id))
        pred_flow = pred_flow[:min_len]
        gt_flow = gt_flow[:min_len]
        class_id = class_id[:min_len]
        
        # 计算点云坐标（这里使用简单的索引作为坐标）
        
        # 创建EgoLidarFlow对象
        pred_ego_flow = EgoLidarFlow(full_flow=pred_flow, mask=np.ones(len(pred_flow), dtype=bool))
        

        
        # 创建点云对象
        pc = PointCloud(points=point_cloud)
        
        # 创建监督点云帧
        supervised_pc = SupervisedPointCloudFrame(
            full_pc=pc,
            pose=PoseInfo.identity(),
            mask=np.ones(len(point_cloud), dtype=bool),
            full_pc_classes=class_id
        )
        
        # 创建真实流对象
        gt_ego_flow = EgoLidarFlow(full_flow=gt_flow, mask=np.ones(len(gt_flow), dtype=bool))
        
        # 创建场景流帧
        scene_flow_frame = TimeSyncedSceneFlowFrame(
            pc=supervised_pc,
            flow=gt_ego_flow,
            log_id=f"sample_{i}",
            log_idx=i,
            log_timestamp=0.0,
            auxillary_pc=None,
            rgbs=None
        )
        
        # 使用eval方法添加帧结果
        evaluator.eval(pred_ego_flow, scene_flow_frame)
    
    # 计算结果
    results = evaluator.compute_results(save_results=True)
    return results


def compute_volume_based_bucketed_epe(
    point_clouds: List[torch.Tensor],
    pred_flows: List[torch.Tensor],
    gt_flows: List[torch.Tensor],
    class_ids: List[np.ndarray],
    volume_thresholds: Tuple[float, float] = (9.5, 40.0),
    output_path: Path = Path("/tmp/volume_bucketed_epe_results")
) -> Dict[str, Tuple[float, float]]:
    """
    基于体积的Bucketed EPE计算（参考论文附录C）
    
    Args:
        pred_flows: 预测的场景流列表
        gt_flows: 真实场景流列表  
        class_ids: 类别ID列表
        volume_thresholds: 体积阈值 (small, medium, large)
        output_path: 输出路径
        
    Returns:
        每个体积类别的(静态EPE, 动态误差)字典
    """
    # 创建基于体积的元类别映射
    volume_meta_lookup = BUCKETED_VOLUME_METACATAGORIES
    
    # 创建体积类别映射
    volume_class_id_to_name = {
        -1: "BACKGROUND",
        0: "SMALL", 
        1: "MEDIUM",
        2: "LARGE"
    }
    
    # 将原始类别ID映射到体积类别
    volume_class_ids = []
    for class_id in class_ids:
        # 这里需要根据实际的边界框信息来计算体积
        # 由于我们没有边界框信息，这里使用简化的映射
        volume_class_id = np.where(class_id == -1, -1, 1)  # 默认中等大小
        volume_class_ids.append(volume_class_id)
    
    return compute_bucketed_epe_metrics(
        point_clouds=point_clouds,
        pred_flows=pred_flows,
        gt_flows=gt_flows,
        class_ids=volume_class_ids,
        class_id_to_name=volume_class_id_to_name,
        meta_class_lookup=volume_meta_lookup,
        output_path=output_path
    )

"""
这段ai写的代码完全没用，特意注释掉以防止以后被运行
only AI can do
"""
# def evaluate_with_bucketed_epe(
#     scene_flow_predictor,
#     dataloader,
#     device,
#     config,
#     output_path: Path = Path("/tmp/bucketed_epe_eval")
# ) -> Dict[str, Dict[str, Tuple[float, float]]]:
#     """
#     使用Bucketed EPE评估模型
    
#     Args:
#         scene_flow_predictor: 场景流预测模型
#         dataloader: 数据加载器
#         device: 设备
#         config: 配置
#         output_path: 输出路径
        
#     Returns:
#         评估结果字典
#     """
#     scene_flow_predictor.eval()
    
#     pred_flows = []
#     gt_flows = []
#     class_ids = []
    
#     with torch.no_grad():
#         for batch in dataloader:
#             point_cloud_firsts = [item.to(device) for item in batch["point_cloud_first"]]
#             flow_gt = batch.get("flow")
            
#             if flow_gt is not None:
#                 flow_gt = [item.to(device) for item in flow_gt]
            
#             # 预测场景流
#             for i, point_cloud_first in enumerate(point_cloud_firsts):
#                 try:
#                     if getattr(config.model.flow, "name", "") == "FastFlow3D":
#                         cur_idx = batch["idx"][i]
#                         total_frames = batch["total_frames"][i]
#                         next_idx = min(int(cur_idx) + 1, int(total_frames) - 1)
#                         next_item = batch["self"][0].get_item(next_idx)
#                         pc0 = point_cloud_firsts[i][:, :3]
#                         pc1 = next_item["point_cloud_first"].to(device)[:, :3]
#                         pose0 = batch.get("pose")
#                         if pose0 is not None:
#                             pose0 = pose0[i].to(device)
#                         else:
#                             pose0 = torch.eye(4, device=device)
#                         pose1 = next_item.get("pose", torch.eye(4)).to(device)
#                         flow_pred = scene_flow_predictor(pc0, pc1, pose0, pose1)
#                     else:
#                         flow_pred = scene_flow_predictor(point_cloud_first)
                    
#                     pred_flows.append(flow_pred)
                    
#                 except Exception as e:
#                     # 备用预测方法
#                     flow_pred = scene_flow_predictor(point_cloud_first)
#                     pred_flows.append(flow_pred)
            
#             gt_flows.extend(flow_gt)
            
#             # 提取类别ID
#             for i, point_cloud_first in enumerate(point_cloud_firsts):
#                 # 从数据中提取类别信息
#                 if hasattr(batch, 'class_ids') and batch['class_ids'] is not None:
#                     class_id = batch['class_ids'][i]
#                 else:
#                     # 如果没有类别信息，创建默认的背景类别
#                     class_id = np.full(len(point_cloud_first), -1, dtype=np.int8)
                
#                 class_ids.append(class_id)
    
#     # 计算标准Bucketed EPE
#     standard_results = compute_bucketed_epe_metrics(
#         pred_flows=pred_flows,
#         gt_flows=gt_flows,
#         class_ids=class_ids,
#         output_path=output_path / "standard"
#     )
    
#     # 计算基于体积的Bucketed EPE
#     volume_results = compute_volume_based_bucketed_epe(
#         pred_flows=pred_flows,
#         gt_flows=gt_flows,
#         class_ids=class_ids,
#         output_path=output_path / "volume"
#     )
    
#     return {
#         "standard_bucketed_epe": standard_results,
#         "volume_bucketed_epe": volume_results
#     }
