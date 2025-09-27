import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

def read_av2_h5(file_path: str, apply_ego_motion: bool = True, timestamp: Optional[str] = None) -> Dict:
    """
    Read a single frame from an AV2 h5 file created by the preprocessing code.
    
    Args:
        file_path (str): Path to the h5 file
        timestamp (str, optional): Specific timestamp to load. If None, returns the first timestamp.
        apply_ego_motion (bool, optional): Whether to apply ego motion to the point cloud.
    Returns:
        dict: Dictionary containing point cloud, flow, and other data
    """
    with h5py.File(file_path, 'r') as f:
        # Get available timestamps
        timestamps = list(f.keys())
        
        # If no timestamp provided, use the first one
        if timestamp is None:
            timestamp = timestamps[0]
        elif timestamp not in timestamps:
            raise ValueError(f"Timestamp {timestamp} not found in {file_path}")
        
        # Get the group for this timestamp
        group = f[timestamp]
        # Always available data
        point_cloud_first = np.array(group['lidar'])

        ground_mask = np.array(group['ground_mask'])
        pose = np.array(group['pose'])
        
        # Data that might not be available for the last frame or test set
        if 'flow' in group:
            flow = np.array(group['flow'])
            flow_is_valid = np.array(group['flow_is_valid'])
            flow_category = np.array(group['flow_category_indices'])
            real_ego_motion = np.array(group['ego_motion'])
            if not apply_ego_motion:
                ego_motion = np.array(group['ego_motion'])
            else:
                ego_motion = np.eye(4).repeat(point_cloud_first.shape[0], axis=0)
        else:
            flow = None
            flow_is_valid = None
            flow_category = None
            real_ego_motion = None
            ego_motion = None
            
        # For evaluation masks (only in val/test sets)
        if 'eval_mask' in group:
            eval_mask = np.array(group['eval_mask'])
        else:
            eval_mask = None
            
        # 在read_av2_h5函数中添加这些字段
        label = None
        if 'label' in group:
            label = np.array(group['label'])
        dufo_label = None
        if 'dufo_label' in group:
            dufo_label = np.array(group['dufo_label'])
        flow_instance_id = None
        if 'flow_instance_id' in group:
            flow_instance_id = np.array(group['flow_instance_id'])
    
    # Create result dictionary
    result = {
        "point_cloud_first": torch.from_numpy(point_cloud_first).float(),
        "ground_mask": torch.from_numpy(ground_mask).bool(),
        "pose": torch.from_numpy(pose).float(),
    }
    if label is not None:
        result["label"] = torch.from_numpy(label).long()
    if dufo_label is not None:
        result["dufo_label"] = torch.from_numpy(dufo_label).long()
    if flow_instance_id is not None:
        result["flow_instance_id"] = torch.from_numpy(flow_instance_id).long()
    if flow is not None:
        result["flow"] = torch.from_numpy(flow).float()
        result["flow_is_valid"] = torch.from_numpy(flow_is_valid).bool()
        result["flow_category"] = torch.from_numpy(flow_category).to(torch.uint8)
        result["ego_motion"] = torch.from_numpy(ego_motion).float()
        result["real_ego_motion"] = torch.from_numpy(real_ego_motion).float()
    
    if eval_mask is not None:
        result["eval_mask"] = torch.from_numpy(eval_mask).bool()
        
    
    return result

DEFAULT_POINT_CLOUD_RANGE = (
    -48,
    -48,
    -2.5,
    48,
    48,
    2.5,
)

def read_av2_scene(file_path: str, apply_ego_motion: bool = True) -> Dict[str, Dict]:
    """
    Read all frames from an AV2 h5 file created by the preprocessing code.
    
    Args:
        file_path (str): Path to the h5 file
    
    Returns:
        Dict[str, Dict]: Dictionary with timestamps as keys and data dictionaries as values
    """
    scene_data = {}
    
    with h5py.File(file_path, 'r') as f:
        # Get available timestamps
        timestamps = list(f.keys())
        
        for timestamp in timestamps:
            scene_data[timestamp] = read_av2_h5(file_path, apply_ego_motion, timestamp)
        center = None
        for i in range(len(timestamps)):
            point_cloud = scene_data[timestamps[i]]["point_cloud_first"]
            point_cloud_cropped_mask = torch.all(
                torch.stack([
                    point_cloud[:, 0] > DEFAULT_POINT_CLOUD_RANGE[0],
                    point_cloud[:, 0] < DEFAULT_POINT_CLOUD_RANGE[3],
                    point_cloud[:, 1] > DEFAULT_POINT_CLOUD_RANGE[1],
                    point_cloud[:, 1] < DEFAULT_POINT_CLOUD_RANGE[4],
                    point_cloud[:, 2] > DEFAULT_POINT_CLOUD_RANGE[2],
                    point_cloud[:, 2] < DEFAULT_POINT_CLOUD_RANGE[5]
                ], dim=0), dim=0)
            scene_data[timestamps[i]]["point_cloud_first_cropped_mask"] = point_cloud_cropped_mask
            if apply_ego_motion:
                pose = scene_data[timestamps[i]]["pose"]
                # Transform points from current frame to first frame
                # 将点从当前帧变换到第一帧
                # P_first = R^T * (P_current - t)
                # where R is rotation matrix and t is translation vector
                # transformed_points = torch.matmul(point_cloud - motion[:3, 3], motion[:3, :3].T)
                point_cloud = point_cloud*torch.tensor([-1,-1,1])
                transformed_points = torch.matmul(point_cloud, pose[:3, :3].T)-pose[:3, 3]*torch.tensor([1,1,1])

                # Transform flow vectors to first frame coordinate system
                # 将flow向量变换到第一帧坐标系
                if "flow" in scene_data[timestamps[i]]:
                    flow = scene_data[timestamps[i]]["flow"]
                    # Flow vectors are also rotated by the same rotation matrix
                    # Flow向量也通过相同的旋转矩阵进行旋转
                    real_ego_motion = scene_data[timestamps[i]]["real_ego_motion"]
                    pc_flow= torch.matmul(flow-real_ego_motion[:3, 3], real_ego_motion[:3, :3])*torch.tensor([-1,-1,1])
                    transformed_flow = torch.matmul(pc_flow, pose[:3, :3].T)#-pose[:3, 3]
                    scene_data[timestamps[i]]["flow"] = transformed_flow
                if center is None:
                    center = transformed_points.mean(dim=0)
                transformed_points = transformed_points - center
                scene_data[timestamps[i]]["point_cloud_first"] = transformed_points

    return scene_data