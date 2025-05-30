import h5py
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

def read_av2_h5(file_path: str, timestamp: Optional[str] = None) -> Dict:
    """
    Read a single frame from an AV2 h5 file created by the preprocessing code.
    
    Args:
        file_path (str): Path to the h5 file
        timestamp (str, optional): Specific timestamp to load. If None, returns the first timestamp.
    
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
            ego_motion = np.array(group['ego_motion'])
        else:
            flow = None
            flow_is_valid = None
            flow_category = None
            ego_motion = None
            
        # For evaluation masks (only in val/test sets)
        if 'eval_mask' in group:
            eval_mask = np.array(group['eval_mask'])
        else:
            eval_mask = None
            
        # If we have flow, try to find the next point cloud
        point_cloud_second = None
        next_timestamp_idx = timestamps.index(timestamp) + 1
        if next_timestamp_idx < len(timestamps) and flow is not None:
            next_timestamp = timestamps[next_timestamp_idx]
            point_cloud_second = np.array(f[next_timestamp]['lidar'])
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
    
    if eval_mask is not None:
        result["eval_mask"] = torch.from_numpy(eval_mask).bool()
        
    if point_cloud_second is not None:
        result["point_cloud_second"] = torch.from_numpy(point_cloud_second).float()
    
    return result

def read_av2_scene(file_path: str) -> Dict[str, Dict]:
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
            scene_data[timestamp] = read_av2_h5(file_path, timestamp)
    
    return scene_data