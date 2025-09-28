#!/usr/bin/env python3
"""
Simple AV2 Complete Sequence Visualization
简单的AV2完整序列可视化
"""

import sys
import os
import numpy as np
import torch

show_flow = True
max_sequence_length = 2
start_sequence_index = 50
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import open3d as o3d
except ImportError:
    print("Error: Open3D not installed. Please install with: pip install open3d")
    sys.exit(1)

from dataset.av2_dataset import AV2SequenceDataset

def main():
    """显示完整的AV2序列，所有帧一起显示在屏幕上"""
    try:
        # Initialize dataset
        dataset = AV2SequenceDataset(
            fix_ego_motion=True,
            max_k=1,
            apply_ego_motion=True
        )
        
        num_frames = len(dataset)
        num_frames = min(num_frames, max_sequence_length)
        # Create all geometries
        all_geometries = []
        
        for i in range(num_frames):
            # Load sample
            sample = dataset.get_item(i+start_sequence_index)
            if not sample:
                break
            
            # Extract data
            points = sample['point_cloud_first']
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            
            # Create point cloud with colors
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # Color points based on type
            colors = np.ones((len(points), 3)) * 0.7  # Default gray
            
            # Color different point types
            if 'background_static_mask' in sample:
                mask = sample['background_static_mask']
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                colors[mask] = [0.5, 0.5, 0.5]  # Gray for background static
            
            if 'foreground_static_mask' in sample:
                mask = sample['foreground_static_mask']
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                colors[mask] = [0.0, 0.8, 0.0]  # Green for foreground static
            
            if 'foreground_dynamic_mask' in sample:
                mask = sample['foreground_dynamic_mask']
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                colors[mask] = [1.0, 0.0, 0.0]  # Red for foreground dynamic
            
            pcd.colors = o3d.utility.Vector3dVector(colors)
            all_geometries.append(pcd)
            
            # Add flow vectors for dynamic points
            if show_flow and 'flow' in sample and 'foreground_dynamic_mask' in sample:
                flow = sample['flow']
                if isinstance(flow, torch.Tensor):
                    flow = flow.detach().cpu().numpy()
                
                dynamic_mask = sample['foreground_dynamic_mask']
                if isinstance(dynamic_mask, torch.Tensor):
                    dynamic_mask = dynamic_mask.detach().cpu().numpy()
                
                if dynamic_mask.sum() > 0:
                    dynamic_points = points[dynamic_mask]
                    dynamic_flow = flow[dynamic_mask]
                    
                    # Create line set for flow vectors
                    end_points = dynamic_points + dynamic_flow
                    all_points = np.vstack((dynamic_points, end_points))
                    
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(all_points)
                    
                    # Create lines
                    lines = [[j, j + len(dynamic_points)] for j in range(len(dynamic_points))]
                    line_set.lines = o3d.utility.Vector2iVector(lines)
                    
                    # Yellow color for flow vectors
                    flow_colors = np.tile([1.0, 1.0, 0.0], (len(dynamic_points), 1))
                    line_set.colors = o3d.utility.Vector3dVector(flow_colors)
                    
                    all_geometries.append(line_set)
        print("num_frames",num_frames)
        print("all_geometries",len(all_geometries))
        # Visualize all frames together
        o3d.visualization.draw_geometries(
            all_geometries[:-1],
            window_name="AV2 Complete Sequence",
            width=1600,
            height=1000
        )
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
