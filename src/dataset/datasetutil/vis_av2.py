import os
import numpy as np
import h5py
from typing import Optional, Dict
import open3d as o3d
from av2 import read_av2_h5
# Read a single frame
data = read_av2_h5("/home/lzq/workspace/gan_seg/demo_data/demo/train/8de6abb6-6589-3da7-8e21-6ecc80004a36.h5")

# Access the point cloud and scene flow
point_cloud = data["point_cloud_first"]  # Shape: [N, 3]
scene_flow = data["flow"]  # Shape: [N, 3]

# Visualize the point clouds and flow
valid_mask = data["flow_is_valid"]
dynamic_mask = data["flow_category"] != 0
valid_mask = valid_mask & dynamic_mask
# Create point clouds
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(data["point_cloud_first"].numpy()[valid_mask])

if "point_cloud_second" in data:
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(data["point_cloud_second"].numpy())
    pcd2.paint_uniform_color([0, 1, 0])  # Green for second point cloud

# Visualize flow vectors
if "flow" in data:
    # Get points that have valid flow

    valid_points = data["point_cloud_first"][valid_mask].numpy()
    valid_flow = data["flow"][valid_mask].numpy()
    
    # Create a line set to visualize the flow
    line_set = o3d.geometry.LineSet()
    
    # Points include starting points and ending points
    end_points = valid_points + valid_flow
    all_points = np.vstack((valid_points, end_points))
    line_set.points = o3d.utility.Vector3dVector(all_points)
    
    # Create lines
    lines = [[i, i + len(valid_points)] for i in range(len(valid_points))]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Set colors based on flow magnitude
    flow_magnitude = np.linalg.norm(valid_flow, axis=1)
    norm_magnitude = flow_magnitude / (np.max(flow_magnitude) + 1e-10)
    colors = np.zeros((len(lines), 3))
    colors[:, 0] = norm_magnitude  # Red channel based on flow magnitude
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd1, pcd2, line_set])