from typing import Optional
import open3d as o3d
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def visualize_vectors(points, vectors, vis=None, color=None, scale=1.0):
    """
    Visualize vectors in Open3D.
    
    Args:
        points (numpy.ndarray): Starting points of vectors, shape (N, 3)
        vectors (numpy.ndarray): Vectors to visualize, shape (N, 3)
        vis (o3d.visualization.Visualizer, optional): Existing visualizer
        color (list, optional): RGB color for vectors, default [1, 0, 0] (red)
        scale (float, optional): Scaling factor for vector lengths
        
    Returns:
        tuple: (visualizer, line_set) - The visualizer and the created line set
    """
    if color is None:
        color = [1, 0, 0]  # Default red color
        
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
    
    # Create line set for vectors
    line_set = o3d.geometry.LineSet()
    
    # Generate points and lines
    end_points = points + vectors * scale
    all_points = np.vstack((points, end_points))
    line_set.points = o3d.utility.Vector3dVector(all_points)
    
    # Create lines connecting start points to end points
    lines = [[i, i + len(points)] for i in range(len(points))]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Set color for all lines
    line_set.colors = o3d.utility.Vector3dVector(color)
    
    # Add to visualizer
    vis.add_geometry(line_set)
    
    return vis, line_set

def update_vector_visualization(line_set, points, vectors, scale=1.0, color=None):
    """
    Update an existing line set with new vectors.
    
    Args:
        line_set (o3d.geometry.LineSet): The line set to update
        points (numpy.ndarray): Starting points of vectors, shape (N, 3)
        vectors (numpy.ndarray): Vectors to visualize, shape (N, 3)
        scale (float, optional): Scaling factor for vector lengths
        color (list, optional): RGB color for vectors
        
    Returns:
        o3d.geometry.LineSet: The updated line set
    """
    # Generate points
    end_points = points + vectors * scale
    all_points = np.vstack((points, end_points))
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.colors = o3d.utility.Vector3dVector(color)
    
    return line_set
