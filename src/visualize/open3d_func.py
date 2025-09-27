from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

def visualize_vectors(points, vectors, vis=None, color=None, scale=1.0):
    """
    Visualize vectors in Open3D.
    
    Args:
        points (numpy.ndarray or torch.Tensor): Starting points of vectors, shape (N, 3)
        vectors (numpy.ndarray or torch.Tensor): Vectors to visualize, shape (N, 3)
        vis (o3d.visualization.Visualizer, optional): Existing visualizer
        color (list or numpy.ndarray, optional): RGB color for vectors, default [1, 0, 0] (red)
                                               Can be single color [R,G,B] or per-vector colors [N,3]
        scale (float, optional): Scaling factor for vector lengths
        
    Returns:
        tuple: (visualizer, line_set) - The visualizer and the created line set
    """
    import open3d as o3d

    # Convert torch tensors to numpy arrays
    if hasattr(points, 'detach'):  # torch tensor
        points = points.detach().cpu().numpy()
    if hasattr(vectors, 'detach'):  # torch tensor
        vectors = vectors.detach().cpu().numpy()
    
    # Ensure numpy arrays
    points = np.asarray(points)
    vectors = np.asarray(vectors)
    
    # Validate input shapes
    assert points.shape[1] == 3, f"Points should have shape (N, 3), got {points.shape}"
    assert vectors.shape[1] == 3, f"Vectors should have shape (N, 3), got {vectors.shape}"
    assert points.shape[0] == vectors.shape[0], f"Points and vectors should have same length, got {points.shape[0]} vs {vectors.shape[0]}"

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
    color = np.asarray(color)
    if color.ndim == 1:
        # Single color for all lines - repeat for each line
        colors = np.tile(color, (len(points), 1))
    else:
        # Per-line colors
        colors = color
    
    assert colors.shape == (len(points), 3), f"Colors should have shape ({len(points)}, 3), got {colors.shape}"
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    # Add to visualizer
    vis.add_geometry(line_set)
    
    return vis, line_set

def update_vector_visualization(line_set, points, vectors, scale=1.0, color=None):
    """
    Update an existing line set with new vectors.
    
    Args:
        line_set (o3d.geometry.LineSet): The line set to update
        points (numpy.ndarray or torch.Tensor): Starting points of vectors, shape (N, 3)
        vectors (numpy.ndarray or torch.Tensor): Vectors to visualize, shape (N, 3)
        scale (float, optional): Scaling factor for vector lengths
        color (list or numpy.ndarray, optional): RGB color for vectors
                                               Can be single color [R,G,B] or per-vector colors [N,3]
        
    Returns:
        o3d.geometry.LineSet: The updated line set
    """
    import open3d as o3d

    # Convert torch tensors to numpy arrays
    if hasattr(points, 'detach'):  # torch tensor
        points = points.detach().cpu().numpy()
    if hasattr(vectors, 'detach'):  # torch tensor
        vectors = vectors.detach().cpu().numpy()
    
    # Ensure numpy arrays
    points = np.asarray(points)
    vectors = np.asarray(vectors)
    
    # Validate input shapes
    assert points.shape[1] == 3, f"Points should have shape (N, 3), got {points.shape}"
    assert vectors.shape[1] == 3, f"Vectors should have shape (N, 3), got {vectors.shape}"
    assert points.shape[0] == vectors.shape[0], f"Points and vectors should have same length, got {points.shape[0]} vs {vectors.shape[0]}"

    # Generate points
    end_points = points + vectors * scale
    all_points = np.vstack((points, end_points))
    line_set.points = o3d.utility.Vector3dVector(all_points)
    
    # Update lines if necessary (in case number of vectors changed)
    lines = [[i, i + len(points)] for i in range(len(points))]
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Update colors if provided
    if color is not None:
        color = np.asarray(color)
        if color.ndim == 1:
            # Single color for all lines - repeat for each line
            colors = np.tile(color, (len(points), 1))
        else:
            # Per-line colors
            colors = color
        
        assert colors.shape == (len(points), 3), f"Colors should have shape ({len(points)}, 3), got {colors.shape}"
        line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set
