import torch
import torch.nn
import torch.nn.functional as F
def weighted_chamfer_distance(points1, points2, weights1=None, weights2=None):
    """
    Compute the weighted Chamfer Distance between two point clouds.

    Args:
        points1 (torch.Tensor): Point cloud 1 of shape (N, D), where N is the number of points and D is the dimension.
        points2 (torch.Tensor): Point cloud 2 of shape (M, D), where M is the number of points and D is the dimension.
        weights1 (torch.Tensor, optional): Weights for points in points1 of shape (N,). Default is uniform weights.
        weights2 (torch.Tensor, optional): Weights for points in points2 of shape (M,). Default is uniform weights.

    Returns:
        torch.Tensor: Weighted Chamfer Distance.
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(points1, points2, p=2)  # Euclidean distance

    # Find nearest neighbors for points1 in points2
    min_dist1, _ = torch.min(dist_matrix, dim=1)
    if weights1 is not None:
        min_dist1 = min_dist1 * weights1

    # Find nearest neighbors for points2 in points1
    min_dist2, _ = torch.min(dist_matrix, dim=0)
    if weights2 is not None:
        min_dist2 = min_dist2 * weights2

    # Compute weighted Chamfer Distance
    chamfer_distance = torch.sum(min_dist1) + torch.sum(min_dist2)
    return chamfer_distance