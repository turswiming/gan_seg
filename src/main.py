import torch
from torch import nn
from torch.nn import functional as F
import open3d as o3d
from per_scene_dataset import PerSceneDataset
from scene_flow_predict_model import SceneFlowPredictor, Neural_Prior ,FLowPredictor
from mask_predict_model import MaskPredictor
from gan_loss import GanLoss
from loss2 import Loss_version2
from opticalflow_loss_3d import OpticalFlowLoss_3d
from objectsmooth import SmoothLoss
dataset = PerSceneDataset()

def pca(pred_mask, num_components=3):
    #PCA to 3D
    
    # Normalize the mask values for better PCA results
    normalized_mask = F.softmax(pred_mask*0.1, dim=1)
    
    # Center the data
    mean = torch.mean(normalized_mask, dim=0, keepdim=True)
    centered_data = normalized_mask - mean
    
    # Compute covariance matrix
    cov = torch.mm(centered_data.t(), centered_data) / (centered_data.size(0) - 1)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort eigenvalues and eigenvectors in descending order
    indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]
    
    # Select top 3 eigenvectors
    top_eigenvectors = eigenvectors[:, :3]
    
    # Project the data onto the top 3 eigenvectors
    color = torch.mm(centered_data, top_eigenvectors)
    
    # Scale to [0, 1] range for visualization
    min_vals = torch.min(color, dim=0, keepdim=True)[0]
    max_vals = torch.max(color, dim=0, keepdim=True)[0]
    color = (color - min_vals) / (max_vals - min_vals + 1e-8)
    return color

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


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PerSceneDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
infinite_loader = infinite_dataloader(dataloader)
sample = next(infinite_loader)
_, N, _ = sample["point_cloud_first"].shape
scene_flow_predictor = FLowPredictor(dim=3,pointSize=N)
scene_flow_predictor.to(device)
slot_num = 6
mask_predictor = MaskPredictor(slot_num=slot_num, point_length=N)
optimizer = torch.optim.AdamW(scene_flow_predictor.parameters(), lr=0.1, weight_decay=0.01)
optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=1, weight_decay=0.01)
criterion = Loss_version2(device=device)
criterion2 = OpticalFlowLoss_3d(device=device)
smooth_loss = SmoothLoss()

mse = nn.MSELoss()
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
gt_pcd = o3d.geometry.PointCloud()
reconstructed_pcd = o3d.geometry.PointCloud()
first_iteration = True
loop_step =10
step = 0
for sample in infinite_loader:
    step += 1
    # if (step // loop_step)%2 == 0:
    #     train_flow_model = True
    #     train_mask_model = False
    # else:
    #     train_flow_model = False
    #     train_mask_model = True
    # if train_flow_model:
    #     scene_flow_predictor.train()
    #     mask_predictor.eval()
    # else:
    #     scene_flow_predictor.eval()
    #     mask_predictor.train()
    pred_flow = scene_flow_predictor(sample["point_cloud_first"].to(device))
    pred_flow = pred_flow.view(-1, 3)
    gt_flow = torch.tensor(sample["flow"])
    gt_flow = gt_flow.to(pred_flow.device)
    pred_mask = mask_predictor(sample)
    loss, reconstructed_points = criterion(sample, pred_mask, pred_flow)
    loss2 = criterion2(sample, pred_mask, pred_flow)
    mse_loss = mse(pred_flow+sample["point_cloud_first"].to(device), reconstructed_points)
    loss = loss *1

    loss2 = loss2 * 10

    mse_loss = mse_loss * 0.001
    smooth_loss_value = smooth_loss(sample["point_cloud_first"].to(device).to(torch.float), pred_mask.permute(1,0).unsqueeze(0).to(torch.float))
    smooth_loss_value = smooth_loss_value * 0.01
    print("loss", loss.item())
    print("loss2", loss2.item())
    print("mse_loss", mse_loss.item())
    print("smooth_loss_value", smooth_loss_value.item())
    optimizer.zero_grad()
    optimizer_mask.zero_grad()
    pred_flow.retain_grad()
    pred_mask.retain_grad()
    sum_loss = loss + loss2 +mse_loss +smooth_loss_value
    # sum_loss = mse_loss
    sum_loss.backward()
    if pred_flow.grad is not None:
        print("pred_flow.grad", pred_flow.grad.std())
    if pred_mask.grad is not None:
        print("pred_mask.grad", pred_mask.grad.std())
    optimizer.step()
    optimizer_mask.step()
    #compare with gt_flow
    pred_flow = pred_flow.view(-1, 3)
    pred_flow = pred_flow.reshape(-1, 3)
    gt_flow = gt_flow.view(-1, 3)
    gt_flow = gt_flow.reshape(-1, 3)
    epe = torch.norm(pred_flow - gt_flow, dim=1)
    print("epe", epe.mean().item())
    pred_point = (sample["point_cloud_first"] + pred_flow.cpu().detach()).numpy()
    pcd.points = o3d.utility.Vector3dVector(pred_point.reshape(-1, 3))
    pred_mask = pred_mask.permute(1, 0)
    pred_mask = pred_mask.reshape(-1, slot_num)
    #PCA to 3D


    color = pca(pred_mask)
    pcd.colors = o3d.utility.Vector3dVector(color.cpu().detach().numpy())
    gt_pcd.points = o3d.utility.Vector3dVector(sample["point_cloud_second"].cpu().detach().numpy().reshape(-1, 3))
    gt_pcd.paint_uniform_color([0, 1, 0])
    reconstructed_pcd.points = o3d.utility.Vector3dVector(reconstructed_points.cpu().detach().numpy().reshape(-1, 3))
    reconstructed_pcd.paint_uniform_color([0, 0, 1])
    if first_iteration:
        # vis.add_geometry(pcd)
        vis.add_geometry(gt_pcd)
        vis.add_geometry(reconstructed_pcd)
        vis , lineset = visualize_vectors(
            sample["point_cloud_first"].reshape(-1, 3),
            pred_flow.cpu().detach().numpy().reshape(-1, 3),
            vis=vis,
            color=color.cpu().detach().numpy().reshape(-1, 3),
            )
        first_iteration = False
    else:
        # vis.update_geometry(pcd)
        lineset = update_vector_visualization(
            lineset,
            sample["point_cloud_first"].reshape(-1, 3),
            pred_flow.cpu().detach().numpy().reshape(-1, 3),
            color=color.cpu().detach().numpy().reshape(-1, 3),
            
        )
        vis.update_geometry(lineset)
        vis.update_geometry(gt_pcd)
        vis.update_geometry(reconstructed_pcd)
    vis.poll_events()
    vis.update_renderer()

# 关闭窗口
vis.destroy_window()