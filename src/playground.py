from pathlib import Path
import argparse
from torch._tensor import Tensor
import matplotlib.pyplot as plt
from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
import numpy as np
from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import ArgoverseSceneFlowSequenceLoader
import torch

def visualize_mask_3d(points, mask, save_path="mask_visualization_3d.png", title="Mask Visualization"):
    """
    Visualize mask on 3D point cloud
    Args:
        points: (N, 3) numpy array
        mask: (N,) numpy array - can be binary, class labels, or probabilities
        save_path: output file path
        title: plot title
    """
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    ax.set_xlim(-48, 48)
    ax.set_ylim(-48, 48)
    ax.set_zlim(-48, 48)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Handle different mask types
    if mask.dtype == bool:
        # Binary mask
        colors = mask.astype(int)
        cmap = plt.cm.RdYlGn
    elif mask.max() <= 1.0 and mask.min() >= 0.0:
        # Probability mask
        colors = mask
        cmap = plt.cm.viridis
    else:
        # Class labels
        colors = mask
        cmap = plt.cm.tab10
    print("colors shape", colors.shape)
    scatter = ax.scatter(
        points[:, 0], 
        points[:, 1], 
        points[:, 2],
        c=colors,
        cmap=cmap,
        s=1,
        alpha=0.8
    )
    
    plt.colorbar(scatter, ax=ax, label='Mask Value')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D visualization to {save_path}")

def visualize_mask_bev(points, mask, save_path="mask_visualization_bev.png", title="Bird's Eye View"):
    """
    Visualize mask in Bird's Eye View (top-down)
    Args:
        points: (N, 3) numpy array
        mask: (N,) numpy array
        save_path: output file path
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Handle different mask types
    if mask.dtype == bool:
        colors = mask.astype(int)
        cmap = plt.cm.RdYlGn
    elif mask.max() <= 1.0 and mask.min() >= 0.0:
        colors = mask
        cmap = plt.cm.viridis
    else:
        colors = mask
        cmap = plt.cm.tab10
    print("colors shape", colors.shape)
    scatter = ax.scatter(
        points[:, 0],  # X
        points[:, 1],  # Y
        c=colors,
        cmap=cmap,
        s=1,
        alpha=0.8
    )
    
    ax.set_xlim(-48, 48)
    ax.set_ylim(-48, 48)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.colorbar(scatter, ax=ax, label='Mask Value')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved BEV visualization to {save_path}")

def visualize_mask_multi_view(points, mask, save_path="mask_visualization_multi.png"):
    """
    Visualize mask from multiple viewpoints
    Args:
        points: (N, 3) numpy array
        mask: (N,) numpy array
        save_path: output file path
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Handle different mask types
    if mask.dtype == bool:
        colors = mask.astype(int)
        cmap = plt.cm.RdYlGn
        vmin, vmax = 0, 1
    elif mask.max() <= 1.0 and mask.min() >= 0.0:
        colors = mask
        cmap = plt.cm.viridis
        vmin, vmax = 0, 1
    else:
        colors = mask
        cmap = plt.cm.tab10
        vmin, vmax = mask.min(), mask.max()
    
    # 3D view
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, cmap=cmap, s=1, vmin=vmin, vmax=vmax)
    ax1.set_title('3D View')
    ax1.set_xlim(-48, 48)
    ax1.set_ylim(-48, 48)
    ax1.set_zlim(-48, 48)
    
    # Bird's Eye View (XY)
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], c=colors, cmap=cmap, s=1, vmin=vmin, vmax=vmax)
    ax2.set_title("Bird's Eye View (XY)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2)
    
    # Side View (XZ)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(points[:, 0], points[:, 2], c=colors, cmap=cmap, s=1, vmin=vmin, vmax=vmax)
    ax3.set_title('Side View (XZ)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Front View (YZ)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(points[:, 1], points[:, 2], c=colors, cmap=cmap, s=1, vmin=vmin, vmax=vmax)
    ax4.set_title('Front View (YZ)')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-view visualization to {save_path}")


def visualize_flow_bev(
    point_cloud_first,
    point_cloud_next,
    flow,
    save_path="flow_bev.png",
    title="Bird's Eye View - Scene Flow",
    xlim=(-50, 50),
    ylim=(-50, 50),
    subsample=8912,  # 子采样以提高可视化速度
    arrow_scale=1.0,  # flow 箭头缩放
):
    """
    俯瞰视角可视化点云和 flow
    
    Args:
        point_cloud_first: (N, 3) - 第一帧点云
        point_cloud_next: (N, 3) - 第二帧点云
        flow: (N, 3) - scene flow 向量
        save_path: 保存路径
        title: 图标题
        xlim: X 轴范围
        ylim: Y 轴范围
        subsample: 子采样数量（加快绘制）
        arrow_scale: flow 箭头长度缩放因子
    """
    # 转换为 numpy
    if torch.is_tensor(point_cloud_first):
        point_cloud_first = point_cloud_first.cpu().numpy()
    if torch.is_tensor(point_cloud_next):
        point_cloud_next = point_cloud_next.cpu().numpy()
    if torch.is_tensor(flow):
        flow = flow.cpu().numpy()
    
    N = point_cloud_first.shape[0]
    
    # 子采样
    if N > subsample:
        indices = np.random.choice(N, subsample, replace=False)
        pc1_vis = point_cloud_first[indices]
        pc2_vis = point_cloud_next[indices]
        flow_vis = flow[indices]
    else:
        pc1_vis = point_cloud_first
        pc2_vis = point_cloud_next
        flow_vis = flow
    
    # 创建图
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 绘制第一帧点云（蓝色）
    ax.scatter(
        pc1_vis[:, 0], 
        pc1_vis[:, 1], 
        c='blue', 
        s=5, 
        alpha=0.6, 
        label='Point Cloud First'
    )
    
    # 绘制第二帧点云（红色）
    ax.scatter(
        pc2_vis[:, 0], 
        pc2_vis[:, 1], 
        c='red', 
        s=5, 
        alpha=0.6, 
        label='Point Cloud Next'
    )

    # 绘制 flow 箭头（绿色）
    # 只绘制部分箭头以避免过于密集
    n_arrows = len(pc1_vis)
    arrow_indices = np.random.choice(len(pc1_vis), n_arrows, replace=False)
    
    for idx in arrow_indices:
        ax.arrow(
            pc1_vis[idx, 0],  # x
            pc1_vis[idx, 1],  # y
            flow_vis[idx, 0] * arrow_scale,  # dx
            flow_vis[idx, 1] * arrow_scale,  # dy
            head_width=0.3,
            head_length=0.5,
            fc='green',
            ec='green',
            alpha=0.7,
            linewidth=0.5
        )
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # 添加统计信息
    flow_magnitude = np.linalg.norm(flow_vis, axis=1)
    stats_text = f"Points: {N}\nFlow mean: {flow_magnitude.mean():.3f}m\nFlow max: {flow_magnitude.max():.3f}m"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存俯瞰图到: {save_path}")

def visualize_mask_per_class(points, mask, save_path="mask_per_class.png"):
    """
    Visualize each class/mask value separately
    Args:
        points: (N, 3) numpy array
        mask: (N,) numpy array with class labels
        save_path: output file path
    """
    unique_classes = np.unique(mask)
    n_classes = len(unique_classes)
    
    # Determine grid size
    cols = min(4, n_classes)
    rows = (n_classes + cols - 1) // cols
    
    fig = plt.figure(figsize=(5*cols, 5*rows))
    
    for idx, cls in enumerate(unique_classes):
        ax = fig.add_subplot(rows, cols, idx+1, projection='3d')
        
        # Filter points for this class
        class_mask = mask == cls
        class_points = points[class_mask]
        
        ax.scatter(class_points[:, 0], class_points[:, 1], class_points[:, 2], s=1, alpha=0.8)
        ax.set_title(f'Class {cls} ({class_mask.sum()} points)')
        ax.set_xlim(-48, 48)
        ax.set_ylim(-48, 48)
        ax.set_zlim(-48, 48)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class visualization to {save_path}")

def main():
    av2_sceneflow_zoo = AV2SceneFlowZoo(
        root_dir=Path("/workspace/av2data/val"),
        expected_camera_shape=(1280, 720),
        eval_args={},
        with_rgb=False,
        flow_data_path=Path("/workspace/av2flow/val"),
        range_crop_type="ego",
        point_size=8192,
        load_flow=False,
        load_boxes=True,
        min_instance_size=50,
    )
    from dataset.kittisf_sceneflow import KittisfSceneFlowDataset
    kittisf_sceneflow_dataset = KittisfSceneFlowDataset(
        data_root=Path("/workspace/kittisf_downwampled/kittisf_downsampled/data"),
        split="train",
        num_points=8192,
        seed=42,
        augmentation=False,
    )
    from torch.utils.data import DataLoader
    dataloader = DataLoader(av2_sceneflow_zoo, batch_size=10, shuffle=False, num_workers=0)
    import tqdm
    from time import sleep
    import time
    time_start = time.time()
    time_last_update = time.time()
    time_seq = []
    all_iter = len(dataloader)
    mask_size = []
    count = 0
    for item in tqdm.tqdm(dataloader, total=all_iter, desc="Processing dataset", unit="samples"):
        #print mask shape
        mask = item["mask"]
        print(f"mask shape: {mask.shape}")
        for i in range(mask.shape[0]):
            print(f"mask[i] shape: {mask[i].shape}")
            if mask[i].shape[0] < 8192:
                continue
            mask_onehot = torch.nn.functional.one_hot(mask[i].long())
            print(f"mask_onehot shape: {mask_onehot.shape}")
            mask_onehot = mask_onehot[:, mask_onehot.sum(dim=0) > 50]
            mask_onehot = mask_onehot.float()
            print(f"mask_onehot shape: {mask_onehot.shape}")
            mask_onehot[:,0] += 0.01 # add a small value to the background class to avoid the background class is not used
            mask_res = torch.argmax(mask_onehot, dim=1)
            mask_size.append(max(mask_res))
            count += 1
        print(f"mask shape: {mask_size[-1]}")
        if count > 500:
            break
    print(f'mean of mask size: {np.mean(mask_size)}')
    print(f'max of mask size: {np.max(mask_size)}')
    print(f'min of mask size: {np.min(mask_size)}')
    print(f'std of mask size: {np.std(mask_size)}')
    print(f'median of mask size: {np.median(mask_size)}')
    print(f"95% of mask size: {np.percentile(mask_size, 95)}")
    print(f"99% of mask size: {np.percentile(mask_size, 99)}")
    exit()
    print(type[str, Tensor](item))
    point_cloud_first = item["point_cloud_first"]
    point_cloud_next = item["point_cloud_next"]
    mask = item["mask"] if "mask" in item else None
    flow = item["flow"] if "flow" in item else None
    class_ids = item["class_ids"]
    
    # Print mask statistics
    if mask is not None:
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask range: [{mask.min()}, {mask.max()}]")
        print(f"Unique values: {np.unique(mask)}")
    from losses.KNNDistanceLoss import KNNDistanceLoss
    knn_distance_loss = KNNDistanceLoss()
    with torch.no_grad():
        knn_distance = knn_distance_loss((point_cloud_first+flow).unsqueeze(0), point_cloud_next.unsqueeze(0), bidirectional=False)
        print(f"KNN distance: {knn_distance}")
    exit()
    # Visualize with class_ids
    print("\nVisualizing class_ids...")
    print("class_ids shape", class_ids.shape)
    print("class_ids", class_ids.max(), class_ids.min())
    visualize_mask_3d(point_cloud_first, class_ids, "class_ids_3d.png", "Class IDs - 3D View")
    visualize_mask_bev(point_cloud_first, class_ids, "class_ids_bev.png", "Class IDs - Bird's Eye View")
    visualize_mask_multi_view(point_cloud_first, class_ids, "class_ids_multi.png")
    visualize_flow_bev(point_cloud_first, point_cloud_next, flow, "flow_bev.png", "Flow - Bird's Eye View")
    
    # If mask exists, visualize it too
    if mask is not None:
        print("\nVisualizing mask...")
        visualize_mask_3d(point_cloud_first, mask, "mask_3d.png", "Mask - 3D View")
        visualize_mask_bev(point_cloud_first, mask, "mask_bev.png", "Mask - Bird's Eye View")
        visualize_mask_multi_view(point_cloud_first, mask, "mask_multi.png")
        
        # # If mask has multiple classes, show per-class visualization
        # if len(np.unique(mask)) > 1 and len(np.unique(mask)) <= 20:
        #     visualize_mask_per_class(point_cloud_first, mask, "mask_per_class.png")
    
    print("\nAll visualizations saved!")

if __name__ == "__main__":
    main()



