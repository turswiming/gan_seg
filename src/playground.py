from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
import numpy as np
from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import ArgoverseSceneFlowSequenceLoader

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
        root_dir=Path("/workspace/av2data/train"),
        expected_camera_shape=(1280, 720),
        eval_args={},
        with_rgb=False,
        flow_data_path=Path("/workspace/av2flow/train"),
        range_crop_type="ego",
        point_size=8192,
        load_flow=False,
        load_boxes=True,
    )
    item = av2_sceneflow_zoo[0]
    print(type(item))
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
    
    # Visualize with class_ids
    print("\nVisualizing class_ids...")
    visualize_mask_3d(point_cloud_first, class_ids, "class_ids_3d.png", "Class IDs - 3D View")
    # visualize_mask_bev(point_cloud_first, class_ids, "class_ids_bev.png", "Class IDs - Bird's Eye View")
    # visualize_mask_multi_view(point_cloud_first, class_ids, "class_ids_multi.png")
    
    # If mask exists, visualize it too
    if mask is not None:
        print("\nVisualizing mask...")
        visualize_mask_3d(point_cloud_first, mask, "mask_3d.png", "Mask - 3D View")
        # # visualize_mask_bev(point_cloud_first, mask, "mask_bev.png", "Mask - Bird's Eye View")
        # # visualize_mask_multi_view(point_cloud_first, mask, "mask_multi.png")
        
        # # If mask has multiple classes, show per-class visualization
        # if len(np.unique(mask)) > 1 and len(np.unique(mask)) <= 20:
        #     visualize_mask_per_class(point_cloud_first, mask, "mask_per_class.png")
    
    print("\nAll visualizations saved!")

if __name__ == "__main__":
    main()



