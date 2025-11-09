"""
Visualization script for Argoverse2 dataset segmentation masks.
Reads the argoverse2 dataset using AV2SceneFlowZoo and visualizes:
- Point clouds with segmentation masks
- Class IDs
- Instance masks
Uses matplotlib to create and save visualizations.
"""

from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
import numpy as np


def visualize_mask_3d(points, mask, save_path="mask_visualization_3d.png", title="Mask Visualization"):
    """
    Visualize mask on 3D point cloud.
    
    Args:
        points: (N, 3) numpy array or torch tensor
        mask: (N,) numpy array or torch tensor - can be binary, class labels, or probabilities
        save_path: output file path
        title: plot title
    """
    # Convert to numpy if needed
    if hasattr(points, 'numpy'):
        points = points.numpy()
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis limits
    ax.set_xlim(-48, 48)
    ax.set_ylim(-48, 48)
    ax.set_zlim(-48, 48)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
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
        cmap = plt.cm.tab20
    
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
    Visualize mask in Bird's Eye View (top-down).
    
    Args:
        points: (N, 3) numpy array or torch tensor
        mask: (N,) numpy array or torch tensor
        save_path: output file path
        title: plot title
    """
    # Convert to numpy if needed
    if hasattr(points, 'numpy'):
        points = points.numpy()
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    
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
        cmap = plt.cm.tab20
    
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
    Visualize mask from multiple viewpoints.
    
    Args:
        points: (N, 3) numpy array or torch tensor
        mask: (N,) numpy array or torch tensor
        save_path: output file path
    """
    # Convert to numpy if needed
    if hasattr(points, 'numpy'):
        points = points.numpy()
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    
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
        cmap = plt.cm.tab20
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
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2)
    
    # Side View (XZ)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(points[:, 0], points[:, 2], c=colors, cmap=cmap, s=1, vmin=vmin, vmax=vmax)
    ax3.set_title('Side View (XZ)')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Front View (YZ)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(points[:, 1], points[:, 2], c=colors, cmap=cmap, s=1, vmin=vmin, vmax=vmax)
    ax4.set_title('Front View (YZ)')
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-view visualization to {save_path}")


def visualize_mask_per_class(points, mask, save_path="mask_per_class.png", max_classes=20):
    """
    Visualize each class/mask value separately.
    
    Args:
        points: (N, 3) numpy array or torch tensor
        mask: (N,) numpy array or torch tensor with class labels
        save_path: output file path
        max_classes: maximum number of classes to visualize
    """
    # Convert to numpy if needed
    if hasattr(points, 'numpy'):
        points = points.numpy()
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    
    unique_classes = np.unique(mask)
    n_classes = min(len(unique_classes), max_classes)
    unique_classes = unique_classes[:n_classes]
    
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
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class visualization to {save_path}")


def predict_mask(mask_predictor, point_cloud, device):
    """
    Predict segmentation mask using the mask predictor model.
    
    Args:
        mask_predictor: Trained mask prediction model
        point_cloud: (N, 3) point cloud tensor
        device: torch device
        
    Returns:
        predicted_mask: (N,) predicted instance labels
        mask_probs: (N, K) predicted probabilities for K clusters
    """
    import torch
    mask_predictor.eval()
    
    with torch.no_grad():
        point_cloud = point_cloud.to(device).float()
        
        # Forward pass through mask predictor
        # mask_predictor typically outputs (K, N) where K is number of masks
        mask_predictor.to(device)
        mask_logits = mask_predictor(point_cloud.unsqueeze(0),point_cloud.unsqueeze(0))
        
        # Convert to probabilities
        if mask_logits.dim() == 2:
            # Shape: (K, N) -> transpose to (N, K)
            mask_probs = torch.softmax(mask_logits, dim=0).T  # (N, K)
        else:
            # Shape: (N, K)
            mask_probs = torch.softmax(mask_logits, dim=-1)
        
        # Get predicted class for each point
        predicted_mask = torch.argmax(mask_probs, dim=-1)  # (N,)
        
    return predicted_mask, mask_probs


def predict_flow(flow_predictor, point_cloud_first, point_cloud_next, device):
    """
    Predict scene flow using the flow predictor model.
    
    Args:
        flow_predictor: Trained flow prediction model
        point_cloud_first: (N, 3) first frame point cloud
        point_cloud_next: (N, 3) second frame point cloud
        device: torch device
        
    Returns:
        predicted_flow: (N, 3) predicted flow vectors
    """
    import torch
    flow_predictor.eval()
    
    with torch.no_grad():
        point_cloud_first = point_cloud_first.to(device).float()
        point_cloud_next = point_cloud_next.to(device).float()
        
        # Check if it's a FlowStep3D model that needs both frames
        from OGCModel.flownet_kitti import FlowStep3D
        if isinstance(flow_predictor, FlowStep3D):
            pc1 = point_cloud_first.unsqueeze(0)  # (1, N, 3)
            pc2 = point_cloud_next.unsqueeze(0)  # (1, N, 3)
            cascade_flow_outs = flow_predictor(pc1, pc2, pc1, pc2, iters=4)
            predicted_flow = cascade_flow_outs[-1].squeeze(0)  # (N, 3)
        else:
            # Simple forward pass for other models
            predicted_flow = flow_predictor(point_cloud_first)
        
    return predicted_flow


def visualize_comparison(points, gt_mask, pred_mask, save_path="comparison.png", title="GT vs Predicted"):
    """
    Visualize ground truth and predicted masks side by side.
    
    Args:
        points: (N, 3) numpy array
        gt_mask: (N,) ground truth mask
        pred_mask: (N,) predicted mask
        save_path: output file path
        title: plot title
    """
    # Convert to numpy if needed
    if hasattr(points, 'numpy'):
        points = points.numpy()
    if hasattr(gt_mask, 'numpy'):
        gt_mask = gt_mask.numpy()
    if hasattr(pred_mask, 'numpy'):
        pred_mask = pred_mask.cpu().numpy()
    
    fig = plt.figure(figsize=(24, 10))
    
    # Ground truth
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    scatter1 = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=gt_mask, cmap=plt.cm.tab20, s=1, alpha=0.8)
    ax1.set_title('Ground Truth', fontsize=16)
    ax1.set_xlim(-48, 48)
    ax1.set_ylim(-48, 48)
    ax1.set_zlim(-48, 48)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    plt.colorbar(scatter1, ax=ax1)
    
    # Predicted
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    scatter2 = ax2.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=pred_mask, cmap=plt.cm.tab20, s=1, alpha=0.8)
    ax2.set_title('Predicted', fontsize=16)
    ax2.set_xlim(-48, 48)
    ax2.set_ylim(-48, 48)
    ax2.set_zlim(-48, 48)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    plt.colorbar(scatter2, ax=ax2)
    
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison visualization to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize Argoverse2 segmentation masks')
    parser.add_argument('--root_dir', type=str, default='/workspace/av2data/train',
                        help='Root directory of AV2 dataset')
    parser.add_argument('--flow_data_path', type=str, default='/workspace/av2flow/train',
                        help='Path to flow data')
    parser.add_argument('--index', type=int, default=0,
                        help='Index of sample to visualize')
    parser.add_argument('--point_size', type=int, default=8192,
                        help='Number of points to sample')
    parser.add_argument('--output_dir', type=str, default='.',
                        help='Output directory for visualizations')
    parser.add_argument('--load_boxes', action='store_true',
                        help='Load bounding boxes (instance masks)')
    parser.add_argument('--load_flow', action='store_false',
                        help='Load flow data')
    parser.add_argument('--visualize_all', action='store_true',
                        help='Create all visualization types')
    parser.add_argument('--checkpoint_path', type=str, 
                        default='/workspace/gan_seg/outputs/exp/20251106_173046question/checkpoints/step_10800.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--predict', action='store_true',
                        help='Use model to predict masks and visualize predictions')
    parser.add_argument('--visualize_flow', action='store_false',
                        help='Visualize predicted flow (as flow magnitude)')
    
    args = parser.parse_args()
    
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models if prediction is requested
    flow_predictor = None
    mask_predictor = None
    from OGCModel.segnet_av2 import MaskFormer3D
    mask_predictor = MaskFormer3D(n_slot=16, n_point=8192, n_transformer_layer=2, transformer_embed_dim=128)
    args.predict = True
    if args.predict:
        print(f"Loading checkpoint from {args.checkpoint_path}...")
        ckpt = torch.load(args.checkpoint_path, map_location=device)
        mask_predictor.load_state_dict(ckpt["mask_predictor"])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    print(f"Loading AV2SceneFlowZoo dataset from {args.root_dir}...")
    dataset = AV2SceneFlowZoo(
        root_dir=Path(args.root_dir),
        expected_camera_shape=(1280, 720),
        eval_args={},
        with_rgb=False,
        flow_data_path=Path(args.flow_data_path),
        range_crop_type="ego",
        point_size=args.point_size,
        load_flow=args.load_flow,
        load_boxes=args.load_boxes,
    )
    
    # Load sample
    print(f"Loading sample {args.index}...")
    item = dataset[args.index]
    
    # Extract data
    point_cloud_first = item["point_cloud_first"]
    point_cloud_next = item["point_cloud_next"]
    mask = item.get("mask", None)
    flow = item.get("flow", None)
    class_ids = item["class_ids"]
    
    # Print statistics
    print(f"\n=== Data Statistics ===")
    print(f"Point cloud shape: {point_cloud_first.shape}")
    print(f"Class IDs shape: {class_ids.shape}")
    print(f"Unique class IDs: {np.unique(class_ids.numpy() if hasattr(class_ids, 'numpy') else class_ids)}")
    
    if mask is not None:
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Unique mask values: {np.unique(mask.numpy() if hasattr(mask, 'numpy') else mask)[:10]}...")  # Show first 10
    
    # === PREDICTION ===
    pred_mask = None
    pred_flow = None
    
    if args.predict and mask_predictor is not None:
        print(f"\n=== Model Prediction ===")
        print("Predicting segmentation mask...")
        pred_mask, mask_probs = predict_mask(mask_predictor, point_cloud_first, device)
        print(f"Predicted mask shape: {pred_mask.shape}")
        print(f"Unique predicted mask values: {np.unique(pred_mask.cpu().numpy())}")
        print(f"Number of predicted clusters: {len(np.unique(pred_mask.cpu().numpy()))}")
        
        if args.visualize_flow and flow_predictor is not None:
            print("Predicting scene flow...")
            pred_flow = predict_flow(flow_predictor, point_cloud_first, point_cloud_next, device)
            print(f"Predicted flow shape: {pred_flow.shape}")
            flow_magnitude = torch.norm(pred_flow, dim=-1)
            print(f"Flow magnitude - min: {flow_magnitude.min():.4f}, max: {flow_magnitude.max():.4f}, mean: {flow_magnitude.mean():.4f}")
    
    # === VISUALIZATION ===
    print(f"\n=== Visualization ===")
    
    # 1. Visualize class IDs (semantic categories)
    print("Visualizing class IDs...")
    visualize_mask_3d(point_cloud_first, class_ids, 
                      output_dir / f"class_ids_3d_sample_{args.index}.png", 
                      "Class IDs - 3D View")
    
    if args.visualize_all:
        visualize_mask_bev(point_cloud_first, class_ids, 
                          output_dir / f"class_ids_bev_sample_{args.index}.png", 
                          "Class IDs - Bird's Eye View")
        visualize_mask_multi_view(point_cloud_first, class_ids, 
                                 output_dir / f"class_ids_multi_sample_{args.index}.png")
        
        # Per-class visualization
        unique_classes = np.unique(class_ids.numpy() if hasattr(class_ids, 'numpy') else class_ids)
        if len(unique_classes) > 1 and len(unique_classes) <= 20:
            visualize_mask_per_class(point_cloud_first, class_ids, 
                                    output_dir / f"class_ids_per_class_sample_{args.index}.png")
    
    # 2. Visualize ground truth instance masks if available
    if mask is not None:
        print("Visualizing ground truth instance masks...")
        visualize_mask_3d(point_cloud_first, mask, 
                         output_dir / f"gt_mask_3d_sample_{args.index}.png", 
                         "Ground Truth Instance Mask - 3D View")
        
        if args.visualize_all:
            visualize_mask_bev(point_cloud_first, mask, 
                              output_dir / f"gt_mask_bev_sample_{args.index}.png", 
                              "Ground Truth Instance Mask - Bird's Eye View")
            visualize_mask_multi_view(point_cloud_first, mask, 
                                     output_dir / f"gt_mask_multi_sample_{args.index}.png")
            
            # Per-instance visualization
            unique_instances = np.unique(mask.numpy() if hasattr(mask, 'numpy') else mask)
            if len(unique_instances) > 1 and len(unique_instances) <= 20:
                visualize_mask_per_class(point_cloud_first, mask, 
                                        output_dir / f"gt_mask_per_instance_sample_{args.index}.png")
    
    # 3. Visualize predicted masks
    if pred_mask is not None:
        print("Visualizing predicted instance masks...")
        visualize_mask_3d(point_cloud_first, pred_mask.cpu(), 
                         output_dir / f"pred_mask_3d_sample_{args.index}.png", 
                         "Predicted Instance Mask - 3D View")
        
        if args.visualize_all:
            visualize_mask_bev(point_cloud_first, pred_mask, 
                              output_dir / f"pred_mask_bev_sample_{args.index}.png", 
                              "Predicted Instance Mask - Bird's Eye View")
            visualize_mask_multi_view(point_cloud_first, pred_mask, 
                                     output_dir / f"pred_mask_multi_sample_{args.index}.png")
        
        # 4. Visualize comparison between GT and predicted
        if mask is not None:
            print("Visualizing GT vs Predicted comparison...")
            visualize_comparison(point_cloud_first, mask, pred_mask,
                               output_dir / f"comparison_sample_{args.index}.png",
                               f"Sample {args.index}: Ground Truth vs Predicted")
    
    # 5. Visualize predicted flow
    if pred_flow is not None:
        print("Visualizing predicted scene flow...")
        import torch
        flow_magnitude = torch.norm(pred_flow, dim=-1)
        visualize_mask_3d(point_cloud_first, flow_magnitude, 
                         output_dir / f"pred_flow_magnitude_sample_{args.index}.png", 
                         "Predicted Flow Magnitude - 3D View")
        
        if args.visualize_all:
            visualize_mask_bev(point_cloud_first, flow_magnitude, 
                              output_dir / f"pred_flow_magnitude_bev_sample_{args.index}.png", 
                              "Predicted Flow Magnitude - Bird's Eye View")
        
        # Visualize ground truth flow if available
        if flow is not None:
            import torch
            gt_flow_magnitude = torch.norm(flow, dim=-1)
            visualize_mask_3d(point_cloud_first, gt_flow_magnitude, 
                             output_dir / f"gt_flow_magnitude_sample_{args.index}.png", 
                             "Ground Truth Flow Magnitude - 3D View")
    
    print(f"\n✓ All visualizations saved to {output_dir}")
    print(f"\n=== Summary ===")
    print(f"- Class IDs visualization: ✓")
    if mask is not None:
        print(f"- Ground truth mask visualization: ✓")
    if pred_mask is not None:
        print(f"- Predicted mask visualization: ✓")
        print(f"- GT vs Predicted comparison: ✓")
    if pred_flow is not None:
        print(f"- Predicted flow visualization: ✓")


if __name__ == "__main__":
    main()