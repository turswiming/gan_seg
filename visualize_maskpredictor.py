#!/usr/bin/env python3
"""
Mask Predictor Visualization Script
MaskPredictor网络可视化脚本

读取保存的maskpredictor网络，读取av2sequence数据集并可视化结果
Loads saved maskpredictor network, reads av2sequence dataset and visualizes results
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed. Please install with: pip install numpy")
    sys.exit(1)

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("Error: PyTorch not installed. Please install PyTorch")
    sys.exit(1)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    print("Warning: Open3D not installed. Only saving results to files.")
    print("Install with: pip install open3d")
    HAS_OPEN3D = False

# Import project modules
try:
    from dataset.av2_dataset import AV2SequenceDataset
    from model.mask_predict_model import OptimizedMaskPredictor, Neural_Mask_Prior, EulerMaskMLP
    from Predictor import get_mask_predictor
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def load_checkpoint(checkpoint_path, device):
    """
    加载checkpoint文件
    Load checkpoint file
    
    Args:
        checkpoint_path (str): 检查点文件路径 / Path to checkpoint file
        device (torch.device): 设备 / Device
        
    Returns:
        dict: 包含模型状态的字典 / Dictionary containing model states
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        print("Available keys in checkpoint:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        return checkpoint
    else:
        print("Checkpoint is not a dictionary, treating as direct model state")
        return {"mask_predictor": checkpoint}

def create_mask_predictor_from_checkpoint(checkpoint, device, point_length=65536):
    """
    从checkpoint创建mask predictor
    Create mask predictor from checkpoint
    
    Args:
        checkpoint (dict): 检查点数据 / Checkpoint data
        device (torch.device): 设备 / Device
        point_length (int): 点云长度 / Point cloud length
        
    Returns:
        torch.nn.Module: 加载的mask predictor模型 / Loaded mask predictor model
    """
    # 尝试从checkpoint中提取mask predictor的状态
    # Try to extract mask predictor state from checkpoint
    mask_state = None
    config = None
    
    if "mask_predictor" in checkpoint:
        mask_state = checkpoint["mask_predictor"]
    elif "model" in checkpoint:
        mask_state = checkpoint["model"]
    else:
        # 假设整个checkpoint就是模型状态
        # Assume the entire checkpoint is the model state
        mask_state = checkpoint
    
    # 尝试从checkpoint中获取配置信息
    # Try to get config from checkpoint
    if "config" in checkpoint:
        config = checkpoint["config"]
        print(f"Found config in checkpoint")
        if hasattr(config, 'model') and hasattr(config.model, 'mask'):
            mask_config = config.model.mask
            print(f"Mask model config: {mask_config}")
            
            # 使用配置信息创建正确的模型
            # Create correct model using config
            if mask_config.name == "EulerMaskMLP":
                from model.eulerflow_raw_mlp import ActivationFn
                print("Creating EulerMaskMLP from config")
                
                # 从配置中获取参数
                # Get parameters from config
                slot_num = mask_config.slot_num
                use_normalization = getattr(mask_config, 'use_normalization', False)
                normalization_type = getattr(mask_config, 'normalization_type', 'group_norm')
                
                if hasattr(mask_config, 'MLP'):
                    mlp_config = mask_config.MLP
                    num_layers = mlp_config.num_layers
                    num_hidden = mlp_config.num_hidden
                else:
                    num_layers = 8
                    num_hidden = 128
                
                model = EulerMaskMLP(
                    slot_num=slot_num,
                    filter_size=num_hidden,
                    act_fn=ActivationFn.LEAKYRELU,
                    layer_size=num_layers,
                    use_normalization=use_normalization,
                    normalization_type=normalization_type,
                )
                
                try:
                    model.load_state_dict(mask_state)
                    print("Successfully loaded EulerMaskMLP from checkpoint")
                    return model.to(device)
                except Exception as e:
                    print(f"Failed to load EulerMaskMLP: {e}")
            
            elif mask_config.name == "OptimizedMask":
                print("Creating OptimizedMaskPredictor from config")
                model = OptimizedMaskPredictor(
                    slot_num=mask_config.slot_num,
                    point_length=point_length
                )
                try:
                    model.load_state_dict(mask_state)
                    print("Successfully loaded OptimizedMaskPredictor from checkpoint")
                    return model.to(device)
                except Exception as e:
                    print(f"Failed to load OptimizedMaskPredictor: {e}")
            
            elif mask_config.name in ["NMP", "EularNMP"]:
                print(f"Creating Neural_Mask_Prior from config ({mask_config.name})")
                model_detail = mask_config.NMP
                input_dim = 3 if mask_config.name == "NMP" else 4
                
                model = Neural_Mask_Prior(
                    input_dim=input_dim,
                    slot_num=mask_config.slot_num,
                    filter_size=model_detail.num_hidden,
                    act_fn=model_detail.activation,
                    layer_size=model_detail.num_layers
                )
                try:
                    model.load_state_dict(mask_state)
                    print("Successfully loaded Neural_Mask_Prior from checkpoint")
                    return model.to(device)
                except Exception as e:
                    print(f"Failed to load Neural_Mask_Prior: {e}")
    
    # 从状态字典推断模型类型和参数 (fallback方法)
    # Infer model type and parameters from state dict (fallback method)
    state_keys = list(mask_state.keys()) if isinstance(mask_state, dict) else []
    
    # 检查是否是OptimizedMaskPredictor (只有tensor2d参数)
    # Check if it's OptimizedMaskPredictor (only has tensor2d parameter)
    if "tensor2d" in state_keys and len(state_keys) == 1:
        print("Detected OptimizedMaskPredictor")
        tensor_shape = mask_state["tensor2d"].shape
        slot_num, point_len = tensor_shape
        
        model = OptimizedMaskPredictor(slot_num=slot_num, point_length=point_len)
        model.load_state_dict(mask_state)
        return model.to(device)
    
    # 检查是否是EulerMaskMLP (复杂的层次结构)
    # Check if it's EulerMaskMLP (complex layer structure)
    elif any("nn_layers.0.1" in key for key in state_keys):
        print("Detected EulerMaskMLP-like structure")
        # EulerMaskMLP有复杂的嵌套结构，创建默认模型并尝试加载
        # EulerMaskMLP has complex nested structure, create default model and try to load
        from model.eulerflow_raw_mlp import ActivationFn
        
        # 尝试推断参数
        # Try to infer parameters
        slot_num = 30  # 默认值 / default value
        filter_size = 96  # 默认值 / default value
        layer_size = 12  # 默认值 / default value
        
        # 尝试从最后一层推断slot数量
        # Try to infer slot number from last layer
        for key in state_keys:
            if "36.weight" in key:  # 最后一层的权重 / last layer weight
                last_weight = mask_state[key]
                slot_num = last_weight.shape[0]
                break
        
        print(f"Inferred EulerMaskMLP parameters: slot_num={slot_num}, filter_size={filter_size}, layer_size={layer_size}")
        
        model = EulerMaskMLP(
            slot_num=slot_num,
            filter_size=filter_size,
            act_fn=ActivationFn.LEAKYRELU,
            layer_size=layer_size,
            use_normalization=True,
            normalization_type="group_norm",
        )
        
        try:
            model.load_state_dict(mask_state)
            print("Successfully loaded EulerMaskMLP")
            return model.to(device)
        except Exception as e:
            print(f"Failed to load EulerMaskMLP with inferred parameters: {e}")
    
    # 检查是否是Neural_Mask_Prior
    # Check if it's Neural_Mask_Prior
    elif any("nn_layers" in key and not "nn_layers.0.1" in key for key in state_keys):
        print("Detected Neural_Mask_Prior")
        # 从第一层推断输入维度
        # Infer input dimension from first layer
        first_layer_key = None
        for key in state_keys:
            if "nn_layers.0" in key and "weight" in key:
                first_layer_key = key
                break
        
        if first_layer_key:
            first_weight = mask_state[first_layer_key]
            input_dim = first_weight.shape[1]
            filter_size = first_weight.shape[0]
            
            # 从最后一层推断slot数量
            # Infer slot number from last layer
            last_layer_key = None
            max_layer_idx = -1
            for key in state_keys:
                if "nn_layers" in key and "weight" in key:
                    # 提取层索引
                    # Extract layer index
                    parts = key.split(".")
                    try:
                        layer_idx = int(parts[1])
                        if layer_idx > max_layer_idx:
                            max_layer_idx = layer_idx
                            last_layer_key = key
                    except (ValueError, IndexError):
                        continue
            
            if last_layer_key:
                last_weight = mask_state[last_layer_key]
                slot_num = last_weight.shape[0]
                
                # 估算层数
                # Estimate number of layers
                layer_count = max_layer_idx // 2 + 1  # 假设每层包含Linear和激活函数
                
                print(f"Inferred Neural_Mask_Prior parameters: input_dim={input_dim}, slot_num={slot_num}, filter_size={filter_size}, layers≈{layer_count}")
                
                # 创建模型并尝试加载
                # Create model and try to load
                model = Neural_Mask_Prior(
                    input_dim=input_dim,
                    slot_num=slot_num,
                    filter_size=filter_size,
                    layer_size=layer_count
                )
                
                try:
                    model.load_state_dict(mask_state)
                    return model.to(device)
                except Exception as e:
                    print(f"Failed to load Neural_Mask_Prior with inferred parameters: {e}")
    
    # 如果无法自动识别，创建默认模型
    # If cannot auto-detect, create default model
    print("Could not detect model type, creating default OptimizedMaskPredictor")
    model = OptimizedMaskPredictor(slot_num=10, point_length=point_length)
    
    try:
        model.load_state_dict(mask_state)
        print("Successfully loaded as OptimizedMaskPredictor")
    except Exception as e:
        print(f"Failed to load as OptimizedMaskPredictor: {e}")
        print("Using random initialized model")
    
    return model.to(device)

def create_colormap(num_colors, distinct=True):
    """
    为不同的mask创建颜色映射
    Create colormap for different masks
    
    Args:
        num_colors (int): 颜色数量 / Number of colors
        distinct (bool): 是否创建区分度高的颜色 / Whether to create distinct colors
        
    Returns:
        numpy.ndarray: 颜色映射数组 [num_colors, 3] / Colormap array [num_colors, 3]
    """
    if distinct and num_colors <= 20:
        # 为少量mask使用预定义的高区分度颜色
        # Use predefined high-contrast colors for small number of masks
        distinct_colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 1.0, 0.0],  # Green
            [0.0, 0.0, 1.0],  # Blue
            [1.0, 1.0, 0.0],  # Yellow
            [1.0, 0.0, 1.0],  # Magenta
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.5, 0.0],  # Orange
            [0.5, 0.0, 1.0],  # Purple
            [0.0, 0.5, 0.0],  # Dark Green
            [0.5, 0.5, 0.0],  # Olive
            [1.0, 0.7, 0.7],  # Light Red
            [0.7, 1.0, 0.7],  # Light Green
            [0.7, 0.7, 1.0],  # Light Blue
            [0.5, 0.5, 0.5],  # Gray
            [0.8, 0.4, 0.0],  # Brown
            [1.0, 0.8, 0.8],  # Pink
            [0.4, 0.8, 0.4],  # Light Olive
            [0.8, 0.8, 0.4],  # Beige
            [0.6, 0.4, 0.8],  # Lavender
            [0.4, 0.6, 0.8],  # Light Blue-Gray
        ]
        colors = distinct_colors[:num_colors]
        if len(colors) < num_colors:
            # 如果需要更多颜色，补充随机颜色
            # Add random colors if need more
            for i in range(len(colors), num_colors):
                np.random.seed(i)  # 确保可重复
                colors.append(np.random.rand(3))
        return np.array(colors)
    else:
        # 使用HSV颜色空间创建均匀分布的颜色
        # Create evenly distributed colors using HSV color space
        colors = []
        for i in range(num_colors):
            if num_colors > 100:
                # 对于大量mask，使用更随机的颜色分布
                # For many masks, use more random color distribution
                np.random.seed(i)
                hue = np.random.rand()
                saturation = 0.7 + 0.3 * np.random.rand()
                value = 0.8 + 0.2 * np.random.rand()
            else:
                hue = i / num_colors
                saturation = 0.8
                value = 0.9
            
            # Convert HSV to RGB
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append(rgb)
        
        return np.array(colors)

def visualize_masks(points, masks, window_name="Mask Visualization", save_path=None, top_k_masks=None):
    """
    可视化点云和预测的masks
    Visualize point cloud and predicted masks
    
    Args:
        points (numpy.ndarray): 点云坐标 [N, 3] / Point cloud coordinates [N, 3]
        masks (numpy.ndarray): Mask预测结果 [slot_num, N] / Mask predictions [slot_num, N]
        window_name (str): 窗口名称 / Window name
        save_path (str, optional): 保存路径 / Save path
        top_k_masks (int, optional): 只显示top-k个最重要的mask / Only show top-k most important masks
    """
    if not HAS_OPEN3D:
        print("Open3D not available, skipping visualization")
        if save_path:
            # 保存数据到文件
            # Save data to file
            np.savez(save_path, points=points, masks=masks)
            print(f"Saved data to {save_path}")
        return
    
    slot_num = masks.shape[0]
    
    # 如果指定了top_k_masks，选择最重要的mask
    # If top_k_masks is specified, select the most important masks
    if top_k_masks is not None and top_k_masks < slot_num:
        # 计算每个mask的重要性（平均激活值）
        # Calculate importance of each mask (mean activation)
        mask_importance = np.mean(masks, axis=1)
        top_k_indices = np.argsort(mask_importance)[-top_k_masks:]
        
        # 创建新的mask矩阵，只包含top-k masks
        # Create new mask matrix with only top-k masks
        filtered_masks = masks[top_k_indices]
        
        # 重新计算softmax，确保概率和为1
        # Recalculate softmax to ensure probabilities sum to 1
        filtered_masks = np.exp(filtered_masks) / np.sum(np.exp(filtered_masks), axis=0, keepdims=True)
        
        # 将masks转换为硬分配
        # Convert masks to hard assignment
        mask_assignments = np.argmax(filtered_masks, axis=0)
        effective_slot_num = top_k_masks
        
        print(f"Selected top {top_k_masks} masks out of {slot_num}")
        print(f"Top mask indices: {top_k_indices}")
        print(f"Top mask importance scores: {mask_importance[top_k_indices]}")
    else:
        # 使用所有masks
        # Use all masks
        mask_assignments = np.argmax(masks, axis=0)
        effective_slot_num = slot_num
    
    # 创建颜色映射
    # Create colormap
    if effective_slot_num <= 20:
        colormap = create_colormap(effective_slot_num, distinct=True)
    else:
        colormap = create_colormap(effective_slot_num, distinct=False)
    
    # 为每个点分配颜色
    # Assign colors to each point
    point_colors = colormap[mask_assignments]
    
    # 创建点云
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    
    # 创建坐标系
    # Create coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    
    # 可视化
    # Visualize
    geometries = [pcd, coord_frame]
    
    print(f"Visualizing {len(points)} points with {effective_slot_num} effective masks")
    assignment_counts = np.bincount(mask_assignments, minlength=effective_slot_num)
    print(f"Mask assignment distribution: {assignment_counts}")
    
    # 显示每个mask的点数统计
    # Show point count statistics for each mask
    print("\nMask usage statistics:")
    for i, count in enumerate(assignment_counts):
        if count > 0:
            percentage = (count / len(points)) * 100
            print(f"  Mask {i:2d}: {count:4d} points ({percentage:5.1f}%)")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=1600,
        height=1000,
        point_show_normal=False
    )
    
    if save_path:
        # 保存点云文件
        # Save point cloud file
        save_file = save_path.replace('.npz', '.ply') if save_path.endswith('.npz') else save_path + '.ply'
        o3d.io.write_point_cloud(save_file, pcd)
        print(f"Saved visualization to {save_file}")

def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description="Visualize MaskPredictor results on AV2 dataset")
    parser.add_argument("--checkpoint", "-c", type=str, 
                       default="/home/lzq/workspace/gan_seg/outputs/ablation/flow_smooth/AV2Sequence/ours_0.01/run_0/checkpoints/step_13000.pt",
                       help="Path to the checkpoint file")
    parser.add_argument("--frame_idx", "-f", type=int, default=50,
                       help="Frame index to visualize")
    parser.add_argument("--save_dir", "-s", type=str, default=None,
                       help="Directory to save visualization results")
    parser.add_argument("--device", "-d", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--top_k_masks", "-k", type=int, default=10,
                       help="Number of top masks to visualize (default: 10, set to -1 for all masks)")
    
    args = parser.parse_args()
    
    # 设置设备
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    try:
        # 1. 加载数据集
        # 1. Load dataset
        print("Loading AV2SequenceDataset...")
        dataset = AV2SequenceDataset(
            fix_ego_motion=True,
            max_k=1,
            apply_ego_motion=True
        )
        
        # 获取数据样本
        # Get data sample
        print(f"Loading frame {args.frame_idx}...")
        sample = dataset.get_item(args.frame_idx)
        
        if not sample:
            print(f"Failed to load frame {args.frame_idx}")
            return
        
        # 提取点云数据
        # Extract point cloud data
        points = sample['point_cloud_first']
        idx = sample['idx']
        total_frames = sample['total_frames']
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()
        
        print(f"Loaded point cloud with {len(points)} points")
        
        # 2. 加载模型
        # 2. Load model
        print("Loading mask predictor model...")
        checkpoint = load_checkpoint(args.checkpoint, device)
        
        # 创建mask predictor
        # Create mask predictor
        mask_predictor = create_mask_predictor_from_checkpoint(
            checkpoint, device, point_length=len(points)
        )
        mask_predictor.eval()
        
        print(f"Loaded mask predictor: {type(mask_predictor).__name__}")
        
        # 3. 进行推理
        # 3. Perform inference
        print("Running inference...")
        points_tensor = torch.from_numpy(points).float().to(device)
        
        with torch.no_grad():
            # 根据模型类型调用不同的forward方法
            # Call different forward methods based on model type
            if isinstance(mask_predictor, EulerMaskMLP):
                # EulerMaskMLP needs additional parameters: pc, idx, total_entries
                # For visualization, we use dummy values for idx and total_entries
                masks_pred = mask_predictor(points_tensor, idx=idx, total_entries=total_frames)
                # EulerMaskMLP的输出应该是 [N, slot_num]，需要转置为 [slot_num, N]
                # EulerMaskMLP output should be [N, slot_num], need to transpose to [slot_num, N]
                print(f"EulerMaskMLP output shape: {masks_pred.shape}")
                print(f"Points length: {len(points_tensor)}")
                
                # 检查输出格式并转置
                # Check output format and transpose
                if masks_pred.shape[0] == len(points_tensor) and masks_pred.shape[1] < masks_pred.shape[0]:
                    # 如果第一维等于点数且第二维较小，说明是 [N, slot_num] 格式
                    # If first dim equals point count and second dim is smaller, it's [N, slot_num] format
                    masks_pred = masks_pred.permute(1, 0)  # 转置为 [slot_num, N]
                    print(f"Transposed to: {masks_pred.shape}")
                elif masks_pred.shape[1] == len(points_tensor):
                    # 如果第二维等于点数，已经是 [slot_num, N] 格式
                    # If second dim equals point count, already [slot_num, N] format
                    print("Already in correct format [slot_num, N]")
                else:
                    print(f"Warning: Unexpected output shape {masks_pred.shape} for {len(points_tensor)} points")
            elif isinstance(mask_predictor, Neural_Mask_Prior):
                # Neural_Mask_Prior needs point coordinates as input
                masks_pred = mask_predictor(points_tensor)
                # 确保输出格式正确 [slot_num, N]
                # Ensure correct output format [slot_num, N]
                if masks_pred.shape[0] == len(points_tensor):
                    masks_pred = masks_pred.permute(1, 0)
            else:
                # OptimizedMaskPredictor doesn't need input coordinates
                masks_pred = mask_predictor(points_tensor)
            
            # 应用softmax获得概率分布
            # Apply softmax to get probability distribution
            masks_prob = F.softmax(masks_pred, dim=0)
            masks_np = masks_prob.detach().cpu().numpy()
        
        print(f"Generated masks with shape: {masks_np.shape}")
        
        # 4. 可视化结果
        # 4. Visualize results
        print("Visualizing results...")
        
        # 保存路径
        # Save path
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"frame_{args.frame_idx}_masks")
        
        # 可视化
        # Visualize
        top_k = None if args.top_k_masks == -1 else args.top_k_masks
        visualize_masks(
            points, 
            masks_np,
            window_name=f"MaskPredictor Results - Frame {args.frame_idx}",
            save_path=save_path,
            top_k_masks=top_k
        )
        
        # 打印统计信息
        # Print statistics
        print("\n=== Results Summary ===")
        print(f"Frame: {args.frame_idx}")
        print(f"Points: {len(points)}")
        print(f"Masks: {masks_np.shape[0]}")
        print(f"Mask probabilities range: [{masks_np.min():.4f}, {masks_np.max():.4f}]")
        
        # 显示每个mask的平均概率
        # Show average probability for each mask
        mask_means = np.mean(masks_np, axis=1)
        print("\nMask average probabilities:")
        for i, mean_prob in enumerate(mask_means):
            print(f"  Mask {i:2d}: {mean_prob:.4f}")
        
        print("\nVisualization completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
