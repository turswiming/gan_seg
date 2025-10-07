#!/usr/bin/env python3
"""
Mask Predictor Sequence Visualization Script
MaskPredictor网络序列可视化脚本

循环播放所有数据条目的mask预测结果
Loop through all data entries and visualize mask prediction results
"""

import sys
import os
import argparse
from pathlib import Path
import time

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
    
    # 如果无法从配置创建，使用fallback方法
    # If cannot create from config, use fallback method
    print("Using fallback method to create model")
    
    # 检查是否是EulerMaskMLP (复杂的层次结构)
    # Check if it's EulerMaskMLP (complex layer structure)
    state_keys = list(mask_state.keys()) if isinstance(mask_state, dict) else []
    if any("nn_layers.0.1" in key for key in state_keys):
        print("Detected EulerMaskMLP-like structure")
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
    
    # 创建默认模型
    # Create default model
    print("Creating default OptimizedMaskPredictor")
    model = OptimizedMaskPredictor(slot_num=10, point_length=point_length)
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

class SequenceVisualizer:
    """序列可视化器类 / Sequence visualizer class"""
    
    def __init__(self, mask_predictor, dataset, device, top_k_masks=10):
        """
        初始化序列可视化器
        Initialize sequence visualizer
        
        Args:
            mask_predictor: 训练好的mask预测模型 / Trained mask predictor model
            dataset: AV2SequenceDataset实例 / AV2SequenceDataset instance
            device: 计算设备 / Computing device
            top_k_masks: 显示的top-k mask数量 / Number of top-k masks to show
        """
        self.mask_predictor = mask_predictor
        self.dataset = dataset
        self.device = device
        self.top_k_masks = top_k_masks
        self.vis = None
        self.current_geometries = []
        
        # 预计算颜色映射
        # Pre-compute colormap
        self.colormap = create_colormap(top_k_masks, distinct=True)
        
    def predict_masks(self, points, idx, total_frames):
        """
        预测单帧的masks
        Predict masks for a single frame
        
        Args:
            points (numpy.ndarray): 点云坐标 [N, 3]
            idx (int): 帧索引
            total_frames (int): 总帧数
            
        Returns:
            numpy.ndarray: 预测的masks [top_k, N]
        """
        points_tensor = torch.from_numpy(points).float().to(self.device)
        
        with torch.no_grad():
            if isinstance(self.mask_predictor, EulerMaskMLP):
                # EulerMaskMLP需要额外参数
                # EulerMaskMLP needs additional parameters
                masks_pred = self.mask_predictor(points_tensor, idx=idx, total_entries=total_frames)
                
                # 检查输出格式并转置
                # Check output format and transpose
                if masks_pred.shape[0] == len(points_tensor) and masks_pred.shape[1] < masks_pred.shape[0]:
                    masks_pred = masks_pred.permute(1, 0)  # [slot_num, N]
                
            elif isinstance(self.mask_predictor, Neural_Mask_Prior):
                masks_pred = self.mask_predictor(points_tensor)
                if masks_pred.shape[0] == len(points_tensor):
                    masks_pred = masks_pred.permute(1, 0)
            else:
                masks_pred = self.mask_predictor(points_tensor)
            
            # 应用softmax获得概率分布
            # Apply softmax to get probability distribution
            masks_prob = F.softmax(masks_pred, dim=0)
            masks_np = masks_prob.detach().cpu().numpy()
        
        # 选择top-k最重要的masks
        # Select top-k most important masks
        if self.top_k_masks is not None and self.top_k_masks < masks_np.shape[0]:
            mask_importance = np.mean(masks_np, axis=1)
            top_k_indices = np.argsort(mask_importance)[-self.top_k_masks:]
            filtered_masks = masks_np[top_k_indices]
            
            # 重新计算softmax
            # Recalculate softmax
            filtered_masks = np.exp(filtered_masks) / np.sum(np.exp(filtered_masks), axis=0, keepdims=True)
            return filtered_masks
        
        return masks_np
    
    def create_point_cloud_geometry(self, points, masks):
        """
        创建点云几何体
        Create point cloud geometry
        
        Args:
            points (numpy.ndarray): 点云坐标 [N, 3]
            masks (numpy.ndarray): Mask预测结果 [top_k, N]
            
        Returns:
            o3d.geometry.PointCloud: 点云几何体
        """
        # 转换为硬分配
        # Convert to hard assignment
        mask_assignments = np.argmax(masks, axis=0)
        
        # 分配颜色
        # Assign colors
        point_colors = self.colormap[mask_assignments]
        
        # 创建点云
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        return pcd
    
    def init_visualizer(self, fix_camera=True):
        """
        初始化可视化器
        Initialize visualizer
        
        Args:
            fix_camera (bool): 是否固定摄像机视角 / Whether to fix camera viewpoint
        """
        if not HAS_OPEN3D:
            print("Open3D not available")
            return False
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="MaskPredictor Sequence Visualization", width=1600, height=1000)
        
        # 添加坐标系
        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
        self.vis.add_geometry(coord_frame)
        
        # 设置初始摄像机视角
        # Set initial camera viewpoint
        if fix_camera:
            self.setup_camera_view()
        
        return True
    
    def setup_camera_view(self):
        """
        设置固定的摄像机视角 (默认前视角)
        Setup fixed camera viewpoint (default front view)
        """
        self.apply_camera_preset("front")
    
    def apply_camera_preset(self, preset="front"):
        """
        应用摄像机预设
        Apply camera preset
        
        Args:
            preset (str): 预设名称 / Preset name
                - "front": 前视角 / Front view
                - "top": 鸟瞰视角 / Top view  
                - "diagonal": 斜视角 / Diagonal view
                - "side": 侧视角 / Side view
        """
        if not self.vis:
            return
        
        view_control = self.vis.get_view_control()
        
        if preset == "front":
            # 前视角 (从前方看)
            # Front view (looking from front)
            view_control.set_front([0, 0, -1])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, -1, 0])
            view_control.set_zoom(0.5)
            print("Camera set to front view")
            
        elif preset == "top":
            # 鸟瞰视角 (从上方看)
            # Bird's eye view (looking from above)
            view_control.set_front([0, 0, -1])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 1, 0])
            view_control.set_zoom(0.3)
            print("Camera set to top view")
            
        elif preset == "diagonal":
            # 斜视角 (45度角)
            # Diagonal view (45 degree angle)
            view_control.set_front([1, 1, -1])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 0, 1])
            view_control.set_zoom(0.4)
            print("Camera set to diagonal view")
            
        elif preset == "side":
            # 侧视角 (从侧面看)
            # Side view (looking from side)
            view_control.set_front([1, 0, 0])
            view_control.set_lookat([0, 0, 0])
            view_control.set_up([0, 0, 1])
            view_control.set_zoom(0.4)
            print("Camera set to side view")
            
        else:
            print(f"Unknown camera preset: {preset}, using front view")
            self.apply_camera_preset("front")
    
    def save_camera_parameters(self, filename="camera_params.json"):
        """
        保存当前摄像机参数到文件
        Save current camera parameters to file
        
        Args:
            filename (str): 保存文件名 / Save filename
        """
        if not self.vis:
            return
        
        view_control = self.vis.get_view_control()
        camera_params = view_control.convert_to_pinhole_camera_parameters()
        
        # 保存到JSON文件
        # Save to JSON file
        o3d.io.write_pinhole_camera_parameters(filename, camera_params)
        print(f"Camera parameters saved to {filename}")
    
    def load_camera_parameters(self, filename="camera_params.json"):
        """
        从文件加载摄像机参数
        Load camera parameters from file
        
        Args:
            filename (str): 参数文件名 / Parameter filename
        """
        if not self.vis or not os.path.exists(filename):
            return False
        
        try:
            camera_params = o3d.io.read_pinhole_camera_parameters(filename)
            view_control = self.vis.get_view_control()
            view_control.convert_from_pinhole_camera_parameters(camera_params)
            print(f"Camera parameters loaded from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load camera parameters: {e}")
            return False
    
    def update_frame(self, frame_idx, preserve_camera=True):
        """
        更新单帧显示
        Update single frame display
        
        Args:
            frame_idx (int): 帧索引
            preserve_camera (bool): 是否保持摄像机位置不变
            
        Returns:
            bool: 是否成功更新
        """
        try:
            # 保存当前摄像机状态 (如果需要保持)
            # Save current camera state (if preserve is needed)
            camera_params = None
            if preserve_camera and self.vis:
                view_control = self.vis.get_view_control()
                camera_params = view_control.convert_to_pinhole_camera_parameters()
            
            # 获取数据样本
            # Get data sample
            sample = self.dataset.get_item(frame_idx)
            if not sample:
                return False
            
            # 提取数据
            # Extract data
            points = sample['point_cloud_first']
            idx = sample['idx']
            total_frames = sample['total_frames']
            
            if isinstance(points, torch.Tensor):
                points = points.detach().cpu().numpy()
            
            # 预测masks
            # Predict masks
            masks = self.predict_masks(points, idx, total_frames)
            
            # 创建新的点云几何体
            # Create new point cloud geometry
            new_pcd = self.create_point_cloud_geometry(points, masks)
            
            # 移除旧的点云
            # Remove old point cloud
            for geom in self.current_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)  # 不重置边界框
            self.current_geometries.clear()
            
            # 添加新的点云
            # Add new point cloud
            self.vis.add_geometry(new_pcd, reset_bounding_box=False)  # 不重置边界框
            self.current_geometries.append(new_pcd)
            
            # 恢复摄像机状态
            # Restore camera state
            if preserve_camera and camera_params is not None:
                view_control = self.vis.get_view_control()
                view_control.convert_from_pinhole_camera_parameters(camera_params)
            
            # 更新显示
            # Update display
            self.vis.update_geometry(new_pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            
            # 显示信息
            # Display info
            assignment_counts = np.bincount(np.argmax(masks, axis=0), minlength=masks.shape[0])
            print(f"\rFrame {frame_idx:3d}: {len(points):4d} points, {masks.shape[0]} masks, "
                  f"distribution: {assignment_counts}", end="", flush=True)
            
            return True
            
        except Exception as e:
            print(f"\nError updating frame {frame_idx}: {e}")
            return False
    
    def play_sequence(self, start_frame=0, end_frame=None, fps=5, loop=True, save_frames=False, save_dir=None, 
                     fix_camera=True, camera_preset="front", save_camera_params=False):
        """
        播放序列
        Play sequence
        
        Args:
            start_frame (int): 起始帧
            end_frame (int): 结束帧 (None表示到最后)
            fps (float): 播放帧率
            loop (bool): 是否循环播放
            save_frames (bool): 是否保存帧
            save_dir (str): 保存目录
            fix_camera (bool): 是否固定摄像机
            camera_preset (str): 摄像机预设 ("front", "top", "diagonal")
            save_camera_params (bool): 是否保存摄像机参数
        """
        if not self.init_visualizer(fix_camera=fix_camera):
            print("Failed to initialize visualizer")
            return
        
        # 设置摄像机预设
        # Set camera preset
        if fix_camera:
            self.apply_camera_preset(camera_preset)
            print(f"Camera position will be preserved during sequence playback")
        
        if end_frame is None:
            end_frame = len(self.dataset)
        
        end_frame = min(end_frame, len(self.dataset))
        
        if save_frames and save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        frame_delay = 1.0 / fps
        first_frame_processed = False
        
        try:
            while True:  # 外层循环用于循环播放
                print(f"\nPlaying sequence from frame {start_frame} to {end_frame-1} at {fps} FPS")
                
                for frame_idx in range(start_frame, end_frame):
                    start_time = time.time()
                    
                    # 更新帧，保持摄像机位置
                    # Update frame, preserve camera position
                    if not self.update_frame(frame_idx, preserve_camera=fix_camera):
                        print(f"\nFailed to update frame {frame_idx}")
                        continue
                    
                    # 在第一帧后保存摄像机参数（如果需要）
                    # Save camera parameters after first frame (if needed)
                    if not first_frame_processed and save_camera_params:
                        if save_dir:
                            camera_file = os.path.join(save_dir, "camera_params.json")
                        else:
                            camera_file = "camera_params.json"
                        self.save_camera_parameters(camera_file)
                        first_frame_processed = True
                    
                    # 保存帧（如果需要）
                    # Save frame (if needed)
                    if save_frames and save_dir:
                        screenshot_path = os.path.join(save_dir, f"frame_{frame_idx:04d}.png")
                        self.vis.capture_screen_image(screenshot_path)
                    
                    # 控制帧率
                    # Control frame rate
                    elapsed = time.time() - start_time
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                    
                    # 检查窗口是否被关闭
                    # Check if window is closed
                    if not self.vis.poll_events():
                        print("\nVisualization window closed")
                        return
                
                if not loop:
                    break
                
                print(f"\nLoop completed, restarting...")
                
        except KeyboardInterrupt:
            print(f"\nPlayback interrupted by user")
        
        finally:
            if self.vis:
                self.vis.destroy_window()

def main():
    """主函数 / Main function"""
    parser = argparse.ArgumentParser(description="Visualize MaskPredictor sequence on AV2 dataset")
    parser.add_argument("--checkpoint", "-c", type=str, 
                       default="/home/lzq/workspace/gan_seg/outputs/ablation/flow_smooth/AV2Sequence/ours_0.01/run_0/checkpoints/step_13000.pt",
                       help="Path to the checkpoint file")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="Start frame index")
    parser.add_argument("--end_frame", type=int, default=None,
                       help="End frame index (None for all)")
    parser.add_argument("--fps", type=float, default=2.0,
                       help="Playback frame rate")
    parser.add_argument("--top_k_masks", "-k", type=int, default=30,
                       help="Number of top masks to visualize")
    parser.add_argument("--device", "-d", type=str, default="auto",
                       help="Device to use (cuda/cpu/auto)")
    parser.add_argument("--loop", action="store_true", default=True,
                       help="Loop playback")
    parser.add_argument("--save_frames", action="store_true",
                       help="Save frames as images")
    parser.add_argument("--save_dir", "-s", type=str, default=None,
                       help="Directory to save frames")
    parser.add_argument("--camera_preset", type=str, default="front",
                       choices=["front", "top", "diagonal", "side"],
                       help="Camera viewpoint preset")
    parser.add_argument("--no_fix_camera", action="store_true",
                       help="Don't fix camera view (allow manual control)")
    parser.add_argument("--save_camera", action="store_true",
                       help="Save camera parameters after first frame")
    
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
        
        print(f"Dataset length: {len(dataset)}")
        
        # 2. 加载模型
        # 2. Load model
        print("Loading mask predictor model...")
        checkpoint = load_checkpoint(args.checkpoint, device)
        
        # 创建mask predictor
        # Create mask predictor
        mask_predictor = create_mask_predictor_from_checkpoint(
            checkpoint, device, point_length=65536
        )
        mask_predictor.eval()
        
        print(f"Loaded mask predictor: {type(mask_predictor).__name__}")
        
        # 3. 创建序列可视化器
        # 3. Create sequence visualizer
        visualizer = SequenceVisualizer(
            mask_predictor=mask_predictor,
            dataset=dataset,
            device=device,
            top_k_masks=args.top_k_masks
        )
        
        # 4. 播放序列
        # 4. Play sequence
        print(f"Starting sequence playback...")
        print(f"Camera preset: {args.camera_preset}")
        print(f"Fixed camera: {not args.no_fix_camera}")
        print(f"Controls: Close window or press Ctrl+C to stop")
        
        visualizer.play_sequence(
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            fps=args.fps,
            loop=args.loop,
            save_frames=args.save_frames,
            save_dir=args.save_dir,
            fix_camera=not args.no_fix_camera,
            camera_preset=args.camera_preset,
            save_camera_params=args.save_camera
        )
        
        print("Sequence playback completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
