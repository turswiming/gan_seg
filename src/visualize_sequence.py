"""
Point Cloud Sequence Visualization Tool
Visualize entire point cloud sequences in a single Open3D window
"""

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
from typing import List, Optional
import time

from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_scene_flow import (
    ArgoverseSceneFlowSequenceLoader,
)


class PointCloudSequenceVisualizer:
    """Point Cloud Sequence Visualizer"""
    
    def __init__(self, sequence_loader: ArgoverseSceneFlowSequenceLoader, 
                 sequence_idx: int = 0, 
                 max_frames: Optional[int] = None,
                 point_size: float = 1.0):
        """
        Initialize the visualizer
        
        Args:
            sequence_loader: Sequence loader
            sequence_idx: Sequence index to visualize
            max_frames: Maximum number of frames, None for all frames
            point_size: Size of point cloud points
        """
        self.loader = sequence_loader
        self.sequence_idx = sequence_idx
        self.max_frames = max_frames
        self.point_size = point_size
        
        # 获取序列
        self.sequence = self.loader[sequence_idx]
        self.total_frames = len(self.sequence)
        if self.max_frames is not None:
            self.total_frames = min(self.total_frames, self.max_frames)
        
        print(f"Sequence {sequence_idx} total frames: {self.total_frames}")
        
        # Store point cloud data for all frames
        self.point_clouds = []
        self.colors = []
        self.current_frame = 0
        # Optional flow data aligned to rendered points (numpy arrays per frame)
        self.flow_points_list = []  # list[np.ndarray | None], each shape (M,3)
        self.flow_vecs_list = []    # list[np.ndarray | None], each shape (M,3)
        # Optional flow data aligned to rendered points (numpy arrays per frame)
        self.flow_points_list = []  # list[np.ndarray | None], each shape (M,3)
        self.flow_vecs_list = []    # list[np.ndarray | None], each shape (M,3)
        
    def load_sequence_data(self):
        """Load point cloud data for the entire sequence"""
        print("Loading sequence data...")
        
        for frame_idx in range(self.total_frames):
            try:
                # Load frame data
                frame, _ = self.sequence.load(frame_idx, 0, with_flow=True)
                
                # Get point cloud data - use global coordinate system
                if hasattr(frame.pc, 'global_pc'):
                    print("Global pc found")
                    points = frame.pc.global_pc.points
                    print("unfull global pc points shape", points.shape)
                    # print(frame.keys())
                else:
                    print("No global pc found, using ego coordinate system")
                    # Fallback: use ego coordinate system
                    points = frame.pc.pc.points
                
                self.point_clouds.append(points)
                
                # Assign different colors for each frame
                color = self._get_frame_color(frame_idx)
                self.colors.append(color)

                # Try to extract flow (in global frame) aligned with visible points
                flow_points = None
                flow_vecs = None
                try:
                    flow = getattr(frame, "flow", None)
                    if flow is not None:
                        flow_mask = getattr(flow, "mask", None)
                        # Prefer full global pc when available
                        # full_global_pc = getattr(frame.pc, "full_global_pc", None)
                        global_pc = getattr(frame.pc, "global_pc", None)
                        if global_pc is not None:
                            base_points = global_pc.points
                            if hasattr(flow, "valid_flow") and flow.valid_flow is not None:
                                vecs = flow.valid_flow
                                if vecs.shape[0] == base_points.shape[0]:
                                    flow_points = base_points
                                    flow_vecs = vecs
                except Exception as e:
                    print(f"Warning: failed to extract flow for frame {frame_idx}: {e}")

                self.flow_points_list.append(flow_points)
                self.flow_vecs_list.append(flow_vecs)

                if frame_idx % 10 == 0:
                    print(f"Loaded {frame_idx + 1}/{self.total_frames} frames")
                    
            except Exception as e:
                print(f"Error loading frame {frame_idx}: {e}")
                continue
                
        print(f"Successfully loaded {len(self.point_clouds)} frames")
    
    def _get_frame_color(self, frame_idx: int) -> np.ndarray:
        """Generate different colors for each frame"""
        # Use HSV color space to generate evenly distributed colors
        hue = (frame_idx / self.total_frames) % 1.0
        saturation = 0.8
        value = 0.9
        
        # HSV to RGB
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return np.array([r, g, b])
    
    def create_combined_pointcloud(self, frame_indices: List[int]) -> o3d.geometry.PointCloud:
        """Create combined point cloud"""
        if not frame_indices:
            return o3d.geometry.PointCloud()
        
        # Combine point clouds from all specified frames
        all_points = []
        all_colors = []
        
        for frame_idx in frame_indices:
            if frame_idx < len(self.point_clouds):
                points = self.point_clouds[frame_idx]
                color = self.colors[frame_idx]
                
                all_points.append(points)
                # Assign frame color to each point
                frame_colors = np.tile(color, (len(points), 1))
                all_colors.append(frame_colors)
        
        if not all_points:
            return o3d.geometry.PointCloud()
        
        # Combine all point clouds
        combined_points = np.vstack(all_points)
        combined_colors = np.vstack(all_colors)
        
        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(combined_points)
        pcd.colors = o3d.utility.Vector3dVector(combined_colors)
        
        return pcd
    
    def visualize_sequence(self, show_all_frames: bool = True, 
                          frame_step: int = 1,
                          auto_play: bool = False,
                          play_speed: float = 1.0,
                          render_to_image: bool = True):
        """
        Visualize sequence
        
        Args:
            show_all_frames: Whether to show all frames
            frame_step: Frame step size
            auto_play: Whether to auto play
            play_speed: Playback speed multiplier
            render_to_image: Whether to render as images
        """
        if render_to_image:
            # Image rendering mode
            self._render_sequence_to_images(show_all_frames, frame_step)
            return
        
        # Create visualization window
        vis = o3d.visualization.Visualizer()
        
        try:
            vis.create_window(window_name=f"Point Cloud Sequence - Sequence {self.sequence_idx}", 
                             width=1200, height=800)
        except Exception as e:
            print(f"Cannot create Open3D window: {e}")
            print("Possible reasons:")
            print("1. No display or X11 forwarding")
            print("2. Running in headless server environment")
            print("3. Open3D version incompatibility")
            print("\nTrying offline mode to save point clouds...")
            self._save_pointcloud_offline(show_all_frames, frame_step)
            return
        
        # Set rendering options
        render_option = vis.get_render_option()
        if render_option is not None:
            render_option.point_size = self.point_size
            render_option.background_color = np.array([0.1, 0.1, 0.1])
        else:
            print("Warning: Cannot get render options")
        
        if show_all_frames:
            # Show all frames
            frame_indices = list(range(0, self.total_frames, frame_step))
            pcd = self.create_combined_pointcloud(frame_indices)
            vis.add_geometry(pcd)
            print(f"Showing {len(frame_indices)} frames combined point cloud")
        else:
            # Single frame mode
            pcd = self.create_combined_pointcloud([0])
            vis.add_geometry(pcd)
            print("Single frame mode, use keyboard controls")
        
        # Set camera view
        ctr = vis.get_view_control()
        # ctr.set_front([0, 0, -1])
        # ctr.set_up([0, -1, 0])
        
        # Auto play mode
        if auto_play and not show_all_frames:
            self._auto_play_sequence(vis, play_speed)
        else:
            # Interactive mode
            self._interactive_mode(vis, show_all_frames)
    
    def _auto_play_sequence(self, vis, play_speed: float):
        """Auto play sequence"""
        print("Auto play mode - Press ESC to exit")
        
        frame_duration = 1.0 / play_speed  # Display time per frame
        
        for frame_idx in range(self.total_frames):
            # Update point cloud
            pcd = self.create_combined_pointcloud([frame_idx])
            vis.clear_geometry()
            vis.add_geometry(pcd)
            
            # Update window
            vis.poll_events()
            vis.update_renderer()
            
            # Wait
            time.sleep(frame_duration)
            
            # Check for exit
            if not vis.poll_events():
                break
    
    def _interactive_mode(self, vis, show_all_frames: bool):
        """Interactive mode"""
        if show_all_frames:
            print("Interactive mode - Use mouse to rotate/zoom to view combined point cloud")
            print("Press ESC to exit")
        else:
            print("Interactive mode - Keyboard controls:")
            print("  A/D: Previous/Next frame")
            print("  R: Reset view")
            print("  ESC: Exit")
        
        # Main loop
        while True:
            if not vis.poll_events():
                break
            vis.update_renderer()
            
            # Keyboard controls (only in single frame mode)
            if not show_all_frames:
                # Keyboard event handling can be added here
                pass
    
    def _save_pointcloud_offline(self, show_all_frames: bool, frame_step: int):
        """Offline mode: Save point clouds to files"""
        import os
        
        # Create output directory
        output_dir = f"pointcloud_sequence_{self.sequence_idx}"
        os.makedirs(output_dir, exist_ok=True)
        
        if show_all_frames:
            # Save combined point cloud
            frame_indices = list(range(0, self.total_frames, frame_step))
            pcd = self.create_combined_pointcloud(frame_indices)
            
            if len(pcd.points) > 0:
                output_file = f"{output_dir}/combined_sequence.ply"
                o3d.io.write_point_cloud(output_file, pcd)
                print(f"Combined point cloud saved to: {output_file}")
                print(f"Contains {len(pcd.points)} points")
            else:
                print("No point cloud data to save")
        else:
            # Save each frame
            for frame_idx in range(self.total_frames):
                if frame_idx < len(self.point_clouds):
                    pcd = self.create_combined_pointcloud([frame_idx])
                    if len(pcd.points) > 0:
                        output_file = f"{output_dir}/frame_{frame_idx:04d}.ply"
                        o3d.io.write_point_cloud(output_file, pcd)
                        print(f"Frame {frame_idx} saved to: {output_file}")
        
        print(f"\nAll files saved to directory: {output_dir}")
        print("You can open these files with MeshLab, CloudCompare or other point cloud viewers")
    
    def _render_sequence_to_images(self, show_all_frames: bool, frame_step: int):
        """Render sequence as JPEG images (using matplotlib, no Open3D GUI required).

        Also draws scene flow vectors (quiver) when available.
        """
        import os
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create output directory
        output_dir = f"pointcloud_images_sequence_{self.sequence_idx}"
        os.makedirs(output_dir, exist_ok=True)
        
        if show_all_frames:
            # Render combined point cloud
            frame_indices = list(range(0, self.total_frames, frame_step))
            print(f"Frame indices: {frame_indices}")
            pcd = self.create_combined_pointcloud(frame_indices)
            
            if len(pcd.points) > 0:
                output_file = f"{output_dir}/combined_sequence.jpg"
                # Aggregate flows from selected frames (downsample for readability)
                flow_points, flow_vecs = self._gather_flows(frame_indices, max_quivers=200000)
                print(f"Flow points: {flow_points.shape}, Flow vecs: {flow_vecs.shape}")

                # self._render_pointcloud_with_matplotlib(
                #     pcd,
                #     output_file,
                #     title=f"Sequence {self.sequence_idx} - Combined Point Cloud ({len(frame_indices)} frames)",
                #     flow_points=flow_points,
                #     flow_vecs=flow_vecs,
                # )

                # Save accompanying NPZ with points/colors and flow
                npz_path = output_file.replace('.jpg', '.npz')
                pc_points = np.asarray(pcd.points)
                pc_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                if pc_colors is None:
                    pc_colors = np.zeros((len(pc_points), 3), dtype=np.float32)
                if flow_points is None:
                    flow_points = np.zeros((0, 3), dtype=np.float32)
                if flow_vecs is None:
                    flow_vecs = np.zeros((0, 3), dtype=np.float32)
                np.savez(
                    npz_path,
                    points=pc_points.astype(np.float32),
                    colors=pc_colors.astype(np.float32),
                    flow_points=flow_points.astype(np.float32),
                    flow_vecs=flow_vecs.astype(np.float32),
                    frame_indices=np.array(frame_indices, dtype=np.int32),
                )
                print(f"Combined point cloud image saved to: {output_file}")
                print(f"Combined NPZ saved to: {npz_path}")
            else:
                print("No point cloud data to render")
        else:
            # Render each frame
            for frame_idx in range(self.total_frames):
                if frame_idx < len(self.point_clouds):
                    pcd = self.create_combined_pointcloud([frame_idx])
                    if len(pcd.points) > 0:
                        output_file = f"{output_dir}/frame_{frame_idx:04d}.jpg"
                        flow_points, flow_vecs = self._gather_flows([frame_idx], max_quivers=20000)
                        # print(f"Flow points: {flow_points.shape}, Flow vecs: {flow_vecs.shape}")
                        # self._render_pointcloud_with_matplotlib(
                        #     pcd,
                        #     output_file,
                        #     title=f"Sequence {self.sequence_idx} - Frame {frame_idx}",
                        #     flow_points=flow_points,
                        #     flow_vecs=flow_vecs,
                        # )

                        # Save per-frame NPZ
                        npz_path = output_file.replace('.jpg', '.npz')
                        pc_points = np.asarray(pcd.points)
                        pc_colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                        if pc_colors is None:
                            pc_colors = np.zeros((len(pc_points), 3), dtype=np.float32)
                        if flow_points is None:
                            flow_points = np.zeros((0, 3), dtype=np.float32)
                        if flow_vecs is None:
                            flow_vecs = np.zeros((0, 3), dtype=np.float32)
                        np.savez(
                            npz_path,
                            points=pc_points.astype(np.float32),
                            colors=pc_colors.astype(np.float32),
                            flow_points=flow_points.astype(np.float32),
                            flow_vecs=flow_vecs.astype(np.float32),
                            frame_indices=np.array([frame_idx], dtype=np.int32),
                        )
                        print(f"Frame NPZ saved to: {npz_path}")
                        print(f"Frame {frame_idx} image saved to: {output_file}")
        
        print(f"\nAll images saved to directory: {output_dir}")
    
    def _render_pointcloud_with_matplotlib(self, pcd, output_file, title="Point Cloud Visualization", 
                                          figsize=(120, 80), dpi=100,
                                          flow_points: Optional[np.ndarray] = None,
                                          flow_vecs: Optional[np.ndarray] = None):
        """Render point cloud as image using matplotlib (no GUI required).

        Optionally overlays scene flow vectors using a 3D quiver plot.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Set matplotlib to non-interactive backend
        plt.switch_backend('Agg')
        
        # Get point cloud data
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        # Create figure
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # # Plot point cloud
        # if colors is not None:
        #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
        #               c=colors, s=self.point_size, alpha=0.6)
        # else:
        #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
        #               s=self.point_size, alpha=0.6)
        
        # Set title and labels
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Set background color
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        
        # Set axis colors
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.title.set_color('white')
        
        # Set tick colors
        ax.tick_params(colors='white')
        
        # Set axes
        ax.grid(True, alpha=0.3)
        
        # Auto adjust view angle
        ax.view_init(elev=20, azim=45)

        # Draw scene flow vectors if provided
        if flow_points is not None and flow_vecs is not None and len(flow_points) > 0:
            # Normalize vector colors by magnitude
            magnitudes = np.linalg.norm(flow_vecs, axis=1)
            if magnitudes.max() > 0:
                norm_mag = magnitudes / magnitudes.max()
            else:
                norm_mag = magnitudes
            # Use a colormap for vectors
            import matplotlib.cm as cm
            cmap = cm.get_cmap('viridis')
            colors_quiver = cmap(norm_mag)
            # Quiver in 3D
            ax.quiver(
                flow_points[:, 0], flow_points[:, 1], flow_points[:, 2],
                flow_vecs[:, 0], flow_vecs[:, 1], flow_vecs[:, 2],
                length=1.0, normalize=False, linewidth=1, alpha=0.8
            )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        
        # Cleanup
        plt.close(fig)

    def _gather_flows(self, frame_indices: List[int], max_quivers: int = 5000) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Collect and subsample flow points/vectors for given frames.

        Returns (points, vecs) as float32 numpy arrays or (None, None) if unavailable.
        """
        pts_list = []
        vecs_list = []
        for idx in frame_indices:
            if idx < len(self.flow_points_list):
                p = self.flow_points_list[idx]
                v = self.flow_vecs_list[idx]
                if p is not None and v is not None and len(p) == len(v) and len(p) > 0:
                    pts_list.append(p)
                    vecs_list.append(v)

        if not pts_list:
            return None, None

        pts = np.concatenate(pts_list, axis=0).astype(np.float32)
        vecs = np.concatenate(vecs_list, axis=0).astype(np.float32)

        # # Subsample if too many arrows
        # if len(pts) > max_quivers:
        #     sel = np.random.choice(len(pts), size=max_quivers, replace=False)
        #     pts = pts[sel]
        #     vecs = vecs[sel]

        return pts, vecs


def main():
    parser = argparse.ArgumentParser(description="Point Cloud Sequence Visualization Tool")
    parser.add_argument("--raw_dir", type=str, required=True,
                       help="AV2 raw data path")
    parser.add_argument("--flow_dir", type=str, default=None,
                       help="Flow data path (optional)")
    parser.add_argument("--use_gt_flow", action="store_true",
                       help="Use ground truth flow data")
    parser.add_argument("--sequence_idx", type=int, default=0,
                       help="Sequence index to visualize")
    parser.add_argument("--max_frames", type=int, default=None,
                       help="Maximum number of frames")
    parser.add_argument("--show_all", action="store_true",
                       help="Show combined point cloud of all frames")
    parser.add_argument("--auto_play", action="store_true",
                       help="Auto play mode")
    parser.add_argument("--play_speed", type=float, default=2.0,
                       help="Playback speed")
    parser.add_argument("--point_size", type=float, default=1.0,
                       help="Point cloud point size")
    parser.add_argument("--offline", action="store_true",
                       help="Offline mode: save point clouds to files without displaying window")
    parser.add_argument("--render_images", action="store_true",
                       help="Render as JPEG images")
    
    args = parser.parse_args()
    
    # Create data loader
    loader = ArgoverseSceneFlowSequenceLoader(
        raw_data_path=Path(args.raw_dir),
        flow_data_path=Path(args.flow_dir) if args.flow_dir else None,
        use_gt_flow=True,
    )
    
    # Get available sequences
    seq_ids = loader.get_sequence_ids()
    print(f"Available sequences: {len(seq_ids)}")
    
    if args.sequence_idx >= len(seq_ids):
        print(f"Sequence index {args.sequence_idx} out of range (0-{len(seq_ids)-1})")
        return
    
    # Create visualizer
    visualizer = PointCloudSequenceVisualizer(
        loader, 
        sequence_idx=args.sequence_idx,
        max_frames=args.max_frames,
        point_size=args.point_size
    )
    
    # Load data
    visualizer.load_sequence_data()
    
    if not visualizer.point_clouds:
        print("No frame data loaded successfully")
        return
    
    # Start visualization
    if args.offline:
        print("Using offline mode...")
        visualizer._save_pointcloud_offline(
            show_all_frames=args.show_all,
            frame_step=1
        )
    elif args.render_images:
        print("Using image rendering mode...")
        visualizer.visualize_sequence(
            show_all_frames=args.show_all,
            frame_step=1,
            render_to_image=True
        )
    else:
        visualizer.visualize_sequence(
            show_all_frames=args.show_all,
            auto_play=args.auto_play,
            play_speed=args.play_speed,
            render_to_image=False
        )


if __name__ == "__main__":
    main()
