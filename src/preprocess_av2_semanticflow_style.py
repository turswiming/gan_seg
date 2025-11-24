#!/usr/bin/env python3
"""
Argoverse2 Dataset Preprocessing - SemanticFlow Style

This script preprocesses Argoverse2 dataset following the SemanticFlow approach:
1. Convert coordinate frames: ego → city → sensor
2. Use HD Map for ground detection
3. Compute scene flow dynamically from annotations
4. Expand bounding boxes (+0.2m)
5. Handle dynamic objects
6. Support self-supervised learning (DUFOmap + HDBSCAN)
"""

import argparse
import concurrent.futures
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any
from zipfile import ZipFile

import fire
import h5py
import numpy as np
import pandas as pd
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.constants import AnnotationCategories
from av2.geometry.se3 import SE3
from av2.map.map_api import ArgoverseStaticMap
from av2.structures.cuboid import Cuboid, CuboidList
from av2.structures.sweep import Sweep
from av2.utils.io import read_feather
from tqdm import tqdm
from typing import Final

BOUNDING_BOX_EXPANSION: Final = 0.2
CATEGORY_TO_INDEX: Final = {
    **{"NONE": 0},
    **{k.value: i + 1 for i, k in enumerate(AnnotationCategories)},
}


def convert_pose_dataframe_to_SE3(pose_df: pd.DataFrame):
    """Convert pose dataframe to SE3."""
    qw, qx, qy, qz = pose_df[["qw", "qx", "qy", "qz"]].values.squeeze()
    tx, ty, tz = pose_df[["tx_m", "ty_m", "tz_m"]].values.squeeze()

    rotation = np.array([[1 - 2 * (qy ** 2 + qz ** 2),
                          2 * (qx * qy - qw * qz),
                          2 * (qx * qz + qw * qy)],
                         [2 * (qx * qy + qw * qz),
                          1 - 2 * (qx ** 2 + qz ** 2),
                          2 * (qy * qz - qw * qx)],
                         [2 * (qx * qz - qw * qy),
                          2 * (qy * qz + qw * qx),
                          1 - 2 * (qx ** 2 + qy ** 2)]])

    translation = np.array([tx, ty, tz])
    return SE3(rotation, translation)


def read_ego_SE3_sensor(data_dir: Path):
    """Read ego to sensor SE3 transformations."""
    calib_path = data_dir / "calibration" / "egovehicle_SE3_sensor.feather"
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    calib_df = read_feather(calib_path)
    ego2sensor = {}
    for _, row in calib_df.iterrows():
        sensor_name = row['sensor_name']
        qw, qx, qy, qz = row["qw"], row["qx"], row["qy"], row["qz"]
        tx, ty, tz = row["tx_m"], row["ty_m"], row["tz_m"]

        rotation = np.array([[1 - 2 * (qy ** 2 + qz ** 2),
                              2 * (qx * qy - qw * qz),
                              2 * (qx * qz + qw * qy)],
                             [2 * (qx * qy + qw * qz),
                              1 - 2 * (qx ** 2 + qz ** 2),
                              2 * (qy * qz - qw * qx)],
                             [2 * (qx * qz - qw * qy),
                              2 * (qy * qz + qw * qx),
                              1 - 2 * (qx ** 2 + qy ** 2)]])

        translation = np.array([tx, ty, tz])
        ego2sensor[sensor_name] = SE3(rotation, translation)

    return ego2sensor


def create_eval_mask(data_mode: str, output_dir: Path, mask_dir: str):
    """
    Create evaluation mask for val and test splits.
    Need download the official mask file run: `s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" .`
    Check more in SemanticFlow assets/README.md
    """
    mask_file_path = Path(mask_dir) / f"{data_mode}-masks.zip"
    if not mask_file_path.exists():
        print(f'{mask_file_path} not found, please download the mask file for official evaluation.')
        print(f'Run: s5cmd --no-sign-request cp "s3://argoverse/tasks/3d_scene_flow/zips/*" {mask_dir}/')
        return
    
    # Extract the mask file
    extracted_dir = Path(mask_dir) / f"{data_mode}-masks"
    if not extracted_dir.exists():
        print(f'Extracting {mask_file_path}...')
        with ZipFile(mask_file_path, 'r') as zipObj:
            zipObj.extractall(extracted_dir)
    
    data_index = []
    # List scene ids
    if not extracted_dir.exists():
        print(f'Extracted directory {extracted_dir} not found after extraction.')
        return
        
    scene_ids = [d for d in os.listdir(extracted_dir) if os.path.isdir(extracted_dir / d)]
    
    for scene_id in tqdm(scene_ids, desc=f'Create {data_mode} eval mask', ncols=100):
        scene_mask_dir = extracted_dir / scene_id
        if not scene_mask_dir.exists():
            continue
            
        timestamps = sorted([
            int(file.replace('.feather', ''))
            for file in os.listdir(scene_mask_dir)
            if file.endswith('.feather')
        ])
        
        if not (output_dir / f'{scene_id}.h5').exists():
            continue
            
        with h5py.File(output_dir / f'{scene_id}.h5', 'r+') as f:
            for ts in timestamps:
                key = str(ts)
                if key not in f.keys():
                    print(f'{scene_id}/{key} not found in h5 file')
                    continue
                    
                group = f[key]
                mask_file = scene_mask_dir / f"{key}.feather"
                if not mask_file.exists():
                    print(f'Mask file {mask_file} not found')
                    continue
                    
                mask = pd.read_feather(mask_file).to_numpy().astype(bool)
                
                # Handle different mask shapes
                if mask.ndim > 1:
                    mask = mask.squeeze()
                if mask.ndim == 0:
                    mask = np.array([mask])
                    
                # Check if mask size matches point cloud size
                if 'lidar' in group:
                    pc_size = group['lidar'].shape[0]
                    if mask.shape[0] != pc_size:
                        print(f'Mask size mismatch for {scene_id}/{key}: mask={mask.shape[0]}, pc={pc_size}')
                        continue
                
                if 'eval_mask' in group:
                    del group['eval_mask']
                group.create_dataset('eval_mask', data=mask)
                data_index.append([scene_id, key])

    # Create eval index file
    eval_index_path = output_dir / 'index_eval.pkl'
    with open(eval_index_path, 'wb') as f:
        pickle.dump(data_index, f)
    print(f"Created eval index with {len(data_index)} samples at {eval_index_path}")


def read_pose_pc_ground(data_dir: Path, log_id: str, timestamp: int,
                        avm: ArgoverseStaticMap, split: str):
    """Read pose, point cloud, and ground mask using HD Map."""
    # Load poses
    pose_path = data_dir / log_id / "city_SE3_egovehicle.feather"
    ego2sensor_pose = read_ego_SE3_sensor(data_dir / log_id)['up_lidar']

    if not pose_path.exists():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")

    log_poses_df = read_feather(pose_path)
    filtered_log_poses_df = log_poses_df[log_poses_df["timestamp_ns"].isin([timestamp])]

    if len(filtered_log_poses_df) == 0:
        raise ValueError(f"Timestamp {timestamp} not found in pose file")

    pose = convert_pose_dataframe_to_SE3(filtered_log_poses_df)

    # Load point cloud
    sweep_path = data_dir / log_id / "sensors" / "lidar" / f"{timestamp}.feather"
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep file not found: {sweep_path}")

    pc = Sweep.from_feather(sweep_path).xyz

    # Get ground points using HD Map (in city coordinates)
    global_pc = pose.transform_point_cloud(pc)
    is_ground = avm.get_ground_points_boolean(global_pc)

    # Convert to sensor coordinates for output
    pc_sensor = ego2sensor_pose.inverse().transform_point_cloud(pc)

    return pc_sensor, pose, is_ground


def compute_sceneflow(data_dir: Path, log_id: str, timestamps: Tuple[int, int],
                     split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SE3]:
    """Compute scene flow from annotations."""
    # Load poses and point clouds
    log_map_dirpath = data_dir / log_id / "map"
    avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

    pc_list = []
    pose_list = []

    for ts in timestamps:
        pc, pose, _ = read_pose_pc_ground(data_dir, log_id, ts, avm, split)
        pc_list.append(pc)
        pose_list.append(pose)

    # Load cuboids (only for sensor dataset)
    annotations_path = data_dir / log_id / "annotations.feather"
    if annotations_path.exists() and split in ['train', 'val']:
        # Load annotations from disk
        cuboid_list = CuboidList.from_feather(annotations_path)
        raw_data = read_feather(annotations_path)
        ids = raw_data.track_uuid.to_numpy()
        
        # Build timestamp to cuboid mapping
        timestamp_cuboid_index = defaultdict(dict)
        for id, cuboid in zip(ids, cuboid_list.cuboids):
            timestamp_cuboid_index[cuboid.timestamp_ns][id] = cuboid
        
        cuboids = [timestamp_cuboid_index.get(ts, {}) for ts in timestamps]

        # Compute ego motion flow (for background points)
        ego1_SE3_ego0 = pose_list[1].inverse().compose(pose_list[0])
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        
        flow = ego1_SE3_ego0.transform_point_cloud(pc_list[0]) - pc_list[0]
        flow = flow.astype(np.float32)

        # Initialize validity mask and categories
        valid_mask = np.ones(len(pc_list[0]), dtype=np.bool_)
        class_indices = np.zeros(len(pc_list[0]), dtype=np.uint8)

        # Process dynamic objects
        for track_id, cuboid_0 in cuboids[0].items():
            # Expand bounding box
            cuboid_0.length_m += BOUNDING_BOX_EXPANSION
            cuboid_0.width_m += BOUNDING_BOX_EXPANSION

            # Get points inside cuboid
            obj_pts, obj_mask = cuboid_0.compute_interior_points(pc_list[0])
            
            # Assign category using SemanticFlow's category mapping
            class_indices[obj_mask] = CATEGORY_TO_INDEX[str(cuboid_0.category)]

            # Check if object exists in next frame
            if track_id in cuboids[1]:
                cuboid_1 = cuboids[1][track_id]

                # Compute object motion
                c1_SE3_c0 = cuboid_1.dst_SE3_object.compose(cuboid_0.dst_SE3_object.inverse())
                obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                flow[obj_mask] = obj_flow.astype(np.float32)
            else:
                # Object disappeared, mark flow as invalid
                valid_mask[obj_mask] = False
    else:
        # No annotations, use ego motion only
        ego1_SE3_ego0 = pose_list[1].inverse().compose(pose_list[0])
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        flow = ego1_SE3_ego0.transform_point_cloud(pc_list[0]) - pc_list[0]
        flow = flow.astype(np.float32)
        valid_mask = np.ones(len(pc_list[0]), dtype=np.bool_)
        class_indices = np.zeros(len(pc_list[0]), dtype=np.uint8)

    return flow, valid_mask, class_indices, ego1_SE3_ego0


def create_group_data(group, pc, ground_mask, pose, flow=None, flow_valid=None,
                     flow_category=None, ego_motion=None, eval_mask=None,
                     dufo_label=None, label=None):
    """Create HDF5 group with all data."""
    group.create_dataset('lidar', data=pc.astype(np.float32))
    group.create_dataset('ground_mask', data=ground_mask.astype(bool))
    group.create_dataset('pose', data=pose.transform_matrix.astype(np.float32))

    if flow is not None:
        group.create_dataset('flow', data=flow.astype(np.float32))
        group.create_dataset('flow_is_valid', data=flow_valid.astype(bool))
        group.create_dataset('flow_category_indices', data=flow_category.astype(np.uint8))
        group.create_dataset('ego_motion', data=ego_motion.transform_matrix.astype(np.float32))

    if eval_mask is not None:
        group.create_dataset('eval_mask', data=eval_mask.astype(bool))

    if dufo_label is not None:
        group.create_dataset('dufo_label', data=dufo_label.astype(np.uint8))

    if label is not None:
        group.create_dataset('label', data=label.astype(np.int16))


def process_log(data_dir: Path, output_dir: Path, log_id: str, split: str,
               with_flow: bool = True, with_ssl: bool = False):
    """Process a single log/sequence."""
    try:
        log_map_dirpath = data_dir / log_id / "map"
        if not log_map_dirpath.exists():
            warnings.warn(f"No map directory for {log_id}, skipping...")
            return None

        avm = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

        # Get all LiDAR timestamps
        lidar_dir = data_dir / log_id / "sensors" / "lidar"
        if not lidar_dir.exists():
            warnings.warn(f"No LiDAR directory for {log_id}, skipping...")
            return None

        timestamps = sorted([
            int(f.stem) for f in lidar_dir.glob("*.feather")
        ])

        if len(timestamps) < 2:
            warnings.warn(f"Not enough frames in {log_id}, skipping...")
            return None

        # Create output file
        output_file = output_dir / f"{log_id}.h5"

        processed_pairs = []

        with h5py.File(output_file, 'w') as f:
            # Process consecutive pairs
            for i in range(len(timestamps) - 1):
                ts0, ts1 = timestamps[i], timestamps[i + 1]

                try:
                    # Read data for both frames
                    pc0, pose0, ground0 = read_pose_pc_ground(data_dir, log_id, ts0, avm, split)
                    pc1, pose1, ground1 = read_pose_pc_ground(data_dir, log_id, ts1, avm, split)

                    # Compute flow if annotations available
                    if with_flow and split in ['train', 'val']:
                        flow, valid_mask, class_indices, ego_motion = compute_sceneflow(
                            data_dir, log_id, (ts0, ts1), split
                        )
                    else:
                        flow = None
                        valid_mask = None
                        class_indices = None
                        ego_motion = None

                    # Create group for frame 0
                    group_name = str(ts0)
                    group = f.create_group(group_name)

                    # Store data
                    create_group_data(
                        group=group,
                        pc=pc0,
                        ground_mask=ground0,
                        pose=pose0,
                        flow=flow,
                        flow_valid=valid_mask,
                        flow_category=class_indices,
                        ego_motion=ego_motion,
                    )

                    processed_pairs.append([log_id, str(ts0)])

                except Exception as e:
                    warnings.warn(f"Error processing frame {ts0} in {log_id}: {e}")
                    continue

        if len(processed_pairs) == 0:
            output_file.unlink()
            return None

        return processed_pairs

    except Exception as e:
        warnings.warn(f"Error processing log {log_id}: {e}")
        return None


def create_reading_index(data_dir: Path):
    """Create index file for all processed data."""
    data_index = []

    for file_name in os.listdir(data_dir):
        if not file_name.endswith(".h5"):
            continue

        scene_id = file_name.split(".")[0]

        try:
            with h5py.File(data_dir / file_name, 'r') as f:
                timestamps = sorted(f.keys(), key=lambda x: int(x))
                for timestamp in timestamps:
                    data_index.append([scene_id, timestamp])
        except Exception as e:
            warnings.warn(f"Error reading {file_name}: {e}")
            continue

    with open(data_dir / 'index_total.pkl', 'wb') as f:
        pickle.dump(data_index, f)

    print(f"Created index with {len(data_index)} samples")


def main(argo_dir: str,
         output_dir: str,
         split: str = 'train',
         av2_type: str = 'sensor',
         nproc: int = 1,
         with_flow: bool = True,
         with_ssl: bool = False,
         mask_dir: Optional[str] = None,
         only_index: bool = False):
    """
    Main preprocessing function.
    
    Args:
        argo_dir: Root directory of Argoverse2 dataset
        output_dir: Output directory for preprocessed data
        split: Dataset split ('train', 'val', or 'test')
        av2_type: Dataset type ('sensor' or 'lidar')
        nproc: Number of processes for parallel processing
        with_flow: Whether to compute scene flow (requires annotations)
        with_ssl: Whether to include self-supervised learning labels
        mask_dir: Directory containing official evaluation masks (required for val/test)
        only_index: If True, only create index files without processing data
    """

    data_dir = Path(argo_dir) / av2_type / split
    output_dir = Path(output_dir) / av2_type / split

    if only_index:
        create_reading_index(output_dir)
        if split in ['val', 'test'] and mask_dir:
            create_eval_mask(split, output_dir, mask_dir)
        return

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing Argoverse2 {av2_type} dataset ({split} split)")
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of processes: {nproc}")
    print(f"With flow: {with_flow}")
    print(f"With SSL: {with_ssl}")

    # Get all log IDs
    log_ids = sorted([
        d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / "sensors" / "lidar").exists()
    ])

    print(f"Found {len(log_ids)} logs to process")

    # Process logs
    if nproc > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = {
                executor.submit(process_log, data_dir, output_dir, log_id, split, with_flow, with_ssl): log_id
                for log_id in log_ids
            }

            all_pairs = []
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(log_ids), desc="Processing logs"):
                log_id = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_pairs.extend(result)
                except Exception as e:
                    warnings.warn(f"Error processing {log_id}: {e}")
    else:
        all_pairs = []
        for log_id in tqdm(log_ids, desc="Processing logs"):
            result = process_log(data_dir, output_dir, log_id, split, with_flow, with_ssl)
            if result is not None:
                all_pairs.extend(result)

    print(f"Successfully processed {len(all_pairs)} frame pairs")

    # Create index file
    create_reading_index(output_dir)

    # Create eval mask for val and test splits
    if split in ['val', 'test'] and mask_dir:
        print(f"\nCreating evaluation masks for {split} split...")
        create_eval_mask(split, output_dir, mask_dir)

    print("Preprocessing completed!")


if __name__ == "__main__":
    fire.Fire(main)
