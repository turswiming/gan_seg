from dataclasses import dataclass
from pathlib import Path
import numpy as np
from functools import cached_property

from bucketed_scene_flow_eval.datastructures import (
    SE3,
    BoundingBox,
    TimeSyncedAVLidarData,
    TimeSyncedSceneFlowBoxFrame,
    TimeSyncedSceneFlowFrame,
    InstanceId,
)
from bucketed_scene_flow_eval.utils import load_feather

from .argoverse_scene_flow import ArgoverseNoFlowSequence, ArgoverseNoFlowSequenceLoader

BLEED = 0.2
import pickle
import pandas as pd
class LazyAnnotationDict:
    """Lazy-loading dictionary for bounding box annotations."""
    
    def __init__(self, annotations_file: Path):
        self.annotations_file = annotations_file
        self._cache = {}  # 缓存已加载的 timestamp
        self._df = None   # DataFrame 懒加载
    
    @cached_property
    def _annotation_df(self) -> pd.DataFrame:
        """Lazy load the dataframe once."""
        return load_feather(self.annotations_file, verbose=False)
    
    def get(self, timestamp: int, default=None):
        """Get boxes for a specific timestamp, loading only when needed."""
        # 如果已缓存，直接返回
        if timestamp in self._cache:
            return self._cache[timestamp]
        
        # 从 DataFrame 中过滤出该 timestamp 的数据
        rows = self._annotation_df[self._annotation_df['timestamp_ns'] == timestamp]
        
        if len(rows) == 0:
            self._cache[timestamp] = default if default is not None else []
            return self._cache[timestamp]
        
        # 只解析这个 timestamp 的 boxes
        boxes = []
        for _, row in rows.iterrows():
            pose = SE3.from_rot_w_x_y_z_translation_x_y_z(
                row["qw"], row["qx"], row["qy"], row["qz"],
                row["tx_m"], row["ty_m"], row["tz_m"],
            )
            boxes.append(
                BoundingBox(
                    pose=pose,
                    length=row["length_m"],
                    width=row["width_m"],
                    height=row["height_m"],
                    track_uuid=row["track_uuid"],
                    category=row["category"],
                )
            )
        
        # 缓存结果
        self._cache[timestamp] = boxes
        return boxes
    
    def __getitem__(self, timestamp: int):
        """Support dict-like access."""
        result = self.get(timestamp)
        if result is None:
            raise KeyError(timestamp)
        return result

class ArgoverseBoxAnnotationSequence(ArgoverseNoFlowSequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.timestamp_to_boxes = self._prep_bbox_annotations()
    @cached_property
    def timestamp_to_boxes(self) -> LazyAnnotationDict:
        """Lazily load annotation dictionary."""
        annotations_file = self.dataset_dir / "annotations.feather"
        assert annotations_file.exists(), f"Annotations file {annotations_file} does not exist"
        return LazyAnnotationDict(annotations_file)

    def _prep_bbox_annotations(self) -> dict[int, list[BoundingBox]]:
        annotations_file = self.dataset_dir / "annotations.feather"
        assert annotations_file.exists(), f"Annotations file {annotations_file} does not exist"
        annotation_df = load_feather(annotations_file)
        # Index(['timestamp_ns', 'track_uuid', 'category', 'length_m', 'width_m',
        #         'height_m', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m',
        #         'num_interior_pts'],
        #         dtype='object')

        # Convert to dictionary keyed by timestamp_ns int
        timestamp_to_annotations: dict[int, list[BoundingBox]] = {}
        for _, row in annotation_df.iterrows():
            timestamp_ns = row["timestamp_ns"]
            if timestamp_ns not in timestamp_to_annotations:
                timestamp_to_annotations[timestamp_ns] = []
            pose = SE3.from_rot_w_x_y_z_translation_x_y_z(
                row["qw"],
                row["qx"],
                row["qy"],
                row["qz"],
                row["tx_m"],
                row["ty_m"],
                row["tz_m"],
            )
            timestamp_to_annotations[timestamp_ns].append(
                BoundingBox(
                    pose=pose,
                    length=row["length_m"],
                    width=row["width_m"],
                    height=row["height_m"],
                    track_uuid=row["track_uuid"],
                    category=row["category"],
                )
            )
        return timestamp_to_annotations

    def load(
        self, idx: int, relative_to_idx: int, with_flow: bool = False
    ) -> tuple[TimeSyncedSceneFlowBoxFrame, TimeSyncedAVLidarData]:
        scene_flow_frame, lidar_data = super().load(idx, relative_to_idx, with_flow)
        timestamp = self.timestamp_list[idx]
        boxes = self.timestamp_to_boxes.get(timestamp, [])
        cache_path = self.dataset_dir /"cache" / f"{timestamp}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                instance_ids = pickle.load(f)
            return TimeSyncedSceneFlowBoxFrame(
                **vars(scene_flow_frame), boxes=boxes, instance_ids=instance_ids
            ), lidar_data
        else:
        # Compute per-point instance ids via point-in-box using boxes' poses and sizes
            instance_ids = self._compute_instance_ids(scene_flow_frame, boxes)
            if not cache_path.parent.exists():
                cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(instance_ids, f)
            return TimeSyncedSceneFlowBoxFrame(
                **vars(scene_flow_frame), boxes=boxes, instance_ids=instance_ids
            ), lidar_data

    def _compute_instance_ids(
        self, scene_flow_frame: TimeSyncedSceneFlowFrame, boxes: list[BoundingBox]
    ) -> np.ndarray:
        num_points = len(scene_flow_frame.pc.full_pc)
        instance_ids = np.full((num_points,), fill_value=-1, dtype=InstanceId)
        if len(boxes) == 0 or num_points == 0:
            return instance_ids

        # Use global coordinates for stable assignments
        points_global = scene_flow_frame.pc.full_ego_pc.to_array()

        # Map track_uuid to stable small integer ids
        track_to_id: dict[str, int] = {}
        next_id = 0
        for box in boxes:
            if box.track_uuid not in track_to_id:
                track_to_id[box.track_uuid] = next_id
                next_id += 1

        # For each box, compute mask of points inside oriented box
        for box in boxes:
            # Transform points to box local frame: box_T_global^{-1}
            global_T_box = box.pose.inverse()
            pts_local = global_T_box.transform_points(points_global)
            half_l = (box.length / 2.0) + BLEED
            half_w = (box.width / 2.0) + BLEED
            half_h = (box.height / 2.0) + BLEED
            in_x = np.logical_and(pts_local[:, 0] >= -half_l, pts_local[:, 0] <= half_l)
            in_y = np.logical_and(pts_local[:, 1] >= -half_w, pts_local[:, 1] <= half_w)
            in_z = np.logical_and(pts_local[:, 2] >= -half_h, pts_local[:, 2] <= half_h)
            inside = in_x & in_y & in_z
            if inside.any():
                instance_ids[inside] = track_to_id[box.track_uuid]

        return instance_ids


class ArgoverseBoxAnnotationSequenceLoader(ArgoverseNoFlowSequenceLoader):

    def _load_sequence_uncached(self, sequence_id: str) -> ArgoverseBoxAnnotationSequence:
        assert (
            sequence_id in self.sequence_id_to_raw_data
        ), f"sequence_id {sequence_id} does not exist"
        return ArgoverseBoxAnnotationSequence(
            sequence_id,
            self.sequence_id_to_raw_data[sequence_id],
            self.sequence_id_to_raw_data[sequence_id],
            with_classes=False,
            **self.load_sequence_kwargs,
        )

    def cache_folder_name(self) -> str:
        return f"av2_box_data_use_gt_flow_{self.use_gt_flow}_raw_data_path_{self.raw_data_path}_No_flow_data_path"
