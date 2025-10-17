from dataclasses import dataclass
from pathlib import Path
import numpy as np

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


class ArgoverseBoxAnnotationSequence(ArgoverseNoFlowSequence):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timestamp_to_boxes = self._prep_bbox_annotations()

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

        # Compute per-point instance ids via point-in-box using boxes' poses and sizes
        instance_ids = self._compute_instance_ids(scene_flow_frame, boxes)

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
        points_global = scene_flow_frame.pc.full_global_pc.to_array()

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
            half_l = box.length / 2.0
            half_w = box.width / 2.0
            half_h = box.height / 2.0
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
