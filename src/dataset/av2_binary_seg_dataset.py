import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Iterable, Optional

from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo


class AV2BinarySegDataset(Dataset):
    """
    Wraps AV2SceneFlowZoo to provide foreground/background labels for supervision.

    Foreground points are defined either by an explicit class-id allowlist or
    by thresholding class ids (> 0 by default). The dataset always returns a
    fixed number of points as configured via `point_size`, matching the
    internal downsampling in AV2SceneFlowZoo.
    """

    def __init__(
        self,
        *,
        root_dir: str,
        flow_data_path: str,
        expected_camera_shape: tuple[int, int, int],
        subsequence_length: int,
        sliding_window_step_size: int,
        with_rgb: bool,
        range_crop_type: str,
        point_size: int,
        with_ground: bool = False,
        use_gt_flow: bool = False,
        eval_type: str = "bucketed_epe",
        load_flow: bool = False,
        load_boxes: bool = False,
        min_instance_size: int = 50,
        cache_root: str = "/tmp/",
        log_subset: Optional[Iterable[str]] = None,
        min_flow_threshold: float = 0.05,
        fg_class_ids: Optional[Iterable[int]] = None,
    ):
        super().__init__()
        self.point_size = point_size
        self.fg_class_ids = set(fg_class_ids) if fg_class_ids is not None else None

        self.dataset = AV2SceneFlowZoo(
            root_dir=Path(root_dir),
            expected_camera_shape=expected_camera_shape,
            eval_args=dict(),
            with_rgb=with_rgb,
            flow_data_path=Path(flow_data_path),
            range_crop_type=range_crop_type,
            point_size=point_size,
            subsequence_length=subsequence_length,
            sliding_window_step_size=sliding_window_step_size,
            with_ground=with_ground,
            use_gt_flow=use_gt_flow,
            eval_type=eval_type,
            load_flow=load_flow,
            load_boxes=load_boxes,
            min_instance_size=min_instance_size,
            cache_root=Path(cache_root),
            log_subset=list(log_subset) if log_subset is not None else None,
            min_flow_threshold=min_flow_threshold,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        points = sample["point_cloud_first"].float()

        class_ids = sample["class_ids"]
        if not isinstance(class_ids, torch.Tensor):
            class_ids = torch.from_numpy(class_ids)
        binary_labels = self._class_ids_to_binary(class_ids)

        return {
            "points": points,
            "labels": binary_labels,
            "sequence_id": sample.get("sequence_id"),
        }

    def _class_ids_to_binary(self, class_ids: torch.Tensor) -> torch.Tensor:
        if self.fg_class_ids is not None:
            fg_mask = torch.zeros_like(class_ids, dtype=torch.bool)
            for class_id in self.fg_class_ids:
                fg_mask |= class_ids == class_id
        else:
            fg_mask = class_ids > 0
        return fg_mask.long()

