from bucketed_scene_flow_eval.datasets.argoverse2 import Argoverse2CausalSceneFlow
from pathlib import Path
import torch
from utils.bucketed_epe_utils import extract_classid_from_argoverse2_data
from bucketed_scene_flow_eval.datasets.argoverse2.argoverse_raw_data import ArgoverseRawSequence
import numpy as np
from typing import Optional
from utils.fps_utils import fps_downsample_with_attributes


class AV2SceneFlowZoo(Argoverse2CausalSceneFlow):
    def __init__(
        self,
        root_dir: Path,
        expected_camera_shape: tuple,
        eval_args: dict,
        with_rgb: bool,
        flow_data_path: Path,
        range_crop_type: str,
        point_size: int = 8192,
        subsequence_length: int = 2,
        sliding_window_step_size: int = 1,
        with_ground: bool = False,
        use_gt_flow: bool = True,
        eval_type: str = "bucketed_epe",
        load_flow: bool = True,
        load_boxes: bool = False,
        min_instance_size: int = 50,
        cache_root: Path = Path("/tmp/"),
        log_subset: Optional[list[str]] = None,
    ):
        super().__init__(
            root_dir=root_dir,
            subsequence_length=subsequence_length,
            sliding_window_step_size=sliding_window_step_size,
            with_ground=with_ground,
            use_gt_flow=use_gt_flow,
            eval_type=eval_type,
            expected_camera_shape=expected_camera_shape,
            eval_args=eval_args,
            with_rgb=with_rgb,
            flow_data_path=flow_data_path,
            range_crop_type=range_crop_type,
            load_flow=load_flow,
            load_boxes=load_boxes,
            log_subset=log_subset,
            cache_root=cache_root,
        )
        self.root_dir = root_dir
        self.point_size = point_size
        self.min_instance_size = min_instance_size
        self.downsample_generator = torch.Generator().manual_seed(0)

    def transform_pc(self, pc: "torch.Tensor", transform: "torch.Tensor") -> "torch.Tensor":
        """
        Transform an Nx3 point cloud by a 4x4 transformation matrix.
        """

        homogenious_pc = torch.cat((pc, torch.ones((pc.shape[0], 1), device=pc.device)), dim=1)
        homogenious_pc = homogenious_pc @ transform.T
        return homogenious_pc[:, :3]

    def ego_to_global_flow(
        self, ego_pc: torch.Tensor, ego_flow: torch.Tensor, ego_to_global: torch.Tensor
    ) -> torch.Tensor:
        # Expect an Nx3 point cloud, an Nx3 ego flow, and a 4x4 transformation matrix
        # Return an Nx3 global flow

        assert ego_pc.shape[1] == 3, f"Expected Nx3 point cloud, got {ego_pc.shape}"
        assert ego_flow.shape[1] == 3, f"Expected Nx3 ego flow, got {ego_flow.shape}"
        assert ego_to_global.shape == (
            4,
            4,
        ), f"Expected 4x4 transformation matrix, got {ego_to_global.shape}"

        assert ego_pc.shape == ego_flow.shape, f"Expected same shape, got {ego_pc.shape} != {ego_flow.shape}"

        flowed_ego_pc = ego_pc + ego_flow

        # Transform both the point cloud and the flowed point cloud into the global frame
        global_pc = self.transform_pc(ego_pc, ego_to_global)
        global_flowed_pc = self.transform_pc(flowed_ego_pc, ego_to_global)
        global_flow = global_flowed_pc - global_pc
        return global_flow

    def __getitem__(self, index):
        sample = super().__getitem__(index // 4)

        point_cloud_first = sample[0].pc.full_global_pc.points
        point_cloud_next = sample[1].pc.full_global_pc.points

        valid_mask_first = sample[0].pc.mask
        valid_mask_next = sample[1].pc.mask

        # centroid_first = np.mean(sample[0].pc.global_pc.points, axis=0)
        # centroid_next = np.mean(sample[1].pc.global_pc.points, axis=0)
        # centroid = (centroid_first + centroid_next) / 2
        # if index // 2 == 0:
        #     half_mask_first_1 = point_cloud_first[:, 0] - centroid[0] < 0
        #     half_mask_next_1 = point_cloud_next[:, 0] - centroid[0] < 0.5
        # else:
        #     half_mask_first_1 = centroid[0] - point_cloud_first[:, 0] < 0
        #     half_mask_next_1 = centroid[0] - point_cloud_next[:, 0] < 0.5

        # if index / 2 // 2 == 0:
        #     half_mask_first_2 = centroid[1] - point_cloud_first[:, 1] < 0
        #     half_mask_next_2 = centroid[1] - point_cloud_next[:, 1] < 0.5
        # else:
        #     half_mask_first_2 = point_cloud_first[:, 1] - centroid[1] < 0
        #     half_mask_next_2 = point_cloud_next[:, 1] - centroid[1] < 0.5

        # valid_mask_first = valid_mask_first & half_mask_first_1 & half_mask_first_2
        # valid_mask_next = valid_mask_next & half_mask_next_1 & half_mask_next_2
        # valid_mask_first = np.ones_like(valid_mask_first)
        # valid_mask_next = np.ones_like(valid_mask_next)
        point_cloud_first = point_cloud_first[valid_mask_first, :]
        point_cloud_next = point_cloud_next[valid_mask_next, :]

        point_cloud_first = torch.from_numpy(point_cloud_first).float()
        point_cloud_next = torch.from_numpy(point_cloud_next).float()

        # process flow
        flow = sample[0].flow.full_flow[valid_mask_first, :]
        transform = sample[0].pc.pose.ego_to_global.to_array()
        transform = torch.from_numpy(transform).float()
        flow = torch.from_numpy(flow).float()
        flow = self.ego_to_global_flow(point_cloud_first, flow, transform)

        class_ids = extract_classid_from_argoverse2_data(sample[0])
        class_ids = class_ids[valid_mask_first,]
        if hasattr(sample[0], "instance_ids"):
            mask = sample[0].instance_ids[valid_mask_first]
            mask = torch.from_numpy(mask)
        else:
            mask = None
        (point_cloud_first, point_cloud_next, mask, flow, class_ids) = self.downsample_point_cloud(
            point_cloud_first, point_cloud_next, mask, flow, class_ids, self.point_size
        )
        if mask is not None:
            mask = mask + 1  # add 1 to the mask to avoid the background class
            # print mask larger than 10
            mask_onehot = torch.nn.functional.one_hot(mask.long())
            mask_onehot = mask_onehot[:, mask_onehot.sum(dim=0) > self.min_instance_size]
            mask_onehot = mask_onehot.float()
            mask_onehot[:, 0] += 0.01
            # add a small value to the background class to avoid the background class is not used
            mask = torch.argmax(mask_onehot, dim=1)
        new_sample = {
            "point_cloud_first": point_cloud_first,
            "point_cloud_next": point_cloud_next,
            "flow": flow,
            "class_ids": class_ids,
            "sequence_id": sample[0].log_id,
        }
        if mask is not None:
            new_sample["mask"] = mask

        return new_sample

    def downsample_point_cloud(
        self,
        point_cloud_first: torch.Tensor,
        point_cloud_next: torch.Tensor,
        mask: torch.Tensor | None,
        flow: torch.Tensor,
        class_ids: torch.Tensor,
        size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # using farthest point sampling to downsample the point cloud
        self.downsample_generator.manual_seed(point_cloud_first.shape[0])
        random_indices_first = torch.randint(
            0, point_cloud_first.shape[0], (size,), generator=self.downsample_generator
        )
        point_cloud_first = point_cloud_first[random_indices_first, :]
        if mask is not None:
            mask = mask[random_indices_first,]
        flow = flow[random_indices_first, :]
        class_ids = class_ids[random_indices_first,]
        self.downsample_generator.manual_seed(point_cloud_next.shape[0])
        random_indices_next = torch.randint(0, point_cloud_next.shape[0], (size,), generator=self.downsample_generator)
        point_cloud_next = point_cloud_next[random_indices_next, :]
        return point_cloud_first, point_cloud_next, mask, flow, class_ids

    def downsample_point_cloud_with_fps(
        self,
        point_cloud_first: torch.Tensor,
        point_cloud_next: torch.Tensor,
        mask: torch.Tensor | None,
        flow: torch.Tensor,
        class_ids: torch.Tensor,
        size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Downsample point clouds using GPU-based Farthest Point Sampling (FPS).
        Uses parallel processing for first and next point clouds when GPU is available.

        Args:
            point_cloud_first: First point cloud [N, 3]
            point_cloud_next: Next point cloud [M, 3]
            mask: Instance mask [N] or None
            flow: Scene flow [N, 3]
            class_ids: Class IDs [N]
            size: Target number of points

        Returns:
            Downsampled point clouds and corresponding attributes
        """
        use_gpu = torch.cuda.is_available()
        original_device = point_cloud_first.device

        # Prepare data for GPU if available
        if use_gpu:
            point_cloud_first_gpu = point_cloud_first.to("cuda")
            point_cloud_next_gpu = point_cloud_next.to("cuda")
            mask_gpu = mask.to("cuda") if mask is not None else None
            flow_gpu = flow.to("cuda") if flow is not None else None
            if isinstance(class_ids, np.ndarray):
                class_ids_gpu = torch.from_numpy(class_ids).to("cuda")
            else:
                class_ids_gpu = class_ids.to("cuda") if class_ids is not None else None
        else:
            point_cloud_first_gpu = point_cloud_first
            point_cloud_next_gpu = point_cloud_next
            mask_gpu = mask
            flow_gpu = flow
            if isinstance(class_ids, np.ndarray):
                class_ids_gpu = torch.from_numpy(class_ids)
            else:
                class_ids_gpu = class_ids

        # Sequential processing (parallel streams disabled due to CUDA kernel thread-safety issues)
        # The furthest_point_sample CUDA kernel may not be thread-safe for concurrent execution
        need_downsample_first = point_cloud_first.shape[0] > size
        need_downsample_next = point_cloud_next.shape[0] > size

        # Process sequentially to avoid CUDA kernel conflicts
        if need_downsample_first:
            # Sequential processing (fallback for CPU or when only one needs downsampling)
            if need_downsample_first:
                point_cloud_first, mask, flow, class_ids = fps_downsample_with_attributes(
                    point_cloud_first_gpu, size, mask_gpu, flow_gpu, class_ids_gpu
                )
            else:
                point_cloud_first = point_cloud_first_gpu
                mask = mask_gpu
                flow = flow_gpu
                class_ids = class_ids_gpu

            if need_downsample_next:
                point_cloud_next = fps_downsample_with_attributes(point_cloud_next_gpu, size)[0]
            else:
                point_cloud_next = point_cloud_next_gpu

        return point_cloud_first, point_cloud_next, mask, flow, class_ids

    def __len__(self):
        return super().__len__()
