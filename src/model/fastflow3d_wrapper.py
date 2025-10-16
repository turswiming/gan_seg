import os
import sys
from typing import Optional, Tuple

import torch


class FastFlow3DInference:
    """
    Lightweight adapter to run SceneFlowZoo FastFlow3D from this repo.

    Usage:
        wrapper = FastFlow3DInference(scene_flow_zoo_root="/workspace/gan_seg/SceneFlowZoo", ckpt_path=None, device="cuda")
        flow = wrapper.predict(pc0, pc1, pose0, pose1)
    """

    def __init__(self, scene_flow_zoo_root: str, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        if scene_flow_zoo_root not in sys.path:
            sys.path.append(scene_flow_zoo_root)

        # Delayed imports after sys.path update
        from SceneFlowZoo.configs.pseudoimage import (  # type: ignore
            VOXEL_SIZE as SFZ_VOXEL_SIZE,
            PSEUDO_IMAGE_DIMS as SFZ_PSEUDO_IMAGE_DIMS,
            POINT_CLOUD_RANGE as SFZ_POINT_CLOUD_RANGE,
        )
        from SceneFlowZoo.models.feed_forward.fast_flow_3d import (  # type: ignore
            FastFlow3D,
            FastFlow3DBackboneType,
            FastFlow3DHeadType,
        )

        self._TorchFullFrameInputSequence = __import__(
            "SceneFlowZoo.dataloaders.dataclasses", fromlist=["TorchFullFrameInputSequence"]
        ).TorchFullFrameInputSequence
        self._LoaderType = __import__(
            "bucketed_scene_flow_eval.interfaces", fromlist=["LoaderType"]
        ).LoaderType

        self.device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

        # Build FastFlow3D model
        self.model = FastFlow3D(
            VOXEL_SIZE=SFZ_VOXEL_SIZE,
            PSEUDO_IMAGE_DIMS=SFZ_PSEUDO_IMAGE_DIMS,
            POINT_CLOUD_RANGE=SFZ_POINT_CLOUD_RANGE,
            FEATURE_CHANNELS=64,
            SEQUENCE_LENGTH=2,
            bottleneck_head=FastFlow3DHeadType.LINEAR,
            backbone=FastFlow3DBackboneType.UNET,
        ).to(self.device)

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            # Allow both direct state_dict and wrapped
            state_dict = state.get("state_dict", state)
            try:
                self.model.load_state_dict(state_dict, strict=False)
            except Exception:
                self.model.load_state_dict(state, strict=False)

        self.model.eval()

    def _pad_pair(self, pc0: torch.Tensor, pc1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pad two point clouds to same length into a fixed-size array representation and return masks.
        Returns:
            full_pc: (2, PadN, 3)
            full_mask: (2, PadN)
            pad_n: scalar length
        """
        assert pc0.dim() == 2 and pc0.shape[1] == 3
        assert pc1.dim() == 2 and pc1.shape[1] == 3
        n0 = pc0.shape[0]
        n1 = pc1.shape[0]
        pad_n = max(n0, n1)

        def _pad(pc: torch.Tensor, target_n: int) -> Tuple[torch.Tensor, torch.Tensor]:
            pad_len = target_n - pc.shape[0]
            if pad_len > 0:
                pad = torch.zeros((pad_len, 3), dtype=pc.dtype, device=pc.device)
                pc_padded = torch.cat([pc, pad], dim=0)
            else:
                pc_padded = pc
            mask = torch.zeros((target_n,), dtype=torch.float32, device=pc.device)
            mask[: pc.shape[0]] = 1.0
            return pc_padded, mask

        pc0_pad, m0 = _pad(pc0, pad_n)
        pc1_pad, m1 = _pad(pc1, pad_n)
        full_pc = torch.stack([pc0_pad, pc1_pad], dim=0)
        full_mask = torch.stack([m0, m1], dim=0)
        return full_pc, full_mask, torch.tensor(pad_n, device=pc0.device)

    def _build_dummy_gt(self, pad_n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # (K-1=1, PadN, 3), (K-1=1, PadN), (K-1=1, PadN)
        gt_flowed = torch.zeros((1, pad_n, 3), dtype=torch.float32, device=device)
        gt_flowed_mask = torch.zeros((1, pad_n), dtype=torch.float32, device=device)
        gt_class = torch.zeros((1, pad_n), dtype=torch.float32, device=device)
        return gt_flowed, gt_flowed_mask, gt_class

    def predict(self, pc0: torch.Tensor, pc1: torch.Tensor, pose0: torch.Tensor, pose1: torch.Tensor) -> torch.Tensor:
        """
        Run FastFlow3D to predict ego-frame flow for pc0.

        Args:
            pc0: (N0, 3) tensor in sensor frame at t0
            pc1: (N1, 3) tensor in sensor frame at t1
            pose0: (4, 4) ego_to_global at t0
            pose1: (4, 4) ego_to_global at t1
        Returns:
            flow0: (N0, 3) ego-frame flow for points in pc0
        """
        device = self.device
        pc0 = pc0.to(device).float()
        pc1 = pc1.to(device).float()
        pose0 = pose0.to(device).float()
        pose1 = pose1.to(device).float()

        full_pc, full_mask, pad_n = self._pad_pair(pc0, pc1)
        gt_flowed, gt_flowed_mask, gt_class = self._build_dummy_gt(int(pad_n.item()), device)

        # K=2 frames
        pc_poses_sensor_to_ego = torch.stack([torch.eye(4, device=device), torch.eye(4, device=device)], dim=0)
        pc_poses_ego_to_global = torch.stack([pose0, pose1], dim=0)

        # Empty RGB slots
        rgb_images = torch.zeros((2, 0, 4, 0, 0), dtype=torch.float32, device=device)
        rgb_poses_sensor_to_ego = torch.zeros((2, 0, 4, 4), dtype=torch.float32, device=device)
        rgb_poses_ego_to_global = torch.zeros((2, 0, 4, 4), dtype=torch.float32, device=device)
        rgb_projected_points = torch.zeros((2, 0, int(pad_n.item()), 2), dtype=torch.float32, device=device)
        rgb_projected_points_mask = torch.zeros((2, 0, int(pad_n.item())), dtype=torch.float32, device=device)

        seq = self._TorchFullFrameInputSequence(
            dataset_idx=0,
            sequence_log_id="custom",
            sequence_idx=0,
            full_pc=full_pc,
            full_pc_mask=full_mask,
            full_pc_gt_flowed=gt_flowed,
            full_pc_gt_flowed_mask=gt_flowed_mask,
            full_pc_gt_class=gt_class,
            pc_poses_sensor_to_ego=pc_poses_sensor_to_ego,
            pc_poses_ego_to_global=pc_poses_ego_to_global,
            auxillary_pc=None,
            rgb_images=rgb_images,
            rgb_poses_sensor_to_ego=rgb_poses_sensor_to_ego,
            rgb_poses_ego_to_global=rgb_poses_ego_to_global,
            rgb_projected_points=rgb_projected_points,
            rgb_projected_points_mask=rgb_projected_points_mask,
            loader_type=self._LoaderType.CAUSAL,
        )

        with torch.no_grad():
            outputs = self.model.inference_forward([seq], logger=None)
        out = outputs[0]

        # out.get_full_ego_flow(0) returns (PadN, 3) with mask out.get_full_flow_mask(0)
        full_flow = out.get_full_ego_flow(0)
        full_flow_mask = out.get_full_flow_mask(0)
        flow0 = full_flow[full_mask[0].bool() & full_flow_mask]
        # Align to original N0 length (drop any extra due to voxel mask)
        return flow0[: pc0.shape[0]]



