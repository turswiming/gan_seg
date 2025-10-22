from bucketed_scene_flow_eval.datasets.argoverse2 import Argoverse2CausalSceneFlow
from pathlib import Path
import torch
from utils.bucketed_epe_utils import extract_classid_from_argoverse2_data
class AV2SceneFlowZoo(Argoverse2CausalSceneFlow):
    def __init__(self, 
    root_dir: Path, 

    expected_camera_shape: tuple, 
    eval_args: dict, 
    with_rgb: bool, 
    flow_data_path: Path, 
    range_crop_type: str,
    downsample_factor: int,
    subsequence_length: int=2, 
    sliding_window_step_size: int=1, 
    with_ground: bool=False, 
    use_gt_flow: bool=True, 
    eval_type: str="bucketed_epe", 
    load_flow: bool=True,
    load_boxes: bool=False):
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
            load_boxes=load_boxes
        )
        self.downsample_factor = downsample_factor
    def transform_pc(self,pc: "torch.Tensor", transform: "torch.Tensor") -> "torch.Tensor":

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

        assert (
            ego_pc.shape == ego_flow.shape
        ), f"Expected same shape, got {ego_pc.shape} != {ego_flow.shape}"

        flowed_ego_pc = ego_pc + ego_flow

        # Transform both the point cloud and the flowed point cloud into the global frame
        global_pc = self.transform_pc(ego_pc, ego_to_global)
        global_flowed_pc = self.transform_pc(flowed_ego_pc, ego_to_global)
        global_flow = global_flowed_pc - global_pc
        return global_flow
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        point_cloud_first = sample[0].pc.full_global_pc.points[sample[0].pc.mask,:]
        point_cloud_first = point_cloud_first[::self.downsample_factor,:]
        point_cloud_next = sample[1].pc.full_global_pc.points[sample[1].pc.mask,:]
        point_cloud_next = point_cloud_next[::self.downsample_factor,:]
        flow = sample[0].flow.full_flow[sample[0].pc.mask,:]
        flow = flow[::self.downsample_factor,:]
        #convert flow to global frame
        transform = sample[0].pc.pose.ego_to_global.to_array()
        transform = torch.from_numpy(transform).float()
        point_cloud_first = torch.from_numpy(point_cloud_first).float()
        point_cloud_next = torch.from_numpy(point_cloud_next).float()
        flow = torch.from_numpy(flow).float()
        flow = self.ego_to_global_flow(point_cloud_first,flow,transform)
        if  hasattr(sample[0], "instance_ids"):
            mask = sample[0].instance_ids[sample[0].pc.mask]
            mask = mask[::self.downsample_factor]
            mask = torch.from_numpy(mask)
        else:
            mask = None

        new_sample = {
            "point_cloud_first": point_cloud_first,
            "point_cloud_next": point_cloud_next,
            "flow": flow,
        }
        if mask is not None:
            new_sample["mask"] = mask

        class_ids = extract_classid_from_argoverse2_data(sample[0])
        class_ids = class_ids[sample[0].pc.mask,]
        class_ids = class_ids[::self.downsample_factor,]
        new_sample["class_ids"] = class_ids
        return new_sample

    def __len__(self):
        return super().__len__()