"""
Forward pass utilities for scene flow and mask prediction.

This module contains forward pass functions that were
previously in main.py, organized for better modularity.
"""

import torch
from model.eulerflow_raw_mlp import QueryDirection
from SceneFlowZoo.models.feed_forward.fast_flow_3d import FastFlow3D
from OGCModel.flownet_kitti import FlowStep3D


def forward_scene_flow(point_cloud_firsts, point_cloud_nexts, sample, flow_predictor, config, train_flow, device):
    """Perform forward pass for scene flow prediction.

    Args:
        point_cloud_firsts: List of first point clouds
        point_cloud_nexts: List of next point clouds
        sample: Data sample
        flow_predictor: Scene flow model
        config: Configuration object
        train_flow: Whether in training mode
        device: Training device

    Returns:
        tuple: (pred_flow, reverse_pred_flow, longterm_pred_flow)
    """
    pred_flow = []
    reverse_pred_flow = []
    longterm_pred_flow = {}

    if train_flow:
        for i in range(len(point_cloud_firsts)):
            if config.model.flow.name in config.model.euler_flow_models:
                if sample["k"][i] == 1:
                    pred_flow.append(
                        flow_predictor(
                            point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i], QueryDirection.FORWARD
                        )
                    )
                    reverse_pred_flow.append(
                        flow_predictor(
                            point_cloud_nexts[i],
                            sample["idx"][i] + 1,
                            sample["total_frames"][i],
                            QueryDirection.REVERSE,
                        )
                    )
                else:
                    # Multi-step prediction
                    pred_pc = point_cloud_firsts[i].clone()
                    for k in range(0, sample["k"][i]):
                        pred_flow_temp = flow_predictor(
                            pred_pc, sample["idx"][i] + k, sample["total_frames"][i], QueryDirection.FORWARD
                        )
                        pred_pc = pred_pc + pred_flow_temp
                        longterm_pred_flow[sample["idx"][i] + k + 1] = pred_pc.clone()
                        if k == 0:
                            pred_flow.append(pred_flow_temp)

                    # Reverse multi-step prediction
                    pred_pc = point_cloud_nexts[i].clone()
                    for k in range(0, sample["k"][i]):
                        pred_flow_temp = flow_predictor(
                            pred_pc, sample["idx"][i] - k + 1, sample["total_frames"][i], QueryDirection.REVERSE
                        )
                        pred_pc = pred_pc + pred_flow_temp
                        longterm_pred_flow[sample["idx"][i] - k] = pred_pc.clone()
                        if k == 0:
                            reverse_pred_flow.append(pred_flow_temp)
            else:
                pred_flow.append(flow_predictor(point_cloud_firsts[i]))
    else:
        with torch.no_grad():
            for i in range(len(point_cloud_firsts)):
                if config.model.flow.name in config.model.euler_flow_models:
                    pred_flow.append(
                        flow_predictor(
                            point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i], QueryDirection.FORWARD
                        )
                    )
                else:
                    pred_flow.append(flow_predictor(point_cloud_firsts[i]))

    return pred_flow, reverse_pred_flow, longterm_pred_flow


def forward_mask_prediction(point_cloud_firsts, sample, mask_predictor, config, train_mask):
    """Perform forward pass for mask prediction.

    Args:
        point_cloud_firsts: List of first point clouds
        sample: Data sample
        mask_predictor: Mask prediction model
        config: Configuration object
        train_mask: Whether in training mode

    Returns:
        list: Predicted masks
    """
    pred_mask = []

    if train_mask:
        for i in range(len(point_cloud_firsts)):
            if config.model.mask.name in config.model.euler_mask_models:
                mask = mask_predictor(point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i])
                mask = mask.permute(1, 0)
                pred_mask.append(mask)
            else:
                pred_mask.append(mask_predictor(point_cloud_firsts[i]))
    else:
        with torch.no_grad():
            for i in range(len(point_cloud_firsts)):
                if config.model.mask.name in config.model.euler_mask_models:
                    mask = mask_predictor(point_cloud_firsts[i], sample["idx"][i], sample["total_frames"][i])
                    mask = mask.permute(1, 0)
                    pred_mask.append(mask)
                else:
                    pred_mask.append(mask_predictor(point_cloud_firsts[i]))

    return pred_mask


def inference_models_general(
    flow_predictor, mask_predictor, sample, config, train_flow=False, downsample=None, augment_params=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    point_cloud_firsts = [s["point_cloud_first"].to(device).float() for s in sample]
    point_cloud_nexts = [s["point_cloud_next"].to(device).float() for s in sample]
    gt_flows = [s["flow"].to(device).float() for s in sample]
    if hasattr(config.training, "use_icp_inference_flow") and not config.training.use_icp_inference_flow:
        return_value = forward_scene_flow_general_old(
            point_cloud_firsts,
            point_cloud_nexts,
            flow_predictor,
            config.dataset.name,
            train_flow=train_flow,
            return_modified_point_cloud=False,
        )
    else:
        return_value = forward_scene_flow_general(
            point_cloud_firsts,
            point_cloud_nexts,
            flow_predictor,
            config.dataset.name,
            train_flow=train_flow,
            return_modified_point_cloud=False,
        )

    if train_flow and config.model.flow.name == "FlowStep3D":
        flow, cascade_flow_outs = return_value
    else:
        flow = return_value
        cascade_flow_outs = None
    if downsample is not None:
        point_cloud_firsts = [item[::downsample, :] for item in point_cloud_firsts]
        point_cloud_nexts = [item[::downsample, :] for item in point_cloud_nexts]
        flow = [item[::downsample, :] for item in flow]
        if cascade_flow_outs is not None:
            for i in range(len(cascade_flow_outs)):
                cascade_flow_outs[i] = [item[::downsample, :] for item in cascade_flow_outs[i]]
    # augment transform
    for i in range(len(point_cloud_firsts)):
        if cascade_flow_outs is not None:
            cascade_flow_outs_item = [item[i] for item in cascade_flow_outs]
        else:
            cascade_flow_outs_item = None
        (point_cloud_firsts[i], point_cloud_nexts[i], flow[i], cascade_flow_outs_item) = augment_transform(
            point_cloud_firsts[i], point_cloud_nexts[i], flow[i], cascade_flow_outs_item, augment_params
        )
        if cascade_flow_outs is not None:
            for j in range(len(cascade_flow_outs)):
                cascade_flow_outs[j][i] = cascade_flow_outs_item[j]
    pred_masks = forward_mask_prediction_general(point_cloud_firsts, mask_predictor)
    return flow, pred_masks, point_cloud_firsts, point_cloud_nexts, cascade_flow_outs


def forward_scene_flow_general_old(
    point_cloud_firsts,
    point_cloud_nexts,
    flow_predictor,
    dataset_name,
    train_flow=False,
    return_modified_point_cloud=False,
):
    if isinstance(flow_predictor, FlowStep3D):
        if dataset_name != "KITTISF_new":
            point_cloud_firsts = [pc[:, [0, 2, 1]] for pc in point_cloud_firsts]
            point_cloud_nexts = [pc[:, [0, 2, 1]] for pc in point_cloud_nexts]
        point_cloud_firsts = [pc.unsqueeze(0) for pc in point_cloud_firsts]
        point_cloud_nexts = [pc.unsqueeze(0) for pc in point_cloud_nexts]
        flow_outs = []
        pc1_list = []
        pc2_list = []
        for pc1, pc2 in zip(point_cloud_firsts, point_cloud_nexts):

            pc1 = pc1.squeeze(0)
            pc2 = pc2.squeeze(0)
            pc1_list.append(pc1)
            pc2_list.append(pc2)
        pc1_input = torch.cat([pc.unsqueeze(0) for pc in pc1_list], dim=0)
        pc2_input = torch.cat([pc.unsqueeze(0) for pc in pc2_list], dim=0)
        casecde_flow_outs = flow_predictor(pc1_input, pc2_input, pc1_input, pc2_input, iters=5)
        flow_outs = casecde_flow_outs[-1]
        if dataset_name != "KITTISF_new":
            flow_outs = [pc[:, [0, 2, 1]] for pc in flow_outs]
            point_cloud_firsts = [pc[:, [0, 2, 1]] for pc in point_cloud_firsts]
            point_cloud_nexts = [pc[:, [0, 2, 1]] for pc in point_cloud_nexts]
            for flow_step in casecde_flow_outs:
                flow_step = [pc[:, [0, 2, 1]] for pc in flow_step]
        if train_flow:
            if return_modified_point_cloud:
                return flow_outs, casecde_flow_outs[:-1], point_cloud_firsts
            else:
                return flow_outs, casecde_flow_outs[:-1]
        if return_modified_point_cloud:
            return flow_outs, point_cloud_nexts
        else:
            return flow_outs
    elif isinstance(flow_predictor, FastFlow3D):
        first_masks = [torch.ones(pc.shape[0], device=pc.device).bool() for pc in point_cloud_firsts]
        next_masks = [torch.ones(pc.shape[0], device=pc.device).bool() for pc in point_cloud_nexts]
        first_inputs = [[pc, mask] for pc, mask in zip(point_cloud_firsts, first_masks)]
        next_inputs = [[pc, mask] for pc, mask in zip(point_cloud_nexts, next_masks)]
        transform_inputs = [
            [torch.eye(4, device=pc.device), torch.eye(4, device=pc.device)] for pc in point_cloud_firsts
        ]
        flow_out_seqs = flow_predictor._model_forward(first_inputs, next_inputs, transform_inputs)
        flow = [item.ego_flows.squeeze(0) for item in flow_out_seqs]
        return flow
    else:
        raise ValueError(f"Flow predictor {flow_predictor.name} not supported")


def forward_scene_flow_general(
    point_cloud_firsts,
    point_cloud_nexts,
    flow_predictor,
    dataset_name,
    train_flow=False,
    return_modified_point_cloud=False,
):
    if isinstance(flow_predictor, FlowStep3D):

        # if dataset_name == "KITTISF_new":
        #     ground_masks = [pc[:,1] < -1.4 for pc in point_cloud_firsts]
        #     original_point_cloud_firsts = point_cloud_firsts.copy()
        #     original_point_cloud_nexts = point_cloud_nexts.copy()
        #     point_cloud_firsts = [pc[~ground_mask] for pc,ground_mask in zip(point_cloud_firsts,ground_masks)]
        #     point_cloud_nexts = [pc[~ground_mask] for pc,ground_mask in zip(point_cloud_nexts,ground_masks)]
        point_cloud_firsts = [pc.unsqueeze(0) for pc in point_cloud_firsts]
        point_cloud_nexts = [pc.unsqueeze(0) for pc in point_cloud_nexts]
        # if all point_cloud_firsts and point_cloud_nexts are the same, then only forward once
        if (
            all(pc1.shape == pc2.shape for pc1, pc2 in zip(point_cloud_firsts, point_cloud_nexts))
            and len(set([pc.shape[1] for pc in point_cloud_firsts])) == 1
            and False
        ):
            point_cloud_firsts = torch.cat(point_cloud_firsts, dim=0)
            point_cloud_nexts = torch.cat(point_cloud_nexts, dim=0)
            casecde_flow_outs = flow_predictor(
                point_cloud_firsts, point_cloud_nexts, point_cloud_firsts, point_cloud_nexts, iters=5
            )
            flow_outs = casecde_flow_outs[-1]
            flow_outs = [flow_outs[i].squeeze(0) for i in range(len(flow_outs))]
        else:
            flow_outs = []
            cascade_flow_pred_res_batch = []
            pc1_list = []
            pc2_list = []
            pc1_ground_mask_list = []
            pc2_ground_mask_list = []
            T_list = []
            flow_pred_org_list = []
            for pc1, pc2 in zip(point_cloud_firsts, point_cloud_nexts):
                from dataset.kittisf_sceneflow import get_global_transform_matrix

                pc1 = pc1.squeeze(0)
                pc2 = pc2.squeeze(0)
                pc1_ground_mask = pc1[:, 1] < -1.4
                pc2_ground_mask = pc2[:, 1] < -1.4
                T = get_global_transform_matrix(pc1[~pc1_ground_mask], pc2[~pc2_ground_mask])
                # ensure T is torch tensor on same device/dtype as pc1
                T = torch.from_numpy(T)
                T = T.to(device=pc1.device, dtype=pc1.dtype)

                rot, transl = T[:3, :3], T[:3, 3]

                # compute with torch: einsum('ij,nj->ni', rot, pc1) == pc1 @ rot.T
                flow_pred_org = pc1 @ rot.T + transl - pc1
                flow_pred_org = flow_pred_org.to(dtype=torch.float32)
                pc1 = pc1 @ rot.T + transl
                pc1 = pc1.to(dtype=torch.float32)
                pc1_list.append(pc1)
                pc2_list.append(pc2)
                pc1_ground_mask_list.append(pc1_ground_mask)
                pc2_ground_mask_list.append(pc2_ground_mask)
                flow_pred_org_list.append(flow_pred_org)
                T_list.append(T)
            pc1_input = torch.cat([pc.unsqueeze(0) for pc in pc1_list], dim=0)
            pc2_input = torch.cat([pc.unsqueeze(0) for pc in pc2_list], dim=0)
            point_cloud_firsts = [pc.squeeze(0) for pc in pc1_input]
            point_cloud_nexts = [pc.squeeze(0) for pc in pc2_input]
            flow_preds = flow_predictor(pc1_input, pc2_input, pc1_input, pc2_input, iters=4)
            flow_prd_lsq = flow_preds[-1]
            flow_outs = []
            for flow_pred_org, flow_prd_lsq_item, pc1_ground_mask in zip(
                flow_pred_org_list, flow_prd_lsq, pc1_ground_mask_list
            ):

                flow_pred_res = flow_pred_org.clone()
                flow_pred_res[~pc1_ground_mask] += flow_prd_lsq_item[~pc1_ground_mask]
                flow_outs.append(flow_pred_res)
            casecde_flow_outs = []
            for flow_pred_step in flow_preds:
                flow_pred_step_res = []
                # 拆成列表
                flow_pred_step_list = [flow_pred_step[i] for i in range(flow_pred_step.shape[0])]
                for flow_pred_org, flow_prd_lsq_item, pc1_ground_mask in zip(
                    flow_pred_org_list, flow_pred_step_list, pc1_ground_mask_list
                ):

                    flow_pred_res = flow_pred_org.clone()
                    flow_pred_res[~pc1_ground_mask] += flow_prd_lsq_item[~pc1_ground_mask]
                    flow_pred_step_res.append(flow_pred_res)
                casecde_flow_outs.append(flow_pred_step_res)
                pass
        if train_flow:
            if return_modified_point_cloud:
                return flow_outs, casecde_flow_outs[:, -1], point_cloud_firsts
            else:
                return flow_outs, casecde_flow_outs[:-1]
        if return_modified_point_cloud:
            return flow_outs, point_cloud_firsts
        else:
            return flow_outs
    elif isinstance(flow_predictor, FastFlow3D):
        first_masks = [torch.ones(pc.shape[0], device=pc.device).bool() for pc in point_cloud_firsts]
        next_masks = [torch.ones(pc.shape[0], device=pc.device).bool() for pc in point_cloud_nexts]
        first_inputs = [[pc, mask] for pc, mask in zip(point_cloud_firsts, first_masks)]
        next_inputs = [[pc, mask] for pc, mask in zip(point_cloud_nexts, next_masks)]
        transform_inputs = [
            [torch.eye(4, device=pc.device), torch.eye(4, device=pc.device)] for pc in point_cloud_firsts
        ]
        flow_out_seqs = flow_predictor._model_forward(first_inputs, next_inputs, transform_inputs)
        flow = [item.ego_flows.squeeze(0) for item in flow_out_seqs]
        return flow
    else:
        raise ValueError(f"Flow predictor {flow_predictor.name} not supported")


def forward_mask_prediction_general(pc_tensors, mask_predictor):
    pred_masks = []
    if len(set([pc.shape[1] for pc in pc_tensors])) == 1:
        pc_tensors = torch.cat([pc.unsqueeze(0) for pc in pc_tensors], dim=0)
        pred_mask = mask_predictor.forward(pc_tensors, pc_tensors)
        for i in range(pred_mask.shape[0]):
            pred_masks.append(pred_mask[i].permute(1, 0))
        return pred_masks
    for pc_tensor in pc_tensors:
        pc_tensor = pc_tensor.unsqueeze(0).contiguous()
        pred_mask = mask_predictor.forward(pc_tensor, pc_tensor)
        pred_mask = pred_mask.squeeze(0)
        pred_mask = pred_mask.permute(1, 0)
        pred_masks.append(pred_mask)
    return pred_masks


def augment_transform(pc1, pc2, flow, cascade_flow_outs, aug_params):
    angle_range = aug_params.angle_range
    translation_range = aug_params.translation_range
    scale_range = aug_params.scale_range
    mirror_x = aug_params.mirror_x
    mirror_z = aug_params.mirror_z
    # decentralize point cloud
    center_point = (pc1.mean(0) + pc2.mean(0)) / 2 * torch.tensor([1, 0, 1]).to(pc1.device)
    pc1 = pc1 - center_point
    pc2 = pc2 - center_point
    # random rotation along y axis
    angle = torch.rand(1).item() * (angle_range[1] - angle_range[0]) + angle_range[0]  # uniform(-π/4, π/4)
    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    rot = torch.tensor(
        [[cos_a.item(), 0, sin_a.item()], [0, 1, 0], [-sin_a.item(), 0, cos_a.item()]], dtype=torch.float32
    )
    rot = rot.to(pc1.device)
    pc1 = pc1 @ rot.T
    pc2 = pc2 @ rot.T
    flow = flow @ rot.T
    if cascade_flow_outs is not None:
        for i in range(len(cascade_flow_outs)):
            cascade_flow = cascade_flow_outs[i]
            cascade_flow = cascade_flow @ rot.T
            cascade_flow_outs[i] = cascade_flow
    # random translation
    translation = torch.rand(3).to(pc1.device) * 2 - 1  # uniform(-1, 1)
    translation = translation * torch.tensor([translation_range[0], translation_range[1], translation_range[2]]).to(
        pc1.device
    )
    translation = translation.to(pc1.device)
    pc1 = pc1 + translation
    pc2 = pc2 + translation

    # random scaling
    scale = torch.rand(1).item()
    scale = scale * (scale_range[1] - scale_range[0]) + scale_range[0]
    pc1 = pc1 * scale
    pc2 = pc2 * scale
    flow = flow * scale
    if cascade_flow_outs is not None:
        for i in range(len(cascade_flow_outs)):
            cascade_flow = cascade_flow_outs[i]
            cascade_flow = cascade_flow * scale
            cascade_flow_outs[i] = cascade_flow
    # random mirror in x, z axis
    if mirror_x:
        mirror_x = torch.rand(1).item()
        if mirror_x < 0.5:
            pc1[:, 0] = -pc1[:, 0]
            pc2[:, 0] = -pc2[:, 0]
            flow[:, 0] = -flow[:, 0]
            if cascade_flow_outs is not None:
                for i in range(len(cascade_flow_outs)):
                    cascade_flow = cascade_flow_outs[i]
                    cascade_flow[:, 0] = -cascade_flow[:, 0]
                    cascade_flow_outs[i] = cascade_flow
    if mirror_z:
        mirror_z = torch.rand(1).item()
        if mirror_z < 0.5:
            pc1[:, 2] = -pc1[:, 2]
            pc2[:, 2] = -pc2[:, 2]
            flow[:, 2] = -flow[:, 2]
            if cascade_flow_outs is not None:
                for i in range(len(cascade_flow_outs)):
                    cascade_flow = cascade_flow_outs[i]
                    cascade_flow[:, 2] = -cascade_flow[:, 2]
                    cascade_flow_outs[i] = cascade_flow
    # if mirror_z < 0.5:
    #     pc1[:, 2] = -pc1[:, 2]
    #     pc2[:, 2] = -pc2[:, 2]
    #     flow[:, 2] = -flow[:, 2]

    # Convert back to numpy for compatibility
    return pc1, pc2, flow, cascade_flow_outs


def downsample_point_clouds(pred_flow, pred_mask, point_cloud_firsts, point_cloud_nexts, downsample_factor):
    if pred_flow is not None:
        pred_flow = [item[::downsample_factor, :] for item in pred_flow]
    if pred_mask is not None:
        pred_mask = [item[:, ::downsample_factor] for item in pred_mask]
    if point_cloud_firsts is not None:
        point_cloud_firsts = [item[::downsample_factor, :] for item in point_cloud_firsts]
    if point_cloud_nexts is not None:
        point_cloud_nexts = [item[::downsample_factor, :] for item in point_cloud_nexts]
    return pred_flow, pred_mask, point_cloud_firsts, point_cloud_nexts
