from utils.metrics import calculate_miou, calculate_epe
import torch
import torch.nn.functional as F
from utils.visualization_utils import remap_instance_labels, create_label_colormap, color_mask
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
def evaluate_predictions(
        pred_flows, 
        gt_flows, 
        pred_masks, 
        gt_masks, 
        device, 
        writer=None, 
        step=0,
        argoverse2=False,
        background_static_mask=None,
        foreground_static_mask=None,
        foreground_dynamic_mask=None
        ):
    """
    Evaluate model predictions by computing EPE and mIoU metrics.
    
    Args:
        pred_flows (list[torch.Tensor]): Predicted scene flows
        gt_flows (list[torch.Tensor]): Ground truth scene flows
        pred_masks (list[torch.Tensor]): Predicted instance masks
        gt_masks (list[torch.Tensor]): Ground truth instance masks
        device (torch.device): Device to run computations on
        writer (SummaryWriter): TensorBoard writer for logging
        step (int): Current training step
        
    Returns:
        tuple: (epe_mean, miou_mean) containing the computed metrics
    """
    # Compute EPE
    epe_mean = calculate_epe(pred_flows, gt_flows)
    # tqdm.write(f"\rEPE: {epe_mean.item()}", end="")
    if writer is not None:
        writer.add_scalar("epe", epe_mean.item(), step)
    
    # Compute mIoU
    miou_list = []
    for i in range(len(pred_masks)):
        gt_mask = remap_instance_labels(gt_masks[i])
        # tqdm.write(f"gt_mask size {max(gt_mask)}")
        miou_list.append(
            calculate_miou(
                pred_masks[i], 
                F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(device=device)
            )
        )
    miou_mean = torch.mean(torch.stack(miou_list))
    # tqdm.write(f"miou {miou_mean.item()}")
    if writer is not None:
        writer.add_scalar("miou", miou_mean.item(), step)
    
    if argoverse2:
        #calculate epe in three different masks
        if background_static_mask is not None:
            bg_epe = calculate_epe(
                [pred_flows[i][background_static_mask[i]] for i in range(len(pred_flows))],
                [gt_flows[i][background_static_mask[i]] for i in range(len(gt_flows))]
            )
            if writer is not None:
                writer.add_scalar("epe_bg", bg_epe.item(), step)
        if foreground_static_mask is not None:
            fg_static_epe = calculate_epe(
                [pred_flows[i][foreground_static_mask[i]] for i in range(len(pred_flows))],
                [gt_flows[i][foreground_static_mask[i]] for i in range(len(gt_flows))]
            )
            if writer is not None:
                writer.add_scalar("epe_fg_static", fg_static_epe.item(), step)
        if foreground_dynamic_mask is not None:
            fg_dynamic_epe = calculate_epe(
                [pred_flows[i][foreground_dynamic_mask[i]] for i in range(len(pred_flows))],
                [gt_flows[i][foreground_dynamic_mask[i]] for i in range(len(gt_flows))]
            )
            if writer is not None:
                writer.add_scalar("epe_fg_dynamic", fg_dynamic_epe.item(), step)

        if foreground_dynamic_mask is not None and background_static_mask is not None and foreground_static_mask is not None:
            threeway_mean = (fg_dynamic_epe + fg_static_epe + bg_epe) / 3.0
            if writer is not None:
                writer.add_scalar("epe_threeway_mean", threeway_mean.item(), step)

    if argoverse2:
        return epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean
    else:
        return epe_mean, miou_mean, None, None, None, None

def eval_model(scene_flow_predictor, mask_predictor, dataloader, config, device, writer, step):
    """
    Evaluate the model on the validation dataset.
    
    Args:
        scene_flow_predictor (nn.Module): Scene flow prediction model
        mask_predictor (nn.Module): Mask prediction model
        dataloader (DataLoader): DataLoader for the validation dataset
        config (dict): Configuration dictionary
        device (torch.device): Device to run computations on
        writer (SummaryWriter): TensorBoard writer for logging
        step (int): Current training step
    """
    scene_flow_predictor.eval()
    mask_predictor.eval()

    pred_flows = []
    gt_flows = []
    pred_masks = []
    gt_masks = []

    background_static_masks = []
    foreground_static_masks = []
    foreground_dynamic_masks = []

    # Try to ensure dataloader starts from initial state when possible
    iterator = None

    iterator = iter(dataloader)


    with torch.no_grad():
        for batch in iterator:
            point_cloud_firsts = [item.to(device) for item in batch["point_cloud_first"]]
            flow_gt = batch.get("flow")
            if flow_gt is not None:
                flow_gt = [item.to(device) for item in flow_gt]
            # Predict scene flow
            for i  in range(len(point_cloud_firsts)):
                try:
                    if getattr(config.model.flow, "name", "") == "FastFlow3D":
                        cur_idx = batch["idx"][i]
                        total_frames = batch["total_frames"][i]
                        # fetch next frame within bounds
                        next_idx = min(int(cur_idx) + 1, int(total_frames) - 1)
                        next_item = batch["self"][0].get_item(next_idx)
                        pc0 = point_cloud_firsts[i][:, :3]
                        pc1 = next_item["point_cloud_first"].to(device)[:, :3]
                        pose0 = batch.get("pose")
                        if pose0 is not None:
                            pose0 = pose0[i].to(device)
                        else:
                            pose0 = torch.eye(4, device=device)
                        pose1 = next_item.get("pose", torch.eye(4)).to(device)
                        flow_pred = scene_flow_predictor(pc0, pc1, pose0, pose1)
                    else:
                        flow_pred = scene_flow_predictor(point_cloud_firsts[i])
                    pred_flows.extend([flow_pred])
                except Exception as e:
                    from model.eulerflow_raw_mlp import QueryDirection
                    pred_flows.append(scene_flow_predictor(point_cloud_firsts[i], batch["idx"][i], batch["total_frames"][i], QueryDirection.FORWARD))  # Shape: [B, N, 3]

            gt_flows.extend(flow_gt)

            for point_cloud_first in point_cloud_firsts:
                if config.model.mask.name in ["EulerMaskMLP", "EulerMaskMLPResidual", "EulerMaskMLPRoutine"]:
                    masks_pred = mask_predictor(point_cloud_first, batch["idx"][i], batch["total_frames"][i])
                    masks_pred = masks_pred.permute(1, 0)
                else:
                    masks_pred = mask_predictor(point_cloud_first)
                pred_masks.extend([masks_pred])

            # optional Argoverse masks
            bsm = batch.get("background_static_mask")
            fsm = batch.get("foreground_static_mask")
            fdm = batch.get("foreground_dynamic_mask")
            gt_masks.extend(batch["dynamic_instance_mask"])
            if bsm is not None:
                background_static_masks.extend([m for m in bsm])
            if fsm is not None:
                foreground_static_masks.extend([m for m in fsm])
            if fdm is not None:
                foreground_dynamic_masks.extend([m for m in fdm])

    # Convert mask lists to None if empty
    bg_masks = background_static_masks if len(background_static_masks) > 0 else None
    fg_static_masks = foreground_static_masks if len(foreground_static_masks) > 0 else None
    fg_dynamic_masks = foreground_dynamic_masks if len(foreground_dynamic_masks) > 0 else None

    # Evaluate predictions and log metrics
    epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean = evaluate_predictions(
        pred_flows,
        gt_flows,
        pred_masks,
        gt_masks,
        device,
        writer=writer,
        step=step,
        argoverse2=config.dataset.name in ["AV2","AV2Sequence"],
        background_static_mask=bg_masks,
        foreground_static_mask=fg_static_masks,
        foreground_dynamic_mask=fg_dynamic_masks,
    )

    return epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean

def eval_model_general(scene_flow_predictor, mask_predictor, dataloader, config, device, writer, step):
    """
    General evaluator aligned with new data structures (TimeSyncedSceneFlowFrame).
    Expects each batch to provide a list of frames or a dict with key 'frames'.
    """
    scene_flow_predictor.eval()
    mask_predictor.eval()

    pred_flows = []
    gt_flows = []
    pred_masks = []
    gt_masks = []

    background_static_masks = []
    foreground_static_masks = []
    foreground_dynamic_masks = []

    with torch.no_grad():
        for batch in dataloader:
            # Normalize batch format to list[TimeSyncedSceneFlowFrame]
            if isinstance(batch, dict) and "frames" in batch:
                frames = batch["frames"]
            else:
                frames = batch  # assume batch is already list[TimeSyncedSceneFlowFrame]

            # Per-frame eval
            for frame in frames:
                # 1) Inputs: global point cloud (masked) and GT flow
                # pc points (visible subset) in world frame
                pc_points = torch.from_numpy(frame.pc.global_pc.points).to(device=device, dtype=torch.float32)  # [N,3]
                # GT flow in ego coordinates; align to visible points
                gt_mask_bool = frame.flow.mask  # numpy bool over full_pc
                # valid_flow is already masked to mask==True
                gt_flow_np = frame.flow.valid_flow  # [M,3] where M=sum(mask)
                # visible pc points correspond to frame.pc.pc (mask applied).
                # Align shape: both should have same length; if not, re-index via pc.mask
                visible_mask = frame.pc.mask  # numpy bool over full_pc
                # Map GT to visible subset: expected gt_mask_bool == visible_mask
                if gt_flow_np.shape[0] != visible_mask.sum():
                    # Fallback: align by mask intersection
                    common = (gt_mask_bool & visible_mask)
                    gt_flow_np = frame.flow.full_flow[common]

                gt_flow = torch.from_numpy(gt_flow_np).to(device=device, dtype=torch.float32)  # [N,3] aligned to pc_points

                # 2) Predict scene flow. If your predictor expects pc0->pc1 or only pc0, adapt accordingly.
                # Minimal API: predictor(pc_points) -> [N,3]
                try:
                    flow_pred = scene_flow_predictor(pc_points)  # torch [N,3]
                except TypeError:
                    # Alternative signature: predictor(pc_points, extra_args...)
                    flow_pred = scene_flow_predictor(pc_points)

                pred_flows.append(flow_pred)
                gt_flows.append(gt_flow)

                # 3) Predict masks
                # If your mask predictor expects a single tensor [N,3]:
                try:
                    masks_pred = mask_predictor(pc_points)  # shape [K,N] or [N,K]; normalize to [K,N]
                except TypeError:
                    masks_pred = mask_predictor(pc_points)
                if masks_pred.dim() == 2 and masks_pred.shape[0] < masks_pred.shape[1]:
                    # [K,N]
                    pred_masks.append(masks_pred)
                elif masks_pred.dim() == 2:
                    # [N,K] -> [K,N]
                    pred_masks.append(masks_pred.transpose(0, 1).contiguous())
                else:
                    # If mask predictor returns dict
                    logits = masks_pred.get("pred_logits", None)
                    masks = masks_pred.get("pred_masks", None)
                    if logits is not None and masks is not None:
                        # reduce to per-point segmentation if needed; for simplicity keep masks as-is
                        pred_masks.append(masks.squeeze(0))
                    else:
                        # fallback: identity single-class mask
                        pred_masks.append(torch.ones(1, pc_points.shape[0], device=device, dtype=torch.float32))

                # 4) GT masks: if you have dynamic instance mask per point
                # Here we use a single-class placeholder. Replace with your ground-truth mask if available:
                gt_masks.append(torch.zeros(pc_points.shape[0], device=device, dtype=torch.long))

                # 5) Optional AV2 three-way masks
                # If your dataloader provides per-point background/foreground masks, append them; otherwise skip.
                if isinstance(batch, dict):
                    for name, buf in [("background_static_mask", background_static_masks),
                                      ("foreground_static_mask", foreground_static_masks),
                                      ("foreground_dynamic_mask", foreground_dynamic_masks)]:
                        arr = batch.get(name, None)
                        if arr is not None:
                            # Expect numpy bool aligned to visible points
                            vis_mask = torch.from_numpy(arr).to(device=device)
                            buf.append(vis_mask)

    # Convert optionals
    bg_masks = background_static_masks if len(background_static_masks) > 0 else None
    fg_static_masks = foreground_static_masks if len(foreground_static_masks) > 0 else None
    fg_dynamic_masks = foreground_dynamic_masks if len(foreground_dynamic_masks) > 0 else None

    epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean = evaluate_predictions(
        pred_flows,
        gt_flows,
        pred_masks,
        gt_masks,
        device,
        writer=writer,
        step=step,
        argoverse2=(getattr(config.dataset, "name", "") in ["AV2", "AV2Sequence"]),
        background_static_mask=bg_masks,
        foreground_static_mask=fg_static_masks,
        foreground_dynamic_mask=fg_dynamic_masks,
    )
    return epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean