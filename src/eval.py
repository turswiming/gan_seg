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
                    flow_pred = scene_flow_predictor(point_cloud_firsts[i])
                    pred_flows.extend([flow_pred])
                except Exception as e:
                    from model.eulerflow_raw_mlp import QueryDirection
                    pred_flows.append(scene_flow_predictor(point_cloud_firsts[i], batch["idx"][i], batch["total_frames"][i], QueryDirection.FORWARD))  # Shape: [B, N, 3]

            gt_flows.extend(flow_gt)

            for point_cloud_first in point_cloud_firsts:
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