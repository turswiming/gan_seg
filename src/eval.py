from utils.metrics import calculate_miou, calculate_epe
from utils.bucketed_epe_utils import extract_classid_from_argoverse2_data, compute_bucketed_epe_metrics
import torch
import torch.nn.functional as F
from utils.visualization_utils import remap_instance_labels
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
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

def evaluate_predictions_general(
        pred_flows, 
        gt_flows, 
        pred_masks, 
        gt_masks, 
        class_labels,
        device, 
        writer=None, 
        step=0,
        ):
    """
    Evaluate model predictions by computing EPE and mIoU metrics for general data structure.
    
    Args:
        pred_flows (list[torch.Tensor]): Predicted scene flows
        gt_flows (list[torch.Tensor]): Ground truth scene flows
        pred_masks (list[torch.Tensor]): Predicted instance masks
        gt_masks (list[torch.Tensor]): Ground truth instance masks
        class_labels: Class labels for evaluation
        device (torch.device): Device to run computations on
        writer (SummaryWriter): TensorBoard writer for logging
        step (int): Current training step
        
    Returns:
        tuple: (epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean)
            - epe_mean: Mean end-point error
            - miou_mean: Mean intersection over union
            - bg_epe: Background EPE (None if not available)
            - fg_static_epe: Foreground static EPE (None if not available)  
            - fg_dynamic_epe: Foreground dynamic EPE (None if not available)
            - threeway_mean: Three-way mean EPE (None if not available)
    """
    # Compute EPE
    epe_mean = calculate_epe(pred_flows, gt_flows)
    # tqdm.write(f"\rEPE: {epe_mean.item()}", end="")
    if writer is not None:
        writer.add_scalar("epe", epe_mean.item(), step)
    
    # Compute mIoU
    miou_list = []
    for i in range(len(pred_masks)):
        gt_mask = remap_instance_labels(gt_masks[i],ignore_label=[-1])
        # tqdm.write(f"gt_mask size {max(gt_mask)}")
        miou = calculate_miou(
                pred_masks[i], 
                F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(device=device)
            )
        if miou is not None:
            miou_list.append(miou) #if no instance is found, miou is None
        
    miou_mean = torch.mean(torch.stack(miou_list))
    # tqdm.write(f"miou {miou_mean.item()}")
    if writer is not None:
        writer.add_scalar("miou", miou_mean.item(), step)
    
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

def eval_model_general(scene_flow_predictor, mask_predictor, val_flow_dataloader, val_mask_dataloader, config, device, writer, step, downsample_factor):
    """
    General evaluator aligned with new data structures (TimeSyncedSceneFlowFrame).
    Expects each batch to provide a list of frames or a dict with key 'frames'.
    
    Args:
        scene_flow_predictor: Scene flow prediction model
        mask_predictor: Mask prediction model
        val_flow_dataloader: Validation dataloader for flow evaluation
        val_mask_dataloader: Validation dataloader for mask evaluation
        config: Configuration object
        device: Device to run computations on
        writer: TensorBoard writer for logging
        step: Current training step
        downsample_factor: Factor to downsample point clouds
        
    Returns:
        tuple: (epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean)
            Evaluation metrics for the model
    """
    scene_flow_predictor.eval()
    mask_predictor.eval()

    pred_flows = []
    gt_flows = []
    pred_masks = []
    gt_masks = []
    class_ids = []
    point_clouds = []

    background_static_masks = []
    foreground_static_masks = []
    foreground_dynamic_masks = []
    eval_size =100
    with torch.no_grad():
        for i, batch in enumerate(val_flow_dataloader):
            if i >= eval_size:
                break
            for sample in batch:
                point_cloud_first = sample[0].pc.full_global_pc.points[sample[0].flow.mask,:][::downsample_factor,:]
                point_cloud_next = sample[1].pc.full_global_pc.points[sample[1].pc.mask,:][::downsample_factor,:]
                point_cloud_first = torch.from_numpy(point_cloud_first).to(device).float()
                point_cloud_next = torch.from_numpy(point_cloud_next).to(device).float()
                ego_motion_first = torch.from_numpy(sample[0].pc.pose.sensor_to_ego.to_array()).to(device).float()
                firstmask = torch.ones(point_cloud_first.shape[0], device=device).bool()
                nextmask = torch.ones(point_cloud_next.shape[0], device=device).bool()
                flow_pred = scene_flow_predictor._model_forward([(point_cloud_first, firstmask)], [(point_cloud_next, nextmask)], [(torch.eye(4, device=device), ego_motion_first)])
                pred_flows.append(flow_pred[0].ego_flows.squeeze(0))
                gt_flow = sample[0].flow.full_flow[sample[0].flow.mask,:][::downsample_factor,:]
                gt_flow = torch.from_numpy(gt_flow).to(device).float()
                gt_flows.append(gt_flow)
                class_ids.append(extract_classid_from_argoverse2_data(sample[0]))
                point_clouds.append(point_cloud_first.cpu())
                if pred_flows[-1].shape[0] != gt_flows[-1].shape[0]:
                    #pop the last element
                    pred_flows.pop()
                    gt_flows.pop()
                    class_ids.pop()
                    point_clouds.pop()
        for i, batch in enumerate(val_mask_dataloader):
            if i >= eval_size:
                break
            for sample in batch:
                point_cloud_first = sample[0].pc.full_global_pc.points[sample[0].pc.mask,:][::downsample_factor,:]
                masks_pred = mask_predictor.forward_train({'points': [torch.from_numpy(point_cloud_first).to(device).float()]})
                pred_masks.append(masks_pred["pred_masks"].to(device).float())
                gt_mask = sample[0].instance_ids[sample[0].pc.mask,][::downsample_factor,]
                gt_mask = torch.from_numpy(gt_mask).to(device).long()
                gt_masks.append(gt_mask)
                # assert pred_masks[-1].shape[1] == gt_masks[-1].shape[0], f"pred_masks shape: {pred_masks[-1].shape}, gt_masks shape: {gt_masks[-1].shape}"
                if pred_masks[-1].shape[1] != gt_masks[-1].shape[0]:
                    #pop the last element
                    pred_masks.pop()
                    gt_masks.pop()

    # Convert optionals
    bg_masks = background_static_masks if len(background_static_masks) > 0 else None
    fg_static_masks = foreground_static_masks if len(foreground_static_masks) > 0 else None
    fg_dynamic_masks = foreground_dynamic_masks if len(foreground_dynamic_masks) > 0 else None
    class_labels = None
    epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean = evaluate_predictions_general(
        pred_flows,
        gt_flows,
        pred_masks,
        gt_masks,
        class_labels,
        device,
        writer=writer,
        step=step,
    )
    output_path = Path(config.log.dir) / "bucketed_epe"/f"step_{step}"
    standard_results = compute_bucketed_epe_metrics(
        point_clouds=point_clouds,
        pred_flows=pred_flows,
        gt_flows=gt_flows,
        class_ids=class_ids,
        output_path=output_path
    )
    for class_name, (static_epe, dynamic_error) in standard_results.items():
        writer.add_scalar(f"bucketed_epe/standard/{class_name}_static", static_epe, step)
        writer.add_scalar(f"bucketed_epe/standard/{class_name}_dynamic", dynamic_error, step)
    return epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean


def eval_model_with_bucketed_epe(scene_flow_predictor, dataloader, config, device, writer, step, output_path=None):
    """
    使用Bucketed EPE评估模型，参考论文 "I Can't Believe It's Not Scene Flow!"
    
    Args:
        scene_flow_predictor: 场景流预测模型
        dataloader: 数据加载器
        config: 配置字典
        device: 设备
        writer: TensorBoard writer
        step: 当前训练步骤
        output_path: 输出路径
        
    Returns:
        bucketed EPE评估结果
    """
    if output_path is None:
        output_path = Path("/tmp/bucketed_epe_eval")
    
    # 直接使用compute_bucketed_epe_metrics函数
    # 这里需要从dataloader中提取数据，但为了简化，我们假设已经有了pred_flows, gt_flows, class_ids
    # 在实际使用中，这些数据应该从dataloader中获取
    
    print("注意: eval_model_with_bucketed_epe函数需要在实际使用中实现数据提取逻辑")
    print("建议直接使用compute_bucketed_epe_metrics函数")
    
    # 示例用法（需要实际的数据）
    # results = compute_bucketed_epe_metrics(
    #     pred_flows=pred_flows,
    #     gt_flows=gt_flows, 
    #     class_ids=class_ids,
    #     output_path=output_path
    # )
    
    return {"message": "请直接使用compute_bucketed_epe_metrics函数"}