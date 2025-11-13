from typing import Sequence
from utils.metrics import calculate_miou, calculate_epe
from utils.bucketed_epe_utils import (
    extract_classid_from_argoverse2_data,
    compute_bucketed_epe_metrics,
)
import torch
import torch.nn.functional as F
from utils.visualization_utils import remap_instance_labels
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from OGCModel.icp_util import icp
import numpy as np
from bucketed_scene_flow_eval.interfaces.abstract_dataset import LoaderType


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
    foreground_dynamic_mask=None,
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
                F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(device=device),
            )
        )
    miou_mean = torch.mean(torch.stack(miou_list))
    # tqdm.write(f"miou {miou_mean.item()}")
    if writer is not None:
        writer.add_scalar("miou", miou_mean.item(), step)

    if argoverse2:
        # calculate epe in three different masks
        if background_static_mask is not None:
            bg_epe = calculate_epe(
                [pred_flows[i][background_static_mask[i]] for i in range(len(pred_flows))],
                [gt_flows[i][background_static_mask[i]] for i in range(len(gt_flows))],
            )
            if writer is not None:
                writer.add_scalar("epe_bg", bg_epe.item(), step)
        if foreground_static_mask is not None:
            fg_static_epe = calculate_epe(
                [pred_flows[i][foreground_static_mask[i]] for i in range(len(pred_flows))],
                [gt_flows[i][foreground_static_mask[i]] for i in range(len(gt_flows))],
            )
            if writer is not None:
                writer.add_scalar("epe_fg_static", fg_static_epe.item(), step)
        if foreground_dynamic_mask is not None:
            fg_dynamic_epe = calculate_epe(
                [pred_flows[i][foreground_dynamic_mask[i]] for i in range(len(pred_flows))],
                [gt_flows[i][foreground_dynamic_mask[i]] for i in range(len(gt_flows))],
            )
            if writer is not None:
                writer.add_scalar("epe_fg_dynamic", fg_dynamic_epe.item(), step)

        if (
            foreground_dynamic_mask is not None
            and background_static_mask is not None
            and foreground_static_mask is not None
        ):
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
    min_points=100,
    type="val",
    config=None,
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
        min_points (int): Minimum number of points to consider for evaluation
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
        writer.add_scalar(f"{type}/epe", epe_mean.item(), step)

    # Compute mIoU
    miou_list = []
    for i in range(len(pred_masks)):
        gt_mask = remap_instance_labels(gt_masks[i])
        # tqdm.write(f"gt_mask size {max(gt_mask)}")
        miou = calculate_miou(
            pred_masks[i],
            F.one_hot(gt_mask.to(torch.long)).permute(1, 0).to(device=device),
            min_points=min_points,
        )
        if miou is not None:
            miou_list.append(miou)  # if no instance is found, miou is None

    miou_mean = torch.mean(torch.stack(miou_list))
    # tqdm.write(f"miou {miou_mean.item()}")
    if writer is not None:
        writer.add_scalar(f"{type}/miou", miou_mean.item(), step)

        return epe_mean, miou_mean, None, None, None, None


def eval_model(flow_predictor, mask_predictor, dataloader, config, device, writer, step):
    """
    Evaluate the model on the validation dataset.

    Args:
        flow_predictor (nn.Module): Scene flow prediction model
        mask_predictor (nn.Module): Mask prediction model
        dataloader (DataLoader): DataLoader for the validation dataset
        config (dict): Configuration dictionary
        device (torch.device): Device to run computations on
        writer (SummaryWriter): TensorBoard writer for logging
        step (int): Current training step
    """
    flow_predictor.eval()
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
            for i in range(len(point_cloud_firsts)):
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
                        flow_pred = flow_predictor(pc0, pc1, pose0, pose1)
                    else:
                        flow_pred = flow_predictor(point_cloud_firsts[i])
                    pred_flows.extend([flow_pred])
                except Exception as e:
                    from model.eulerflow_raw_mlp import QueryDirection

                    pred_flows.append(
                        flow_predictor(
                            point_cloud_firsts[i],
                            batch["idx"][i],
                            batch["total_frames"][i],
                            QueryDirection.FORWARD,
                        )
                    )  # Shape: [B, N, 3]

            gt_flows.extend(flow_gt)

            for point_cloud_first in point_cloud_firsts:
                if config.model.mask.name in [
                    "EulerMaskMLP",
                    "EulerMaskMLPResidual",
                    "EulerMaskMLPRoutine",
                ]:
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
        argoverse2=config.dataset.name in ["AV2", "AV2Sequence"],
        background_static_mask=bg_masks,
        foreground_static_mask=fg_static_masks,
        foreground_dynamic_mask=fg_dynamic_masks,
    )

    return epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean


def eval_model_general(
    flow_predictor,
    mask_predictor,
    val_flow_dataloader,
    val_mask_dataloader,
    config,
    device,
    writer,
    step,
    type="val",
    save_sample=False,
):
    """
    General evaluator aligned with new data structures (TimeSyncedSceneFlowFrame).
    Expects each batch to provide a list of frames or a dict with key 'frames'.

    Args:
        flow_predictor: Scene flow prediction model
        mask_predictor: Mask prediction model
        val_flow_dataloader: Validation dataloader for flow evaluation
        val_mask_dataloader: Validation dataloader for mask evaluation
        config: Configuration object
        device: Device to run computations on
        writer: TensorBoard writer for logging
        step: Current training step

    Returns:
        tuple: (epe_mean, miou_mean, bg_epe, fg_static_epe, fg_dynamic_epe, threeway_mean)
            Evaluation metrics for the model
    """
    flow_predictor.eval()
    mask_predictor.eval()
    skip_val_mask = getattr(config.eval, "skip_val_mask", False)
    skip_val_flow = getattr(config.eval, "skip_val_flow", False)
    pred_flows = []
    gt_flows = []
    pred_masks = []
    gt_masks = []
    class_ids = []
    point_clouds = []
    sequence_ids_flow = []
    sequence_ids_mask = []
    background_static_masks = []
    foreground_static_masks = []
    foreground_dynamic_masks = []
    eval_size = min(config.eval.eval_size, len(val_flow_dataloader))
    flow_eval_size = eval_size
    mask_eval_size = eval_size
    if hasattr(config.eval, "eval_flow_size"):
        flow_eval_size = config.eval.eval_flow_size
    if hasattr(config.eval, "eval_mask_size"):
        mask_eval_size = config.eval.eval_mask_size
    with torch.no_grad():
        print("Evaluating flow model")
        # add a progress bar
        with tqdm(total=flow_eval_size, desc="Evaluating flow model") as pbar:
            for i, batch in enumerate(val_flow_dataloader):
                pbar.update(1)
                if i >= flow_eval_size:
                    break
                if getattr(config.model.flow, "name", "") == "FlowStep3D":
                    from utils.forward_utils import forward_scene_flow_general, forward_scene_flow_general_old

                    point_cloud_firsts = [s["point_cloud_first"].to(device).float() for s in batch]
                    point_cloud_nexts = [s["point_cloud_next"].to(device).float() for s in batch]
                    if (
                        hasattr(config.training, "use_icp_inference_flow")
                        and not config.training.use_icp_inference_flow
                    ):
                        flow_pred = forward_scene_flow_general_old(
                            point_cloud_firsts,
                            point_cloud_nexts,
                            flow_predictor,
                            config.dataset.name,
                        )
                    else:
                        flow_pred = forward_scene_flow_general(
                            point_cloud_firsts,
                            point_cloud_nexts,
                            flow_predictor,
                            config.dataset.name,
                        )
                    pred_flows.extend(flow_pred)
                    gt_flows.extend([s["flow"].to(device).float() for s in batch])
                    point_clouds.extend(point_cloud_firsts)
                    if "class_ids" in batch[0].keys():
                        class_ids.extend([s["class_ids"] for s in batch])
                    sequence_ids_flow.extend([s["sequence_id"] for s in batch])
                    continue
                for sample in batch:
                    point_cloud_first = sample["point_cloud_first"]
                    point_cloud_first = point_cloud_first.to(device).float()
                    point_cloud_next = sample["point_cloud_next"].to(device).float()
                    point_cloud_first_ones = torch.ones(
                        point_cloud_first.shape[0], device=point_cloud_first.device
                    ).bool()
                    point_cloud_next_ones = torch.ones(point_cloud_next.shape[0], device=point_cloud_next.device).bool()
                    transform = torch.eye(4, device=point_cloud_first.device)
                    flow_pred = flow_predictor._model_forward(
                        [[point_cloud_first, point_cloud_first_ones]],
                        [[point_cloud_next, point_cloud_next_ones]],
                        [[transform, transform]],
                    )[0].ego_flows.squeeze(0)
                    # because the transform is identity, here it is the global flow

                    pred_flows.append(flow_pred)

                    gt_flows.append(sample["flow"].to(device).float())
                    # print the magnitude of the flow
                    if "class_ids" in sample.keys():
                        class_ids.append(sample["class_ids"])
                    point_clouds.append(point_cloud_first)
                    sequence_ids_flow.append(sample["sequence_id"])
        # exit(0)
        print("Evaluating mask model")
        point_clouds_mask = []
        with tqdm(total=mask_eval_size, desc="Evaluating mask model") as pbar:
            for i, batch in enumerate(val_mask_dataloader):
                pbar.update(1)
                if i >= mask_eval_size:
                    break
                point_cloud_firsts = [
                    s["point_cloud_first"][:: config.training.mask_downsample_factor, :].to(device).float()
                    for s in batch
                ]
                from utils.forward_utils import forward_mask_prediction_general

                # decentralize point cloud
                for i in range(len(point_cloud_firsts)):
                    from utils.forward_utils import augment_transform

                    point_cloud_firsts[i], _, _, _ = augment_transform(
                        point_cloud_firsts[i],
                        point_cloud_firsts[i],
                        torch.zeros(point_cloud_firsts[i].shape[0], 3).to(point_cloud_firsts[i].device),
                        None,
                        config.training.augment_params,
                    )
                gt_mask = [s["mask"].to(device).long()[:: config.training.mask_downsample_factor] for s in batch]

                masks_pred = forward_mask_prediction_general(point_cloud_firsts, mask_predictor)
                point_clouds_mask.extend(point_cloud_firsts)
                pred_masks.extend(masks_pred)
                gt_masks.extend(gt_mask)
                sequence_ids_mask.extend([s["sequence_id"] for s in batch])
    class_labels = None
    print("Evaluating predictions")
    if save_sample:
        flow_obj = {
            "pred_flows": pred_flows,
            "gt_flows": gt_flows,
            "point_clouds": point_clouds,
            "sequence_ids_flow": sequence_ids_flow,
        }
        mask_obj = {
            "pred_masks": pred_masks,
            "gt_masks": gt_masks,
            "sequence_ids_mask": sequence_ids_mask,
            "point_clouds_mask": point_clouds_mask,
        }
        torch.save(flow_obj, Path(config.log.dir) / f"{type}_flow_sample.pth")
        torch.save(mask_obj, Path(config.log.dir) / f"{type}_mask_sample.pth")
    evaluate_predictions_general(
        pred_flows,
        gt_flows,
        pred_masks,
        gt_masks,
        class_labels,
        device,
        writer=writer,
        step=step,
        min_points=config.eval.min_points,
        type=type,
        config=config,
    )
    output_path = Path(config.log.dir) / "bucketed_epe" / f"step_{step}"
    if config.dataset.name in ["AV2_SceneFlowZoo"]:
        standard_results = compute_bucketed_epe_metrics(
            point_clouds=point_clouds,
            pred_flows=pred_flows,
            gt_flows=gt_flows,
            class_ids=class_ids,
            output_path=output_path,
        )
        for class_name, (static_epe, dynamic_error) in standard_results.items():
            writer.add_scalar(f"bucketed_epe/standard/{class_name}_static", static_epe, step)
            writer.add_scalar(f"bucketed_epe/standard/{class_name}_dynamic", dynamic_error, step)
