from utils.metrics import calculate_miou, calculate_epe
import torch
import torch.nn.functional as F
from utils.visualization_utils import remap_instance_labels, create_label_colormap, color_mask
from torch.utils.tensorboard import SummaryWriter
def evaluate_predictions(pred_flows, gt_flows, pred_masks, gt_masks, device, writer, step):
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
    writer.add_scalar("miou", miou_mean.item(), step)
    
    return epe_mean, miou_mean