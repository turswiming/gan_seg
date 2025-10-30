"""
Main training script for scene flow and mask prediction.

This script handles the complete training pipeline including:
- Configuration loading and setup
- Dataset and model initialization
- Training loop with loss computation
- Visualization and logging
"""

# Standard library imports
import datetime
import argparse
import os
import shutil
import gc
import time
# Third party imports
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from config.config import print_config
from tqdm import tqdm
from pathlib import Path
from typing import List
import random
# Local imports
from eval import evaluate_predictions, eval_model, eval_model_general
from utils.config_utils import load_config_with_inheritance, save_config_and_code
from utils.dataloader_utils import create_dataloaders, infinite_dataloader  
from config.config import correct_datatype
from OGCModel.flownet_kitti import FlowStep3D
# Import refactored modules
from utils.model_utils import (
    setup_device_and_training, initialize_models_and_optimizers, 
    initialize_loss_functions, initialize_visualization,
    setup_checkpointing, load_checkpoint, create_checkpoint_saver
)
from utils.forward_utils import forward_scene_flow, forward_mask_prediction
from utils.training_utils import (
    determine_training_modes, set_model_training_modes,
    log_gradient_debug_info, perform_optimization_step,
    handle_checkpoint_saving, handle_evaluation_general,
    log_prediction_histograms, handle_visualization, cleanup_memory
)
from losses.loss_functions import compute_all_losses, compute_all_losses_general

from SceneFlowZoo.dataloaders import (
    TorchFullFrameInputSequence,
    TorchFullFrameOutputSequence,
    TorchFullFrameOutputSequenceWithDistance,
)
from OGCModel.icp_util import icp
#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def augment_transform(pc1, pc2, flow):
    #decentralize point cloud
    center_point = (pc1.mean(0)+pc2.mean(0))/2*torch.tensor([1,0,1]).to(pc1.device)
    pc1 = pc1 - center_point
    pc2 = pc2 - center_point
    #random rotation along y axis
    angle = torch.rand(1).item() * (np.pi/2) - np.pi/4  # uniform(-π/4, π/4)
    cos_a, sin_a = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
    rot = torch.tensor([[cos_a.item(), 0, sin_a.item()], 
                        [0, 1, 0], 
                        [-sin_a.item(), 0, cos_a.item()]], dtype=torch.float32)
    rot = rot.to(pc1.device)
    pc1 = pc1 @ rot.T
    pc2 = pc2 @ rot.T
    flow = flow @ rot.T
    
    #random translation
    translation = torch.rand(3).to(pc1.device) * 2 - 1  # uniform(-1, 1)
    translation[1]*=0
    translation = translation.to(pc1.device)
    pc1 = pc1 + translation
    pc2 = pc2 + translation
    
    #random scaling
    scale = torch.rand(1).item() * 0.1 + 0.95  # uniform(0.95, 1.05)
    pc1 = pc1 * scale
    pc2 = pc2 * scale
    flow = flow * scale
    
    #random mirror in x, z axis
    mirror_x = torch.rand(1).item()
    # mirror_z = torch.rand(1).item()
    if mirror_x < 0.5:
        pc1[:, 0] = -pc1[:, 0]
        pc2[:, 0] = -pc2[:, 0]
        flow[:, 0] = -flow[:, 0]
    # if mirror_z < 0.5:
    #     pc1[:, 2] = -pc1[:, 2]
    #     pc2[:, 2] = -pc2[:, 2]
    #     flow[:, 2] = -flow[:, 2]
    
    # Convert back to numpy for compatibility
    return pc1, pc2, flow

def create_dataloaders_general(config):
    """
    Create dataloaders for general training with AV2 SceneFlowZoo dataset.
    
    Args:
        config: Configuration object containing dataset parameters
        
    Returns:
        tuple: (dataloader, val_flow_dataloader, val_mask_dataloader)
            - dataloader: Main training dataloader
            - val_flow_dataloader: Validation dataloader for flow evaluation
            - val_mask_dataloader: Validation dataloader for mask evaluation
    """
    if config.dataset.name == "AV2_SceneFlowZoo":
        from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
        
        dataset = AV2SceneFlowZoo(
            root_dir=Path(config.dataset.AV2_SceneFlowZoo.root_dir),
            subsequence_length=config.dataset.AV2_SceneFlowZoo.subsequence_length,
            sliding_window_step_size=config.dataset.AV2_SceneFlowZoo.sliding_window_step_size,
            with_ground=config.dataset.AV2_SceneFlowZoo.with_ground,
            use_gt_flow=config.dataset.AV2_SceneFlowZoo.use_gt_flow,
            eval_type=config.dataset.AV2_SceneFlowZoo.eval_type,
            expected_camera_shape=config.dataset.AV2_SceneFlowZoo.expected_camera_shape,
            eval_args=dict(),
            with_rgb=config.dataset.AV2_SceneFlowZoo.with_rgb,
            flow_data_path=Path(config.dataset.AV2_SceneFlowZoo.flow_data_path),
            range_crop_type="ego",
            downsample_factor=config.dataset.AV2_SceneFlowZoo.downsample_factor,
        )
    elif config.dataset.name == "KITTISF_new":
        from dataset.kittisf_sceneflow import KittisfSceneFlowDataset
        dataset = KittisfSceneFlowDataset(
            data_root=config.dataset.KITTISF_new.data_root,
            split="train",
            num_points=config.dataset.KITTISF_new.num_points,
            seed=config.dataset.KITTISF_new.seed,
            augmentation=False
        )

    if config.dataset.val_name == "AV2_SceneFlowZoo_val":
        from dataset.av2_sceneflow_zoo import AV2SceneFlowZoo
        val_flow_config = config.dataset.AV2_SceneFlowZoo_val_flow
        val_flow_dataset = AV2SceneFlowZoo(
            root_dir=Path(val_flow_config.root_dir),
            with_ground=val_flow_config.with_ground,
            use_gt_flow=val_flow_config.use_gt_flow,
            eval_type=val_flow_config.eval_type,
            expected_camera_shape=val_flow_config.expected_camera_shape,
            eval_args=dict(output_path=val_flow_config.eval_args_output_path),
            with_rgb=val_flow_config.with_rgb,
            flow_data_path=Path(val_flow_config.flow_data_path),
            range_crop_type="ego",
            load_flow=True,
            load_boxes=False,
            downsample_factor=config.dataset.AV2_SceneFlowZoo.downsample_factor,
        )
        val_mask_config = config.dataset.AV2_SceneFlowZoo_val_mask
        val_mask_dataset = AV2SceneFlowZoo(
            root_dir=Path(val_mask_config.root_dir),
            with_ground=val_mask_config.with_ground,
            use_gt_flow=val_mask_config.use_gt_flow,
            eval_type=val_mask_config.eval_type,
            expected_camera_shape=val_mask_config.expected_camera_shape,
            eval_args=dict(output_path=val_mask_config.eval_args_output_path),
            with_rgb=val_mask_config.with_rgb,
            flow_data_path=Path(val_mask_config.flow_data_path),
            range_crop_type="ego",
            load_flow=False,
            load_boxes=True,
            downsample_factor=config.dataset.AV2_SceneFlowZoo.downsample_factor,
        )
    elif config.dataset.val_name == "KITTISF_new":
        from dataset.kittisf_sceneflow import KittisfSceneFlowDataset
        val_flow_dataset = KittisfSceneFlowDataset(
            data_root=config.dataset.KITTISF_new.data_root,
            split="val",
            num_points=config.dataset.KITTISF_new.num_points,
            seed=config.dataset.KITTISF_new.seed,
            augmentation=False
        )
        val_mask_dataset = KittisfSceneFlowDataset(
            data_root=config.dataset.KITTISF_new.data_root,
            split="val",
            num_points=config.dataset.KITTISF_new.num_points,
            seed=config.dataset.KITTISF_new.seed,
            augmentation=False
        )
    collect_fn = lambda x: x
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.dataloader.batchsize, 
        shuffle=True, 
        num_workers=config.dataloader.num_workers,
        collate_fn=collect_fn
    )
    val_flow_dataloader = torch.utils.data.DataLoader(
        val_flow_dataset, 
        batch_size=10, 
        shuffle=False, 
        num_workers=config.dataloader.num_workers,
        collate_fn=collect_fn
    )
    val_mask_dataloader = torch.utils.data.DataLoader(
        val_mask_dataset, 
        batch_size=10, 
        shuffle=False, 
        num_workers=config.dataloader.num_workers,
        collate_fn=collect_fn
    )
    return (dataset, dataloader, 
            val_flow_dataset, val_flow_dataloader, 
            val_mask_dataset, val_mask_dataloader)
    pass
def forward_scene_flow_general(point_cloud_firsts, point_cloud_nexts, flow_predictor,dataset_name, train_flow=False):
    if isinstance(flow_predictor, FlowStep3D):

        # if dataset_name == "KITTISF_new":
        #     ground_masks = [pc[:,1] < -1.4 for pc in point_cloud_firsts]
        #     original_point_cloud_firsts = point_cloud_firsts.copy()
        #     original_point_cloud_nexts = point_cloud_nexts.copy()
        #     point_cloud_firsts = [pc[~ground_mask] for pc,ground_mask in zip(point_cloud_firsts,ground_masks)]
        #     point_cloud_nexts = [pc[~ground_mask] for pc,ground_mask in zip(point_cloud_nexts,ground_masks)]
        point_cloud_firsts = [pc.unsqueeze(0) for pc in point_cloud_firsts]
        point_cloud_nexts = [pc.unsqueeze(0) for pc in point_cloud_nexts]
        #if all point_cloud_firsts and point_cloud_nexts are the same, then only forward once
        if all(pc1.shape == pc2.shape for pc1, pc2 in zip(point_cloud_firsts, point_cloud_nexts)) \
            and len(set([pc.shape[1] for pc in point_cloud_firsts])) == 1 and False:
            point_cloud_firsts = torch.cat(point_cloud_firsts, dim=0)
            point_cloud_nexts = torch.cat(point_cloud_nexts, dim=0)
            casecde_flow_outs = flow_predictor(point_cloud_firsts, point_cloud_nexts,point_cloud_firsts, point_cloud_nexts, iters=5)
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
            for pc1,pc2 in zip(point_cloud_firsts, point_cloud_nexts):
                from dataset.kittisf_sceneflow import get_global_transform_matrix
                pc1 = pc1.squeeze(0)
                pc2 = pc2.squeeze(0)
                pc1_ground_mask = pc1[:,1] < -1.4
                pc2_ground_mask = pc2[:,1] < -1.4
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
            flow_preds = flow_predictor(
                pc1_input, 
                pc2_input,
                pc1_input, 
                pc2_input, iters=5)
            flow_prd_lsq = flow_preds[-1]
            flow_outs = []
            for flow_pred_org, flow_prd_lsq_item, pc1_ground_mask in \
             zip(flow_pred_org_list, flow_prd_lsq, pc1_ground_mask_list):

                flow_pred_res = flow_pred_org.clone()
                flow_pred_res[~pc1_ground_mask] += flow_prd_lsq_item[~pc1_ground_mask]
                flow_outs.append(flow_pred_res)
            casecde_flow_outs = []
            for i, flow_pred_step in enumerate(flow_preds):
                flow_pred_step_res = []
                for flow_pred_org, flow_prd_lsq_item, pc1_ground_mask in \
                 zip(flow_pred_org_list, flow_pred_step, pc1_ground_mask_list):

                        flow_pred_res = flow_pred_org.clone()
                        flow_pred_res[~pc1_ground_mask] += flow_prd_lsq_item[~pc1_ground_mask]
                        flow_pred_step_res.append(flow_pred_res)
                casecde_flow_outs.append(flow_pred_step_res)
                pass

        if train_flow:
            return flow_outs, casecde_flow_outs[:-1]
        return flow_outs
    elif isinstance(flow_predictor, FastFlow3D):
        first_masks = [torch.ones(pc.shape[0],device=pc.device).bool() for pc in point_cloud_firsts]
        next_masks = [torch.ones(pc.shape[0],device=pc.device).bool() for pc in point_cloud_nexts]
        first_inputs = [[pc,mask] for pc,mask in zip(point_cloud_firsts,first_masks)]
        next_inputs = [[pc,mask] for pc,mask in zip(point_cloud_nexts,next_masks)]
        transform_inputs = [[torch.eye(4,device=pc.device),torch.eye(4,device=pc.device)] for pc in point_cloud_firsts]
        flow_out_seqs = flow_predictor._model_forward(first_inputs, next_inputs, transform_inputs)
        flow = [item.ego_flows.squeeze(0) for item in flow_out_seqs]
        return flow
    else:
        raise ValueError(f"Flow predictor {flow_predictor.name} not supported")

def forward_mask_prediction_general(pc_tensors, mask_predictor):
    pred_masks = []
    if len(set([pc.shape[1] for pc in pc_tensors])) == 1:
        pc_tensors = torch.cat([pc.unsqueeze(0) for pc in pc_tensors],dim=0)
        pred_mask = mask_predictor.forward(pc_tensors,pc_tensors)
        for i in range(pred_mask.shape[0]):
            pred_masks.append(pred_mask[i].permute(1,0))
        return pred_masks
    for pc_tensor in pc_tensors:
        pc_tensor = pc_tensor.unsqueeze(0).contiguous()
        pred_mask = mask_predictor.forward(pc_tensor,pc_tensor)
        pred_mask = pred_mask.squeeze(0)
        pred_mask = pred_mask.permute(1,0)
        pred_masks.append(pred_mask)
    return pred_masks

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def inference_models(flow_predictor,mask_predictor, sample,dataset_name, train_flow=False,downsample=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    point_cloud_firsts = [s["point_cloud_first"].to(device).float() for s in sample]
    point_cloud_nexts = [s["point_cloud_next"].to(device).float() for s in sample]
    return_value = forward_scene_flow_general(point_cloud_firsts, point_cloud_nexts, flow_predictor,dataset_name,train_flow=train_flow)
    if train_flow:
        flow,cascade_flow_outs = return_value
    else:
        flow = return_value
        cascade_flow_outs = None
    if downsample is not None:
        point_cloud_firsts = [item[::downsample,:] for item in point_cloud_firsts]
        point_cloud_nexts = [item[::downsample,:] for item in point_cloud_nexts]
        flow = [item[::downsample,:] for item in flow]
        if cascade_flow_outs is not None:
            for i in range(len(cascade_flow_outs)):
                cascade_flow_outs[i] = [item[::downsample,:] for item in cascade_flow_outs[i]]
    #augment transform
    for i in range(len(point_cloud_firsts)):
        point_cloud_firsts[i], point_cloud_nexts[i], flow[i] = augment_transform(point_cloud_firsts[i], point_cloud_nexts[i], flow[i])
    pred_masks = forward_mask_prediction_general(point_cloud_firsts, mask_predictor)
    return flow,pred_masks,point_cloud_firsts,point_cloud_nexts,cascade_flow_outs

def downsample_point_clouds(pred_flow,pred_mask,point_cloud_firsts,point_cloud_nexts,downsample_factor):
    if pred_flow is not None:
        pred_flow = [item[::downsample_factor,:] for item in pred_flow]
    if pred_mask is not None:
        pred_mask = [item[:,::downsample_factor] for item in pred_mask]
    if point_cloud_firsts is not None:
        point_cloud_firsts = [item[::downsample_factor,:] for item in point_cloud_firsts]
    if point_cloud_nexts is not None:
        point_cloud_nexts = [item[::downsample_factor,:] for item in point_cloud_nexts]
    return pred_flow,pred_mask,point_cloud_firsts,point_cloud_nexts

def main(config, writer):
    """
    Main training function.
    
    Args:
        config: Configuration object containing all training parameters
        writer: TensorBoard SummaryWriter for logging
    """
    # Setup device and basic configurations
    device = setup_device_and_training()
    set_seed(config.seed)
    # Create dataloaders
    (dataset, dataloader,
    val_flow_dataset, val_flow_dataloader,
    val_mask_dataset, val_mask_dataloader) = create_dataloaders_general(config)

    # # Initialize models, optimizers and schedulers
    (mask_predictor, flow_predictor, optimizer_flow, optimizer_mask, 
     alter_scheduler, scene_flow_smoothness_scheduler, mask_scheduler) = initialize_models_and_optimizers(config,None,device)
    #infinite dataloader
    inf_dataloader = infinite_dataloader(dataloader)
    # # Initialize loss functions
    loss_functions = initialize_loss_functions(config, device)
    
    # # Initialize visualization
    # if config.vis.show_window:
    #     vis, pcd, gt_pcd, reconstructed_pcd = initialize_visualization(config)
    
    # # Setup checkpointing
    checkpoint_dir, save_every_iters, step, resume, resume_path = setup_checkpointing(config, device)
    
    # # Load checkpoint if resuming
    step = load_checkpoint(config, flow_predictor, 
                          mask_predictor, optimizer_flow, optimizer_mask, alter_scheduler, mask_scheduler)
    
    # Create checkpoint saver
    save_checkpoint = create_checkpoint_saver(checkpoint_dir, flow_predictor, mask_predictor,
                                            optimizer_flow, optimizer_mask, alter_scheduler, config, mask_scheduler)
    
    first_iteration = True
    loss_dict_move_average = []
    # Main training loop
    with tqdm(inf_dataloader, desc="Training", total=config.training.max_iter-step) as inf_dataloader:
        tqdm.write("Starting training...")
        for sample in inf_dataloader:
            step += 1
            
            handle_evaluation_general(
                config=config, 
                step=step, 
                flow_predictor=flow_predictor, 
                mask_predictor=mask_predictor, 
                val_flow_dataloader=val_flow_dataloader, 
                val_mask_dataloader=val_mask_dataloader, 
                train_dataloader=dataloader,
                device=device, writer=writer)


            if step > config.training.max_iter:
                tqdm.write("Reached maximum training iterations, stopping.")
                break
            # Determine training modes and set model states
            train_flow, train_mask = determine_training_modes(step, config, alter_scheduler)
            set_model_training_modes(flow_predictor, mask_predictor, train_flow, train_mask)  
            flow_predictor.to(device)
            mask_predictor.to(device)
            
            try:
                (pred_flow,
                pred_mask,
                point_cloud_firsts,
                point_cloud_nexts,
                cascade_flow_outs) = inference_models(
                    flow_predictor,
                    mask_predictor,
                    sample,
                    config.dataset.name, 
                    train_flow=train_flow,downsample=config.training.mask_downsample_factor)
            except Exception as e:
                print(e)
                import traceback
                print(traceback.format_exc())
                continue

            # Compute all losses
            loss_dict, total_loss, reconstructed_points = compute_all_losses_general(
                config=config, loss_functions=loss_functions, flow_predictor=flow_predictor, mask_predictor=mask_predictor,
                point_cloud_firsts=point_cloud_firsts, point_cloud_nexts=point_cloud_nexts, pred_flow=pred_flow, pred_mask=pred_mask, step=step, scene_flow_smoothness_scheduler=scene_flow_smoothness_scheduler,
                train_flow=train_flow, train_mask=train_mask, device=device,cascade_flow_outs=cascade_flow_outs)

            # Log to tensorboard
            loss_dict_move_average.append(loss_dict)
            if step % config.log.tensorboard_log_interval == 0:
                loss_mean_dict = {name: np.mean([loss_dict[name].cpu().item() for loss_dict in loss_dict_move_average]) for name in loss_dict_move_average[0].keys()}
                loss_mean_dict["total_loss"] = np.sum([loss_mean_dict[key] for key in loss_mean_dict.keys()])
                writer.add_scalars("losses", loss_mean_dict, step)
                loss_dict_move_average = []
            
            # Log gradient debugging information
            log_gradient_debug_info(config, writer, loss_dict, flow_predictor, mask_predictor, step)
            
            # Perform optimization step
            optimization_success = perform_optimization_step(
                config, total_loss, optimizer_flow, optimizer_mask,
                flow_predictor, mask_predictor, train_flow, train_mask)
            
            if not optimization_success:
                continue


            alter_scheduler.step()
            # Step mask scheduler if available
            if mask_scheduler is not None:
                mask_scheduler.step()
            
            # Handle checkpoint saving
            handle_checkpoint_saving(save_every_iters, step, checkpoint_dir, save_checkpoint)
            
            # Handle evaluation
            
            # Clear memory cache
            if step % config.hardware.clear_cache_interval == 0:
                cleanup_memory()
            
            # Log prediction histograms
            log_prediction_histograms(config, writer, pred_flow, pred_mask, step)
            
            # Handle visualization
            # first_iteration, lineset, lineset_gt = handle_visualization(
            #     config, vis, pcd, gt_pcd, point_cloud_firsts, point_cloud_nexts,
            #     pred_flow, pred_mask, sample, first_iteration)
            
            pass  # end loop 

    # Cleanup
    if config.vis.show_window:
        vis.destroy_window()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scene Flow and Mask Prediction")
    parser.add_argument("--config", type=str, default="config/baseconfig.yaml", 
                       help="Path to the config file")
    
    args, unknown = parser.parse_known_args()
    config_obj = load_config_with_inheritance(args.config)

    # print_config(config_obj)
    
    cli_opts = OmegaConf.from_cli()
    
    # Merge configs
    config = OmegaConf.merge(config_obj, cli_opts)
    config = correct_datatype(config)
    print_config(config)

    # Setup logging directory
    if config.log.dir == "":
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.log.dir = f"../outputs/exp/{time_str}"
    
    writer = SummaryWriter(log_dir=config.log.dir)
    
    # Save config and code
    save_config_and_code(config, config.log.dir)

    # Start training
    main(config, writer)