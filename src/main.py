#standard library
import datetime
import argparse
import os
import shutil

#third party library
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from omegaconf import OmegaConf
from config.config import print_config
from tqdm import tqdm
#local library
import open3d as o3d
from dataset.av2_dataset import AV2Dataset
from dataset.per_scene_dataset import PerSceneDataset

from model.scene_flow_predict_model import FLowPredictor, Neural_Prior ,SceneFlowPredictor
from model.mask_predict_model import MaskPredictor

from losses.ChamferDistanceLoss import ChamferDistanceLoss
from losses.ReconstructionLoss import ReconstructionLoss
from losses.PointSmoothLoss import PointSmoothLoss
from losses.FlowSmoothLoss import FlowSmoothLoss

from visualize.open3d_func import visualize_vectors, update_vector_visualization
from visualize.pca import pca
from Predictor import get_mask_predictor,get_scene_flow_predictor

#import tensorboard


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def main(config ,writer):
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.dataset.name == "AV2":
        dataset = AV2Dataset()
    elif config.dataset.name == "MOVI_F":
        dataset = PerSceneDataset()
    else:
        raise ValueError("Dataset not supported")
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.dataloader.batchsize, 
        shuffle=True,
        collate_fn=lambda batch: {
            "point_cloud_first": torch.stack([item["point_cloud_first"] for item in batch]),
            "point_cloud_second": torch.stack([item["point_cloud_second"] for item in batch]),
            "flow": torch.stack([item["flow"] for item in batch])
        }
    )
    infinite_loader = infinite_dataloader(dataloader)
    sample = next(infinite_loader)
    batch_size, N, _ = sample["point_cloud_first"].shape
    mask_predictor = get_mask_predictor(config.model.mask,N)
    scene_flow_predictor = get_scene_flow_predictor(config.model.flow,N)
    scene_flow_predictor.to(device)

    optimizer = torch.optim.AdamW(scene_flow_predictor.parameters(), lr=config.model.flow.lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=config.model.mask.lr)

    reconstructionLoss = ReconstructionLoss(device)
    chamferLoss = ChamferDistanceLoss()
    flowSmoothLoss = FlowSmoothLoss(device)
    pointsmoothloss = PointSmoothLoss()
    flowRecLoss = nn.MSELoss()
    if config.vis.show_window:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        pcd = o3d.geometry.PointCloud()
        gt_pcd = o3d.geometry.PointCloud()
        reconstructed_pcd = o3d.geometry.PointCloud()
    first_iteration = True
    step = 0
    epe = None
    for sample in tqdm(infinite_loader, total=config.training.max_iter):
        if epe is not None:
            tqdm.write(f"epe: {epe.mean().item():.4f}")
        step += 1
        if step > config.training.max_iter:
            break
        # if (step // loop_step)%2 == 0:
        #     train_flow_model = True
        #     train_mask_model = False
        # else:
        #     train_flow_model = False
        #     train_mask_model = True
        # if train_flow_model:
        #     scene_flow_predictor.train()
        #     mask_predictor.eval()
        # else:
        #     scene_flow_predictor.eval()
        #     mask_predictor.train()
        
        # Get predictions with batch dimension
        pred_flow = scene_flow_predictor(sample["point_cloud_first"].to(device))  # Shape: [B, N, 3]
        gt_flow = sample["flow"].to(device)  # Shape: [B, N, 3]
        pred_mask = mask_predictor(sample)  # Shape: [B, K, N]

        #compute losses
        if config.lr_multi.rec_loss>0 or config.lr_multi.rec_flow_loss>0:
            rec_loss, reconstructed_points = reconstructionLoss(sample, pred_mask, pred_flow)
            rec_loss = rec_loss * config.lr_multi.rec_loss
        else:
            rec_loss = torch.tensor(0.0, device=device)
        if config.lr_multi.scene_flow_smoothness>0:
            scene_flow_smooth_loss = flowSmoothLoss(sample, pred_mask, pred_flow)
            scene_flow_smooth_loss = scene_flow_smooth_loss * config.lr_multi.scene_flow_smoothness
        else:
            scene_flow_smooth_loss = torch.tensor(0.0, device=device)
        if config.lr_multi.rec_flow_loss>0:
            # Apply flow to get predicted second point cloud
            point_cloud_first = sample["point_cloud_first"].to(device)
            pred_second_points = point_cloud_first + pred_flow
            rec_flow_loss = flowRecLoss(pred_second_points, reconstructed_points)
            rec_flow_loss = rec_flow_loss * config.lr_multi.rec_flow_loss
        else:
            rec_flow_loss = torch.tensor(0.0, device=device)
        if config.lr_multi.flow_loss>0:
            # Apply flow to get predicted second point cloud
            point_cloud_first = sample["point_cloud_first"].to(device)
            pred_second_points = point_cloud_first + pred_flow
            flow_loss = chamferLoss(pred_second_points, sample["point_cloud_second"].to(device))
            flow_loss = flow_loss * config.lr_multi.flow_loss
        else:
            flow_loss = torch.tensor(0.0, device=device)
        if config.lr_multi.point_smooth_loss>0:
            point_smooth_loss = pointsmoothloss(sample["point_cloud_first"].to(device), pred_mask)
            point_smooth_loss = point_smooth_loss * config.lr_multi.point_smooth_loss
        else:
            point_smooth_loss = torch.tensor(0.0, device=device)

        #add all losses together
        loss = rec_loss + flow_loss + scene_flow_smooth_loss + rec_flow_loss + point_smooth_loss
        print("rec_loss", rec_loss.item())
        print("flow_loss", flow_loss.item())
        print("scene_flow_smooth_loss", scene_flow_smooth_loss.item())
        print("rec_flow_loss", rec_flow_loss.item())
        print("point_smooth_loss", point_smooth_loss.item())
        print("iteration", step)
        #add all losses to tensorboard in one chart
        writer.add_scalars("losses", {
            "rec_loss": rec_loss.item(),
            "flow_loss": flow_loss.item(),
            "scene_flow_smooth_loss": scene_flow_smooth_loss.item(),
            "rec_flow_loss": rec_flow_loss.item(),
            "point_smooth_loss": point_smooth_loss.item(),
            "total_loss": loss.item(),
        }, step)
        
        optimizer.zero_grad()
        optimizer_mask.zero_grad()
        loss.backward()
        
        # Log gradients if needed
        if hasattr(pred_flow, 'grad') and pred_flow.grad is not None:
            print("pred_flow.grad", pred_flow.grad.std().item())
        if hasattr(pred_mask, 'grad') and pred_mask.grad is not None:
            print("pred_mask.grad", pred_mask.grad.std().item())
            
        optimizer.step()
        optimizer_mask.step()
        
        #compare with gt_flow
        epe = torch.norm(pred_flow - gt_flow, dim=2, p=2)  # Shape: [B, N]
        epe_mean = epe.mean()
        print("epe", epe_mean.item())

        #PCA to 3D
        writer.add_scalar("epe", epe_mean.item(), step)

        #visualize
        if config.vis.show_window and sample["point_cloud_first"].shape[0] > 0:
            # Use the first batch item for visualization
            batch_idx = 0
            
            # Get point clouds and predictions for visualization
            point_cloud_first = sample["point_cloud_first"][batch_idx].cpu().numpy()
            point_cloud_second = sample["point_cloud_second"][batch_idx].cpu().numpy()
            current_pred_flow = pred_flow[batch_idx].cpu().detach().numpy()
            current_pred_mask = pred_mask[batch_idx].cpu().detach()
            
            # PCA for coloring
            current_pred_mask = current_pred_mask.permute(1, 0)  # Change to [N, K] for PCA
            color = pca(current_pred_mask)
            
            # Predicted point cloud
            pred_point = point_cloud_first + current_pred_flow
            pcd.points = o3d.utility.Vector3dVector(pred_point)
            pcd.colors = o3d.utility.Vector3dVector(color.numpy())
            
            # Ground truth point cloud
            gt_pcd.points = o3d.utility.Vector3dVector(point_cloud_second)
            gt_pcd.paint_uniform_color([0, 1, 0])
            
            # Reconstructed point cloud if available
            if "reconstructed_points" in locals():
                current_reconstructed = reconstructed_points[batch_idx].cpu().detach().numpy()
                reconstructed_pcd.points = o3d.utility.Vector3dVector(current_reconstructed)
                reconstructed_pcd.paint_uniform_color([0, 0, 1])
                
            if first_iteration:
                # vis.add_geometry(pcd)
                vis.add_geometry(gt_pcd)
                if "reconstructed_points" in locals():
                    vis.add_geometry(reconstructed_pcd)
                vis, lineset = visualize_vectors(
                    point_cloud_first,
                    current_pred_flow,
                    vis=vis,
                    color=color.numpy(),
                    )
                first_iteration = False
            else:
                # vis.update_geometry(pcd)
                lineset = update_vector_visualization(
                    lineset,
                    point_cloud_first,
                    current_pred_flow,
                    color=color.numpy(),
                )
                vis.update_geometry(lineset)
                vis.update_geometry(gt_pcd)
                if "reconstructed_points" in locals():
                    vis.update_geometry(reconstructed_pcd)
            vis.poll_events()
            vis.update_renderer()

    # 关闭窗口
    if config.vis.show_window:
        vis.destroy_window()

def load_config_with_inheritance(config_path):
    config = OmegaConf.load(config_path)
    base_config_path = config_path
    while '__base__' in config:
        base_config_path = os.path.join(os.path.dirname(base_config_path), config.__base__)
        print(f"Loading base config from: {base_config_path}")
        base_config = OmegaConf.load(base_config_path)
        # 移除_base_字段避免干扰
        config.pop('__base__')
        config = OmegaConf.merge(base_config, config)
    return config

# 使用

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scene Flow and Mask Prediction")
    parser.add_argument("--config", type=str, default="config/baseconfig.yaml", help="Path to the config file")
    
    args, unknown = parser.parse_known_args()
    config_obj = load_config_with_inheritance(args.config)

    print_config(config_obj)
    cli_opts = OmegaConf.from_cli()
    print_config(cli_opts)
    # Merge the config with command line options
    config = OmegaConf.merge(config_obj, cli_opts)
    #init summary writer
    if config.log.dir =="":
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config.log.dir = f"../outputs/exp/{time_str}"
    
    writer = SummaryWriter(log_dir=config.log.dir)
    # Save config file to config.log.dir
    config_save_path = f"{config.log.dir}/config.yaml"
    with open(config_save_path, "w") as config_file:
        OmegaConf.save(config=config, f=config_file)
    print(f"Config file saved to {config_save_path}")

    # Save all .py files in the current folder to config.log.dir/code
    code_save_path = os.path.join(config.log.dir, "code")
    os.makedirs(code_save_path, exist_ok=True)
    current_folder = os.path.dirname(os.path.abspath(__file__))
    code_extensions = (".py",".cu",".cpp",".h")
    for root, dirs, files in os.walk(current_folder):
        for file_name in files:
            if file_name.endswith(code_extensions):
                full_file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(full_file_path, current_folder)
                destination_path = os.path.join(code_save_path, relative_path)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy(full_file_path, destination_path)

    print(f"All .py files including subdirectories saved to {code_save_path}")

    main(config, writer)