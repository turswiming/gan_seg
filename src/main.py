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

#local library
import open3d as o3d
from dataset.av2_dataset import AV2Dataset

from model.scene_flow_predict_model import FLowPredictor, Neural_Prior ,SceneFlowPredictor
from model.mask_predict_model import MaskPredictor

from losses.ChamferDistanceLoss import ChamferDistanceLoss
from losses.ReconstructionLoss import ReconstructionLoss
from losses.PointSmoothLoss import PointSmoothLoss
from losses.FlowSmoothLoss import FlowSmoothLoss

from visualize.open3d_func import visualize_vectors, update_vector_visualization
from visualize.pca import pca
#import tensorboard


def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch

def main(config ,writer):
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = AV2Dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    loop_step = 10
    # scene_flow_smooth_shaduler = lambda it : 10000
    infinite_loader = infinite_dataloader(dataloader)
    sample = next(infinite_loader)
    _, N, _ = sample["point_cloud_first"].shape
    scene_flow_predictor = Neural_Prior()
    # scene_flow_predictor = FLowPredictor(pointSize=N)
    # scene_flow_predictor = SceneFlowPredictor(layer_num=8)
    scene_flow_predictor.to(device)
    slot_num = 10
    mask_predictor = MaskPredictor(slot_num=slot_num, point_length=N)
    optimizer = torch.optim.Adam(scene_flow_predictor.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=1)

    reconstructionLoss = ReconstructionLoss(device)
    chamferLoss = ChamferDistanceLoss()
    flowSmoothLoss = FlowSmoothLoss(device)
    pointsmoothloss = PointSmoothLoss()
    flowRecLoss = nn.MSELoss()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    pcd = o3d.geometry.PointCloud()
    gt_pcd = o3d.geometry.PointCloud()
    reconstructed_pcd = o3d.geometry.PointCloud()
    first_iteration = True
    loop_step =10
    step = 0
    for sample in infinite_loader:
        step += 1
        if (step // loop_step)%2 == 0:
            train_flow_model = True
            train_mask_model = False
        else:
            train_flow_model = False
            train_mask_model = True
        if train_flow_model:
            scene_flow_predictor.train()
            mask_predictor.eval()
        else:
            scene_flow_predictor.eval()
            mask_predictor.train()
        pred_flow = scene_flow_predictor(sample["point_cloud_first"].to(device))
        pred_flow = pred_flow.view(-1, 3)
        gt_flow = torch.tensor(sample["flow"])
        gt_flow = gt_flow.to(pred_flow.device)
        pred_mask = mask_predictor(sample)

        #compute losses
        if config.lr_multi.rec_loss>0 and config.lr_multi.rec_flow_loss>0:
            rec_loss, reconstructed_points = reconstructionLoss(sample, pred_mask, pred_flow)
            rec_loss = rec_loss *config.lr_multi.rec_loss
        else:
            rec_loss = torch.tensor(0.0)
        if config.lr_multi.scene_flow_smoothness>0:
            scene_flow_smooth_loss = flowSmoothLoss(sample, pred_mask, pred_flow)
            scene_flow_smooth_loss = scene_flow_smooth_loss * config.lr_multi.scene_flow_smoothness
        else:
            scene_flow_smooth_loss = torch.tensor(0.0)
        if config.lr_multi.rec_flow_loss>0:
            rec_flow_loss = flowRecLoss(pred_flow+sample["point_cloud_first"].to(device), reconstructed_points)
            rec_flow_loss = rec_flow_loss * config.lr_multi.rec_flow_loss
        else:
            rec_flow_loss = torch.tensor(0.0)
        if config.lr_multi.flow_loss>0:
            flow_loss = chamferLoss(pred_flow+sample["point_cloud_first"].to(device), sample["point_cloud_second"].to(device))
            flow_loss = flow_loss * config.lr_multi.flow_loss
        else:
            flow_loss = torch.tensor(0.0)
        if config.lr_multi.point_smooth_loss>0:
            point_smooth_loss = pointsmoothloss(sample["point_cloud_first"].to(device).to(torch.float), pred_mask.permute(1,0).unsqueeze(0).to(torch.float))
            point_smooth_loss = point_smooth_loss * config.lr_multi.point_smooth_loss
        else:
            point_smooth_loss = torch.tensor(0.0)

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
        pred_flow.retain_grad()
        pred_mask.retain_grad()
        # sum_loss = mse_loss
        loss.backward()
        if pred_flow.grad is not None:
            print("pred_flow.grad", pred_flow.grad.std())
        if pred_mask.grad is not None:
            print("pred_mask.grad", pred_mask.grad.std())
        optimizer.step()
        optimizer_mask.step()
        #compare with gt_flow
        pred_flow = pred_flow.view(-1, 3)
        pred_flow = pred_flow.reshape(-1, 3)
        gt_flow = gt_flow.view(-1, 3)
        gt_flow = gt_flow.reshape(-1, 3)
        epe = torch.norm(pred_flow - gt_flow, dim=1,p=2)
        print("epe", epe.mean().item())
        pred_point = (sample["point_cloud_first"] + pred_flow.cpu().detach()).numpy()
        pcd.points = o3d.utility.Vector3dVector(pred_point.reshape(-1, 3))
        pred_mask = pred_mask.permute(1, 0)
        pred_mask = pred_mask.reshape(-1, slot_num)
        #PCA to 3D
        writer.add_scalar("epe", epe.mean().item(), step)

        #visualize
        color = pca(pred_mask)
        pcd.colors = o3d.utility.Vector3dVector(color.cpu().detach().numpy())
        gt_pcd.points = o3d.utility.Vector3dVector(sample["point_cloud_second"].cpu().detach().numpy().reshape(-1, 3))
        gt_pcd.paint_uniform_color([0, 1, 0])
        #if defined reconstructed_points:

        if "reconstructed_points" in locals():
            reconstructed_pcd.points = o3d.utility.Vector3dVector(reconstructed_points.cpu().detach().numpy().reshape(-1, 3))
            reconstructed_pcd.paint_uniform_color([0, 0, 1])
        if first_iteration:
            # vis.add_geometry(pcd)
            vis.add_geometry(gt_pcd)
            if "reconstructed_points" in locals():
                vis.add_geometry(reconstructed_pcd)
            vis , lineset = visualize_vectors(
                sample["point_cloud_first"].reshape(-1, 3),
                pred_flow.cpu().detach().numpy().reshape(-1, 3),
                vis=vis,
                color=color.cpu().detach().numpy().reshape(-1, 3),
                )
            first_iteration = False
        else:
            # vis.update_geometry(pcd)
            lineset = update_vector_visualization(
                lineset,
                sample["point_cloud_first"].reshape(-1, 3),
                pred_flow.cpu().detach().numpy().reshape(-1, 3),
                color=color.cpu().detach().numpy().reshape(-1, 3),
                
            )
            vis.update_geometry(lineset)
            vis.update_geometry(gt_pcd)
            if "reconstructed_points" in locals():
                vis.update_geometry(reconstructed_pcd)
        vis.poll_events()
        vis.update_renderer()

    # 关闭窗口
    vis.destroy_window()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Scene Flow and Mask Prediction")
    parser.add_argument("--config", type=str, default="config/baseconfig.yaml", help="Path to the config file")
    config_obj = OmegaConf.load(parser.parse_args().config)
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