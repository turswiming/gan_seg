#standard library
import datetime

#third party library
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

#local library
import open3d as o3d
from dataset.per_scene_dataset import PerSceneDataset

from model.scene_flow_predict_model import FLowPredictor
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

#init summary writer
time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(log_dir=f"../outputs/exp/{time_str}")
dataset = PerSceneDataset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PerSceneDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
warmup_shaduler = lambda it : min(10+it*0.1,100)
infinite_loader = infinite_dataloader(dataloader)
sample = next(infinite_loader)
_, N, _ = sample["point_cloud_first"].shape
scene_flow_predictor = FLowPredictor(dim=3,pointSize=N)
scene_flow_predictor.to(device)
slot_num = 3
mask_predictor = MaskPredictor(slot_num=slot_num, point_length=N)
optimizer = torch.optim.AdamW(scene_flow_predictor.parameters(), lr=0.03, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=1, weight_decay=0.01)

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
    rec_loss, reconstructed_points = reconstructionLoss(sample, pred_mask, pred_flow)
    scene_flow_smooth_loss = flowSmoothLoss(sample, pred_mask, pred_flow)
    rec_flow_loss = flowRecLoss(pred_flow+sample["point_cloud_first"].to(device), reconstructed_points)
    flow_loss = chamferLoss(pred_flow+sample["point_cloud_first"].to(device), sample["point_cloud_second"].to(device))
    point_smooth_loss = pointsmoothloss(sample["point_cloud_first"].to(device).to(torch.float), pred_mask.permute(1,0).unsqueeze(0).to(torch.float))

    #add weights to losses
    rec_loss = rec_loss *0.1
    flow_loss = flow_loss * 1
    scene_flow_smooth_loss = scene_flow_smooth_loss * warmup_shaduler(step)
    rec_flow_loss = rec_flow_loss * 0.001
    point_smooth_loss = point_smooth_loss * 0.01

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
    reconstructed_pcd.points = o3d.utility.Vector3dVector(reconstructed_points.cpu().detach().numpy().reshape(-1, 3))
    reconstructed_pcd.paint_uniform_color([0, 0, 1])
    if first_iteration:
        # vis.add_geometry(pcd)
        vis.add_geometry(gt_pcd)
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
        vis.update_geometry(reconstructed_pcd)
    vis.poll_events()
    vis.update_renderer()

# 关闭窗口
vis.destroy_window()