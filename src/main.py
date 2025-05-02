import torch
from torch import nn
from torch.nn import functional as F
import open3d as o3d
from per_scene_dataset import PerSceneDataset
from scene_flow_predict_model import SceneFlowPredictor
from mask_predict_model import MaskPredictor
from gan_loss import GanLoss
dataset = PerSceneDataset()

def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PerSceneDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
infinite_loader = infinite_dataloader(dataloader)
sample = next(infinite_loader)
_, N, _ = sample["point_cloud_first"].shape
scene_flow_predictor = SceneFlowPredictor(hidden_dim=128,layer_num=8)
mask_predictor = MaskPredictor(slot_num=8, point_length=N)
optimizer = torch.optim.AdamW(scene_flow_predictor.parameters(), lr=0.001)
optimizer_mask = torch.optim.AdamW(mask_predictor.parameters(), lr=0.1)
criterion = nn.MSELoss()
gan_loss = GanLoss(device=device)
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
gt_pcd = o3d.geometry.PointCloud()
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
    pred_flow = scene_flow_predictor(sample)

    gt_flow = sample["point_cloud_second"] - sample["point_cloud_first"]
    gt_flow = gt_flow.to(pred_flow.device)
    pred_mask = mask_predictor(sample)
    loss = gan_loss(sample, pred_mask, pred_flow)
    print("loss", loss.item())
    optimizer.zero_grad()
    optimizer_mask.zero_grad()
    pred_flow.retain_grad()
    pred_mask.retain_grad()
    loss.backward()
    print("pred_flow.grad", pred_flow.grad.std())
    print("pred_mask.grad", pred_mask.grad.std())
    optimizer.step()
    optimizer_mask.step()
    pred_point = (sample["point_cloud_first"] + pred_flow.cpu().detach()).numpy()
    pcd.points = o3d.utility.Vector3dVector(pred_point.reshape(-1, 3))
    pcd.paint_uniform_color([1, 0, 0])
    gt_pcd.points = o3d.utility.Vector3dVector(sample["point_cloud_second"].cpu().detach().numpy().reshape(-1, 3))
    gt_pcd.paint_uniform_color([0, 1, 0])
    if first_iteration:
        vis.add_geometry(pcd)
        vis.add_geometry(gt_pcd)
        first_iteration = False
    else:
        vis.update_geometry(pcd)
        vis.update_geometry(gt_pcd)
    vis.poll_events()
    vis.update_renderer()

# 关闭窗口
vis.destroy_window()