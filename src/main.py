import torch
from torch import nn
from torch.nn import functional as F
import open3d as o3d
from per_scene_dataset import PerSceneDataset
from scene_flow_predict_model import SceneFlowPredictor
from mask_predict_model import MaskPredictor
dataset = PerSceneDataset()

def infinite_dataloader(dataloader):
    while True:
        for batch in dataloader:
            yield batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = PerSceneDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
infinite_loader = infinite_dataloader(dataloader)
scene_flow_predictor = SceneFlowPredictor(hidden_dim=128,layer_num=8)
mask_predictor = MaskPredictor(slot_num=8, point_length=65536)
optimizer = torch.optim.AdamW(scene_flow_predictor.parameters()+mask_predictor.parameters(), lr=0.001)
criterion = nn.MSELoss()

vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
gt_pcd = o3d.geometry.PointCloud()
first_iteration = True

for sample in infinite_loader:
    pred = scene_flow_predictor(sample)

    gt_flow = sample["point_cloud_second"] - sample["point_cloud_first"]
    gt_flow = gt_flow.to(pred.device)
    loss = criterion(pred, gt_flow)
    print("loss", loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    pred_point = (sample["point_cloud_first"] + pred.cpu().detach()).numpy()
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