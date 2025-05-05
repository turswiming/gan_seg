

import torch
from torch import nn
from torch.nn import functional as F
from .gen_point_traj_flow import process_one_sample
class PerSceneDataset(nn.Module):
    def __init__(self):
        super(PerSceneDataset, self).__init__()
        self.traj = None
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        start = 0
        end  = 1
        if self.traj is None:

            metadata_path = "/home/lzq/workspace/gan_seg/dataset/0/metadata.json"
            dep_img_path = f"/home/lzq/workspace/gan_seg/dataset/0/depth_{start:05d}.tiff"
            seg_img_path = f"/home/lzq/workspace/gan_seg/dataset/0/segmentation_{start:05d}.png"
            self.traj =  process_one_sample(metadata_path, dep_img_path, seg_img_path,f=start)
            dep_img_path_2 = f"/home/lzq/workspace/gan_seg/dataset/0/depth_{end:05d}.tiff"
            seg_img_path_2 = f"/home/lzq/workspace/gan_seg/dataset/0/segmentation_{end:05d}.png"
            traj2 = process_one_sample(metadata_path, dep_img_path_2, seg_img_path_2,f=end)
            #randomly select 1024 points
            movement = self.traj[end]-self.traj[start]
            movement = torch.tensor(movement)
            self.movement_mask = torch.norm(movement, dim=1) > 0.01

            movement2 = traj2[end]-traj2[start]
            movement2 = torch.tensor(movement2)
            self.movement_mask2 = torch.norm(movement2, dim=1) > 0.01
            

            self.point_cloud_first = self.traj[start][self.movement_mask]
            self.point_cloud_second = traj2[end][self.movement_mask2]
        sample = {}
        sample["point_cloud_first"] = self.point_cloud_first
        sample["point_cloud_second"] = self.point_cloud_second
        sample['flow'] = (self.traj[end]-self.traj[start])[self.movement_mask]

        return sample