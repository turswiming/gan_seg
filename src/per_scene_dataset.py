

import torch
from torch import nn
from torch.nn import functional as F
from gen_point_traj_flow import process_one_sample
class PerSceneDataset(nn.Module):
    def __init__(self):
        super(PerSceneDataset, self).__init__()
        self.traj = None
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        if self.traj is None:
            metadata_path = "/home/lzq/workspace/gan_seg/dataset/0/metadata.json"
            dep_img_path = "/home/lzq/workspace/gan_seg/dataset/0/depth_00000.tiff"
            seg_img_path = "/home/lzq/workspace/gan_seg/dataset/0/segmentation_00000.png"
            self.traj =  process_one_sample(metadata_path, dep_img_path, seg_img_path,f=0)
            #randomly select 1024 points
            movement = self.traj[1]-self.traj[0]
            movement = torch.tensor(movement)
            movement_mask = torch.norm(movement, dim=1) > 0.01
            self.point_cloud_first = self.traj[0][movement_mask]
            self.point_cloud_second = self.traj[1][movement_mask]
        sample = {}
        sample["point_cloud_first"] = self.point_cloud_first
        sample["point_cloud_second"] = self.point_cloud_second
        return sample