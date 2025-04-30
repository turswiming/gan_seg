

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

        sample = {}
        sample["point_cloud_first"] = self.traj[0]
        sample["point_cloud_second"] = self.traj[1]
        return sample