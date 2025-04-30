import torch
from torch import nn
from torch.nn import functional as F
class GanLoss():
    def __init__(self, device):
        self.device = device

    
    def forward(self, inputs,pred_mask, pred_flow):
        point_cloud_first = inputs["point_cloud_first"].to(self.device)
        point_cloud_second = inputs["point_cloud_second"].to(self.device)
        # print("point_cloud_first.shape",point_cloud_first.shape)
        # print("point_cloud_second.shape",point_cloud_second.shape)
        # print("self.tensor3d.shape",self.tensor3d.shape)
        loss = self.criterion(point_cloud_first, point_cloud_second)
        return loss