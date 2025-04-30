import torch
from torch import nn
from torch.nn import functional as F

class MaskPredictor(nn.Module):
    def __init__(self, slot_num=1, point_length=65536):
        super(MaskPredictor, self).__init__()
        self.slot_num = slot_num
        self.point_length = point_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.randn((slot_num, point_length))
        softmaxed_tensor = F.softmax(tensor, dim=0)
        self.tensor3d = torch.nn.Parameter(softmaxed_tensor, requires_grad=True)
    
    def forward(self, inputs):
        return self.tensor3d