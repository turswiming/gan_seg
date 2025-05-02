import torch
from torch import nn
from torch.nn import functional as F

class SceneFlowPredictor(nn.Module):
    def __init__(self,hidden_dim=256, layer_num=2):
        super(SceneFlowPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 3
        self.output_dim = 3
        self.input_layer = nn.Linear(self.input_dim, hidden_dim,device=self.device,dtype=torch.float64)
        self.activation_input = nn.ReLU()
        for i in range(layer_num):
            setattr(self, f"linear{i}", nn.Linear(hidden_dim, hidden_dim,device=self.device,dtype=torch.float64))
            setattr(self, f"tanh{i}", nn.ReLU())
        self.output_layer = nn.Linear(hidden_dim, self.output_dim,device=self.device,dtype=torch.float64)
        self.activation_output = nn.ReLU()

    def forward(self, inputs):
        point_cloud = inputs["point_cloud_first"].to(self.device)
        #connver to double
        point_cloud = point_cloud.view(-1, self.input_dim)
        point_cloud = self.input_layer(point_cloud)
        point_cloud = self.activation_input(point_cloud)
        for i in range(self.layer_num):
            linear = getattr(self, f"linear{i}")
            tanh = getattr(self, f"tanh{i}")
            point_cloud_linear = linear(point_cloud)
            point_cloud = tanh(point_cloud_linear)
        point_cloud = point_cloud.view(-1, self.hidden_dim)
        point_cloud = self.output_layer(point_cloud)
        point_cloud = self.activation_output(point_cloud)
        point_cloud = point_cloud.view(-1, self.output_dim)
        flow = point_cloud
        return flow