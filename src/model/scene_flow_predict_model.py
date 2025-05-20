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
        self.input_layer = nn.Linear(self.input_dim, hidden_dim,device=self.device)
        self.activation_input = nn.ReLU()
        for i in range(layer_num):
            setattr(self, f"linear{i}", nn.Linear(hidden_dim, hidden_dim,device=self.device))
            setattr(self, f"relu{i}", nn.ReLU())
        self.output_layer = nn.Linear(hidden_dim, self.output_dim,device=self.device)
        self.activation_output = nn.ReLU()

    def forward(self, inputs):
        point_cloud = inputs.to(self.device)
        #connver to double
        point_cloud = point_cloud.view(-1, self.input_dim)
        point_cloud = self.input_layer(point_cloud)
        point_cloud = self.activation_input(point_cloud)
        for i in range(self.layer_num):
            linear = getattr(self, f"linear{i}")
            tanh = getattr(self, f"relu{i}")
            point_cloud_linear = linear(point_cloud)
            point_cloud = tanh(point_cloud_linear) + point_cloud
        point_cloud = point_cloud.view(-1, self.hidden_dim)
        point_cloud = self.output_layer(point_cloud)
        point_cloud = self.activation_output(point_cloud)
        point_cloud = point_cloud.view(-1, self.output_dim)
        flow = point_cloud
        return flow
    
    
class Neural_Prior(torch.nn.Module):
    def __init__(self, dim_x=3, filter_size=128, act_fn='relu', layer_size=8):
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
            self.nn_layers.append(torch.nn.Linear(filter_size, dim_x))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(dim_x, dim_x)))

    def forward(self, x):
        """ points -> features
            [B, N, 3] -> [B, N, 3]
        """
        for layer in self.nn_layers:
            x = layer(x)
                
        return x
    
class OptimizedFLowPredictor(torch.nn.Module):
    def __init__(self, dim=3, pointSize=128):
        super().__init__()
        self.pointSize = pointSize
        self.dim = dim
        init_noise = torch.randn((pointSize, dim))*0.1
        self.init_noise = torch.nn.Parameter(init_noise, requires_grad=True)
    
    def forward(self, x):
        """ points -> features
            [N, 3] -> [N, 3]
        """
        return self.init_noise