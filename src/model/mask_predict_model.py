"""
Mask Prediction Model implementation for point cloud segmentation.

This module implements a learnable mask predictor that assigns points to different
slots (segments) in a point cloud. It uses a parameter-based approach where the
mask weights are directly optimized during training.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .eulerflow_raw_mlp import EulerFlowMLP, ActivationFn, QueryDirection


class ResidualBlock(nn.Module):
    """
    经典残差块设计，基于ResNet的启发
    Classic residual block design inspired by ResNet
    """
    def __init__(self, dim_in, dim_out, activation_fn=nn.ReLU()):
        super().__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.linear2 = nn.Linear(dim_out, dim_out)
        self.activation = activation_fn
        
        # 捷径连接：如果输入输出维度不同，需要线性投影
        # Shortcut connection: linear projection if input/output dimensions differ
        self.shortcut = nn.Linear(dim_in, dim_out) if dim_in != dim_out else nn.Identity()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out = out + identity  # 核心：残差连接 / Core: residual connection
        out = self.activation(out)
        return out

class OptimizedMaskPredictor(nn.Module):
    """
    Learnable mask predictor for point cloud segmentation.
    
    This model learns to segment point clouds by optimizing a set of parameters
    that represent soft assignments of points to different slots (segments).
    The assignments are normalized using softmax to ensure they sum to 1 across slots.
    
    Attributes:
        slot_num (int): Number of segmentation slots (masks)
        point_length (int): Number of points in the point cloud
        device (torch.device): Device to perform computations on
        tensor3d (nn.Parameter): Learnable parameters for mask prediction [K, N]
    """
    
    def __init__(self, slot_num=1, point_length=65536):
        """
        Initialize the mask predictor.
        
        Args:
            slot_num (int): Number of segmentation slots
            point_length (int): Number of points to process
        """
        super(OptimizedMaskPredictor, self).__init__()
        self.slot_num = slot_num
        self.point_length = point_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.randn((slot_num, point_length), device=self.device)
        softmaxed_tensor = F.softmax(tensor, dim=0)
        self.tensor2d = torch.nn.Parameter(softmaxed_tensor, requires_grad=True)
    
    def forward(self, inputs):
        """
        Predict segmentation masks for a batch of point clouds.
        
        Args:
            inputs (dict): Input dictionary containing:
                - point_cloud_first (torch.Tensor): First point cloud [B, N, 3]
                
        Returns:
            torch.Tensor: Predicted segmentation masks [B, K, N] where:
                - B is the batch size
                - K is the number of slots
                - N is the number of points
        """
        # Repeat the parameter tensor for each batch item
        return self.tensor2d 
    
class Neural_Mask_Prior(torch.nn.Module):
    """
    Neural network based mask predictor for point cloud segmentation.
    
    This model uses a multi-layer perceptron to learn point cloud segmentation
    by mapping 3D coordinates to slot assignment probabilities. The network
    architecture is configurable in terms of depth, width, and activation functions.
    
    Attributes:
        layer_size (int): Number of hidden layers
        nn_layers (nn.ModuleList): List of neural network layers
    """

    def __init__(self, input_dim=3, slot_num=10, filter_size=128, act_fn='sigmoid', layer_size=8, dropout=0.2):
        """
        Initialize the neural mask predictor.
        
        Args:
            dim_x (int): Input dimension (default: 3 for xyz coordinates)
            slot_num (int): Number of segmentation slots
            filter_size (int): Width of hidden layers
            act_fn (str): Activation function ('relu' or 'sigmoid')
            layer_size (int): Number of hidden layers
            dropout (float): Dropout probability
        """
        super().__init__()
        self.layer_size = layer_size
        
        self.nn_layers = torch.nn.ModuleList([])
        # input layer (default: xyz -> 128)
        if layer_size >= 1:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(input_dim, filter_size)))
            if act_fn == 'relu':
                self.nn_layers.append(torch.nn.ReLU())
            elif act_fn == 'sigmoid':
                self.nn_layers.append(torch.nn.Sigmoid())
            elif act_fn == "leakyrelu":
                self.nn_layers.append(torch.nn.LeakyReLU())
                #add normalization
            # self.nn_layers.append(torch.nn.BatchNorm1d(filter_size))
            for _ in range(layer_size-1):
                self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(filter_size, filter_size)))
                if act_fn == 'relu':
                    self.nn_layers.append(torch.nn.ReLU())
                elif act_fn == 'sigmoid':
                    self.nn_layers.append(torch.nn.Sigmoid())
                elif act_fn == "leakyrelu":
                    self.nn_layers.append(torch.nn.LeakyReLU())
            self.nn_layers.append(torch.nn.Linear(filter_size, slot_num))
        else:
            self.nn_layers.append(torch.nn.Sequential(torch.nn.Linear(input_dim, slot_num)))

    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x (torch.Tensor): Input point cloud coordinates [N, 3]
            
        Returns:
            torch.Tensor: Predicted slot assignment probabilities [slot_num, N]
        """
        layer_num = 0
        for layer in self.nn_layers:
            layer_num += 1
            # print(f"layer_num: {layer_num}, x_std{x.std()}, x_mean: {x.mean()}")
            x = layer(x)
        # x = F.softmax(x, dim=1)
        return x.permute(1, 0)
    

class EulerMaskMLP(EulerFlowMLP):
    def __init__(
        self, 
        slot_num=10, 
        filter_size=128, 
        act_fn: ActivationFn = ActivationFn.LEAKYRELU,
        layer_size=8, 
        use_normalization: bool = False,
        normalization_type: str = "group_norm",  # 默认使用group_norm节省内存
    ):
        super().__init__(
            output_dim=slot_num,
            latent_dim=filter_size,
            act_fn=act_fn,
            num_layers=layer_size,
            use_normalization=use_normalization,
            normalization_type=normalization_type,
        )
        self.nn_layers = torch.nn.Sequential(self.nn_layers)

    def forward(self, pc, idx, total_entries):
        return super().forward(pc, idx, total_entries, QueryDirection.FORWARD)


class EulerMaskMLPResidual(nn.Module):
    """
    基于残差块的EulerMaskMLP实现，直接初始化编码器
    EulerMaskMLP implementation with residual blocks, directly initializing encoder
    """
    def __init__(
        self, 
        slot_num=10, 
        filter_size=128, 
        act_fn: ActivationFn = ActivationFn.LEAKYRELU,
        layer_size=8, 
        encoder=None,  # 可选的编码器
    ):
        super().__init__()
        self.slot_num = slot_num
        self.filter_size = filter_size
        self.act_fn = act_fn
        self.layer_size = layer_size
        
        # 初始化编码器
        if encoder is None:
            from .eulerflow_raw_mlp import SimpleEncoder
            self.encoder = SimpleEncoder()
        else:
            self.encoder = encoder
            
        # 构建基于残差块的网络
        self._build_residual_network()

    def _build_residual_network(self):
        """构建基于残差块的网络结构"""
        # 获取激活函数
        activation_fn = self._get_activation_fn()
        
        # 获取编码器输出维度
        encoder_dim = len(self.encoder)
        
        # 构建残差网络
        residual_layers = []
        
        # 第一个残差块：从编码器输出到隐藏维度
        residual_layers.append(ResidualBlock(encoder_dim, self.filter_size, activation_fn))
        
        # 中间的残差块：隐藏维度到隐藏维度
        for _ in range(self.layer_size - 1):
            residual_layers.append(ResidualBlock(self.filter_size, self.filter_size, activation_fn))
        
        # 输出层：从隐藏维度到输出维度
        residual_layers.append(nn.Linear(self.filter_size, self.slot_num))
        
        # 构建完整网络：编码器 + 残差块
        self.nn_layers = torch.nn.Sequential(
            self.encoder,       # 编码器
            *residual_layers    # 残差块
        )

    def _get_activation_fn(self):
        """获取激活函数"""
        if self.act_fn == ActivationFn.RELU:
            return nn.ReLU()
        elif self.act_fn == ActivationFn.LEAKYRELU:
            return nn.LeakyReLU()
        elif self.act_fn == ActivationFn.SIGMOID:
            return nn.Sigmoid()
        else:
            return nn.ReLU()  # 默认使用ReLU

    def forward(self, pc, idx, total_entries):
        """
        前向传播
        Forward pass
        """
        entries = (pc, idx, total_entries, QueryDirection.FORWARD)
        return self.nn_layers(entries)

class EulerMaskMLPRoutine(nn.Module):
    """
    包含多个浅层EulerMaskMLP子网络的路由网络
    Routine network containing multiple shallow EulerMaskMLP sub-networks
    
    根据idx选择输入到对应的子网络，每个子网络处理bucketsize个连续的idx
    Selects input to corresponding sub-network based on idx, each sub-network handles bucketsize consecutive idxs
    """
    
    def __init__(
        self,
        slot_num=20,
        filter_size=128,
        act_fn: ActivationFn = ActivationFn.LEAKYRELU,
        layer_size=6,  # 浅层网络，层数较少
        bucketsize=30,  # 每个子网络处理的idx范围大小
        max_buckets=20,  # 最大子网络数量
        use_normalization: bool = False,
        normalization_type: str = "group_norm",
    ):
        super().__init__()
        self.slot_num = slot_num
        self.filter_size = filter_size
        self.act_fn = act_fn
        self.layer_size = layer_size
        self.bucketsize = bucketsize
        self.max_buckets = max_buckets
        self.use_normalization = use_normalization
        self.normalization_type = normalization_type
        
        # 创建多个浅层EulerMaskMLP子网络
        # Create multiple shallow EulerMaskMLP sub-networks
        self.sub_networks = nn.ModuleList()
        for i in range(max_buckets):
            sub_network = EulerMaskMLPResidual(
                slot_num=slot_num,
                filter_size=filter_size,
                act_fn=act_fn,
                layer_size=layer_size,
            )
            self.sub_networks.append(sub_network)
    
    def get_bucket_index(self, idx):
        """
        根据idx计算对应的bucket索引
        Calculate bucket index based on idx
        
        Args:
            idx (int or torch.Tensor): 输入索引 / Input index
            
        Returns:
            int: bucket索引 / bucket index
        """
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        
        bucket_idx = idx // self.bucketsize
        # 确保bucket索引在有效范围内
        # Ensure bucket index is within valid range
        bucket_idx = min(bucket_idx, self.max_buckets - 1)
        return bucket_idx
    
    def forward(self, pc, idx, total_entries):
        """
        前向传播，根据idx选择对应的子网络
        Forward pass, select corresponding sub-network based on idx
        
        Args:
            pc (torch.Tensor): 点云数据 / Point cloud data
            idx (int or torch.Tensor): 索引 / Index
            total_entries (int): 总条目数 / Total entries
            
        Returns:
            torch.Tensor: 预测的mask / Predicted mask
        """
        # 获取对应的bucket索引
        # Get corresponding bucket index
        bucket_idx = self.get_bucket_index(idx)
        
        # 选择对应的子网络进行前向传播
        # Select corresponding sub-network for forward pass
        selected_network = self.sub_networks[bucket_idx]
        
        return selected_network(pc, idx, total_entries)
    
    def get_network_info(self):
        """
        获取网络信息
        Get network information
        
        Returns:
            dict: 网络信息字典 / Network information dictionary
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_sub_networks': len(self.sub_networks),
            'bucketsize': self.bucketsize,
            'max_buckets': self.max_buckets,
            'slot_num': self.slot_num,
            'filter_size': self.filter_size,
            'layer_size': self.layer_size,
        }
    
    def get_sub_network_parameters(self, bucket_idx):
        """
        获取指定子网络的参数
        Get parameters of specified sub-network
        
        Args:
            bucket_idx (int): 子网络索引 / Sub-network index
            
        Returns:
            dict: 子网络参数字典 / Sub-network parameters dictionary
        """
        if bucket_idx >= len(self.sub_networks):
            raise ValueError(f"Bucket index {bucket_idx} out of range [0, {len(self.sub_networks)-1}]")
        
        sub_network = self.sub_networks[bucket_idx]
        return dict(sub_network.named_parameters())
    
    def freeze_sub_network(self, bucket_idx):
        """
        冻结指定子网络的参数
        Freeze parameters of specified sub-network
        
        Args:
            bucket_idx (int): 子网络索引 / Sub-network index
        """
        if bucket_idx >= len(self.sub_networks):
            raise ValueError(f"Bucket index {bucket_idx} out of range [0, {len(self.sub_networks)-1}]")
        
        for param in self.sub_networks[bucket_idx].parameters():
            param.requires_grad = False
    
    def unfreeze_sub_network(self, bucket_idx):
        """
        解冻指定子网络的参数
        Unfreeze parameters of specified sub-network
        
        Args:
            bucket_idx (int): 子网络索引 / Sub-network index
        """
        if bucket_idx >= len(self.sub_networks):
            raise ValueError(f"Bucket index {bucket_idx} out of range [0, {len(self.sub_networks)-1}]")
        
        for param in self.sub_networks[bucket_idx].parameters():
            param.requires_grad = True