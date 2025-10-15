"""
EulerFlow Residual MLP implementation for scene flow prediction.

This module implements a residual-based flow predictor that uses residual blocks
for improved flow prediction performance. It extends the base EulerFlowMLP with
residual connections for better gradient flow and training stability.
"""

import torch
from torch import nn
from torch.nn import functional as F
from .eulerflow_raw_mlp import EulerFlowMLP, ActivationFn, QueryDirection, BaseEncoder, SimpleEncoder


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


class EulerFlowMLPResidual(nn.Module):
    """
    基于残差块的EulerFlowMLP实现，直接初始化编码器
    EulerFlowMLP implementation with residual blocks, directly initializing encoder
    """
    def __init__(
        self, 
        output_dim: int = 3,
        latent_dim: int = 128, 
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 8, 
        encoder: BaseEncoder = None,  # 可选的编码器
        use_normalization: bool = False,
        normalization_type: str = "layer_norm",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.act_fn = act_fn
        self.num_layers = num_layers
        self.use_normalization = use_normalization
        self.normalization_type = normalization_type
        
        # 初始化编码器
        if encoder is None:
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
        residual_layers.append(ResidualBlock(encoder_dim, self.latent_dim, activation_fn))
        
        # 中间的残差块：隐藏维度到隐藏维度
        for _ in range(self.num_layers - 1):
            residual_layers.append(ResidualBlock(self.latent_dim, self.latent_dim, activation_fn))
        
        # 输出层：从隐藏维度到输出维度
        residual_layers.append(nn.Linear(self.latent_dim, self.output_dim))
        
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
        elif self.act_fn == ActivationFn.SINC:
            return nn.SiLU()  # 使用SiLU作为SINC的近似
        else:
            return nn.ReLU()  # 默认使用ReLU

    def forward(self, pc, idx, total_entries, query_direction=QueryDirection.FORWARD):
        """
        前向传播
        Forward pass
        
        Args:
            pc (torch.Tensor): Point cloud data [N, 3]
            idx (int): Frame index
            total_entries (int): Total number of frames
            query_direction (QueryDirection): Query direction (FORWARD/REVERSE)
            
        Returns:
            torch.Tensor: Predicted flow vectors [N, 3]
        """
        entries = (pc, idx, total_entries, query_direction)
        return self.nn_layers(entries)


class EulerFlowMLPRoutine(nn.Module):
    """
    包含多个浅层EulerFlowMLP子网络的路由网络
    Routine network containing multiple shallow EulerFlowMLP sub-networks
    
    根据idx选择输入到对应的子网络，每个子网络处理bucketsize个连续的idx
    Selects input to corresponding sub-network based on idx, each sub-network handles bucketsize consecutive idxs
    """
    
    def __init__(
        self,
        output_dim: int = 3,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 6,  # 浅层网络，层数较少
        bucketsize: int = 30,  # 每个子网络处理的idx范围大小
        max_buckets: int = 20,  # 最大子网络数量
        use_normalization: bool = False,
        normalization_type: str = "layer_norm",
    ):
        super().__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.act_fn = act_fn
        self.num_layers = num_layers
        self.bucketsize = bucketsize
        self.max_buckets = max_buckets
        self.use_normalization = use_normalization
        self.normalization_type = normalization_type
        
        # 创建多个浅层EulerFlowMLP子网络
        # Create multiple shallow EulerFlowMLP sub-networks
        self.sub_networks = nn.ModuleList()
        for i in range(max_buckets):
            sub_network = EulerFlowMLPResidual(
                output_dim=output_dim,
                latent_dim=latent_dim,
                act_fn=act_fn,
                num_layers=num_layers,
                use_normalization=use_normalization,
                normalization_type=normalization_type,
            )
            self.sub_networks.append(sub_network)
    
    def get_bucket_index(self, idx):
        """
        根据idx获取对应的bucket索引
        Get bucket index based on idx
        """
        return min(idx // self.bucketsize, self.max_buckets - 1)
    
    def forward(self, pc, idx, total_entries, query_direction=QueryDirection.FORWARD):
        """
        前向传播，根据idx选择对应的子网络
        Forward pass, select corresponding sub-network based on idx
        """
        bucket_idx = self.get_bucket_index(idx)
        return self.sub_networks[bucket_idx](pc, idx, total_entries, query_direction)
