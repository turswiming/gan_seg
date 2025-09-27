import torch
import torch.nn as nn
import enum


class ActivationFn(enum.Enum):
    RELU = "relu"
    SIGMOID = "sigmoid"
    SINC = "sinc"  # https://openreview.net/forum?id=0Lqyut1y7M
    GAUSSIAN = "gaussian"  # https://arxiv.org/abs/2204.05735
    LEAKYRELU = "leakyrelu"


class SinC(nn.Module):
    def __init__(self):
        super(SinC, self).__init__()

    def forward(self, x):
        return torch.sinc(x)


class Gaussian(nn.Module):
    def __init__(
        self,
        sigma: float = 0.1,  # GARF default value
    ):
        super(Gaussian, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        # From https://github.com/sfchng/Gaussian-Activated-Radiance-Fields/blob/74d72387bb2526755a8d6c07f6f900ec6a1be594/model/nerf_gaussian.py#L457-L464
        return (-0.5 * (x) ** 2 / self.sigma**2).exp()


import torch
import torch.nn as nn

class ScaleOnlyBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(ScaleOnlyBatchNorm1d, self).__init__(num_features, eps, momentum, affine, track_running_stats)
        if self.affine:
            # 初始化 bias 为 0，并且不跟踪其梯度
            self.bias = None  # 不使用 bias 参数
            self.weight = nn.Parameter(torch.ones(num_features))  # 只保留 scale 参数

    def forward(self, input):
        # 复用 BatchNorm1d 的运行均值和方差计算
        self._check_input_dim(input)

        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # 使用累积平均
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        # 计算归一化
        output = nn.functional.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            None,  # 不使用 bias
            self.training or not self.track_running_stats,
            exponential_average_factor,
            self.eps,
        )
        return output

class NSFPRawMLP(nn.Module):

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 8,
        with_compile: bool = True,
        use_normalization: bool = False,
        normalization_type: str = "layer_norm",  # "layer_norm", "group_norm", "none"
    ):
        super().__init__()
        self.layer_size = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.act_fn = act_fn
        self.use_normalization = use_normalization
        self.normalization_type = normalization_type
        self.nn_layers = torch.nn.Sequential(*self._make_model())
        if with_compile:
            # 暂时禁用 torch.compile 以避免兼容性问题
            # self.nn_layers = torch.compile(self.nn_layers, dynamic=True)
            pass

    def _get_activation_fn(self) -> nn.Module:
        match self.act_fn:
            case ActivationFn.RELU:
                return torch.nn.ReLU()
            case ActivationFn.SIGMOID:
                return torch.nn.Sigmoid()
            case ActivationFn.SINC:
                return SinC()
            case ActivationFn.GAUSSIAN:
                return Gaussian()
            case ActivationFn.LEAKYRELU:
                return torch.nn.LeakyReLU()
            case _:
                raise ValueError(f"Unsupported activation function: {self.act_fn}")

    def _get_normalization_layer(self) -> nn.Module:
        """创建指定类型的normalization层"""
        if self.normalization_type == "layer_norm":
            return torch.nn.LayerNorm(self.latent_dim, eps=1e-6)
        elif self.normalization_type == "group_norm":
            # GroupNorm使用更少的内存，将通道分成16组
            num_groups = min(16, self.latent_dim // 4)  # 确保每组至少4个通道
            num_groups = max(1, num_groups)  # 至少1组
            return torch.nn.GroupNorm(num_groups, self.latent_dim, eps=1e-6)
        elif self.normalization_type == "batch_norm":
            return ScaleOnlyBatchNorm1d(self.latent_dim, eps=1e-6)
        else:
            raise ValueError(f"Unsupported normalization type: {self.normalization_type}")

    def _make_model(self) -> torch.nn.ModuleList:
        nn_layers = torch.nn.ModuleList([])
        if self.layer_size <= 1:
            nn_layers.append(torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.output_dim)))
            return nn_layers

        # First layer
        nn_layers.append(torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.latent_dim)))
        if self.use_normalization:
            nn_layers.append(self._get_normalization_layer())
        nn_layers.append(self._get_activation_fn())
        
        # Hidden layers
        for _ in range(self.layer_size - 1):
            nn_layers.append(torch.nn.Sequential(torch.nn.Linear(self.latent_dim, self.latent_dim)))
            if self.use_normalization:
                nn_layers.append(self._get_normalization_layer())
            nn_layers.append(self._get_activation_fn())
        
        # Output layer (no normalization on output)
        nn_layers.append(torch.nn.Linear(self.latent_dim, self.output_dim))

        return nn_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.nn_layers(x)
