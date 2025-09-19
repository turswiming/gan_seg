# GAN_SEG - 场景流预测和实例分割 / Scene Flow Prediction and Instance Segmentation

## 项目概述 / Project Overview

这是一个基于深度学习的3D点云场景流预测和实例分割系统。该项目实现了多种神经网络模型，用于预测点云中的运动流场和进行实例分割。

This is a deep learning-based 3D point cloud scene flow prediction and instance segmentation system. The project implements various neural network models for predicting motion flow fields in point clouds and performing instance segmentation.

## 核心功能模块 / Core Functional Modules

### 1. 主训练模块 / Main Training Module

#### `src/main.py`
- **功能 / Function**: 主训练脚本，处理完整的训练流程
- **核心函数 / Key Functions**:
  - `main(config, writer)`: 主训练函数，协调整个训练过程
  - 支持多种损失函数组合训练 / Supports training with multiple loss function combinations
  - 集成TensorBoard日志记录 / Integrated TensorBoard logging
  - 支持实时可视化 / Real-time visualization support

#### `src/eval.py`
- **功能 / Function**: 模型评估模块
- **核心函数 / Key Functions**:
  - `evaluate_predictions()`: 计算EPE和mIoU指标
  - `eval_model()`: 在验证数据集上评估模型
  - 支持Argoverse2数据集的三类别评估 / Supports three-category evaluation for Argoverse2

### 2. 模型架构 / Model Architecture

#### `src/Predictor.py`
- **功能 / Function**: 模型工厂函数
- **核心函数 / Key Functions**:
  - `get_scene_flow_predictor()`: 创建场景流预测器
  - `get_mask_predictor()`: 创建掩码预测器
  - 支持多种模型架构选择 / Supports multiple model architecture choices

#### `src/model/eulerflow_raw_mlp.py`
- **功能 / Function**: 欧拉流MLP模型实现
- **核心类 / Key Classes**:
  - `EulerFlowMLP`: 基于时间编码的场景流预测模型
  - `SimpleEncoder`: 简单的时间和方向编码器
  - `FourierTemporalEmbedding`: 傅里叶时间嵌入编码器
  - 支持前向和反向查询 / Supports forward and reverse queries

#### `src/model/scene_flow_predict_model.py`
- **功能 / Function**: 场景流预测模型
- **核心类 / Key Classes**:
  - `Neural_Prior`: 基于神经网络的先验模型
  - `OptimizedFLowPredictor`: 基于参数优化的流预测器

#### `src/model/mask_predict_model.py`
- **功能 / Function**: 掩码预测模型
- **核心类 / Key Classes**:
  - `OptimizedMaskPredictor`: 基于参数优化的掩码预测器
  - `Neural_Mask_Prior`: 基于神经网络的掩码先验模型

#### `src/model/nsfp_raw_mlp.py`
- **功能 / Function**: NSFP原始MLP实现
- **核心类 / Key Classes**:
  - `NSFPRawMLP`: 可配置的多层感知机
  - `ActivationFn`: 支持多种激活函数（ReLU, Sigmoid, SinC, Gaussian）

### 3. 数据处理 / Data Processing

#### `src/dataset/av2_dataset.py`
- **功能 / Function**: AV2数据集加载器
- **核心类 / Key Classes**:
  - `AV2PerSceneDataset`: 单场景数据集加载器
  - `AV2SequenceDataset`: 序列数据集加载器
  - 支持ego-motion补偿 / Supports ego-motion compensation
  - 提供前景/背景/动态/静态掩码 / Provides foreground/background/dynamic/static masks

#### `src/utils/dataloader_utils.py`
- **功能 / Function**: 数据加载工具
- **核心函数 / Key Functions**:
  - `infinite_dataloader()`: 创建无限循环数据加载器
  - `create_dataloaders()`: 根据配置创建数据加载器
  - 支持多种数据集类型 / Supports multiple dataset types

### 4. 损失函数 / Loss Functions

#### `src/losses/ChamferDistanceLoss.py`
- **功能 / Function**: 内存高效的Chamfer距离损失
- **核心类 / Key Classes**:
  - `ChamferDistanceLoss`: 分块处理的Chamfer距离计算
  - 支持双向距离计算 / Supports bidirectional distance calculation
  - 内存优化实现 / Memory-optimized implementation

#### `src/losses/FlowSmoothLoss.py`
- **功能 / Function**: 流场平滑损失
- **核心类 / Key Classes**:
  - `FlowSmoothLoss`: 基于二次流近似的平滑损失
  - `ScaleGradient`: 自定义梯度缩放函数
  - 支持L1/L2损失标准 / Supports L1/L2 loss criteria

#### `src/losses/KDTreeDistanceLoss.py`
- **功能 / Function**: KD树距离损失
- **核心类 / Key Classes**:
  - `KDTreeDistanceLoss`: 基于KD树的最近邻距离损失
  - 支持距离截断和缓存 / Supports distance truncation and caching

#### `src/losses/KNNDistanceLoss.py`
- **功能 / Function**: K近邻距离损失
- **核心类 / Key Classes**:
  - `KNNDistanceLoss`: 基于PyTorch3D的KNN距离损失
  - `TruncatedKNNDistanceLoss`: 截断版本的KNN距离损失
  - 支持双向和单向距离计算 / Supports bidirectional and unidirectional distance calculation

#### `src/losses/ReconstructionLoss.py`
- **功能 / Function**: 重建损失
- **核心类 / Key Classes**:
  - `ReconstructionLoss`: 点云重建损失
  - 包含SVD刚体变换拟合 / Includes SVD rigid transformation fitting
  - 支持软KNN插值 / Supports soft KNN interpolation

### 5. 评估指标 / Evaluation Metrics

#### `src/utils/metrics.py`
- **功能 / Function**: 评估指标计算
- **核心函数 / Key Functions**:
  - `calculate_miou()`: 计算平均交并比(mIoU)
  - `calculate_epe()`: 计算端点误差(EPE)
  - 支持实例分割评估 / Supports instance segmentation evaluation

### 6. 配置管理 / Configuration Management

#### `src/config/config.py`
- **功能 / Function**: 配置文件处理
- **核心函数 / Key Functions**:
  - `print_config()`: 打印配置信息
  - `correct_datatype()`: 数据类型修正

#### `src/utils/config_utils.py`
- **功能 / Function**: 配置工具
- **核心函数 / Key Functions**:
  - `load_config_with_inheritance()`: 支持继承的配置加载
  - `save_config_and_code()`: 保存配置和代码文件

### 7. 调度器 / Scheduler

#### `src/alter_scheduler.py`
- **功能 / Function**: 交替训练调度器
- **核心类 / Key Classes**:
  - `AlterScheduler`: 控制流预测和掩码预测的交替训练
  - 支持自定义训练步数配置 / Supports custom training step configuration

### 8. 可视化工具 / Visualization Tools

#### `src/utils/visualization_utils.py`
- **功能 / Function**: 可视化工具函数
- **核心函数 / Key Functions**:
  - `remap_instance_labels()`: 重映射实例标签
  - `color_mask()`: 为掩码生成颜色
  - `create_label_colormap()`: 创建标签颜色映射

#### `src/visualize/open3d_func.py`
- **功能 / Function**: Open3D可视化功能
- **核心函数 / Key Functions**:
  - `visualize_vectors()`: 可视化向量场
  - `update_vector_visualization()`: 更新向量可视化

## 支持的数据集 / Supported Datasets

1. **Argoverse 2 (AV2)**: 自动驾驶场景流数据集 / Autonomous driving scene flow dataset
2. **KITTI Scene Flow**: 经典场景流数据集 / Classic scene flow dataset  
3. **MOViF**: 合成动态场景数据集 / Synthetic dynamic scene dataset

## 支持的模型架构 / Supported Model Architectures

1. **EulerFlowMLP**: 基于欧拉流的时间感知MLP / Time-aware MLP based on Euler flow
2. **NSFP**: 神经场景流先验 / Neural Scene Flow Prior
3. **OptimizedFlow**: 基于参数优化的流预测 / Parameter optimization-based flow prediction
4. **OptimizedMask**: 基于参数优化的掩码预测 / Parameter optimization-based mask prediction

## 主要特性 / Key Features

- 🎯 **多任务学习**: 同时进行场景流预测和实例分割 / Multi-task learning for scene flow and instance segmentation
- ⚡ **内存高效**: 分块处理大型点云 / Memory-efficient processing for large point clouds  
- 🔄 **交替训练**: 智能的流预测和掩码预测交替训练策略 / Alternating training strategy for flow and mask prediction
- 📊 **丰富评估**: 支持多种评估指标和可视化 / Comprehensive evaluation metrics and visualization
- 🛠️ **模块化设计**: 易于扩展的模块化架构 / Modular architecture for easy extension
- 📈 **实时监控**: TensorBoard集成的训练监控 / Real-time training monitoring with TensorBoard

## 使用方法 / Usage

```bash
# 训练模型 / Train model
python src/main.py --config config/baseconfig.yaml

# 使用自定义配置 / Use custom configuration  
python src/main.py --config config/custom.yaml model.flow.lr=0.001

# 评估模型 / Evaluate model
python src/eval.py --config config/eval.yaml --checkpoint path/to/checkpoint
```

## 配置文件结构 / Configuration Structure

配置文件支持继承，可以通过`__base__`字段继承基础配置。主要配置项包括：

Configuration files support inheritance through the `__base__` field. Main configuration items include:

- `dataset`: 数据集配置 / Dataset configuration
- `model`: 模型架构配置 / Model architecture configuration  
- `training`: 训练参数配置 / Training parameter configuration
- `loss`: 损失函数权重配置 / Loss function weight configuration
- `vis`: 可视化配置 / Visualization configuration

## 依赖项 / Dependencies

- PyTorch >= 1.9.0
- PyTorch3D  
- Open3D
- OmegaConf
- TensorBoard
- NumPy
- tqdm

## 许可证 / License

请参考项目根目录下的LICENSE文件。/ Please refer to the LICENSE file in the project root directory.
