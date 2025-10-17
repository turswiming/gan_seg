# 点云序列可视化工具使用说明

## 概述
本项目提供了两个点云序列可视化工具，用于在Open3D窗口中可视化Argoverse2数据集的点云序列。

## 工具说明

### 1. 完整版可视化工具 (`visualize_sequence.py`)
功能丰富的可视化工具，支持多种显示模式。

**特性：**
- 支持显示所有帧的合并点云
- 支持单帧模式
- 支持自动播放模式
- 可配置点云大小、播放速度等参数

**使用方法：**
```bash
# 基本使用 - 显示所有帧的合并点云
python src/visualize_sequence.py --raw_dir /path/to/av2/data --show_all

# 渲染为JPEG图片（推荐）
python src/visualize_sequence.py --raw_dir /path/to/av2/data --render_images --show_all

# 自动播放模式
python src/visualize_sequence.py --raw_dir /path/to/av2/data --auto_play --play_speed 2.0

# 限制帧数
python src/visualize_sequence.py --raw_dir /path/to/av2/data --max_frames 20 --show_all
```

**参数说明：**
- `--raw_dir`: AV2原始数据路径（必需）
- `--flow_dir`: 流数据路径（可选）
- `--use_gt_flow`: 使用真实流数据
- `--sequence_idx`: 要可视化的序列索引（默认0）
- `--max_frames`: 最大帧数限制
- `--show_all`: 显示所有帧的合并点云
- `--auto_play`: 自动播放模式
- `--play_speed`: 播放速度（默认2.0）
- `--point_size`: 点云点的大小（默认1.0）
- `--render_images`: 渲染为JPEG图片（推荐）
- `--offline`: 离线模式，保存点云文件

### 2. 简化版可视化工具 (`visualize_simple.py`)
简单易用的可视化工具，适合快速查看。

**特性：**
- 简单易用，参数少
- 自动为每帧分配不同颜色
- 适合快速预览

**使用方法：**
```bash
# 基本使用
python src/visualize_simple.py --raw_dir /path/to/av2/data

# 指定序列和帧数
python src/visualize_simple.py --raw_dir /path/to/av2/data --sequence_idx 1 --max_frames 15
```

**参数说明：**
- `--raw_dir`: AV2原始数据路径（必需）
- `--sequence_idx`: 要可视化的序列索引（默认0）
- `--max_frames`: 最大帧数（默认10）

## 使用示例

### 示例1：快速预览第一个序列
```bash
python src/visualize_simple.py --raw_dir /workspace/av2_data
```

### 示例2：渲染为JPEG图片（推荐）
```bash
python src/visualize_sequence.py --raw_dir /workspace/av2_data --render_images --show_all --max_frames 20
```

### 示例3：查看特定序列的所有帧
```bash
python src/visualize_sequence.py --raw_dir /workspace/av2_data --sequence_idx 2 --show_all --max_frames 50
```

### 示例4：自动播放模式
```bash
python src/visualize_sequence.py --raw_dir /workspace/av2_data --auto_play --play_speed 1.5 --max_frames 30
```

## 可视化说明

### 颜色编码
- 每帧使用不同的颜色显示
- 颜色按HSV色彩空间均匀分布
- 同一帧内的所有点使用相同颜色

### 交互控制
- **鼠标左键拖拽**: 旋转视角
- **鼠标滚轮**: 缩放
- **鼠标右键拖拽**: 平移
- **ESC键**: 退出

### 坐标系
- 使用全局坐标系显示点云
- 如果全局坐标不可用，则使用ego坐标系

## 注意事项

1. **数据路径**: 确保`--raw_dir`指向正确的AV2数据目录
2. **内存使用**: 显示大量帧时可能占用较多内存
3. **性能**: 点云数量过多时可能影响渲染性能
4. **依赖**: 需要安装Open3D库

## 故障排除

### 常见问题

1. **"没有成功加载任何帧数据"**
   - 检查数据路径是否正确
   - 确认数据格式是否支持

2. **"序列索引超出范围"**
   - 使用`--sequence_idx 0`查看可用序列
   - 检查数据目录是否包含序列数据

3. **渲染性能问题**
   - 减少`--max_frames`参数
   - 降低`--point_size`参数

## 技术细节

### 数据访问
- 使用`frame.pc.global_pc.points`获取全局坐标点云
- 备用方案使用`frame.pc.pc.points`获取ego坐标点云

### 颜色生成
- 使用HSV色彩空间生成均匀分布的颜色
- 饱和度0.8，亮度0.9，色调按帧索引分布

### 渲染设置
- 背景色: 深灰色 (0.1, 0.1, 0.1)
- 点大小: 可配置
- 窗口大小: 1200x800
