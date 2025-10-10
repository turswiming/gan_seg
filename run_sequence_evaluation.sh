#!/bin/bash

# Script to run sequence mIoU evaluation
# 运行序列mIoU评估的脚本

# Set the checkpoint path (update this to your actual checkpoint path)
# 设置检查点路径（请更新为您的实际检查点路径）
CHECKPOINT_PATH="/workspace/gan_seg/outputs/exp/20251009_200151/checkpoints/step_8000.pt"

# Set the config path
# 设置配置文件路径
CONFIG_PATH="/workspace/gan_seg/outputs/exp/20251009_153404/config.yaml"

# Run the evaluation script
# 运行评估脚本
echo "Starting sequence mIoU evaluation..."
echo "开始序列mIoU评估..."

python evaluate_sequence_miou.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --config "$CONFIG_PATH" \
    --device auto \
    --max-frames 170 \
    --output "sequence_miou_results.pt" \
    --visualize \
    --save-plots

echo "Evaluation completed!"
echo "评估完成！"
