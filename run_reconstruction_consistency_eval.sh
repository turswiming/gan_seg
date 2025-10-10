#!/bin/bash

# Script to run reconstruction loss consistency evaluation
# 运行重建损失一致性评估的脚本

echo "Starting reconstruction loss consistency evaluation..."
echo "开始重建损失一致性评估..."

# Run the evaluation script
# 运行评估脚本
cd /workspace/gan_seg
python src/ablation/reconstruction_loss_consistency_eval.py

echo "Reconstruction loss consistency evaluation completed!"
echo "重建损失一致性评估完成！"



