#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用4张GPU
export RESUME_CHECKPOINT="checkpoints_optimized/best_model.pt"  # 从最佳模型继续训练

echo "启动全数据集训练..."
nohup python -m scripts.train_full_data > train_full.log 2>&1 &
echo "训练已在后台启动，日志保存在train_full.log"
echo "可以使用 'tail -f train_full.log' 查看训练进度" 