#!/usr/bin/env python
import os
import argparse
from scripts.train_large import train as train_original
import sys

def main():
    """使用全量数据进行长期训练的主入口"""
    # 使修改在sys.argv生效
    sys.argv = [
        sys.argv[0],  # 脚本名称
        "--epochs", "15",  # 增加到15个epoch
        "--data", "wiki-full-zh",  # 数据集路径
        "--save-dir", "checkpoints_full", # 保存目录
        "--d-model", "768",  # 模型维度
        "--batch-size", "32",  # 批量大小
        "--grad-accum-steps", "4",  # 增加梯度累积步数为4
        "--num-rev-blocks", "6",  # Enigma RevBlock层数
        "--num-rotors", "4",  # Enigma转子数量
        "--num-transformer-layers", "12",  # Transformer层数
        "--num-heads", "12",  # 注意力头数
        "--lr", "8e-4",  # 略微降低学习率以增加稳定性
        "--warmup-ratio", "0.1",  # 预热步数比例
        "--weight-decay", "1e-3",  # 权重衰减
        "--use-amp",  # 使用混合精度训练
        "--amp-dtype", "float16",  # 明确指定使用fp16精度
        "--use-alibi",  # 使用ALiBi位置编码
        "--max-len", "8192",  # 最大序列长度
        "--tokenizer", "bert-chinese-base",  # 使用bert tokenizer
        "--use-saved-tokenizer",  # 使用保存的分词器
        "--saved-tokenizer-path", "tokenizer/tokenizer.pkl",  # 分词器路径
        "--save-every", "5000",  # 每5000步保存一次
        "--eval-every", "1000",  # 每1000步评估一次
        "--use-gumbel-sinkhorn",  # 启用Gumbel-Sinkhorn软置换
        "--gumbel-temp-min", "0.1",  # Gumbel-Sinkhorn最小温度
        "--gumbel-temp-max", "1.0",  # Gumbel-Sinkhorn最大温度
        "--anneal-every", "2000",  # 每2000步进行一次温度退火
        "--enable-flow-training",  # 启用Flow模型训练
        "--flow-weight", "0.1",  # Flow模型损失权重
        "--flow-prior", "gaussian",  # Flow模型先验分布
    ]
    
    # 替换命令行参数
    if len(sys.argv) > 1:
        # 允许从命令行覆盖一些参数
        for i in range(1, len(sys.argv)):
            arg = sys.argv[i]
            if arg.startswith("--"):
                param_name = arg[2:]
                # 查找默认参数中该参数的位置
                for j in range(1, len(sys.argv)):
                    if sys.argv[j].startswith(f"--{param_name}"):
                        # 如果参数后有值，则还要覆盖值
                        if j + 1 < len(sys.argv) and not sys.argv[j + 1].startswith("--"):
                            sys.argv[j + 1] = sys.argv[i + 1]
                        break
                else:
                    # 如果没找到，说明是一个新的参数，直接添加
                    sys.argv.append(arg)
                    if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                        sys.argv.append(sys.argv[i + 1])
    
    # 如果指定了从哪个检查点恢复训练
    resume_checkpoint = os.environ.get("RESUME_CHECKPOINT")
    if resume_checkpoint:
        sys.argv.extend(["--resume", resume_checkpoint])
    
    # 打印训练命令
    print("训练命令:")
    print(" ".join(sys.argv))
    
    # 调用原始训练函数
    train_original()

if __name__ == "__main__":
    main() 