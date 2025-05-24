#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EnigmaLM 训练启动脚本
使用防过拟合的最佳实践配置
"""

import subprocess
import sys
import os

def main():
    """启动训练"""
    print("🚀 启动 EnigmaLM 训练")
    print("=" * 50)
    
    # 检查数据是否存在
    data_path = "wiki-full-zh/processed/train_seq256_bert_fast.pt"
    if not os.path.exists(data_path):
        print("❌ 训练数据不存在！")
        print(f"请确保 {data_path} 文件存在")
        return 1
    
    print("✅ 数据文件检查通过")
    print(f"🎯 使用防过拟合配置训练 EnigmaLM")
    print()
    
    # 启动训练脚本
    try:
        subprocess.run([sys.executable, "scripts/train.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        return 1
    
    print("🎉 训练完成！")
    return 0

if __name__ == "__main__":
    exit(main()) 