#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Flow生成模型和Gumbel-Sinkhorn软置换功能
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from enigma.model import Enigma, EnigmaLM
from enigma.jacobian_logdet import EnigmaFlow
from enigma.gumbel_sinkhorn import GumbelSinkhornRotorStack, GumbelSinkhorn


def test_flow_model():
    """测试Flow模型功能"""
    print("测试Flow模型功能...")
    # 创建Enigma模型
    model = Enigma(
        d=64,
        num_rev_blocks=2,
        num_rotors=2,
        plugboard_sparsity=0.1
    )
    
    # 创建Flow模型
    flow = model.create_flow_model(prior='gaussian')
    
    # 测试采样
    samples = flow.sample(num_samples=2)
    print(f"Flow采样形状: {samples.shape}")
    
    # 测试log_prob
    x = torch.randn(2, 64)
    log_probs = flow.log_prob(x)
    print(f"Flow log_prob: {log_probs}")
    
    print("Flow模型测试通过！\n")
    return True


def test_gumbel_sinkhorn():
    """测试Gumbel-Sinkhorn功能"""
    print("测试Gumbel-Sinkhorn功能...")
    # 创建使用Gumbel-Sinkhorn的Enigma模型
    model = Enigma(
        d=64,
        num_rev_blocks=2,
        num_rotors=2,
        use_gumbel_sinkhorn=True,
        gumbel_temp_min=0.1,
        gumbel_temp_max=1.0
    )
    
    # 测试前向传播
    x = torch.randn(2, 64)
    y = model(x)
    print(f"使用Gumbel-Sinkhorn的Enigma输出形状: {y.shape}")
    
    # 测试温度退火
    temps = model.anneal_gumbel_temperatures()
    print(f"Gumbel-Sinkhorn温度退火: {temps}")
    
    # 测试逆操作
    x_recon = model.inverse(y)
    error = torch.norm(x - x_recon) / torch.norm(x)
    print(f"Gumbel-Sinkhorn重构误差: {error.item()}")
    
    print("Gumbel-Sinkhorn测试通过！\n")
    return True


def test_enigma_lm():
    """测试EnigmaLM模型"""
    print("测试EnigmaLM模型...")
    # 创建EnigmaLM模型
    model = EnigmaLM(
        vocab_size=21128,  # BERT中文词表大小
        d=64,
        num_rev_blocks=2,
        num_rotors=2,
        num_transformer_layers=2,
        num_heads=2,
        use_gumbel_sinkhorn=True
    )
    
    # 测试前向传播
    tokens = torch.randint(0, 21128, (2, 10))  # 批量大小为2，序列长度为10
    logits = model(tokens)
    print(f"EnigmaLM输出形状: {logits.shape}")
    
    # 测试Flow模型创建
    flow = model.create_flow_model()
    samples = flow.sample(num_samples=2)
    print(f"EnigmaLM Flow采样形状: {samples.shape}")
    
    print("EnigmaLM测试通过！\n")
    return True


if __name__ == "__main__":
    print("开始测试Flow和Gumbel-Sinkhorn功能...\n")
    
    # 测试Flow模型
    test_flow_model()
    
    # 测试Gumbel-Sinkhorn
    test_gumbel_sinkhorn()
    
    # 测试EnigmaLM
    test_enigma_lm()
    
    print("所有测试完成！") 