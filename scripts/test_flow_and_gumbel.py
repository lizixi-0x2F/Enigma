#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Flow生成模型和Gumbel-Sinkhorn软置换功能
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from enigma.model import Enigma
from enigma.jacobian_logdet import EnigmaFlow
from enigma.gumbel_sinkhorn import GumbelSinkhornRotorStack, GumbelSinkhorn


def test_flow_model():
    """测试Flow生成模型"""
    print("=== 测试Flow生成模型 ===")
    
    # 创建Enigma模型
    d = 32  # 使用较小的维度进行测试
    model = Enigma(
        d=d,
        num_rev_blocks=3,
        num_rotors=2,
        plugboard_sparsity=0.1
    )
    
    # 方法1：使用factory方法创建Flow模型
    flow_model1 = model.create_flow_model(prior='gaussian')
    
    # 方法2：直接创建Flow模型
    flow_model2 = EnigmaFlow(model, prior='gaussian')
    
    # 测试采样功能
    print("测试生成样本...")
    num_samples = 100
    with torch.no_grad():
        samples = flow_model1.sample(num_samples=num_samples)
    
    print(f"生成的样本形状: {samples.shape}")
    
    # 测试对数概率计算
    print("测试计算对数概率...")
    x = torch.randn(10, d)
    log_probs = flow_model1.log_prob(x)
    
    print(f"样本对数概率形状: {log_probs.shape}")
    print(f"样本对数概率值: {log_probs}")
    
    # 测试Flow模型的可逆性
    print("测试Flow模型的可逆性...")
    z = torch.randn(5, d)
    with torch.no_grad():
        x = flow_model1.inverse(z)
        z_reconstructed = flow_model1.forward(x)
    
    reconstruction_error = torch.norm(z - z_reconstructed) / torch.norm(z)
    print(f"Flow模型重构误差: {reconstruction_error.item()}")
    
    return flow_model1


def test_gumbel_sinkhorn():
    """测试Gumbel-Sinkhorn软置换"""
    print("\n=== 测试Gumbel-Sinkhorn软置换 ===")
    
    # 创建单个Gumbel-Sinkhorn层进行测试
    dim = 8
    gumbel = GumbelSinkhorn(dim=dim, temperature=1.0)
    
    # 测试生成置换矩阵
    print("测试生成置换矩阵...")
    with torch.no_grad():
        perm_matrix = gumbel()
    
    print(f"生成的置换矩阵形状: {perm_matrix.shape}")
    print("置换矩阵:")
    print(perm_matrix)
    
    # 验证是双随机矩阵
    row_sums = perm_matrix.sum(dim=1)
    col_sums = perm_matrix.sum(dim=0)
    
    print(f"行和: {row_sums}")
    print(f"列和: {col_sums}")
    
    # 测试温度退火
    print("测试温度退火...")
    temps = []
    perms = []
    
    # 模拟退火过程
    temp = 1.0
    for i in range(10):
        gumbel.temperature = temp
        with torch.no_grad():
            perm = gumbel()
        
        temps.append(temp)
        # 计算离散化程度（最大值与总和的比值）
        discreteness = (perm.max(dim=1)[0] / perm.sum(dim=1)).mean().item()
        perms.append(discreteness)
        
        temp *= 0.8  # 降低温度
    
    print(f"温度变化: {temps}")
    print(f"离散化程度变化: {perms}")
    
    # 可视化退火过程
    plt.figure(figsize=(10, 5))
    plt.plot(temps, perms, 'o-', label='离散化程度')
    plt.xlabel('温度')
    plt.ylabel('离散化程度')
    plt.title('Gumbel-Sinkhorn温度退火效果')
    plt.legend()
    plt.grid(True)
    
    # 保存图像到文件
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/gumbel_annealing.png')
    print("退火过程图像已保存至 outputs/gumbel_annealing.png")
    
    return gumbel


def test_enigma_with_gumbel_sinkhorn():
    """测试使用Gumbel-Sinkhorn转子的Enigma模型"""
    print("\n=== 测试使用Gumbel-Sinkhorn转子的Enigma模型 ===")
    
    # 创建使用Gumbel-Sinkhorn转子的Enigma模型
    d = 32
    model = Enigma(
        d=d,
        num_rev_blocks=3,
        num_rotors=3,
        use_gumbel_sinkhorn=True,
        gumbel_temp_min=0.1,
        gumbel_temp_max=1.0
    )
    
    # 测试前向传播
    print("测试前向传播...")
    x = torch.randn(10, d)
    y = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    
    # 测试可逆性
    print("测试可逆性...")
    x_reconstructed = model.inverse(y)
    error = torch.norm(x - x_reconstructed) / torch.norm(x)
    
    print(f"重构误差: {error.item()}")
    
    # 测试温度退火
    print("测试Gumbel-Sinkhorn温度退火...")
    for i in range(5):
        temp = model.anneal_gumbel_temperatures()
        print(f"退火后温度: {temp}")
    
    # 与普通Enigma模型比较
    print("与普通Enigma模型比较...")
    standard_model = Enigma(
        d=d,
        num_rev_blocks=3,
        num_rotors=3,
        use_gumbel_sinkhorn=False
    )
    
    # 计算前向传播时间
    import time
    
    # 测试Gumbel-Sinkhorn模型速度
    start_time = time.time()
    for _ in range(100):
        y_gumbel = model(x)
    gumbel_time = time.time() - start_time
    
    # 测试标准模型速度
    start_time = time.time()
    for _ in range(100):
        y_standard = standard_model(x)
    standard_time = time.time() - start_time
    
    print(f"Gumbel-Sinkhorn模型前向传播时间: {gumbel_time:.4f}秒")
    print(f"标准模型前向传播时间: {standard_time:.4f}秒")
    print(f"速度比例(Gumbel/标准): {gumbel_time/standard_time:.2f}")
    
    return model


def test_combined_features():
    """测试Flow模型和Gumbel-Sinkhorn的结合使用"""
    print("\n=== 测试Flow模型和Gumbel-Sinkhorn的结合使用 ===")
    
    # 创建使用Gumbel-Sinkhorn转子的Enigma模型
    d = 32
    model = Enigma(
        d=d,
        num_rev_blocks=3,
        num_rotors=3,
        use_gumbel_sinkhorn=True,
        gumbel_temp_min=0.1,
        gumbel_temp_max=1.0
    )
    
    # 创建基于Gumbel-Sinkhorn Enigma的Flow模型
    flow_model = model.create_flow_model(prior='gaussian')
    
    # 测试采样
    print("测试采样...")
    samples = flow_model.sample(num_samples=50)
    print(f"采样形状: {samples.shape}")
    
    # 测试可逆性
    print("测试可逆性...")
    z = torch.randn(10, d)
    with torch.no_grad():
        x = flow_model.inverse(z)
        z_reconstructed = flow_model.forward(x)
    
    error = torch.norm(z - z_reconstructed) / torch.norm(z)
    print(f"重构误差: {error.item()}")
    
    # 测试温度退火如何影响Flow模型
    print("测试温度退火对Flow模型的影响...")
    log_probs_before = flow_model.log_prob(samples[:5])
    
    # 执行多次退火
    for i in range(5):
        model.anneal_gumbel_temperatures()
    
    log_probs_after = flow_model.log_prob(samples[:5])
    
    print(f"退火前对数概率: {log_probs_before}")
    print(f"退火后对数概率: {log_probs_after}")
    print(f"对数概率变化: {log_probs_after - log_probs_before}")
    
    return flow_model


if __name__ == "__main__":
    # 设置随机种子以保证可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 运行测试
    flow_model = test_flow_model()
    gumbel = test_gumbel_sinkhorn()
    enigma_gumbel = test_enigma_with_gumbel_sinkhorn()
    combined = test_combined_features()
    
    print("\n所有测试完成!") 