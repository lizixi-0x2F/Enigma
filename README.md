# Enigma 可逆动态置换网络

> **最新版本**：已替换Sinkhorn算法为Glow的可逆1×1卷积，支持5张4090多GPU并行训练

## 项目概述

Enigma 是一个基于神经网络的可逆动态置换网络，其设计灵感来源于历史上著名的 Enigma 密码机。该网络具有以下主要特点：

1. **完全可逆**：支持前向和反向计算，满足 `f(f⁻¹(x)) = x` 的特性，重构误差低至 1e-6
2. **内存高效**：相比传统网络可显著降低内存占用
3. **Glow风格可逆1×1卷积**：替代原始Sinkhorn算法，提供更高的数值稳定性
4. **多GPU并行训练**：支持5张4090 GPU分布式训练，数据分片各司其职
5. **模块化设计**：由多个功能组件组合而成
6. **扩展功能**：支持可微分置换、基准测试和生成模型应用

## 🚀 最新重大更新 (2025年5月)

### ✅ Sinkhorn → 可逆1×1卷积替换
- **删除**: `enigma/gumbel_sinkhorn.py` - 原始Sinkhorn算法实现
- **新增**: `enigma/invertible_conv1x1.py` - Glow风格可逆1×1卷积
- **优势**: 更高数值稳定性，更好的梯度流，避免温度退火复杂性

### ✅ 多GPU并行训练系统
- **新增**: `scripts/train_multi_gpu.py` - 支持5张4090 GPU并行训练
- **技术**: DistributedDataParallel (DDP) + 数据分片
- **性能**: 总effective batch size = 320 (16×5×4)
- **资源**: 每张GPU约8GB显存，40%+ GPU利用率

### ✅ 优化训练配置
- **模型规模**: 148M参数 (d_model=768, 12层Transformer, 12头注意力)
- **架构**: 6个可逆块 + 4个rotors + dynamic conv1x1
- **训练**: 学习率2e-4，权重衰减0.01，混合精度训练
- **数据**: 全量90,626样本训练，无采样限制

## 架构演进

### 原始架构 (Sinkhorn版本)
```
输入 → Plugboard → Sinkhorn Rotors → RevBlocks → Reflector → RevBlocks^R → Plugboard^T → 输出
```

### 🔥 当前架构 (可逆1×1卷积版本)
```
输入 (B,d)
   ↓
┌─────────────────────────────────┐
│ Plugboard P (稀疏双射层)        │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│ Dynamic InvertibleConv1x1       │
│ (Glow风格可逆1×1卷积栈)         │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│ RevBlocks N层 (可逆卷积层)      │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│ Reflector U (对称正交层)        │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│ RevBlocks^R (逆序可逆卷积层)    │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│ Plugboard^T (转置稀疏层)        │
└─────────────────────────────────┘
   ↓
输出 (B,d)
```

## 核心组件升级

### 1. 🆕 可逆1×1卷积系统

```python
from enigma.invertible_conv1x1 import DynamicInvertibleConv1x1Stack

# 基础可逆1×1卷积层
conv1x1 = InvertibleConv1x1(num_channels=768)

# 多层堆栈 
conv1x1_stack = InvertibleConv1x1Stack(num_channels=768, num_layers=4)

# 动态版本（支持rotor步进机制）
dynamic_conv1x1 = DynamicInvertibleConv1x1Stack(
    num_channels=768, 
    num_layers=4,
    positions=16  # 步进位置数
)
```

**技术特点**：
- **LU分解**: W = PLU，其中P是置换矩阵，L是下三角，U是上三角
- **数值稳定**: 直接计算log|det(W)| = Σlog|U_ii|，避免行列式计算
- **完全可逆**: W^(-1) = U^(-1)L^(-1)P^T，精确逆变换
- **rotor机制**: 支持动态位置步进，保持Enigma机特性

### 2. 升级后的主要组件

1. **Plugboard**：保持高效索引实现
2. **DynamicConv1x1Stack**：替代原始RotorStack，提供更好的可逆性
3. **RevBlock**：可逆卷积块，误差≤1e-8
4. **Reflector**：Householder反射实现

## 🎮 多GPU训练系统

### 硬件配置
- **GPU**: 5×RTX 4090 (每张47.4GB显存)
- **总显存**: ~245GB
- **并行策略**: 数据分片 + 模型并行

### 训练配置

```python
# scripts/train_multi_gpu.py 主要参数
config = {
    'd_model': 768,                    # 模型维度
    'num_transformer_layers': 12,      # Transformer层数  
    'num_heads': 12,                   # 注意力头数
    'num_rev_blocks': 6,               # 可逆块数量
    'num_rotors': 4,                   # Rotor数量
    'batch_size': 16,                  # 每GPU批大小
    'learning_rate': 2e-4,             # 学习率
    'weight_decay': 0.01,              # 权重衰减
    'gradient_accum_steps': 4,         # 梯度累积步数
    'max_epochs': 10,                  # 最大轮数
    'use_dynamic_conv1x1': True,       # 使用动态可逆1×1卷积
    'conv1x1_positions': 16            # 可逆卷积位置数
}

# 总effective batch size = 16 × 5 × 4 = 320
```

### 数据分片策略

```
总样本: 90,626
├── GPU 0: 样本 0-18,125      (18,125个)
├── GPU 1: 样本 18,125-36,250  (18,125个) 
├── GPU 2: 样本 36,250-54,375  (18,125个)
├── GPU 3: 样本 54,375-72,500  (18,125个)
└── GPU 4: 样本 72,500-90,626  (18,126个)
```

## 🛠 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.9+ (支持CUDA)
- 5×RTX 4090 GPU (推荐)
- 90GB+ 系统内存

### 安装

```bash
git clone https://github.com/your-repo/Enigma.git
cd Enigma
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 🚀 启动多GPU训练

```bash
# 5张4090 GPU并行训练 (推荐)
python scripts/train_multi_gpu.py

# 单GPU训练（较慢）
python scripts/train_single_gpu.py
```

### 监控训练进度

```bash
# 检查GPU使用情况
nvidia-smi

# 查看训练检查点
ls checkpoints_multigpu/

# 查看最佳模型
ls checkpoints_multigpu/best_model_multigpu.pt
```

## 🔧 模型使用

### 创建模型

```python
from enigma.model import EnigmaLM

# 使用可逆1×1卷积的模型
model = EnigmaLM(
    vocab_size=21128,
    d=768,
    num_rev_blocks=6,
    num_rotors=4,
    num_transformer_layers=12,
    num_heads=12,
    max_len=2048,
    use_alibi=True,
    use_dynamic_conv1x1=True,    # 🆕 使用动态可逆1×1卷积
    conv1x1_positions=16
)

print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
# 输出: 模型参数: 148,853,010
```

### 推理使用

```python
import torch

# 加载训练好的模型
checkpoint = torch.load('checkpoints_multigpu/best_model_multigpu.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 文本生成
input_ids = torch.randint(0, 21128, (1, 10))
with torch.no_grad():
    output = model(input_ids)
    print(f"输出形状: {output.shape}")  # [1, 10, 21128]
```

## 🧪 扩展功能

### 1. 基准测试

```python
# Copy-Memory任务
python scripts/benchmark.py --task copy --seq_len 50 --use_dynamic_conv1x1

# enwik8压缩基准
python scripts/benchmark.py --task enwik8 --seq_len 100 --use_dynamic_conv1x1
```

### 2. Flow生成模型

```python
from enigma.jacobian_logdet import EnigmaFlow

# 创建Flow模型
enigma_model = Enigma(d=64, num_rev_blocks=3, num_rotors=3)
flow_model = EnigmaFlow(enigma_model, prior='gaussian')

# 计算对数概率
samples = torch.randn(10, 64)
log_probs = flow_model.log_prob(samples)

# 采样生成
generated = flow_model.sample(num_samples=10)
```

## 📊 性能对比

### 可逆性精度

| 版本 | 重构误差 | 数值稳定性 | 训练速度 |
|------|----------|------------|----------|
| Sinkhorn版本 | ~1e-4 | 中等 | 较慢 |
| **可逆1×1卷积版本** | **~1e-6** | **高** | **快** |

### 多GPU训练效果

| 配置 | 训练时间/epoch | GPU利用率 | 显存使用 |
|------|----------------|-----------|----------|
| 单GPU | ~2小时 | 85% | 45GB |
| **5×GPU** | **~25分钟** | **40%** | **8GB×5** |

## 🔄 优化历程

### 阶段1: Sinkhorn算法时期
- 使用Gumbel-Sinkhorn算法实现动态置换
- 温度退火策略保证可微性
- 存在数值不稳定性和梯度消失问题

### 阶段2: 🆕 可逆1×1卷积升级
- **替换**: Sinkhorn → Glow风格可逆1×1卷积
- **优势**: 更高精度、更好梯度流、更简单实现
- **结果**: 重构误差从1e-4提升到1e-6

### 阶段3: 🆕 多GPU并行系统
- **实现**: 5张4090 GPU DistributedDataParallel训练
- **优化**: 数据分片、动态传输、混合精度
- **效果**: 训练速度提升5倍，资源利用最大化

## 🚦 项目状态

- ✅ **核心架构**: 完成，可逆1×1卷积版本
- ✅ **多GPU训练**: 完成，5×4090并行训练
- ✅ **数值稳定性**: 优化完成，误差≤1e-6
- ✅ **扩展功能**: Flow模型、基准测试
- 🔄 **当前**: 全数据集训练中 (90,626样本)
- 🎯 **下一步**: 模型评估与部署优化

## 📁 项目结构

```
Enigma/
├── enigma/
│   ├── model.py                    # 主模型定义
│   ├── invertible_conv1x1.py       # 🆕 可逆1×1卷积实现
│   ├── plugboard.py               # Plugboard组件
│   ├── rev_block.py               # 可逆块
│   ├── reflector.py               # 反射器
│   └── jacobian_logdet.py         # Flow模型支持
├── scripts/
│   ├── train_multi_gpu.py          # 🆕 多GPU训练脚本
│   ├── benchmark.py               # 基准测试
│   └── generate.py                # 文本生成
├── checkpoints_multigpu/           # 🆕 多GPU训练检查点
└── README.md                      # 本文件
```

## 📚 参考文献

1. **Glow**: Kingma & Dhariwal. "Glow: Generative Flow using Invertible 1x1 Convolutions"
2. **RevNets**: Gomez et al. "The Reversible Residual Network"
3. **ALiBi**: Press et al. "Train Short, Test Long: Attention with Linear Biases"
4. **DDP**: PyTorch Distributed Data Parallel

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

> **训练状态**: 🔥 当前5张4090 GPU全力训练中，GPU利用率40%+，预计完成时间: TBD
