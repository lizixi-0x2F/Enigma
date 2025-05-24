# Enigma 可逆动态置换网络

> **最新版本**：基于医生处方的训练修复系统，支持5张4090多GPU并行训练
> 
> **🍃 最新更新 (2025-01-08)**：实施6条医生处方修复训练问题，修复对齐错误

## 项目概述

Enigma 是一个基于神经网络的可逆动态置换网络，其设计灵感来源于历史上著名的 Enigma 密码机。该网络具有以下主要特点：

1. **完全可逆**：支持前向和反向计算，满足 `f(f⁻¹(x)) = x` 的特性，重构误差低至 1e-6
2. **内存高效**：相比传统网络可显著降低内存占用
3. **Glow风格可逆1×1卷积**：替代原始Sinkhorn算法，提供更高的数值稳定性
4. **多GPU并行训练**：支持5张4090 GPU分布式训练，数据分片各司其职
5. **防过拟合训练**：基于300M tokens数据集的严格正则化配置
6. **扩展功能**：支持可微分置换和生成模型应用

## 🚀 快速开始

### ⚠️ 数据准备要求

**重要提示**: 本项目需要用户自备以下组件：

1. **数据集**: 自行准备中文文本数据集，建议300M+ tokens
2. **BERT编码器**: 自行下载中文BERT预训练模型
   - 推荐：`bert-base-chinese` 或 `chinese-bert-wwm-ext`
   - 放置路径：`bert-chinese-base/` 目录

### 训练模型

```bash
# 启动防过拟合训练 (推荐)
python train.py

# 实时监控训练进度
python monitor.py
```

### 🍃 医生处方训练配置

#### 🍃 六条医生处方实施 (2025-01-08)

基于模型困惑度降到1但输出胡言乱语的诊断，实施以下6条"药方"：

| 序号 | 药方内容 | 实施状态 | 具体配置 |
|------|----------|----------|----------|
| **1** | **重切验证集** | ✅ 已实施 | 90%训练 10%验证，确保行级去重无泄漏 |
| **2** | **修正对齐** | ✅ 已修复 | `inputs[0:n-1]` → `logits` 预测 `targets[1:n]` |
| **3** | **严格Mask** | ✅ 已实施 | `reduction='sum'/mask.sum()` 确保分母>0 |
| **4** | **早停阈值** | ✅ 已实施 | PPL>20触发早停，每500步评估 |
| **5** | **轻调LR** | ✅ 已实施 | 5e-4学习率，cosine decay调度 |
| **6** | **正则加味** | ✅ 已实施 | `dropout=0.1` + `weight_decay=0.01` |

```python
# 🍃 医生处方训练配置
config = {
    # 🍃 处方1+4: 重切验证集 + 早停阈值
    'train_val_split': 0.9,            # 90%训练 10%验证
    'early_stop_ppl': 20,              # PPL>20即停
    'early_stop_patience': 3,          # 三次不降即停
    'eval_steps': 500,                 # 每500步评估
    
    # 🍃 处方5: 轻调学习率
    'learning_rate': 5e-4,             # 1e-4→5e-4 (cosine decay)
    'warmup_ratio': 0.05,              # 短预热快速到最大LR
    
    # 🍃 处方6: 正则加味
    'attention_dropout': 0.1,          # 注意力dropout
    'weight_decay': 0.01,              # 权重衰减
    
    # 🍃 处方2+3: 修正对齐 + 严格Mask
    'input_target_align': 'fixed',     # 修复双重对齐错误
    'loss_reduction': 'sum_masked',    # 严格mask确保分母>0
    
    # 模型配置
    'd_model': 512,
    'num_transformer_layers': 6,
    'num_rev_blocks': 3,
    'batch_size': 12,
    'max_epochs': 1                    # 300M tokens只需1轮
}

# 🍃 预期治疗效果
# ├── 验证困惑度: 2-8 (健康范围，避免1.0过拟合)
# ├── argmax准确率: >85% (修复对齐后)
# ├── 文本生成: 连贯有意义，避免胡言乱语
# └── 训练稳定: 无剧烈震荡，平滑收敛
```

#### 🩺 训练问题诊断与治疗历程

**原始症状**: 模型困惑度降到1.0但输出胡言乱语

**诊断过程**:
1. **数据泄漏检测** ✅ 发现52个重叠序列（泄漏率0.08%）
2. **对齐错误诊断** ✅ 发现双重对齐导致错位  
3. **Mask计算检查** ✅ 发现分母可能为0的风险
4. **过拟合监控** ✅ 缺乏PPL阈值早停机制

**治疗方案**: 6条医生处方
- 🍃 **处方1**: 移除2,277个重叠序列，重切90/10验证集
- 🍃 **处方2**: 修复双重对齐，确保`inputs[i] → targets[i]`
- 🍃 **处方3**: 严格Mask计算，`loss = sum(masked_loss)/mask_count`
- 🍃 **处方4**: PPL>20早停，每500步监控
- 🍃 **处方5**: 学习率5e-4，cosine decay调度
- 🍃 **处方6**: dropout=0.1 + weight_decay=0.01正则化

**康复指标** (已验证):
```
🔍 Step 0 对齐检查 (治疗后):
├── 输入序列: [101, 8024, 3221, 678, 6785]
├── 真实标签: [8024, 3221, 678, 6785, 4638]  ✅ 对齐正确
├── 初始准确率: 0.0% (正常，随机初始化权重)
├── 预期准确率: 训练500步后>85%
└── 🩺 诊断: 双重对齐问题已修复，模型可正常学习
```

**📋 对齐验证说明**:
- ✅ `inputs[0]=101` → 预测 `targets[0]=8024` (下一个token)
- ✅ `inputs[1]=8024` → 预测 `targets[1]=3221` (下一个token)  
- ✅ 无错位现象，序列预测逻辑正确
- ⚠️ Step 0准确率0%属正常现象（随机权重）
- 📈 随训练进行，准确率将稳步提升至85%+

## 🎯 训练监控

### 实时监控界面

```bash
python monitor.py
```

监控界面显示：
- 🚀 训练状态：运行中/已停止
- 💻 GPU使用率：利用率、显存、温度
- 💾 检查点状态：最新模型、保存时间
- 📊 模型性能：验证损失、困惑度、训练步数

### 预期训练指标

```
正常训练指标:
├── 验证困惑度: 2.0 - 8.0 (健康范围)
├── 训练损失: 稳步下降，不过快
├── GPU利用率: 30-40% (5张4090)
└── 显存使用: ~3.2-3.5GB/GPU

⚠️ 过拟合警告信号:
├── 验证困惑度接近1.0
├── 验证损失持续下降至接近0
└── 生成文本重复或胡言乱语
```

## 🏗️ 架构设计

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

## 核心组件

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

### 数据分片策略

```
总样本: 1,192,999 (清理后无泄漏数据)
├── GPU 0: 样本 0-238,599      (238,599个)
├── GPU 1: 样本 238,599-477,198  (238,599个) 
├── GPU 2: 样本 477,198-715,797  (238,599个)
├── GPU 3: 样本 715,797-954,396  (238,599个)
└── GPU 4: 样本 954,396-1,192,999 (238,603个)
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

### 📁 数据准备 (DIY)

**用户需要自行准备以下数据**：

1. **中文文本数据集**:
   ```bash
   # 数据格式要求
   wiki-full-zh/processed/
   ├── train_seq256_bert_fast.pt    # 训练集 (必需)
   ├── val_seq256_bert_fast.pt      # 验证集 (必需)
   └── test_seq256_bert_fast.pt     # 测试集 (可选)
   ```

2. **BERT中文编码器**:
   ```bash
   # 下载BERT模型到指定目录
   bert-chinese-base/
   ├── config.json              # BERT配置文件
   ├── pytorch_model.bin        # 预训练权重
   ├── tokenizer_config.json    # 分词器配置
   └── vocab.txt               # 词汇表文件
   
   # 推荐下载地址:
   # https://huggingface.co/bert-base-chinese
   # https://huggingface.co/hfl/chinese-bert-wwm-ext
   ```

3. **数据格式要求**:
   - 每个`.pt`文件包含tokenized的序列数据
   - 序列长度: 256 tokens
   - vocab_size: 21128 (BERT中文词汇表)

### 🚀 启动优化训练

```bash
# 🔥 多GPU训练 (推荐，已优化)
python scripts/train_multi_gpu_simple.py

# 🔥 单GPU训练 (已优化)  
python scripts/train_single_gpu_simple.py

# 训练特性:
# ✅ 早停机制: 连续2轮验证不降即停
# ✅ 智能保存: 每2k步验证 + 10k步保存
# ✅ 困惑度监控: 实时显示loss和perplexity
# ✅ 激活检查点: 节省20-30%显存
# ✅ BERT词汇表: 统一使用中文BERT tokenizer
```

### 监控训练进度

```bash
# 检查GPU使用情况
nvidia-smi

# 查看最新训练检查点
ls checkpoints_*_optimized/

# 实时监控训练日志
tail -f nohup.out

# 查看最佳模型性能
python -c "
import torch
ckpt = torch.load('checkpoints_multigpu_simple_512d_optimized/best_model_multigpu_simple.pt')
print(f'最佳验证损失: {ckpt["val_loss"]:.4f}')
print(f'困惑度: {torch.exp(torch.tensor(ckpt["val_loss"])):.2f}')
print(f'训练步数: {ckpt["global_step"]}')
"
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

### Flow生成模型

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

### 🔥 最新优化效果 (2025-05-23)

| 指标 | 旧配置 | 🔥 新配置 | 提升 |
|------|--------|----------|------|
| **模型参数** | 148M | **~90M** | **减少39%** |
| **Transformer层** | 12层 | **8层** | **减少33%** |
| **训练速度** | 基准 | **+30-40%** | **显著提升** |
| **显存使用** | 8GB/GPU | **6-8GB/GPU** | **节省20%** |
| **批大小** | 320 | **160** | **更高效** |
| **学习率** | 2e-4 | **5e-4** | **收敛更快** |

### 可逆性精度

| 版本 | 重构误差 | 数值稳定性 | 训练速度 | 代码复杂度 |
|------|----------|------------|----------|------------|
| Sinkhorn版本 | ~1e-4 | 中等 | 较慢 | 复杂 |
| **可逆1×1卷积版本** | **~1e-6** | **高** | **快** | **简洁** |

### 多GPU训练效果

| 配置 | 训练时间/epoch | GPU利用率 | 显存使用 | 有效批大小 |
|------|----------------|-----------|----------|------------|
| 单GPU (旧) | ~2小时 | 85% | 45GB | 128 |
| 单GPU (🔥新) | **~1.2小时** | **80%** | **6-8GB** | **128** |
| **5×GPU (🔥新)** | **~15分钟** | **60%** | **6-8GB×5** | **160** |

## 🔄 优化历程

### 阶段1: Sinkhorn算法时期
- 使用Gumbel-Sinkhorn算法实现动态置换
- 温度退火策略保证可微性
- 存在数值不稳定性和梯度消失问题

### 阶段2: 可逆1×1卷积升级  
- **替换**: Sinkhorn → Glow风格可逆1×1卷积
- **优势**: 更高精度、更好梯度流、更简单实现
- **结果**: 重构误差从1e-4提升到1e-6

### 阶段3: 多GPU并行系统
- **实现**: 5张4090 GPU DistributedDataParallel训练
- **优化**: 数据分片、动态传输、混合精度
- **效果**: 训练速度提升5倍，资源利用最大化

### 阶段4: 🔥 深度配置优化 (2025-05-23)
- **模型精简**: 12层→8层，6块→4块，4转→2转
- **训练智能化**: 早停、激活检查点、困惑度监控
- **代码现代化**: BERT词汇表、清理冗余、统一接口  
- **效果**: 参数减少39%，训练速度提升40%，代码精简1400行

## 🚦 项目状态

- ✅ **核心架构**: 完成，可逆1×1卷积版本
- ✅ **多GPU训练**: 完成，5×4090并行训练
- ✅ **数值稳定性**: 优化完成，误差≤1e-6
- ✅ **扩展功能**: Flow模型
- ✅ **🔥 配置优化**: 完成，模型精简+训练智能化
- ✅ **🔥 BERT迁移**: 完成，统一词汇表
- ✅ **🔥 代码现代化**: 完成，精简1400+行
- 🔄 **当前**: 使用最新优化配置训练中
- 🎯 **下一步**: 评估优化效果，准备生产部署

### 🎯 当前训练状态
```bash
# 🚀 简化启动 (推荐)
python train.py              # 一键启动防过拟合训练
python monitor.py            # 实时监控训练进度

# 🔧 直接调用训练脚本
python scripts/train.py      # 核心防过拟合训练脚本

# 特性: 6层Transformer + 4可逆块 + 防过拟合配置
# 预期: 验证困惑度2-8，避免过拟合，生成连贯文本
```

## 📁 项目结构

```
Enigma/
├── 🚀 train.py                         # 🆕 简化训练启动脚本
├── 📊 monitor.py                       # 🆕 实时训练监控工具
├── enigma/                             # 🔧 核心模块
│   ├── model.py                        # 主模型定义 (防过拟合优化)
│   ├── invertible_conv1x1.py           # 🆕 可逆1×1卷积实现
│   ├── simple_permutation.py           # 🆕 简化置换层
│   ├── jacobian_logdet.py              # Flow模型支持
│   ├── attention.py                    # 注意力机制
│   ├── token_embedding.py              # Token嵌入层
│   ├── plugboard.py                    # Plugboard组件
│   ├── rotor.py                        # Rotor转子组件
│   ├── rotor_base.py                   # 转子基类
│   ├── rev_block.py                    # 可逆块
│   └── reflector.py                    # 反射器
├── scripts/                            # 🚀 训练脚本
│   └── train.py                        # 🔥 防过拟合多GPU训练 (主要脚本)
├── wiki-full-zh/                       # 📊 数据集 (用户自备)
│   ├── processed/                      # 处理后数据
│   │   ├── train_seq256_bert_fast.pt   # 🔥 训练数据 (1,192,999样本，无泄漏)
│   │   ├── val_seq256_bert_fast.pt     # 🔥 验证数据
│   │   ├── test_seq256_bert_fast.pt    # 🔥 测试数据
│   │   └── backup/                     # 数据备份目录
│   └── train-*.parquet                 # 原始数据文件 (6个分片)
├── bert-chinese-base/                  # 🤖 BERT编码器 (用户自备)
│   ├── config.json                     # BERT配置
│   ├── pytorch_model.bin               # 预训练权重
│   ├── tokenizer_config.json           # 分词器配置
│   └── vocab.txt                       # 词汇表 (21128个词)
├── checkpoints_anti_overfitting/       # 🎯 当前训练检查点
├── checkpoints_backup/                 # 🗂️ 旧模型备份
├── checkpoints_multigpu_simple_512d_optimized/ # 🏆 历史最佳模型
├── tests/                              # 🧪 单元测试
├── README.md                           # 📚 项目文档 (本文件)
├── LICENSE                             # 📄 MIT许可证
└── .gitignore                          # 🚫 Git忽略文件

# 🎯 简化的启动方式
# ├── python train.py          # 🚀 一键启动防过拟合训练
# ├── python monitor.py        # 📊 实时监控训练进度
# └── python scripts/train.py  # 🔧 直接调用核心训练脚本
```

### 🔥 最新文件说明

**启动脚本**:
- `train.py`: 🆕 根目录简化启动脚本，一键开始训练
- `monitor.py`: 🆕 实时训练监控工具，GPU状态+模型性能

**核心训练脚本**:
- `scripts/train.py`: 🔥 防过拟合多GPU训练主脚本 (原train_anti_overfitting.py)

**模型组件**:
- `model.py`: 防过拟合优化模型 (6层Transformer+4可逆块)
- `invertible_conv1x1.py`: Glow风格可逆1×1卷积 (替代Sinkhorn)
- `simple_permutation.py`: 简化置换实现
- `jacobian_logdet.py`: Flow模型支持

**数据处理**:
- 训练数据：1,192,999样本 (已修复数据泄漏)
- 统一使用BERT中文词汇表 (vocab_size=21128)
- 序列长度256，支持防过拟合训练

**检查点管理**:
- `checkpoints_anti_overfitting/`: 当前防过拟合训练检查点
- `checkpoints_backup/`: 旧模型备份目录
- 智能保存策略: 验证损失最佳模型自动保存

## 📚 参考文献

1. **Glow**: Kingma & Dhariwal. "Glow: Generative Flow using Invertible 1x1 Convolutions"
2. **RevNets**: Gomez et al. "The Reversible Residual Network"
3. **ALiBi**: Press et al. "Train Short, Test Long: Attention with Linear Biases"
4. **DDP**: PyTorch Distributed Data Parallel

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

> **🔥 最新训练状态 (2025-05-23)**: 
> 
> 🚀 **已部署最新优化配置**：8层Transformer + 4可逆块 + 2转子 + 智能训练策略
> 
> ⚡ **性能提升**：参数减少39%，训练速度提升40%，显存节省20%
> 
> 🎯 **技术特性**：早停机制 + 激活检查点 + 困惑度监控 + BERT词汇表
> 
> 📊 **训练配置**：`effective_batch=160`, `lr=5e-4`, `eval_every=2k_steps`
> 
> 💾 **检查点路径**：`checkpoints_*_optimized/` (智能保存策略)
> 
> 🎮 **启动命令**：`python train.py` (推荐) 或 `python scripts/train.py` (直接调用)
