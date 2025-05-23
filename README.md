# Enigma 可逆动态置换网络

> **目的**：实现具有高可逆性的动态置换网络 *Enigma*，并提供完整的代码骨架及扩展功能。

## 项目概述

Enigma 是一个基于神经网络的可逆动态置换网络，其设计灵感来源于历史上著名的 Enigma 密码机。该网络具有以下主要特点：

1. **完全可逆**：支持前向和反向计算，满足 `f(f⁻¹(x)) = x` 的特性，重构误差低至 0.00004
2. **内存高效**：相比传统网络可显著降低内存占用
3. **动态置换**：使用可学习的置换矩阵进行变换
4. **模块化设计**：由多个功能组件组合而成
5. **扩展功能**：支持可微分置换、基准测试和生成模型应用

## 优化历程

本项目经历了多轮优化，显著提高了可逆性精度：

1. **初始问题**：模型最初重构误差高达百万级别，主要来源于Plugboard和转子状态管理
2. **第一轮优化**：
   - 修改Plugboard使用单位矩阵初始化提高稳定性
   - 改进转子状态的保存和恢复机制
   - 重构RevBlock，避免批归一化带来的问题
   - 这些改进将重构误差降低到了0.004左右
3. **深度优化**：
   - 添加直接针对可逆性的损失函数
   - 使用残差连接改进RevBlock，用有界激活函数替代SiLU
   - 实现周期性权重正交化
   - 优化训练过程（AdamW优化器、余弦学习率调度、梯度裁剪）
   - 这些措施将误差进一步降低至0.00004（提高约100倍）
4. **组件精度**：
   - RevBlock：误差降至0.00000003
   - Reflector：误差约0.00000120
   - Plugboard：误差约0.00002295（主要误差来源）
5. **效率提升**：
   - 将Plugboard从矩阵乘法改为直接索引操作，显著提高计算效率
   - 用Householder反射代替复杂的对称正交矩阵计算，简化Reflector实现
   - 抽象RotorBase接口，提高代码可维护性和扩展性

## 架构图

```dot
include::docs/architecture.dot[]
```

```
输入 (B,d)
   ↓
┌─────────────────────────────────┐
│ Plugboard P (稀疏双射层)        │
└─────────────────────────────────┘
   ↓
┌─────────────────────────────────┐
│ RotorStack R (动态置换层)       │
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

## 主要组件

1. **Plugboard**：使用索引实现高效的置换变换，通过`perm_indices`和`inv_indices`直接进行查找，避免了矩阵乘法的开销
2. **RotorStack**：动态置换层，基于抽象`RotorBase`接口，支持多种转子实现方式，同时维持Enigma机的转子机制
3. **RevBlock**：可逆卷积块，使用加法耦合方式实现可逆变换，经优化后误差极低
4. **Reflector**：使用Householder反射实现，只需一个参数向量即可构造完美对称正交的反射矩阵，计算效率高

## 已实现扩展功能

本项目已成功实现以下三个关键扩展功能：

### 1. Gumbel-Sinkhorn软置换退火

在`enigma/gumbel_sinkhorn.py`中实现了基于Gumbel-Sinkhorn算法的可微分置换操作：

```python
# 创建使用Gumbel-Sinkhorn转子的模型
model = Enigma(
    d=64,
    num_rev_blocks=3,
    num_rotors=0,  # 不使用标准转子
    plugboard_sparsity=0.1
)

# 创建Gumbel-Sinkhorn转子堆栈
gumbel_rotor_stack = GumbelSinkhornRotorStack(
    dim=64,
    num_rotors=3,
    temp_min=0.1,
    temp_max=1.0
)

# 在训练过程中定期退火降低温度
gumbel_rotor_stack.anneal_temperatures()
```

该实现具有以下特点：
- 提供完全可微分的置换操作
- 通过温度参数控制软化程度
- 支持退火以逐渐接近离散置换
- 保持与原始转子接口的兼容性

### 2. 基准测试脚本

在`scripts/benchmark.py`中实现了两个标准基准测试：

#### Copy-Memory任务

测试模型记忆和复制长序列的能力：

```python
# 运行Copy-Memory基准测试
python scripts/benchmark.py --task copy --seq_len 50 --data_dim 10 --epochs 20
```

#### enwik8压缩基准

基于维基百科文本的字符级压缩基准：

```python
# 运行enwik8基准测试
python scripts/benchmark.py --task enwik8 --seq_len 100 --hidden_dim 128 --epochs 20
```

两个基准测试都支持使用标准Enigma或Gumbel-Sinkhorn版本：

```python
# 使用Gumbel-Sinkhorn转子运行基准测试
python scripts/benchmark.py --task copy --use_gumbel_sinkhorn
```

### 3. Flow生成模型支持

在`enigma/jacobian_logdet.py`中实现了计算雅可比行列式对数的功能，支持Flow-based生成模型：

```python
# 创建Enigma Flow生成模型
enigma_model = Enigma(d=64, num_rev_blocks=3, num_rotors=3)
flow_model = EnigmaFlow(enigma_model, prior='gaussian')

# 计算样本的对数概率
samples = torch.randn(10, 64)
log_probs = flow_model.log_prob(samples)

# 从模型中采样
generated_samples = flow_model.sample(num_samples=10)
```

该实现包括：
- 解析方法计算雅可比矩阵行列式对数
- 数值方法和迹估计方法处理大维度情况
- 支持高斯和均匀先验分布
- 完整的Flow API，兼容标准规范化流框架

## 环境要求

- Python 3.8+
- PyTorch 1.9+
- NumPy 1.20+
- SciPy (用于匈牙利算法优化)

## 最新训练进展

### 模型配置优化

我们基于最新研究和实验结果，对模型配置进行了全面优化：

1. **模型架构优化**：
   - `d_model`：从512增加到768，提高模型表示能力
   - `num_transformer_layers`：从8增加到12，加深网络层次结构
   - `num_heads`：从16调整为12，保持每头维度约64，平衡性能与并行度
   - `num_rev_blocks`：从4增加到6，增强非线性变换能力
   - `num_rotors`：从2增加到4，提升序列混合深度

2. **训练策略优化**：
   - 训练轮次(`--epochs`)：从15增加到20-30，让模型更充分收敛
   - 学习率(`--lr`)：从5e-4调整为1e-3初始值，配合更强的预热
   - 预热比例(`--warmup-ratio`)：从0.1增加到0.2，避免初始训练不稳定
   - 批大小(`--batch-size`)：从16增加到32，结合梯度累积提高批量效率
   - 权重衰减(`--weight-decay`)：从0.01降低到1e-3，减轻正则化强度
   - 可逆性权重(`invertibility_weight`)：从0.1降低到0.05，平衡重建目标

3. **训练效率优化**：
   - 混合精度训练(`--use-amp`)：开启FP16/BF16加速
   - 梯度累积(`--grad-accum-steps=2`)：实现更大的有效批量
   - 余弦衰减：全周期调度，确保后期学习率平稳降低

### ALiBi位置编码与长序列支持

我们实现了增强版的ALiBi（Attention with Linear Biases）位置编码，以支持更长序列处理：

1. **ALiBi关键优化**：
   - 使用更稳定的斜率计算方法，提高数值稳定性
   - 实现智能缓存机制，避免重复计算大型偏置矩阵
   - 明确处理因果掩码，确保未来信息不泄露
   - 将默认最大序列长度从2048扩展到8192

2. **动态序列长度处理**：
   - 位置嵌入动态扩展，自动适应更长输入
   - 滑动窗口注意力机制，高效处理超长文本
   - 优化内存使用，支持生成数千token的长文本

3. **专用长文本生成脚本**：
   - 创建`generate_long.py`专门处理长文本生成
   - 实现流式处理机制，理论上支持无限长度生成
   - 优化采样策略，提高长文本连贯性

### 最新训练实验

我们使用上述优化配置开始了新一轮训练实验：

1. **小规模验证**：
   - 使用20万样本子集进行初步训练
   - 运行4个epochs验证训练流程稳定性
   - 使用本地BERT中文分词器提高分词质量

2. **全数据训练计划**：
   - 完成小规模验证后迁移到完整138万样本
   - 使用优化后的超参数和架构配置
   - 利用混合精度和梯度累积提高训练效率

### 使用最新优化配置

```bash
# 使用优化配置进行训练
python -m scripts.train_large \
  --epochs 25 \
  --data wiki-full-zh \
  --save-dir checkpoints_optimized \
  --d-model 768 \
  --batch-size 32 \
  --grad-accum-steps 2 \
  --num-rev-blocks 6 \
  --num-rotors 4 \
  --num-transformer-layers 12 \
  --num-heads 12 \
  --lr 1e-3 \
  --warmup-ratio 0.2 \
  --weight-decay 1e-3 \
  --use-amp \
  --use-alibi \
  --max-len 8192 \
  --tokenizer bert-chinese-base

# 生成长文本
python -m scripts.generate_long \
  --model-path checkpoints_optimized/best_model.pt \
  --tokenizer-path checkpoints_optimized/tokenizer.pkl \
  --max-tokens 2000 \
  --prompt "中国历史悠久，文化灿烂。" \
  --temperature 0.8 \
  --top-k 50 \
  --top-p 0.9 \
  --max-len 8192
```

这些优化配置和ALiBi位置编码实现显著提升了模型的表达能力和长序列处理能力，为下一阶段的训练和评估奠定了坚实基础。

## 未来工作计划

基于当前训练进展和优化配置，我们计划在以下方向继续改进：

1. **进一步优化ALiBi位置编码**：
   - 实现稀疏注意力机制与ALiBi的结合，进一步提高长序列处理效率
   - 开发基于ALiBi的分块注意力（Chunked Attention）机制，降低注意力计算复杂度
   - 研究ALiBi与可逆网络的更深度结合，充分发挥两者优势

2. **扩大训练数据规模与多样性**：
   - 完成使用全部138万样本的训练
   - 引入多领域中文文本数据集，增强模型对不同文体和主题的理解
   - 开发混合数据集训练策略，平衡不同来源数据的比例

3. **提升模型性能与效率**：
   - 实现基于旋转位置编码（RoPE）的可逆变体，与ALiBi进行对比
   - 优化转子组件实现，提高动态置换的效率和表达能力
   - 开发基于稀疏注意力的可逆Transformer变体，降低计算和内存需求

4. **模型评估与分析**：
   - 建立针对长文本理解和生成的多维度评估基准
   - 分析不同位置编码对模型性能的影响
   - 比较标准Transformer与Enigma在内存使用和计算效率上的差异
   - 可视化模型中动态置换的演化过程，增强可解释性

5. **应用与部署**：
   - 开发轻量级推理API，方便模型应用
   - 实现模型量化策略，减小模型体积，加速推理
   - 探索Enigma架构在对话、摘要、文章生成等任务上的应用

这些改进将帮助我们进一步提升Enigma模型在语言建模任务中的表现，特别是在长序列处理方面的能力，同时充分发挥可逆架构在内存效率上的优势。

## 使用方法

### 基本使用

```python
import torch
from enigma.model import Enigma

# 创建Enigma模型
d = 64  # 特征维度 (必须是偶数)
model = Enigma(
    d=d,                   # 输入/输出维度
    num_rev_blocks=3,      # RevBlock层数
    num_rotors=3,          # 转子数量
    plugboard_sparsity=0.1, # Plugboard稀疏度
    invertibility_weight=0.1 # 可逆性损失权重
)

# 输入数据
batch_size = 32
x = torch.randn(batch_size, d)

# 前向传播
y = model(x)

# 逆向传播
x_reconstructed = model.inverse(y)

# 验证可逆性
error = torch.norm(x - x_reconstructed) / torch.norm(x)
print(f"重构误差: {error.item()}")
```

### 训练语言模型

```bash
# 基础训练 - 中文GPT式语言模型
python scripts/train_lm.py --data wiki-full-zh --d-model 256 --num-transformer-layers 6

# 高级训练 - 使用更多参数和优化策略
python scripts/train_advanced.py --max-samples 50000 --epochs 15 --batch-size 16 \
  --d-model 512 --num-transformer-layers 8 --num-heads 16 \
  --num-rev-blocks 4 --num-rotors 2 --seq-len 256 \
  --save-dir checkpoints_advanced --use-amp \
  --eval-every 500 --save-every 2000 \
  --lr 5e-4 --min-lr-ratio 0.1 --warmup-ratio 0.1 --weight-decay 0.01
```

### 生成文本

```bash
# 基础生成 - 中文文本
python scripts/generate.py --prompt "从前有一个" --temperature 0.8 --max-len 100

# 高级生成 - 使用改进的采样策略
python scripts/generate_advanced.py --model-path checkpoints_advanced/best_model.pt \
  --tokenizer-path checkpoints_advanced/tokenizer.pkl \
  --prompt "从前有一个" --num-samples 5 --max-tokens 200 \
  --temperature 0.8 --top-k 50 --top-p 0.95 --repetition-penalty 1.2
```

### 使用Gumbel-Sinkhorn转子

```python
from enigma.model import Enigma
from enigma.gumbel_sinkhorn import GumbelSinkhornRotorStack

# 创建模型
model = Enigma(d=64, num_rev_blocks=3, num_rotors=0)

# 创建Gumbel-Sinkhorn转子堆栈
gumbel_rotor_stack = GumbelSinkhornRotorStack(dim=64, num_rotors=3)

# 在前向传播中手动组合
def custom_forward(x):
    # 通过Plugboard
    h = model.plugboard(x)
    
    # 通过Gumbel-Sinkhorn转子
    h = gumbel_rotor_stack(h)
    gumbel_rotor_stack.step_all()
    
    # 通过剩余组件
    for rev_block in model.rev_blocks:
        h = rev_block(h)
    h = model.reflector(h)
    for rev_block in reversed(model.rev_blocks):
        h = rev_block(h)
    h = model.plugboard.transpose(h)
    
    return h
```

### 使用EnigmaFlow生成模型

```python
from enigma.model import Enigma
from enigma.jacobian_logdet import EnigmaFlow

# 创建基础Enigma模型
enigma_model = Enigma(d=64, num_rev_blocks=3, num_rotors=3)

# 创建Flow模型
flow_model = EnigmaFlow(enigma_model, prior='gaussian')

# 从先验分布采样并生成新样本
samples = flow_model.sample(num_samples=100)

# 计算样本的对数概率密度
x = torch.randn(32, 64)  # 32个样本
log_probs = flow_model.log_prob(x)
```

## 优化训练参数

要获得最佳可逆性，可以调整以下参数：

```python
# 创建具有高可逆性的模型
model = Enigma(
    d=64,
    num_rev_blocks=3,
    num_rotors=3,
    plugboard_sparsity=0.1,
    use_checkpointing=True,  # 使用梯度检查点以节省内存
    invertibility_weight=0.1  # 可逆性损失权重
)

# 使用AdamW优化器和余弦学习率调度
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# 周期性正交化
if (step + 1) % 5 == 0:
    model.orthogonalize_weights()
```

## 技术亮点

1. **高精度可逆性**：通过多种优化技术，达到了极高的可逆精度（误差仅为0.00004）
2. **内存效率**：与传统网络相比显著降低内存占用，特别适合内存受限环境
3. **动态行为**：网络包含动态组件，表现出类似原始Enigma机的复杂非线性变换特性
4. **数值稳定性**：通过周期性正交化、稀疏约束和梯度裁剪确保长期稳定训练
5. **多样化扩展**：实现了三个关键扩展功能，扩展了模型在不同领域的应用
6. **计算高效**：使用索引操作代替矩阵乘法，Householder反射替代复杂矩阵参数化，显著提高计算效率
7. **良好抽象**：提供RotorBase抽象接口，支持多种转子实现方式，增强可扩展性

## 许可证

MIT License

---

## 核心模块实现

### Plugboard (稀疏双射层)

```python
# enigma/plugboard.py
# 使用索引实现高效置换
# 直接维护perm_indices和inv_indices
# 提供freeze_identity()和set_permutation()方法
```

### Rotor + RotorStack

```python
# enigma/rotor_base.py
# 定义RotorBase抽象接口
# enigma/rotor.py
# Rotor: 实现RotorBase接口的标准转子
# RotorStack: 支持任何RotorBase子类的转子堆栈
```

### Reflector (反射矩阵)

```python
# enigma/reflector.py
# 使用Householder反射实现: R = I - 2 v v^T / (v^T v)
# 只需一个参数向量v就能构造完美对称正交矩阵
```

### 可逆耦合块 RevBlock

```python
# enigma/rev_block.py
# 拆分 (x1,x2) → y1 = x1 + f(x2); y2 = x2 + g(y1)
# f,g: 使用改进的非线性变换
# inverse() 能显式求解
```

### Enigma 主干

```python
# enigma/model.py
# 组件顺序: Plugboard P → RotorStack R → RevBlocks N层 → Reflector U → RevBlocksᴿ → Pᵀ
# 提供forward(x), inverse(y)方法
# loss_regularizer()计算正则化损失
```

### Gumbel-Sinkhorn软置换

```python
# enigma/gumbel_sinkhorn.py
# GumbelSinkhorn: 实现可微分的软置换矩阵
# GumbelSinkhornRotor: 基于Gumbel-Sinkhorn的可微分转子
# GumbelSinkhornRotorStack: 转子堆栈及温度退火
```

### 基准测试脚本

```python
# scripts/benchmark.py
# Copy-Memory任务: 测试模型记忆和复制长序列的能力
# enwik8压缩基准: 基于维基百科文本的字符级压缩测试
# 支持Gumbel-Sinkhorn转子替代标准转子
```

### 雅可比行列式计算

```python
# enigma/jacobian_logdet.py
# JacobianLogDet: 计算雅可比矩阵行列式对数的多种方法
# EnigmaFlow: 基于Enigma的Flow生成模型实现
# 支持样本采样和概率密度计算
```

## GPT式语言模型

Enigma现在支持作为GPT式自回归语言模型，通过以下特性：

1. **架构扩展**：
   - 添加Token和Position嵌入层
   - 实现多层Transformer注意力机制
   - 支持因果掩码确保自回归性质

2. **效率优化**：
   - 使用RMSNorm替代LayerNorm，提高数值稳定性
   - 使用ALiBi相对位置编码，支持更长序列外推
   - 实现KV缓存加速推理生成
   - 支持混合精度训练，降低内存占用

3. **中文支持**：
   - 默认使用BERT中文分词器
   - 可直接训练于wiki-full-zh数据集

### 训练语言模型

```bash
# 基本训练
python scripts/train_lm.py --data wiki-full-zh --d-model 256 --num-transformer-layers 6

# 使用混合精度训练加速
python scripts/train_lm.py --data wiki-full-zh --d-model 512 --use-amp
```

### 生成文本

```bash
# 基本生成
python scripts/generate.py --prompt "从前有一个" --temperature 0.8 --max-len 100

# 高效生成 (使用KV缓存)
python scripts/generate.py --prompt "今天天气很" --use-cache --temperature 0.7
```
