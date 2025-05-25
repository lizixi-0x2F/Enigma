[English](./README_EN.md)

# Enigma 语言模型

> “如果你不能用简单的语言解释，那就说明你还不够理解它。” — 理查德·费曼

> **座右铭：**
> “雄关漫道真如铁；
> 而今迈步从头越。
> 从头越，苍山如海,
> 残阳如血”

基于可逆神经网络的语言模型，支持全序列预训练与超高效 LoRA 微调，驱动于字符级分词，秉承费曼的清晰思考精神。

## 🚀 快速开始

### 1. 环境搭建

```bash
pip install -r requirements.txt
```

### 2. 构建完整分词器

从 `sft_data.jsonl` 提取原始文本，训练字符级分词器：

```bash
python build_full_tokenizer.py
```

分词器文件将保存在 `enigma_tokenizer/` 目录下。

### 3. 预训练

使用 \~90 万样本进行全序列预训练：

```bash
accelerate launch --config_file accelerate_config.yaml pretrain.py
```

**配置要点：**

* **数据集：** `am_0.9M_processed_hf/`（900K 样本，HF 格式）
* **分词器：** 字符级，位于 `enigma_tokenizer/`
* **序列长度：** 2048（不截断）
* **GPU：** 5×4090 并行
* **总 batch size：** 5 卡 × 4/卡 = 20（可累积至 32）
* **学习率：** 3e-4
* **最大步数：** 5000
* **早停：** 验证 PPL ≤ 25

### 4. LoRA 指令微调

```bash
accelerate launch --config_file accelerate_config.yaml lora_sft.py
```

**LoRA-SFT 配置：**

* **数据集：** `sft_data.jsonl`（instruction + input → output）
* **基座模型：** `output_full_sequence/final_model`
* **分词器：** `enigma_tokenizer/`
* **序列长度：** 512（微调上下文）
* **GPU：** 5×4090 并行
* **batch：** 4/卡 × 累积 8 = 32
* **学习率：** 1e-4
* **LoRA rank：** 16，alpha：32（缩放因子=2）
* **最大步数：** 1000
* **早停：** loss <0.1 或 PPL ≤25

**LoRA 挂载层：** Q/K/V/O 投影、前馈网络 FC1/FC2、回归头

### 5. 聊天测试

```bash
python enigma_chat.py
```

使用自定义分词器进行交互式对话。

## 📊 监控训练

* **TensorBoard：**

  ```bash
  tensorboard --logdir output_full_sequence/runs  
  tensorboard --logdir output_lora_sft/runs
  ```
* **GPU 使用率：**

  ```bash
  nvidia-smi -l 3
  ```

## 📁 项目结构

```text
Enigma/
├─ enigma/                 # 核心模型代码
│  ├─ modeling_enigma.py   # EnigmaForCausalLM 主模型
│  ├─ attention.py         # Transformer 注意力层
│  ├─ rev_block.py         # 可逆块实现
│  ├─ rotor.py             # Enigma 转子逻辑
│  └─ ...
├─ pretrain.py             # 预训练脚本
├─ lora_sft.py             # LoRA 微调脚本
├─ enigma_chat.py          # 聊天测试脚本
├─ build_full_tokenizer.py # 分词器构建脚本
├─ enigma_tokenizer/       # 分词器文件
├─ output_full_sequence/   # 预训练输出
│  └─ final_model/         # 最终预训练模型
├─ output_lora_sft/        # LoRA 微调输出
│  └─ lora_adapter/        # LoRA 适配器权重
├─ am_0.9M_processed_hf/   # 预训练数据（HF 缓存）
├─ sft_data.jsonl          # SFT 数据 (JSONL)
├─ accelerate_config.yaml  # 分布式配置
└─ config.json             # Enigma 模型配置
```

## 🧩 架构概述

Enigma 模型由以下几个关键模块组成：

- **字符嵌入层**：将字符映射到高维向量空间，保留原始字符信息
- **可逆 Enigma 块**：基于可逆神经网络和可逆卷积，实现特征的可逆变换与高效记忆
- **Transformer 注意力层**：多头自注意力机制，用于捕捉全局上下文依赖
- **前馈网络层 (FC1/FC2)**：两层全连接网络，提供非线性表征能力
- **语言模型输出层 (LM Head)**：线性映射到词汇表大小，用于下一个字符预测
- **LoRA 微调**：在注意力投影层和前馈网络层注入低秩适配器，实现高效参数微调

此外，模型采用 **ALiBi** 位置编码，支持最长 2048 长度序列训练；参数量约 **50M**，并可通过梯度检查点进一步降低显存开销。

## 🔧 核心特性

* **字符级分词器**：\~11K 词汇，多语言支持
* **可逆架构**：RevBlock + 可逆卷积 + ALiBi
* **全序列支持**：最大 2048 tokens
* **LoRA 微调**：仅训练 <6% 参数
* **FP16 + 梯度检查点**：显存友好
* **早停机制**：基于 PPL 自动终止
* **分布式训练**：Multi-GPU 支持

## 📖 理念

启发自费曼对"清晰"的执着，Enigma 将复杂的可逆网络拆解成可理解的模块，逐块验证、逐层构建。正如费曼会逐步拆解机器原理，我们也一步步构建语言智能——一个可逆块，一个 LoRA 适配器，直到每段代码都如清泉般可追溯。

> "唯有你能清晰解释的事物，你才真正掌握。"

让我们一起，从曾经的铁壁密码，到今日的智能对话，迈步从头越，共探 AI 世界的新征程！
