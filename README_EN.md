[简体中文](./README_SC.md)

# Enigma Language Model

> “If you can't explain it simply, you don't understand it well enough.” — Richard Feynman

> **Motto:**
> "The mighty pass stretches on, solid as iron;
> Yet now we press forward, starting anew to climb.
> From the start we climb, mountains like seas,
> The dying sun bleeds in crimson skies."

A reversible neural network–based language model that supports full-sequence pretraining and ultra-efficient LoRA fine-tuning, driven by a character-level tokenizer and inspired by Feynman's spirit of clear thinking.

## 🚀 Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Build the Full-Sequence Tokenizer

Extract raw text from `sft_data.jsonl` to train a character-level tokenizer:

```bash
python build_full_tokenizer.py
```

This will output the full tokenizer under `enigma_tokenizer/`.

### 3. Pretraining

Pretrain on \~900K samples with full-sequence inputs:

```bash
accelerate launch --config_file accelerate_config.yaml pretrain.py
```

**Configuration highlights:**

* **Dataset:** `am_0.9M_processed_hf/` (900K samples, HF format)
* **Tokenizer:** character-level, `enigma_tokenizer/`
* **Sequence length:** 2048 (no truncation)
* **GPUs:** 5×4090 distributed
* **Effective batch size:** 8 GPUs × 4 per\_device = 32
* **Learning rate:** 3e-4
* **Max steps:** 5000
* **Early-stopping:** PPL ≤ 25

### 4. LoRA Supervised Fine-Tuning

```bash
accelerate launch --config_file accelerate_config.yaml lora_sft.py
```

**LoRA-SFT settings:**

* **Dataset:** `sft_data.jsonl` (instruction + input → output)
* **Base model:** `output_full_sequence/final_model`
* **Tokenizer:** `enigma_tokenizer/`
* **Sequence length:** 512 (SFT context)
* **GPUs:** 5×4090
* **Batch:** 4 per\_device × 8 accum = 32
* **Learning rate:** 1e-4
* **LoRA rank:** 16, alpha: 32 (scaling=2)
* **Max steps:** 1000
* **Early-stopping:** loss <0.1 OR PPL ≤25

**LoRA injection**: Q/K/V/O projections, FFN FC1/FC2, LM head

### 5. Chat Ready

```bash
python enigma_chat.py
```

Interact with Enigma using your custom tokenizer.

## 📊 Monitoring

* **TensorBoard:**

  ```bash
  tensorboard --logdir output_full_sequence/runs
  tensorboard --logdir output_lora_sft/runs
  ```
* **GPU util:**

  ```bash
  nvidia-smi -l 3
  ```

## 📁 Project Structure

```text
Enigma/
├─ enigma/                 # Core model code
│  ├─ modeling_enigma.py   # EnigmaForCausalLM
│  ├─ attention.py         # Transformer attention
│  ├─ rev_block.py         # Reversible block
│  ├─ rotor.py             # Enigma rotor logic
│  └─ ...
├─ pretrain.py             # Pretraining script
├─ lora_sft.py             # LoRA fine-tuning script
├─ enigma_chat.py          # Interactive chat client
├─ build_full_tokenizer.py # Tokenizer builder
├─ enigma_tokenizer/       # Tokenizer files
├─ output_full_sequence/   # Pretrain outputs
│  └─ final_model/
├─ output_lora_sft/        # SFT outputs
│  └─ lora_adapter/
├─ am_0.9M_processed_hf/   # Pretrain data (HF cache)
├─ sft_data.jsonl          # SFT data (JSONL)
├─ accelerate_config.yaml  # Distributed config
└─ config.json             # Enigma model config
```

## 🧩 Architecture Overview

The Enigma model consists of the following key modules:

- **Character Embedding**: maps each character into a high-dimensional vector space, preserving raw input information
- **Reversible Enigma Block**: uses reversible neural networks and invertible convolutions to perform efficient, invertible feature transformations
- **Transformer Attention Layers**: multi-head self-attention mechanism for capturing global context dependencies
- **Feed-Forward Networks (FC1/FC2)**: two-layer fully connected networks providing non-linear representation power
- **Language Modeling Head (LM Head)**: linear projection to the vocabulary size for next-token prediction
- **LoRA Fine-Tuning**: injects low-rank adapters into attention projection and feed-forward layers for efficient parameter-efficient tuning

Additionally, Enigma adopts **ALiBi** positional encoding to support sequences up to 2048 tokens, has approximately **50M** parameters, and can leverage gradient checkpointing to reduce memory footprint.

## 🔧 Core Features

* **Character-level tokenizer** (\~11K vocab, multi-language support)
* **Reversible architecture** (RevBlock + dynamic conv + ALiBi)
* **Full-sequence support** (max 2048 tokens)
* **LoRA fine-tuning** (<6% params trained)
* **FP16 & gradient checkpointing** for memory efficiency
* **Early-stopping** based on validation PPL
* **Distributed training** across multiple GPUs

## 📖 Philosophy

Inspired by Richard Feynman's relentless pursuit of clarity, Enigma transforms complex reversible networks into an accessible, efficient, and transparent language model. Just as Feynman would rebuild the machine piece by piece, here we reconstruct language intelligence—one reversible block, one LoRA adapter at a time—so that every step remains understandable and improvable.

> "No one can truly understand a concept until they can explain it in simple words."

Embark on this journey: from the iron gates of encryption to the boundless horizon of human–machine dialogue, Enigma invites you to climb again—*chì là́ng*—and discover new vistas of clarity.
