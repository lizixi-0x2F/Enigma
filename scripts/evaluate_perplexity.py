import torch
import torch.nn as nn
import argparse
import os
import pickle
import numpy as np
import math
import pandas as pd
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from enigma.model import EnigmaLM


class BertChineseTokenizer:
    """BERT中文分词器包装类"""
    
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.cls_token_id  # 使用CLS作为开始标记
        self.eos_token_id = self.tokenizer.sep_token_id  # 使用SEP作为结束标记
        self.vocab_size = len(self.tokenizer)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    
    def encode(self, text, add_special_tokens=True):
        """编码文本为token ids"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=512,
            truncation=True
        )
    
    def decode(self, ids, skip_special_tokens=True):
        """解码token ids为文本"""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


class SimpleDataset(Dataset):
    """使用少量样本进行评估的简化数据集"""
    
    def __init__(self, tokenizer, seq_len=256, num_samples=50):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples = []
        
        # 创建一些简单的测试样本
        test_prompts = [
            "中国历史悠久，文化灿烂。",
            "人工智能技术正在快速发展。",
            "自然语言处理是计算机科学的重要分支。",
            "深度学习模型可以处理各种复杂任务。",
            "互联网改变了人们的生活方式。"
        ]
        
        for prompt in test_prompts:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) < seq_len:
                tokens = tokens + [self.tokenizer.pad_token_id] * (seq_len - len(tokens))
            else:
                tokens = tokens[:seq_len]
            self.samples.append(torch.tensor(tokens, dtype=torch.long))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return torch.stack(batch, dim=0)


def load_tokenizer(tokenizer_path):
    """加载分词器"""
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"成功从 {tokenizer_path} 加载分词器")
        return tokenizer
    except Exception as e:
        print(f"加载分词器时出错: {e}")
        raise


def load_model(model_path, config, device='cuda'):
    """加载模型"""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 创建模型
        model = EnigmaLM(
            vocab_size=config['vocab_size'],
            d=config['d_model'],
            num_rev_blocks=config['num_rev_blocks'],
            num_rotors=config['num_rotors'],
            num_transformer_layers=config['num_transformer_layers'],
            num_heads=config['num_heads']
        ).to(device)
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功从 {model_path} 加载模型")
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        raise


def calculate_perplexity(model, dataset, device, pad_token_id, batch_size=2):
    """计算模型在数据集上的困惑度"""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='none')
    
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="计算困惑度"):
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            
            logits = model(inputs)
            
            # 计算每个token的损失
            losses = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            # 创建掩码，忽略padding token
            mask = (targets != pad_token_id).reshape(-1)
            
            # 计算非padding token的损失总和和数量
            total_loss += losses[mask].sum().item()
            total_tokens += mask.sum().item()
    
    # 计算平均损失
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    # 计算困惑度
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss


def generate_text(model, tokenizer, prompt, max_tokens=50, temperature=0.7, device='cuda'):
    """生成文本"""
    model.eval()
    
    # 编码提示文本
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # 生成文本
    generated_tokens = []
    
    with torch.no_grad():
        # 输入提示
        current_input = input_ids
        
        # 生成序列
        for _ in range(max_tokens):
            # 获取预测
            logits = model(current_input)
            
            # 获取最后一个token的预测
            next_token_logits = logits[:, -1, :].squeeze(0)
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 添加到生成序列
            generated_tokens.append(next_token.item())
            
            # 更新输入
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
            
            # 如果生成了EOS token，就停止生成
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = getattr(tokenizer, 'sep_token_id', None)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    
    # 解码生成的token序列
    generated_text = tokenizer.decode(input_ids[0].tolist() + generated_tokens, skip_special_tokens=True)
    
    return generated_text.replace(' ', '')


def parse_args():
    parser = argparse.ArgumentParser(description='评估EnigmaLM模型的困惑度')
    parser.add_argument('--model-path', type=str, default='checkpoints_best/best_model.pt', help='模型检查点路径')
    parser.add_argument('--tokenizer-path', type=str, default='checkpoints_best/tokenizer.pkl', help='分词器路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--d-model', type=int, default=768, help='模型维度')
    parser.add_argument('--num-rev-blocks', type=int, default=6, help='RevBlock层数')
    parser.add_argument('--num-rotors', type=int, default=4, help='转子数量')
    parser.add_argument('--num-transformer-layers', type=int, default=12, help='Transformer层数')
    parser.add_argument('--num-heads', type=int, default=12, help='注意力头数')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载分词器
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # 模型配置
    config = {
        'vocab_size': tokenizer.vocab_size,
        'd_model': args.d_model,
        'num_rev_blocks': args.num_rev_blocks,
        'num_rotors': args.num_rotors,
        'num_transformer_layers': args.num_transformer_layers,
        'num_heads': args.num_heads
    }
    
    # 加载模型
    model = load_model(args.model_path, config, args.device)
    
    # 创建简单数据集进行评估
    dataset = SimpleDataset(tokenizer)
    
    # 计算困惑度
    print("\n计算困惑度...")
    perplexity, loss = calculate_perplexity(model, dataset, args.device, tokenizer.pad_token_id)
    print(f"困惑度 (Perplexity): {perplexity:.4f}")
    print(f"平均损失: {loss:.4f}")
    
    # 生成一些样本文本
    print("\n生成样本文本:")
    test_prompts = [
        "中国的四大发明",
        "人工智能的发展",
        "计算机科学的应用"
    ]
    
    for i, prompt in enumerate(test_prompts):
        generated_text = generate_text(model, tokenizer, prompt, device=args.device)
        print(f"\n样本 {i+1}:")
        print(f"提示: {prompt}")
        print(f"生成: {generated_text}")


if __name__ == "__main__":
    main() 