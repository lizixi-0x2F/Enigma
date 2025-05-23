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
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge import Rouge
from enigma.model import EnigmaLM

# 确保NLTK资源可用
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


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


class TextDataset(Dataset):
    """使用BERT中文分词器的文本数据集"""
    
    def __init__(self, data_dir, seq_len=256, max_samples=None, tokenizer=None):
        self.seq_len = seq_len
        self.samples = []
        
        print(f"正在加载数据集: {data_dir}")
        
        # 加载数据
        files = sorted(glob.glob(f"{data_dir}/*.parquet"))
        
        if len(files) == 0:
            raise ValueError(f"找不到parquet文件: {data_dir}/*.parquet")
        
        # 进度条显示加载进度
        with tqdm(total=len(files), desc="加载parquet文件") as pbar:
            text_samples = []
            for file in files:
                # 只读取text列，减少内存使用
                df = pd.read_parquet(file, columns=['text'])
                if 'text' in df.columns:
                    text_samples.extend(df['text'].tolist())
                pbar.update(1)
                
                # 如果达到最大样本数，则停止加载
                if max_samples and len(text_samples) >= max_samples:
                    text_samples = text_samples[:max_samples]
                    break
        
        print(f"加载了 {len(text_samples)} 段文本")
        
        # 初始化分词器或使用提供的分词器
        if tokenizer is None:
            print("初始化BERT中文分词器...")
            self.tokenizer = BertChineseTokenizer()
        else:
            self.tokenizer = tokenizer
            
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # 分词和截断为固定长度的序列
        print("对文本进行分词...")
        for text in tqdm(text_samples):
            # 编码文本
            tokens = self.tokenizer.encode(text)
            
            # 处理长文本，分割为多个固定长度的序列
            for i in range(0, len(tokens), seq_len):
                chunk = tokens[i:i + seq_len]
                
                # 如果序列长度不足，则填充
                if len(chunk) < seq_len:
                    chunk = chunk + [self.pad_token_id] * (seq_len - len(chunk))
                
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"创建了 {len(self.samples)} 个训练样本")
    
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


def calculate_perplexity(model, dataset, device, pad_token_id, batch_size=4):
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


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.7, 
                 top_k=50, top_p=0.9, device='cuda'):
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
                
            # 应用top-k和top-p过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过阈值的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 0] = 0  # 保留最高概率的token
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=0, index=sorted_indices, src=sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = -float('Inf')
                
            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 添加到生成序列
            generated_tokens.append(next_token.item())
            
            # 更新输入
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
            
            # 如果生成了EOS token，就停止生成
            if next_token.item() == tokenizer.sep_token_id:
                break
    
    # 解码生成的token序列
    generated_text = tokenizer.decode(input_ids[0].tolist() + generated_tokens, skip_special_tokens=True)
    # 移除提示部分，只保留生成的内容
    if prompt in generated_text:
        generated_text = generated_text[len(prompt):]
    
    return generated_text.replace(' ', '')


def calculate_bleu(references, candidates):
    """计算BLEU分数"""
    # 分词
    tokenized_refs = [[nltk.word_tokenize(ref)] for ref in references]
    tokenized_cands = [nltk.word_tokenize(cand) for cand in candidates]
    
    # 计算BLEU分数
    smoothing = SmoothingFunction().method1
    bleu1 = corpus_bleu(tokenized_refs, tokenized_cands, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = corpus_bleu(tokenized_refs, tokenized_cands, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu4 = corpus_bleu(tokenized_refs, tokenized_cands, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return {
        'bleu-1': bleu1,
        'bleu-2': bleu2,
        'bleu-4': bleu4
    }


def calculate_rouge(references, candidates):
    """计算ROUGE分数"""
    rouge = Rouge()
    
    # 确保输入非空
    valid_pairs = [(ref, cand) for ref, cand in zip(references, candidates) if ref.strip() and cand.strip()]
    
    if not valid_pairs:
        return {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}
    
    refs, cands = zip(*valid_pairs)
    
    try:
        scores = rouge.get_scores(cands, refs, avg=True)
        return {
            'rouge-1': scores['rouge-1']['f'],
            'rouge-2': scores['rouge-2']['f'],
            'rouge-l': scores['rouge-l']['f']
        }
    except Exception as e:
        print(f"计算ROUGE分数时出错: {e}")
        return {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0}


def parse_args():
    parser = argparse.ArgumentParser(description='评估EnigmaLM模型在验证集上的性能')
    parser.add_argument('--model-path', type=str, default='checkpoints_best/best_model.pt', help='模型检查点路径')
    parser.add_argument('--tokenizer-path', type=str, default='checkpoints_best/tokenizer.pkl', help='分词器路径')
    parser.add_argument('--data-dir', type=str, default='wiki-full-zh', help='数据集目录')
    parser.add_argument('--max-samples', type=int, default=1000, help='最大样本数量')
    parser.add_argument('--seq-len', type=int, default=256, help='序列长度')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--batch-size', type=int, default=4, help='批量大小')
    parser.add_argument('--num-gen-samples', type=int, default=10, help='生成样本的数量')
    parser.add_argument('--gen-max-tokens', type=int, default=100, help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
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
    
    # 加载验证数据集
    print("加载验证数据集...")
    dataset = TextDataset(args.data_dir, seq_len=args.seq_len, max_samples=args.max_samples, tokenizer=tokenizer)
    
    # 随机分割出验证集 (使用5%的数据作为验证集)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"验证集大小: {len(val_dataset)}")
    
    # 计算困惑度
    print("\n计算困惑度...")
    perplexity, loss = calculate_perplexity(model, val_dataset, args.device, tokenizer.pad_token_id, args.batch_size)
    print(f"验证集困惑度 (Perplexity): {perplexity:.4f}")
    print(f"验证集损失: {loss:.4f}")
    
    # 生成一些样本文本以进行BLEU/ROUGE评估
    print("\n生成样本文本进行BLEU/ROUGE评估...")
    
    references = []
    generated_texts = []
    
    # 从验证集中选择一些样本作为参考
    for i in range(min(args.num_gen_samples, len(val_dataset))):
        # 获取原始样本文本
        sample_tokens = val_dataset[i].tolist()
        reference_text = tokenizer.decode(sample_tokens, skip_special_tokens=True).replace(' ', '')
        
        # 只使用前20%的文本作为提示
        prompt_length = max(10, int(len(reference_text) * 0.2))
        prompt = reference_text[:prompt_length]
        
        # 生成文本
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.gen_max_tokens,
            temperature=args.temperature,
            device=args.device
        )
        
        # 参考文本是原始文本除了提示的部分
        reference = reference_text[prompt_length:]
        
        references.append(reference)
        generated_texts.append(generated_text)
        
        print(f"\n样本 {i+1}:")
        print(f"提示: {prompt}")
        print(f"参考: {reference[:100]}..." if len(reference) > 100 else f"参考: {reference}")
        print(f"生成: {generated_text[:100]}..." if len(generated_text) > 100 else f"生成: {generated_text}")
    
    # 计算BLEU分数
    if references and generated_texts:
        print("\n计算BLEU分数...")
        bleu_scores = calculate_bleu(references, generated_texts)
        print(f"BLEU-1: {bleu_scores['bleu-1']:.4f}")
        print(f"BLEU-2: {bleu_scores['bleu-2']:.4f}")
        print(f"BLEU-4: {bleu_scores['bleu-4']:.4f}")
        
        # 计算ROUGE分数
        print("\n计算ROUGE分数...")
        rouge_scores = calculate_rouge(references, generated_texts)
        print(f"ROUGE-1: {rouge_scores['rouge-1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge-2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rouge-l']:.4f}")
    else:
        print("没有足够的样本来计算BLEU/ROUGE分数")


if __name__ == "__main__":
    main() 