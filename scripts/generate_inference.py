import torch
import pickle
import argparse
import os
from enigma.model import EnigmaLM
import numpy as np
import time
from tqdm import tqdm
from transformers import BertTokenizer

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

def parse_args():
    parser = argparse.ArgumentParser(description='使用EnigmaLM模型生成文本')
    parser.add_argument('--model-path', type=str, default='checkpoints_best/best_model.pt', help='模型检查点路径')
    parser.add_argument('--tokenizer-path', type=str, default='checkpoints_best/tokenizer.pkl', help='分词器路径')
    parser.add_argument('--prompt', type=str, default='中国的历史', help='生成文本的提示')
    parser.add_argument('--num-samples', type=int, default=3, help='生成的样本数量')
    parser.add_argument('--max-tokens', type=int, default=200, help='生成的最大token数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='使用的设备')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--top-k', type=int, default=50, help='top-k采样')
    parser.add_argument('--top-p', type=float, default=0.95, help='nucleus采样概率阈值')
    parser.add_argument('--repetition-penalty', type=float, default=1.2, 
                        help='重复惩罚系数，>1会降低已生成token的概率')
    return parser.parse_args()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """对logits进行top-k和top-p过滤"""
    top_k = min(top_k, logits.size(-1))  # 安全检查
    
    if top_k > 0:
        # 移除所有不在top k的token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        # 计算累积概率分布
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将第一个token保留（避免全部被过滤）
        sorted_indices_to_remove[..., 0] = 0

        # 恢复原始索引顺序并过滤
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits

def apply_repetition_penalty(next_token_logits, generated_tokens, penalty=1.0):
    """应用重复惩罚"""
    if len(generated_tokens) > 0:
        for token in set(generated_tokens):
            # 如果token在生成的序列中出现过，就降低其概率
            next_token_logits[token] /= penalty
    return next_token_logits

def generate_text(model, tokenizer, prompt='', max_tokens=100, temperature=0.8, 
                 top_k=50, top_p=0.95, repetition_penalty=1.2, device='cuda'):
    """ 生成文本函数 """
    model.eval()
    
    # 编码提示文本
    if prompt:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # 如果没有提示，则使用BOS token开始
        input_ids = torch.tensor([tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else tokenizer.cls_token_id], 
                                 dtype=torch.long, device=device).unsqueeze(0)
    
    # 初始化生成的token列表
    generated_tokens = input_ids[0].tolist()
    
    # 初始化KV缓存
    kv_cache = None
    
    # 开始生成
    with torch.no_grad():
        for _ in tqdm(range(max_tokens), desc="生成中"):
            # 获取模型输出
            outputs = model(input_ids)
            
            # 检查输出格式
            if isinstance(outputs, tuple):
                logits = outputs[0]
                if len(outputs) > 1:
                    kv_cache = outputs[1:]
            else:
                logits = outputs
                kv_cache = None
            
            # 获取最后一个token的logits
            if len(logits.shape) == 3:
                next_token_logits = logits[:, -1, :].squeeze(0)
            else:
                # 如果是二维张量，直接使用
                next_token_logits = logits.squeeze(0)
            
            # 应用重复惩罚
            if repetition_penalty > 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, 
                    generated_tokens, 
                    repetition_penalty
                )
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 应用top-k和top-p过滤
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=top_k, 
                top_p=top_p
            )
            
            # 采样下一个token
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 添加到生成的序列
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # 如果生成了EOS token，就停止生成
            if next_token.item() == (tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.sep_token_id):
                break
    
    # 解码生成的token序列
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 后处理：移除字符之间的空格
    text = raw_text.replace(" ", "")
    return text

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

def main():
    args = parse_args()
    
    # 加载分词器
    tokenizer = load_tokenizer(args.tokenizer_path)
    
    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # 固定模型配置
    config = {
        'vocab_size': 21128,
        'd_model': 512,
        'num_rev_blocks': 4,
        'num_rotors': 2,
        'num_transformer_layers': 8,
        'num_heads': 16
    }
    
    print(f"使用固定的模型配置: {config}")
    
    # 创建模型
    model = EnigmaLM(
        vocab_size=config['vocab_size'],
        d=config['d_model'],
        num_rev_blocks=config['num_rev_blocks'],
        num_rotors=config['num_rotors'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads']
    ).to(args.device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print("模型加载成功")
    
    # 生成多个样本
    print(f"\n生成 {args.num_samples} 个样本，每个最多 {args.max_tokens} 个token")
    print(f"温度: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}, 重复惩罚: {args.repetition_penalty}")
    
    if args.prompt:
        print(f"提示: {args.prompt}")
    
    for i in range(args.num_samples):
        start_time = time.time()
        
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device
        )
        
        generation_time = time.time() - start_time
        
        print(f"\n样本 {i+1}:")
        if args.prompt:
            print(f"提示: {args.prompt}")
        print(f"生成: {generated_text}")
        print(f"生成时间: {generation_time:.2f}秒, 速度: {args.max_tokens/generation_time:.2f} token/秒")
        print("-" * 50)

if __name__ == "__main__":
    main() 