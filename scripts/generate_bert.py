import torch
import pickle
import argparse
import os
from enigma.model import EnigmaLM
import numpy as np
import time
from tqdm import tqdm
from transformers import BertTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description='使用EnigmaLM模型和BERT分词器生成中文文本')
    parser.add_argument('--model-path', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--tokenizer-path', type=str, default=None, help='分词器路径(pickle文件)')
    parser.add_argument('--bert-tokenizer', type=str, default='bert-base-chinese', 
                       help='BERT分词器名称(若未提供tokenizer-path)')
    parser.add_argument('--prompt', type=str, default='', help='生成文本的提示')
    parser.add_argument('--num-samples', type=int, default=5, help='生成的样本数量')
    parser.add_argument('--max-tokens', type=int, default=200, help='生成的最大token数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='使用的设备')
    parser.add_argument('--temperature', type=float, default=0.8, help='采样温度')
    parser.add_argument('--top-k', type=int, default=50, help='top-k采样')
    parser.add_argument('--top-p', type=float, default=0.95, help='nucleus采样概率阈值')
    parser.add_argument('--repetition-penalty', type=float, default=1.2, 
                        help='重复惩罚系数，>1会降低已生成token的概率')
    parser.add_argument('--no-repeat-ngram-size', type=int, default=3,
                        help='禁止重复的n元组大小，0表示不使用')
    return parser.parse_args()

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ 对logits应用top-k和/或top-p过滤 """
    top_k = min(top_k, logits.size(-1))  # 安全检查
    
    if top_k > 0:
        # 移除所有不在top-k中的token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # 对概率进行排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将第一个token移出移除列表
        sorted_indices_to_remove[..., 0] = 0

        # 回溯到原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits

def apply_repetition_penalty(logits, generated_tokens, penalty=1.0):
    """ 应用重复惩罚 """
    if len(generated_tokens) == 0 or penalty == 1.0:
        return logits
    
    # 获取已生成token的logits
    for token in set(generated_tokens):
        idx = token
        # 对于已生成的token，减少其概率
        if logits[idx] > 0:
            logits[idx] /= penalty
        else:
            logits[idx] *= penalty
    
    return logits

def no_repeat_ngram_blocking(generated_tokens, logits, n_gram=3, filter_value=-float('Inf')):
    """阻止生成重复的n元组"""
    if len(generated_tokens) < n_gram or n_gram <= 0:
        return logits
    
    # 当前(n-1)元组
    current_ngram_prefix = tuple(generated_tokens[-(n_gram-1):])
    
    # 查找所有已经出现过的n元组
    for i in range(len(generated_tokens) - n_gram + 1):
        # 如果找到与当前(n-1)元组相同的前缀
        if tuple(generated_tokens[i:i+n_gram-1]) == current_ngram_prefix:
            # 禁止生成相应的下一个token
            next_token = generated_tokens[i+n_gram-1]
            logits[next_token] = filter_value
    
    return logits

class BertChineseTokenizer:
    """BERT中文分词器包装类"""
    
    def __init__(self, model_name='bert-base-chinese'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.cls_token_id  # 使用CLS作为开始标记
        self.eos_token_id = self.tokenizer.sep_token_id  # 使用SEP作为结束标记
        self.vocab_size = len(self.tokenizer)
    
    def encode(self, text):
        """编码文本为token ids"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加CLS和SEP标记
            max_length=512,
            truncation=True
        )
    
    def decode(self, ids):
        """解码token ids为文本"""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

def generate_text(model, tokenizer, prompt='', max_tokens=100, temperature=0.8, 
                 top_k=50, top_p=0.95, repetition_penalty=1.2, 
                 no_repeat_ngram_size=3, device='cuda'):
    """ 生成文本函数 """
    model.eval()
    
    # 处理提示
    if prompt:
        prompt_tokens = tokenizer.encode(prompt)
        context = torch.tensor(prompt_tokens, dtype=torch.long).unsqueeze(0).to(device)
    else:
        # 使用开始标记作为初始上下文
        context = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).to(device)
    
    generated_tokens = []
    past_tokens = context[0].tolist()
    
    # 生成文本
    with torch.no_grad():
        for _ in range(max_tokens):
            # 获取预测
            logits = model(context)
            
            # 取最后一个token的logits
            next_token_logits = logits[0, -1, :].clone()
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits /= temperature
            
            # 应用重复惩罚
            if repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, past_tokens, repetition_penalty
                )
            
            # 应用n元组重复阻止
            if no_repeat_ngram_size > 0:
                next_token_logits = no_repeat_ngram_blocking(
                    past_tokens, next_token_logits, no_repeat_ngram_size
                )
            
            # 应用top-k和top-p过滤
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, top_k=top_k, top_p=top_p
            )
            
            # 采样下一个token
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # 如果生成了结束标记，停止生成
            if next_token == tokenizer.eos_token_id:
                break
            
            # 将新token添加到生成序列
            generated_tokens.append(next_token)
            past_tokens.append(next_token)
            
            # 更新上下文
            context = torch.cat([context, torch.tensor([[next_token]], device=device)], dim=1)
    
    # 解码生成的token序列
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text

def load_tokenizer(args):
    """加载分词器"""
    if args.tokenizer_path:
        print(f"从文件加载分词器: {args.tokenizer_path}")
        with open(args.tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        print(f"初始化BERT分词器: {args.bert_tokenizer}")
        tokenizer = BertChineseTokenizer(args.bert_tokenizer)
    
    return tokenizer

def load_model(args, tokenizer):
    """加载模型"""
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # 获取模型配置
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"使用检查点中的模型配置")
    else:
        # 从模型状态推断配置
        model_state = checkpoint['model_state_dict']
        
        # 这需要与模型定义匹配
        vocab_size = tokenizer.vocab_size
        
        # 判断是否存在嵌入层
        if 'embed.token_emb.weight' in model_state:
            d_model = model_state['embed.token_emb.weight'].size(1)
        else:
            # 尝试从其他层获取维度
            for key, value in model_state.items():
                if 'weight' in key and len(value.size()) == 2:
                    d_model = value.size(1)
                    break
            else:
                d_model = 512  # 默认值
        
        # 检查是否有转子层
        num_rotors = 0
        for key in model_state.keys():
            if 'rotors' in key:
                num_rotors = 2  # 假设有转子
                break
        
        # 计算RevBlock数量
        num_rev_blocks = 0
        for key in model_state.keys():
            if 'enigma_core.rev_blocks' in key and '.0.' in key:  # 计算第一个模块的数量
                num_rev_blocks += 1
        
        # 计算Transformer层数
        num_transformer_layers = 0
        for key in model_state.keys():
            if 'transformers' in key and '.0.' in key:  # 计算第一个Transformer的层数
                num_transformer_layers += 1
        
        # 如果找不到transformer层，尝试其他方式
        if num_transformer_layers == 0:
            for i in range(20):  # 尝试最多20层
                if f'transformer.layers.{i}.self_attn.W_q' in model_state:
                    num_transformer_layers = i + 1
        
        if num_transformer_layers == 0:
            num_transformer_layers = 8  # 默认值
        
        # 计算注意力头数
        num_heads = 16  # 默认值
        for key in model_state.keys():
            if 'self_attn.W_q' in key:
                qkv_dim = model_state[key].size(0)
                num_heads = qkv_dim // d_model
                break
        
        config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'num_rev_blocks': num_rev_blocks or 4,  # 使用默认值，如果没有检测到
            'num_rotors': num_rotors,
            'num_transformer_layers': num_transformer_layers,
            'num_heads': num_heads
        }
        print("从模型状态推断的配置:")
    
    print(f"模型配置: {config}")
    
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
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model

def main():
    args = parse_args()
    
    # 加载分词器
    tokenizer = load_tokenizer(args)
    
    # 加载模型
    model = load_model(args, tokenizer)
    print("模型加载成功")
    
    # 生成多个样本
    print(f"\n生成 {args.num_samples} 个样本，每个最多 {args.max_tokens} 个token")
    print(f"温度: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}, 重复惩罚: {args.repetition_penalty}")
    print(f"禁止重复的n元组大小: {args.no_repeat_ngram_size}")
    
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
            no_repeat_ngram_size=args.no_repeat_ngram_size,
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