import torch
import argparse
import os
import pickle
from tqdm import tqdm
from enigma.model import EnigmaLM

def load_tokenizer(tokenizer_path):
    """加载分词器"""
    with open(tokenizer_path, 'rb') as f:
        return pickle.load(f)

def load_model(model_path, config, device='cuda'):
    """加载模型"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    model = EnigmaLM(
        vocab_size=config['vocab_size'],
        d=config['d_model'],
        num_rev_blocks=config['num_rev_blocks'],
        num_rotors=config['num_rotors'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        max_len=config['max_len'],
        use_alibi=True
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功从 {model_path} 加载模型")
    
    return model

def generate_text(model, tokenizer, prompt, max_tokens=2000, temperature=0.7, 
                 top_k=50, top_p=0.9, device='cuda', print_progress=True):
    """生成长文本"""
    model.eval()
    
    # 编码提示文本
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    # 生成文本
    generated_tokens = []
    
    with torch.no_grad():
        # 输入提示
        current_input = input_ids
        
        # 生成序列
        progress_bar = tqdm(range(max_tokens), desc="生成文本") if print_progress else range(max_tokens)
        for _ in progress_bar:
            # 如果序列长度超过模型支持的最大长度，裁剪输入
            if current_input.size(1) > model.max_len:
                current_input = current_input[:, -model.max_len:]
            
            # 获取预测
            logits = model(current_input)
            
            # 获取最后一个token的预测
            next_token_logits = logits[:, -1, :].squeeze(0)
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
                
            # 应用top-k过滤
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
                
            # 应用top-p过滤
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
            if hasattr(tokenizer, 'eos_token_id') and next_token.item() == tokenizer.eos_token_id:
                break
            if hasattr(tokenizer, 'sep_token_id') and next_token.item() == tokenizer.sep_token_id:
                break
    
    # 解码生成的token序列
    generated_text = tokenizer.decode(generated_tokens)
    
    # 返回原始提示和生成的文本
    return prompt + generated_text

def parse_args():
    parser = argparse.ArgumentParser(description='使用EnigmaLM生成长文本')
    parser.add_argument('--model-path', type=str, default='checkpoints_large/best_model.pt', help='模型检查点路径')
    parser.add_argument('--tokenizer-path', type=str, default='checkpoints_large/tokenizer.pkl', help='分词器路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--max-tokens', type=int, default=2000, help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top-k', type=int, default=50, help='top-k过滤')
    parser.add_argument('--top-p', type=float, default=0.9, help='top-p过滤')
    parser.add_argument('--d-model', type=int, default=768, help='模型维度')
    parser.add_argument('--num-rev-blocks', type=int, default=6, help='RevBlock层数')
    parser.add_argument('--num-rotors', type=int, default=4, help='转子数量')
    parser.add_argument('--num-transformer-layers', type=int, default=12, help='Transformer层数')
    parser.add_argument('--num-heads', type=int, default=12, help='注意力头数')
    parser.add_argument('--max-len', type=int, default=8192, help='模型支持的最大序列长度')
    parser.add_argument('--prompt', type=str, default='中国历史悠久，文化灿烂。', help='生成的起始提示')
    parser.add_argument('--output-file', type=str, default=None, help='输出文件路径，不指定则输出到控制台')
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
        'num_heads': args.num_heads,
        'max_len': args.max_len
    }
    
    # 加载模型
    model = load_model(args.model_path, config, args.device)
    
    # 生成文本
    print(f"使用提示: {args.prompt}")
    print(f"生成最大token数: {args.max_tokens}")
    print("开始生成...")
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=args.device
    )
    
    # 输出结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        print(f"生成的文本已保存到: {args.output_file}")
    else:
        print("\n" + "="*50)
        print("生成的文本:")
        print("="*50)
        print(generated_text)
        print("="*50)

if __name__ == "__main__":
    main() 