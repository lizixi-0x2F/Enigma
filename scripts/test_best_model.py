import torch
import pickle
import argparse
import os
import sys
import time
from enigma.model import EnigmaLM

def parse_args():
    parser = argparse.ArgumentParser(description='测试最新训练的EnigmaLM模型')
    parser.add_argument('--model-path', type=str, default='checkpoints_advanced/best_model.pt', help='模型路径')
    parser.add_argument('--tokenizer-path', type=str, default='checkpoints_advanced/tokenizer.pkl', help='分词器路径')
    parser.add_argument('--prompt', type=str, default='中国历史', help='提示文本')
    parser.add_argument('--max-tokens', type=int, default=200, help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度参数')
    parser.add_argument('--top-k', type=int, default=50, help='top-k采样')
    parser.add_argument('--top-p', type=float, default=0.95, help='top-p(nucleus)采样')
    parser.add_argument('--repetition-penalty', type=float, default=1.2, help='重复惩罚，>1会降低已生成token的概率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
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
        sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将第一个token移出移除列表（确保至少有一个token可选）
        sorted_indices_to_remove[..., 0] = 0

        # 回溯到原始索引
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    
    return logits

def apply_repetition_penalty(logits, generated_tokens, penalty=1.0):
    """ 应用重复惩罚，降低已生成token的概率 """
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

def main():
    args = parse_args()
    
    # 加载分词器
    print(f"加载分词器: {args.tokenizer_path}")
    with open(args.tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # 加载模型权重
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model_state = checkpoint['model_state_dict']
    
    # 获取模型参数
    vocab_size = model_state['embed.token_emb.weight'].size(0)
    d_model = model_state['embed.token_emb.weight'].size(1)
    
    # 计算Transformer层数
    num_transformer_layers = 0
    for key in model_state.keys():
        if key.startswith('transformer_layers.'):
            layer_num = int(key.split('.')[1])
            num_transformer_layers = max(num_transformer_layers, layer_num + 1)
    
    # 计算RevBlock数量
    num_rev_blocks = 0
    for key in model_state.keys():
        if key.startswith('enigma_core.rev_blocks.'):
            layer_num = int(key.split('.')[2])
            num_rev_blocks = max(num_rev_blocks, layer_num + 1)
    
    # 估计其他参数
    num_rotors = 2  # 根据训练参数估计
    num_heads = 16  # 根据训练参数估计
    
    print(f"模型参数: vocab_size={vocab_size}, d_model={d_model}, layers={num_transformer_layers}, rev_blocks={num_rev_blocks}")
    
    # 重新创建模型
    model = EnigmaLM(
        vocab_size=vocab_size,
        d=d_model,
        num_rev_blocks=4,  # 使用训练脚本中的参数
        num_rotors=2,      # 使用训练脚本中的参数
        num_transformer_layers=8,  # 使用训练脚本中的参数
        num_heads=16       # 使用训练脚本中的参数
    ).to(args.device)
    
    # 加载权重
    try:
        model.load_state_dict(model_state)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)
    
    # 生成文本
    print(f"\n生成文本 (提示: '{args.prompt}')")
    print(f"参数: 温度={args.temperature}, top-k={args.top_k}, top-p={args.top_p}, 重复惩罚={args.repetition_penalty}")
    
    # 编码提示
    input_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(args.device)
    
    # 生成
    model.eval()
    output_ids = []
    
    with torch.no_grad():
        # 初始输入
        current_input = input_tensor
        
        # 保存所有生成的token用于重复惩罚
        all_tokens = current_input[0].tolist()
        
        # 开始生成
        start_time = time.time()
        for _ in range(args.max_tokens):
            # 获取预测
            logits = model(current_input)
            next_token_logits = logits[0, -1, :].clone()  # 获取最后一个token的logits
            
            # 应用温度
            if args.temperature != 1.0:
                next_token_logits = next_token_logits / args.temperature
            
            # 应用重复惩罚
            if args.repetition_penalty != 1.0:
                next_token_logits = apply_repetition_penalty(next_token_logits, all_tokens, args.repetition_penalty)
            
            # 应用top-k和top-p过滤
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=args.top_k, 
                top_p=args.top_p
            )
            
            # 采样下一个token
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            # 添加到生成序列
            output_ids.append(next_token)
            all_tokens.append(next_token)
            
            # 更新输入
            current_input = torch.cat([current_input, torch.tensor([[next_token]], device=args.device)], dim=1)
        
        # 计算速度
        generation_time = time.time() - start_time
        tokens_per_second = args.max_tokens / generation_time
    
    # 解码并显示结果
    output_text = tokenizer.decode(output_ids)
    print(f"\n生成结果: {args.prompt}{output_text}")
    print(f"\n生成用时: {generation_time:.2f}秒")
    print(f"生成速度: {tokens_per_second:.2f} tokens/秒")

if __name__ == "__main__":
    main() 