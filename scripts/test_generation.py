import torch
import argparse
import pickle
import os
from enigma.model import EnigmaLM

class BertChineseTokenizer:
    """BERT中文分词器包装类"""
    
    def __init__(self, model_name='bert-base-chinese'):
        from transformers import BertTokenizer
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


def generate_text(model, tokenizer, prompt, max_tokens=100, temperature=0.7, top_k=50, top_p=0.9, device='cuda'):
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
            eos_token_id = getattr(tokenizer, 'eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = getattr(tokenizer, 'sep_token_id', None)
            
            if eos_token_id is not None and next_token.item() == eos_token_id:
                break
    
    # 解码生成的token序列
    generated_text = tokenizer.decode(input_ids[0].tolist() + generated_tokens, skip_special_tokens=True)
    
    # 移除提示部分，只保留生成的内容
    if prompt in generated_text:
        generated_text = generated_text[len(prompt):]
    
    return generated_text.replace(' ', '')


def parse_args():
    parser = argparse.ArgumentParser(description='测试EnigmaLM模型的文本生成')
    parser.add_argument('--model-path', type=str, default='checkpoints_best/best_model.pt', help='模型检查点路径')
    parser.add_argument('--tokenizer-path', type=str, default='checkpoints_best/tokenizer.pkl', help='分词器路径')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='设备')
    parser.add_argument('--max-tokens', type=int, default=100, help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top-k', type=int, default=50, help='top-k过滤')
    parser.add_argument('--top-p', type=float, default=0.9, help='top-p过滤')
    parser.add_argument('--num-samples', type=int, default=3, help='每个提示生成的样本数')
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
    checkpoint = torch.load(args.model_path, map_location=args.device)
    model = EnigmaLM(
        vocab_size=config['vocab_size'],
        d=config['d_model'],
        num_rev_blocks=config['num_rev_blocks'],
        num_rotors=config['num_rotors'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads']
    ).to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功从 {args.model_path} 加载模型")
    
    # 定义要测试的提示词
    prompts = [
        "中国的历史",
        "人工智能",
        "计算机科学",
        "自然语言处理",
        "机器学习算法",
        "深度学习模型",
        "智能机器人",
        "大数据分析",
        "神经网络",
        "人类与机器"
    ]
    
    # 对每个提示词生成多个样本
    for prompt in prompts:
        print(f"\n提示: {prompt}")
        print("-" * 50)
        
        for i in range(args.num_samples):
            generated_text = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device
            )
            
            print(f"样本 {i+1}: {generated_text}")
        
        print("-" * 50)


if __name__ == "__main__":
    main() 