#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import json
from pathlib import Path

# 添加enigma模块到路径
sys.path.append(str(Path(__file__).parent))

from transformers import PreTrainedTokenizer
from enigma.modeling_enigma import EnigmaConfig, EnigmaForCausalLM

# 🔥 完整Enigma Tokenizer
class EnigmaTokenizer(PreTrainedTokenizer):
    """基于真实数据训练的Enigma Tokenizer"""
    
    def __init__(self, vocab_file="enigma_tokenizer/vocab.json", **kwargs):
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)
        
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        
        # 设置特殊token
        super().__init__(
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            **kwargs
        )
    
    @property
    def vocab_size(self):
        return len(self._vocab)
    
    def _tokenize(self, text):
        """字符级分词"""
        return list(text)
    
    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab.get("<unk>", 0))
    
    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, "<unk>")
    
    def get_vocab(self):
        return self._vocab
    
    def encode(self, text, add_special_tokens=True, max_length=None, truncation=True):
        """编码文本为token ids"""
        tokens = self._tokenize(text)
        
        if add_special_tokens:
            tokens = ["<s>"] + tokens + ["</s>"]
        
        if max_length and truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        token_ids = [self._convert_token_to_id(token) for token in tokens]
        return token_ids
    
    def decode(self, token_ids, skip_special_tokens=True):
        """解码token ids为文本"""
        tokens = [self._convert_id_to_token(id) for id in token_ids]
        
        if skip_special_tokens:
            special_tokens = ["<s>", "</s>", "<pad>", "<unk>"]
            tokens = [token for token in tokens if token not in special_tokens]
        
        return "".join(tokens)

class EnigmaChat:
    def __init__(self, model_path="output_full_sequence/final_model"):
        print("🔧 初始化Enigma聊天系统...")
        
        # 加载tokenizer
        print("📖 加载Enigma Tokenizer...")
        self.tokenizer = EnigmaTokenizer()
        print(f"✅ Tokenizer加载完成，词汇量: {self.tokenizer.vocab_size:,}")
        
        # 加载模型
        print(f"🤖 加载模型: {model_path}")
        self.config = EnigmaConfig.from_pretrained(model_path)
        self.model = EnigmaForCausalLM.from_pretrained(model_path)
        self.model.eval()
        
        # 检查GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"🚀 模型加载完成，设备: {self.device}")
        print(f"📊 模型参数量: {self.model.num_parameters():,}")
        
    def generate_response(self, user_input, max_length=512, temperature=0.8, top_p=0.9):
        """生成回复"""
        # 构建对话格式
        prompt = f"<s>[INST] 你是Enigma\n\n{user_input} [/INST] "
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        print(f"🔤 输入长度: {len(input_ids)} tokens")
        
        # 生成回复
        with torch.no_grad():
            # 简单的贪心解码
            generated_ids = input_ids.copy()
            
            for _ in range(max_length):
                # 获取当前输入
                current_input = torch.tensor([generated_ids]).to(self.device)
                
                # 前向传播
                outputs = self.model(current_input)
                logits = outputs.logits
                
                # 获取下一个token的概率分布
                next_token_logits = logits[0, -1, :]
                
                # 应用温度
                next_token_logits = next_token_logits / temperature
                
                # Top-p采样
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # 移除累积概率超过top_p的token
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # 添加到生成序列
                generated_ids.append(next_token)
                
                # 检查是否生成了结束token
                if next_token == self.tokenizer._convert_token_to_id("</s>"):
                    break
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # 提取回复部分
        if "[/INST]" in generated_text:
            response = generated_text.split("[/INST]", 1)[1].strip()
            if response.endswith("</s>"):
                response = response[:-4].strip()
        else:
            response = generated_text
        
        return response
    
    def chat(self):
        """开始聊天循环"""
        print("\n🎉 Enigma聊天系统已启动！")
        print("💡 输入 'quit' 或 'exit' 退出")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n👤 你: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Enigma正在思考...")
                response = self.generate_response(user_input)
                print(f"🤖 Enigma: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 生成回复时出错: {e}")
                continue

def main():
    """主函数"""
    print("🚀 启动Enigma聊天系统")
    print("=" * 50)
    
    try:
        chat_system = EnigmaChat()
        chat_system.chat()
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 