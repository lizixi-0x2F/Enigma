#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from collections import Counter
from pathlib import Path
import jieba
from transformers import BertTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents

class FullTokenizerBuilder:
    """构建满血tokenizer - 从真实数据中提取语义词汇"""
    
    def __init__(self, vocab_size=21128):
        self.vocab_size = vocab_size
        self.texts = []
        
    def extract_texts_from_sft_data(self, sft_file="sft_data.jsonl"):
        """从SFT数据中提取所有文本"""
        print(f"📖 从 {sft_file} 提取文本...")
        
        texts = []
        with open(sft_file, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 提取instruction, input, output
                    instruction = data.get('instruction', '')
                    input_text = data.get('input', '')
                    output = data.get('output', '')
                    
                    # 清理思维链内容（<think>...</think>）
                    output_clean = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
                    
                    # 收集所有文本
                    if instruction: texts.append(instruction)
                    if input_text: texts.append(input_text)
                    if output_clean: texts.append(output_clean)
                    
                    if line_no % 10000 == 0:
                        print(f"  已处理 {line_no} 行...")
                        
                except Exception as e:
                    print(f"⚠️ 第{line_no}行解析错误: {e}")
                    continue
        
        print(f"✅ 提取完成，共 {len(texts)} 个文本片段")
        self.texts = texts
        return texts
    
    def build_wordpiece_tokenizer(self):
        """构建WordPiece tokenizer（类似BERT）"""
        print("🔧 构建WordPiece tokenizer...")
        
        # 初始化tokenizer
        tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
        
        # 预处理器 - 分割空格和标点
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.Punctuation()
        ])
        
        # 训练器
        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", 
                           "<s>", "</s>", "[INST]", "[/INST]", "<think>", "</think>"],
            min_frequency=2,
            continuing_subword_prefix="##"
        )
        
        # 训练tokenizer
        print("📚 开始训练tokenizer...")
        tokenizer.train_from_iterator(self.texts, trainer)
        
        # 设置解码器
        tokenizer.decoder = decoders.WordPiece()
        
        # 设置后处理器
        tokenizer.post_processor = processors.BertProcessing(
            sep=("[SEP]", tokenizer.token_to_id("[SEP]")),
            cls=("[CLS]", tokenizer.token_to_id("[CLS]"))
        )
        
        print("✅ WordPiece tokenizer构建完成")
        return tokenizer
    
    def build_bpe_tokenizer(self):
        """构建BPE tokenizer"""
        print("🔧 构建BPE tokenizer...")
        
        # 初始化BPE tokenizer
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        
        # 预处理器
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.WhitespaceSplit(),
            pre_tokenizers.ByteLevel(add_prefix_space=False)
        ])
        
        # 训练器
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<unk>", "<s>", "</s>", "[INST]", "[/INST]", 
                           "<think>", "</think>", "<pad>"],
            min_frequency=2
        )
        
        # 训练
        print("📚 开始训练BPE tokenizer...")
        tokenizer.train_from_iterator(self.texts, trainer)
        
        # 解码器
        tokenizer.decoder = decoders.ByteLevel()
        
        print("✅ BPE tokenizer构建完成")
        return tokenizer
    
    def build_char_level_tokenizer(self):
        """构建字符级tokenizer（适合中文）"""
        print("🔧 构建字符级tokenizer...")
        
        # 统计所有字符
        char_counter = Counter()
        for text in self.texts:
            char_counter.update(text)
        
        # 构建词汇表
        special_tokens = ["<unk>", "<s>", "</s>", "[INST]", "[/INST]", 
                         "<think>", "</think>", "<pad>"]
        
        # 取高频字符
        most_common_chars = char_counter.most_common(self.vocab_size - len(special_tokens))
        
        vocab = {}
        # 添加特殊token
        for i, token in enumerate(special_tokens):
            vocab[token] = i
        
        # 添加字符
        for char, freq in most_common_chars:
            if char not in vocab:
                vocab[char] = len(vocab)
        
        print(f"✅ 字符级tokenizer构建完成，词汇量: {len(vocab)}")
        return vocab
    
    def save_tokenizer(self, tokenizer, tokenizer_type, save_dir="enigma_tokenizer"):
        """保存tokenizer"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        if tokenizer_type in ["wordpiece", "bpe"]:
            # 保存HuggingFace格式
            tokenizer.save(str(save_path / "tokenizer.json"))
            
            # 转换为HuggingFace tokenizer
            hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
            hf_tokenizer.save_pretrained(save_path)
            
        elif tokenizer_type == "char":
            # 保存字符级词汇表
            with open(save_path / "vocab.json", 'w', encoding='utf-8') as f:
                json.dump(tokenizer, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Tokenizer已保存到: {save_path}")
        return save_path
    
    def test_tokenizer(self, tokenizer, tokenizer_type):
        """测试tokenizer"""
        print("\n🧪 测试tokenizer...")
        
        test_texts = [
            "你好，我是Enigma",
            "1+1等于几？",
            "[INST] 请解释人工智能 [/INST]",
            "这是一个测试句子，包含中文和English混合内容。"
        ]
        
        for text in test_texts:
            print(f"\n📝 原文: {text}")
            
            if tokenizer_type in ["wordpiece", "bpe"]:
                encoded = tokenizer.encode(text)
                tokens = encoded.tokens
                ids = encoded.ids
                decoded = tokenizer.decode(ids)
                
                print(f"🔢 Token IDs: {ids[:10]}...")  # 只显示前10个
                print(f"🎯 Tokens: {tokens[:10]}...")
                print(f"📖 解码: {decoded}")
                
            elif tokenizer_type == "char":
                ids = [tokenizer.get(char, tokenizer.get("<unk>", 0)) for char in text]
                chars = [k for k, v in tokenizer.items() if v in ids[:10]]
                
                print(f"🔢 Char IDs: {ids[:10]}...")
                print(f"📖 对应字符: {chars}")

def main():
    """主函数"""
    print("🚀 构建Enigma满血Tokenizer")
    print("=" * 60)
    
    builder = FullTokenizerBuilder(vocab_size=21128)
    
    # 1. 提取训练数据文本
    texts = builder.extract_texts_from_sft_data()
    
    if not texts:
        print("❌ 没有提取到文本数据！")
        return
    
    print(f"\n📊 数据统计:")
    print(f"  总文本数: {len(texts)}")
    print(f"  总字符数: {sum(len(text) for text in texts):,}")
    print(f"  平均长度: {sum(len(text) for text in texts) / len(texts):.1f}")
    
    # 2. 构建不同类型的tokenizer
    print("\n" + "=" * 60)
    
    # 选择tokenizer类型
    tokenizer_type = input("选择tokenizer类型 (1: WordPiece, 2: BPE, 3: 字符级) [默认: 3]: ").strip()
    
    if tokenizer_type == "1":
        tokenizer = builder.build_wordpiece_tokenizer()
        save_path = builder.save_tokenizer(tokenizer, "wordpiece")
        builder.test_tokenizer(tokenizer, "wordpiece")
        
    elif tokenizer_type == "2":
        tokenizer = builder.build_bpe_tokenizer()
        save_path = builder.save_tokenizer(tokenizer, "bpe")
        builder.test_tokenizer(tokenizer, "bpe")
        
    else:  # 默认字符级
        tokenizer = builder.build_char_level_tokenizer()
        save_path = builder.save_tokenizer(tokenizer, "char")
        builder.test_tokenizer(tokenizer, "char")
    
    print(f"\n🎉 满血tokenizer构建完成！")
    print(f"📁 保存位置: {save_path}")
    print("💡 现在可以用这个tokenizer重新训练模型或进行推理")

if __name__ == "__main__":
    main() 