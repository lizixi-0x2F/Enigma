#!/usr/bin/env python
import os
import torch
import pickle
from tqdm import tqdm
import pandas as pd
import glob

def main():
    """创建并保存处理好的样本，以供训练使用"""
    # 参数设置
    data_dir = "wiki-full-zh"
    seq_len = 256
    max_samples = 200000  # 限制样本数量
    tokenizer_path = "tokenizer/tokenizer.pkl"
    
    # 加载分词器
    print(f"从 {tokenizer_path} 加载分词器...")
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # 确保processed目录存在
    processed_dir = os.path.join(data_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    processed_samples_path = os.path.join(processed_dir, f"processed_samples_seq{seq_len}.pt")
    
    # 加载数据
    files = sorted(glob.glob(f"{data_dir}/*.parquet"))
    
    if len(files) == 0:
        raise ValueError(f"找不到parquet文件: {data_dir}/*.parquet")
    
    # 进度条显示加载进度
    samples = []
    with tqdm(total=len(files), desc="加载数据") as pbar:
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
    
    # 分词和截断为固定长度的序列
    print("对文本进行分词...")
    pad_token_id = tokenizer.pad_token_id
    
    for text in tqdm(text_samples):
        # 编码文本
        tokens = tokenizer.encode(text)
        
        # 处理长文本，分割为多个固定长度的序列
        for i in range(0, len(tokens), seq_len):
            chunk = tokens[i:i + seq_len]
            
            # 如果序列长度不足，则填充
            if len(chunk) < seq_len:
                chunk = chunk + [pad_token_id] * (seq_len - len(chunk))
            
            samples.append(torch.tensor(chunk, dtype=torch.long))
    
    print(f"创建了 {len(samples)} 个训练样本")
    
    # 保存处理好的样本
    print(f"保存样本到 {processed_samples_path}...")
    torch.save(samples, processed_samples_path)
    print("完成!")

if __name__ == "__main__":
    main() 