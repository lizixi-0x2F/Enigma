#!/usr/bin/env python3
"""
分析wiki-full-zh数据集的规模和处理情况
"""

import pandas as pd
import torch
import os
from pathlib import Path

def analyze_raw_data():
    """分析原始parquet文件"""
    print("🔍 分析原始parquet文件...")
    
    parquet_files = sorted([f for f in os.listdir('wiki-full-zh') if f.endswith('.parquet')])
    total_rows = 0
    total_size_mb = 0
    
    for file in parquet_files:
        file_path = f'wiki-full-zh/{file}'
        df = pd.read_parquet(file_path)
        file_size = os.path.getsize(file_path) / 1024 / 1024
        
        print(f"   {file}: {len(df):,} 行, {file_size:.1f}MB")
        total_rows += len(df)
        total_size_mb += file_size
    
    print(f"📊 原始数据总计: {total_rows:,} 行, {total_size_mb:.1f}MB")
    return total_rows

def analyze_processed_data():
    """分析处理后的数据"""
    print("\n🔍 分析处理后的数据...")
    
    processed_file = 'wiki-full-zh/processed/processed_samples_seq256.pt'
    if os.path.exists(processed_file):
        data = torch.load(processed_file, map_location='cpu')
        file_size = os.path.getsize(processed_file) / 1024 / 1024
        
        print(f"   处理后样本数: {len(data):,}")
        print(f"   样本形状: {data[0].shape}")
        print(f"   数据类型: {data[0].dtype}")
        print(f"   文件大小: {file_size:.1f}MB")
        print(f"   词汇表范围: {data[0].min().item()} - {data[0].max().item()}")
        
        return len(data)
    else:
        print("   ❌ 处理后文件不存在")
        return 0

def check_sample_content():
    """检查样本内容"""
    print("\n🔍 检查样本内容...")
    
    # 检查第一个parquet文件的内容
    df = pd.read_parquet('wiki-full-zh/train-00000-of-00006.parquet')
    print(f"   原始数据列: {list(df.columns)}")
    
    if 'text' in df.columns:
        sample_text = df['text'].iloc[0]
        print(f"   样本文本长度: {len(sample_text)} 字符")
        print(f"   样本文本预览: {sample_text[:200]}...")

def main():
    print("📈 Enigma数据集分析报告")
    print("=" * 50)
    
    # 分析原始数据
    total_raw = analyze_raw_data()
    
    # 分析处理后数据
    total_processed = analyze_processed_data()
    
    # 检查样本内容
    check_sample_content()
    
    # 计算处理比例
    print("\n📋 数据处理总结:")
    if total_raw > 0 and total_processed > 0:
        ratio = (total_processed / total_raw) * 100
        print(f"   处理比例: {ratio:.2f}% ({total_processed:,} / {total_raw:,})")
        print(f"   未处理数据: {total_raw - total_processed:,} 行")
    
    # 给出建议
    print("\n💡 扩展建议:")
    if total_raw > total_processed:
        print(f"   🎯 可以处理更多数据：还有 {total_raw - total_processed:,} 行未处理")
        print(f"   🚀 建议处理全部数据以提升模型性能")
        print(f"   💾 预估全量处理后文件大小: ~{(total_raw / total_processed) * 202:.0f}MB")

if __name__ == "__main__":
    main() 