import pandas as pd
import sys
import os
import glob

def examine_dataset(data_dir, num_samples=5):
    """
    检查数据集的结构和内容
    
    参数:
        data_dir: 数据集目录路径
        num_samples: 要显示的样本数量
    """
    print(f"检查数据集: {data_dir}")
    
    # 获取所有parquet文件
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not parquet_files:
        print(f"错误: 在 {data_dir} 中没有找到parquet文件")
        return
    
    print(f"找到 {len(parquet_files)} 个parquet文件:")
    for file_path in parquet_files:
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
        print(f"  - {os.path.basename(file_path)}: {file_size:.2f} MB")
    
    # 读取第一个文件的元数据
    print("\n检查第一个文件的结构...")
    first_file = parquet_files[0]
    try:
        df = pd.read_parquet(first_file)
        print(f"行数: {len(df)}")
        print(f"列: {df.columns.tolist()}")
        print(f"数据类型:\n{df.dtypes}")
        
        # 显示一些统计信息
        if 'text' in df.columns:
            text_lengths = df['text'].str.len()
            print(f"\n文本长度统计:")
            print(f"  最小长度: {text_lengths.min()}")
            print(f"  最大长度: {text_lengths.max()}")
            print(f"  平均长度: {text_lengths.mean():.2f}")
            print(f"  中位数长度: {text_lengths.median()}")
        
        # 显示样本数据
        print(f"\n显示 {num_samples} 个样本:")
        for i, row in df.head(num_samples).iterrows():
            if 'text' in df.columns:
                text = row['text']
                # 限制显示长度，避免输出过长
                display_text = text[:500] + "..." if len(text) > 500 else text
                print(f"\n样本 {i}:")
                print(f"{display_text}")
            else:
                print(f"\n样本 {i}:")
                print(row)
        
        # 估算整个数据集的大小
        total_rows = len(df)
        for file_path in parquet_files[1:]:
            df_info = pd.read_parquet(file_path, columns=[])
            total_rows += len(df_info)
        
        print(f"\n估计整个数据集包含 {total_rows} 个样本")
        
    except Exception as e:
        print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    data_dir = "wiki-full-zh"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    num_samples = 3
    if len(sys.argv) > 2:
        num_samples = int(sys.argv[2])
    
    examine_dataset(data_dir, num_samples) 