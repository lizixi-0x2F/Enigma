import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='清理检查点，只保留最佳模型和分词器')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints_large', help='检查点目录')
    parser.add_argument('--best-model-name', type=str, default='best_model.pt', help='最佳模型文件名')
    parser.add_argument('--tokenizer-name', type=str, default='tokenizer.pkl', help='分词器文件名')
    parser.add_argument('--output-dir', type=str, default='checkpoints_best', help='输出目录')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不实际删除文件')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 检查最佳模型和分词器是否存在
    best_model_path = os.path.join(args.checkpoints_dir, args.best_model_name)
    tokenizer_path = os.path.join(args.checkpoints_dir, args.tokenizer_name)
    
    if not os.path.exists(best_model_path):
        print(f"错误：最佳模型 {best_model_path} 不存在")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"错误：分词器 {tokenizer_path} 不存在")
        return
    
    # 复制最佳模型和分词器到输出目录
    best_model_output = os.path.join(args.output_dir, args.best_model_name)
    tokenizer_output = os.path.join(args.output_dir, args.tokenizer_name)
    
    if args.dry_run:
        print(f"[试运行] 将复制 {best_model_path} 到 {best_model_output}")
        print(f"[试运行] 将复制 {tokenizer_path} 到 {tokenizer_output}")
    else:
        print(f"复制 {best_model_path} 到 {best_model_output}")
        shutil.copy2(best_model_path, best_model_output)
        
        print(f"复制 {tokenizer_path} 到 {tokenizer_output}")
        shutil.copy2(tokenizer_path, tokenizer_output)
    
    # 获取检查点目录中的所有文件
    all_files = os.listdir(args.checkpoints_dir)
    checkpoint_files = [f for f in all_files if f.endswith('.pt') and f != args.best_model_name]
    
    # 显示要删除的文件
    print(f"找到 {len(checkpoint_files)} 个检查点文件需要删除")
    
    if args.dry_run:
        for file in checkpoint_files:
            print(f"[试运行] 将删除 {os.path.join(args.checkpoints_dir, file)}")
    else:
        for file in checkpoint_files:
            file_path = os.path.join(args.checkpoints_dir, file)
            print(f"删除 {file_path}")
            os.remove(file_path)
    
    print("完成！")
    print(f"保留的文件：")
    print(f"- {best_model_output}")
    print(f"- {tokenizer_output}")

if __name__ == "__main__":
    main() 