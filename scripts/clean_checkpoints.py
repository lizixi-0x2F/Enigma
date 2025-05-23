import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='清理检查点，只保留指定的模型和分词器')
    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints_optimized', help='检查点目录')
    parser.add_argument('--keep-models', type=str, nargs='+', default=['best_model.pt', 'final_model.pt'], 
                        help='要保留的模型文件名列表')
    parser.add_argument('--tokenizer-name', type=str, default='tokenizer.pkl', help='分词器文件名')
    parser.add_argument('--output-dir', type=str, default='', 
                        help='输出目录，为空则不复制文件，仅删除不需要的检查点')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不实际删除文件')
    parser.add_argument('--delete-all-checkpoints', action='store_true', 
                        help='删除所有checkpoint_前缀的文件')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 检查要保留的模型是否存在
    model_paths = []
    for model_name in args.keep_models:
        model_path = os.path.join(args.checkpoints_dir, model_name)
        if os.path.exists(model_path):
            model_paths.append((model_name, model_path))
        else:
            print(f"警告：模型 {model_path} 不存在")
    
    if not model_paths:
        print(f"错误：没有找到任何要保留的模型")
        return
    
    # 检查分词器是否存在
    tokenizer_path = os.path.join(args.checkpoints_dir, args.tokenizer_name)
    if not os.path.exists(tokenizer_path):
        print(f"警告：分词器 {tokenizer_path} 不存在")
    
    # 如果指定了输出目录，则复制文件
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 复制要保留的模型到输出目录
        for model_name, model_path in model_paths:
            model_output = os.path.join(args.output_dir, model_name)
            
            if args.dry_run:
                print(f"[试运行] 将复制 {model_path} 到 {model_output}")
            else:
                print(f"复制 {model_path} 到 {model_output}")
                shutil.copy2(model_path, model_output)
        
        # 复制分词器到输出目录
        if os.path.exists(tokenizer_path):
            tokenizer_output = os.path.join(args.output_dir, args.tokenizer_name)
            
            if args.dry_run:
                print(f"[试运行] 将复制 {tokenizer_path} 到 {tokenizer_output}")
            else:
                print(f"复制 {tokenizer_path} 到 {tokenizer_output}")
                shutil.copy2(tokenizer_path, tokenizer_output)
    
    # 获取检查点目录中的所有文件
    all_files = os.listdir(args.checkpoints_dir)
    
    # 确定要删除的文件
    files_to_delete = []
    keep_model_names = [name for name, _ in model_paths]
    
    for f in all_files:
        if f.endswith('.pt') and f not in keep_model_names:
            if args.delete_all_checkpoints:
                # 如果指定了删除所有checkpoint_前缀的文件，不管是epoch还是step
                if f.startswith('checkpoint_'):
                    files_to_delete.append(f)
            else:
                # 否则仅删除checkpoint_step前缀的文件，保留epoch文件
                if f.startswith('checkpoint_step'):
                    files_to_delete.append(f)
    
    # 显示要删除的文件
    print(f"\n找到 {len(files_to_delete)} 个检查点文件需要删除")
    
    if args.dry_run:
        for file in files_to_delete:
            print(f"[试运行] 将删除 {os.path.join(args.checkpoints_dir, file)}")
    else:
        total_size_mb = 0
        for file in files_to_delete:
            file_path = os.path.join(args.checkpoints_dir, file)
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_size_mb += size_mb
            print(f"删除 {file_path} ({size_mb:.2f} MB)")
            os.remove(file_path)
        print(f"\n共释放空间: {total_size_mb/1024:.2f} GB")
    
    print("\n完成！")
    print(f"保留的文件：")
    for model_name, model_path in model_paths:
        print(f"- {model_name}")
    if os.path.exists(tokenizer_path):
        print(f"- {args.tokenizer_name}")

if __name__ == "__main__":
    main() 