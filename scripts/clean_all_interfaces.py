#!/usr/bin/env python
import os
import argparse
import shutil
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='清理所有多余的接口文件，只保留优化过的核心文件')
    parser.add_argument('--scripts-dir', type=str, default='scripts', help='脚本目录')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不实际删除文件')
    parser.add_argument('--backup-dir', type=str, default='scripts_backup_all', 
                        help='备份目录，为空则不备份')
    parser.add_argument('--force', action='store_true', help='强制删除，不询问确认')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 要保留的核心文件
    keep_files = [
        # 优化过的新接口
        'inference_service.py',      # 包含Beam Search和Contrastive Search
        'generate_from_latest.py',   # 更新的命令行接口
        'start_api_server.py',       # 启动API服务
        'test_model_loading.py',     # 测试模型加载
        
        # 优化过的训练文件
        'train_large.py',            # 添加了分词器加载功能
        'train_full_data.py',        # 全数据集训练
        
        # 实用工具
        'clean_checkpoints.py',      # 清理检查点
        'clean_interface_files.py',  # 清理接口
        'clean_all_interfaces.py',   # 本脚本
        
        # Python包相关
        '__init__.py',
        '__pycache__'
    ]
    
    # 检查脚本目录是否存在
    if not os.path.exists(args.scripts_dir):
        print(f"错误：脚本目录 {args.scripts_dir} 不存在")
        return
    
    # 获取脚本目录中的所有文件
    all_files = os.listdir(args.scripts_dir)
    
    # 确定要删除的文件
    files_to_delete = []
    for f in all_files:
        # 忽略目录
        if os.path.isdir(os.path.join(args.scripts_dir, f)):
            continue
        # 如果不在保留列表中，则删除
        if f not in keep_files:
            files_to_delete.append(f)
    
    # 显示要删除的文件
    print(f"\n找到 {len(files_to_delete)} 个可删除的接口文件:")
    for f in files_to_delete:
        print(f"  - {f}")
    
    # 显示要保留的文件
    print(f"\n将保留 {len(keep_files)} 个核心接口文件:")
    for f in keep_files:
        if f in all_files or os.path.isdir(os.path.join(args.scripts_dir, f)):
            print(f"  - {f}")
    
    # 确认是否继续
    if not args.force and not args.dry_run:
        confirm = input("\n确定要删除这些文件吗? [y/N] ")
        if confirm.lower() != 'y':
            print("操作已取消")
            return
    
    # 创建备份目录（如果指定）
    if args.backup_dir:
        backup_dir = args.backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        print(f"\n创建备份目录: {backup_dir}")
    
    # 处理每个待删除的文件
    print("\n开始处理文件...")
    for filename in files_to_delete:
        file_path = os.path.join(args.scripts_dir, filename)
        
        # 如果指定了备份，先备份文件
        if args.backup_dir:
            backup_path = os.path.join(args.backup_dir, filename)
            if args.dry_run:
                print(f"[试运行] 将复制 {file_path} 到 {backup_path}")
            else:
                print(f"备份 {file_path} 到 {backup_path}")
                shutil.copy2(file_path, backup_path)
        
        # 删除文件
        if args.dry_run:
            print(f"[试运行] 将删除 {file_path}")
        else:
            print(f"删除 {file_path}")
            os.remove(file_path)
    
    print("\n清理完成!")

if __name__ == "__main__":
    main() 