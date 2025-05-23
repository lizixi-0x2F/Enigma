#!/usr/bin/env python
import os
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='清理和整理接口文件')
    parser.add_argument('--scripts-dir', type=str, default='scripts', help='脚本目录')
    parser.add_argument('--dry-run', action='store_true', help='试运行，不实际删除文件')
    parser.add_argument('--backup-dir', type=str, default='scripts_backup', 
                        help='备份目录，为空则不备份')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 待删除的文件列表
    files_to_delete = [
        'generate_inference.py',   # 被 inference_service.py 替代
        'generate_bert.py'         # 新接口已整合此功能
    ]
    
    # 检查脚本目录是否存在
    if not os.path.exists(args.scripts_dir):
        print(f"错误：脚本目录 {args.scripts_dir} 不存在")
        return
    
    # 创建备份目录（如果指定）
    if args.backup_dir:
        backup_dir = args.backup_dir
        os.makedirs(backup_dir, exist_ok=True)
        print(f"创建备份目录: {backup_dir}")
    
    # 处理每个待删除的文件
    for filename in files_to_delete:
        file_path = os.path.join(args.scripts_dir, filename)
        
        if os.path.exists(file_path):
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
        else:
            print(f"文件不存在: {file_path}")
    
    # 显示保留的接口文件
    print("\n保留的接口文件:")
    important_files = [
        'inference_service.py',
        'generate_from_latest.py',
        'start_api_server.py',
        'test_model_loading.py'
    ]
    
    for filename in important_files:
        file_path = os.path.join(args.scripts_dir, filename)
        if os.path.exists(file_path):
            print(f"- {filename}")
        else:
            print(f"- {filename} (不存在)")
    
    print("\n清理完成!")

if __name__ == "__main__":
    main() 