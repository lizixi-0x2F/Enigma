import os
import sys
import time
import glob
import argparse
from datetime import datetime, timedelta

def get_newest_checkpoint(checkpoint_dir):
    """获取最新的检查点文件"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    if not checkpoint_files:
        return None, None
    
    # 按修改时间排序
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    newest_file = checkpoint_files[0]
    
    # 获取修改时间
    mod_time = datetime.fromtimestamp(os.path.getmtime(newest_file))
    
    return newest_file, mod_time

def get_checkpoint_stats(checkpoint_dir):
    """获取检查点统计信息"""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_*.pt"))
    epoch_files = [f for f in checkpoint_files if "epoch" in f]
    step_files = [f for f in checkpoint_files if "step" in f]
    
    has_best = os.path.exists(os.path.join(checkpoint_dir, "best_model.pt"))
    
    return {
        "total": len(checkpoint_files),
        "epochs": len(epoch_files),
        "steps": len(step_files),
        "has_best": has_best
    }

def monitor_checkpoints(checkpoint_dir, interval=60):
    """监控检查点目录"""
    print(f"开始监控检查点目录: {checkpoint_dir}")
    print("按Ctrl+C停止监控")
    print("-" * 80)
    
    last_file = None
    start_time = datetime.now()
    
    try:
        while True:
            # 获取最新检查点
            newest_file, mod_time = get_newest_checkpoint(checkpoint_dir)
            stats = get_checkpoint_stats(checkpoint_dir)
            
            # 清屏
            if os.name == 'nt':  # Windows
                os.system('cls')
            else:  # Linux/Mac
                os.system('clear')
            
            # 显示基本信息
            print(f"监控目录: {checkpoint_dir}")
            print(f"监控开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"已运行: {str(datetime.now() - start_time).split('.')[0]}")
            print("-" * 80)
            
            # 显示检查点统计
            print(f"检查点总数: {stats['total']}")
            print(f"完成的Epoch数: {stats['epochs']}")
            print(f"保存的Step数: {stats['steps']}")
            print(f"是否有最佳模型: {'是' if stats['has_best'] else '否'}")
            
            # 显示最新检查点信息
            if newest_file:
                elapsed = datetime.now() - mod_time
                print("-" * 80)
                print(f"最新检查点: {os.path.basename(newest_file)}")
                print(f"保存时间: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"距今: {str(elapsed).split('.')[0]}")
                
                # 如果有新检查点，显示提醒
                if newest_file != last_file and last_file is not None:
                    print("\n新检查点已生成!")
                
                last_file = newest_file
            else:
                print("-" * 80)
                print("尚未找到检查点文件")
            
            # 显示优化配置信息
            print("-" * 80)
            print("当前训练配置:")
            print("- 模型尺寸: 768维")
            print("- 批量大小: 32 (有效批量 64，使用梯度累积)")
            print("- RevBlock层数: 6")
            print("- 转子数量: 4")
            print("- Transformer层数: 12")
            print("- 注意力头数: 12")
            print("- 学习率: 1e-3 (带预热)")
            print("- 预热比例: 0.2")
            print("- 权重衰减: 1e-3")
            print("- 使用混合精度: 是")
            print("- 使用ALiBi位置编码: 是")
            print("- 最大序列长度: 8192")
            
            # 等待下一次检查
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n停止监控")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="监控EnigmaLM模型训练进度")
    parser.add_argument("--dir", type=str, default="checkpoints_optimized", help="检查点目录")
    parser.add_argument("--interval", type=int, default=30, help="检查间隔(秒)")
    args = parser.parse_args()
    
    monitor_checkpoints(args.dir, args.interval) 