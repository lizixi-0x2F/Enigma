#!/usr/bin/env python3
import os
import time
import glob
import subprocess
import argparse

def get_latest_screen_output(screen_id):
    """获取最新的screen输出"""
    tmp_file = "/tmp/enigma_training_output.txt"
    try:
        subprocess.run(f"screen -S {screen_id} -X hardcopy {tmp_file}", shell=True)
        with open(tmp_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"获取screen输出失败: {str(e)}"

def get_latest_log_tail(num_lines=20):
    """从屏幕输出中获取最新的日志行"""
    screen_id = "enigma_full_train"
    content = get_latest_screen_output(screen_id)
    lines = content.split('\n')
    return '\n'.join(lines[-num_lines:])

def get_gpu_info():
    """获取GPU使用情况"""
    try:
        output = subprocess.check_output("nvidia-smi", shell=True).decode('utf-8')
        return output
    except Exception as e:
        return f"获取GPU信息失败: {str(e)}"

def get_checkpoints_info(checkpoint_dir="checkpoints_large"):
    """获取检查点信息"""
    if not os.path.exists(checkpoint_dir):
        return "检查点目录不存在"
    
    checkpoints = glob.glob(f"{checkpoint_dir}/*.pt")
    if not checkpoints:
        return "暂无检查点文件"
    
    result = []
    result.append(f"检查点总数: {len(checkpoints)}")
    result.append("最新检查点:")
    
    # 按修改时间排序
    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    for cp in checkpoints[:3]:  # 显示最新的3个检查点
        size_mb = os.path.getsize(cp) / 1024 / 1024
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(cp)))
        result.append(f"  {cp} ({size_mb:.2f} MB, {mtime})")
    
    return '\n'.join(result)

def main():
    parser = argparse.ArgumentParser(description="监控EnigmaLM训练进度")
    parser.add_argument("--interval", type=int, default=30, help="更新间隔(秒)")
    parser.add_argument("--lines", type=int, default=20, help="显示的日志行数")
    args = parser.parse_args()
    
    try:
        while True:
            os.system('clear')
            print("=" * 80)
            print("【EnigmaLM训练监控】")
            print("=" * 80)
            
            # 显示GPU信息
            print("\n【GPU使用情况】")
            print(get_gpu_info())
            
            # 显示检查点信息
            print("\n【检查点信息】")
            print(get_checkpoints_info())
            
            # 显示训练日志
            print(f"\n【最新训练日志 (最后 {args.lines} 行)】")
            print(get_latest_log_tail(args.lines))
            
            print("\n" + "=" * 80)
            print(f"监控更新于: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"按 Ctrl+C 退出监控")
            print("=" * 80)
            
            time.sleep(args.interval)
    
    except KeyboardInterrupt:
        print("\n退出监控")

if __name__ == "__main__":
    main() 