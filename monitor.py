#!/usr/bin/env python3
"""
训练监控脚本
"""

import os
import time
import subprocess

def monitor_training():
    print("🔍 Enigma训练监控")
    print("=" * 50)
    
    while True:
        try:
            # 检查进程
            result = subprocess.run(['pgrep', '-f', 'python train.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                pid = result.stdout.strip()
                print(f"✅ 训练进程运行中 (PID: {pid})")
                
                # 检查检查点目录
                checkpoint_dir = "checkpoints_final"
                if os.path.exists(checkpoint_dir):
                    files = os.listdir(checkpoint_dir)
                    if files:
                        print(f"📁 检查点文件: {len(files)} 个")
                        for f in sorted(files)[-3:]:  # 显示最近3个文件
                            print(f"   - {f}")
                    else:
                        print("⏳ 暂无检查点文件")
                else:
                    print("📁 检查点目录未创建")
                
                # 检查GPU使用率
                try:
                    gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', 
                                               '--format=csv,noheader,nounits'], 
                                              capture_output=True, text=True)
                    if gpu_result.returncode == 0:
                        gpu_usage = gpu_result.stdout.strip()
                        print(f"🎮 GPU使用率: {gpu_usage}%")
                except:
                    print("🎮 GPU信息不可用")
                    
            else:
                print("❌ 训练进程未运行")
                break
                
            print("-" * 30)
            time.sleep(10)  # 每10秒检查一次
            
        except KeyboardInterrupt:
            print("\n👋 监控结束")
            break
        except Exception as e:
            print(f"❌ 监控错误: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_training() 