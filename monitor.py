#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控脚本 - 实时跟踪防过拟合训练进度
"""

import os
import time
import torch
import subprocess
from datetime import datetime

def check_gpu_usage():
    """检查GPU使用情况"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True)
        return result.stdout.strip().split('\n')
    except:
        return ["GPU信息获取失败"]

def check_training_status():
    """检查训练状态"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        if 'train_anti_overfitting' in result.stdout:
            return "✅ 训练正在运行"
        else:
            return "❌ 训练已停止"
    except:
        return "❓ 状态检查失败"

def check_checkpoints():
    """检查检查点状态"""
    checkpoint_dir = 'checkpoints_anti_overfitting'
    if not os.path.exists(checkpoint_dir):
        return "📁 检查点目录还未创建"
    
    files = os.listdir(checkpoint_dir)
    if not files:
        return "📁 检查点目录为空"
    
    # 查找最新的最佳模型
    best_models = [f for f in files if 'best_model' in f]
    checkpoints = [f for f in files if 'checkpoint' in f]
    
    info = []
    if best_models:
        for model in best_models:
            path = os.path.join(checkpoint_dir, model)
            size = os.path.getsize(path) / (1024*1024)  # MB
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            info.append(f"🏆 {model} ({size:.1f}MB, {mtime.strftime('%H:%M:%S')})")
    
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        path = os.path.join(checkpoint_dir, latest_checkpoint)
        size = os.path.getsize(path) / (1024*1024)  # MB
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        info.append(f"💾 {latest_checkpoint} ({size:.1f}MB, {mtime.strftime('%H:%M:%S')})")
    
    return '\n'.join(info) if info else "📁 没有找到有效的检查点"

def load_model_info():
    """加载模型信息"""
    checkpoint_dir = 'checkpoints_anti_overfitting'
    best_model_path = os.path.join(checkpoint_dir, 'best_model_anti_overfitting.pt')
    
    if not os.path.exists(best_model_path):
        return "❓ 最佳模型还未保存"
    
    try:
        checkpoint = torch.load(best_model_path, map_location='cpu')
        val_loss = checkpoint.get('val_loss', 'N/A')
        perplexity = checkpoint.get('perplexity', 'N/A')
        global_step = checkpoint.get('global_step', 'N/A')
        
        history = checkpoint.get('training_history', {})
        train_losses = history.get('training_losses', [])
        val_losses = history.get('validation_losses', [])
        
        info = [
            f"📊 验证损失: {val_loss:.4f}" if isinstance(val_loss, float) else f"📊 验证损失: {val_loss}",
            f"🎯 困惑度: {perplexity:.2f}" if isinstance(perplexity, float) else f"🎯 困惑度: {perplexity}",
            f"📈 训练步数: {global_step}"
        ]
        
        if train_losses and val_losses:
            info.append(f"📉 训练损失趋势: {len(train_losses)} 个点")
            info.append(f"📊 验证损失趋势: {len(val_losses)} 个点")
            
            if len(val_losses) >= 2:
                trend = val_losses[-1] - val_losses[0]
                trend_emoji = "📈" if trend > 0 else "📉"
                info.append(f"{trend_emoji} 验证损失变化: {trend:+.4f}")
        
        return '\n'.join(info)
    except Exception as e:
        return f"❌ 模型信息读取失败: {e}"

def main():
    """主监控循环"""
    print("🔍 EnigmaLM 防过拟合训练监控")
    print("=" * 60)
    
    try:
        while True:
            # 清屏
            os.system('clear' if os.name == 'posix' else 'cls')
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🕐 当前时间: {current_time}")
            print("=" * 60)
            
            # 训练状态
            print("🚀 训练状态:")
            print(f"   {check_training_status()}")
            print()
            
            # GPU状态
            print("💻 GPU使用情况:")
            gpu_info = check_gpu_usage()
            for i, info in enumerate(gpu_info):
                if info.strip():
                    parts = info.split(', ')
                    if len(parts) >= 5:
                        name, util, mem_used, mem_total, temp = parts
                        util_pct = util.strip()
                        mem_pct = int(mem_used) / int(mem_total) * 100
                        print(f"   GPU {i}: {util_pct}% 使用率, {mem_pct:.1f}% 显存 ({mem_used}MB/{mem_total}MB), {temp}°C")
            print()
            
            # 检查点状态
            print("💾 检查点状态:")
            checkpoint_info = check_checkpoints()
            for line in checkpoint_info.split('\n'):
                print(f"   {line}")
            print()
            
            # 模型性能
            print("📊 模型性能:")
            model_info = load_model_info()
            for line in model_info.split('\n'):
                print(f"   {line}")
            print()
            
            print("=" * 60)
            print("💡 预期指标:")
            print("   - 验证困惑度应在 2-8 范围内")
            print("   - 避免困惑度接近 1 (过拟合)")
            print("   - 训练应在 1 epoch 内完成")
            print()
            print("按 Ctrl+C 退出监控")
            
            # 等待30秒
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 监控已退出")
    except Exception as e:
        print(f"\n❌ 监控出错: {e}")

if __name__ == "__main__":
    main() 