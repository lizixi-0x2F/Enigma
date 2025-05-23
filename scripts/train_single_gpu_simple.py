#!/usr/bin/env python3
"""
简化版单GPU训练脚本 - 使用BERT词汇表
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
import time
import math
from tqdm import tqdm

# 导入模型
import sys
sys.path.append('.')
from enigma.model import EnigmaLM

class SimpleSingleGPUDataset(Dataset):
    """简化版单GPU数据集"""
    
    def __init__(self, data_path):
        print(f"🚀 加载数据: {data_path}")
        
        # 加载数据
        self.samples = torch.load(data_path, map_location='cpu', weights_only=False)
        print(f"✅ 加载完成，总样本数: {len(self.samples)}")
        
        # 使用BERT词汇表设置
        self.vocab_size = 21128  # BERT中文词汇表大小
        self.pad_token_id = 0    # BERT的[PAD] token
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def get_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """余弦学习率调度器"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_single_gpu(config):
    """单GPU训练函数"""
    print(f"🎯 启动单GPU训练")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 使用设备: {device}")
    
    # 加载数据集
    dataset = SimpleSingleGPUDataset(data_path=config['data_path'])
    
    # 数据分片 - 95%训练，5%验证
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"📊 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
    # 创建模型
    model = EnigmaLM(
        vocab_size=dataset.vocab_size,
        d=config['d_model'],
        num_rev_blocks=config['num_rev_blocks'],
        num_rotors=config['num_rotors'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        max_len=2048,
        use_alibi=True,
        use_dynamic_conv1x1=True,
        conv1x1_positions=16
    ).to(device)
    
    # 启用激活检查点以节省显存
    if config.get('use_checkpointing', True):
        from torch.utils.checkpoint import checkpoint
        print("✅ 启用激活检查点以节省显存")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📐 模型参数: {total_params:,}")
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )
    
    # 学习率调度器
    total_steps = (len(train_loader) // config['gradient_accum_steps']) * config['max_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_scheduler(optimizer, warmup_steps, total_steps)
    
    # 其他设置
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    try:
        scaler = GradScaler('cuda')  # 新版本API
    except:
        scaler = GradScaler()  # 旧版本API
    
    print(f"📈 总步数: {total_steps}, 预热步数: {warmup_steps}")
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 训练循环
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']}")
        
        for step, tokens in enumerate(progress_bar):
            tokens = tokens.to(device, non_blocking=True)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            
            # 前向传播 (使用混合精度)
            try:
                with autocast('cuda'):  # 新版本API
                    logits = model(inputs)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = loss / config['gradient_accum_steps']
            except:
                with autocast():  # 旧版本API
                    logits = model(inputs)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    loss = loss / config['gradient_accum_steps']
            
            # 反向传播
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            
            # 参数更新
            if (step + 1) % config['gradient_accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 更新进度条
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f'{loss.item() * config["gradient_accum_steps"]:.4f}',
                    'lr': f'{current_lr:.2e}',
                    'step': global_step
                })
                
                # 验证和保存
                if global_step % config['eval_steps'] == 0:
                    model.eval()
                    val_loss = 0
                    val_steps = 0
                    
                    with torch.no_grad():
                        for val_tokens in val_loader:
                            if val_steps >= 50:  # 限制验证步数
                                break
                            val_tokens = val_tokens.to(device)
                            val_inputs, val_targets = val_tokens[:, :-1], val_tokens[:, 1:]
                            val_logits = model(val_inputs)
                            val_loss += criterion(val_logits.reshape(-1, val_logits.size(-1)), 
                                                val_targets.reshape(-1)).item()
                            val_steps += 1
                    
                    val_loss /= val_steps
                    perplexity = math.exp(val_loss)
                    print(f"\n📊 Step {global_step}, 验证损失: {val_loss:.4f}, 困惑度: {perplexity:.2f}")
                    
                    # 保存最佳模型并检查早停
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'val_loss': val_loss,
                            'config': config
                        }, f"{config['save_dir']}/best_model_single_gpu.pt")
                        print(f"💾 保存最佳模型 (验证损失: {val_loss:.4f}, 困惑度: {perplexity:.2f})")
                    else:
                        patience_counter += 1
                        if patience_counter >= config['early_stop_patience']:
                            print(f"🔴 早停：验证损失连续{config['early_stop_patience']}次未改善")
                            return
                    
                    model.train()
                
                # 定期保存检查点
                if global_step % config['save_steps'] == 0:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'config': config
                    }, f"{config['save_dir']}/checkpoint_step_{global_step}.pt")
                    print(f"💾 保存检查点: step_{global_step}")
        
        # 计算平均损失
        epoch_loss = epoch_loss / len(train_loader) * config['gradient_accum_steps']
        epoch_time = time.time() - start_time
        
        print(f"🎯 Epoch {epoch+1}/{config['max_epochs']} 完成")
        print(f"   平均训练损失: {epoch_loss:.4f}")
        print(f"   耗时: {epoch_time:.1f}秒")
        print(f"   当前学习率: {scheduler.get_last_lr()[0]:.2e}")
        
        # 定期保存检查点
        if (epoch + 1) % 2 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'config': config
            }, f"{config['save_dir']}/checkpoint_epoch_{epoch+1}.pt")
            print(f"💾 保存检查点: epoch_{epoch+1}")

def main():
    print(f"🚀 启动简化版单GPU训练")
    
    # 训练配置
    config = {
        # 数据配置
        'data_path': 'wiki-full-zh/processed/train_seq256_bert_fast.pt',
        
        # 模型配置
        'd_model': 512,             # 中等维度，平衡表达力与计算量
        'num_transformer_layers': 8,  # 8层自注意力，足够捕捉中长程依赖
        'num_heads': 8,             # 每头维度64
        'num_rev_blocks': 4,        # 4层可逆耦合，保持非线性变换能力
        'num_rotors': 2,            # 2个转子即可提供动态置换
        
        # 训练配置
        'batch_size': 64,           # 批量越大，吞吐越高
        'learning_rate': 5e-4,      # 学习率稍低，更稳定
        'weight_decay': 1e-3,       # 轻度权重衰减防过拟合
        'max_epochs': 5,            # 数据量大时少跑几轮即可
        'gradient_accum_steps': 2,  # 若显存紧张，等效batch=128
        'warmup_ratio': 0.1,        # 预热10%，快速进入收敛区间
        
        # 优化设置
        'use_checkpointing': True,  # 激活检查点，省下中间激活存储
        'early_stop_patience': 2,   # 验证集上连续2轮不降即停
        
        # 保存和验证配置
        'eval_steps': 2000,         # 每2k步在验证集上计算困惑度
        'save_steps': 10000,        # 每10k步存一次，防止意外中断
        'save_dir': 'checkpoints_single_gpu_512d_optimized'  # 区分保存目录
    }
    
    # 计算effective batch size
    effective_batch_size = config['batch_size'] * config['gradient_accum_steps']
    
    print(f"🎯 训练配置:")
    print(f"   d_model: {config['d_model']}")
    print(f"   num_transformer_layers: {config['num_transformer_layers']}")
    print(f"   num_heads: {config['num_heads']}")
    print(f"   num_rev_blocks: {config['num_rev_blocks']}")
    print(f"   num_rotors: {config['num_rotors']}")
    print(f"   batch_size: {config['batch_size']}")
    print(f"   learning_rate: {config['learning_rate']}")
    print(f"   weight_decay: {config['weight_decay']}")
    print(f"   max_epochs: {config['max_epochs']}")
    print(f"   gradient_accum_steps: {config['gradient_accum_steps']}")
    print(f"   effective batch size: {effective_batch_size}")
    
    # 启动训练
    train_single_gpu(config)
    
    print("🎉 单GPU训练完成!")

if __name__ == "__main__":
    main() 