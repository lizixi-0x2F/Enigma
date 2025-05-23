#!/usr/bin/env python3
"""
简化版多GPU训练脚本 - 使用BERT词汇表
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
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

def collate_fn(batch):
    """可序列化的collate函数"""
    return torch.stack(batch)

class SimpleMultiGPUDataset(Dataset):
    """简化版多GPU数据集"""
    
    def __init__(self, data_path, gpu_id=0, num_gpus=5):
        print(f"🚀 [GPU {gpu_id}] 加载数据: {data_path}")
        
        # 加载数据
        all_samples = torch.load(data_path, map_location='cpu', weights_only=False)
        
        # 数据分片
        total_samples = len(all_samples)
        samples_per_gpu = total_samples // num_gpus
        start_idx = gpu_id * samples_per_gpu
        
        if gpu_id == num_gpus - 1:
            end_idx = total_samples
        else:
            end_idx = start_idx + samples_per_gpu
            
        self.samples = all_samples[start_idx:end_idx]
        print(f"✅ [GPU {gpu_id}] 分配样本: {start_idx}-{end_idx} ({len(self.samples)} 个)")
        
        # 使用BERT词汇表设置
        self.vocab_size = 21128  # BERT中文词汇表大小
        self.pad_token_id = 0    # BERT的[PAD] token
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def setup(rank, world_size):
    """初始化分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'  # 换个端口
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练"""
    dist.destroy_process_group()

def get_cosine_scheduler(optimizer, warmup_steps, total_steps):
    """余弦学习率调度器"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train_worker(rank, world_size, config):
    """训练worker进程"""
    print(f"🎯 [GPU {rank}] 启动训练")
    
    # 初始化分布式
    setup(rank, world_size)
    
    # 加载数据集
    dataset = SimpleMultiGPUDataset(
        data_path=config['data_path'],
        gpu_id=rank,
        num_gpus=world_size
    )
    
    # 数据分片 - 95%训练，5%验证
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    print(f"📊 [GPU {rank}] 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    
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
    ).to(rank)
    
    # DDP包装
    model = DDP(model, device_ids=[rank])
    
    # 启用激活检查点以节省显存
    if rank == 0 and config.get('use_checkpointing', True):
        print("✅ 启用激活检查点以节省显存")
    
    if rank == 0:
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
    
    if rank == 0:
        print(f"📈 总步数: {total_steps}, 预热步数: {warmup_steps}")
        os.makedirs(config['save_dir'], exist_ok=True)
    
    # 训练循环
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['max_epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        # 进度条只在主进程显示
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']}")
        else:
            progress_bar = train_loader
        
        for step, tokens in enumerate(progress_bar):
            tokens = tokens.to(rank, non_blocking=True)
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
                
                # 验证和保存（主进程）
                if rank == 0 and global_step % config['eval_steps'] == 0:
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for i in range(min(50, len(val_dataset))):
                            val_tokens = val_dataset[i].unsqueeze(0).to(rank)
                            val_inputs, val_targets = val_tokens[:, :-1], val_tokens[:, 1:]
                            val_logits = model(val_inputs)
                            val_loss += criterion(val_logits.reshape(-1, val_logits.size(-1)), 
                                                val_targets.reshape(-1)).item()
                    
                    val_loss /= min(50, len(val_dataset))
                    perplexity = math.exp(val_loss)
                    print(f"\n📊 Step {global_step}, 验证损失: {val_loss:.4f}, 困惑度: {perplexity:.2f}")
                    
                    # 保存最佳模型并检查早停
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save({
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'val_loss': val_loss,
                            'config': config
                        }, f"{config['save_dir']}/best_model_multigpu_simple.pt")
                        print(f"💾 保存最佳模型 (验证损失: {val_loss:.4f}, 困惑度: {perplexity:.2f})")
                    else:
                        patience_counter += 1
                        if patience_counter >= config['early_stop_patience']:
                            print(f"🔴 早停：验证损失连续{config['early_stop_patience']}次未改善")
                            cleanup()
                            return
                    
                    model.train()
                
                # 定期保存检查点（主进程）
                if rank == 0 and global_step % config['save_steps'] == 0:
                    torch.save({
                        'model_state_dict': model.module.state_dict(),
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
        
        if rank == 0:
            print(f"🎯 Epoch {epoch+1}/{config['max_epochs']} 完成")
            print(f"   平均训练损失: {epoch_loss:.4f}")
            print(f"   耗时: {epoch_time:.1f}秒")
            print(f"   当前学习率: {scheduler.get_last_lr()[0]:.2e}")
            
            # 定期保存检查点
            if (epoch + 1) % 2 == 0:
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'config': config
                }, f"{config['save_dir']}/checkpoint_epoch_{epoch+1}.pt")
                print(f"💾 保存检查点: epoch_{epoch+1}")
    
    # 清理
    cleanup()

def main():
    # 检查GPU数量
    world_size = torch.cuda.device_count()
    print(f"🚀 启动简化版多GPU训练")
    print(f"🎯 检测到 {world_size} 张GPU")
    
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
        
        # 训练配置 (多GPU调整)
        'batch_size': 16,           # 每GPU批大小，总effective=16×5×2=160
        'learning_rate': 5e-4,      # 学习率稍低，更稳定
        'weight_decay': 1e-3,       # 轻度权重衰减防过拟合
        'max_epochs': 5,            # 数据量大时少跑几轮即可
        'gradient_accum_steps': 2,  # 若显存紧张，等效batch更大
        'warmup_ratio': 0.1,        # 预热10%，快速进入收敛区间
        
        # 优化设置
        'use_checkpointing': True,  # 激活检查点，省下中间激活存储
        'early_stop_patience': 2,   # 验证集上连续2轮不降即停
        
        # 保存和验证配置
        'eval_steps': 2000,         # 每2k步在验证集上计算困惑度
        'save_steps': 10000,        # 每10k步存一次，防止意外中断
        'save_dir': 'checkpoints_multigpu_simple_512d_optimized'  # 区分保存目录
    }
    
    # 计算effective batch size
    effective_batch_size = config['batch_size'] * world_size * config['gradient_accum_steps']
    
    print(f"🎯 训练配置:")
    print(f"   d_model: {config['d_model']}")
    print(f"   num_transformer_layers: {config['num_transformer_layers']}")
    print(f"   num_heads: {config['num_heads']}")
    print(f"   num_rev_blocks: {config['num_rev_blocks']}")
    print(f"   num_rotors: {config['num_rotors']}")
    print(f"   batch_size: {config['batch_size']} (每GPU)")
    print(f"   learning_rate: {config['learning_rate']}")
    print(f"   weight_decay: {config['weight_decay']}")
    print(f"   max_epochs: {config['max_epochs']}")
    print(f"   gradient_accum_steps: {config['gradient_accum_steps']}")
    print(f"   总effective batch size: {effective_batch_size}")
    
    # 启动多进程训练
    mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    
    print("🎉 多GPU训练完成!")

if __name__ == "__main__":
    main() 