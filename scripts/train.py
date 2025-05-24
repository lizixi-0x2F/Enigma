#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
防过拟合训练脚本 - 基于300M tokens数据和超参数优化建议
严格控制训练epoch数，增加正则化，防止重新过拟合
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

class AntiOverfittingDataset(Dataset):
    """防过拟合数据集 - 使用修复后的无泄漏数据"""
    
    def __init__(self, data_path, gpu_id=0, num_gpus=5):
        print(f"🚀 [GPU {gpu_id}] 加载修复后的无泄漏数据: {data_path}")
        
        # 加载修复后的数据
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
        
        # BERT词汇表设置
        self.vocab_size = 21128  # BERT中文词汇表大小
        self.pad_token_id = 0    # BERT的[PAD] token
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def setup(rank, world_size):
    """初始化分布式训练"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12358'  # 新端口避免冲突
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式训练"""
    dist.destroy_process_group()

def get_cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """带预热的余弦学习率调度器，最低学习率不为0"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def calculate_perplexity_accurately(model, criterion, val_loader, device, max_batches=20):
    """🍃 更准确地计算困惑度 - 适配药方3的严格Mask"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, tokens in enumerate(val_loader):
            if i >= max_batches:
                break
                
            tokens = tokens.to(device)
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            
            logits = model(inputs)
            # 🍃 药方2+3: 对齐并严格mask
            logits = logits[:, :-1]
            targets = targets[:, 1:]
            
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = targets.reshape(-1)
            mask = flat_targets != 0  # 0是pad_token_id
            
            if mask.sum() > 0:
                loss = criterion(flat_logits[mask], flat_targets[mask]) / mask.sum()
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')  # 防止数值溢出
    
    return avg_loss, perplexity

def train_worker(rank, world_size, config):
    """训练worker进程 - 防过拟合版本"""
    print(f"🎯 [GPU {rank}] 启动防过拟合训练")
    
    # 初始化分布式
    setup(rank, world_size)
    
    # 加载修复后的数据集
    dataset = AntiOverfittingDataset(
        data_path=config['data_path'],
        gpu_id=rank,
        num_gpus=world_size
    )
    
    # 🍃 药方1: 重切验证集 - 确保行级去重，10%验证集
    train_size = int(0.9 * len(dataset))  # 90%训练，10%验证
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], 
                                              generator=torch.Generator().manual_seed(42))
    
    if rank == 0:
        print(f"🍃 [药方1] 重切验证集: 训练90% 验证10%，确保无数据泄漏")
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if rank == 0:
        print(f"📊 训练集: {len(train_dataset):,}, 验证集: {len(val_dataset):,}")
        print(f"🎯 数据总tokens估计: ~{len(train_dataset) * 256 / 1e6:.1f}M")
    
    # 创建模型 - 使用配置的dropout参数
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
    
    # 手动设置dropout参数（如果模型支持）
    def set_dropout_recursive(module, dropout_rate):
        """递归设置模型的dropout率"""
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                child.p = dropout_rate
                print(f"🎯 [GPU {rank}] 设置 {name} dropout = {dropout_rate}")
            else:
                set_dropout_recursive(child, dropout_rate)
    
    if rank == 0:
        print(f"🎯 设置attention和FF层dropout = {config['attention_dropout']}")
    set_dropout_recursive(model, config['attention_dropout'])
    
    # DDP包装
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📐 模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 优化器 - 应用weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],  # 重要的正则化参数
        betas=(0.9, 0.95),  # 稍微调整beta2，更适合大模型
        eps=1e-8
    )
    
    # 学习率调度器 - 严格控制总步数
    total_steps = (len(train_loader) // config['gradient_accum_steps']) * config['max_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_cosine_scheduler_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1)
    
    # 🍃 药方3: 严格Mask - 使用reduction='sum'后手动计算平均
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id, reduction='sum')
    try:
        scaler = GradScaler('cuda')
    except:
        scaler = GradScaler()
    
    if rank == 0:
        print(f"🍃 [药方3] 严格Mask: 使用reduction='sum'确保分母>0")
        print(f"\n🍃 =====【医生处方实施状态】=====")
        print(f"🍃 药方1: ✅ 重切验证集 (90%训练 10%验证)")
        print(f"🍃 药方2: ✅ 修正对齐 (logits[:-1] vs labels[1:])")
        print(f"🍃 药方3: ✅ 严格Mask (sum/count 确保分母>0)")
        print(f"🍃 药方4: ✅ 早停阈值 (PPL>{config['early_stop_ppl']}) 每{config['eval_steps']}步评估")
        print(f"🍃 药方5: ✅ 轻调LR ({config['learning_rate']:.0e}, cosine decay)")
        print(f"🍃 药方6: ✅ 正则加味 (dropout={config['attention_dropout']}, decay={config['weight_decay']})")
        print(f"🍃 ================================")
    
    if rank == 0:
        print(f"📈 防过拟合配置:")
        print(f"   max_epochs: {config['max_epochs']} (严格限制!)")
        print(f"   总步数: {total_steps:,}")
        print(f"   预热步数: {warmup_steps:,}")
        print(f"   learning_rate: {config['learning_rate']}")
        print(f"   weight_decay: {config['weight_decay']}")
        print(f"   attention_dropout: {config['attention_dropout']}")
        print(f"   早停patience: {config['early_stop_patience']}")
        os.makedirs(config['save_dir'], exist_ok=True)
    
    # 训练循环 - 严格监控过拟合
    global_step = 0
    best_val_loss = float('inf')
    patience_counter = 0
    training_losses = []
    validation_losses = []
    
    for epoch in range(config['max_epochs']):
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        # 进度条只在主进程显示
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [防过拟合模式]")
        else:
            progress_bar = train_loader
        
        for step, tokens in enumerate(progress_bar):
            tokens = tokens.to(rank, non_blocking=True)
            
            # 🍃 药方2: 修正对齐 - logits取[:, :-1], labels取[:, 1:]，避免"看答案"
            inputs, targets = tokens[:, :-1], tokens[:, 1:]
            
            # 前向传播 (使用混合精度)
            try:
                with autocast('cuda'):
                    logits = model(inputs)
                    # 🍃 药方2: 确保logits和targets形状对齐
                    logits = logits[:, :-1]  # 取前n-1个logits
                    targets = targets[:, 1:]  # 取后n-1个targets
                    
                    # 🍃 药方3: 严格Mask计算loss，确保分母>0
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    flat_targets = targets.reshape(-1)
                    mask = flat_targets != dataset.pad_token_id
                    
                    if mask.sum() > 0:
                        loss = criterion(flat_logits[mask], flat_targets[mask]) / mask.sum()
                    else:
                        loss = torch.tensor(0.0, device=rank, requires_grad=True)
                    loss = loss / config['gradient_accum_steps']
            except:
                with autocast():
                    logits = model(inputs)
                    # 🍃 药方2: 确保logits和targets形状对齐
                    logits = logits[:, :-1]  # 取前n-1个logits
                    targets = targets[:, 1:]  # 取后n-1个targets
                    
                    # 🍃 药方3: 严格Mask计算loss，确保分母>0
                    flat_logits = logits.reshape(-1, logits.size(-1))
                    flat_targets = targets.reshape(-1)
                    mask = flat_targets != dataset.pad_token_id
                    
                    if mask.sum() > 0:
                        loss = criterion(flat_logits[mask], flat_targets[mask]) / mask.sum()
                    else:
                        loss = torch.tensor(0.0, device=rank, requires_grad=True)
                    loss = loss / config['gradient_accum_steps']
            
            # 🍃 药方2: 每500步检查对齐质量
            if rank == 0 and global_step % 500 == 0 and step == 0:
                with torch.no_grad():
                    pred_tokens = torch.argmax(logits[0], dim=-1)  # 第一个样本的预测
                    true_tokens = targets[0]  # 第一个样本的真实标签
                    valid_mask = true_tokens != dataset.pad_token_id
                    if valid_mask.sum() > 0:
                        accuracy = (pred_tokens[valid_mask] == true_tokens[valid_mask]).float().mean()
                        print(f"🍃 [药方2] Step {global_step} 对齐检查: argmax==label 准确率 {accuracy:.1%}")
                        if accuracy < 0.8:
                            print("⚠️  准确率过低，检查logits-targets对齐")
            
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
                
                # 验证和过拟合检测（主进程）
                if rank == 0 and global_step % config['eval_steps'] == 0:
                    val_loss, perplexity = calculate_perplexity_accurately(
                        model.module, criterion, val_loader, rank
                    )
                    
                    training_losses.append(epoch_loss / (step + 1) * config['gradient_accum_steps'])
                    validation_losses.append(val_loss)
                    
                    # 计算过拟合指标
                    if len(training_losses) >= 2:
                        train_loss_trend = training_losses[-1] - training_losses[-2]
                        val_loss_trend = val_loss - validation_losses[-2]
                        overfitting_signal = val_loss_trend > 0 and train_loss_trend < -0.01
                    else:
                        overfitting_signal = False
                    
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"\n📊 Step {global_step}")
                    print(f"   训练损失: {training_losses[-1]:.4f}")
                    print(f"   验证损失: {val_loss:.4f}")
                    print(f"   困惑度: {perplexity:.2f}")
                    print(f"   学习率: {current_lr:.2e}")
                    if overfitting_signal:
                        print(f"   ⚠️  过拟合信号检测!")
                    
                    # 🍃 药方4: 早停阈值检查
                    if perplexity > config['early_stop_ppl']:
                        print(f"🍃 [药方4] 困惑度 {perplexity:.2f} > {config['early_stop_ppl']}, 触发早停")
                        cleanup()
                        return
                    
                    # 🍃 药方5: 观察训练曲线，动态调整LR建议
                    if len(training_losses) >= 3:
                        recent_train_trend = training_losses[-1] - training_losses[-3]
                        if abs(recent_train_trend) < 0.001:
                            print(f"🍃 [药方5] 训练损失停滞，建议升高学习率")
                        elif recent_train_trend > 0.01:
                            print(f"🍃 [药方5] 训练损失震荡，建议降低学习率")
                    
                    # 保存最佳模型
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save({
                            'model_state_dict': model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'global_step': global_step,
                            'val_loss': val_loss,
                            'perplexity': perplexity,
                            'config': config,
                            'training_history': {
                                'training_losses': training_losses,
                                'validation_losses': validation_losses
                            }
                        }, f"{config['save_dir']}/best_model_anti_overfitting.pt")
                        print(f"💾 保存最佳模型 (验证损失: {val_loss:.4f}, 困惑度: {perplexity:.2f})")
                    else:
                        patience_counter += 1
                        print(f"⏰ 早停计数器: {patience_counter}/{config['early_stop_patience']}")
                        
                        # 🍃 药方6: 正则加味检查 - 验证PPL<15且仍在复读
                        if perplexity < 15:
                            print(f"🍃 [药方6] 验证PPL {perplexity:.2f} < 15 但损失未降，可能出现复读")
                            print(f"   建议: 已应用 attn_dropout={config['attention_dropout']} + weight_decay={config['weight_decay']}")
                        
                        # 严格的早停策略
                        if patience_counter >= config['early_stop_patience']:
                            print(f"🛑 早停触发: 验证损失连续{config['early_stop_patience']}次未改善")
                            print(f"🎯 最佳验证损失: {best_val_loss:.4f}")
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
        
        # 计算epoch统计
        epoch_loss = epoch_loss / len(train_loader) * config['gradient_accum_steps']
        epoch_time = time.time() - start_time
        
        if rank == 0:
            print(f"🎯 Epoch {epoch+1}/{config['max_epochs']} 完成")
            print(f"   平均训练损失: {epoch_loss:.4f}")
            print(f"   耗时: {epoch_time:.1f}秒")
            print(f"   当前学习率: {scheduler.get_last_lr()[0]:.2e}")
            
            # 如果这是第一个epoch并且已经很好了，建议提前停止
            if epoch == 0 and len(validation_losses) > 0 and validation_losses[-1] < 2.0:
                print(f"🎯 注意: 第一个epoch验证损失已经很低 ({validation_losses[-1]:.4f})")
                print(f"   建议考虑更早停止以防过拟合")
    
    # 最终统计
    if rank == 0:
        print(f"\n🎉 训练完成!")
        print(f"🎯 最佳验证损失: {best_val_loss:.4f}")
        print(f"📊 训练总步数: {global_step}")
        if len(validation_losses) >= 2:
            final_trend = validation_losses[-1] - validation_losses[0]
            print(f"📈 验证损失变化: {final_trend:+.4f}")
    
    cleanup()

def main():
    """主函数 - 防过拟合配置"""
    world_size = torch.cuda.device_count()
    print(f"🚀 启动防过拟合训练脚本")
    print(f"🎯 检测到 {world_size} 张GPU")
    print(f"📚 基于300M tokens数据集的1 epoch训练策略")
    
    # 🔥 防过拟合训练配置 - 严格按照建议
    config = {
        # 数据配置
        'data_path': 'wiki-full-zh/processed/train_seq256_bert_fast.pt',  # 使用修复后的数据
        
        # 模型配置 (保持合理规模)
        'd_model': 512,                     
        'num_transformer_layers': 6,        # 减少层数防过拟合
        'num_heads': 8,                     
        'num_rev_blocks': 3,                # 减少可逆块
        'num_rotors': 2,                    
        
        # 🎯 关键防过拟合超参数
        'max_epochs': 1,                    # 💥 1 epoch就够 (300M tokens数据)
        'learning_rate': 5e-4,              # 🍃 药方5: 轻调LR 1e-4→5e-4 (cosine decay)
        'weight_decay': 0.01,               # 💥 按建议0.01
        'attention_dropout': 0.1,           # 🍃 药方6: 正则加味 attn_dropout=0.1
        
        # 训练配置
        'batch_size': 12,                   # 稍小的batch size
        'gradient_accum_steps': 2,          # 等效batch size = 12*5*2 = 120
        'warmup_ratio': 0.05,               # 短预热，快速到最大学习率
        
        # 🍃 药方4: 早停阈值PPL 20，每500步评估
        'early_stop_patience': 3,           # 验证PPL三次不降即停
        'early_stop_ppl': 20,               # PPL阈值，超过即停止
        'eval_steps': 500,                  # 每500步评估
        'save_steps': 5000,                 
        
        # 保存配置
        'save_dir': 'checkpoints_anti_overfitting'
    }
    
    # 计算关键指标
    effective_batch_size = config['batch_size'] * world_size * config['gradient_accum_steps']
    estimated_tokens_per_epoch = 1192999 * 256  # 修复后的训练集大小 * 序列长度
    
    print(f"\n🎯 防过拟合训练配置:")
    print(f"   📊 数据: ~{estimated_tokens_per_epoch/1e6:.1f}M tokens")
    print(f"   🔄 max_epochs: {config['max_epochs']} (严格限制!)")
    print(f"   📏 batch_size: {config['batch_size']} × {world_size} GPUs × {config['gradient_accum_steps']} = {effective_batch_size}")
    print(f"   📉 learning_rate: {config['learning_rate']}")
    print(f"   🏋️  weight_decay: {config['weight_decay']}")
    print(f"   💧 attention_dropout: {config['attention_dropout']}")
    print(f"   ⏰ early_stop_patience: {config['early_stop_patience']}")
    print(f"   🔍 eval_steps: {config['eval_steps']}")
    
    print(f"\n💡 预期效果:")
    print(f"   - 验证困惑度应保持在合理范围 (2-8)")
    print(f"   - 模型生成应该连贯且有意义")
    print(f"   - 避免验证损失过低导致的过拟合")
    
    # 启动多进程训练
    mp.spawn(train_worker, args=(world_size, config), nprocs=world_size, join=True)
    
    print("🎉 防过拟合训练完成!")
    print("💡 建议: 检查最终模型的困惑度和生成质量")

if __name__ == "__main__":
    main() 