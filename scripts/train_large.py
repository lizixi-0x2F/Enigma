import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from tqdm import tqdm
import pandas as pd
import glob
import math
import time
import pickle
from transformers import BertTokenizer
from enigma.model import EnigmaLM
from torch.amp import GradScaler, autocast

class BertChineseTokenizer:
    """使用预训练的BERT中文分词器"""
    
    def __init__(self, model_name='bert-base-chinese', max_length=512):
        # 检查是否为本地路径
        if os.path.exists(model_name):
            print(f"从本地路径加载分词器: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        else:
            print(f"从Hugging Face加载分词器: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        self.max_length = max_length
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.cls_token_id  # 使用CLS作为开始标记
        self.eos_token_id = self.tokenizer.sep_token_id  # 使用SEP作为结束标记
        self.vocab_size = len(self.tokenizer)
    
    def encode(self, text):
        """编码文本为token ids"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,  # 添加CLS和SEP标记
            max_length=self.max_length,
            truncation=True
        )
    
    def decode(self, ids):
        """解码token ids为文本"""
        return self.tokenizer.decode(ids, skip_special_tokens=True)

class TextDataset(Dataset):
    """使用BERT中文分词器的文本数据集"""
    
    def __init__(self, data_dir, seq_len=256, max_samples=None, tokenizer_name='bert-base-chinese', 
                 use_saved_tokenizer=True, saved_tokenizer_path='checkpoints_optimized/tokenizer.pkl',
                 skip_tokenization=False):
        self.seq_len = seq_len
        self.samples = []
        
        print(f"正在加载数据集: {data_dir}")
        
        # 首先尝试加载分词器
        print("初始化分词器...")
        
        # 尝试加载保存的分词器
        tokenizer_loaded = False
        if use_saved_tokenizer and os.path.exists(saved_tokenizer_path):
            try:
                with open(saved_tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                print(f"成功从 {saved_tokenizer_path} 加载分词器")
                tokenizer_loaded = True
            except Exception as e:
                print(f"加载保存的分词器时出错: {e}")
                print("将创建新的分词器")
                if skip_tokenization:
                    raise ValueError("指定了跳过分词但无法加载分词器")
        
        # 如果没有成功加载保存的分词器，则创建新的分词器
        if not tokenizer_loaded:
            print("初始化BERT中文分词器...")
            self.tokenizer = BertChineseTokenizer(tokenizer_name)
        
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # 尝试从保存的样本文件加载
        # 确保processed目录存在
        processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        processed_samples_path = os.path.join(processed_dir, f"processed_samples_seq{seq_len}.pt")
        
        if os.path.exists(processed_samples_path):
            print(f"从处理好的文件加载样本: {processed_samples_path}")
            try:
                # 直接从预处理文件加载样本
                self.samples = torch.load(processed_samples_path)
                if max_samples and len(self.samples) > max_samples:
                    print(f"限制样本数量为 {max_samples}")
                    self.samples = self.samples[:max_samples]
                print(f"成功加载 {len(self.samples)} 个预处理样本")
                return  # 提前返回，跳过后续处理
            except Exception as e:
                print(f"加载预处理样本时出错: {e}")
                if skip_tokenization:
                    raise ValueError(f"指定了跳过分词但无法加载预处理样本: {e}")
                print("将重新处理原始文本...")
                self.samples = []  # 重置样本列表
        elif skip_tokenization:
            raise ValueError(f"指定了跳过分词但找不到预处理样本文件: {processed_samples_path}")
        
        # 如果没有预处理的样本文件或加载失败，则处理原始文本
        # 加载数据
        files = sorted(glob.glob(f"{data_dir}/*.parquet"))
        
        if len(files) == 0:
            raise ValueError(f"找不到parquet文件: {data_dir}/*.parquet")
        
        # 进度条显示加载进度
        with tqdm(total=len(files), desc="加载parquet文件") as pbar:
            text_samples = []
            for file in files:
                # 只读取text列，减少内存使用
                df = pd.read_parquet(file, columns=['text'])
                if 'text' in df.columns:
                    text_samples.extend(df['text'].tolist())
                pbar.update(1)
                
                # 如果达到最大样本数，则停止加载
                if max_samples and len(text_samples) >= max_samples:
                    text_samples = text_samples[:max_samples]
                    break
        
        print(f"加载了 {len(text_samples)} 段文本")
        
        # 分词和截断为固定长度的序列
        print("对文本进行分词...")
        for text in tqdm(text_samples):
            # 编码文本
            tokens = self.tokenizer.encode(text)
            
            # 处理长文本，分割为多个固定长度的序列
            for i in range(0, len(tokens), seq_len):
                chunk = tokens[i:i + seq_len]
                
                # 如果序列长度不足，则填充
                if len(chunk) < seq_len:
                    chunk = chunk + [self.pad_token_id] * (seq_len - len(chunk))
                
                self.samples.append(torch.tensor(chunk, dtype=torch.long))
        
        print(f"创建了 {len(self.samples)} 个训练样本")
        
        # 保存处理好的样本以便下次使用
        try:
            torch.save(self.samples, processed_samples_path)
            print(f"处理好的样本已保存到 {processed_samples_path}")
        except Exception as e:
            print(f"保存处理好的样本时出错: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """
    创建带有预热的余弦学习率调度器
    
    参数:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率与初始学习率的比例
        last_epoch: 上次epoch的索引
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 线性预热
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 预热后的余弦退火
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def collate_fn(batch):
    """将样本堆叠为batch"""
    return torch.stack(batch)

def parse_args():
    parser = argparse.ArgumentParser(description='大规模EnigmaLM语言模型训练 (BERT分词器)')
    parser.add_argument('--data', type=str, default='wiki-full-zh', help='数据集路径')
    parser.add_argument('--batch-size', type=int, default=32, help='批量大小')
    parser.add_argument('--grad-accum-steps', type=int, default=2, help='梯度累积步数，用于增大有效批量')
    parser.add_argument('--d-model', type=int, default=768, help='模型维度')
    parser.add_argument('--seq-len', type=int, default=256, help='序列长度')
    parser.add_argument('--max-len', type=int, default=8192, help='模型支持的最大序列长度')
    parser.add_argument('--num-rev-blocks', type=int, default=6, help='Enigma RevBlock层数')
    parser.add_argument('--num-rotors', type=int, default=4, help='Enigma转子数量')
    parser.add_argument('--num-transformer-layers', type=int, default=12, help='Transformer层数')
    parser.add_argument('--num-heads', type=int, default=12, help='注意力头数')
    parser.add_argument('--lr', type=float, default=1e-3, help='最大学习率')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1, help='最小学习率比例')
    parser.add_argument('--warmup-ratio', type=float, default=0.2, help='预热步数比例')
    parser.add_argument('--weight-decay', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=25, help='训练轮次')
    parser.add_argument('--save-dir', type=str, default='checkpoints_large', help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='训练设备')
    parser.add_argument('--use-amp', action='store_true', default=True, help='是否使用混合精度训练')
    parser.add_argument('--amp-dtype', type=str, default='float16', choices=['float16', 'bfloat16'], 
                        help='混合精度训练的数据类型')
    parser.add_argument('--use-alibi', action='store_true', default=True, help='是否使用ALiBi位置编码')
    parser.add_argument('--max-samples', type=int, default=None, help='最大加载样本数，None表示全部加载')
    parser.add_argument('--save-every', type=int, default=2000, help='每多少步保存一次检查点')
    parser.add_argument('--eval-every', type=int, default=500, help='每多少步评估一次')
    parser.add_argument('--eval-samples', type=int, default=100, help='评估样本数')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--tokenizer', type=str, default='bert-base-chinese', help='BERT分词器名称')
    parser.add_argument('--use-saved-tokenizer', action='store_true', default=True, 
                        help='是否使用已保存的分词器')
    parser.add_argument('--saved-tokenizer-path', type=str, default='checkpoints_optimized/tokenizer.pkl', 
                        help='已保存的分词器路径')
    parser.add_argument('--skip-tokenization', action='store_true', default=False,
                        help='跳过分词处理，直接使用已处理的样本文件')
    # 添加Gumbel-Sinkhorn相关参数
    parser.add_argument('--use-gumbel-sinkhorn', action='store_true', default=False, 
                        help='是否使用Gumbel-Sinkhorn软置换')
    parser.add_argument('--gumbel-temp-min', type=float, default=0.1, help='Gumbel-Sinkhorn最小温度')
    parser.add_argument('--gumbel-temp-max', type=float, default=1.0, help='Gumbel-Sinkhorn最大温度')
    parser.add_argument('--anneal-every', type=int, default=2000, help='每多少步执行一次温度退火')
    # 添加Flow模型训练相关参数
    parser.add_argument('--enable-flow-training', action='store_true', default=False, 
                        help='是否启用Flow模型训练')
    parser.add_argument('--flow-weight', type=float, default=0.1, help='Flow模型损失权重')
    parser.add_argument('--flow-prior', type=str, default='gaussian', choices=['gaussian', 'uniform'], 
                        help='Flow模型先验分布')
    return parser.parse_args()

def evaluate(model, dataset, device, pad_token_id, num_samples=100):
    """评估模型在验证集上的性能"""
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            tokens = dataset[i].unsqueeze(0).to(device)
            inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / min(num_samples, len(dataset))

def train():
    args = parse_args()
    
    # 确保保存目录存在
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据集
    print(f"加载数据集: {args.data}")
    dataset = TextDataset(
        args.data, 
        seq_len=args.seq_len, 
        max_samples=args.max_samples, 
        tokenizer_name=args.tokenizer,
        use_saved_tokenizer=args.use_saved_tokenizer,
        saved_tokenizer_path=args.saved_tokenizer_path,
        skip_tokenization=args.skip_tokenization
    )
    
    # 分割训练集和验证集 (95% 训练，5% 验证)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 保存分词器
    tokenizer_path = os.path.join(args.save_dir, "tokenizer.pkl")
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(dataset.tokenizer, f)
    print(f"分词器已保存到 {tokenizer_path}")
    
    # 创建模型
    model = EnigmaLM(
        vocab_size=dataset.vocab_size,
        d=args.d_model,
        num_rev_blocks=args.num_rev_blocks,
        num_rotors=args.num_rotors,
        num_transformer_layers=args.num_transformer_layers,
        num_heads=args.num_heads,
        max_len=args.max_len,
        use_alibi=args.use_alibi,
        use_gumbel_sinkhorn=args.use_gumbel_sinkhorn,
        gumbel_temp_min=args.gumbel_temp_min,
        gumbel_temp_max=args.gumbel_temp_max
    ).to(args.device)
    
    # 创建Flow模型（如果启用）
    flow_model = None
    if args.enable_flow_training:
        print("启用Flow模型训练")
        flow_model = model.create_flow_model(prior=args.flow_prior)
    
    # 定义优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 初始化梯度
    optimizer.zero_grad()
    
    # 计算总步数和预热步数
    # 考虑梯度累积，实际更新步数减少
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    # 创建学习率调度器
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.pad_token_id)
    
    # 混合精度训练
    scaler = torch.amp.GradScaler() if args.use_amp else None
    
    # 恢复训练检查点（如果指定）
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"加载检查点: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"恢复训练: 从epoch {start_epoch}, 步数 {global_step}")
    
    # 训练循环
    print(f"开始训练，总步数: {total_steps}, 预热步数: {warmup_steps}")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start_time = time.time()
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for step, tokens in enumerate(train_loader):
                tokens = tokens.to(args.device)
                
                # 输入和目标
                inputs = tokens[:, :-1]  # 除了最后一个token
                targets = tokens[:, 1:]  # 除了第一个token
                
                # 混合精度训练
                if args.use_amp:
                    with torch.amp.autocast('cuda', dtype=torch.float16 if args.amp_dtype == 'float16' else torch.bfloat16):
                        # 前向传播
                        logits = model(inputs)
                        # 计算语言模型损失
                        lm_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                        
                        # 如果启用Flow模型训练，添加Flow损失
                        flow_loss = 0.0
                        if args.enable_flow_training and flow_model is not None:
                            # 随机获取一批样本进行Flow训练
                            batch_size = tokens.size(0)
                            with torch.no_grad():
                                # 从先验分布采样
                                z = torch.randn(batch_size, args.d_model, device=args.device)
                            
                            # 计算Flow模型损失（负对数似然）
                            x = flow_model.inverse(z)  # 生成样本
                            flow_log_probs = flow_model.log_prob(x)  # 计算对数概率
                            flow_loss = -flow_log_probs.mean() * args.flow_weight
                        
                        # 组合损失
                        loss = lm_loss + flow_loss
                        # 梯度累积：缩放损失
                        loss = loss / args.grad_accum_steps
                    
                    # 缩放梯度并执行反向传播
                    scaler.scale(loss).backward()
                    
                    # 仅在累积完成后更新参数
                    if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        # 学习率调度（确保在优化器步骤之后）
                        scheduler.step()
                        # 仅在实际更新参数时增加global_step
                        global_step += 1
                else:
                    # 常规训练
                    # 前向传播
                    logits = model(inputs)
                    # 计算语言模型损失
                    lm_loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    
                    # 如果启用Flow模型训练，添加Flow损失
                    flow_loss = 0.0
                    if args.enable_flow_training and flow_model is not None:
                        # 随机获取一批样本进行Flow训练
                        batch_size = tokens.size(0)
                        with torch.no_grad():
                            # 从先验分布采样
                            z = torch.randn(batch_size, args.d_model, device=args.device)
                        
                        # 计算Flow模型损失（负对数似然）
                        x = flow_model.inverse(z)  # 生成样本
                        flow_log_probs = flow_model.log_prob(x)  # 计算对数概率
                        flow_loss = -flow_log_probs.mean() * args.flow_weight
                    
                    # 组合损失
                    loss = lm_loss + flow_loss
                    # 梯度累积：缩放损失
                    loss = loss / args.grad_accum_steps
                    
                    # 反向传播
                    loss.backward()
                    
                    # 仅在累积完成后更新参数
                    if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        # 学习率调度（确保在优化器步骤之后）
                        scheduler.step()
                        # 仅在实际更新参数时增加global_step
                        global_step += 1
                
                # 获取当前学习率
                current_lr = scheduler.get_last_lr()[0]
                
                # 更新进度条
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "avg_loss": f"{epoch_loss/(step+1):.4f}",
                    "lr": f"{current_lr:.8f}"
                })
                
                # 如果使用Gumbel-Sinkhorn，定期执行温度退火
                if args.use_gumbel_sinkhorn and global_step % args.anneal_every == 0:
                    temps = model.anneal_gumbel_temperatures()
                    print(f"\nStep {global_step}, 执行Gumbel-Sinkhorn温度退火: {temps}")
                
                # 定期评估
                if global_step % args.eval_every == 0:
                    val_loss = evaluate(model, val_dataset, args.device, dataset.pad_token_id, args.eval_samples)
                    print(f"\nStep {global_step}, 验证损失: {val_loss:.4f}")
                    
                    # 如果是最佳模型，单独保存
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save({
                            'epoch': epoch,
                            'global_step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'val_loss': val_loss,
                            'best_val_loss': best_val_loss
                        }, os.path.join(args.save_dir, "best_model.pt"))
                        print(f"保存最佳模型，验证损失: {val_loss:.4f}")
                    
                    # 恢复训练模式
                    model.train()
                
                # 定期保存模型
                if global_step % args.save_every == 0:
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'val_loss': best_val_loss
                    }, os.path.join(args.save_dir, f"checkpoint_step{global_step}.pt"))
        
        # 计算每个epoch的训练时间
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} 完成，用时 {epoch_time:.2f} 秒，平均损失: {epoch_loss/len(train_loader):.4f}")
        
        # 每个epoch结束后保存模型
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'val_loss': best_val_loss
        }, os.path.join(args.save_dir, f"checkpoint_epoch{epoch+1}.pt"))
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'val_loss': best_val_loss
    }, os.path.join(args.save_dir, "final_model.pt"))
    
    print(f"训练完成，模型已保存到 {args.save_dir}")
    print(f"最佳验证损失: {best_val_loss:.4f}")

if __name__ == '__main__':
    train() 