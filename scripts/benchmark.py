import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import gzip
import requests
from io import BytesIO

from enigma.model import Enigma
from enigma.gumbel_sinkhorn import GumbelSinkhornRotorStack


class BenchmarkDataset:
    """
    基准测试数据集加载器
    
    支持两种基准任务:
    1. Copy-Memory: 测试模型记忆和复制长序列的能力
    2. enwik8: 基于维基百科文本压缩的基准，测试模型的压缩能力
    """
    def __init__(self, task='copy', seq_len=100, batch_size=32, num_batches=100, 
                 data_dim=10, device='cpu'):
        """
        初始化数据集
        
        参数:
            task: 'copy'或'enwik8'
            seq_len: 序列长度
            batch_size: 批量大小
            num_batches: 批次数量
            data_dim: 数据维度(仅用于copy任务)
            device: 设备
        """
        self.task = task
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.data_dim = data_dim
        self.device = device
        
        # 准备数据
        if task == 'copy':
            self.prepare_copy_data()
        elif task == 'enwik8':
            self.prepare_enwik8_data()
        else:
            raise ValueError(f"不支持的任务: {task}")
    
    def prepare_copy_data(self):
        """准备Copy-Memory任务的随机数据"""
        # 生成用于训练的随机序列
        self.data = [self._generate_copy_batch() for _ in range(self.num_batches)]
    
    def _generate_copy_batch(self):
        """生成一个Copy-Memory批次"""
        # 序列长度是输入序列长度+1个分隔符+输入长度（期望输出）
        seq_length = 2 * self.seq_len + 1
        
        # 随机生成输入序列（one-hot编码）
        x_int = torch.randint(0, self.data_dim - 1, (self.batch_size, self.seq_len))
        x = torch.zeros(self.batch_size, seq_length, self.data_dim)
        
        # 填充输入序列
        for b in range(self.batch_size):
            for t in range(self.seq_len):
                x[b, t, x_int[b, t]] = 1.0
        
        # 添加分隔符
        x[:, self.seq_len, self.data_dim - 1] = 1.0
        
        # 生成目标序列 - 前半部分是零，后半部分是输入序列的复制
        y = torch.zeros_like(x)
        for b in range(self.batch_size):
            for t in range(self.seq_len):
                y[b, self.seq_len + 1 + t, x_int[b, t]] = 1.0
        
        return x.to(self.device), y.to(self.device)
    
    def prepare_enwik8_data(self):
        """准备enwik8数据集"""
        # 下载enwik8数据（如果尚未下载）
        data_path = "enwik8.gz"
        if not os.path.exists(data_path):
            print("正在下载enwik8数据...")
            url = "http://mattmahoney.net/dc/enwik8.zip"
            try:
                response = requests.get(url)
                with open("enwik8.zip", "wb") as f:
                    f.write(response.content)
                
                # 解压缩
                import zipfile
                with zipfile.ZipFile("enwik8.zip", "r") as zip_ref:
                    zip_ref.extractall(".")
                
                # 压缩为gzip格式（便于处理）
                with open("enwik8", "rb") as f_in:
                    with gzip.open(data_path, "wb") as f_out:
                        f_out.write(f_in.read())
                
                # 清理
                os.remove("enwik8.zip")
                os.remove("enwik8")
            except:
                print("下载失败，使用随机数据代替")
                # 创建随机数据
                data = np.random.randint(0, 256, size=100000000, dtype=np.uint8)
                with gzip.open(data_path, "wb") as f:
                    f.write(data.tobytes())
        
        # 加载数据
        with gzip.open(data_path, "rb") as f:
            data = f.read()
        
        # 将数据转换为张量
        data = np.frombuffer(data, dtype=np.uint8)
        data = torch.from_numpy(data).long()
        
        # 划分训练集和验证集
        train_size = int(0.9 * len(data))
        self.train_data = data[:train_size]
        self.val_data = data[train_size:]
        
        # 准备批次
        self.data = []
        for _ in range(self.num_batches):
            # 随机选择起始点
            indices = torch.randint(0, len(self.train_data) - self.seq_len - 1, (self.batch_size,))
            x_batch = []
            y_batch = []
            
            for idx in indices:
                # 输入是从idx开始的seq_len个字符
                x = self.train_data[idx:idx + self.seq_len]
                # 目标是从idx+1开始的seq_len个字符（预测下一个字符）
                y = self.train_data[idx + 1:idx + self.seq_len + 1]
                
                x_batch.append(x)
                y_batch.append(y)
            
            # 转换为one-hot编码
            x_one_hot = torch.zeros(self.batch_size, self.seq_len, 256)
            for b in range(self.batch_size):
                for t in range(self.seq_len):
                    x_one_hot[b, t, x_batch[b][t]] = 1.0
            
            y_tensor = torch.stack(y_batch)
            
            self.data.append((x_one_hot.to(self.device), y_tensor.to(self.device)))
    
    def __len__(self):
        """返回批次数量"""
        return self.num_batches
    
    def __getitem__(self, idx):
        """获取指定索引的批次"""
        return self.data[idx]


class CopyMemoryModel(nn.Module):
    """
    Copy-Memory任务的模型
    
    使用Enigma作为序列处理的核心组件
    """
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len, 
                 num_rev_blocks=3, num_rotors=3, use_gumbel_sinkhorn=False):
        super(CopyMemoryModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Enigma核心
        self.enigma = Enigma(
            d=hidden_dim,
            num_rev_blocks=num_rev_blocks,
            num_rotors=num_rotors if not use_gumbel_sinkhorn else 0,
            plugboard_sparsity=0.1,
            invertibility_weight=0.1
        )
        
        # 可选的Gumbel-Sinkhorn转子堆栈
        self.use_gumbel_sinkhorn = use_gumbel_sinkhorn
        if use_gumbel_sinkhorn:
            self.gumbel_rotor_stack = GumbelSinkhornRotorStack(
                dim=hidden_dim, 
                num_rotors=num_rotors
            )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 形状为[batch_size, seq_len, input_dim]的输入序列
            
        返回:
            形状为[batch_size, seq_len, output_dim]的输出序列
        """
        batch_size, seq_len, _ = x.shape
        outputs = []
        
        # 对序列中的每个时间步进行处理
        for t in range(seq_len):
            # 获取当前输入
            x_t = x[:, t, :]
            
            # 输入投影
            h = self.input_proj(x_t)
            
            # 通过Enigma
            if self.use_gumbel_sinkhorn:
                # 先通过Enigma的Plugboard
                h = self.enigma.plugboard(h)
                
                # 通过Gumbel-Sinkhorn转子堆栈
                h = self.gumbel_rotor_stack(h)
                self.gumbel_rotor_stack.step_all()
                
                # 通过剩余的Enigma组件
                for rev_block in self.enigma.rev_blocks:
                    h = rev_block(h)
                h = self.enigma.reflector(h)
                for rev_block in reversed(self.enigma.rev_blocks):
                    h = rev_block(h)
                h = self.enigma.plugboard.transpose(h)
            else:
                # 使用标准Enigma
                h = self.enigma(h)
            
            # 输出投影
            out = self.output_proj(h)
            outputs.append(out)
        
        # 堆叠所有时间步的输出
        return torch.stack(outputs, dim=1)


class Enwik8Model(nn.Module):
    """
    enwik8基准任务的模型
    
    使用Enigma作为字符级压缩和预测的核心组件
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_rev_blocks=3, num_rotors=3,
                 use_gumbel_sinkhorn=False):
        super(Enwik8Model, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Enigma核心
        self.enigma = Enigma(
            d=hidden_dim,
            num_rev_blocks=num_rev_blocks,
            num_rotors=num_rotors if not use_gumbel_sinkhorn else 0,
            plugboard_sparsity=0.1,
            invertibility_weight=0.1
        )
        
        # 可选的Gumbel-Sinkhorn转子堆栈
        self.use_gumbel_sinkhorn = use_gumbel_sinkhorn
        if use_gumbel_sinkhorn:
            self.gumbel_rotor_stack = GumbelSinkhornRotorStack(
                dim=hidden_dim, 
                num_rotors=num_rotors
            )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 形状为[batch_size, seq_len, input_dim]的输入序列
            
        返回:
            形状为[batch_size, seq_len, output_dim]的字符预测
        """
        batch_size, seq_len, _ = x.shape
        
        # 首先通过投影层压缩输入
        h = self.input_proj(x.reshape(-1, self.input_dim))
        h = h.reshape(batch_size, seq_len, self.hidden_dim)
        
        # 对每个时间步进行处理
        outputs = []
        h_state = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        for t in range(seq_len):
            # 组合当前输入和隐藏状态
            h_t = h[:, t, :] + h_state
            
            # 通过Enigma
            if self.use_gumbel_sinkhorn:
                # 先通过Enigma的Plugboard
                h_out = self.enigma.plugboard(h_t)
                
                # 通过Gumbel-Sinkhorn转子堆栈
                h_out = self.gumbel_rotor_stack(h_out)
                self.gumbel_rotor_stack.step_all()
                
                # 通过剩余的Enigma组件
                for rev_block in self.enigma.rev_blocks:
                    h_out = rev_block(h_out)
                h_out = self.enigma.reflector(h_out)
                for rev_block in reversed(self.enigma.rev_blocks):
                    h_out = rev_block(h_out)
                h_out = self.enigma.plugboard.transpose(h_out)
            else:
                # 使用标准Enigma
                h_out = self.enigma(h_t)
            
            # 更新隐藏状态
            h_state = h_out
            
            # 输出投影
            out = self.output_proj(h_out)
            outputs.append(out)
        
        # 堆叠所有时间步的输出
        return torch.stack(outputs, dim=1)


def train_copy_memory(args):
    """训练Copy-Memory任务"""
    print(f"开始Copy-Memory任务训练 (使用{'Gumbel-Sinkhorn' if args.use_gumbel_sinkhorn else '标准'} Enigma)")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # 创建数据集
    dataset = BenchmarkDataset(
        task='copy',
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=args.batches_per_epoch,
        data_dim=args.data_dim,
        device=device
    )
    
    # 创建模型
    model = CopyMemoryModel(
        input_dim=args.data_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.data_dim,
        seq_len=2 * args.seq_len + 1,  # 输入+分隔符+输出
        num_rev_blocks=args.num_rev_blocks,
        num_rotors=args.num_rotors,
        use_gumbel_sinkhorn=args.use_gumbel_sinkhorn
    ).to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 损失函数
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练循环
    train_losses = []
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(dataset), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, (x, y) in enumerate(dataset):
                # 前向传播
                y_pred = model(x)
                
                # 计算损失
                loss = criterion(y_pred, y)
                
                # 如果使用Enigma，添加可逆性损失
                if hasattr(model.enigma, 'loss_regularizer'):
                    reg_loss = model.enigma.loss_regularizer() * args.reg_weight
                    loss += reg_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 如果使用Gumbel-Sinkhorn，执行温度退火
                if args.use_gumbel_sinkhorn:
                    model.gumbel_rotor_stack.anneal_temperatures()
                
                # 更新损失
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataset)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.6f}")
        
        # 定期正交化Enigma权重
        if (epoch + 1) % 5 == 0 and hasattr(model.enigma, 'orthogonalize_weights'):
            model.enigma.orthogonalize_weights()
    
    # 评估
    model.eval()
    test_batch_x, test_batch_y = dataset[0]  # 使用第一个批次作为测试
    
    with torch.no_grad():
        y_pred = model(test_batch_x)
        # 转换为类别
        y_pred_class = torch.argmax(y_pred, dim=-1)
        y_true_class = torch.argmax(test_batch_y, dim=-1)
        
        # 计算准确率
        correct = (y_pred_class == y_true_class).float()
        # 仅考虑需要复制的部分
        target_correct = correct[:, args.seq_len+1:]
        accuracy = target_correct.mean().item()
        
        print(f"测试准确率: {accuracy:.4f}")
    
    # 绘制损失曲线
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Copy-Memory任务训练损失')
        plt.grid(True)
        plt.savefig('copy_memory_loss.png')
        print("训练损失曲线已保存到 copy_memory_loss.png")
    except:
        print("无法创建损失曲线图")
    
    # 保存模型
    torch.save(model.state_dict(), "copy_memory_model.pth")
    print("模型已保存到 copy_memory_model.pth")


def train_enwik8(args):
    """训练enwik8基准任务"""
    print(f"开始enwik8任务训练 (使用{'Gumbel-Sinkhorn' if args.use_gumbel_sinkhorn else '标准'} Enigma)")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    
    # 创建数据集
    dataset = BenchmarkDataset(
        task='enwik8',
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=args.batches_per_epoch,
        device=device
    )
    
    # 创建模型
    model = Enwik8Model(
        input_dim=256,  # 一个字节的可能值
        hidden_dim=args.hidden_dim,
        output_dim=256,  # 预测下一个字节
        num_rev_blocks=args.num_rev_blocks,
        num_rotors=args.num_rotors,
        use_gumbel_sinkhorn=args.use_gumbel_sinkhorn
    ).to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    train_losses = []
    bits_per_byte_history = []
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(total=len(dataset), desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch_idx, (x, y) in enumerate(dataset):
                # 前向传播
                y_pred = model(x)
                
                # 重新形状以匹配交叉熵损失的期望
                y_pred = y_pred.reshape(-1, 256)
                y = y.reshape(-1)
                
                # 计算损失
                loss = criterion(y_pred, y)
                
                # 如果使用Enigma，添加可逆性损失
                if hasattr(model.enigma, 'loss_regularizer'):
                    reg_loss = model.enigma.loss_regularizer() * args.reg_weight
                    loss += reg_loss
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 如果使用Gumbel-Sinkhorn，执行温度退火
                if args.use_gumbel_sinkhorn:
                    model.gumbel_rotor_stack.anneal_temperatures()
                
                # 更新损失
                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix({"loss": loss.item()})
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataset)
        train_losses.append(avg_loss)
        
        # 计算bits per byte (对数据压缩的估计)
        bits_per_byte = avg_loss / np.log(2)
        bits_per_byte_history.append(bits_per_byte)
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.6f}, Bits per Byte: {bits_per_byte:.4f}")
        
        # 定期正交化Enigma权重
        if (epoch + 1) % 5 == 0 and hasattr(model.enigma, 'orthogonalize_weights'):
            model.enigma.orthogonalize_weights()
    
    # 绘制损失曲线
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(bits_per_byte_history)
        plt.xlabel('Epoch')
        plt.ylabel('Bits per Byte')
        plt.title('enwik8任务压缩性能')
        plt.grid(True)
        plt.savefig('enwik8_compression.png')
        print("压缩性能曲线已保存到 enwik8_compression.png")
    except:
        print("无法创建压缩性能曲线图")
    
    # 保存模型
    torch.save(model.state_dict(), "enwik8_model.pth")
    print("模型已保存到 enwik8_model.pth")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Enigma基准测试')
    
    # 任务选择
    parser.add_argument('--task', type=str, default='copy', choices=['copy', 'enwik8'],
                        help='基准任务: copy或enwik8')
    
    # 数据参数
    parser.add_argument('--seq_len', type=int, default=50,
                        help='Copy-Memory任务的序列长度或enwik8任务的上下文窗口大小')
    parser.add_argument('--data_dim', type=int, default=10,
                        help='Copy-Memory任务的数据维度')
    
    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Enigma隐藏层维度')
    parser.add_argument('--num_rev_blocks', type=int, default=3,
                        help='RevBlock层数')
    parser.add_argument('--num_rotors', type=int, default=3,
                        help='转子数量')
    parser.add_argument('--use_gumbel_sinkhorn', action='store_true',
                        help='使用Gumbel-Sinkhorn转子代替标准转子')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量大小')
    parser.add_argument('--batches_per_epoch', type=int, default=100,
                        help='每个epoch的批次数')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练轮次')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--reg_weight', type=float, default=0.01,
                        help='正则化权重')
    parser.add_argument('--no_cuda', action='store_true',
                        help='禁用CUDA')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 选择任务并开始训练
    if args.task == 'copy':
        train_copy_memory(args)
    else:  # enwik8
        train_enwik8(args) 