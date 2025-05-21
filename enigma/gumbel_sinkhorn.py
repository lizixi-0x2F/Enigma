import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GumbelSinkhorn(nn.Module):
    """
    Gumbel-Sinkhorn置换层
    
    实现可微分的软置换矩阵，通过Gumbel-Sinkhorn算法近似离散置换
    
    参数:
        dim (int): 置换维度
        n_iters (int): Sinkhorn迭代次数
        temperature (float): 温度参数，控制软化程度
        noise_factor (float): Gumbel噪声系数
        annealing (bool): 是否使用温度退火
        hard (bool): 是否在前向传播中返回离散的置换矩阵
    """
    def __init__(self, dim, n_iters=20, temperature=1.0, noise_factor=1.0, 
                 annealing=True, hard=False):
        super(GumbelSinkhorn, self).__init__()
        self.dim = dim
        self.n_iters = n_iters
        self.temperature = temperature
        self.noise_factor = noise_factor
        self.annealing = annealing
        self.hard = hard
        self.anneal_rate = 0.9
        self.min_temp = 0.1
        
        # 可学习的logits矩阵
        self.logits = nn.Parameter(torch.zeros(dim, dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化logits为接近单位矩阵的值"""
        identity = torch.eye(self.dim)
        # 为单位矩阵添加微小扰动
        noise = torch.randn_like(identity) * 0.01
        self.logits.data.copy_(torch.log(identity + 1e-5) + noise)
    
    def gumbel_noise(self, shape, device):
        """生成Gumbel噪声"""
        # 从均匀分布中采样
        u = torch.rand(shape, device=device)
        # 转换为Gumbel分布
        return -torch.log(-torch.log(u + 1e-10) + 1e-10)
    
    def sinkhorn_normalization(self, log_alpha, n_iters=20, temp=1.0):
        """
        Sinkhorn标准化算法，将一个矩阵转换为双随机矩阵
        
        参数:
            log_alpha: 输入的对数矩阵
            n_iters: 迭代次数
            temp: 温度参数
            
        返回:
            双随机矩阵
        """
        n = log_alpha.size(0)
        log_alpha = log_alpha / temp
        
        for _ in range(n_iters):
            # 行归一化
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
            # 列归一化
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)
            
        return torch.exp(log_alpha)
    
    def soft_to_hard_perm(self, soft_perm):
        """
        将软置换矩阵转换为硬置换矩阵
        
        使用匈牙利算法（线性分配问题）
        
        参数:
            soft_perm: 软置换矩阵
            
        返回:
            硬置换矩阵
        """
        n = soft_perm.size(0)
        device = soft_perm.device
        
        # CPU上运行匈牙利算法
        soft_perm_cpu = soft_perm.detach().cpu().numpy()
        
        # 使用scipy实现匈牙利算法
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(-soft_perm_cpu)
            hard_perm = torch.zeros_like(soft_perm)
            hard_perm[row_ind, col_ind] = 1.0
        except ImportError:
            # 若不可用，则使用贪婪近似
            hard_perm = torch.zeros_like(soft_perm)
            # 依次找出每行的最大值并置1
            for i in range(n):
                j = torch.argmax(soft_perm[i])
                hard_perm[i, j] = 1.0
                # 防止列重复
                soft_perm[:, j] = -float('inf')
        
        return hard_perm.to(device)
    
    def forward(self, x=None, temperature=None):
        """
        前向传播，生成软置换矩阵
        
        参数:
            x: 可选输入，不影响置换生成
            temperature: 可选温度参数，覆盖默认温度
            
        返回:
            双随机矩阵，近似置换矩阵
        """
        batch_size = 1 if x is None else x.size(0)
        device = self.logits.device
        
        # 使用当前温度或传入的温度
        temp = temperature if temperature is not None else self.temperature
        
        # 应用Gumbel噪声
        if self.training and self.noise_factor > 0:
            noise = self.gumbel_noise(self.logits.shape, device) * self.noise_factor
            log_alpha = self.logits + noise
        else:
            log_alpha = self.logits
        
        # 通过Sinkhorn算法获得双随机矩阵
        soft_perm = self.sinkhorn_normalization(log_alpha, self.n_iters, temp)
        
        # 如果需要，转换为离散置换矩阵
        if self.hard:
            # 在前向传播中返回离散置换，反向传播中使用软置换
            hard_perm = self.soft_to_hard_perm(soft_perm)
            # Straight-through estimator
            soft_perm_hard = (hard_perm - soft_perm).detach() + soft_perm
            return soft_perm_hard
        
        return soft_perm
    
    def inverse(self, y):
        """
        逆操作，使用置换矩阵的转置
        
        参数:
            y: 输入张量
            
        返回:
            应用逆置换后的张量
        """
        # 置换矩阵的逆就是其转置
        soft_perm = self.forward()
        return torch.matmul(y, soft_perm)
    
    def anneal_temperature(self):
        """退火降低温度"""
        if self.annealing:
            self.temperature = max(self.min_temp, self.temperature * self.anneal_rate)
            return self.temperature
        return self.temperature


class GumbelSinkhornRotor(nn.Module):
    """
    基于Gumbel-Sinkhorn的可微分转子
    
    将Gumbel-Sinkhorn用于Enigma转子，实现可微分的置换操作
    
    参数:
        dim (int): 转子维度
        n_iters (int): Sinkhorn迭代次数
        temperature (float): 起始温度
        noise_factor (float): Gumbel噪声强度
    """
    def __init__(self, dim, n_iters=20, temperature=1.0, noise_factor=1.0):
        super(GumbelSinkhornRotor, self).__init__()
        self.dim = dim
        self.position = nn.Parameter(torch.tensor(0), requires_grad=False)
        
        # Gumbel-Sinkhorn置换层
        self.permutation = GumbelSinkhorn(
            dim=dim, 
            n_iters=n_iters, 
            temperature=temperature,
            noise_factor=noise_factor,
            annealing=True
        )
        
        # 步进模式（每次前向传播后是否移动转子）
        self.stepping = True
        
    def forward(self, x):
        """
        前向传播，应用置换
        
        参数:
            x: 形状为[batch_size, dim]的输入张量
            
        返回:
            形状为[batch_size, dim]的置换后张量
        """
        # 获取当前置换矩阵
        perm_matrix = self.permutation()
        
        # 应用置换
        y = torch.matmul(x, perm_matrix)
        
        return y
    
    def inverse(self, y):
        """
        逆操作，应用置换的逆
        
        参数:
            y: 形状为[batch_size, dim]的输入张量
            
        返回:
            形状为[batch_size, dim]的逆置换后张量
        """
        # 置换矩阵的逆是其转置
        perm_matrix = self.permutation()
        perm_inverse = perm_matrix.transpose(-1, -2)
        
        # 应用逆置换
        x = torch.matmul(y, perm_inverse)
        
        return x
    
    def step(self):
        """步进转子，增加内部位置计数器"""
        if self.stepping:
            with torch.no_grad():
                self.position.add_(1)
                if self.position >= self.dim:
                    self.position.zero_()
    
    def set_position(self, pos):
        """设置转子位置"""
        with torch.no_grad():
            self.position.copy_(torch.tensor(pos % self.dim, dtype=self.position.dtype))


class GumbelSinkhornRotorStack(nn.Module):
    """
    基于Gumbel-Sinkhorn的转子堆栈
    
    参数:
        dim (int): 输入维度
        num_rotors (int): 转子数量
        temp_min (float): 最小温度
        temp_max (float): 最大温度
        noise_factor (float): Gumbel噪声强度
    """
    def __init__(self, dim, num_rotors=3, temp_min=0.1, temp_max=1.0, noise_factor=1.0):
        super(GumbelSinkhornRotorStack, self).__init__()
        self.dim = dim
        self.num_rotors = num_rotors
        
        # 创建多个转子，温度递减
        temps = torch.linspace(temp_max, temp_min, num_rotors)
        self.rotors = nn.ModuleList([
            GumbelSinkhornRotor(
                dim=dim,
                temperature=temps[i].item(),
                noise_factor=noise_factor
            ) for i in range(num_rotors)
        ])
        
    def forward(self, x):
        """
        前向传播，依次通过所有转子
        
        参数:
            x: 形状为[batch_size, dim]的输入张量
            
        返回:
            形状为[batch_size, dim]的输出张量
        """
        y = x
        for rotor in self.rotors:
            y = rotor(y)
        return y
    
    def inverse(self, y):
        """
        逆操作，逆序通过所有转子的逆操作
        
        参数:
            y: 形状为[batch_size, dim]的输入张量
            
        返回:
            形状为[batch_size, dim]的输出张量
        """
        x = y
        for rotor in reversed(self.rotors):
            x = rotor.inverse(x)
        return x
    
    def step_all(self):
        """步进所有转子"""
        for i, rotor in enumerate(self.rotors):
            rotor.step()
            # 只有当前转子到达一个周期才步进下一个
            if rotor.position.item() != 0 and i < len(self.rotors) - 1:
                break
    
    def anneal_temperatures(self):
        """对所有转子进行温度退火"""
        temps = []
        for rotor in self.rotors:
            temp = rotor.permutation.anneal_temperature()
            temps.append(temp)
        return temps 