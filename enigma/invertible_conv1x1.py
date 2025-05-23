import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class InvertibleConv1x1(nn.Module):
    """
    Glow风格的可逆1×1卷积层 - 简化稳定版本
    
    使用直接的权重矩阵参数化，通过正交初始化确保可逆性
    
    参数:
        dim (int): 输入/输出维度
    """
    
    def __init__(self, dim):
        super(InvertibleConv1x1, self).__init__()
        self.dim = dim
        
        # 直接参数化权重矩阵，使用正交初始化
        w_init = torch.randn(dim, dim)
        w_init = torch.linalg.qr(w_init)[0]  # 正交化确保可逆
        self.register_parameter('weight', nn.Parameter(w_init))
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 形状为[batch_size, dim]的输入张量
            
        返回:
            y: 形状为[batch_size, dim]的输出张量
        """
        # 使用权重矩阵进行变换
        y = torch.matmul(x, self.weight.T)
        return y
    
    def inverse(self, y):
        """
        逆变换
        
        参数:
            y: 形状为[batch_size, dim]的输入张量
            
        返回:
            x: 形状为[batch_size, dim]的输出张量
        """
        # 计算权重矩阵的逆
        weight_inv = torch.inverse(self.weight)
        x = torch.matmul(y, weight_inv.T)
        return x
    
    def log_det(self):
        """返回雅可比行列式的对数"""
        return torch.log(torch.abs(torch.det(self.weight)))


class InvertibleConv1x1Stack(nn.Module):
    """
    多个可逆1×1卷积层的堆栈，用于替换转子堆栈
    
    参数:
        dim (int): 输入维度
        num_layers (int): 层数（对应原来的转子数量）
    """
    
    def __init__(self, dim, num_layers=3):
        super(InvertibleConv1x1Stack, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # 创建多个1x1卷积层
        self.layers = nn.ModuleList([
            InvertibleConv1x1(dim) 
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        """
        前向传播，依次通过所有层
        
        参数:
            x: 形状为[batch_size, dim]的输入张量
            
        返回:
            y: 形状为[batch_size, dim]的输出张量
        """
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
    
    def inverse(self, y):
        """
        逆变换，逆序通过所有层的逆操作
        
        参数:
            y: 形状为[batch_size, dim]的输入张量
            
        返回:
            x: 形状为[batch_size, dim]的输出张量
        """
        x = y
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
    
    def log_det(self):
        """返回整个堆栈的雅可比行列式对数"""
        total_log_det = 0.0
        for layer in self.layers:
            total_log_det = total_log_det + layer.log_det()
        return total_log_det
    
    # 为了兼容原来的接口，添加以下方法
    def permute(self, x):
        """兼容原转子堆栈接口的前向传播"""
        return self.forward(x)
    
    def inverse_permute(self, y):
        """兼容原转子堆栈接口的逆变换"""
        return self.inverse(y)
    
    def step_all(self):
        """兼容原转子堆栈接口的步进操作（1x1卷积不需要步进）"""
        pass


class DynamicInvertibleConv1x1(nn.Module):
    """
    动态可逆1×1卷积，权重随时间步变化 - 简化版本
    
    这个版本更接近原始转子的概念，权重会根据位置参数动态调整
    
    参数:
        dim (int): 输入维度
        num_positions (int): 位置数量（类似转子的位置）
    """
    
    def __init__(self, dim, num_positions=None):
        super(DynamicInvertibleConv1x1, self).__init__()
        self.dim = dim
        self.num_positions = num_positions if num_positions is not None else dim
        
        # 当前位置
        self.register_buffer('position', torch.tensor(0, dtype=torch.long))
        
        # 为每个位置存储不同的权重矩阵
        weights = []
        for i in range(self.num_positions):
            w_init = torch.randn(dim, dim)
            w_init = torch.linalg.qr(w_init)[0]  # 正交化确保可逆
            weights.append(w_init)
        self.register_parameter('weight_all', nn.Parameter(torch.stack(weights)))
    
    def _get_current_weight(self):
        """获取当前位置的权重矩阵"""
        pos = self.position.item()
        return self.weight_all[pos]
    
    def forward(self, x):
        """前向传播"""
        W = self._get_current_weight()
        y = torch.matmul(x, W.T)
        return y
    
    def inverse(self, y):
        """逆变换"""
        W = self._get_current_weight()
        W_inv = torch.inverse(W)
        x = torch.matmul(y, W_inv.T)
        return x
    
    def step(self):
        """步进到下一个位置"""
        with torch.no_grad():
            self.position.add_(1)
            if self.position >= self.num_positions:
                self.position.zero_()
    
    def log_det(self):
        """返回当前位置的雅可比行列式对数"""
        W = self._get_current_weight()
        return torch.log(torch.abs(torch.det(W)))


class DynamicInvertibleConv1x1Stack(nn.Module):
    """
    动态可逆1×1卷积堆栈，完全替换转子堆栈
    
    参数:
        dim (int): 输入维度
        num_layers (int): 层数
        num_positions (int): 每层的位置数量
    """
    
    def __init__(self, dim, num_layers=3, num_positions=None):
        super(DynamicInvertibleConv1x1Stack, self).__init__()
        self.dim = dim
        self.num_layers = num_layers
        
        # 创建多个动态1x1卷积层
        self.layers = nn.ModuleList([
            DynamicInvertibleConv1x1(dim, num_positions) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        """前向传播"""
        y = x
        for layer in self.layers:
            y = layer(y)
        return y
    
    def inverse(self, y):
        """逆变换"""
        x = y
        for layer in reversed(self.layers):
            x = layer.inverse(x)
        return x
    
    def step_all(self):
        """步进所有层"""
        for layer in self.layers:
            layer.step()
    
    def log_det(self):
        """返回整个堆栈的雅可比行列式对数"""
        total_log_det = 0.0
        for layer in self.layers:
            total_log_det = total_log_det + layer.log_det()
        return total_log_det
    
    # 为了兼容原来的接口
    def permute(self, x):
        """兼容原转子堆栈接口的前向传播"""
        return self.forward(x)
    
    def inverse_permute(self, y):
        """兼容原转子堆栈接口的逆变换"""
        return self.inverse(y) 