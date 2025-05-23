import torch
import torch.nn as nn
import numpy as np


class Reflector(nn.Module):
    """
    Householder反射矩阵：使用可学习向量v构造反射矩阵R = I - 2 v v^T / (v^T v)
    
    参数:
        d (int): 输入/输出维度
    """
    
    def __init__(self, d):
        super().__init__()
        # 使用标准正态分布初始化向量v
        self.v = nn.Parameter(torch.randn(d))
    
    def forward(self, x):
        """
        应用Householder反射变换
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的经过反射变换的输出张量
        """
        v = self.v
        # 计算Householder反射: R = I - 2 v v^T / (v^T v)
        # 即 R(x) = x - 2 <x,v> v / (v^T v)
        v_norm_squared = (v @ v).clamp(min=1e-8)  # 避免除以零
        beta = 2.0 / v_norm_squared
        
        # 高效实现: 避免显式构造完整的矩阵
        # 计算 <x,v>
        x_dot_v = torch.matmul(x, v)  # [B]
        
        # 计算 <x,v> v / (v^T v)
        scaled_proj = beta * torch.outer(x_dot_v, v)  # [B, d]
        
        # 最终结果: x - 2 <x,v> v / (v^T v)
        return x - scaled_proj
    
    def inverse(self, y):
        """
        应用反射变换的逆操作（由于R是对称正交的，逆操作与正向操作相同）
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的经过反射变换的输出张量
        """
        # 与前向传播相同，因为Householder反射矩阵是对称正交的 (R = R^T = R^-1)
        return self.forward(y)
    
    def orth_constraint(self):
        """
        计算正交约束惩罚项
        
        Householder反射器天然满足正交性，因此返回0
        
        返回:
            Tensor: 始终为0的张量
        """
        return torch.tensor(0.0, device=self.v.device) 