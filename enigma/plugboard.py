import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import LongTensor


class Plugboard(nn.Module):
    """
    稀疏双射层：使用索引实现置换变换
    
    参数:
        d (int): 输入/输出维度
        sparsity (float): 稀疏度参数，在索引实现中不再使用，保留参数兼容性
    """
    
    def __init__(self, d, sparsity=0.1):
        super().__init__()
        # 原有稀疏矩阵参数移除
        # 新增索引参数
        self.register_buffer('perm_indices', torch.arange(d, dtype=torch.long))
        self.register_buffer('inv_indices', torch.arange(d, dtype=torch.long))
        self.d = d
    
    def forward(self, x):
        """
        前向传播: 使用索引直接选取
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        # 直接用索引选取
        return x[:, self.perm_indices]
    
    def inverse(self, y):
        """
        逆向传播: 使用逆索引选取
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        return y[:, self.inv_indices]
    
    def transpose(self, x):
        """
        转置变换: 对于置换矩阵，转置等同于逆
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        return self.inverse(x)
    
    def transpose_inverse(self, y):
        """
        转置逆变换: 对于置换矩阵，转置的逆等同于原始变换
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        return self.forward(y)
    
    def freeze_identity(self):
        """将置换设置为恒等变换"""
        self.perm_indices = torch.arange(self.d, dtype=torch.long, device=self.perm_indices.device)
        self.inv_indices = torch.arange(self.d, dtype=torch.long, device=self.inv_indices.device)
    
    def set_permutation(self, perm: LongTensor):
        """
        设置自定义置换
        
        参数:
            perm (LongTensor): 长度为d的置换向量
        """
        self.perm_indices = perm.clone()
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(self.d, dtype=torch.long, device=perm.device)
        self.inv_indices = inv
    
    def l1_reg(self):
        """计算L1正则化项，在索引实现中始终为0"""
        return torch.tensor(0.0, device=self.perm_indices.device) 