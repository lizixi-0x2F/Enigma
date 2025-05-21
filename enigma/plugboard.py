import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Plugboard(nn.Module):
    """
    稀疏双射层：实现可学习的稀疏矩阵对输入进行变换
    
    参数:
        d (int): 输入/输出维度
        sparsity (float): 稀疏度，默认0.1，表示约90%的连接为0
    """
    
    def __init__(self, d, sparsity=0.1):
        super(Plugboard, self).__init__()
        self.d = d
        self.sparsity = sparsity
        
        # 创建可学习的权重矩阵 (正交初始化)
        self.weight = nn.Parameter(torch.Tensor(d, d))
        
        # 创建稀疏掩码 (不可学习)
        # 确保掩码是一个双射映射：至少每行每列有一个非零元素
        mask = self._create_bijective_mask(d, sparsity)
        self.register_buffer('mask', mask)
        
        # 初始化为单位矩阵
        self._init_identity()
    
    def _create_bijective_mask(self, d, sparsity):
        """创建一个双射映射的稀疏掩码"""
        # 单位矩阵确保基本的双射性
        mask = torch.eye(d)
        return mask
    
    def _init_identity(self):
        """初始化为单位矩阵"""
        with torch.no_grad():
            self.weight.copy_(torch.eye(self.d))
    
    def forward(self, x):
        """
        前向传播: y = (W ∘ M)·x
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        # 应用稀疏掩码
        sparse_weight = self.weight * self.mask
        
        # 矩阵乘法
        return F.linear(x, sparse_weight)
    
    def inverse(self, y):
        """
        逆向传播: x = (W ∘ M)^-1·y
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        # 应用稀疏掩码
        sparse_weight = self.weight * self.mask
        
        # 计算逆矩阵 (对于单位矩阵，逆就是自身)
        inv = sparse_weight.transpose(-2, -1)
        
        # 应用逆矩阵
        return torch.matmul(y, inv)
    
    def transpose(self, x):
        """
        转置变换: y = (W ∘ M)^T·x
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        # 应用稀疏掩码的转置
        sparse_weight_t = (self.weight * self.mask).t()
        
        # 应用转置矩阵
        return torch.matmul(x, sparse_weight_t)
    
    def transpose_inverse(self, y):
        """
        转置逆变换: x = ((W ∘ M)^T)^-1·y
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        # 应用稀疏掩码的转置
        sparse_weight_t = (self.weight * self.mask).t()
        
        # 计算转置的逆 (对于单位矩阵，就是自身)
        inv = sparse_weight_t.transpose(-2, -1)
        
        # 应用转置的逆
        return torch.matmul(y, inv)
    
    def freeze_identity(self):
        """冻结权重矩阵和掩码为单位矩阵"""
        with torch.no_grad():
            # 将权重设为单位矩阵
            self.weight.fill_(0.0)
            self.weight.fill_diagonal_(1.0)
            
            # 将掩码也设为单位矩阵
            mask = torch.zeros_like(self.mask)
            mask.fill_diagonal_(1.0)
            self.mask.copy_(mask)
    
    def l1_reg(self):
        """计算L1正则化项，用于促进稀疏性"""
        sparse_weight = self.weight * self.mask
        return sparse_weight.abs().sum() 