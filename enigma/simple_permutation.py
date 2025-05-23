import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSoftPermutation(nn.Module):
    """
    简化的软置换层 - 使用Gumbel Softmax替代Sinkhorn
    计算复杂度: O(n²) vs Sinkhorn的 O(n² * iters)
    """
    def __init__(self, dim, temperature=1.0, hard=False):
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.hard = hard
        
        # 学习的logits矩阵
        self.logits = nn.Parameter(torch.randn(dim, dim) * 0.1)
        
    def forward(self, x=None):
        """简单的行列归一化，无需迭代"""
        # 行归一化
        row_normalized = F.softmax(self.logits / self.temperature, dim=1)
        
        if self.hard:
            # 硬置换：每行选择最大值位置
            indices = torch.argmax(row_normalized, dim=1)
            hard_perm = torch.zeros_like(row_normalized)
            hard_perm[torch.arange(self.dim), indices] = 1.0
            # Straight-through estimator
            return (hard_perm - row_normalized).detach() + row_normalized
        
        return row_normalized


class FastRotor(nn.Module):
    """快速转子实现"""
    def __init__(self, dim, temperature=1.0):
        super().__init__()
        self.permutation = SimpleSoftPermutation(dim, temperature)
        
    def forward(self, x):
        perm_matrix = self.permutation()
        return torch.matmul(x, perm_matrix)
        
    def inverse(self, y):
        perm_matrix = self.permutation()
        return torch.matmul(y, perm_matrix.transpose(-1, -2))


class FastRotorStack(nn.Module):
    """快速转子堆栈"""
    def __init__(self, dim, num_rotors=3, temp_min=0.5, temp_max=1.0):
        super().__init__()
        temps = torch.linspace(temp_max, temp_min, num_rotors)
        self.rotors = nn.ModuleList([
            FastRotor(dim, temps[i].item()) for i in range(num_rotors)
        ])
        
    def forward(self, x):
        y = x
        for rotor in self.rotors:
            y = rotor(y)
        return y
        
    def inverse(self, y):
        x = y
        for rotor in reversed(self.rotors):
            x = rotor.inverse(x)
        return x 