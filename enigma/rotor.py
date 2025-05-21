import torch
import torch.nn as nn
import numpy as np


def shift(permutation, step):
    """
    对置换矩阵进行循环移位
    
    参数:
        permutation (Tensor): 置换矩阵
        step (int): 移动步数
        
    返回:
        Tensor: 移位后的置换矩阵
    """
    d = permutation.size(0)
    indices = torch.arange(d, device=permutation.device)
    indices = (indices + step) % d
    return permutation[indices]


class Rotor(nn.Module):
    """
    实现Enigma机转子机制的动态置换层
    
    参数:
        d (int): 输入/输出维度
        notch_pos (int, optional): 缺口位置，默认为d//2
    """
    
    def __init__(self, d, notch_pos=None):
        super(Rotor, self).__init__()
        self.d = d
        self.notch_pos = notch_pos if notch_pos is not None else d // 2
        
        # 初始化为有效的置换矩阵
        perm = self._generate_valid_permutation(d)
        self.register_buffer('permutation', perm)
        
        # 当前位置
        self.register_buffer('position', torch.tensor(0, dtype=torch.long))
        
        # 预计算并缓存所有可能位置的置换和逆置换
        # 这消除了动态计算带来的数值误差
        self._precompute_permutations()
    
    def _generate_valid_permutation(self, d):
        """生成一个有效的置换矩阵（确保每个元素都有唯一的映射）"""
        return torch.randperm(d)
    
    def _precompute_permutations(self):
        """预计算所有位置的置换和逆置换，并缓存结果"""
        forward_perms = []
        inverse_perms = []
        
        for pos in range(self.d):
            # 计算当前位置的置换
            current_perm = shift(self.permutation, pos)
            forward_perms.append(current_perm)
            
            # 计算逆置换
            inv_indices = torch.zeros_like(current_perm)
            inv_indices[current_perm] = torch.arange(self.d, device=current_perm.device)
            inverse_perms.append(inv_indices)
        
        # 将所有置换组合成一个张量
        self.register_buffer('cached_forward_perms', torch.stack(forward_perms))
        self.register_buffer('cached_inverse_perms', torch.stack(inverse_perms))
    
    def forward(self, x):
        """
        应用当前置换状态
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的经过置换的输出张量
        """
        # 获取当前位置的预计算置换
        current_perm = self.cached_forward_perms[self.position]
        
        # 应用置换
        return x[:, current_perm]
    
    def inverse(self, y):
        """
        应用置换的逆操作
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的逆置换后的输出张量
        """
        # 获取当前位置的预计算逆置换
        current_inv_perm = self.cached_inverse_perms[self.position]
        
        # 应用逆置换
        return y[:, current_inv_perm]
    
    def step(self):
        """
        更新转子位置
        
        返回:
            bool: 如果越过缺口则返回True，否则返回False
        """
        # 保存旧位置
        old_position = self.position.item()
        
        # 更新位置
        self.position = (self.position + 1) % self.d
        
        # 检查是否越过缺口
        return self.position.item() == self.notch_pos


class RotorStack(nn.Module):
    """
    多个转子的组合，模拟Enigma机的多转子机制
    
    参数:
        d (int): 输入/输出维度
        num_rotors (int): 转子数量
    """
    
    def __init__(self, d, num_rotors):
        super(RotorStack, self).__init__()
        self.d = d
        self.num_rotors = num_rotors
        
        # 创建多个转子，确保每个转子使用不同的置换
        self.rotors = nn.ModuleList()
        for i in range(num_rotors):
            # 使用不同的种子初始化每个转子
            torch.manual_seed(42 + i)
            self.rotors.append(Rotor(d, notch_pos=int(d * (i+1) / (num_rotors+1))))
    
    def forward(self, x):
        """
        顺序应用所有转子
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的经过所有转子处理的输出张量
        """
        # 依次通过每个转子
        for rotor in self.rotors:
            x = rotor(x)
        return x
    
    def inverse(self, y):
        """
        逆序应用所有转子的逆操作
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的经过所有转子逆操作处理的输出张量
        """
        # 逆序依次通过每个转子的逆操作
        for rotor in reversed(self.rotors):
            y = rotor.inverse(y)
        return y
    
    def step_all(self):
        """更新所有转子位置，具有缺口进位机制"""
        # 模拟Enigma机的转子进位机制
        carry = True  # 第一个转子总是转动
        
        for rotor in self.rotors:
            if carry:
                carry = rotor.step()  # 如果前一个转子触发缺口，则当前转子步进
            else:
                break  # 如果没有进位，则停止步进 