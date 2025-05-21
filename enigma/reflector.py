import torch
import torch.nn as nn
import numpy as np


class Reflector(nn.Module):
    """
    反射矩阵：实现对称正交变换，满足 U == U^T == U^-1
    
    参数:
        d (int): 输入/输出维度
    """
    
    def __init__(self, d):
        super(Reflector, self).__init__()
        self.d = d
        
        # 初始化一个随机对称正交矩阵
        U = self._init_symmetrical_orthogonal_matrix(d)
        self.U = nn.Parameter(U)
        
        # 上次应用参数化后的矩阵
        self.register_buffer('last_orthogonal', torch.zeros_like(U))
        self.register_buffer('needs_reparameterize', torch.tensor(True))
    
    def _init_symmetrical_orthogonal_matrix(self, d):
        """
        生成随机对称正交矩阵
        
        方法：首先生成随机正交矩阵Q，然后计算U = Q·Q^T，这样U自然满足对称正交
        
        参数:
            d (int): 矩阵维度
            
        返回:
            Tensor: 对称正交矩阵
        """
        # 生成随机矩阵
        A = torch.randn(d, d)
        
        # QR分解得到正交矩阵Q
        Q, _ = torch.linalg.qr(A)
        
        # 计算对称正交矩阵 (这保证U同时是对称的且正交的)
        U = torch.matmul(Q, Q.transpose(-2, -1))
        
        return U
    
    def _reparameterize(self):
        """对参数进行重新参数化，确保严格对称正交"""
        with torch.no_grad():
            # 先确保对称性
            symmetric_U = 0.5 * (self.U + self.U.transpose(-2, -1))
            
            # 进行特征值分解
            try:
                # 由于矩阵是对称的，可以使用更高效的对称特征值分解
                e, v = torch.linalg.eigh(symmetric_U)
                
                # 对特征值进行标准化 (确保其绝对值为1)
                e_sign = torch.sign(e)
                # 避免零特征值
                e_sign[e_sign == 0] = 1
                
                # 重构确保严格对称正交
                orthogonal_U = torch.matmul(v, torch.matmul(torch.diag(e_sign), v.transpose(-2, -1)))
                
                # 缓存结果
                self.last_orthogonal.copy_(orthogonal_U)
                self.needs_reparameterize.fill_(False)
                
                return orthogonal_U
            except:
                # 如果分解失败，使用原始方法
                A = torch.randn_like(self.U)
                Q, _ = torch.linalg.qr(A)
                orthogonal_U = torch.matmul(Q, Q.transpose(-2, -1))
                
                # 缓存结果
                self.last_orthogonal.copy_(orthogonal_U)
                self.needs_reparameterize.fill_(False)
                
                return orthogonal_U
    
    def forward(self, x):
        """
        应用反射变换
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的经过反射变换的输出张量
        """
        # 检查是否需要重新参数化
        if self.needs_reparameterize or self.training:
            orthogonal_U = self._reparameterize()
        else:
            orthogonal_U = self.last_orthogonal
            
        # 因为训练后参数会变化，所以每次前向传播后标记需要重新参数化
        if self.training:
            self.needs_reparameterize.fill_(True)
        
        return torch.matmul(x, orthogonal_U)
    
    def inverse(self, y):
        """
        应用反射变换的逆操作（由于U是对称正交的，逆操作与正向操作相同）
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的经过反射变换的输出张量
        """
        # 与前向传播相同，因为U是对称正交的 (U = U^T = U^-1)
        return self.forward(y)
    
    def orth_constraint(self):
        """
        计算正交约束惩罚项 ‖U^T·U - I‖^2
        
        返回:
            Tensor: 正交性偏差的平方范数
        """
        # 计算 U^T·U
        UTU = torch.matmul(self.U.transpose(-2, -1), self.U)
        
        # 计算与单位矩阵的差
        I = torch.eye(self.d, device=self.U.device)
        diff = UTU - I
        
        # 计算对称性的偏差
        sym_diff = self.U - self.U.transpose(-2, -1)
        
        # 返回平方范数 (包括正交性和对称性约束)
        return torch.norm(diff, p='fro') ** 2 + torch.norm(sym_diff, p='fro') ** 2 