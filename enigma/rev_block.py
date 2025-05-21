import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    卷积块，用于RevBlock中的函数f和g
    
    参数:
        channels (int): 输入/输出通道数
    """
    
    def __init__(self, channels):
        super(ConvBlock, self).__init__()
        
        # 两层全连接网络替代卷积，避免形状和归一化问题
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)
        
        # 添加比例因子，控制输出幅度，有助于提高数值稳定性
        # 使用低初始值并且要求正值，确保加法耦合可逆性
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 权重L2正则化
        self.weight_decay = 0.001
        
        # 初始化为近似恒等映射
        self._init_near_identity()
    
    def _init_near_identity(self):
        """初始化为近似恒等映射，避免初始数值不稳定"""
        with torch.no_grad():
            # 使用正交初始化，然后缩放到接近零
            nn.init.orthogonal_(self.fc1.weight)
            nn.init.orthogonal_(self.fc2.weight)
            
            self.fc1.weight.mul_(0.01)
            self.fc2.weight.mul_(0.01)
            
            nn.init.zeros_(self.fc1.bias)
            nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        参数:
            x (Tensor): 形状为 [B, C, 1] 的输入张量
            
        返回:
            Tensor: 形状为 [B, C, 1] 的输出张量
        """
        # 重塑为FC层可接受的形状
        B, C, _ = x.shape
        x_flat = x.view(B, C)
        
        # 第一层 + Leaky ReLU
        x_flat = F.leaky_relu(self.fc1(x_flat), negative_slope=0.1)
        
        # 第二层 + TanH (有界激活函数)
        x_flat = torch.tanh(self.fc2(x_flat))
        
        # 缩放输出，确保不会导致数值爆炸
        # 使用绝对值确保scale始终为正，维持可逆性
        scale_factor = torch.abs(self.scale)
        x_flat = x_flat * scale_factor
        
        # 恢复原始形状
        return x_flat.view(B, C, 1)
    
    def weight_regularization(self):
        """权重正则化项，控制变换的幅度"""
        reg = (self.fc1.weight.norm(2) + self.fc2.weight.norm(2)) * self.weight_decay
        return reg


class RevBlock(nn.Module):
    """
    可逆耦合块，基于加法耦合方式实现可逆变换
    
    参数:
        d (int): 输入/输出维度 (必须是偶数)
    """
    
    def __init__(self, d):
        super(RevBlock, self).__init__()
        assert d % 2 == 0, "维度d必须是偶数"
        
        self.d = d
        half_d = d // 2
        
        # 两个耦合函数
        self.F = ConvBlock(half_d)
        self.G = ConvBlock(half_d)
        
        # 添加残差连接比例因子，进一步稳定数值
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.9)
        
        # 缓存中间结果以节省内存
        self.register_buffer('y1_cache', None)
        self.register_buffer('y2_cache', None)
        self.register_buffer('needs_recompute', torch.tensor(True))
    
    def _reshape_to_conv(self, x, half_d):
        """辅助函数，将张量重塑为卷积格式"""
        B = x.size(0)
        return x.view(B, half_d, 1)
    
    def _reshape_from_conv(self, x, half_d):
        """辅助函数，从卷积格式重塑回原始格式"""
        B = x.size(0)
        return x.view(B, half_d)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        B = x.size(0)
        half_d = self.d // 2
        
        # 将输入拆分为两半
        x1, x2 = torch.split(x, half_d, dim=1)  # [B, half_d]
        
        # 重塑为适合卷积的形状
        x1_conv = self._reshape_to_conv(x1, half_d)
        x2_conv = self._reshape_to_conv(x2, half_d)
        
        # 确保residual_scale介于0.5和1之间，提高稳定性
        res_scale = 0.5 + 0.5 * torch.sigmoid(self.residual_scale)
        
        # 应用加法耦合，使用前面定义的F和G函数，添加残差连接
        # y1 = res_scale * x1 + F(x2)
        y1_conv = res_scale * x1_conv + self.F(x2_conv)
        # y2 = res_scale * x2 + G(y1)
        y2_conv = res_scale * x2_conv + self.G(y1_conv)
        
        # 重塑回原始形状
        y1 = self._reshape_from_conv(y1_conv, half_d)
        y2 = self._reshape_from_conv(y2_conv, half_d)
        
        # 合并输出
        y = torch.cat([y1, y2], dim=1)  # [B, d]
        
        return y
    
    def inverse(self, y):
        """
        逆向传播
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        B = y.size(0)
        half_d = self.d // 2
        
        # 将输入拆分为两半
        y1, y2 = torch.split(y, half_d, dim=1)  # [B, half_d]
        
        # 重塑为适合卷积的形状
        y1_conv = self._reshape_to_conv(y1, half_d)
        y2_conv = self._reshape_to_conv(y2, half_d)
        
        # 确保residual_scale介于0.5和1之间，提高稳定性
        res_scale = 0.5 + 0.5 * torch.sigmoid(self.residual_scale)
        
        # 逆向计算:
        # x2 = (y2 - G(y1)) / res_scale
        x2_conv = (y2_conv - self.G(y1_conv)) / res_scale
        # x1 = (y1 - F(x2)) / res_scale
        x1_conv = (y1_conv - self.F(x2_conv)) / res_scale
        
        # 重塑回原始形状
        x1 = self._reshape_from_conv(x1_conv, half_d)
        x2 = self._reshape_from_conv(x2_conv, half_d)
        
        # 合并输出
        x = torch.cat([x1, x2], dim=1)  # [B, d]
        
        return x
    
    def weight_regularization(self):
        """计算权重正则化损失"""
        return self.F.weight_regularization() + self.G.weight_regularization()
    
    def forward_with_checkpointing(self, x):
        """
        使用检查点机制的前向传播，用于减少内存使用
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        # 启用梯度检查点以节省内存
        if torch.is_grad_enabled() and any(p.requires_grad for p in self.parameters()):
            return torch.utils.checkpoint.checkpoint(self.forward, x)
        else:
            return self.forward(x) 