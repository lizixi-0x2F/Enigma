import torch
import torch.nn as nn
import numpy as np
from torch.autograd.functional import jacobian


class JacobianLogDet:
    """
    计算可逆网络的雅可比矩阵行列式对数
    
    用于Flow-based生成模型，提供变量变换的概率密度变换
    """
    
    @staticmethod
    def compute_logdet_analytical(model, x):
        """
        分析法计算雅可比行列式对数
        
        针对Enigma网络的特殊结构进行优化计算
        
        参数:
            model: Enigma模型实例
            x: 输入张量，形状为[batch_size, d]
            
        返回:
            雅可比行列式对数值
        """
        batch_size, d = x.shape
        
        # 获取模型组件
        plugboard = model.plugboard
        rotor_stack = model.rotor_stack
        rev_blocks = model.rev_blocks
        reflector = model.reflector
        
        # 初始化总对数行列式为0
        log_det = torch.zeros(batch_size, device=x.device)
        
        # 1. Plugboard的雅可比矩阵是固定的稀疏矩阵
        # 计算其行列式的对数
        with torch.no_grad():
            if hasattr(plugboard, 'weight') and hasattr(plugboard, 'mask'):
                weight = plugboard.weight * plugboard.mask
                # 对每个样本计算行列式，使用LU分解
                for i in range(batch_size):
                    try:
                        # 使用torch.linalg.slogdet以获得稳定的计算结果
                        sign, logdet = torch.linalg.slogdet(weight)
                        # 只有当sign为1时才是有效的
                        if sign.item() != 1:
                            # 如果sign为-1或0，使用备用方法计算
                            logdet = torch.log(torch.abs(torch.det(weight)) + 1e-10)
                        log_det[i] += logdet
                    except:
                        # 备用计算方法
                        logdet = torch.log(torch.abs(torch.det(weight)) + 1e-10)
                        log_det[i] += logdet
        
        # 2. RotorStack的雅可比矩阵是置换矩阵，行列式为1/-1
        # 置换矩阵的行列式依赖于置换的奇偶性
        # 但行列式的绝对值始终为1，所以对数值为0
        
        # 3. RevBlocks使用加法耦合，雅可比矩阵是下三角或上三角
        # 对于加法耦合层，雅可比矩阵的行列式为1
        # 所以RevBlocks不会贡献到总对数行列式
        
        # 4. Reflector的雅可比矩阵是正交矩阵，行列式为1/-1
        # 由于它是对称正交的，行列式为1或-1，取对数后为0
        
        return log_det
    
    @staticmethod
    def compute_logdet_numerical(model, x, eps=1e-4):
        """
        数值法计算雅可比行列式对数
        
        通过有限差分近似计算雅可比矩阵，适用于任何可逆网络
        
        参数:
            model: 可逆网络模型
            x: 输入张量，形状为[batch_size, d]
            eps: 有限差分步长
            
        返回:
            雅可比行列式对数值
        """
        batch_size, d = x.shape
        log_dets = []
        
        for i in range(batch_size):
            x_i = x[i:i+1]  # 保持维度
            
            # 定义一个函数以计算雅可比矩阵
            def func(z):
                z_batch = z.unsqueeze(0)  # 添加批量维度
                return model(z_batch).squeeze(0)
            
            # 使用autograd计算雅可比矩阵
            jac = jacobian(func, x_i.squeeze(0))
            jac = jac.reshape(d, d)
            
            # 计算行列式的对数
            try:
                sign, logdet = torch.linalg.slogdet(jac)
                if sign.item() != 1:
                    # 如果sign为-1或0，使用备用方法计算
                    logdet = torch.log(torch.abs(torch.det(jac)) + 1e-10)
            except:
                # 备用计算方法
                logdet = torch.log(torch.abs(torch.det(jac)) + 1e-10)
            
            log_dets.append(logdet)
        
        return torch.stack(log_dets)
    
    @staticmethod
    def compute_logdet_with_trace(model, x, eps=1e-4):
        """
        使用迹估计法计算雅可比行列式对数
        
        对于大维度输入，通过随机投影和迹估计来近似计算
        
        参数:
            model: 可逆网络模型
            x: 输入张量，形状为[batch_size, d]
            eps: 有限差分步长
            
        返回:
            雅可比行列式对数值的近似值
        """
        batch_size, d = x.shape
        device = x.device
        log_dets = torch.zeros(batch_size, device=device)
        
        # 对于每个样本，使用迹估计
        for i in range(batch_size):
            x_i = x[i:i+1]  # 保持维度
            
            # 使用Hutchinson迹估计器
            # 对数行列式 = log(det(J)) = trace(log(J))
            
            # 创建多个随机向量以提高估计精度
            num_estimates = min(10, d)  # 使用10个估计或维度大小(取较小值)
            
            total_estimate = 0
            for j in range(num_estimates):
                # 创建随机单位向量
                v = torch.randn(d, device=device)
                v = v / torch.norm(v)
                
                # 计算方向导数
                x_plus = x_i + eps * v
                x_minus = x_i - eps * v
                
                with torch.no_grad():
                    y_plus = model(x_plus)
                    y_minus = model(x_minus)
                
                # 计算差商近似雅可比矩阵与向量的乘积
                jv = (y_plus - y_minus) / (2 * eps)
                
                # 计算向量与雅可比矩阵乘积的范数
                jv_norm = torch.norm(jv)
                
                # 累加估计值
                total_estimate += torch.log(jv_norm + 1e-10) + torch.log(torch.tensor(d, device=device))
            
            # 取平均值作为最终估计
            log_dets[i] = total_estimate / num_estimates
        
        return log_dets


def add_flow_logdet_to_enigma(model_class):
    """
    为Enigma模型添加计算雅可比行列式对数的功能
    
    这是一个装饰器函数，扩展了Enigma模型的功能
    
    参数:
        model_class: Enigma模型类
        
    返回:
        扩展了flow功能的Enigma模型类
    """
    original_forward = model_class.forward
    original_inverse = model_class.inverse
    
    def new_forward(self, x, compute_logdet=False):
        """扩展的前向传播，可选地计算对数行列式"""
        y = original_forward(self, x)
        
        if compute_logdet:
            # 计算雅可比行列式对数
            logdet = JacobianLogDet.compute_logdet_analytical(self, x)
            return y, logdet
        return y
    
    def new_inverse(self, y, compute_logdet=False):
        """扩展的逆操作，可选地计算对数行列式"""
        x = original_inverse(self, y)
        
        if compute_logdet:
            # 计算逆变换的雅可比行列式对数（与前向变换的负值）
            logdet = -JacobianLogDet.compute_logdet_analytical(self, x)
            return x, logdet
        return x
    
    # 替换方法
    model_class.forward = new_forward
    model_class.inverse = new_inverse
    
    return model_class


class EnigmaFlow(nn.Module):
    """
    基于Enigma的Flow生成模型
    
    将Enigma模型转换为规范化流模型，用于生成建模
    
    参数:
        enigma_model: 基础Enigma模型
        prior: 先验分布类型，'gaussian'或'uniform'
    """
    def __init__(self, enigma_model, prior='gaussian'):
        super(EnigmaFlow, self).__init__()
        
        # 基础Enigma模型
        self.enigma = enigma_model
        
        # 先验分布类型
        self.prior_type = prior
        self.d = enigma_model.d
    
    def forward(self, x):
        """
        前向传播，计算变换后的值和对数概率
        
        参数:
            x: 输入样本
            
        返回:
            z: 变换后的样本
            log_prob: 对数概率密度
        """
        # 使用Enigma变换样本
        z = self.enigma(x)
        
        # 计算雅可比行列式对数
        logdet = JacobianLogDet.compute_logdet_analytical(self.enigma, x)
        
        # 计算先验分布下的对数概率
        if self.prior_type == 'gaussian':
            log_prior = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * self.d * torch.log(torch.tensor(2 * np.pi))
        else:  # uniform先验
            log_prior = torch.zeros(z.size(0), device=z.device)
        
        # 计算样本的对数概率
        log_prob = log_prior + logdet
        
        return z
    
    def inverse(self, z):
        """
        反向传播，从隐空间生成样本
        
        参数:
            z: 隐空间样本
            
        返回:
            x: 生成的样本
            log_prob: 对数概率密度
        """
        # 使用Enigma的逆变换
        x = self.enigma.inverse(z)
        
        # 计算雅可比行列式对数（逆变换的对数行列式是负的）
        logdet = -JacobianLogDet.compute_logdet_analytical(self.enigma, x)
        
        # 计算样本的对数概率
        if self.prior_type == 'gaussian':
            log_prior = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * self.d * torch.log(torch.tensor(2 * np.pi))
        else:  # uniform先验
            log_prior = torch.zeros(z.size(0), device=z.device)
        
        log_prob = log_prior + logdet
        
        return x
    
    def sample(self, num_samples):
        """
        从模型中采样
        
        参数:
            num_samples: 采样数量
            
        返回:
            samples: 生成的样本
        """
        # 从先验分布中采样
        if self.prior_type == 'gaussian':
            z = torch.randn(num_samples, self.d, device=next(self.parameters()).device)
        else:  # uniform先验
            z = torch.rand(num_samples, self.d, device=next(self.parameters()).device) * 2 - 1
        
        # 通过模型的逆变换生成样本
        samples = self.inverse(z)
        
        return samples
    
    def log_prob(self, x):
        """
        计算样本的对数概率密度
        
        参数:
            x: 输入样本
            
        返回:
            log_prob: 对数概率密度
        """
        z = self.forward(x)
        
        # 计算雅可比行列式对数
        logdet = JacobianLogDet.compute_logdet_analytical(self.enigma, x)
        
        # 计算先验分布下的对数概率
        if self.prior_type == 'gaussian':
            log_prior = -0.5 * torch.sum(z ** 2, dim=1) - 0.5 * self.d * torch.log(torch.tensor(2 * np.pi))
        else:  # uniform先验
            log_prior = torch.zeros(z.size(0), device=z.device)
        
        # 计算样本的对数概率
        log_prob = log_prior + logdet
        
        return log_prob 