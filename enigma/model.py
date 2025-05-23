import torch
import torch.nn as nn
from enigma.plugboard import Plugboard
from enigma.rotor import create_rotor_stack
from enigma.reflector import Reflector
from enigma.rev_block import RevBlock


class Enigma(nn.Module):
    """
    Enigma 可逆动态置换网络
    
    结构：
    Plugboard P → RotorStack R → RevBlocks N层 → Reflector U → RevBlocksᴿ → Pᵀ
    
    参数:
        d (int): 输入/输出维度 (必须是偶数)
        num_rev_blocks (int): RevBlock层数
        num_rotors (int): 转子数量
        plugboard_sparsity (float): Plugboard稀疏度
        use_checkpointing (bool): 是否使用梯度检查点以节省内存
        invertibility_weight (float): 可逆性损失权重
    """
    
    def __init__(self, d, num_rev_blocks=3, num_rotors=3, plugboard_sparsity=0.1, 
                 use_checkpointing=False, invertibility_weight=0.05):
        super(Enigma, self).__init__()
        
        assert d % 2 == 0, "维度d必须是偶数"
        self.d = d
        self.num_rev_blocks = num_rev_blocks
        self.num_rotors = num_rotors
        self.use_checkpointing = use_checkpointing
        self.invertibility_weight = invertibility_weight
        
        # Plugboard - 稀疏双射层
        self.plugboard = Plugboard(d, plugboard_sparsity)
        
        # RotorStack - 动态置换层
        self.rotor_stack = create_rotor_stack(d, num_rotors)
        
        # RevBlocks - 可逆卷积层
        self.rev_blocks = nn.ModuleList([
            RevBlock(d) for _ in range(num_rev_blocks)
        ])
        
        # Reflector - 对称正交层
        self.reflector = Reflector(d)
        
        # 用于保存转子状态
        self.register_buffer('saved_positions', torch.zeros(num_rotors, dtype=torch.long))
        
        # 初始化为好的起点
        self._initialize_components()
        
        # 为了计算可逆性损失
        self.register_buffer('last_invertibility_error', torch.tensor(0.0))
    
    def _initialize_components(self):
        """优化初始化各组件以提高数值稳定性"""
        # 使用单位矩阵初始化Plugboard，避免初始训练不稳定
        self.plugboard.freeze_identity()
        
        # 初始化RevBlock的比例因子，避免数值溢出
        for rev_block in self.rev_blocks:
            with torch.no_grad():
                # 使用更保守的初始值
                rev_block.F.scale.fill_(0.005)
                rev_block.G.scale.fill_(0.005)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        batch_size = x.size(0)
        
        # 1. 通过Plugboard - 稀疏双射变换
        y = self.plugboard(x)
        
        # 2. 通过RotorStack - 动态置换
        # 保存当前转子状态，用于逆操作
        self._save_rotor_positions()
        
        y = self.rotor_stack.permute(y)
        # 前向传播后步进转子
        self.rotor_stack.step_all()
        
        # 3. 通过RevBlocks - 可逆卷积层
        for i, rev_block in enumerate(self.rev_blocks):
            if self.use_checkpointing and self.training:
                y = rev_block.forward_with_checkpointing(y)
            else:
                y = rev_block(y)
        
        # 4. 通过Reflector - 对称正交变换
        y = self.reflector(y)
        
        # 5. 通过RevBlocks (逆序) - 可逆卷积层
        for i, rev_block in enumerate(reversed(self.rev_blocks)):
            if self.use_checkpointing and self.training:
                y = rev_block.forward_with_checkpointing(y)
            else:
                y = rev_block(y)
        
        # 6. 通过Plugboard的转置变换
        y = self.plugboard.transpose(y)
        
        # 如果是训练模式，计算可逆性误差并更新
        if self.training and self.invertibility_weight > 0:
            with torch.no_grad():
                x_reconstructed = self.inverse(y.detach())
                error = torch.norm(x - x_reconstructed) / (torch.norm(x) + 1e-8)
                self.last_invertibility_error = error
        
        return y
    
    def _save_rotor_positions(self):
        """保存当前转子位置到单一tensor中"""
        positions = []
        for i, rotor in enumerate(self.rotor_stack.rotors):
            positions.append(rotor.position.clone())
        
        # 将所有位置保存到一个tensor中
        if positions:  # 确保positions非空
            self.saved_positions = torch.stack(positions)
        else:
            # 如果没有转子，创建一个空的张量
            self.saved_positions = torch.zeros(0, dtype=torch.long, device=next(self.parameters()).device)
    
    def _restore_rotor_positions(self):
        """从单一tensor中恢复保存的转子位置"""
        if len(self.saved_positions) == 0 or len(self.rotor_stack.rotors) == 0:
            return  # 如果没有转子或没有保存的位置，直接返回
            
        for i, rotor in enumerate(self.rotor_stack.rotors):
            rotor.position.copy_(self.saved_positions[i])
    
    def inverse(self, y):
        """
        逆向传播
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
        """
        # 1. 通过Plugboard转置的逆变换
        x = self.plugboard.transpose_inverse(y)
        
        # 2. 通过RevBlocks (正序) - 可逆卷积层的逆操作
        for rev_block in self.rev_blocks:
            x = rev_block.inverse(x)
        
        # 3. 通过Reflector - 对称正交变换 (由于是对称正交的，逆操作与正向相同)
        x = self.reflector(x)
        
        # 4. 通过RevBlocks (逆序) - 可逆卷积层的逆操作
        for rev_block in reversed(self.rev_blocks):
            x = rev_block.inverse(x)
        
        # 5. 恢复转子状态到前向传播前，然后进行逆操作
        self._restore_rotor_positions()
        x = self.rotor_stack.inverse_permute(x)
        
        # 6. 通过Plugboard的逆变换
        x = self.plugboard.inverse(x)
        
        return x
    
    def loss_regularizer(self):
        """计算网络的正则化损失"""
        # Plugboard的L1稀疏正则
        plug_reg = self.plugboard.l1_reg() * 0.1  # 降低稀疏性约束权重
        
        # Reflector的正交约束
        orth_reg = self.reflector.orth_constraint() * 10.0  # 增强正交约束权重
        
        # RevBlock的梯度约束
        scale_reg = sum(torch.norm(rev_block.F.scale) + torch.norm(rev_block.G.scale) 
                         for rev_block in self.rev_blocks) * 0.01
        
        # 可逆性损失（如果正在训练且设置了权重）
        inv_loss = 0.0
        if self.training and self.invertibility_weight > 0:
            inv_loss = self.last_invertibility_error * self.invertibility_weight
        
        # 组合正则化项
        reg_loss = plug_reg + orth_reg + scale_reg + inv_loss
        
        return reg_loss
    
    def check_invertibility(self, x, atol=1e-5):
        """
        检查模型对给定输入的可逆性
        
        参数:
            x (Tensor): 测试输入张量
            atol (float): 容忍误差
            
        返回:
            tuple: (是否可逆, 重构误差)
        """
        with torch.no_grad():
            # 前向传播
            y = self(x)
            
            # 逆向传播
            x_reconstructed = self.inverse(y)
            
            # 计算误差
            error = torch.norm(x - x_reconstructed) / (torch.norm(x) + 1e-8)
            
            # 判断是否满足可逆性要求
            is_invertible = error < atol
            
            return is_invertible.item() if hasattr(is_invertible, 'item') else bool(is_invertible), error.item()
            
    def orthogonalize_weights(self):
        """对模型权重进行正交化处理，提高可逆性"""
        # 强制Reflector保持正交
        self.reflector._reparameterize()
        
        # 确保Plugboard尽可能接近正交
        with torch.no_grad():
            sparse_weight = self.plugboard.weight * self.plugboard.mask
            # 使用SVD正交化
            try:
                U, S, V = torch.svd(sparse_weight)
                orthogonal_weight = torch.matmul(U, V.t())
                # 保持掩码
                self.plugboard.weight.copy_(orthogonal_weight)
            except:
                pass

class EnigmaLM(nn.Module):
    """
    基于Enigma架构的自回归语言模型
    
    参数:
        vocab_size (int): 词表大小
        d (int): 嵌入和隐藏层维度
        num_rev_blocks (int): Enigma核心的RevBlock层数
        num_rotors (int): Enigma核心的转子数量
        num_transformer_layers (int): Transformer层数
        num_heads (int): 多头注意力的头数
        d_ff (int): 前馈网络的隐藏层维度
        max_len (int): 最大序列长度
        use_alibi (bool): 是否使用ALiBi位置编码
    """
    def __init__(self, vocab_size, d, num_rev_blocks, num_rotors, num_transformer_layers=6, 
                 num_heads=8, d_ff=None, max_len=8192, use_alibi=True):
        super().__init__()
        from enigma.token_embedding import TokenEmbedding
        from enigma.attention import RevTransformerLayer
        
        if d_ff is None:
            d_ff = 4 * d
            
        self.embed = TokenEmbedding(vocab_size, d, max_len)
        self.enigma_core = Enigma(d, num_rev_blocks, num_rotors)
        
        # Transformer层，传递ALiBi参数和最大序列长度
        self.transformer_layers = nn.ModuleList([
            RevTransformerLayer(d, num_heads, d_ff, use_alibi=use_alibi, max_seq_len=max_len) 
            for _ in range(num_transformer_layers)
        ])
        
        # 输出投影层
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        # 共享嵌入和输出权重
        self.lm_head.weight = self.embed.token_emb.weight
        
        # 记录最大序列长度
        self.max_len = max_len

    def forward(self, tokens):
        """
        前向传播
        
        参数:
            tokens (Tensor): 形状为 [B, T] 的输入token序列
            
        返回:
            Tensor: 形状为 [B, T, V] 的logits
        """
        # 嵌入层
        h = self.embed(tokens)             # token + pos embedding
        
        # 生成因果掩码 (防止信息泄露)
        seq_len = tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(tokens.device)
        
        # Enigma核心处理
        B, T, D = h.shape
        h_flat = h.reshape(B*T, D)
        h_flat = self.enigma_core(h_flat)  # 处理所有位置
        h = h_flat.reshape(B, T, D)
        
        # Transformer层处理
        for layer in self.transformer_layers:
            h = layer(h, ~causal_mask)
        
        # 输出投影
        logits = self.lm_head(h)           # (B, T, V)
        return logits 