import torch
import torch.nn as nn
from enigma.plugboard import Plugboard
from enigma.rotor import create_rotor_stack
from enigma.reflector import Reflector
from enigma.rev_block import RevBlock
from enigma.jacobian_logdet import EnigmaFlow, JacobianLogDet
from enigma.gumbel_sinkhorn import GumbelSinkhornRotorStack


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
        use_gumbel_sinkhorn (bool): 是否使用Gumbel-Sinkhorn软置换
        gumbel_temp_min (float): Gumbel-Sinkhorn最小温度
        gumbel_temp_max (float): Gumbel-Sinkhorn最大温度
    """
    
    def __init__(self, d, num_rev_blocks=3, num_rotors=3, plugboard_sparsity=0.1, 
                 use_checkpointing=False, invertibility_weight=0.05,
                 use_gumbel_sinkhorn=False, gumbel_temp_min=0.1, gumbel_temp_max=1.0):
        super(Enigma, self).__init__()
        
        assert d % 2 == 0, "维度d必须是偶数"
        self.d = d
        self.num_rev_blocks = num_rev_blocks
        self.num_rotors = num_rotors
        self.use_checkpointing = use_checkpointing
        self.invertibility_weight = invertibility_weight
        self.use_gumbel_sinkhorn = use_gumbel_sinkhorn
        
        # Plugboard - 稀疏双射层
        self.plugboard = Plugboard(d, plugboard_sparsity)
        
        # RotorStack - 动态置换层
        if use_gumbel_sinkhorn:
            self.rotor_stack = GumbelSinkhornRotorStack(
                dim=d,
                num_rotors=num_rotors,
                temp_min=gumbel_temp_min,
                temp_max=gumbel_temp_max
            )
        else:
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
    
    def forward(self, x, compute_logdet=False):
        """
        前向传播
        
        参数:
            x (Tensor): 形状为 [B, d] 的输入张量
            compute_logdet (bool): 是否计算雅可比行列式对数(用于Flow模型)
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
            (可选) Tensor: 雅可比行列式对数值
        """
        batch_size = x.size(0)
        
        # 1. 通过Plugboard - 稀疏双射变换
        y = self.plugboard(x)
        
        # 2. 通过RotorStack - 动态置换
        # 保存当前转子状态，用于逆操作
        self._save_rotor_positions()
        
        y = self.rotor_stack.permute(y) if not self.use_gumbel_sinkhorn else self.rotor_stack(y)
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
        
        # 如果需要计算雅可比行列式对数(用于Flow模型)
        if compute_logdet:
            logdet = JacobianLogDet.compute_logdet_analytical(self, x)
            return y, logdet
        
        return y
    
    def _save_rotor_positions(self):
        """保存当前转子位置到单一tensor中"""
        if self.use_gumbel_sinkhorn:
            # Gumbel-Sinkhorn转子不需要保存位置
            return
            
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
        if self.use_gumbel_sinkhorn or len(self.saved_positions) == 0:
            return  # Gumbel-Sinkhorn转子不需要恢复位置
            
        if len(self.rotor_stack.rotors) == 0:
            return  # 如果没有转子或没有保存的位置，直接返回
            
        for i, rotor in enumerate(self.rotor_stack.rotors):
            rotor.position.copy_(self.saved_positions[i])
    
    def inverse(self, y, compute_logdet=False):
        """
        逆向传播
        
        参数:
            y (Tensor): 形状为 [B, d] 的输入张量
            compute_logdet (bool): 是否计算雅可比行列式对数(用于Flow模型)
            
        返回:
            Tensor: 形状为 [B, d] 的输出张量
            (可选) Tensor: 雅可比行列式对数值
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
        if self.use_gumbel_sinkhorn:
            x = self.rotor_stack.inverse(x)
        else:
            x = self.rotor_stack.inverse_permute(x)
        
        # 6. 通过Plugboard的逆变换
        x = self.plugboard.inverse(x)
        
        # 如果需要计算雅可比行列式对数(用于Flow模型)
        if compute_logdet:
            # 逆操作的雅可比行列式是正向操作的倒数，所以取负值
            logdet = -JacobianLogDet.compute_logdet_analytical(self, x)
            return x, logdet
            
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
        """检查模型的可逆性
        
        参数:
            x (Tensor): 输入样本
            atol (float): 绝对容差
            
        返回:
            bool: 是否满足可逆性条件
            float: 误差值
        """
        with torch.no_grad():
            y = self.forward(x)
            x_reconstructed = self.inverse(y)
            error = torch.norm(x - x_reconstructed) / (torch.norm(x) + 1e-8)
            is_invertible = error.item() < atol
            return is_invertible, error.item()
    
    def orthogonalize_weights(self):
        """周期性正交化权重以提高数值稳定性"""
        # 对Reflector进行正交化
        self.reflector.orthogonalize()
        
        # 对RevBlock的权重进行调整
        for rev_block in self.rev_blocks:
            with torch.no_grad():
                # 将权重缩放到合理范围
                max_scale = 0.1
                scale_norm = torch.norm(rev_block.F.scale)
                if scale_norm > max_scale:
                    rev_block.F.scale.data *= max_scale / scale_norm
                    
                scale_norm = torch.norm(rev_block.G.scale)
                if scale_norm > max_scale:
                    rev_block.G.scale.data *= max_scale / scale_norm
    
    def anneal_gumbel_temperatures(self):
        """退火降低Gumbel-Sinkhorn的温度"""
        if self.use_gumbel_sinkhorn:
            return self.rotor_stack.anneal_temperatures()
        return 0.0
            
    def create_flow_model(self, prior='gaussian'):
        """创建基于当前Enigma模型的Flow生成模型
        
        参数:
            prior (str): 先验分布类型，'gaussian'或'uniform'
            
        返回:
            EnigmaFlow: Flow生成模型实例
        """
        return EnigmaFlow(self, prior=prior)


class EnigmaLM(nn.Module):
    """
    使用Enigma作为骨干网络的语言模型
    
    参数:
        vocab_size (int): 词汇表大小
        d (int): 模型维度
        num_rev_blocks (int): RevBlock层数
        num_rotors (int): 转子数量
        num_transformer_layers (int): Transformer层数
        num_heads (int): 注意力头数
        d_ff (int): 前馈层维度，默认为4*d
        max_len (int): 最大序列长度
        use_alibi (bool): 是否使用ALiBi位置编码
        use_gumbel_sinkhorn (bool): 是否使用Gumbel-Sinkhorn软置换
        gumbel_temp_min (float): Gumbel-Sinkhorn最小温度
        gumbel_temp_max (float): Gumbel-Sinkhorn最大温度
    """
    
    def __init__(self, vocab_size, d, num_rev_blocks, num_rotors, num_transformer_layers=6, 
                 num_heads=8, d_ff=None, max_len=8192, use_alibi=True,
                 use_gumbel_sinkhorn=False, gumbel_temp_min=0.1, gumbel_temp_max=1.0):
        super(EnigmaLM, self).__init__()
        
        self.d = d
        
        # Token嵌入
        from enigma.token_embedding import TokenEmbedding
        self.token_embedding = TokenEmbedding(vocab_size, d)
        
        # Enigma骨干网络
        self.enigma = Enigma(
            d=d,
            num_rev_blocks=num_rev_blocks,
            num_rotors=num_rotors,
            use_gumbel_sinkhorn=use_gumbel_sinkhorn,
            gumbel_temp_min=gumbel_temp_min,
            gumbel_temp_max=gumbel_temp_max
        )
        
        # Transformer层
        from enigma.attention import TransformerBlock
        d_ff = d_ff or 4 * d
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d, num_heads, d_ff, use_alibi=use_alibi, max_len=max_len)
            for _ in range(num_transformer_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(d)
        self.output_linear = nn.Linear(d, vocab_size, bias=False)
        
        # 权重绑定 (共享嵌入权重和输出层权重)
        self.output_linear.weight = self.token_embedding.embedding.weight
    
    def forward(self, tokens):
        """
        前向传播
        
        参数:
            tokens (LongTensor): 形状为 [B, L] 的输入token序列
            
        返回:
            Tensor: 形状为 [B, L, vocab_size] 的对数概率
        """
        # 获取序列长度和批量大小
        B, L = tokens.shape
        
        # Token嵌入
        x = self.token_embedding(tokens)  # [B, L, d]
        
        # Enigma处理
        enigma_outputs = []
        for i in range(L):
            # 对每个位置应用Enigma
            enigma_out = self.enigma(x[:, i])  # [B, d]
            enigma_outputs.append(enigma_out)
        
        x = torch.stack(enigma_outputs, dim=1)  # [B, L, d]
        
        # Transformer层
        for transformer in self.transformer_blocks:
            x = transformer(x)
        
        # 输出层
        x = self.output_norm(x)
        logits = self.output_linear(x)  # [B, L, vocab_size]
        
        return logits
    
    def anneal_gumbel_temperatures(self):
        """退火降低Gumbel-Sinkhorn的温度"""
        return self.enigma.anneal_gumbel_temperatures()
    
    def create_flow_model(self, prior='gaussian'):
        """创建基于Enigma部分的Flow生成模型"""
        return self.enigma.create_flow_model(prior=prior) 