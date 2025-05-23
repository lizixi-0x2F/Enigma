import torch
import torch.nn as nn
import torch.nn.functional as F
from enigma.token_embedding import build_alibi_bias

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        
    def forward(self, x):
        # x: [B, T, D]
        # 计算RMS (root mean square)
        rms = torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return self.scale * x * rms

class TransformerBlock(nn.Module):
    """标准Transformer块，包含自注意力机制和前馈网络，使用前归一化架构"""
    
    def __init__(self, d_model, num_heads, d_ff, use_alibi=True, max_len=8192, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_alibi = use_alibi
        self.max_len = max_len
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        
        # 层归一化
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # 前馈网络
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi缓存
        self.register_buffer('alibi_bias_cache', None)
    
    def get_alibi_bias(self, seq_len, device):
        """生成ALiBi位置偏置"""
        # 计算ALiBi偏置
        if self.alibi_bias_cache is None or self.alibi_bias_cache.size(-1) < seq_len:
            # 为每个头生成不同的斜率
            slopes = torch.arange(1, self.num_heads + 1, device=device)
            slopes = 1.0 / (slopes * 2.0 ** torch.arange(0, 8, device=device).unsqueeze(1)).flatten()[:self.num_heads]
            
            # 生成距离矩阵
            pos = torch.arange(seq_len, device=device)
            dist = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0)).float()
            
            # 计算ALiBi偏置
            alibi_bias = -slopes.unsqueeze(1).unsqueeze(1) * dist.unsqueeze(0)
            
            # 缓存ALiBi偏置
            self.alibi_bias_cache = alibi_bias
        
        return self.alibi_bias_cache[..., :seq_len, :seq_len]
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, d_model]
        
        返回:
            Tensor: 输出张量，形状为 [batch_size, seq_len, d_model]
        """
        # 获取序列长度和批量大小
        batch_size, seq_len, _ = x.shape
        
        # 前归一化架构
        x_norm = self.norm1(x)
        
        # 使用PyTorch的built-in因果掩码
        # 将alibi偏置应用到QK矩阵乘法的additive_mask上
        if self.use_alibi:
            alibi_bias = self.get_alibi_bias(seq_len, x.device)
            # 将alibi_bias转换为attention模块期望的形状
            # 手动设置is_causal=False，使用我们的自定义掩码
            attn_output, _ = self.attention(
                x_norm, x_norm, x_norm,
                attn_mask=alibi_bias.repeat(batch_size, 1, 1),
                need_weights=False,
                is_causal=False
            )
        else:
            # 不使用ALiBi时，使用PyTorch的内置因果掩码
            attn_output, _ = self.attention(
                x_norm, x_norm, x_norm,
                need_weights=False,
                is_causal=True
            )
        
        # 残差连接
        x = x + self.dropout(attn_output)
        
        # 前馈网络
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        
        # 残差连接
        x = x + mlp_output
        
        return x

class RevTransformerLayer(nn.Module):
    """可逆Transformer层，结合了自注意力机制和前馈网络，使用前归一化架构"""
    
    def __init__(self, d_model, num_heads, d_ff, use_alibi=True, max_seq_len=8192):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_alibi = use_alibi
        self.max_seq_len = max_seq_len
        
        # 多头自注意力
        self.to_qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 前馈网络
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
        # 使用RMSNorm代替LayerNorm
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # ALiBi缓存
        if use_alibi:
            self.register_buffer('alibi_cache', None)
    
    def get_alibi_bias(self, seq_len, device):
        """获取或计算ALiBi偏置"""
        if not self.use_alibi:
            return None
            
        if self.alibi_cache is None or self.alibi_cache.size(-1) < seq_len:
            # 如果缓存不存在或太小，重新计算
            cache_len = max(seq_len, min(self.max_seq_len, 16384))  # 限制缓存大小
            self.alibi_cache = build_alibi_bias(self.num_heads, cache_len, device)
            
        # 返回当前序列长度需要的部分
        return self.alibi_cache[..., :seq_len, :seq_len]
    
    def forward(self, x, attn_mask=None):
        # 前归一化架构 (Pre-LN)
        
        # 注意力子层
        residual = x
        x_norm = self.norm1(x)
        
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(t.size(0), t.size(1), self.num_heads, self.head_dim).transpose(1, 2), qkv)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 应用ALiBi偏置 (如果启用)
        if self.use_alibi:
            seq_len = x.size(1)
            alibi_bias = self.get_alibi_bias(seq_len, x.device)
            scores = scores + alibi_bias
        
        # 应用因果掩码 (如果提供)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(out.size(0), -1, self.d_model)
        out = self.out_proj(out)
        
        # 残差连接
        x = residual + out
        
        # 前馈网络子层
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.ff(x_norm)
        
        return x 