import torch
import torch.nn as nn
import math

def build_alibi_bias(num_heads, seq_len, device=None, cached_alibi=None):
    """
    生成ALiBi (Attention with Linear Biases) 偏置矩阵，支持更长序列并使用缓存优化
    
    参数:
        num_heads: 注意力头数
        seq_len: 序列长度
        device: 设备
        cached_alibi: 之前计算的ALiBi偏置矩阵，如果提供则扩展它
        
    返回:
        bias: 形状为 [1, num_heads, seq_len, seq_len] 的偏置矩阵
    """
    # 检查缓存
    if cached_alibi is not None and cached_alibi.size(-1) >= seq_len:
        # 如果缓存的大小足够，直接返回其子集
        return cached_alibi[..., :seq_len, :seq_len]
    
    # 为每个头生成不同的斜率
    # 使用线性函数而不是指数，提高数值稳定性
    slopes = torch.arange(1, num_heads + 1, device=device)
    slopes = 1.0 / (slopes * 2.0 ** torch.arange(0, 8, device=device).unsqueeze(1)).flatten()[:num_heads]
    
    # 生成距离矩阵 - 使用显式坐标而不是相对位置，提高长序列支持
    pos = torch.arange(seq_len, device=device)
    # 计算每个位置对之间的距离
    dist = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0)).float()
    
    # 为了支持更长序列，将偏置值保持在合理范围内
    # 应用ALiBi公式: -slope * distance
    alibi_bias = -slopes.unsqueeze(1).unsqueeze(1) * dist.unsqueeze(0)
    
    # 处理因果掩码 - 未来位置的偏置设为负无穷
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    alibi_bias = alibi_bias.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
    
    return alibi_bias.unsqueeze(0)  # [1, num_heads, seq_len, seq_len]

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=8192):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.padding_idx = 0  # 默认填充索引
        self.d_model = d_model
        self.max_len = max_len
        
        # ALiBi缓存
        self.register_buffer('alibi_cache', None)
        
    def forward(self, tokens):
        # tokens: (B, T)
        seq_len = tokens.size(1)
        if seq_len > self.max_len:
            # 动态扩展位置嵌入支持的最大长度
            self.max_len = seq_len * 2  # 加倍以避免频繁调整
            # 创建新的位置嵌入
            new_pos_emb = nn.Embedding(self.max_len, self.d_model, device=tokens.device)
            # 复制原有权重
            with torch.no_grad():
                if hasattr(self, 'pos_emb'):
                    new_pos_emb.weight[:self.pos_emb.weight.size(0)] = self.pos_emb.weight
            # 替换原有位置嵌入
            self.pos_emb = new_pos_emb
        
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        
        # 组合词嵌入和位置嵌入
        embeddings = self.token_emb(tokens) + self.pos_emb(positions)
        
        return embeddings
    
    def get_alibi_bias(self, num_heads, seq_len, device=None):
        """获取缓存的ALiBi偏置或计算新的偏置"""
        if device is None:
            device = self.token_emb.weight.device
            
        if self.alibi_cache is None or self.alibi_cache.size(-1) < seq_len:
            # 如果缓存不存在或太小，计算新的ALiBi偏置并缓存
            new_len = max(seq_len, min(self.max_len, 16384))  # 限制最大缓存大小
            self.alibi_cache = build_alibi_bias(num_heads, new_len, device)
            
        # 返回适合当前序列长度的偏置子集
        return self.alibi_cache[..., :seq_len, :seq_len] 