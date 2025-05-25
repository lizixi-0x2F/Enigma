#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HuggingFace兼容的Enigma模型实现
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from transformers.utils import logging

from .model import Enigma, EnigmaLM
from .token_embedding import TokenEmbedding
from .attention import TransformerBlock

logger = logging.get_logger(__name__)


class EnigmaConfig(PretrainedConfig):
    """
    Enigma模型的配置类
    """
    model_type = "enigma"
    
    def __init__(
        self,
        vocab_size=21128,
        hidden_size=512,
        num_revblocks=6,
        num_rotors=4,
        num_transformer_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        max_position_embeddings=2048,
        use_alibi=True,
        use_invertible_conv=True,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
        gradient_checkpointing=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_revblocks = num_revblocks
        self.num_rotors = num_rotors
        self.num_transformer_layers = num_transformer_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.use_alibi = use_alibi
        self.use_invertible_conv = use_invertible_conv
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.gradient_checkpointing = gradient_checkpointing
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )


class EnigmaForCausalLM(PreTrainedModel, GenerationMixin):
    """
    HuggingFace兼容的Enigma因果语言模型
    """
    config_class = EnigmaConfig
    base_model_prefix = "enigma"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock", "RevBlock", "Enigma"]
    
    def __init__(self, config: EnigmaConfig):
        super().__init__(config)
        self.config = config
        
        # 确保hidden_size是偶数（Enigma要求）
        if config.hidden_size % 2 != 0:
            raise ValueError(f"hidden_size ({config.hidden_size}) 必须是偶数")
        
        # Token嵌入层
        self.token_embedding = TokenEmbedding(config.vocab_size, config.hidden_size)
        
        # Enigma核心网络 - 使用1x1卷积转子
        self.enigma = Enigma(
            d=config.hidden_size,
            num_rev_blocks=config.num_revblocks,
            num_rotors=config.num_rotors,
            use_dynamic_conv1x1=config.use_invertible_conv,  # 使用1x1卷积转子
            conv1x1_positions=config.hidden_size,  # 位置数量等于隐藏维度
            use_checkpointing=config.gradient_checkpointing
        )
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config.hidden_size,
                num_heads=config.num_attention_heads,
                d_ff=config.intermediate_size,
                use_alibi=config.use_alibi,
                max_len=config.max_position_embeddings
            )
            for _ in range(config.num_transformer_layers)
        ])
        
        # 输出层
        self.output_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 梯度检查点标志
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # 权重绑定（可选）
        self.tie_weights()
        
        # 初始化权重
        self.init_weights()
    
    def tie_weights(self):
        """绑定输入嵌入和输出投影层的权重"""
        self.lm_head.weight = self.token_embedding.token_emb.weight
    
    def get_input_embeddings(self):
        return self.token_embedding.token_emb
    
    def set_input_embeddings(self, value):
        self.token_embedding.token_emb = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutput]:
        """
        前向传播
        
        Args:
            input_ids: 输入token序列 [batch_size, sequence_length]
            attention_mask: 注意力掩码 [batch_size, sequence_length]
            labels: 标签序列，用于计算损失 [batch_size, sequence_length]
            use_cache: 是否使用缓存（暂未实现）
            output_attentions: 是否输出注意力权重
            output_hidden_states: 是否输出隐藏状态
            return_dict: 是否返回字典格式的输出
        
        Returns:
            CausalLMOutput或元组，包含损失、logits等
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is None:
            raise ValueError("input_ids不能为空")
        
        batch_size, seq_length = input_ids.shape
        
        # 1. Token嵌入
        hidden_states = self.token_embedding(input_ids)  # [B, L, d]
        
        # 2. 通过Enigma网络处理每个位置
        # 注意：Enigma网络对每个token位置独立处理
        enigma_outputs = []
        for i in range(seq_length):
            # 对每个位置应用Enigma变换
            token_hidden = hidden_states[:, i]  # [B, d]
            enigma_out = self.enigma(token_hidden)  # [B, d]
            enigma_outputs.append(enigma_out)
        
        hidden_states = torch.stack(enigma_outputs, dim=1)  # [B, L, d]
        
        # 3. 通过Transformer层
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        for i, transformer_block in enumerate(self.transformer_blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 梯度检查点
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(transformer_block),
                    hidden_states
                )
            else:
                layer_outputs = transformer_block(hidden_states)
            
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            else:
                hidden_states = layer_outputs
        
            # 4. 输出层
            hidden_states = self.output_norm(hidden_states)
            logits = self.lm_head(hidden_states)  # [B, L, vocab_size]
        
        # 5. 计算损失
        loss = None
        if labels is not None:
            # 移位标签用于因果语言建模
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # 添加Enigma正则化损失
            if hasattr(self.enigma, 'loss_regularizer'):
                reg_loss = self.enigma.loss_regularizer()
                loss = loss + reg_loss
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            output = (logits,)
            if loss is not None:
                output = (loss,) + output
            if all_hidden_states is not None:
                output = output + (all_hidden_states,)
            if all_attentions is not None:
                output = output + (all_attentions,)
            return output
        
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, use_cache=None, **kwargs
    ):
        """为生成准备输入"""
        # 如果模型已经运行过一步，我们只需要最后一个token
        if attention_mask is not None and attention_mask.ndim > 1:
            if attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """重新排序缓存以支持beam search"""
        # 当前版本暂不支持KV缓存
        return past_key_values


# 注册模型到transformers
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("enigma", EnigmaConfig)
AutoModelForCausalLM.register(EnigmaConfig, EnigmaForCausalLM) 