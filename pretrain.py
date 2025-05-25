#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import warnings
from pathlib import Path
import json

# 添加enigma模块到路径
sys.path.append(str(Path(__file__).parent))

from accelerate import Accelerator
from transformers import (
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
    PreTrainedTokenizer,
    set_seed
)
from datasets import load_from_disk
import logging

# 导入我们的自定义模型
from enigma.modeling_enigma import EnigmaConfig, EnigmaForCausalLM

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略一些警告
warnings.filterwarnings("ignore", category=UserWarning)

# 🎯 自定义PPL早停回调
class PPLEarlyStoppingCallback(TrainerCallback):
    def __init__(self, ppl_threshold=25, patience=2):
        self.ppl_threshold = ppl_threshold
        self.patience = patience
        self.ppl_below_threshold_count = 0
        self.should_stop = False
        
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        # 每步结束时检查，从当前batch的loss计算PPL
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            # 获取最近的训练loss
            recent_logs = [log for log in state.log_history if 'loss' in log]
            if recent_logs:
                current_loss = recent_logs[-1]['loss']
                current_ppl = torch.exp(torch.tensor(current_loss))
                
                # 每步检查PPL
                if current_ppl < self.ppl_threshold:
                    self.ppl_below_threshold_count += 1
                    if Accelerator.is_main_process:
                        logger.info(f"🎯 步骤{state.global_step}: PPL={current_ppl:.2f} < {self.ppl_threshold} (第{self.ppl_below_threshold_count}次)")
                    
                    if self.ppl_below_threshold_count > self.patience:
                        if Accelerator.is_main_process:
                            logger.info(f"🛑 PPL小于{self.ppl_threshold}已连续{self.ppl_below_threshold_count}次，触发早停！")
                        control.should_training_stop = True
                        self.should_stop = True
                else:
                    # 如果PPL又上升了，重置计数器
                    if self.ppl_below_threshold_count > 0:
                        if Accelerator.is_main_process:
                            logger.info(f"⚠️ 步骤{state.global_step}: PPL={current_ppl:.2f}回升，重置计数器")
                        self.ppl_below_threshold_count = 0

# 🔥 完整Enigma Tokenizer
class EnigmaTokenizer(PreTrainedTokenizer):
    """基于真实数据训练的Enigma Tokenizer"""
    
    def __init__(self, vocab_file="enigma_tokenizer/vocab.json", **kwargs):
        # 加载词汇表
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self._vocab = json.load(f)
        
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        
        # 设置特殊token
        super().__init__(
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            **kwargs
        )
    
    @property
    def vocab_size(self):
        return len(self._vocab)
    
    def _tokenize(self, text):
        """字符级分词"""
        return list(text)
    
    def _convert_token_to_id(self, token):
        return self._vocab.get(token, self._vocab.get("<unk>", 0))
    
    def _convert_id_to_token(self, index):
        return self._id_to_token.get(index, "<unk>")
    
    def get_vocab(self):
        return self._vocab
    
    def save_vocabulary(self, save_directory, filename_prefix=None):
        """保存词汇表文件"""
        if filename_prefix is None:
            filename_prefix = ""
        
        vocab_file = os.path.join(save_directory, f"{filename_prefix}vocab.json")
        
        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self._vocab, f, ensure_ascii=False, indent=2)
        
        return (vocab_file,)

def main():
    # 设置随机种子
    set_seed(42)
    
    # 初始化Accelerator
    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        logger.info(f"⚡ 超级优化训练 - 使用 {accelerator.num_processes} 个GPU")
    
    # 加载配置并保持原始序列长度
    config = EnigmaConfig.from_pretrained("./config.json")
    # 🔥 使用完整序列长度2048，不截断
    config.max_position_embeddings = 2048
    
    if accelerator.is_main_process:
        logger.info(f"🔥 完整序列训练: 2048 (无截断)")
        logger.info(f"模型配置: {config.num_transformer_layers}层, {config.hidden_size}维")
    
    # 🔥 使用完整的Enigma Tokenizer
    if accelerator.is_main_process:
        logger.info("🔧 加载完整Enigma Tokenizer...")
    
    tokenizer = EnigmaTokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    if accelerator.is_main_process:
        logger.info(f"✅ Tokenizer加载完成，词汇量: {tokenizer.vocab_size:,}")
    
    # 创建模型
    if accelerator.is_main_process:
        logger.info("创建模型中...")
    
    model = EnigmaForCausalLM(config)
    
    if accelerator.is_main_process:
        logger.info(f"✅ 模型创建完成，参数量: {model.num_parameters():,}")
    
    # 加载完整数据集，不截断
    if accelerator.is_main_process:
        logger.info("加载完整数据集（不截断序列）...")
    
    dataset = load_from_disk("am_0.9M_processed_hf")
    
    # 直接使用原始数据，不截断
    train_ds = dataset["train"]
    eval_ds = dataset["validation"]
    
    if accelerator.is_main_process:
        logger.info(f"✅ 训练集: {len(train_ds):,} 样本 (序列长度: 2048)")
        logger.info(f"✅ 验证集: {len(eval_ds):,} 样本 (序列长度: 2048)")
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        return_tensors="pt"
    )
    
    # 创建output目录
    if accelerator.is_main_process:
        os.makedirs("output_full_sequence", exist_ok=True)
    
    # 🔥 调整批次大小以适应2048序列长度
    # 序列长度4倍增加，需要减小批次以避免OOM
    per_device_batch_size = 8   # 从32减少到8
    gradient_accumulation = 4   # 从2增加到4，保持有效批次不变
    
    if accelerator.is_main_process:
        effective_batch = per_device_batch_size * gradient_accumulation * accelerator.num_processes
        logger.info(f"⚡ 完整序列批次配置:")
        logger.info(f"  - 每设备批次: {per_device_batch_size}")
        logger.info(f"  - 梯度累积: {gradient_accumulation}")
        logger.info(f"  - GPU数量: {accelerator.num_processes}")
        logger.info(f"  - 有效批次: {effective_batch}")
        logger.info(f"  - 序列长度: 2048 (完整)")
    
    # ⚡ 高性能训练参数（适配完整序列）
    training_args = TrainingArguments(
        output_dir="output_full_sequence",
        per_device_train_batch_size=per_device_batch_size,    # 8，适应2048序列
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=999999,  # 🗑️ 禁用中间checkpoint，设置超大值
        save_total_limit=1,  # 只保留1个checkpoint
        logging_steps=25,
        fp16=True,
        dataloader_num_workers=6,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=3,
        load_best_model_at_end=False,
        report_to=["tensorboard"],
        warmup_steps=200,
        learning_rate=3e-4,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_safetensors=False,              # 🔥 修复权重共享保存问题
        gradient_checkpointing=False,    # 保持关闭获得最大速度
        remove_unused_columns=False,
        # 🔥 分布式优化
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        ddp_bucket_cap_mb=100,
        # 🔥 性能优化
        fp16_full_eval=True,
        max_steps=5000,
    )
    
    # 超高性能Trainer类
    class UltraFastEnigmaTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            
            # 🔥🔥🔥 关键优化：向量化Enigma处理
            input_ids = inputs.get("input_ids")
            batch_size, seq_length = input_ids.shape
            
            # 处理DistributedDataParallel包装的模型
            actual_model = model.module if hasattr(model, 'module') else model
            
            # 1. Token嵌入
            hidden_states = actual_model.token_embedding(input_ids)  # [B, L, d]
            
            # 2. 向量化Enigma处理 - 将所有位置展平一次性处理
            # 原来：for i in range(seq_length): enigma(x[:, i])
            # 优化：enigma(x.view(-1, d)).view(B, L, d)
            flat_hidden = hidden_states.view(-1, hidden_states.size(-1))  # [B*L, d]
            flat_enigma_out = actual_model.enigma(flat_hidden)  # [B*L, d] - 一次性处理所有位置！
            hidden_states = flat_enigma_out.view(batch_size, seq_length, -1)  # [B, L, d]
            
            # 3. 通过Transformer层
            for transformer_block in actual_model.transformer_blocks:
                hidden_states = transformer_block(hidden_states)
            
            # 4. 输出层
            hidden_states = actual_model.output_norm(hidden_states)
            logits = actual_model.lm_head(hidden_states)  # [B, L, vocab_size]
            
            # 5. 计算损失
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = None
            
            return (loss, logits) if return_outputs else loss
        
        def log(self, logs, start_time=None):
            if accelerator.is_main_process and "train_loss" in logs:
                step = self.state.global_step
                
                # 详细性能监控
                if hasattr(self, '_step_times'):
                    import time
                    current_time = time.time()
                    if len(self._step_times) > 0:
                        avg_step_time = (current_time - self._step_times[0]) / len(self._step_times)
                        logger.info(f"⚡ 步骤 {step}: loss={logs['train_loss']:.4f}, 平均步时={avg_step_time:.1f}s")
                    self._step_times.append(current_time)
                    if len(self._step_times) > 10:  # 只保留最近10步
                        self._step_times.pop(0)
                else:
                    self._step_times = []
            
            super().log(logs)
    
    # 创建PPL早停回调
    ppl_callback = PPLEarlyStoppingCallback(ppl_threshold=25, patience=2)
    
    # 创建Trainer
    trainer = UltraFastEnigmaTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[ppl_callback],  # 🎯 添加PPL早停回调
    )
    
    # 开始训练
    if accelerator.is_main_process:
        logger.info("⚡⚡⚡ 开始超级优化训练...")
        logger.info(f"📊 序列长度: 2048 (完整)")
        logger.info(f"📊 有效批次: {per_device_batch_size * gradient_accumulation * accelerator.num_processes}")
        logger.info(f"📊 预期每步时间: ~15-30秒")
        logger.info(f"🎯 目标: PPL ≤ 25")
    
    try:
        import time
        start_time = time.time()
        
        train_result = trainer.train()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if accelerator.is_main_process:
            steps_per_second = trainer.state.global_step / total_time
            logger.info(f"🎉 训练完成！")
            logger.info(f"⚡ 总用时: {total_time:.2f}秒")
            logger.info(f"⚡ 训练速度: {steps_per_second:.3f} steps/秒")
            logger.info(f"⚡ 平均每步: {total_time/trainer.state.global_step:.1f}秒")
            
            # 保存模型
            trainer.save_model("output_full_sequence/final_model")
            tokenizer.save_pretrained("output_full_sequence/final_model")
        
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logger.info("⏹️ 训练被用户中断")
            trainer.save_model("output_full_sequence/interrupted_model")
    
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"❌ 训练出现错误: {e}")
            import traceback
            traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 