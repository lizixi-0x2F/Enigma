#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import warnings
import json
from pathlib import Path

# 添加enigma模块到路径
sys.path.append(str(Path(__file__).parent))

from accelerate import Accelerator
from transformers import (
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    PreTrainedTokenizer,
    set_seed
)
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import logging

# 导入我们的自定义模型
from enigma.modeling_enigma import EnigmaConfig, EnigmaForCausalLM

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略一些警告
warnings.filterwarnings("ignore", category=UserWarning)

# 🎯 自定义Loss早停回调
class LossEarlyStoppingCallback(TrainerCallback):
    def __init__(self, loss_threshold=0.1, patience=3):
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.loss_below_threshold_count = 0
        self.should_stop = False
        
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        # 每步结束时检查，从当前batch的loss计算
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            # 获取最近的训练loss
            recent_logs = [log for log in state.log_history if 'loss' in log]
            if recent_logs:
                current_loss = recent_logs[-1]['loss']
                
                # 每步检查loss
                if current_loss < self.loss_threshold:
                    self.loss_below_threshold_count += 1
                    if Accelerator.is_main_process:
                        logger.info(f"🎯 步骤{state.global_step}: Loss={current_loss:.4f} < {self.loss_threshold} (第{self.loss_below_threshold_count}次)")
                    
                    if self.loss_below_threshold_count > self.patience:
                        if Accelerator.is_main_process:
                            logger.info(f"🛑 Loss小于{self.loss_threshold}已连续{self.loss_below_threshold_count}次，触发早停！")
                        control.should_training_stop = True
                        self.should_stop = True
                else:
                    # 如果Loss又上升了，重置计数器
                    if self.loss_below_threshold_count > 0:
                        if Accelerator.is_main_process:
                            logger.info(f"⚠️ 步骤{state.global_step}: Loss={current_loss:.4f}回升，重置计数器")
                        self.loss_below_threshold_count = 0

# 🎯 自定义PPL早停回调（SFT专用）
class PPLEarlyStoppingCallback(TrainerCallback):
    def __init__(self, ppl_threshold=25.0, patience=2):
        self.ppl_threshold = ppl_threshold
        self.patience = patience
        self.ppl_below_threshold_count = 0
        self.should_stop = False
        
    def on_step_end(self, args, state, control, logs=None, **kwargs):
        # 每步结束时检查PPL
        if hasattr(state, 'log_history') and len(state.log_history) > 0:
            # 获取最近的训练loss
            recent_logs = [log for log in state.log_history if 'loss' in log]
            if recent_logs:
                current_loss = recent_logs[-1]['loss']
                current_ppl = torch.exp(torch.tensor(current_loss)).item()
                
                # 每步检查PPL
                if current_ppl <= self.ppl_threshold:
                    self.ppl_below_threshold_count += 1
                    if Accelerator.is_main_process:
                        logger.info(f"🎯 SFT步骤{state.global_step}: Loss={current_loss:.4f}, PPL={current_ppl:.2f} ≤ {self.ppl_threshold} (第{self.ppl_below_threshold_count}次)")
                    
                    if self.ppl_below_threshold_count > self.patience:
                        if Accelerator.is_main_process:
                            logger.info(f"🛑 PPL≤{self.ppl_threshold}已连续{self.ppl_below_threshold_count}次，SFT微调完成！")
                        control.should_training_stop = True
                        self.should_stop = True
                else:
                    # 如果PPL又上升了，重置计数器
                    if self.ppl_below_threshold_count > 0:
                        if Accelerator.is_main_process:
                            logger.info(f"⚠️ SFT步骤{state.global_step}: Loss={current_loss:.4f}, PPL={current_ppl:.2f}回升，重置计数器")
                        self.ppl_below_threshold_count = 0
                
                # 额外记录PPL进展
                if state.global_step % 10 == 0 and Accelerator.is_main_process:
                    logger.info(f"📊 SFT进度: Step {state.global_step}, Loss={current_loss:.4f}, PPL={current_ppl:.2f}")

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
        logger.info(f"⚡ Enigma LoRA-SFT微调 - 使用 {accelerator.num_processes} 个GPU")
    
    # 加载预训练模型配置
    config = EnigmaConfig.from_pretrained("./output_full_sequence/final_model")
    config.max_position_embeddings = 2048
    
    if accelerator.is_main_process:
        logger.info(f"🤖 从预训练模型加载: output_full_sequence/final_model")
        logger.info(f"📊 模型配置: {config.num_transformer_layers}层, {config.hidden_size}维")
    
    # 🔥 使用完整的Enigma Tokenizer
    if accelerator.is_main_process:
        logger.info("🔧 加载完整Enigma Tokenizer...")
    
    tokenizer = EnigmaTokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    if accelerator.is_main_process:
        logger.info(f"✅ Tokenizer加载完成，词汇量: {tokenizer.vocab_size:,}")
    
    # 加载预训练模型
    if accelerator.is_main_process:
        logger.info("🔄 加载预训练模型...")
    
    model = EnigmaForCausalLM.from_pretrained("./output_full_sequence/final_model")
    
    if accelerator.is_main_process:
        logger.info(f"✅ 预训练模型加载完成，参数量: {model.num_parameters():,}")
    
    # 🎯 配置LoRA - 对注意力和前馈网络进行微调
    if accelerator.is_main_process:
        logger.info("🔍 检查模型层结构...")
        for name, module in model.named_modules():
            if 'transformer_blocks' in name and any(x in name for x in ['attention', 'mlp', 'in_proj', 'out_proj']):
                logger.info(f"  {name}: {type(module).__name__}")
    
    # 🎯 配置LoRA - 对关键层进行低秩增量
    # 注意力投影：in_proj_weight包含Q/K/V，out_proj是O投影
    # 前馈网络：mlp.0是FC1，mlp.2是FC2
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,  # scaling parameter
        target_modules=[
            # 注意力投影层 (Q/K/V/O)
            "attention.in_proj_weight",    # Q/K/V投影 (1536, 512) = 3 * (512, 512)
            "attention.out_proj",          # O投影 (512, 512)
            # 前馈网络层 (FC1/FC2)  
            "mlp.0",                       # FC1 (512 -> 2048)
            "mlp.2",                       # FC2 (2048 -> 512)
            # 输出层也保留
            "lm_head"
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    if accelerator.is_main_process:
        model.print_trainable_parameters()
        logger.info("🎯 LoRA配置完成，开始加载SFT数据...")
    
    # 📚 加载SFT数据并处理
    def load_sft_data(file_path):
        """加载并处理SFT数据"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                
                # 构建对话格式：系统提示词 + 指令 + 回答
                system_prompt = "你是Enigma"
                instruction = item.get('instruction', '')
                input_text = item.get('input', '')
                output_text = item.get('output', '')
                
                # 组合输入
                if input_text:
                    full_instruction = f"{instruction}\n{input_text}"
                else:
                    full_instruction = instruction
                
                # 构建训练文本格式
                prompt = f"<s>[INST] {system_prompt}\n\n{full_instruction} [/INST] {output_text}</s>"
                
                data.append({
                    'text': prompt,
                    'instruction': full_instruction,
                    'output': output_text
                })
        
        return data
    
    # 加载数据
    sft_data = load_sft_data("sft_data.jsonl")
    
    if accelerator.is_main_process:
        logger.info(f"📚 SFT数据加载完成: {len(sft_data):,} 条对话")
        logger.info(f"💡 示例对话长度: {len(sft_data[0]['text'])} 字符")
    
    # 数据预处理函数
    def preprocess_function(examples):
        """预处理函数：tokenize文本（字符级）"""
        texts = [item['text'] for item in examples]
        
        tokenized = []
        for text in texts:
            # 🔥 字符级tokenization - 直接使用字符
            chars = list(text)[:512]  # 截断到512字符进行SFT
            token_ids = [tokenizer._convert_token_to_id(char) for char in chars]
            
            tokenized.append({
                'input_ids': token_ids,
                'labels': token_ids.copy()  # 对于因果语言模型，labels和input_ids相同
            })
        
        return tokenized
    
    # 转换为Dataset
    processed_data = preprocess_function(sft_data)
    train_dataset = Dataset.from_list(processed_data)
    
    # 创建数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # 创建output目录
    if accelerator.is_main_process:
        os.makedirs("output_lora_sft", exist_ok=True)
    
    # 🎯 SFT训练参数（较小批次，适合微调）
    training_args = TrainingArguments(
        output_dir="output_lora_sft",
        per_device_train_batch_size=4,  # 较小批次适合SFT
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,  # 有效批次=4*8*5=160
        num_train_epochs=3,
        eval_strategy="no",  # 禁用评估策略
        save_steps=999999,  # 禁用中间checkpoint
        save_total_limit=1,
        logging_steps=10,  # 更频繁的日志
        fp16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        load_best_model_at_end=False,
        report_to=["tensorboard"],
        warmup_steps=50,
        learning_rate=1e-4,  # SFT使用较小学习率
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_safetensors=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        # 分布式优化
        ddp_find_unused_parameters=False,
        ddp_broadcast_buffers=False,
        ddp_bucket_cap_mb=100,
        # 性能优化
        fp16_full_eval=True,
        max_steps=1000,  # SFT不需要太多步骤
    )
    
    # 超高性能SFT Trainer类
    class UltraFastSFTTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            
            # 🔥🔥🔥 关键优化：向量化Enigma处理（同预训练）
            input_ids = inputs.get("input_ids")
            batch_size, seq_length = input_ids.shape
            
            # 处理DistributedDataParallel包装的模型
            actual_model = model.module if hasattr(model, 'module') else model
            
            # LoRA模型需要通过base_model访问
            if hasattr(actual_model, 'base_model'):
                base_model = actual_model.base_model.model
            else:
                base_model = actual_model
            
            # 1. Token嵌入
            hidden_states = base_model.token_embedding(input_ids)  # [B, L, d]
            
            # 2. 向量化Enigma处理
            flat_hidden = hidden_states.view(-1, hidden_states.size(-1))
            flat_enigma_out = base_model.enigma(flat_hidden)
            hidden_states = flat_enigma_out.view(batch_size, seq_length, -1)
            
            # 3. 通过Transformer层
            for transformer_block in base_model.transformer_blocks:
                hidden_states = transformer_block(hidden_states)
            
            # 4. 输出层
            hidden_states = base_model.output_norm(hidden_states)
            logits = base_model.lm_head(hidden_states)
            
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
                logger.info(f"⚡ SFT步骤 {step}: loss={logs['train_loss']:.4f}")
            
            super().log(logs)
    
    # 创建Loss早停回调（SFT通常loss会降得很低）
    loss_callback = LossEarlyStoppingCallback(loss_threshold=0.1, patience=3)
    
    # 创建PPL早停回调（SFT专用）
    ppl_callback = PPLEarlyStoppingCallback(ppl_threshold=25.0, patience=2)
    
    # 创建Trainer
    trainer = UltraFastSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[loss_callback, ppl_callback],
    )
    
    # 开始SFT训练
    if accelerator.is_main_process:
        logger.info("⚡⚡⚡ 开始Enigma LoRA-SFT微调...")
        logger.info(f"📊 训练数据: {len(sft_data):,} 条")
        logger.info(f"📊 有效批次: {4 * 8 * accelerator.num_processes}")
        logger.info(f"📊 序列长度: 512 (SFT优化)")
        logger.info(f"🎯 目标: Loss < 0.1, PPL ≤ 25")
    
    try:
        import time
        start_time = time.time()
        
        train_result = trainer.train()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if accelerator.is_main_process:
            steps_per_second = trainer.state.global_step / total_time
            logger.info(f"🎉 SFT微调完成！")
            logger.info(f"⚡ 总用时: {total_time:.2f}秒")
            logger.info(f"⚡ 训练速度: {steps_per_second:.3f} steps/秒")
            logger.info(f"⚡ 平均每步: {total_time/trainer.state.global_step:.1f}秒")
            
            # 保存LoRA适配器
            model.save_pretrained("output_lora_sft/lora_adapter")
            tokenizer.save_pretrained("output_lora_sft/lora_adapter")
            
            logger.info("💾 LoRA适配器已保存到 output_lora_sft/lora_adapter")
        
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logger.info("⏹️ SFT训练被用户中断")
            model.save_pretrained("output_lora_sft/interrupted_adapter")
    
    except Exception as e:
        if accelerator.is_main_process:
            logger.error(f"❌ SFT训练出现错误: {e}")
            import traceback
            traceback.print_exc()
        raise

if __name__ == "__main__":
    main() 