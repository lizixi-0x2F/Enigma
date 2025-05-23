import torch
import pickle
import argparse
import os
import json
import time
import math
from tqdm import tqdm
from flask import Flask, request, jsonify
from enigma.model import EnigmaLM
from transformers import BertTokenizer

app = Flask(__name__)

class BertChineseTokenizer:
    """BERT中文分词器包装类"""
    
    def __init__(self, model_name='bert-base-chinese'):
        # 检查是否为本地路径
        if os.path.exists(model_name):
            print(f"从本地路径加载分词器: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name, local_files_only=True)
        else:
            print(f"从Hugging Face加载分词器: {model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            
        self.pad_token_id = self.tokenizer.pad_token_id
        self.bos_token_id = self.tokenizer.cls_token_id  # 使用CLS作为开始标记
        self.eos_token_id = self.tokenizer.sep_token_id  # 使用SEP作为结束标记
        self.vocab_size = len(self.tokenizer)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
    
    def encode(self, text, add_special_tokens=True):
        """编码文本为token ids"""
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=512,
            truncation=True
        )
    
    def decode(self, ids, skip_special_tokens=True):
        """解码token ids为文本"""
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

def load_tokenizer(tokenizer_path):
    """加载分词器"""
    try:
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"成功从 {tokenizer_path} 加载分词器")
        return tokenizer
    except Exception as e:
        print(f"加载分词器时出错: {e}")
        # 尝试从默认位置加载
        try:
            # 直接创建BertChineseTokenizer实例，避免序列化问题
            tokenizer = BertChineseTokenizer("bert-chinese-base")
            print("从默认路径加载分词器成功")
            return tokenizer
        except Exception as e2:
            print(f"从默认路径加载分词器时出错: {e2}")
            raise

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """对logits进行top-k和top-p过滤"""
    top_k = min(top_k, logits.size(-1))  # 安全检查
    
    if top_k > 0:
        # 移除所有不在top k的token
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    
    if top_p > 0.0:
        # 计算累积概率分布
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # 移除累积概率超过阈值的token
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将第一个token保留（避免全部被过滤）
        sorted_indices_to_remove[..., 0] = 0

        # 恢复原始索引顺序并过滤
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    
    return logits

def apply_repetition_penalty(next_token_logits, generated_tokens, penalty=1.0):
    """应用重复惩罚"""
    if len(generated_tokens) > 0:
        for token in set(generated_tokens):
            # 如果token在生成的序列中出现过，就降低其概率
            next_token_logits[token] /= penalty
    return next_token_logits

def beam_search_generate(model, tokenizer, prompt='', beam_size=5, max_tokens=100, 
                        temperature=0.65, top_k=40, top_p=0.88, repetition_penalty=1.4, device='cuda'):
    """使用Beam Search生成文本"""
    model.eval()
    
    # 编码提示文本
    if prompt:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # 如果没有提示，则使用BOS token开始
        input_ids = torch.tensor([tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else tokenizer.cls_token_id], 
                                 dtype=torch.long, device=device).unsqueeze(0)
    
    # 初始化beams: [(token_ids, score)]
    beams = [(input_ids[0].tolist(), 0.0)]
    finished_beams = []
    
    with torch.no_grad():
        for _ in tqdm(range(max_tokens), desc="Beam搜索生成中"):
            if not beams:
                break
                
            # 扩展所有当前beam
            candidates = []
            for beam_tokens, beam_score in beams:
                # 如果beam已经完成（生成了EOS），将其加入finished_beams
                if beam_tokens[-1] == (tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.sep_token_id):
                    finished_beams.append((beam_tokens, beam_score))
                    continue
                
                # 将beam转换为tensor
                curr_input_ids = torch.tensor(beam_tokens, dtype=torch.long, device=device).unsqueeze(0)
                
                # 获取模型输出
                outputs = model(curr_input_ids)
                
                # 获取最后一个token的logits
                next_token_logits = outputs[:, -1, :].squeeze(0)
                
                # 应用重复惩罚
                if repetition_penalty > 1.0:
                    for token in set(beam_tokens):
                        next_token_logits[token] /= repetition_penalty
                
                # 应用温度
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # 应用top-k和top-p过滤
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits, 
                    top_k=top_k, 
                    top_p=top_p
                )
                
                # 计算概率
                probs = torch.softmax(filtered_logits, dim=-1)
                
                # 获取top-beam_size个候选
                top_probs, top_indices = torch.topk(probs, k=beam_size)
                
                # 将候选添加到列表
                for prob, token_id in zip(top_probs, top_indices):
                    # 计算新的beam_score (对数概率之和)
                    new_beam_score = beam_score - torch.log(prob).item()
                    candidates.append((beam_tokens + [token_id.item()], new_beam_score))
            
            # 如果没有候选，则退出
            if not candidates:
                break
            
            # 排序并保留top-beam_size个候选
            beams = sorted(candidates, key=lambda x: x[1])[:beam_size]
    
    # 如果有已完成的beam，使用它们，否则使用剩余的beam
    if finished_beams:
        finished_beams = sorted(finished_beams, key=lambda x: x[1])
        best_beam = finished_beams[0][0]
    else:
        best_beam = sorted(beams, key=lambda x: x[1])[0][0]
    
    # 解码生成的token序列
    raw_text = tokenizer.decode(best_beam, skip_special_tokens=True)
    
    # 后处理：移除字符之间的空格
    text = raw_text.replace(" ", "")
    return text

def contrastive_search_generate(model, tokenizer, prompt='', max_tokens=100, 
                              top_k=4, alpha=0.6, temperature=0.65, repetition_penalty=1.4, device='cuda'):
    """使用Contrastive Search生成文本，减少重复和退化"""
    model.eval()
    
    # 编码提示文本
    if prompt:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # 如果没有提示，则使用BOS token开始
        input_ids = torch.tensor([tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else tokenizer.cls_token_id], 
                                 dtype=torch.long, device=device).unsqueeze(0)
    
    # 初始化生成的token列表
    generated_tokens = input_ids[0].tolist()
    past_key_values = None
    
    # 存储过去的hidden states用于对比
    past_hidden_states = []
    
    with torch.no_grad():
        for _ in tqdm(range(max_tokens), desc="对比搜索生成中"):
            # 获取模型输出
            outputs = model(input_ids)
            
            # 获取最后一个token的logits和hidden states
            next_token_logits = outputs[:, -1, :].squeeze(0)
            hidden_state = outputs[:, -1, :].squeeze(0)
            
            # 应用重复惩罚
            if repetition_penalty > 1.0:
                for token in set(generated_tokens):
                    next_token_logits[token] /= repetition_penalty
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 获取top-k个候选token
            values, indices = torch.topk(next_token_logits, top_k)
            candidates = []
            
            # 计算对比度分数
            for token_id in indices:
                # 创建一个假想的下一个token
                new_hidden_state = hidden_state.clone()
                
                # 计算与历史hidden states的相似度
                if past_hidden_states:
                    # 计算余弦相似度
                    similarities = []
                    for past_state in past_hidden_states:
                        sim = torch.cosine_similarity(new_hidden_state, past_state, dim=0)
                        similarities.append(sim.item())
                    
                    # 获取最大相似度
                    max_similarity = max(similarities) if similarities else 0
                    
                    # 对比分数 = alpha * 语言模型分数 - (1-alpha) * 最大相似度
                    token_prob = torch.softmax(next_token_logits, dim=0)[token_id].item()
                    score = alpha * math.log(token_prob) - (1-alpha) * max_similarity
                else:
                    # 如果没有历史状态，只使用语言模型分数
                    token_prob = torch.softmax(next_token_logits, dim=0)[token_id].item()
                    score = math.log(token_prob)
                
                candidates.append((token_id.item(), score))
            
            # 选择分数最高的token
            next_token = max(candidates, key=lambda x: x[1])[0]
            
            # 存储当前hidden state
            past_hidden_states.append(hidden_state.detach().clone())
            if len(past_hidden_states) > 4:  # 只保留最近的n个状态
                past_hidden_states.pop(0)
            
            # 添加到生成的序列
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
            
            # 如果生成了EOS token，就停止生成
            if next_token == (tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.sep_token_id):
                break
    
    # 解码生成的token序列
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 后处理：移除字符之间的空格
    text = raw_text.replace(" ", "")
    return text

def generate_text(model, tokenizer, prompt='', max_tokens=100, temperature=0.65, 
                 top_k=40, top_p=0.88, repetition_penalty=1.4, device='cuda',
                 use_kv_cache=True):
    """ 生成文本函数，支持KV缓存 """
    model.eval()
    
    # 编码提示文本
    if prompt:
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    else:
        # 如果没有提示，则使用BOS token开始
        input_ids = torch.tensor([tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else tokenizer.cls_token_id], 
                                 dtype=torch.long, device=device).unsqueeze(0)
    
    # 初始化生成的token列表
    generated_tokens = input_ids[0].tolist()
    
    # 初始化KV缓存
    kv_cache = None
    
    # 开始生成
    with torch.no_grad():
        for _ in tqdm(range(max_tokens), desc="生成中"):
            # 获取模型输出
            if use_kv_cache and kv_cache is not None:
                # 只使用最后一个token作为输入，并传递KV缓存
                current_input = input_ids[:, -1].unsqueeze(-1)
                outputs = model(current_input, kv_cache=kv_cache)
            else:
                # 第一次或不使用KV缓存时处理整个序列
                outputs = model(input_ids)
            
            # 检查输出格式
            if isinstance(outputs, tuple):
                logits = outputs[0]
                if len(outputs) > 1:
                    kv_cache = outputs[1:]
            else:
                logits = outputs
                kv_cache = None
            
            # 获取最后一个token的logits
            if len(logits.shape) == 3:
                next_token_logits = logits[:, -1, :].squeeze(0)
            else:
                # 如果是二维张量，直接使用
                next_token_logits = logits.squeeze(0)
            
            # 应用重复惩罚
            if repetition_penalty > 1.0:
                next_token_logits = apply_repetition_penalty(
                    next_token_logits, 
                    generated_tokens, 
                    repetition_penalty
                )
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 应用top-k和top-p过滤
            filtered_logits = top_k_top_p_filtering(
                next_token_logits, 
                top_k=top_k, 
                top_p=top_p
            )
            
            # 采样下一个token
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # 添加到生成的序列
            generated_tokens.append(next_token.item())
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)
            
            # 如果生成了EOS token，就停止生成
            if next_token.item() == (tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else tokenizer.sep_token_id):
                break
    
    # 解码生成的token序列
    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 后处理：移除字符之间的空格（针对中文）
    text = raw_text.replace(" ", "")
    return text

def find_latest_checkpoint(checkpoint_dir="checkpoints_optimized"):
    """查找最新的检查点文件"""
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"检查点目录不存在: {checkpoint_dir}")
    
    # 首先检查是否有best_model.pt或final_model.pt
    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    
    if os.path.exists(best_model_path):
        return best_model_path
    
    if os.path.exists(final_model_path):
        return final_model_path
    
    # 查找最新的epoch或step检查点
    checkpoints = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_") and filename.endswith(".pt"):
            checkpoints.append(os.path.join(checkpoint_dir, filename))
    
    if not checkpoints:
        raise FileNotFoundError(f"在目录 {checkpoint_dir} 中找不到有效的检查点文件")
    
    # 按文件修改时间排序，取最新的
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint

class EnigmaInferenceService:
    """Enigma模型推理服务"""
    
    def __init__(self, checkpoint_dir="checkpoints_optimized", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.tokenizer = None
        self.model_config = None
        
        # 加载模型和分词器
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        """加载最新的模型和分词器"""
        try:
            # 查找最新的检查点
            model_path = find_latest_checkpoint(self.checkpoint_dir)
            print(f"使用模型检查点: {model_path}")
            
            # 加载分词器
            tokenizer_path = os.path.join(self.checkpoint_dir, "tokenizer.pkl")
            self.tokenizer = load_tokenizer(tokenizer_path)
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 从checkpoint提取模型配置
            # 如果checkpoint中没有配置信息，使用训练命令中的默认值
            self.model_config = {
                'vocab_size': getattr(self.tokenizer, 'vocab_size', 21128),
                'd_model': 768,
                'num_rev_blocks': 6,
                'num_rotors': 4,
                'num_transformer_layers': 12,
                'num_heads': 12,
                'max_len': 8192,
                'use_alibi': True
            }
            
            print(f"使用模型配置: {self.model_config}")
            
            # 创建模型
            self.model = EnigmaLM(
                vocab_size=self.model_config['vocab_size'],
                d=self.model_config['d_model'],
                num_rev_blocks=self.model_config['num_rev_blocks'],
                num_rotors=self.model_config['num_rotors'],
                num_transformer_layers=self.model_config['num_transformer_layers'],
                num_heads=self.model_config['num_heads'],
                max_len=self.model_config['max_len'],
                use_alibi=self.model_config['use_alibi']
            ).to(self.device)
            
            # 加载模型权重，使用strict=False忽略额外的键（如alibi_cache）
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("模型和分词器加载成功")
            
            # 计算模型大小
            model_size_mb = sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024  # 4 bytes per float32
            print(f"模型大小: {model_size_mb:.2f} MB")
            
            return True
        except Exception as e:
            print(f"加载模型或分词器时出错: {e}")
            raise
    
    def generate(self, prompt="", max_tokens=200, temperature=0.65, top_k=40, top_p=0.88, repetition_penalty=1.4,
                search_strategy="default", beam_size=4, contrastive_alpha=0.6):
        """生成文本的主要接口"""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型或分词器未加载")
        
        start_time = time.time()
        
        try:
            # 根据不同的搜索策略选择生成函数
            if search_strategy == "beam":
                generated_text = beam_search_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    beam_size=beam_size,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    device=self.device
                )
            elif search_strategy == "contrastive":
                generated_text = contrastive_search_generate(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    top_k=min(top_k, 10),  # 对于对比搜索，我们使用较小的top_k
                    alpha=contrastive_alpha,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    device=self.device
                )
            else:  # default
                generated_text = generate_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    device=self.device,
                    use_kv_cache=True
                )
            
            generation_time = time.time() - start_time
            tokens_per_second = max_tokens / generation_time if generation_time > 0 else 0
            
            result = {
                "generated_text": generated_text,
                "generation_time_seconds": generation_time,
                "tokens_per_second": tokens_per_second,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "search_strategy": search_strategy,
                "settings": {
                    "temperature": temperature,
                    "top_k": top_k,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "beam_size": beam_size if search_strategy == "beam" else None,
                    "contrastive_alpha": contrastive_alpha if search_strategy == "contrastive" else None
                }
            }
            
            return result
        except Exception as e:
            print(f"生成文本时出错: {e}")
            raise

# 创建服务实例
service = None

@app.route('/generate', methods=['POST'])
def api_generate():
    """API端点：生成文本"""
    global service
    
    # 确保服务已初始化
    if service is None:
        try:
            service = EnigmaInferenceService()
        except Exception as e:
            return jsonify({"error": f"初始化服务失败: {str(e)}"}), 500
    
    # 解析参数
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = int(data.get('max_tokens', 200))
        temperature = float(data.get('temperature', 0.65))
        top_k = int(data.get('top_k', 40))
        top_p = float(data.get('top_p', 0.88))
        repetition_penalty = float(data.get('repetition_penalty', 1.4))
        search_strategy = data.get('search_strategy', 'default')
        beam_size = int(data.get('beam_size', 4))
        contrastive_alpha = float(data.get('contrastive_alpha', 0.6))
        
        # 参数验证
        if max_tokens <= 0 or max_tokens > 1000:
            return jsonify({"error": "max_tokens必须在1-1000之间"}), 400
        
        if temperature <= 0 or temperature > 2.0:
            return jsonify({"error": "temperature必须在0-2之间"}), 400
        
        if top_p <= 0 or top_p > 1.0:
            return jsonify({"error": "top_p必须在0-1之间"}), 400
        
        if search_strategy not in ['default', 'beam', 'contrastive']:
            return jsonify({"error": "search_strategy必须是default、beam或contrastive之一"}), 400
    except Exception as e:
        return jsonify({"error": f"参数解析错误: {str(e)}"}), 400
    
    # 生成文本
    try:
        result = service.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            search_strategy=search_strategy,
            beam_size=beam_size,
            contrastive_alpha=contrastive_alpha
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"生成文本失败: {str(e)}"}), 500

@app.route('/reload', methods=['POST'])
def api_reload():
    """API端点：重新加载模型"""
    global service
    
    try:
        if service is None:
            service = EnigmaInferenceService()
        else:
            service.load_model_and_tokenizer()
        return jsonify({"status": "success", "message": "模型重新加载成功"})
    except Exception as e:
        return jsonify({"error": f"重新加载模型失败: {str(e)}"}), 500

@app.route('/model_info', methods=['GET'])
def api_model_info():
    """API端点：获取模型信息"""
    global service
    
    if service is None:
        return jsonify({"error": "服务未初始化"}), 500
    
    try:
        # 获取当前使用的检查点路径
        model_path = find_latest_checkpoint(service.checkpoint_dir)
        
        # 计算模型大小
        model_size_mb = sum(p.numel() for p in service.model.parameters()) * 4 / 1024 / 1024  # 4 bytes per float32
        
        info = {
            "model_checkpoint": model_path,
            "model_config": service.model_config,
            "model_size_mb": round(model_size_mb, 2),
            "device": service.device,
            "tokenizer_vocab_size": getattr(service.tokenizer, 'vocab_size', 'unknown')
        }
        
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": f"获取模型信息失败: {str(e)}"}), 500

def parse_args():
    parser = argparse.ArgumentParser(description='EnigmaLM生成服务')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_optimized', 
                      help='检查点目录，默认为checkpoints_optimized')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机，默认为0.0.0.0')
    parser.add_argument('--port', type=int, default=5000, help='服务端口，默认为5000')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    return parser.parse_args()

def direct_generate(prompt, max_tokens=200):
    """直接生成文本（命令行模式）"""
    global service
    
    if service is None:
        service = EnigmaInferenceService()
    
    result = service.generate(
        prompt=prompt,
        max_tokens=max_tokens
    )
    
    print(f"提示: {prompt}")
    print(f"生成: {result['generated_text']}")
    print(f"生成时间: {result['generation_time_seconds']:.2f}秒")
    print(f"生成速度: {result['tokens_per_second']:.2f} token/秒")

if __name__ == "__main__":
    args = parse_args()
    
    # 初始化服务
    service = EnigmaInferenceService(checkpoint_dir=args.checkpoint_dir)
    
    # 启动Flask服务
    app.run(host=args.host, port=args.port, debug=args.debug) 