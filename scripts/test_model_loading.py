#!/usr/bin/env python
import sys
import os
import traceback
import torch
from scripts.inference_service import EnigmaInferenceService

def test_model_loading(checkpoint_dir="checkpoints_optimized"):
    """测试模型加载和生成功能"""
    print("\n===== 测试模型加载和生成功能 =====")
    
    try:
        # 显示CUDA和PyTorch信息
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA是否可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"当前设备: {torch.cuda.get_device_name(0)}")
            print(f"显存使用: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")
        
        # 检查检查点目录
        print(f"\n检查检查点目录: {checkpoint_dir}")
        if not os.path.exists(checkpoint_dir):
            print(f"错误: 检查点目录不存在: {checkpoint_dir}")
            return False
        
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        tokenizer_file = os.path.exists(os.path.join(checkpoint_dir, "tokenizer.pkl"))
        
        print(f"找到检查点文件: {checkpoint_files}")
        print(f"分词器文件存在: {tokenizer_file}")
        
        if not checkpoint_files:
            print("错误: 找不到模型检查点文件")
            return False
        
        # 尝试初始化服务
        print("\n尝试初始化EnigmaInferenceService...")
        service = EnigmaInferenceService(checkpoint_dir=checkpoint_dir)
        print("服务初始化成功!")
        
        # 尝试生成文本
        test_prompts = ["中国的历史", "人工智能的发展", "今天天气真好"]
        
        for prompt in test_prompts:
            print(f"\n===== 测试提示: '{prompt}' =====")
            try:
                result = service.generate(
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.2
                )
                
                print(f"生成成功! 用时: {result['generation_time_seconds']:.2f}秒")
                print(f"生成速度: {result['tokens_per_second']:.2f} token/秒")
                print(f"生成文本: {result['generated_text']}")
            except Exception as e:
                print(f"生成文本时出错: {e}")
                traceback.print_exc()
        
        return True
    
    except Exception as e:
        print(f"测试过程中出错: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 允许从命令行指定检查点目录
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_optimized"
    success = test_model_loading(checkpoint_dir)
    
    if success:
        print("\n===== 测试完成: 成功 =====")
    else:
        print("\n===== 测试完成: 失败 =====")
        sys.exit(1) 