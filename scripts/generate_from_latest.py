#!/usr/bin/env python
import sys
import argparse
from scripts.inference_service import EnigmaInferenceService

def parse_args():
    parser = argparse.ArgumentParser(description='EnigmaLM模型文本生成')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_optimized', 
                      help='检查点目录，默认为checkpoints_optimized')
    parser.add_argument('--prompt', type=str, default='', help='生成文本的提示')
    parser.add_argument('--max-tokens', type=int, default=200, help='生成的最大token数')
    parser.add_argument('--temperature', type=float, default=0.65, help='采样温度 (默认0.65)')
    parser.add_argument('--top-k', type=int, default=40, help='top-k采样 (默认40)')
    parser.add_argument('--top-p', type=float, default=0.88, help='nucleus采样概率阈值 (默认0.88)')
    parser.add_argument('--repetition-penalty', type=float, default=1.4, 
                      help='重复惩罚系数，>1会降低已生成token的概率 (默认1.4)')
    parser.add_argument('--search-strategy', type=str, choices=['default', 'beam', 'contrastive'], 
                      default='default', help='生成策略: default, beam, contrastive')
    parser.add_argument('--beam-size', type=int, default=4, help='Beam Search的束宽 (默认4)')
    parser.add_argument('--contrastive-alpha', type=float, default=0.6, 
                      help='对比搜索中的对比参数，控制语言模型和相似度的平衡 (默认0.6)')
    parser.add_argument('--interactive', action='store_true', help='交互模式')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化推理服务
    print(f"初始化Enigma推理服务，从目录 {args.checkpoint_dir} 加载最新模型...")
    service = EnigmaInferenceService(checkpoint_dir=args.checkpoint_dir)
    print("初始化完成！")
    
    if args.interactive:
        # 交互模式
        print("\n欢迎使用Enigma中文语言模型!")
        print("输入文本提示并按回车生成内容，输入'exit'或'quit'退出")
        print(f"当前生成策略: {args.search_strategy}, 温度: {args.temperature}, 重复惩罚: {args.repetition_penalty}\n")
        
        while True:
            try:
                prompt = input("提示> ")
                if prompt.lower() in ['exit', 'quit', '退出']:
                    break
                
                if not prompt:
                    continue
                
                result = service.generate(
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    search_strategy=args.search_strategy,
                    beam_size=args.beam_size,
                    contrastive_alpha=args.contrastive_alpha
                )
                
                print(f"\n生成结果:\n{result['generated_text']}")
                print(f"\n生成时间: {result['generation_time_seconds']:.2f}秒")
                print(f"生成速度: {result['tokens_per_second']:.2f} token/秒\n")
                print("-" * 80)
            
            except KeyboardInterrupt:
                print("\n退出中...")
                break
            except Exception as e:
                print(f"错误: {e}")
    else:
        # 单次生成模式
        if not args.prompt:
            print("错误: 非交互模式下需要提供--prompt参数")
            return
        
        try:
            print(f"使用策略: {args.search_strategy}")
            result = service.generate(
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                search_strategy=args.search_strategy,
                beam_size=args.beam_size,
                contrastive_alpha=args.contrastive_alpha
            )
            
            print(f"提示: {args.prompt}")
            print(f"\n生成结果:\n{result['generated_text']}")
            print(f"\n生成时间: {result['generation_time_seconds']:.2f}秒")
            print(f"生成速度: {result['tokens_per_second']:.2f} token/秒")
        
        except Exception as e:
            print(f"生成时出错: {e}")

if __name__ == "__main__":
    main() 