#!/usr/bin/env python
import argparse
from scripts.inference_service import app, EnigmaInferenceService, service

def parse_args():
    parser = argparse.ArgumentParser(description='启动Enigma中文语言模型API服务')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints_optimized', 
                      help='检查点目录，默认为checkpoints_optimized')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务主机，默认为0.0.0.0')
    parser.add_argument('--port', type=int, default=5000, help='服务端口，默认为5000')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 预加载模型
    global service
    print(f"预加载模型从目录 {args.checkpoint_dir}...")
    service = EnigmaInferenceService(checkpoint_dir=args.checkpoint_dir)
    print("模型加载完成！")
    
    # 启动服务
    print(f"\n启动API服务: http://{args.host}:{args.port}")
    print("API端点:")
    print("  POST /generate - 生成文本")
    print("  POST /reload - 重新加载模型")
    print("  GET  /model_info - 获取模型信息")
    print("\n使用示例 (curl):")
    print(f'curl -X POST http://{args.host}:{args.port}/generate -H "Content-Type: application/json" -d \'{{"prompt": "中国的历史", "max_tokens": 100}}\'')
    print("\nCtrl+C 可停止服务\n")
    
    # 启动Flask服务
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main() 