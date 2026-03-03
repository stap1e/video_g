#!/usr/bin/env python3
"""
使用 hf-mirror 下载模型
"""

import os
import subprocess
import argparse

def download_model(model_name="zai-org/CogVideoX-5b", save_dir="./models/CogVideoX-5b"):
    """
    使用 hf-mirror 下载 Hugging Face 模型
    
    Args:
        model_name (str): 模型名称，默认为 zai-org/CogVideoX-5b
        save_dir (str): 保存目录，默认为 ./models/CogVideoX-5b
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建下载命令
    command = [
        "hf", "download",
        "--repo-type", "model",
        "--local-dir", save_dir,
        model_name
    ]
    
    # 设置环境变量使用 hf-mirror
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    print(f"开始下载模型: {model_name}")
    print(f"保存目录: {save_dir}")
    print(f"使用镜像: {env['HF_ENDPOINT']}")
    
    try:
        # 执行下载命令，不捕获输出以显示进度
        print("下载进度:")
        result = subprocess.run(command, env=env, check=True)
        print("\n下载成功!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n下载失败: {e}")
        return False
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="使用 hf-mirror 下载模型")
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14", help="模型名称")
    parser.add_argument("--save-dir", default="/data/lhy_data/HunyuanVideo/clip-vit-large-patch14", help="保存目录")
    args = parser.parse_args()
    
    download_model(args.model_name, args.save_dir)

if __name__ == "__main__":
    main()