#!/usr/bin/env python3
"""
使用 hf-mirror 下载 Hugging Face 数据集
"""

import os
import subprocess
import argparse

def download_dataset(dataset_name="maxin-cn/Taichi-HD", save_dir="./datasets/Taichi-HD", repo_type="dataset"):
    """
    使用 hf-mirror 下载 Hugging Face 数据集
    
    Args:
        dataset_name (str): 数据集名称，如 'maxin-cn/Taichi-HD'
        save_dir (str): 保存目录，默认为 ./datasets/Taichi-HD
        repo_type (str): 仓库类型，默认为 'dataset'
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 构建下载命令
    command = [
        "hf", "download",
        "--repo-type", repo_type,
        "--local-dir", save_dir,
        dataset_name
    ]
    
    # 设置环境变量使用 hf-mirror
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    print(f"开始下载数据集: {dataset_name}")
    print(f"保存目录: {save_dir}")
    print(f"使用镜像: {env['HF_ENDPOINT']}")
    
    try:
        # 执行下载命令，显示进度
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
    parser = argparse.ArgumentParser(description="使用 hf-mirror 下载 Hugging Face 数据集")
    parser.add_argument("--dataset-name", required=True, help="数据集名称")
    parser.add_argument("--save-dir", default="/data/lhy_data/video_data/datasets", help="保存目录")
    parser.add_argument("--repo-type", default="dataset", help="仓库类型")
    args = parser.parse_args()
    
    download_dataset(args.dataset_name, args.save_dir, args.repo_type)

if __name__ == "__main__":
    main()