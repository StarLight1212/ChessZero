#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
国际象棋AlphaZero训练脚本
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import logging
import datetime
import multiprocessing as mp

from src.config import Config
from src.main import train, set_seed


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AlphaZero国际象棋训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--iterations', type=int, default=None, help='训练迭代次数')
    parser.add_argument('--model', type=str, default=None, help='初始模型文件路径（如果不指定则随机初始化或使用best.pt）')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    mp.set_start_method('spawn')
    # torch.multiprocessing.set_sharing_strategy('file_system')
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = Config(args.config)
    
    # 如果指定了模型，更新配置
    if args.model:
        model_path = args.model
        # 确保模型存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            return 1
            
        # 更新配置
        config.config['system']['load_model'] = True
        config.best_model_path = model_path
    
    # 设置日志
    log_file = os.path.join(config.LOG_DIR, f"train_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 开始训练
    logging.info("开始AlphaZero训练")
    train(config, args.iterations)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
