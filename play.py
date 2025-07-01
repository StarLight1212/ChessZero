#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
国际象棋AlphaZero人机对弈脚本
"""

import os
import sys
import argparse
import logging
import datetime

from src.config import Config
from src.gui import start_gui


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AlphaZero国际象棋人机对弈')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, default=None, help='模型文件路径（默认使用best.pt）')
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 加载配置
    config = Config(args.config)
    
    # 如果指定了模型，更新配置
    if args.model:
        config.best_model_path = args.model
        
    # 设置日志
    log_file = os.path.join(config.LOG_DIR, f"play_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 启动GUI
    logging.info("启动国际象棋GUI")
    return start_gui(config)


if __name__ == "__main__":
    sys.exit(main()) 