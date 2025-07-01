import os
import argparse
import random
import numpy as np
import torch
import logging
import time
from datetime import datetime

from .config import Config
from .model import AlphaZeroModel
from .training import Trainer, self_play, Evaluator
from .gui import start_gui
from .utils import setup_logger


def set_seed(seed):
    """设置随机种子，保证结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AlphaZero国际象棋')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--mode', type=str, choices=['train', 'play', 'gui'], default='train', help='运行模式')
    parser.add_argument('--iterations', type=int, default=None, help='训练迭代次数')
    parser.add_argument('--load_model', type=str, default=None, help='加载模型路径')
    return parser.parse_args()


def train(config, iterations=None):
    """
    训练AlphaZero模型
    
    参数:
        config: 配置对象
        iterations: 迭代次数（如果未指定，使用配置中的值）
    """
    # 设置日志
    log_file = os.path.join(config.LOG_DIR, f"train_main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logger("train_main", log_file)
    logger.info("开始训练AlphaZero模型")
    
    # 如果未指定迭代次数，使用配置中的值
    if iterations is None:
        iterations = config.config['training']['num_iterations']
        
    # 创建模型
    model = AlphaZeroModel(config)
    
    # 如果存在最佳模型，加载它
    if os.path.exists(config.best_model_path) and config.config['system']['load_model']:
        logger.info(f"加载最佳模型: {config.best_model_path}")
        model.load_model(config.best_model_path)
    else:
        logger.info("使用随机初始化的模型")
        # 保存初始模型作为当前最佳
        model.save_model(config.best_model_path)
    
    # 创建训练器和评估器
    trainer = Trainer(config, model)
    evaluator = Evaluator(config)
    
    # 开始迭代训练
    for iteration in range(iterations):
        logger.info(f"开始迭代 {iteration+1}/{iterations}")
        iteration_start_time = time.time()
        
        # 1. 自我对弈生成训练数据
        logger.info("开始自我对弈生成训练数据")
        game_data = self_play(config, model)
        logger.info(f"自我对弈完成，生成 {len(game_data)} 条训练数据")
        
        # 2. 使用生成的数据训练模型
        logger.info("开始训练模型")
        trainer.train(game_data)
        
        # 3. 保存当前迭代的模型（持续更新模式）
        model.save_model(config.best_model_path)  # 直接更新最佳模型
        iteration_model_path = os.path.join(config.CHECKPOINT_DIR, f"iteration_{iteration+1}.pt")
        model.save_model(iteration_model_path)
        
        # 4. 每20个episode进行一次评估（可选）
        if (iteration + 1) % 20 == 0:
            logger.info("定期评估模型")
            win_rate = evaluator.evaluate(model)
            logger.info(f"评估完成，当前模型胜率: {win_rate:.4f}")
        
        # 记录迭代耗时
        iteration_time = time.time() - iteration_start_time
        logger.info(f"迭代 {iteration+1} 完成，耗时: {iteration_time:.1f}秒")
    
    logger.info("训练完成!")


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = Config(args.config)
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        train(config, args.iterations)
    elif args.mode == 'gui':
        start_gui(config)
    else:
        print("不支持的模式")


if __name__ == "__main__":
    main() 