import os
import logging
import numpy as np
import random
import torch
import time
import chess
import chess.pgn
from typing import List, Dict, Tuple, Optional, Union
import datetime


def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    设置日志记录器
    
    参数:
        name: 日志记录器名称
        log_file: 日志文件路径
        level: 日志级别
        
    返回:
        日志记录器对象
    """
    logger = logging.getLogger(name)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 设置级别
    logger.setLevel(level)
    
    return logger


def normalize_encoding(policy_array: List[float], valid_moves: List[str], move_lookup: Dict[str, int]) -> List[float]:
    """
    归一化策略编码
    
    参数:
        policy_array: 策略概率数组
        valid_moves: 有效移动列表
        move_lookup: 移动到索引的映射
        
    返回:
        归一化后的策略数组
    """
    # 创建新的策略数组
    normalized_policy = np.zeros_like(policy_array)
    
    # 提取合法移动的概率
    legal_probs = []
    legal_indices = []
    
    for move in valid_moves:
        if move in move_lookup:
            idx = move_lookup[move]
            legal_probs.append(policy_array[idx])
            legal_indices.append(idx)
    
    # 归一化
    legal_probs = np.array(legal_probs)
    if np.sum(legal_probs) > 0:
        legal_probs = legal_probs / np.sum(legal_probs)
    else:
        # 如果所有概率都是0，使用均匀分布
        legal_probs = np.ones_like(legal_probs) / len(legal_probs)
    
    # 更新数组
    for i, idx in enumerate(legal_indices):
        normalized_policy[idx] = legal_probs[i]
    
    return normalized_policy


def save_game(game: chess.pgn.Game, path: str):
    """
    保存棋局到PGN文件
    
    参数:
        game: 棋局对象
        path: 保存路径
    """
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 设置头信息
    game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = "AlphaZero"
    game.headers["Black"] = "AlphaZero"
    game.headers["Event"] = "AlphaZero Self-Play"
    
    # 保存到文件
    with open(path, "a") as f:
        f.write(str(game) + "\n\n")
