from .utils import setup_logger, normalize_encoding, save_game
import os
import logging
import chess.pgn
from typing import List, Dict, Optional
import numpy as np

__all__ = ['setup_logger', 'normalize_encoding', 'save_game']

def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def save_game(game_node, pgn_file):
    """保存棋局到PGN文件"""
    with open(pgn_file, "a") as f:
        print(game_node, file=f)
        print("\n", file=f)

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