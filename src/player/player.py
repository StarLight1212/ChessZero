import random
import chess
from typing import List, Optional, Dict, Tuple
import numpy as np

from ..config import Config
from ..env import ChessEnv
from ..mcts import MCTS


class Player:
    """玩家基类"""
    
    def __init__(self, config: Config):
        """
        初始化玩家
        
        参数:
            config: 配置对象
        """
        self.config = config
        
    def get_action(self, env: ChessEnv, temperature: float = 1.0) -> str:
        """
        获取动作
        
        参数:
            env: 棋盘环境
            temperature: 温度参数（控制探索程度）
            
        返回:
            UCI格式的动作字符串
        """
        raise NotImplementedError("子类需要实现此方法")
    
    def __call__(self, env: ChessEnv, temperature: float = 1.0) -> str:
        """
        调用get_action方法
        
        参数:
            env: 棋盘环境
            temperature: 温度参数
            
        返回:
            UCI格式的动作字符串
        """
        return self.get_action(env, temperature)


class RandomPlayer(Player):
    """随机玩家"""
    
    def get_action(self, env: ChessEnv, temperature: float = 1.0) -> str:
        """
        随机选择一个合法动作
        
        参数:
            env: 棋盘环境
            temperature: 不使用
            
        返回:
            UCI格式的动作字符串
        """
        legal_moves = env.get_legal_moves()
        if not legal_moves:
            return None  # 没有合法动作，认输
        
        random_move = random.choice(legal_moves)
        return random_move.uci()


class HumanPlayer(Player):
    """人类玩家"""
    
    def get_action(self, env: ChessEnv, temperature: float = 1.0) -> str:
        """
        从用户输入获取动作
        
        参数:
            env: 棋盘环境
            temperature: 不使用
            
        返回:
            UCI格式的动作字符串
        """
        legal_moves = env.get_legal_moves()
        legal_moves_uci = [move.uci() for move in legal_moves]
        
        if not legal_moves:
            return None  # 没有合法动作，认输
        
        # 打印当前棋盘状态
        print(env.render())
        
        # 打印合法动作
        print(f"合法动作: {', '.join(legal_moves_uci)}")
        
        # 获取用户输入
        valid_input = False
        while not valid_input:
            move_uci = input("请输入你的动作 (UCI格式，例如e2e4): ")
            
            # 检查是否要认输
            if move_uci.lower() in ['q', 'quit', 'resign']:
                return None
                
            # 检查输入是否合法
            if move_uci in legal_moves_uci:
                valid_input = True
            else:
                print(f"非法动作，请重新输入。合法动作: {', '.join(legal_moves_uci)}")
        
        return move_uci


class AlphaZeroPlayer(Player):
    """AlphaZero AI玩家"""
    
    def __init__(self, config: Config, model):
        """
        初始化AlphaZero玩家
        
        参数:
            config: 配置对象
            model: AlphaZero模型
        """
        super().__init__(config)
        self.model = model
        self.mcts = MCTS(config, model)
        
        # 设置为评估模式
        self.model.eval()
        
    def get_action(self, env: ChessEnv, temperature: float = 1.0) -> str:
        """
        使用MCTS获取最佳动作
        
        参数:
            env: 棋盘环境
            temperature: 温度参数（控制探索程度）
            
        返回:
            UCI格式的动作字符串
        """
        # 如果没有合法动作，认输
        if not env.get_legal_moves():
            return None
            
        # 使用MCTS获取动作概率和选择的动作
        _, action = self.mcts.get_action_prob(env, temperature)
        
        if action is None:
            return None  # 没有选择动作，认输
            
        return action.uci()
