import numpy as np
import os
import time
import random
from multiprocessing import Pool, Manager
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

from ..config import Config
from ..model import AlphaZeroModel
from ..env import ChessEnv, Winner
from ..mcts import MCTS
from ..utils import setup_logger


def play_game(config: Config, cur_model: AlphaZeroModel, best_model: AlphaZeroModel, 
              game_id: int, cur_plays_white: bool = True, verbose: bool = False) -> Tuple[int, int, int]:
    """
    在当前模型和最佳模型之间进行一场比赛
    
    参数:
        config: 配置对象
        cur_model: 当前模型
        best_model: 最佳模型
        game_id: 游戏ID（用于日志）
        cur_plays_white: 当前模型是否执白
        verbose: 是否打印详细信息
        
    返回:
        (当前模型得分, 最佳模型得分, 步数)
    """
    # 创建环境
    env = ChessEnv().reset()
    
    # 为每个模型创建MCTS
    cur_mcts = MCTS(config, cur_model)
    best_mcts = MCTS(config, best_model)
    
    if verbose:
        print(f"评估游戏 {game_id} 开始: 当前模型 {'执白' if cur_plays_white else '执黑'}")
    
    # 记录步数
    step = 0
    
    # 开始时间
    start_time = time.time()
    
    try:
        # 进行游戏直到结束
        while not env.done:                
            # 确定当前执棋的模型
            if (env.white_to_move and cur_plays_white) or (not env.white_to_move and not cur_plays_white):
                # 当前模型的回合
                mcts = cur_mcts
            else:
                # 最佳模型的回合
                mcts = best_mcts
                
            # 使用MCTS获取动作
            _, action = mcts.get_action_prob(env, temperature=0.0)  # 使用温度为0，确定性选择
            
            if action is None:
                # 没有合法动作
                break
                
            # 执行动作
            if verbose and step % 10 == 0:
                print(f"评估游戏 {game_id}, 步骤 {step}, 动作: {action.uci()}")
                
            env.step(action.uci())
            step += 1
            
        # 计算得分
        cur_score, best_score = 0, 0
        
        if env.winner == Winner.WHITE:
            if cur_plays_white:
                cur_score = 1
            else:
                best_score = 1
        elif env.winner == Winner.BLACK:
            if cur_plays_white:
                best_score = 1
            else:
                cur_score = 1
        else:  # 和棋
            cur_score = 0.5
            best_score = 0.5
            
        if verbose:
            print(f"评估游戏 {game_id} 完成，步数: {step}, 结果: {env.result}, 耗时: {time.time() - start_time:.1f}秒")
            print(f"得分 - 当前模型: {cur_score}, 最佳模型: {best_score}")
            
        return cur_score, best_score, step
        
    except Exception as e:
        print(f"评估游戏 {game_id} 发生错误: {e}")
        return 0, 0, step


def evaluate_worker(config_path: str, cur_model_path: str, best_model_path: str, game_id: int,
                   cur_plays_white: bool, result_queue: 'multiprocessing.Queue', verbose: bool = False) -> None:
    """
    评估工作进程
    
    参数:
        config_path: 配置文件路径
        cur_model_path: 当前模型路径
        best_model_path: 最佳模型路径
        game_id: 游戏ID
        cur_plays_white: 当前模型是否执白
        result_queue: 结果队列
        verbose: 是否打印详细信息
    """
    try:
        # 加载配置和模型
        config = Config(config_path)
        cur_model = AlphaZeroModel(config)
        best_model = AlphaZeroModel(config)
        
        if verbose:
            print(f"评估工作进程 {game_id} 开始，加载模型")
            
        # 加载模型
        cur_model.load_model(cur_model_path)
        best_model.load_model(best_model_path)
        
        # 进行一场游戏
        cur_score, best_score, steps = play_game(config, cur_model, best_model, 
                                               game_id, cur_plays_white, verbose)
        
        # 将结果放入队列
        result_queue.put((cur_score, best_score, steps))
        
    except Exception as e:
        print(f"评估工作进程 {game_id} 发生错误: {e}")
        result_queue.put((0, 0, 0))


class Evaluator:
    """模型评估器"""
    
    def __init__(self, config: Config):
        """
        初始化评估器
        
        参数:
            config: 配置对象
        """
        self.config = config
        
        # 设置日志
        log_file = os.path.join(config.LOG_DIR, f"evaluate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.logger = setup_logger("evaluator", log_file)
        
    def evaluate(self, cur_model: AlphaZeroModel, best_model: AlphaZeroModel = None) -> float:
        """
        评估当前模型相对于最佳模型的性能
        
        参数:
            cur_model: 当前模型
            best_model: 最佳模型（如果为None，则仅进行自评估）
            
        返回:
            当前模型的胜率
        """
        # 如果没有提供最佳模型，创建一个新的
        if best_model is None:
            best_model = AlphaZeroModel(self.config)
            if os.path.exists(self.config.best_model_path):
                best_model.load_model(self.config.best_model_path)
            else:
                self.logger.warning("没有找到最佳模型，使用当前模型自我评估")
                best_model = cur_model
        
        # 保存模型到临时文件
        temp_cur_model_path = os.path.join(self.config.CHECKPOINT_DIR, "temp_cur.pt")
        temp_best_model_path = os.path.join(self.config.CHECKPOINT_DIR, "temp_best.pt")
        
        cur_model.save_model(temp_cur_model_path)
        best_model.save_model(temp_best_model_path)
        
        # 设置要进行的游戏数量
        num_games = self.config.EvalConfig.num_games
        
        # 使用多进程
        num_processes = min(self.config.config['system']['num_processes'], num_games)
        self.logger.info(f"开始评估，游戏数量: {num_games}, 进程数: {num_processes}")
        
        # 使用Manager管理进程间共享的队列
        manager = Manager()
        result_queue = manager.Queue()
        
        # 创建进程池
        pool = Pool(processes=num_processes)
        
        # 提交任务，确保当前模型在一半的游戏中执白
        for i in range(num_games):
            cur_plays_white = i < num_games // 2
            pool.apply_async(evaluate_worker, 
                          args=(self.config.BASE_DIR + "/config.yaml", 
                               temp_cur_model_path, 
                               temp_best_model_path,
                               i, 
                               cur_plays_white,
                               result_queue,
                               i == 0))  # 只在第一个游戏中打印详细信息
        
        # 关闭进程池
        pool.close()
        
        # 收集结果
        cur_score_total = 0
        best_score_total = 0
        total_steps = 0
        
        finished_games = 0
        while finished_games < num_games:
            try:
                # 非阻塞方式获取结果，间隔检查
                if not result_queue.empty():
                    cur_score, best_score, steps = result_queue.get(block=False)
                    
                    cur_score_total += cur_score
                    best_score_total += best_score
                    total_steps += steps
                    
                    finished_games += 1
                    self.logger.info(f"评估游戏 {finished_games}/{num_games} 完成, "
                                  f"当前分数: {cur_score_total:.1f}-{best_score_total:.1f}")
                else:
                    # 稍微休眠以减少CPU使用
                    time.sleep(0.1)
                    
            except Exception as e:
                self.logger.error(f"获取评估结果时发生错误: {e}")
        
        # 等待所有进程结束
        pool.join()
        
        # 清理临时文件
        if os.path.exists(temp_cur_model_path):
            os.remove(temp_cur_model_path)
        if os.path.exists(temp_best_model_path):
            os.remove(temp_best_model_path)
        
        # 计算胜率
        win_rate = 0.0
        if finished_games > 0:
            win_rate = cur_score_total / (cur_score_total + best_score_total)
            
        avg_steps = total_steps / finished_games if finished_games > 0 else 0
        
        self.logger.info(f"评估完成, 当前模型胜率: {win_rate:.4f}, 平均步数: {avg_steps:.1f}")
        
        return win_rate
