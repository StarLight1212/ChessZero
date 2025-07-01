import numpy as np
import math
import time
import threading
import chess
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from ..config import Config
from ..env.chess_env import ChessEnv, Winner
from ..utils import normalize_encoding


class Node:
    """MCTS树节点"""
    
    def __init__(self, prior: float):
        self.visit_count = 0
        self.prior = prior  # 先验概率 P(s,a)
        self.value_sum = 0  # 累积价值
        self.children: Dict[chess.Move, 'Node'] = {}
        self.state = None  # 存储节点状态，用于调试

    def expanded(self) -> bool:
        """节点是否已经展开"""
        return len(self.children) > 0

    def value(self) -> float:
        """获取节点的平均价值Q(s,a)"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MCTS:
    """蒙特卡洛树搜索实现"""
    
    def __init__(self, config: Config, model):
        """
        初始化MCTS
        
        参数:
            config: 配置对象
            model: 用于预测策略和价值的模型
        """
        self.config = config
        self.model = model
        self.root = Node(0)
        self.mcts_config = config.MCTSConfig

        # 用于保护树在多线程访问时的锁
        self.node_locks = defaultdict(threading.Lock)

        # 搜索限制
        self.max_depth = self.mcts_config.max_depth
        
        # 缓存变量
        self.legal_moves_cache = {}

    def reset(self):
        """重置MCTS搜索树和缓存"""
        self.root = Node(0)
        self.legal_moves_cache = {}

    def search(self, env: ChessEnv) -> Dict[chess.Move, Tuple[int, float]]:
        """
        执行MCTS搜索并返回根节点上每个动作的访问计数和Q值
        
        参数:
            env: 当前棋盘环境
            
        返回:
            字典，键为动作，值为(访问计数, Q值)的元组
        """
        # 重新设置根节点
        self.root = Node(0)
        
        # 预先获取并缓存合法动作
        board_hash = hash(env.board.fen())
        if board_hash not in self.legal_moves_cache:
            self.legal_moves_cache[board_hash] = (
                list(env.get_legal_moves()),
                [move.uci() for move in env.get_legal_moves()]
            )

        # 执行模拟
        for i in range(self.mcts_config.num_simulations):
            env_copy = env.copy()
            self._simulate(env_copy, self.root)

        # 收集所有动作的访问计数
        action_stats = {}
        for move, child in self.root.children.items():
            action_stats[move] = (child.visit_count, child.value())

        return action_stats

    def _simulate(self, env: ChessEnv, node: Node, depth: int = 0) -> float:
        """
        执行单次MCTS模拟
        
        参数:
            env: 当前棋盘环境的拷贝
            node: 当前搜索树节点
            depth: 当前搜索的深度
            
        返回:
            模拟结果的价值
        """
        # 检查搜索深度限制
        if depth >= self.max_depth:
            # 达到最大深度，使用启发式评估
            return self._evaluate_position(env)

        # 如果游戏结束，返回结果
        if env.done:
            if env.winner == Winner.WHITE:
                return 1.0  # 白方胜利
            elif env.winner == Winner.BLACK:
                return -1.0  # 黑方失败
            else:
                return 0.0  # 和棋

        # 获取并缓存合法动作
        board_hash = hash(env.board.fen())
        if board_hash not in self.legal_moves_cache:
            legal_moves = list(env.get_legal_moves())
            legal_moves_uci = [move.uci() for move in legal_moves]
            self.legal_moves_cache[board_hash] = (legal_moves, legal_moves_uci)
        else:
            legal_moves, legal_moves_uci = self.legal_moves_cache[board_hash]

        # 如果没有合法动作，返回游戏结果
        if not legal_moves:
            if env.board.is_checkmate():
                return -1.0 if env.board.turn == chess.WHITE else 1.0
            else:  # 和棋
                return 0.0

        # 如果是新节点，进行扩展
        if not node.expanded():
            # 获取当前状态的动作概率和价值
            state = env.get_observation()
            policy, value = self.model.predict(state)

            # 使用normalize_encoding函数进行概率归一化
            policy = normalize_encoding(policy, legal_moves_uci, self.config.move_lookup)

            # 为每个合法移动创建子节点
            for move in legal_moves:
                move_uci = move.uci()
                if move_uci in self.config.move_lookup:
                    move_idx = self.config.move_lookup[move_uci]
                    node.children[move] = Node(policy[move_idx])

            # 如果没有子节点（极罕见情况）
            if len(node.children) == 0:
                # 这种情况通常不会发生，但为了健壮性处理
                return 0.0

            # 返回价值（从当前玩家的角度）
            return -value  # 价值是从下一个玩家角度返回的，所以要取负

        # 选择最佳动作并继续模拟
        move = self._select_action(node, env.white_to_move)

        # 检查动作是否有效（通常不需要这个检查，因为我们已经筛选了合法动作）
        if move is None:
            # 这种情况不应该发生，但为了健壮性处理
            print(f"警告: MCTS没有选择任何动作")
            return 0.0

        # 应用虚拟损失
        with self.node_locks[hash(node)]:
            node.children[move].visit_count += 1
            node.children[move].value_sum -= self.mcts_config.virtual_loss

        # 执行动作并继续模拟
        env.step(move.uci())

        # 递归调用得到子节点的价值
        value = self._simulate(env, node.children[move], depth + 1)

        # 更新节点统计信息（减去虚拟损失并加上真实价值）
        with self.node_locks[hash(node)]:
            node.children[move].value_sum += self.mcts_config.virtual_loss + value
            # 访问计数已经在应用虚拟损失时增加过了

        return -value  # 返回负价值，因为价值是相对于当前玩家的

    def _select_action(self, node: Node, to_play_white: bool) -> Optional[chess.Move]:
        """PUCT with proper Dirichlet noise (root only)"""
        EPS = self.mcts_config.dirichlet_eps
        ALPHA = self.mcts_config.dirichlet_alpha

        # ------------ 根节点一次性混入噪声 ------------
        if node is self.root and not hasattr(node, "_dirichlet_done"):
            child_priors = np.array([child.prior for child in node.children.values()], dtype=np.float32)
            noise = np.random.dirichlet([ALPHA] * len(child_priors)).astype(np.float32)
            mixed = (1 - EPS) * child_priors + EPS * noise
            for c, new_p in zip(node.children.values(), mixed):
                c.prior = new_p
            node._dirichlet_done = True  # 标记一次即可

        # ------------ 计算 PUCT ------------
        total_visits = sum(c.visit_count for c in node.children.values())
        sqrt_total = math.sqrt(total_visits + 1)

        best_score, best_move = -float("inf"), None
        for move, child in node.children.items():
            u_val = self.mcts_config.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = child.value() + u_val
            if score > best_score:
                best_score, best_move = score, move
        return best_move

    def _evaluate_position(self, env: ChessEnv) -> float:
        """
        当搜索达到最大深度时评估位置
        
        参数:
            env: 棋盘环境
            
        返回:
            位置评估值
        """
        # 使用神经网络评估位置
        state = env.get_observation()
        _, value = self.model.predict(state)

        # 对于黑方回合，翻转价值
        return value

    def get_action_prob(self, env: ChessEnv, temperature: float = 1.0, add_dirichlet_noise: bool = False) -> Tuple[Dict[chess.Move, float], Optional[chess.Move]]:
        """
        获取每个动作的概率分布
        
        参数:
            env: 当前棋盘环境
            temperature: 温度参数，控制探索程度
            add_dirichlet_noise: 是否添加狄利克雷噪声
            
        返回:
            (action_probs, chosen_action) - 动作概率字典和选择的动作
        """
        # 预先获取并缓存合法动作
        board_hash = hash(env.board.fen())
        if board_hash not in self.legal_moves_cache:
            legal_moves = list(env.get_legal_moves())
            legal_moves_uci = [move.uci() for move in legal_moves]
            self.legal_moves_cache[board_hash] = (legal_moves, legal_moves_uci)
        else:
            legal_moves, legal_moves_uci = self.legal_moves_cache[board_hash]
            
        if not legal_moves:
            # 没有合法移动，返回空字典和None
            return {}, None
        
        # 进行MCTS搜索
        counts = self.search(env)

        # 提取访问计数
        moves = list(counts.keys())
        visit_counts = np.array([count for move, (count, _) in counts.items()])

        if len(moves) == 0:
            # 没有合法移动，返回None
            return {}, None

        if temperature == 0:
            # 贪婪选择
            action_index = np.argmax(visit_counts)
            action_probs = {moves[action_index]: 1.0}
            return action_probs, moves[action_index]
        else:
            # 温度控制的概率分布
            scaled_counts = visit_counts ** (1.0 / temperature)
            probs = scaled_counts / np.sum(scaled_counts)

            # 构建动作到概率的映射
            action_probs = {moves[i]: probs[i] for i in range(len(moves))}

            # 根据归一化后的概率选择动作
            if sum(action_probs.values()) > 0:
                moves_list = list(action_probs.keys())
                probs_list = [action_probs[move] for move in moves_list]
                probs_sum = sum(probs_list)
                if probs_sum > 0:
                    probs_list = [p / probs_sum for p in probs_list]  # 重新归一化
                    chosen_action = np.random.choice(moves_list, p=probs_list)
                else:
                    # 如果所有概率为0，随机选择
                    chosen_action = np.random.choice(moves_list)
            else:
                # 没有合法动作
                chosen_action = None

            return action_probs, chosen_action