import os
import yaml
import numpy as np
import chess
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class ModelConfig:
    """神经网络模型配置"""
    input_shape: Tuple[int, int, int] = (119, 8, 8)  # 114个平面
    filters: int = 256  # 卷积滤波器数量
    residual_blocks: int = 20  # 残差块数量
    dropout: float = 0.3  # Dropout比率
    policy_output_dim: int = 4672  # 策略输出维度 (8*8*73) <-- Corrected dimension
    value_fc_size: int = 256  # 价值全连接层大小


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 10  # 每次迭代训练轮数
    batch_size: int = 256  # 批量大小
    buffer_size: int = 100000  # 回放缓冲区大小
    train_steps_per_epoch: int = 1000  # 每个epoch的训练步数
    lr_init: float = 1.0e-3  # 初始学习率
    lr_min: float = 1.0e-6  # 最小学习率
    weight_decay: float = 1.0e-4  # 权重衰减
    grad_clip: float = 1.0  # 梯度裁剪


@dataclass
class MCTSConfig:
    """MCTS配置"""
    num_simulations: int = 800  # 模拟次数
    c_puct: float = 3.0  # PUCT常数
    dirichlet_alpha: float = 0.03  # Dirichlet噪声参数
    dirichlet_eps: float = 0.25  # Dirichlet噪声权重
    virtual_loss: float = 3.0  # 虚拟损失
    max_depth: int = 40  # 最大搜索深度
    time_limit: float = 5.0  # 搜索时间限制(秒)


@dataclass
class SelfPlayConfig:
    """自我对弈配置"""
    num_games: int = 20  # 每次迭代的游戏数
    temp_threshold: int = 15  # 温度阈值
    buffer_size: int = 100000  # 回放缓冲区大小
    timeout: int = 60  # 游戏超时(秒)


@dataclass
class EvalConfig:
    """评估配置"""
    num_games: int = 40  # 评估游戏数
    update_threshold: float = 0.55  # 胜率阈值，超过此值更新模型


@dataclass
class GUIConfig:
    """GUI配置"""
    window_width: int = 800  # 窗口宽度
    window_height: int = 850  # 窗口高度
    cell_size: int = 80  # 单元格大小
    margin: int = 40  # 边距
    bottom_margin: int = 80  # 底部边距
    fps: int = 30  # 帧率


class Config:
    """项目配置类"""

    def __init__(self, config_path: str = None):
        """
        初始化配置

        参数:
            config_path: 配置文件路径
        """
        # Assuming __file__ is defined correctly relative to your project structure
        # If running interactively, you might need to adjust BASE_DIR calculation
        try:
            self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except NameError:  # Handle case where __file__ is not defined (e.g. interactive)
            self.BASE_DIR = os.path.abspath('.')  # Use current working directory or adjust as needed

        # 如果未指定配置文件，使用默认路径
        if not config_path:
            config_path = os.path.join(self.BASE_DIR, "config.yaml")  # Make sure config.yaml exists here

        # 加载YAML配置 (Handle potential FileNotFoundError)
        self.config = self._load_config(config_path)

        # 设置路径
        self.CHECKPOINT_DIR = os.path.join(self.BASE_DIR, "models")
        self.DATA_DIR = os.path.join(self.BASE_DIR, "data")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")

        # 创建必要的目录
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)

        # 初始化配置对象
        self.ModelConfig = ModelConfig(
            filters=self.config["network"]["num_channels"],
            residual_blocks=self.config["network"]["num_res_blocks"],
            dropout=self.config["network"]["dropout"]
            # policy_output_dim is set by default in the dataclass
        )

        self.TrainingConfig = TrainingConfig(
            epochs=self.config["training"]["epochs"],
            batch_size=self.config["training"]["batch_size"],
            buffer_size=self.config["training"]["max_queue_length"],
            # train_steps_per_epoch might not be in default yaml, add default if missing
            train_steps_per_epoch=self.config["training"].get("train_steps_per_epoch", 1000),
            lr_init=self.config["network"]["learning_rate"]["max"],
            lr_min=self.config["network"]["learning_rate"]["min"],
            weight_decay=self.config["network"]["weight_decay"],
            grad_clip=self.config["network"]["grad_clip"]
        )

        self.MCTSConfig = MCTSConfig(
            num_simulations=self.config["mcts"]["num_sims"],
            c_puct=self.config["mcts"]["cpuct"],
            dirichlet_alpha=self.config["mcts"]["dirichlet_alpha"],
            dirichlet_eps=self.config["mcts"]["dirichlet_epsilon"],
            # virtual_loss might not be in default yaml, add default if missing
            virtual_loss=self.config["mcts"].get("virtual_loss", 3.0),
            max_depth=self.config["mcts"]["max_depth"],
            time_limit=self.config["mcts"]["time_limit"]
        )

        self.SelfPlayConfig = SelfPlayConfig(
            num_games=self.config["training"]["num_episodes"],
            temp_threshold=self.config["game"]["temp_threshold"],
            buffer_size=self.config["training"]["max_queue_length"],
            timeout=self.config["game"]["timeout"]
        )

        self.EvalConfig = EvalConfig(
            num_games=self.config["training"]["arena_compare"],
            update_threshold=self.config["training"]["update_threshold"]
        )

        self.GUIConfig = GUIConfig(
            window_width=self.config["gui"]["window_width"],
            window_height=self.config["gui"]["window_height"],
            cell_size=self.config["gui"]["cell_size"],
            margin=self.config["gui"]["margin"],
            bottom_margin=self.config["gui"]["bottom_margin"],
            fps=self.config["gui"]["fps"]
        )

        # 最佳模型路径
        self.best_model_path = os.path.join(self.CHECKPOINT_DIR, "best.pt")

        # 创建国际象棋动作编码映射
        self._create_move_encoding()

    def _load_config(self, config_path: str) -> Dict:
        """
        加载YAML配置文件

        参数:
            config_path: 配置文件路径

        返回:
            配置字典
        """
        # 加载YAML文件, handle potential encoding issues and FileNotFoundError
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except UnicodeDecodeError:
            with open(config_path, 'r', encoding='gbk') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

    def _create_move_encoding(self):
        """
        创建国际象棋动作的编码映射 (8x8x73 = 4672 labels).
        Follows the AlphaZero paper's move encoding scheme.
        Plane indices:
        0-55: Queen moves (8 directions, 7 distances)
        56-63: Knight moves (8 L-shapes)
        64-72: Pawn underpromotions (3 directions: N, NW, NE x 3 pieces: N, B, R)
        """
        self.move_labels = [""] * 4672  # Initialize with empty strings as placeholders
        self.move_policy_indices = {}  # Maps uci move string to its policy index

        # Helper functions
        square_to_coords = lambda sq: (chess.square_file(sq), chess.square_rank(sq))
        coords_to_square = lambda f, r: chess.square(f, r)
        is_on_board = lambda f, r: 0 <= f <= 7 and 0 <= r <= 7

        # --- Define Move Directions/Offsets ---
        # Queen Moves (N, NE, E, SE, S, SW, W, NW)
        queen_directions = [
            (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)
        ]
        num_queen_moves = 56

        # Knight Moves
        knight_offsets = [
            (-1, 2), (1, 2), (-2, 1), (2, 1), (-2, -1), (2, -1), (-1, -2), (1, -2)
        ]
        num_knight_moves = 8
        knight_plane_start = num_queen_moves  # 56

        # Pawn Underpromotions (Knight, Bishop, Rook)
        # Directions relative to pawn: Forward, Diag Left (NW), Diag Right (NE)
        # Note: We must handle white (rank 6->7) and black (rank 1->0) promotion
        promotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        # (df, dr) for white perspective: forward, capture left, capture right
        pawn_promo_deltas_white = [(0, 1), (-1, 1), (1, 1)]
        num_underpromo_moves = 9
        underpromo_plane_start = num_queen_moves + num_knight_moves  # 64

        total_planes = num_queen_moves + num_knight_moves + num_underpromo_moves  # 56 + 8 + 9 = 73

        # --- Generate Labels ---
        for from_sq in range(64):
            from_f, from_r = square_to_coords(from_sq)

            # 1. Queen Moves (Planes 0-55)
            plane_idx = 0
            for dir_idx, (df, dr) in enumerate(queen_directions):
                for dist in range(1, 8):  # distance 1 to 7
                    to_f, to_r = from_f + df * dist, from_r + dr * dist

                    queen_plane_index = dir_idx * 7 + (dist - 1) # Calculate the correct plane (0-55)
                    label_idx = from_sq * total_planes + queen_plane_index

                    if is_on_board(to_f, to_r):
                        to_sq = coords_to_square(to_f, to_r)
                        # Queen promotion is handled implicitly by queen moves ending on the 8th/1st rank.
                        # The `chess` library handles this if promotion=None and it's a pawn move to back rank.
                        # However, for clarity in mapping, we explicitly add QUEEN promotion if applicable.
                        is_pawn_promo_rank_white = (from_r == 6 and to_r == 7)
                        is_pawn_promo_rank_black = (from_r == 1 and to_r == 0)

                        promo = None
                        # Heuristic: Only consider Queen promo for straight/diagonal moves if starting on pawn ranks
                        if (df == 0 or abs(df) == 1) and (is_pawn_promo_rank_white or is_pawn_promo_rank_black):
                            promo = chess.QUEEN  # Default promotion for these planes is Queen

                        move = chess.Move(from_sq, to_sq, promotion=promo)
                        uci = move.uci()
                        self.move_labels[label_idx] = uci
                        # Store first occurrence of UCI string for lookup
                        if uci not in self.move_policy_indices:
                            self.move_policy_indices[uci] = label_idx
                    else:
                        self.move_labels[label_idx] = "0000"  # Placeholder for off-board target

                    plane_idx += 1

            # 2. Knight Moves (Planes 56-63)
            plane_idx = knight_plane_start
            for i, (df, dr) in enumerate(knight_offsets):
                to_f, to_r = from_f + df, from_r + dr
                label_idx = from_sq * total_planes + plane_idx + i

                if is_on_board(to_f, to_r):
                    to_sq = coords_to_square(to_f, to_r)
                    move = chess.Move(from_sq, to_sq)
                    uci = move.uci()
                    self.move_labels[label_idx] = uci
                    if uci not in self.move_policy_indices:
                        self.move_policy_indices[uci] = label_idx
                else:
                    self.move_labels[label_idx] = "0000"

            # 3. Pawn Underpromotions (Planes 64-72)
            plane_idx = underpromo_plane_start
            # Check both white and black promotion possibilities FROM this square

            # White promotion check (rank 6 -> 7)
            if from_r == 6:
                promo_rank = 7
                for move_type, (df, dr) in enumerate(pawn_promo_deltas_white):
                    to_f, to_r = from_f + df, from_r + dr
                    if is_on_board(to_f, to_r) and to_r == promo_rank:
                        to_sq = coords_to_square(to_f, to_r)
                        for promo_idx, piece in enumerate(promotion_pieces):
                            # Map move type (0: Fwd, 1: CapLeft, 2: CapRight) and piece (0: N, 1: B, 2: R) to plane
                            current_plane = plane_idx + move_type * 3 + promo_idx
                            label_idx = from_sq * total_planes + current_plane

                            move = chess.Move(from_sq, to_sq, promotion=piece)
                            uci = move.uci()
                            self.move_labels[label_idx] = uci
                            # Underpromotion UCI is unique (e.g., e7e8n), store it
                            self.move_policy_indices[uci] = label_idx
                    else:
                        # Fill placeholder even if move is invalid/off-board for this type
                        for promo_idx in range(len(promotion_pieces)):
                            current_plane = plane_idx + move_type * 3 + promo_idx
                            label_idx = from_sq * total_planes + current_plane
                            if not self.move_labels[
                                label_idx]:  # Only fill if not already potentially filled by other logic
                                self.move_labels[label_idx] = "0000"

            # Black promotion check (rank 1 -> 0)
            elif from_r == 1:
                promo_rank = 0
                # Deltas need to be flipped for black: (0, -1), (1, -1), (-1, -1) -> Fwd, CapRight(SE), CapLeft(SW)
                pawn_promo_deltas_black = [(0, -1), (1, -1), (-1, -1)]
                # Map black deltas to AZ plane order: Fwd(N), CapLeft(NW), CapRight(NE)
                # Black Fwd (0,-1) -> Plane Type 0 (N)
                # Black CapLeft (-1,-1) -> Plane Type 1 (NW)
                # Black CapRight (1,-1) -> Plane Type 2 (NE)
                az_plane_order_map = {0: 0, 2: 1,
                                      1: 2}  # Maps index in pawn_promo_deltas_black to AZ plane type index (0,1,2)

                for move_type_idx, (df, dr) in enumerate(pawn_promo_deltas_black):
                    to_f, to_r = from_f + df, from_r + dr
                    az_plane_type = az_plane_order_map[
                        move_type_idx]  # Get the corresponding AZ plane index (0, 1, or 2)

                    if is_on_board(to_f, to_r) and to_r == promo_rank:
                        to_sq = coords_to_square(to_f, to_r)
                        for promo_idx, piece in enumerate(promotion_pieces):
                            current_plane = plane_idx + az_plane_type * 3 + promo_idx
                            label_idx = from_sq * total_planes + current_plane

                            move = chess.Move(from_sq, to_sq, promotion=piece)
                            uci = move.uci()
                            self.move_labels[label_idx] = uci
                            self.move_policy_indices[uci] = label_idx
                    else:
                        for promo_idx in range(len(promotion_pieces)):
                            current_plane = plane_idx + az_plane_type * 3 + promo_idx
                            label_idx = from_sq * total_planes + current_plane
                            if not self.move_labels[label_idx]:
                                self.move_labels[label_idx] = "0000"

            # Fill any remaining underpromotion plane placeholders for non-promotion start ranks
            if from_r != 1 and from_r != 6:
                for i in range(num_underpromo_moves):
                    label_idx = from_sq * total_planes + underpromo_plane_start + i
                    if not self.move_labels[label_idx]:
                        self.move_labels[label_idx] = "0000"

        # Final verification and creation of the reverse lookup (index to UCI)
        if len(self.move_labels) != 4672:
            print(f"Warning: Expected 4672 labels, generated {len(self.move_labels)}")
            # Pad if necessary, though the logic should prevent this
            self.move_labels.extend(["0000"] * (4672 - len(self.move_labels)))

        # Create the index-to-move lookup (useful for decoding network output)
        self.index_to_move = {i: uci for i, uci in enumerate(self.move_labels)}

        # Keep the original move_lookup name consistent with the old code if needed,
        # but `move_policy_indices` is more descriptive for the UCI -> Index mapping.
        self.move_lookup = self.move_policy_indices

        # print(f"Generated {len(self.move_labels)} move labels.")
        # print(f"Created UCI->Index lookup with {len(self.move_lookup)} entries.")
        # print("Example labels:", self.move_labels[0:5], self.move_labels[73:78], self.move_labels[4665:])
        # Example lookup: print("Index for e2e4:", self.move_lookup.get('e2e4', 'Not found'))
        # Example lookup: print("Index for e7e8q:", self.move_lookup.get('e7e8q', 'Not found')) # Should map to a queen move plane index
        # Example lookup: print("Index for e7e8n:", self.move_lookup.get('e7e8n', 'Not found')) # Should map to an underpromotion plane index

# Example Usage (assuming you have a config.yaml or run interactively)
# config = Config()
# print(f"Policy output dimension in ModelConfig: {config.ModelConfig.policy_output_dim}")
# print(f"Total generated move labels: {len(config.move_labels)}")
# print(f"Lookup map size (UCI -> Index): {len(config.move_lookup)}")
# print(f"Reverse map size (Index -> UCI): {len(config.index_to_move)}")
# print(f"Label for index 0: {config.index_to_move[0]}") # e.g., a1a2 if queen move N, dist 1
# print(f"Label for index 73: {config.index_to_move[73]}") # e.g., a2a3 if queen move N, dist 1
# print(f"Index for 'e2e4': {config.move_lookup.get('e2e4')}") # Should exist
# print(f"Index for 'a7a8n': {config.move_lookup.get('a7a8n')}") # Should exist (underpromotion)
# print(f"Index for 'a7a8q': {config.move_lookup.get('a7a8q')}") # Should exist (queen move plane)
# print(f"Label for index 4671: {config.index_to_move[4671]}") # Last possible move label
