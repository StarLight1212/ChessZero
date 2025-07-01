import chess
import chess.pgn
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
import enum
import copy

# Define chess game result enum
class Winner(enum.Enum):
    BLACK = -1  # Black wins
    DRAW = 0    # Draw
    WHITE = 1   # White wins
    NONE = None     # Game not finished


class ChessEnv:
    """
    Chess environment class, manages board state and generates 
    neural network input features according to AlphaZero specification.
    """

    def __init__(self, history_length: int = 8):
        """
        Initializes the chess environment.
        Args:
            history_length (int): Number of past board states (T) to include in features.
        """
        self.board = chess.Board()
        self.num_halfmoves = 0 # Tracks halfmoves since environment start (may differ from board.halfmove_clock)
        self.winner = Winner.NONE
        self.resigned = False
        self.result = None
        # History stores (board_copy, turn) tuples for T steps
        self.history_length = history_length 
        self.state_history: List[Tuple[chess.Board, chess.Color]] = [] 
        self._add_to_history() # Add initial state

    def reset(self):
        """Resets the board to the initial state."""
        self.board = chess.Board()
        self.num_halfmoves = 0
        self.winner = Winner.NONE
        self.resigned = False
        self.result = None
        self.state_history = []
        self._add_to_history() # Add initial state after reset
        return self

    def update(self, board: chess.Board):
        """Updates the environment using a given board state."""
        # Warning: Updating directly might desync history unless carefully managed.
        # It's usually better to replay moves or reset and play to the position.
        self.board = chess.Board(board.fen())
        self.winner = Winner.NONE
        self.resigned = False
        self.result = None # Result might be unknown
        # History should ideally be reconstructed if using update.
        # For simplicity here, we clear and add current state.
        self.state_history = [] 
        self._add_to_history()
        return self

    def step(self, action: Optional[str]) -> bool:
        """
        Executes one action (move).

        Args:
            action: UCI format move string (e.g., "e2e4"), or None to resign.

        Returns:
            True if the action was successfully executed, False otherwise.
        """
        if self.is_game_over():
            # print("Warning: Trying to step in a finished game.")
            return False

        if action is None:
            # Resign
            self.resigned = True
            self._update_winner()
            return True

        try:
            move = chess.Move.from_uci(action)

            # If it's a legal move, push it
            if move in self.board.legal_moves:
                 # Store state *before* the move for history
                self._add_to_history() 
                self.board.push(move)
                self.num_halfmoves += 1
                self._update_winner()
                return True
            else:
                # print(f"Illegal move attempted: {action}")
                return False
        except Exception as e:
            print(f"Error during step execution: {e}")
            return False

    def _add_to_history(self):
        """Adds the current board state and turn to the history."""
        # Store a copy of the board and whose turn it is
        self.state_history.append((self.board.copy(), self.board.turn))
        # Keep only the last T states
        if len(self.state_history) > self.history_length:
            self.state_history.pop(0)

    def _update_winner(self):
        if self.resigned:
            if self.board.turn == chess.WHITE:
                self.winner = Winner.BLACK
                self.result = "0-1"
            else:
                self.winner = Winner.WHITE
                self.result = "1-0"
            return

        # 手动检查游戏是否结束
        if not list(self.board.legal_moves):  # 没有合法移动
            if self.board.is_checkmate():
                self.winner = Winner.BLACK if self.board.turn == chess.WHITE else Winner.WHITE
                self.result = "0-1" if self.winner == Winner.BLACK else "1-0"
            else:  # 和棋（僵局）
                self.winner = Winner.DRAW
                self.result = "1/2-1/2"
        elif self.board.halfmove_clock >= 100 or self.board.is_repetition(3):
            self.winner = Winner.DRAW
            self.result = "1/2-1/2"
        else:
            self.winner = Winner.NONE
            self.result = "*"

    def is_game_over(self) -> bool:
        """Checks if the game has ended."""
        # Use internal winner state which includes resignation and adjudication
        return self.winner != Winner.NONE

    @property
    def done(self) -> bool:
        """Property indicating if the game is over."""
        return self.is_game_over()

    @property
    def white_to_move(self) -> bool:
        """Property indicating if it's White's turn."""
        return self.board.turn == chess.WHITE

    def get_legal_moves(self) -> List[chess.Move]:
        """Gets all legal moves."""
        return list(self.board.legal_moves)

    def get_legal_moves_uci(self) -> List[str]:
        """Gets all legal moves in UCI format."""
        return [move.uci() for move in self.board.legal_moves]

    def adjudicate(self):
        """Adjudicates the game based on material value if not already finished."""
        if self.winner != Winner.NONE:
            return # Already decided

        # Calculate material value for both sides
        white_value = self._get_material_value(chess.WHITE)
        black_value = self._get_material_value(chess.BLACK)

        diff = white_value - black_value

        # Adjudicate based on material difference (simplified rule)
        if diff > 3:  # White leads by more than 3 pawn equivalents
            self.winner = Winner.WHITE
            self.result = "1-0"
        elif diff < -3: # Black leads by more than 3 pawn equivalents
            self.winner = Winner.BLACK
            self.result = "0-1"
        else: # Material is close, adjudicate as draw
            self.winner = Winner.DRAW
            self.result = "1/2-1/2"

    def _get_material_value(self, color: chess.Color) -> float:
        """Calculates the material value for a given color."""
        value = 0
        # Piece values: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9
        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            if piece.color == color:
                if piece.piece_type == chess.PAWN:
                    value += 1
                elif piece.piece_type == chess.KNIGHT:
                    value += 3
                elif piece.piece_type == chess.BISHOP:
                    value += 3
                elif piece.piece_type == chess.ROOK:
                    value += 5
                elif piece.piece_type == chess.QUEEN:
                    value += 9
        return value

    def get_observation(self) -> np.ndarray:
        """
        Generates the 119x8x8 input tensor for the neural network,
        following the AlphaZero paper structure: (M*T + L) channels.
        M=14 (P1 pieces(6), P2 pieces(6), Repetitions(2))
        T=8 (History length)
        L=7 (Color, MoveCount, P1 Castling(2), P2 Castling(2), NoProgress)

        Returns:
            np.ndarray: The 119x8x8 feature tensor.
        """
        # M = 14 planes per time step
        # L = 7 constant planes
        # Total channels = M * T + L = 14 * 8 + 7 = 112 + 7 = 119
        observation = np.zeros((119, 8, 8), dtype=np.float32)

        current_player_color = self.board.turn
        opponent_color = not current_player_color

        # --- M * T Features (Repeating History Block) ---
        history_idx = len(self.state_history) - 1 # Index of the current state
        
        # Iterate T=8 steps back in time (or fewer if history is short)
        for i in range(self.history_length):
            hist_offset = self.history_length - 1 - i # 0 for current, 1 for t-1, ..., 7 for t-7
            
            state_idx = history_idx - hist_offset
            
            board_planes = np.zeros((14, 8, 8), dtype=np.float32) # 12 piece planes + 2 repetition

            if state_idx >= 0:
                # Get the board state from history at this time step
                historic_board, turn_at_history = self.state_history[state_idx]
                
                # Perspective is always the *current* player
                piece_planes_12 = self._get_oriented_piece_planes(historic_board, current_player_color)
                board_planes[0:12] = piece_planes_12
                
                # Repetition planes (relative to the *current* board state)
                # Paper: "the repetition count for that position" - implies current pos.
                # Plane M-2: Position has repeated >= 1 time before current
                if self.board.is_repetition(2): 
                    board_planes[12].fill(1.0)
                # Plane M-1: Position has repeated >= 2 times before current
                if self.board.is_repetition(3):
                    board_planes[13].fill(1.0)

            # Assign the 14 planes for this time step
            start_channel = i * 14
            observation[start_channel : start_channel + 14] = board_planes

        # --- L Constant Features ---
        l_channel_start = self.history_length * 14 # Start index for L planes (112)

        # L1: Colour (1.0 for White, 0.0 for Black)
        observation[l_channel_start].fill(1.0 if current_player_color == chess.WHITE else 0.0)

        # L2: Total move count (Normalized - choose a reasonable max, e.g., 200)
        # AlphaZero might use a different scaling or just the raw number. 
        # Normalizing helps NN stability. Let's normalize by 100 for consistency.
        observation[l_channel_start + 1].fill(self.board.fullmove_number / 100.0) 

        # L3-L4: P1 (Current Player) Castling Rights
        if self.board.has_kingside_castling_rights(current_player_color):
            observation[l_channel_start + 2].fill(1.0)
        if self.board.has_queenside_castling_rights(current_player_color):
            observation[l_channel_start + 3].fill(1.0)
            
        # L5-L6: P2 (Opponent) Castling Rights
        if self.board.has_kingside_castling_rights(opponent_color):
            observation[l_channel_start + 4].fill(1.0)
        if self.board.has_queenside_castling_rights(opponent_color):
            observation[l_channel_start + 5].fill(1.0)
            
        # L7: No-progress count (50-move rule counter, normalized)
        observation[l_channel_start + 6].fill(self.board.halfmove_clock / 100.0) # Normalize by 100 (max is 100 halfmoves)

        # Final check of total channels: 112 + 7 = 119
        assert l_channel_start + 7 == 119

        return observation

    def _get_oriented_piece_planes(self, board: chess.Board, pov_color: chess.Color) -> np.ndarray:
        """
        Gets 12 piece planes oriented to the perspective of pov_color.
        Planes 0-5: pov_color's pieces (P, N, B, R, Q, K)
        Planes 6-11: opponent's pieces (P, N, B, R, Q, K)
        If pov_color is Black, the board coordinates are flipped vertically.

        Args:
            board: The chess.Board state.
            pov_color: The color of the player whose perspective is used (chess.WHITE or chess.BLACK).

        Returns:
            np.ndarray: A (12, 8, 8) numpy array of piece planes.
        """
        oriented_planes = np.zeros((12, 8, 8), dtype=np.float32)
        opponent_color = not pov_color

        # Piece type to index mapping (within a color group)
        piece_to_idx = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                rank = chess.square_rank(square)  # 0-7
                file = chess.square_file(square)  # 0-7
                
                # Apply vertical flip if perspective is Black
                if pov_color == chess.BLACK:
                    rank = 7 - rank 
                    # File remains the same as vertical flip doesn't change file index

                plane_idx_base = 0 if piece.color == pov_color else 6
                piece_idx = piece_to_idx[piece.piece_type]
                plane_idx = plane_idx_base + piece_idx

                oriented_planes[plane_idx, rank, file] = 1.0

        return oriented_planes


    def copy(self):
        """Creates a deep copy of the environment."""
        new_env = ChessEnv(history_length=self.history_length)
        new_env.board = self.board.copy()
        new_env.num_halfmoves = self.num_halfmoves
        new_env.winner = self.winner
        new_env.resigned = self.resigned
        new_env.result = self.result
        # Deep copy history (contains tuples of board copies and colors)
        new_env.state_history = [(b.copy(), turn) for b, turn in self.state_history] 
        return new_env

    def render(self, mode='unicode') -> str:
        """Renders the board as a string (unicode or standard ASCII)."""
        if mode == 'unicode':
            return self.board.unicode(invert_color=self.white_to_move) # Show from current player's view
        else:
            return str(self.board)

# # --- Example Usage & Basic Checks ---
# if __name__ == "__main__":
#     env = ChessEnv()
#     print("Initial Board:")
#     print(env.render())
    
#     # Check initial observation shape
#     obs = env.get_observation()
#     print(f"\nInitial Observation Shape: {obs.shape}")
#     assert obs.shape == (119, 8, 8)
#     print("Shape check passed.")

#     # Make a few moves
#     print("\nMaking moves: e2e4, e7e5, g1f3")
#     env.step("e2e4")
#     env.step("e7e5")
#     env.step("g1f3")
    
#     print("\nBoard after moves:")
#     print(env.render())
    
#     # Check observation after moves
#     obs_after_moves = env.get_observation()
#     print(f"\nObservation Shape after moves: {obs_after_moves.shape}")
#     assert obs_after_moves.shape == (119, 8, 8)
    
#     # Verify some features manually (basic check)
#     # Current player is Black (after Nf3)
#     assert env.white_to_move == False 
    
#     # Check Color plane (L1 = index 112) - Should be 0.0 for Black
#     print(f"Color plane value (should be 0.0): {obs_after_moves[112, 0, 0]}")
#     assert obs_after_moves[112, 0, 0] == 0.0 
    
#     # Check P1 pieces (Black) - e.g., Black pawn at e5
#     # Perspective is Black, so e5 is rank 3, file 4 after flipping (7-4=3)
#     # P1 Pawn plane index = 6 (0-5 White, 6-11 Black)
#     # History step T=8 (index 7), M=14 planes. Channel = 7*14 + 6 = 98 + 6 = 104
#     print(f"P1 (Black) Pawn at e5 (flipped rank 3, file 4) in current step (plane 104): {obs_after_moves[104, 3, 4]}")
#     assert obs_after_moves[104, 3, 4] == 1.0

#     # Check P2 pieces (White) - e.g., White Knight at f3
#     # Perspective is Black, f3 is rank 5, file 5 after flipping (7-2=5)
#     # P2 Knight plane index = 1 (0-5 White pieces for P2)
#     # History step T=8 (index 7), M=14 planes. Channel = 7*14 + 1 = 98 + 1 = 99
#     print(f"P2 (White) Knight at f3 (flipped rank 5, file 5) in current step (plane 99): {obs_after_moves[99, 5, 5]}")
#     assert obs_after_moves[99, 5, 5] == 1.0

#     # Check history (e.g., initial state)
#     # Step T=1 (index 0), M=14 planes. 
#     # Initial state had White perspective implicitly when adding to history.
#     # Let's check White Pawn at e2 in step T=1 (channel 0*14 + 0 = 0)
#     # Original e2 is rank 1, file 4. Perspective is now Black, so flipped rank is 7-1=6.
#     print(f"P2 (White) Pawn at e2 (flipped rank 6, file 4) in history step T=1 (plane 0): {obs_after_moves[0, 6, 4]}")
#     # This piece is not there anymore in the current board, let's check the state from history directly
#     initial_board_hist, initial_turn = env.state_history[0]
#     obs_initial_reconstructed = env._get_oriented_piece_planes(initial_board_hist, env.board.turn) # Check initial board from current perspective
#     print(f"Recon P2(W) Pawn at e2 (flipped rank 6, file 4) from initial state: {obs_initial_reconstructed[0, 6, 4]}")
#     assert obs_after_moves[0, 6, 4] == 1.0 # Should be 1 in the first history frame (T=1)


#     # Check repetition planes (should be 0 initially)
#     # Current state: channels 7*14+12 = 110 and 7*14+13 = 111
#     print(f"Repetition plane 1 (current): {obs_after_moves[110, 0, 0]}")
#     print(f"Repetition plane 2 (current): {obs_after_moves[111, 0, 0]}")
#     assert obs_after_moves[110, 0, 0] == 0.0
#     assert obs_after_moves[111, 0, 0] == 0.0

#     # Check Castling rights (L3-L6, channels 114-117)
#     # White lost Nf3, Black lost e5. Both sides still have rights.
#     # P1 (Black): Kingside=1, Queenside=1
#     # P2 (White): Kingside=1, Queenside=1
#     print(f"P1 KS Castling (plane 114): {obs_after_moves[114, 0, 0]}") # P1 is Black
#     print(f"P1 QS Castling (plane 115): {obs_after_moves[115, 0, 0]}")
#     print(f"P2 KS Castling (plane 116): {obs_after_moves[116, 0, 0]}") # P2 is White
#     print(f"P2 QS Castling (plane 117): {obs_after_moves[117, 0, 0]}")
#     assert obs_after_moves[114, 0, 0] == 1.0 # Black KS
#     assert obs_after_moves[115, 0, 0] == 1.0 # Black QS
#     assert obs_after_moves[116, 0, 0] == 1.0 # White KS
#     assert obs_after_moves[117, 0, 0] == 1.0 # White QS

#     print("\nBasic feature checks passed.")

#     # Test game end
#     env.reset()
#     # Fool's Mate
#     env.step("f2f3")
#     env.step("e7e5")
#     env.step("g2g4")
#     env.step("d8h4") # Checkmate
#     print("\nBoard after Fool's Mate:")
#     print(env.render())
#     print(f"Game over: {env.is_game_over()}")
#     print(f"Winner: {env.winner}")
#     print(f"Result: {env.result}")
#     assert env.is_game_over()
#     assert env.winner == Winner.BLACK