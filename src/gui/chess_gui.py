# --- (Keep all previous imports and class definitions: PromotionDialog, ChessSquare, AIThread) ---
import sys
import os
import time
import chess
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                           QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
                           QDialog, QMessageBox, QComboBox, QAction, QFileDialog)
from PyQt5.QtSvg import QSvgRenderer
from PyQt5.QtGui import QPixmap, QIcon, QColor, QPainter, QPalette, QFont
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
import logging
import traceback # Import traceback for detailed error logging

# --- Assumed imports (keep as before) ---
script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..config import Config
    from ..env import ChessEnv, Winner
    from ..model import AlphaZeroModel
    from ..player import AlphaZeroPlayer, HumanPlayer, RandomPlayer
    from ..utils import setup_logger
except ImportError as e:
    print(f"Error importing project modules: {e}")
    # --- (Keep Dummy classes as before for fallback) ---
    class Config:
        class GUIConfig: cell_size = 60; window_width = 600; window_height = 700
        LOG_DIR = "logs"; best_model_path = None; CHECKPOINT_DIR = "models"
        class Play: simulation_num_per_move=50; thinking_loop=1; c_puct=1.0; parallel_search_num=8; search_threads=8; dirichlet_alpha=0.3; tau_decay_rate=0.99; virtual_loss=1
        class Model: cnn_filter_num=32; cnn_filter_size=3; res_layer_num=3; l2_reg=1e-4; value_fc_size=64; input_depth=18
    class ChessEnv:
        def __init__(self): self.board = chess.Board(); self.done=False; self.white_to_move=True; self.winner=None; self.result="*"; self._update_game_state()
        def reset(self): self.board.reset(); self.done=False; self.white_to_move=True; self.winner=None; self.result="*"; self._update_game_state(); return self
        def step(self, uci):
             print(f"Dummy env.step received: {uci}")
             try:
                 move = self.board.parse_uci(uci)
                 # !!! CRITICAL DEBUG STEP: Check if move is legal *right before* pushing
                 if move in self.board.legal_moves:
                     print(f"Dummy env: Move {uci} IS in legal_moves before push.")
                     self.board.push(move)
                     self.white_to_move = self.board.turn == chess.WHITE
                     self._update_game_state()
                     print(f"Dummy env: Move {uci} successful. New FEN: {self.board.fen()}")
                     return True
                 else:
                     # This is likely where the problem is if the GUI check passed!
                     print(f"!!!!!!!!!!!! ERROR in Dummy env.step !!!!!!!!!!!!")
                     print(f"Move {uci} was NOT in legal_moves right before push.")
                     print(f"Current FEN: {self.board.fen()}")
                     print(f"Turn: {'White' if self.board.turn else 'Black'}")
                     legal_moves_list = [m.uci() for m in self.board.legal_moves]
                     print(f"Legal moves NOW: {legal_moves_list}")
                     print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                     return False
             except ValueError as e: # Illegal move format OR illegal move push
                 print(f"Dummy env.step ValueError for {uci}: {e}")
                 return False
             except Exception as e:
                 print(f"Dummy env.step unexpected error for {uci}: {e}")
                 return False
        def get_legal_moves(self): return list(self.board.legal_moves)
        def copy(self): # Basic copy for dummy
            new_env = ChessEnv(); new_env.board = self.board.copy(); new_env.done = self.done
            new_env.white_to_move = self.white_to_move; new_env.winner = self.winner; new_env.result=self.result
            return new_env
        def _update_game_state(self): # Dummy update state
             self.done = self.board.is_game_over()
             if self.done:
                 self.result = self.board.result()
                 if self.result == "1-0": self.winner = Winner.WHITE
                 elif self.result == "0-1": self.winner = Winner.BLACK
                 else: self.winner = Winner.DRAW # Draw
             else:
                 self.winner = None
                 self.result = "*"
    class Winner: WHITE=1; BLACK=2; DRAW=0
    class AlphaZeroModel:
        def __init__(self, cfg): pass
        def load_model(self, pth): print(f"Dummy load_model: {pth}"); return False
    class AlphaZeroPlayer:
        def __init__(self, cfg, mdl): pass
        def get_action(self, env, temp): print("Dummy AI move"); time.sleep(0.5); return None
    class HumanPlayer:
        def __init__(self, cfg): pass
    class RandomPlayer:
        def __init__(self, cfg): pass
        def get_action(self, env, temp):
            print("Dummy Random move thinking...")
            legal_moves = env.get_legal_moves()
            if not legal_moves: return None
            import random
            chosen_move = random.choice(legal_moves)
            print(f"Dummy Random move chose: {chosen_move.uci()}")
            time.sleep(0.5) # Simulate thinking
            return chosen_move.uci()
    def setup_logger(name, file): logging.basicConfig(level=logging.DEBUG); return logging.getLogger(name) # Set level to DEBUG
# --- End Assumed imports ---

# --- (Keep PIECE_SVG_FILES, IMAGE_DIR, PromotionDialog, ChessSquare, AIThread classes as defined in the previous corrected version) ---
# Make sure ChessSquare uses the improved update_style and set_highlight from previous version

# Map piece symbols to SVG filenames (ensure these files exist in 'images/' subdir)
PIECE_SVG_FILES = {
    'P': 'wP.svg', 'N': 'wN.svg', 'B': 'wB.svg', 'R': 'wR.svg', 'Q': 'wQ.svg', 'K': 'wK.svg',
    'p': 'bP.svg', 'n': 'bN.svg', 'b': 'bB.svg', 'r': 'bR.svg', 'q': 'bQ.svg', 'k': 'bK.svg'
}
IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'images') # Path to 'images' subdirectory


class PromotionDialog(QDialog):
    """升变选择对话框"""

    def __init__(self, parent=None, is_white=True):
        super().__init__(parent)
        self.selected_piece = None
        self.is_white = is_white
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("棋子升变")
        layout = QGridLayout()

        # 可选的升变棋子
        pieces = ['Q', 'R', 'B', 'N'] if self.is_white else ['q', 'r', 'b', 'n']
        piece_names = ['后', '车', '象', '马']

        # 创建棋子按钮
        for i, (piece_symbol, name) in enumerate(zip(pieces, piece_names)):
            btn = QPushButton(name)
            icon_filename = PIECE_SVG_FILES.get(piece_symbol)

            if icon_filename:
                image_path = os.path.join(IMAGE_DIR, icon_filename)
                if os.path.exists(image_path):
                    icon = QIcon(image_path)
                    btn.setIcon(icon)
                    btn.setIconSize(QSize(50, 50)) # Adjust size as needed
                else:
                    print(f"Warning: Image file not found: {image_path}")
                    btn.setText(f"{name} ({piece_symbol})") # Fallback text
                    btn.setIcon(QIcon()) # Clear icon
            else:
                 btn.setText(f"{name} ({piece_symbol})") # Fallback text
                 btn.setIcon(QIcon())

            # 指定按钮功能
            piece_type = piece_symbol.lower()
            # Use checked signal for lambda with arguments
            btn.clicked.connect(lambda checked=False, p=piece_type: self.select_piece(p))


            layout.addWidget(btn, 0, i)

        self.setLayout(layout)

    def select_piece(self, piece):
        """设置选择的棋子"""
        self.selected_piece = piece
        self.accept()


class ChessSquare(QPushButton):
    """棋盘方格按钮"""

    def __init__(self, row, col, size=60): # Default size reduced slightly
        super().__init__()
        self.row = row
        self.col = col
        self.size = size
        self.piece = None
        self.highlighted_type = None # None, 'selected', 'possible', 'last'
        self.is_dark = (row + col) % 2 != 0 # Store if the square is dark or light
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        self.setFixedSize(self.size, self.size)
        self.update_style() # Use a single method for styling

    def update_style(self):
        """更新方格颜色和边框"""
        style = "QPushButton { border: none; " # Base style with no border

        # Determine background color based on highlight type and base color
        if self.highlighted_type == 'selected':
            style += "background-color: #add8e6;" # Light Blue
        elif self.highlighted_type == 'possible':
            style += "background-color: #90ee90;" # Light Green
        elif self.highlighted_type == 'last':
            style += "background-color: #ffffb3;" # Light Yellow
        else:
            # Standard square color
            if self.is_dark:
                style += "background-color: #b58863;" # Dark Brown
            else:
                style += "background-color: #f0d9b5;" # Light Tan

        style += " }"
        self.setStyleSheet(style)
        # self.update() # update() can sometimes cause issues, try without first

    def set_piece(self, piece_symbol):
        """设置方格上的棋子 (using SVG)"""
        self.piece = piece_symbol
        self.update_icon()

    def update_icon(self):
        """更新方格图标 (using SVG)"""
        if self.piece:
            icon_filename = PIECE_SVG_FILES.get(self.piece)
            if icon_filename:
                image_path = os.path.join(IMAGE_DIR, icon_filename)
                if os.path.exists(image_path):
                    icon = QIcon(image_path)
                    self.setIcon(icon)
                    self.setIconSize(QSize(int(self.size * 0.85), int(self.size * 0.85)))
                    self.setText("")
                else:
                    # print(f"Warning: Icon file not found: {image_path}") # Reduce log spam
                    self.setText(self.piece)
                    self.setIcon(QIcon())
            else:
                # print(f"Warning: No SVG defined for piece: {self.piece}") # Reduce log spam
                self.setText(self.piece)
                self.setIcon(QIcon())
        else:
            self.setText("")
            self.setIcon(QIcon())

    def set_highlight(self, highlight_type):
        """设置高亮类型: None, 'selected', 'possible', 'last'"""
        # Only update if the type changes to avoid unnecessary redraws
        if self.highlighted_type != highlight_type:
            self.highlighted_type = highlight_type
            self.update_style()


class AIThread(QThread):
    """AI思考线程"""
    move_selected = pyqtSignal(str)

    def __init__(self, player, env):
        super().__init__()
        self.player = player
        # Ensure ChessEnv.copy() is robust for threading
        try:
            # CRITICAL: Make sure env.copy() creates a truly independent board state
            self.env_copy = env.copy()
            if not hasattr(self.env_copy, 'board') or self.env_copy.board is env.board:
                 # This check is important if copy is shallow
                 print("Error: AIThread detected shallow copy of environment/board!")
                 # Fallback: create a new board from FEN? self.env_copy.board = chess.Board(env.board.fen())
                 self.env_copy = None # Signal error if copy is bad
        except Exception as e:
             print(f"Error copying environment for AI thread: {e}")
             traceback.print_exc()
             self.env_copy = None # Indicate failure

    def run(self):
        """执行AI思考"""
        if not self.env_copy:
             self.move_selected.emit("error")
             return
        if not self.player:
             print("Error: AI Player object is None in AIThread.")
             self.move_selected.emit("error")
             return

        try:
            move = self.player.get_action(self.env_copy, temperature=0) # Use temp=0 for best move
            self.move_selected.emit(move if move else "resign")
        except AttributeError as e:
             print(f"AI thinking error (AttributeError): {e}. Is the AI player initialized correctly?")
             traceback.print_exc()
             self.move_selected.emit("error")
        except Exception as e:
            print(f"AI thinking error: {e}")
            traceback.print_exc() # Print full traceback for debugging
            self.move_selected.emit("error")


class ChessboardWidget(QWidget):
    """棋盘控件"""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.env = ChessEnv().reset()
        self.selected_square_coord = None
        self.squares = {}
        self.human_color = chess.WHITE
        self.last_move_uci = None

        log_dir = getattr(config, 'LOG_DIR', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'play_gui.log')
        # Set logging level to DEBUG to capture more info
        self.logger = setup_logger("chess_gui", log_file)
        self.logger.setLevel(logging.DEBUG) # Ensure debug messages are processed
        self.logger.info("Chess GUI Initializing...")
        self.logger.debug(f"Initial FEN: {self.env.board.fen()}")

        self.model = None
        self.ai_player = None
        self.load_ai_model()

        self.human_player = HumanPlayer(self.config)
        self.ai_thread = None
        self.game_active = True

        self.init_ui()

        # If AI plays White, start its move after UI is ready
        if self.human_color == chess.BLACK:
             self.logger.info("Human plays Black, starting AI (White) move timer.")
             QTimer.singleShot(500, self.ai_move)

    # --- (load_ai_model, init_ui, get_square_widget, get_square_index methods mostly unchanged from previous version) ---
    # --- Ensure init_ui sets the layout and calls update_board_display and update_status_label ---
    def load_ai_model(self, model_path=None):
        """加载AI模型 (optional path overrides default)"""
        path_to_load = model_path or getattr(self.config, 'best_model_path', None)
        best_model_path_exists = path_to_load and os.path.exists(path_to_load)

        try:
            self.logger.info("Attempting to load AI model...")
            self.model = AlphaZeroModel(self.config) # Re-instantiate

            if best_model_path_exists:
                self.logger.info(f"Loading model from: {path_to_load}")
                if self.model.load_model(path_to_load):
                    self.ai_player = AlphaZeroPlayer(self.config, self.model)
                    self.logger.info("Successfully loaded AlphaZeroPlayer.")
                    return True
                else:
                    self.logger.error(f"Model loading failed from {path_to_load}. Using RandomPlayer.")
            else:
                 if path_to_load: self.logger.warning(f"Model file not found: {path_to_load}.")
                 else: self.logger.warning("No best_model_path configured or found.")

            self.logger.warning("Using RandomPlayer as fallback/default.")
            # Ensure RandomPlayer is instantiated if needed
            if not isinstance(self.ai_player, RandomPlayer):
                 self.ai_player = RandomPlayer(self.config)
            return False # Indicate model wasn't loaded as AlphaZero

        except AttributeError as e:
             self.logger.error(f"Error loading AI model (AttributeError): {e}. Check AlphaZeroModel/Player init or config.")
             traceback.print_exc()
             self.ai_player = RandomPlayer(self.config)
             return False
        except Exception as e:
            self.logger.error(f"Error loading AI model: {e}")
            traceback.print_exc()
            self.ai_player = RandomPlayer(self.config)
            return False

    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Arial", 14, QFont.Bold))
        main_layout.addWidget(self.status_label)

        board_layout = QGridLayout()
        board_layout.setSpacing(0)

        gui_cfg = getattr(self.config, 'GUIConfig', None)
        cell_size = getattr(gui_cfg, 'cell_size', 60)

        for qt_row in range(8):
            for qt_col in range(8):
                square_widget = ChessSquare(qt_row, qt_col, size=cell_size)
                square_widget.clicked.connect(self.square_clicked)
                board_layout.addWidget(square_widget, qt_row, qt_col)
                self.squares[(qt_row, qt_col)] = square_widget

        font = QFont("Arial", 10)
        for col in range(8):
            file_label = QLabel(chess.FILE_NAMES[col])
            file_label.setAlignment(Qt.AlignCenter); file_label.setFont(font)
            board_layout.addWidget(file_label, 8, col)
        for row in range(8):
            rank_label = QLabel(chess.RANK_NAMES[7 - row])
            rank_label.setAlignment(Qt.AlignCenter); rank_label.setFont(font)
            board_layout.addWidget(rank_label, row, 8)

        board_container = QWidget()
        board_container.setLayout(board_layout)
        main_layout.addWidget(board_container, stretch=1)

        control_layout = QHBoxLayout()
        new_game_btn = QPushButton("New Game")
        new_game_btn.clicked.connect(self.new_game)
        control_layout.addWidget(new_game_btn)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["Play as White", "Play as Black"])
        self.color_combo.setCurrentIndex(0 if self.human_color == chess.WHITE else 1)
        self.color_combo.currentIndexChanged.connect(self.change_side)
        control_layout.addWidget(self.color_combo)

        resign_btn = QPushButton("Resign")
        resign_btn.clicked.connect(self.resign)
        control_layout.addWidget(resign_btn)
        main_layout.addLayout(control_layout)

        self.update_board_display()
        self.update_status_label()

    def get_square_widget(self, square_index: chess.Square) -> ChessSquare | None:
        """Get the ChessSquare widget corresponding to a chess.Square index."""
        rank = chess.square_rank(square_index) # 0-7 (rank 1 to 8)
        file = chess.square_file(square_index) # 0-7 (file a to h)
        qt_row = 7 - rank
        qt_col = file
        return self.squares.get((qt_row, qt_col))

    def get_square_index(self, qt_row: int, qt_col: int) -> chess.Square:
        """Convert Qt row/col to chess.Square index."""
        rank = 7 - qt_row
        file = qt_col
        # Basic validation
        if 0 <= rank <= 7 and 0 <= file <= 7:
             return chess.square(file, rank)
        else:
             raise ValueError(f"Invalid Qt coordinates for chess square: row={qt_row}, col={qt_col}")


    def square_clicked(self):
        """处理方格点击事件 - Enhanced Debugging"""
        sender = self.sender()
        if not isinstance(sender, ChessSquare): return

        qt_row, qt_col = sender.row, sender.col
        try:
             clicked_square_index = self.get_square_index(qt_row, qt_col)
        except ValueError as e:
             self.logger.error(f"Error getting square index: {e}")
             return
        clicked_square_coord = (qt_row, qt_col)

        # Log current state *before* any action
        self.logger.debug(f"--- square_clicked START ({chess.square_name(clicked_square_index)}) ---")
        self.logger.debug(f"Game Active: {self.game_active}, Human Color: {'W' if self.human_color else 'B'}")
        self.logger.debug(f"Env Turn: {'W' if self.env.white_to_move else 'B'}, Board Turn: {'W' if self.env.board.turn else 'B'}")
        self.logger.debug(f"Current FEN: {self.env.board.fen()}")
        self.logger.debug(f"Selected Coord: {self.selected_square_coord}")

        if not self.game_active:
            self.logger.debug("Ignoring click: Game not active.")
            return
        is_human_turn = (self.env.white_to_move == (self.human_color == chess.WHITE))
        if not is_human_turn:
            self.logger.debug("Ignoring click: Not human's turn.")
            # Deselect if clicked during opponent's turn
            if self.selected_square_coord:
                self.clear_all_highlights()
                self.selected_square_coord = None
            return

        piece_on_clicked = self.env.board.piece_at(clicked_square_index)

        if self.selected_square_coord is None:
            # 1. Try to select a piece
            if piece_on_clicked and piece_on_clicked.color == self.human_color:
                self.logger.info(f"Selected piece at {chess.square_name(clicked_square_index)}")
                self.selected_square_coord = clicked_square_coord
                self.clear_all_highlights()
                sender.set_highlight('selected')
                self.highlight_legal_moves(clicked_square_index) # Highlight moves from the selected square
            else:
                self.logger.debug("Clicked empty square or opponent's piece - no selection.")
                self.clear_all_highlights() # Should already be clear, but ensure
                self.selected_square_coord = None
        else:
            # 2. Piece already selected - try to move or change selection
            selected_qt_row, selected_qt_col = self.selected_square_coord
            from_square_index = self.get_square_index(selected_qt_row, selected_qt_col)
            to_square_index = clicked_square_index # This is the target

            if clicked_square_coord == self.selected_square_coord: # Clicked same square
                self.logger.debug("Clicked selected square again: Deselecting.")
                self.clear_all_highlights()
                self.selected_square_coord = None
                return

            # Construct potential move
            # Check promotion FIRST, as it changes the move object
            is_promo = self.is_promotion_move(chess.Move(from_square_index, to_square_index))
            promotion_piece = None
            if is_promo:
                 promotion_dialog = PromotionDialog(self, is_white=self.human_color)
                 if promotion_dialog.exec_():
                     promotion_choice = promotion_dialog.selected_piece
                     promotion_piece = self.get_promotion_piece_type(promotion_choice)
                     self.logger.debug(f"Promotion selected: {promotion_choice} -> {promotion_piece}")
                 else:
                     self.logger.debug("Promotion cancelled.")
                     # Keep selection, maybe clear highlights? Or just return? Let's return.
                     # self.clear_all_highlights(keep_last_move=True) # Clear possible moves only
                     # selected_widget = self.squares.get(self.selected_square_coord)
                     # if selected_widget: selected_widget.set_highlight('selected')
                     return # Don't proceed if promotion is cancelled

            # Create the final move object (potentially with promotion)
            move = chess.Move(from_square_index, to_square_index, promotion=promotion_piece)
            move_uci = move.uci()

            # --- CRITICAL DEBUGGING STEP ---
            # Check legality *using the environment's state* right before attempting the move
            current_legal_moves = self.env.get_legal_moves()
            is_legal_in_env = move in current_legal_moves

            self.logger.debug(f"Attempting move: {move_uci}")
            self.logger.debug(f"Checking if '{move_uci}' is in env's legal moves:")
            # Log first few and last few legal moves for comparison
            legal_moves_uci = [m.uci() for m in current_legal_moves]
            log_subset = legal_moves_uci[:5] + ["..."] + legal_moves_uci[-5:] if len(legal_moves_uci) > 10 else legal_moves_uci
            self.logger.debug(f"Env legal moves ({len(legal_moves_uci)}): {log_subset}")
            self.logger.debug(f"Move '{move_uci}' is legal according to env? {is_legal_in_env}")
            # --- END CRITICAL DEBUGGING ---

            if is_legal_in_env:
                # If the check here passes, the subsequent call to make_move -> env.step
                # SHOULD also pass, unless env.step has different logic or state.
                self.logger.info(f"Move {move_uci} deemed legal by GUI check. Calling make_move.")
                self.make_move(move_uci) # Let make_move handle execution and AI trigger
                # Selection is reset inside make_move upon success/failure normally
                # If make_move fails unexpectedly, selection might remain, handle there.
            else:
                # Move is NOT legal according to env.get_legal_moves()
                self.logger.warning(f"Move {move_uci} is NOT in env.get_legal_moves().")
                # Check if clicked on another own piece to change selection
                if piece_on_clicked and piece_on_clicked.color == self.human_color:
                    self.logger.debug(f"Changing selection to {chess.square_name(clicked_square_index)}")
                    self.selected_square_coord = clicked_square_coord
                    self.clear_all_highlights()
                    sender.set_highlight('selected')
                    self.highlight_legal_moves(clicked_square_index)
                else:
                    # Clicked invalid target or opponent piece: Deselect
                    self.logger.debug("Clicked invalid target square: Deselecting.")
                    # Provide user feedback about the illegal move attempt
                    QMessageBox.warning(self, "Invalid Move", f"Move {move_uci} is not legal.")
                    self.clear_all_highlights()
                    self.selected_square_coord = None
        self.logger.debug(f"--- square_clicked END ---")


    def is_promotion_move(self, move: chess.Move) -> bool:
        """Checks if a move is a pawn promotion."""
        # Need the board state to check the piece type
        piece = self.env.board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False
        to_rank_index = chess.square_rank(move.to_square)
        return (piece.color == chess.WHITE and to_rank_index == 7) or \
               (piece.color == chess.BLACK and to_rank_index == 0)

    def get_promotion_piece_type(self, piece_char: str) -> int | None:
        """Converts promotion choice ('q', 'r', 'b', 'n') to chess.PieceType."""
        if not piece_char: return None # Handle cancellation case
        piece_map = {'q': chess.QUEEN, 'r': chess.ROOK, 'b': chess.BISHOP, 'n': chess.KNIGHT}
        return piece_map.get(piece_char.lower(), chess.QUEEN)

    def highlight_legal_moves(self, from_square_index: chess.Square):
        """Highlights all legal destination squares for the piece at from_square_index."""
        try:
            legal_moves = [m for m in self.env.get_legal_moves() if m.from_square == from_square_index]
            self.logger.debug(f"Highlighting {len(legal_moves)} moves from {chess.square_name(from_square_index)}")
            for move in legal_moves:
                to_sq_widget = self.get_square_widget(move.to_square)
                if to_sq_widget:
                    to_sq_widget.set_highlight('possible')
        except Exception as e:
             self.logger.error(f"Error highlighting legal moves: {e}")
             traceback.print_exc()


    def clear_all_highlights(self, keep_last_move=True):
        """Clears selection and possible highlights. Optionally keeps last move."""
        # Don't reset selected_square_coord here, manage it in square_clicked
        for square_widget in self.squares.values():
             # Clear 'selected' and 'possible' highlights
             if square_widget.highlighted_type in ['selected', 'possible']:
                 square_widget.set_highlight(None)
             # If not keeping last move, clear it too
             elif not keep_last_move and square_widget.highlighted_type == 'last':
                  square_widget.set_highlight(None)

        # Re-apply last move highlight if requested and needed
        if keep_last_move and self.last_move_uci:
            try:
                last_move = chess.Move.from_uci(self.last_move_uci)
                from_sq_widget = self.get_square_widget(last_move.from_square)
                to_sq_widget = self.get_square_widget(last_move.to_square)
                # Only set 'last' if not currently selected or possible
                if from_sq_widget and from_sq_widget.highlighted_type is None:
                     from_sq_widget.set_highlight('last')
                if to_sq_widget and to_sq_widget.highlighted_type is None:
                     to_sq_widget.set_highlight('last')
            except ValueError:
                 self.logger.error(f"Invalid UCI string stored for last move: {self.last_move_uci}")
                 self.last_move_uci = None


    def make_move(self, move_uci: str):
        """Executes a move, updates state, triggers AI if needed."""
        if not self.game_active:
            self.logger.warning(f"Attempted move '{move_uci}' while game not active.")
            return

        current_player = "White" if self.env.white_to_move else "Black"
        self.logger.info(f"--- make_move START ({move_uci} by {current_player}) ---")

        if move_uci.lower() == "resign":
            self.handle_resignation(resigning_player_is_white=self.env.white_to_move)
            self.logger.info(f"--- make_move END (Resign) ---")
            return

        # --- MORE DEBUGGING before calling env.step ---
        self.logger.debug(f"FEN before env.step({move_uci}): {self.env.board.fen()}")
        self.logger.debug(f"Turn before env.step: {'W' if self.env.board.turn else 'B'}")
        try:
            move_obj = self.env.board.parse_uci(move_uci) # Parse again for safety check? Or trust GUI?
            self.logger.debug(f"Move '{move_uci}' parsed as: {move_obj}")
            is_legal_now = move_obj in self.env.board.legal_moves
            self.logger.debug(f"Is move legal *right now* according to board.legal_moves? {is_legal_now}")
            if not is_legal_now:
                 self.logger.error(f"!!!!! STATE MISMATCH !!!!! Move {move_uci} was legal in GUI check but NOT legal just before env.step!")
                 # Log more state
                 legal_now_uci = [m.uci() for m in self.env.board.legal_moves]
                 self.logger.error(f"Legal moves according to board NOW: {legal_now_uci}")
                 # Don't attempt the step, show error, reset selection
                 QMessageBox.critical(self, "State Error", f"Internal state mismatch prevented move {move_uci}. Please report this.")
                 self.clear_all_highlights()
                 self.selected_square_coord = None
                 self.update_status_label() # Refresh status
                 self.logger.info(f"--- make_move END (State Mismatch Error) ---")
                 return
        except ValueError as e:
             # This happens if the UCI is invalid format, or if the move is illegal for the current position
             # according to python-chess's board.parse_uci (which implicitly checks basic legality like piece movement)
             # *or* if board.push itself fails later.
             self.logger.error(f"ValueError parsing/checking move {move_uci} just before env.step: {e}")
             # It's possible env.step() handles this, but log it here too.
        # --- END MORE DEBUGGING ---


        # Attempt to make the move in the environment
        step_successful = False
        try:
            step_successful = self.env.step(move_uci) # Call the environment's step method
        except Exception as e:
            self.logger.error(f"!!! Unexpected Error during env.step({move_uci}) call !!!")
            self.logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Runtime Error", f"An error occurred in the game engine:\n{e}")
            self.game_active = False # Halt game

        self.logger.info(f"env.step({move_uci}) returned: {step_successful}")

        if step_successful:
             self.last_move_uci = move_uci
             self.selected_square_coord = None # Clear selection on successful move
             self.update_board_display() # Update visual board (clears old hl, shows new state, adds last move hl)
             self.update_status_label()

             if self.env.done:
                 self.game_active = False
                 self.show_game_result()
                 self.logger.info("Game finished.")
             else:
                 is_ai_turn = (self.env.white_to_move != (self.human_color == chess.WHITE))
                 if is_ai_turn:
                     self.logger.info("Triggering AI move timer.")
                     QTimer.singleShot(100, self.ai_move) # Short delay
        else:
             # env.step returned False - indicates illegal move internally
             self.logger.error(f"Move {move_uci} rejected by env.step() internal logic.")
             # Provide feedback to user
             QMessageBox.warning(self, "Invalid Move", f"The move '{move_uci}' was rejected by the game engine.")
             # Clear selection state as the move ultimately failed
             self.clear_all_highlights()
             self.selected_square_coord = None

        self.logger.info(f"--- make_move END ({move_uci}) ---")

    # --- (ai_move, on_ai_move_received, on_ai_thread_finished unchanged) ---
    def ai_move(self):
        """Initiates the AI thinking process in a separate thread."""
        self.logger.debug("--- ai_move called ---")
        if not self.game_active:
            self.logger.info("AI move request ignored: game not active.")
            return

        is_ai_turn = (self.env.white_to_move != (self.human_color == chess.WHITE))
        if not is_ai_turn:
             self.logger.warning("AI move requested, but it's not AI's turn.")
             # Force status update just in case
             self.update_status_label()
             return

        if self.ai_thread and self.ai_thread.isRunning():
            self.logger.warning("AI is already thinking!")
            return

        if not self.ai_player:
             self.logger.error("Cannot start AI move: AI player not loaded.")
             QMessageBox.critical(self, "Error", "AI Player not available.")
             return

        self.status_label.setText("AI is thinking...")
        QApplication.processEvents()

        self.logger.info("Starting AI thinking thread...")
        self.logger.debug(f"FEN passed to AI thread: {self.env.board.fen()}") # Log state passed to AI
        self.ai_thread = AIThread(self.ai_player, self.env) # Pass current env
        self.ai_thread.move_selected.connect(self.on_ai_move_received)
        self.ai_thread.finished.connect(self.on_ai_thread_finished)
        self.ai_thread.start()

    def on_ai_move_received(self, move_uci: str):
        """Handles the move received from the AI thread."""
        self.logger.info(f"--- on_ai_move_received START ({move_uci}) ---")
        if not self.game_active:
             self.logger.info("Ignoring AI move: Game ended while AI was thinking.")
             self.logger.info(f"--- on_ai_move_received END (Game Inactive) ---")
             return

        if move_uci == "error":
            self.logger.error("AI thread reported an error.")
            self.status_label.setText("AI Error! Try again?")
            QMessageBox.warning(self, "AI Error", "The AI encountered an error while thinking.")
        elif move_uci == "resign":
             self.logger.info("AI resigns.")
             self.handle_resignation(resigning_player_is_white=self.env.white_to_move)
        elif not move_uci:
             self.logger.error("AI thread returned an empty move.")
             self.status_label.setText("AI Error (Empty Move)! Try again?")
             QMessageBox.warning(self, "AI Error", "The AI returned an empty move.")
        else:
            # AI returned a potential move UCI string
            self.logger.info(f"AI proposed move: {move_uci}. Calling make_move.")
            self.make_move(move_uci) # Execute the AI's move

        self.logger.info(f"--- on_ai_move_received END ({move_uci}) ---")


    def on_ai_thread_finished(self):
        """Called when the AI thread terminates."""
        self.logger.debug("AI thinking thread finished.")
        # Check if thread object still exists before trying to delete or access
        if self.ai_thread is not None:
             # Clean up reference to the thread
             # self.ai_thread.deleteLater() # Optionally schedule for deletion
             self.ai_thread = None
        # Status label should have been updated by subsequent actions

    def update_board_display(self):
        """Updates the visual representation of the board."""
        self.logger.debug("--- update_board_display START ---")
        # 1. Clear selection/possible highlights, keep last move highlight
        self.clear_all_highlights(keep_last_move=True)

        # 2. Place pieces according to the current board state
        current_fen = self.env.board.fen() # Get FEN once for logging
        self.logger.debug(f"Updating display for FEN: {current_fen}")
        for square_index in chess.SQUARES:
            piece = self.env.board.piece_at(square_index)
            square_widget = self.get_square_widget(square_index)
            if square_widget:
                square_widget.set_piece(piece.symbol() if piece else None)
                # Ensure squares *without* last move highlight are reset correctly
                if square_widget.highlighted_type == 'last':
                     is_last = False
                     if self.last_move_uci:
                         try:
                             last_m = chess.Move.from_uci(self.last_move_uci)
                             sq_idx = self.get_square_index(square_widget.row, square_widget.col)
                             if sq_idx == last_m.from_square or sq_idx == last_m.to_square:
                                 is_last = True
                         except: pass # Ignore errors parsing last move here
                     if not is_last:
                          square_widget.set_highlight(None) # Remove stale 'last' highlight


        # 3. Re-apply selection highlight if a piece is selected (should be rare here)
        if self.selected_square_coord:
             self.logger.warning("Selected square coord found during update_board_display - unexpected?")
             selected_widget = self.squares.get(self.selected_square_coord)
             if selected_widget: selected_widget.set_highlight('selected')

        # 4. Ensure last move highlights are definitely set (redundant with clear_all_highlights?)
        # self.apply_last_move_highlight() # Call separate function if needed

        QApplication.processEvents() # Process pending UI updates maybe? Careful with loops.
        self.logger.debug("--- update_board_display END ---")


    # --- (update_status_label, show_game_result, new_game, change_side, handle_resignation, resign methods mostly unchanged) ---
    def update_status_label(self):
        """Updates the text of the status label based on game state."""
        # Simplified logic combining active/done checks
        msg = ""
        if self.game_active and not self.env.done:
             turn_color_env = "White" if self.env.white_to_move else "Black"
             turn_color_board = "White" if self.env.board.turn else "Black"
             if turn_color_env != turn_color_board: # State mismatch check
                  self.logger.warning(f"!!! Turn Mismatch! env.white_to_move={self.env.white_to_move} but board.turn={self.env.board.turn}")
             is_human_turn = (self.env.board.turn == self.human_color) # Use board.turn as primary source?
             player = "Your" if is_human_turn else "AI's"
             msg = f"{player} turn ({turn_color_board})"
             if self.env.board.is_check():
                 msg += " - Check!"
        else: # Game is over or inactive
             self.game_active = False # Ensure inactive
             result_str = self.env.board.result() # Get result from board
             self.env.result = result_str # Sync env result
             if self.env.board.is_checkmate():
                  winner = "Black" if self.env.board.turn else "White"
                  msg = f"Checkmate! {winner} Wins!"
                  self.env.winner = Winner.BLACK if self.env.board.turn else Winner.WHITE
             elif self.env.board.is_stalemate():
                  msg = "Stalemate! Draw."; self.env.winner = Winner.DRAW
             elif self.env.board.is_insufficient_material():
                  msg = "Insufficient Material. Draw."; self.env.winner = Winner.DRAW
             # Add other draw conditions if needed (75 moves, 5-rep)
             elif result_str == "1-0": msg = "Game Over: White Wins!"; self.env.winner = Winner.WHITE
             elif result_str == "0-1": msg = "Game Over: Black Wins!"; self.env.winner = Winner.BLACK
             elif result_str == "1/2-1/2": msg = "Game Over: Draw!"; self.env.winner = Winner.DRAW
             else: msg = f"Game Over ({result_str})" # Catch-all

        self.status_label.setText(msg)
        # self.logger.debug(f"Status updated: {msg}") # Reduce log spam
        # QApplication.processEvents() # Avoid calling this too often

    def show_game_result(self):
        """Displays a message box with the final game result."""
        # Use the winner determined in update_status_label based on board state
        title = "Game Over"
        message = f"Game finished. Result: {self.env.result}" # Default message

        if self.env.winner == Winner.WHITE:
            title = "White Wins!"
            message = f"White wins! ({self.env.result})"
        elif self.env.winner == Winner.BLACK:
            title = "Black Wins!"
            message = f"Black wins! ({self.env.result})"
        elif self.env.winner == Winner.DRAW:
            title = "Draw"
            message = f"The game is a draw! ({self.env.result})"

        self.logger.info(f"Displaying result: {title} - {message}")
        QMessageBox.information(self, title, message)

    def new_game(self):
        """Starts a new game."""
        self.logger.info("--- new_game START ---")
        if self.ai_thread and self.ai_thread.isRunning():
            self.logger.warning("Terminating active AI thread for new game.")
            # Use quit() and wait() for graceful shutdown if possible
            self.ai_thread.quit()
            if not self.ai_thread.wait(1000): # Wait 1 sec
                 self.logger.warning("AI thread did not quit gracefully, terminating.")
                 self.ai_thread.terminate()
                 self.ai_thread.wait() # Wait after terminate too
            self.ai_thread = None

        self.env.reset() # Reset the environment
        self.selected_square_coord = None
        self.last_move_uci = None
        self.game_active = True
        self.logger.info(f"New game started. Human is {'White' if self.human_color else 'Black'}.")
        self.logger.debug(f"FEN after reset: {self.env.board.fen()}")

        self.update_board_display() # Update visuals
        self.update_status_label()

        if self.human_color == chess.BLACK:
             self.logger.info("Human is Black, starting AI (White) move timer.")
             QTimer.singleShot(100, self.ai_move)
        self.logger.info("--- new_game END ---")


    def change_side(self, index: int):
        """Changes the human player's color and starts a new game."""
        new_color = chess.WHITE if index == 0 else chess.BLACK
        if new_color == self.human_color: return

        side_str = "White" if new_color == chess.WHITE else "Black"
        self.logger.info(f"Changing human player color to: {side_str}")
        self.human_color = new_color
        # Update the combo box in case this was called programmatically
        self.color_combo.setCurrentIndex(index)
        self.new_game()


    def handle_resignation(self, resigning_player_is_white: bool):
        """Handles resignation logic."""
        if not self.game_active: return

        winner_color = "Black" if resigning_player_is_white else "White"
        loser_color = "White" if resigning_player_is_white else "Black"
        self.logger.info(f"{loser_color} resigns. {winner_color} wins.")

        self.env.winner = Winner.BLACK if resigning_player_is_white else Winner.WHITE
        self.env.result = "0-1" if resigning_player_is_white else "1-0"
        self.env.done = True
        self.game_active = False

        self.update_status_label()
        self.show_game_result()
        # Update display to clear highlights etc.
        self.update_board_display()


    def resign(self):
        """Handles the human player resigning."""
        if not self.game_active: return

        is_human_turn = (self.env.board.turn == self.human_color)
        # Allow resigning even if it's not your turn? Standard GUIs often do.
        # if not is_human_turn:
        #      self.logger.warning("Resign clicked, but not human's turn.")
        #      return

        reply = QMessageBox.question(self, "Confirm Resignation",
                                     "Are you sure you want to resign?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.logger.info("Human player resigns.")
            self.handle_resignation(resigning_player_is_white=(self.human_color == chess.WHITE))


# --- (ChessGUI class definition largely unchanged, ensure it uses the updated ChessboardWidget) ---
class ChessGUI(QMainWindow):
    """国际象棋GUI主窗口"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        if not hasattr(config, 'GUIConfig'): config.GUIConfig = type('DummyGUI', (object,), {'window_width':600, 'window_height':700, 'cell_size':60})()
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle("AlphaZero Chess GUI (Debug Enabled)")
        self.resize(self.config.GUIConfig.window_width, self.config.GUIConfig.window_height)
        try: # Set window icon safely
             wking_path = os.path.join(IMAGE_DIR, PIECE_SVG_FILES.get('K', 'wK.svg'))
             if os.path.exists(wking_path): self.setWindowIcon(QIcon(wking_path))
        except Exception as e: self.config.logger.warning(f"Could not set window icon: {e}")

        self.statusBar().showMessage("Ready")
        self.chessboard = ChessboardWidget(self.config, self)
        self.setCentralWidget(self.chessboard)
        self.create_menus()
        self.show()
    def create_menus(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        try: # Set action icons safely
            new_icon_path = os.path.join(IMAGE_DIR, PIECE_SVG_FILES.get('P', 'wP.svg'))
            new_icon = QIcon(new_icon_path) if os.path.exists(new_icon_path) else QIcon.fromTheme("document-new")
            exit_icon = QIcon.fromTheme("application-exit")
        except Exception as e: self.config.logger.warning(f"Could not load menu icons: {e}"); new_icon=QIcon(); exit_icon=QIcon()

        new_game_action = QAction(new_icon, "&New Game", self); new_game_action.setShortcut("Ctrl+N")
        new_game_action.setStatusTip("Start a new chess game"); new_game_action.triggered.connect(self.chessboard.new_game)
        file_menu.addAction(new_game_action)

        load_model_action = QAction("&Load AI Model...", self); load_model_action.setStatusTip("Load AI model checkpoint")
        load_model_action.triggered.connect(self.load_model_dialog); file_menu.addAction(load_model_action)
        file_menu.addSeparator()
        exit_action = QAction(exit_icon, "&Exit", self); exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application"); exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        game_menu = menu_bar.addMenu("&Game")
        resign_action = QAction("&Resign", self); resign_action.setStatusTip("Resign current game")
        resign_action.triggered.connect(self.chessboard.resign); game_menu.addAction(resign_action)

        help_menu = menu_bar.addMenu("&Help")
        about_action = QAction("&About", self); about_action.setStatusTip("Show application information")
        about_action.triggered.connect(self.show_about_dialog); help_menu.addAction(about_action)
    def load_model_dialog(self):
        start_dir = getattr(self.config, 'CHECKPOINT_DIR', '.')
        model_path, _ = QFileDialog.getOpenFileName(self,"Select AI Model File",start_dir,"PyTorch Model Files (*.pt *.pth);;All Files (*)")
        if model_path:
            self.statusBar().showMessage(f"Loading model: {os.path.basename(model_path)}...")
            QApplication.processEvents()
            success = self.chessboard.load_ai_model(model_path)
            if success:
                 QMessageBox.information(self, "Model Loaded", f"Successfully loaded AI model:\n{model_path}")
                 self.statusBar().showMessage("AI Model Loaded Successfully")
                 self.chessboard.new_game() # Start new game with new model
            else:
                 QMessageBox.warning(self, "Load Failed", f"Could not load AI model from:\n{model_path}\n\nFalling back to default/random player.")
                 self.statusBar().showMessage("Failed to load AI model")
    def show_about_dialog(self):
        QMessageBox.about(self,"About AlphaZero Chess","<h2>AlphaZero Chess GUI</h2><p>Interface for playing against an AlphaZero engine.</p><p>Built with Python, PyQt5, python-chess.</p>")
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Confirm Exit',"Are you sure you want to exit?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
             if self.chessboard.ai_thread and self.chessboard.ai_thread.isRunning():
                  self.chessboard.logger.info("Attempting graceful AI thread shutdown...")
                  self.chessboard.ai_thread.quit()
                  if not self.chessboard.ai_thread.wait(1000):
                       self.chessboard.logger.warning("AI thread did not quit gracefully, terminating.")
                       self.chessboard.ai_thread.terminate()
                       self.chessboard.ai_thread.wait()
             self.chessboard.logger.info("Application closing.")
             event.accept()
        else:
             event.ignore()


# --- (start_gui function and __main__ block unchanged) ---
def start_gui(config):
    """Starts the PyQt5 GUI application."""
    # Basic checks
    if not os.path.isdir(IMAGE_DIR):
        print(f"Error: 'images' directory not found at: {IMAGE_DIR}")
        # Try to create it?
        try: os.makedirs(IMAGE_DIR); print(f"Created directory: {IMAGE_DIR}. Please add SVG files.")
        except OSError as e: print(f"Could not create images directory: {e}"); return 1
    if not hasattr(config, 'LOG_DIR'): config.LOG_DIR = 'logs'
    if not hasattr(config, 'logger'): # Ensure logger exists on config if ChessGUI uses it
         config.logger = setup_logger("chess_config", os.path.join(config.LOG_DIR, 'config.log'))
         config.logger.setLevel(logging.DEBUG)


    app = QApplication(sys.argv)
    app.setApplicationName("AlphaZero Chess")
    gui = ChessGUI(config)
    return app.exec_()

# if __name__ == '__main__':
#     class TestConfig: # Keep the TestConfig definition as before
#         class GUIConfig: cell_size = 60; window_width = 600; window_height = 700
#         LOG_DIR = "logs"; best_model_path = None; CHECKPOINT_DIR = "models"
#         class Play: simulation_num_per_move=50; thinking_loop=1; c_puct=1.0; parallel_search_num=8; search_threads=8; dirichlet_alpha=0.3; tau_decay_rate=0.99; virtual_loss=1
#         class Model: cnn_filter_num=32; cnn_filter_size=3; res_layer_num=3; l2_reg=1e-4; value_fc_size=64; input_depth=18
#         # Add logger to test config
#         logger = setup_logger("test_config", os.path.join(LOG_DIR, 'test_config.log'))
#         logger.setLevel(logging.DEBUG)
#
#
#     print("Running GUI in standalone mode with test config.")
#     if not os.path.exists(IMAGE_DIR): os.makedirs(IMAGE_DIR); print(f"Created directory: {IMAGE_DIR}. Add SVGs.")
#     test_config = TestConfig()
#     sys.exit(start_gui(test_config))