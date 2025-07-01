#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self‑Play generator (fixed empty‑stats bug)
==========================================
• Temperature schedule 30/60 plies
• Uses MCTS visit count as π label
• Gracefully handles positions where MCTS returns empty dict (no legal moves)
"""

from __future__ import annotations

import os
import time
import numpy as np
import chess
from datetime import datetime
from typing import List, Tuple, Optional
from multiprocessing import Pool
from tqdm import tqdm

from ..config import Config
from ..model import AlphaZeroModel
from ..env import ChessEnv, Winner
from ..mcts import MCTS
from ..utils import save_game, setup_logger

# ───────────────────────── helpers ──────────────────────────

def _temperature(step: int) -> float:
    if step < 30:
        return 1.0
    if step < 60:
        return 0.5
    return 0.0

# ───────────────────────── single game ─────────────────────

def play_game(cfg: Config, model: AlphaZeroModel, gid: int, verbose: bool = False) -> Tuple[List, Optional[chess.pgn.Game]]:
    """Run one self‑play game and output (state, π, z) list and PGN."""
    env = ChessEnv().reset()
    mcts = MCTS(cfg, model)

    import chess.pgn  # local import
    pgn_root = chess.pgn.Game()
    node = pgn_root

    data: List = []
    step = 0
    start = time.time()

    while not env.done:
        state = env.get_observation()
        temp = _temperature(step)

        stats = mcts.search(env)  # {move: (N, Q)}

        if not stats:
            # 没有合法着法（应当是将死/和棋）；终止循环
            break

        moves, visits = zip(*[(m, s[0]) for m, s in stats.items()])
        visits = np.asarray(visits, dtype=np.float32)

        if temp == 0:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
        else:
            probs = visits ** (1.0 / temp)
            probs /= probs.sum()

        move = np.random.choice(moves, p=probs)

        # 构建 π 向量
        pi_vec = np.zeros(len(cfg.move_labels), dtype=np.float32)
        for m, p in zip(moves, probs):
            pi_vec[cfg.move_lookup[m.uci()]] = p

        data.append((state, pi_vec, None))

        node = node.add_variation(move)
        env.step(move.uci())
        step += 1

    # ------- assign rewards --------
    if env.winner == Winner.WHITE:
        result_val = 1.0
    elif env.winner == Winner.BLACK:
        result_val = -1.0
    else:
        result_val = 0.0

    for ply, tup in enumerate(data):
        st, pi, _ = tup
        reward = result_val if ply % 2 == 0 else -result_val
        data[ply] = (st, pi, reward)

    pgn_root.headers["Result"] = env.result or "*"

    if verbose:
        print(f"Game {gid} finished | steps {step} | result {env.result} | {time.time()-start:.1f}s")

    return data, pgn_root

# ───────────────────────── worker & runner ─────────────────────

def _worker(args):
    cfg_path, mdl_path, gid, vb = args
    cfg = Config(cfg_path)
    mdl = AlphaZeroModel(cfg)
    if os.path.exists(mdl_path):
        mdl.load_model(mdl_path)
    return play_game(cfg, mdl, gid, vb)


def self_play(cfg: Config, model: AlphaZeroModel, num_games: int | None = None) -> List:
    if num_games is None:
        num_games = cfg.SelfPlayConfig.num_games

    logger = setup_logger("self_play", os.path.join(cfg.LOG_DIR, f"self_play_{datetime.now():%Y%m%d_%H%M%S}.log"))
    logger.info("Self‑play start | games=%d", num_games)

    tmp_model = os.path.join(cfg.CHECKPOINT_DIR, "tmp_sp.pt")
    model.save_model(tmp_model)

    args = [(cfg.BASE_DIR + "/config.yaml", tmp_model, i, i == 0) for i in range(num_games)]
    procs = min(cfg.config['system']['num_processes'], num_games)

    all_samples: List = []
    pgn_out = os.path.join(cfg.DATA_DIR, f"selfplay_{datetime.now():%Y%m%d_%H%M%S}.pgn")

    with Pool(procs) as pool, tqdm(total=num_games, desc="Self‑Play") as bar:
        for d, g in pool.imap(_worker, args):
            if d:
                all_samples.extend(d)
            if g:
                save_game(g, pgn_out)
            bar.update(1)

    os.remove(tmp_model)
    logger.info("Self‑play done | samples=%d", len(all_samples))
    return all_samples
