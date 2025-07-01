#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re‑implemented AlphaZero Trainer (Chess)
========================================
• Correct policy‑loss using visit‑count soft labels (cross‑entropy)
• Cosine LR schedule + proper weight decay on weights only
• DataLoader pin_memory=False → MPS/CPU friendly
• Safe gradient clipping & mixed precision hooks

Compatible with existing package layout: `project_root/src/training/trainer.py`
"""

from __future__ import annotations

import os
import time
import math
import random
from collections import deque
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..config import Config
from ..model import AlphaZeroModel
from ..utils import setup_logger


# ─────────────────────────── Dataset & Buffer ────────────────────────────

class ReplayBuffer:
    """Fixed‑size circular buffer storing (state, π, z)."""

    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, game_data: List[Tuple]):
        for triplet in game_data:
            self.buffer.append(triplet)

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def clear(self):
        self.buffer.clear()


class AlphaZeroDataset(Dataset):
    """Tensor wrapper for replay data."""

    def __init__(self, data: List[Tuple], cfg: Config):
        action_size = len(cfg.move_labels)

        states, policies, values = [], [], []
        for s, pi, z in data:
            states.append(s.astype(np.float32))

            # `pi` is already length‑4672 numpy array (soft label)
            if isinstance(pi, dict):
                vec = np.zeros(action_size, dtype=np.float32)
                for uci, p in pi.items():
                    idx = cfg.move_lookup[uci]
                    vec[idx] = p
                policies.append(vec)
            else:
                policies.append(pi.astype(np.float32))

            values.append(np.float32(z))

        self.states = np.stack(states)
        self.policies = np.stack(policies)
        self.values = np.array(values, dtype=np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


# ─────────────────────────────── Trainer ─────────────────────────────────

class Trainer:
    """Trains AlphaZero network on self‑play data."""

    def __init__(self, cfg: Config, model: AlphaZeroModel):
        self.cfg = cfg
        self.model = model  # wraps AlphaZeroNetwork & device
        self.tc = cfg.TrainingConfig

        # Optimiser (l2 only on weights, not bias/BN)
        wd = self.tc.weight_decay
        decay, no_decay = [], []
        for n, p in self.model.model.named_parameters():
            if p.requires_grad:
                (decay if p.dim() > 1 else no_decay).append(p)
        params = [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.opt = optim.Adam(params, lr=self.tc.lr_init, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.tc.epochs, eta_min=self.tc.lr_min)

        self.replay = ReplayBuffer(self.tc.buffer_size)
        lf = os.path.join(cfg.LOG_DIR, f"train_{datetime.now():%Y%m%d_%H%M%S}.log")
        self.logger = setup_logger("trainer", lf)

        self.policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.value_loss_fn = nn.MSELoss()

    # ────────────────────────────────────────────────────────────────────
    def train(self, game_data: List[Tuple], epochs: int | None = None, batch_size: int | None = None, ckpt_interval: int = 10):
        if epochs is None:
            epochs = self.tc.epochs
        if batch_size is None:
            batch_size = self.tc.batch_size

        self.replay.add(game_data)
        if len(self.replay) < batch_size:
            self.logger.warning("Insufficient replay data: %d < %d; skipping training", len(self.replay), batch_size)
            return

        dataset = AlphaZeroDataset(self.replay.sample(len(self.replay)), self.cfg)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.cfg.config['system']['num_workers'], pin_memory=False)

        best_loss = math.inf
        self.model.train()
        start = time.time()

        for epoch in range(1, epochs + 1):
            ploss_sum = vloss_sum = tot_loss_sum = 0.0
            num_batches = 0

            for s, pi_target, z in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
                s = s.to(self.model.device)
                pi_target = pi_target.to(self.model.device)
                z = z.to(self.model.device)

                self.opt.zero_grad(set_to_none=True)
                pi_logits, v_pred = self.model.model(s)

                # Policy loss (cross‑entropy with visit count distribution)
                log_pi = torch.log_softmax(pi_logits, dim=1)
                policy_loss = -(pi_target * log_pi).sum(dim=1).mean()

                value_loss = self.value_loss_fn(v_pred.view(-1), z)
                loss = policy_loss + value_loss

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.model.parameters(), self.tc.grad_clip)
                self.opt.step()

                ploss_sum += policy_loss.item()
                vloss_sum += value_loss.item()
                tot_loss_sum += loss.item()
                num_batches += 1

            self.scheduler.step()
            avg_pl, avg_vl, avg_tl = [x / num_batches for x in (ploss_sum, vloss_sum, tot_loss_sum)]
            lr_now = self.opt.param_groups[0]['lr']
            self.logger.info("Epoch %d| π %.4f | v %.4f | Σ %.4f | lr %.6f | %.1fs", epoch, avg_pl, avg_vl, avg_tl, lr_now, time.time() - start)

            # Checkpointing
            if epoch % ckpt_interval == 0 or epoch == epochs:
                ckpt = os.path.join(self.cfg.CHECKPOINT_DIR, f"checkpoint_{int(time.time())}.pt")
                self.model.save_model(ckpt)

            if avg_tl < best_loss:
                best_loss = avg_tl
                self.model.save_model(os.path.join(self.cfg.CHECKPOINT_DIR, "best.pt"))

        self.logger.info("Training finished | epochs %d | best loss %.4f | %.1fs", epochs, best_loss, time.time() - start)
