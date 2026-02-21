from __future__ import annotations

"""
Core replay-buffer and gradient-step utilities for self-play training.

External library usage:
- `collections.deque`: fixed-size replay memory.
- `random.sample`: uniform sampling from replay buffer.
- `numpy`: batch assembly from stored samples.
- `torch` + `torch.nn.functional`: forward pass and losses.
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F


class ReplayBuffer:
    """Stores `(state, policy_target, value_target)` tuples from self-play."""
    def __init__(self, maxlen: int = 50000):
        self.buf = deque(maxlen=maxlen)

    def add_many(self, samples):
        self.buf.extend(samples)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        states = np.stack([b[0] for b in batch], axis=0)
        pis = np.stack([b[1] for b in batch], axis=0)
        vals = np.array([b[2] for b in batch], dtype=np.float32)
        return states, pis, vals

    def __len__(self):
        return len(self.buf)


def train_step(net, opt, states, target_pi, target_v, device="cpu"):
    """
    One optimization step on a sampled replay batch.

    Loss = policy cross-entropy (against MCTS policy target) + value MSE.
    """
    net.train()
    x = torch.from_numpy(states).to(device)
    pi_t = torch.from_numpy(target_pi).to(device)
    v_t = torch.from_numpy(target_v).to(device)

    logits, v = net(x)

    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(pi_t * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(v, v_t)

    loss = policy_loss + value_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())
