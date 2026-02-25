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


def train_step_with_feedback(
    net,
    opt,
    states,
    target_pi,
    target_v,
    fb_states,
    fb_good_idx,
    fb_bad_idx,
    fb_weights,
    device="cpu",
    feedback_weight: float = 0.2,
    feedback_margin: float = 0.2,
):
    """
    One optimization step with:
    - Replay policy CE + value MSE
    - Optional preference ranking loss on feedback pairs
    """
    net.train()

    x = torch.from_numpy(states).to(device)
    pi_t = torch.from_numpy(target_pi).to(device)
    v_t = torch.from_numpy(target_v).to(device)

    logits, v = net(x)
    logp = F.log_softmax(logits, dim=1)
    policy_loss = -(pi_t * logp).sum(dim=1).mean()
    value_loss = F.mse_loss(v, v_t)
    base_loss = policy_loss + value_loss

    feedback_loss = torch.tensor(0.0, device=device)
    if fb_states is not None and len(fb_states) > 0 and float(feedback_weight) > 0.0:
        fx = torch.from_numpy(fb_states).to(device)
        good_idx = torch.from_numpy(fb_good_idx).to(device)
        bad_idx = torch.from_numpy(fb_bad_idx).to(device)
        w = torch.from_numpy(fb_weights).to(device)

        fb_logits, _ = net(fx)
        good_logit = fb_logits.gather(1, good_idx.view(-1, 1)).squeeze(1)
        bad_logit = fb_logits.gather(1, bad_idx.view(-1, 1)).squeeze(1)

        # Encourage good move logit > bad move logit by a margin.
        margin_term = float(feedback_margin) - (good_logit - bad_logit)
        per_sample = F.relu(margin_term) * w
        feedback_loss = per_sample.mean()

    loss = base_loss + float(feedback_weight) * feedback_loss

    opt.zero_grad()
    loss.backward()
    opt.step()

    return (
        float(loss.item()),
        float(policy_loss.item()),
        float(value_loss.item()),
        float(feedback_loss.item()),
    )
