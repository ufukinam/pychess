from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encode import ACTION_SIZE

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(x + out)


class AlphaZeroNet(nn.Module):
    def __init__(self, in_channels: int = 18, channels: int = 32, num_blocks: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.pol_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.pol_bn = nn.BatchNorm2d(2)
        self.pol_fc = nn.Linear(2 * 8 * 8, ACTION_SIZE)

        # Value head
        self.val_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.val_bn = nn.BatchNorm2d(1)
        self.val_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        x: (B, 18, 8, 8)
        returns:
          policy_logits: (B, ACTION_SIZE)
          value: (B,) in [-1, 1]
        """
        z = self.blocks(self.stem(x))

        p = F.relu(self.pol_bn(self.pol_conv(z)))
        p = p.view(p.size(0), -1)
        policy_logits = self.pol_fc(p)

        v = F.relu(self.val_bn(self.val_conv(z)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.val_fc1(v))
        v = torch.tanh(self.val_fc2(v)).squeeze(1)

        return policy_logits, v
