from typing import Callable

import numpy as np
import torch
import torch.nn as nn


# We adjust the Discriminator as simple as possible

class DiscHead(nn.Module):
    def __init__(self, channels: int, c_dim: int, cmap_dim: int = 3):
        super().__init__()
        self.channels = channels
        self.c_dim = c_dim
        self.cmap_dim = cmap_dim



        self.cls = nn.Sequential(
            nn.Linear(channels, cmap_dim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(cmap_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        out = self.cls(x)
        return out


# Make the Discriminator as simple as possible
class ProjectedDiscriminator(nn.Module):
    def __init__(self, c_dim: int):
        super().__init__()
        self.c_dim = c_dim
        self.head = DiscHead(c_dim, c_dim)

    def train(self, mode: bool = True):
        self.head.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x: torch.Tensor, c=None,side='D'):
        # Apply discriminator heads.
        """
        x: image features from generator
        c: text features from clip
        """
        if side=='D':
            out1 = self.head(x)
            out2 = self.head(c)
            result = torch.cat([out1, out2], dim=0)
            labels = torch.cat([torch.ones_like(out1), torch.zeros_like(out2)], dim=0)
            return torch.nn.functional.binary_cross_entropy_with_logits(result, labels)
        elif side=='G':
            out1 = self.head(x)
            labels =torch.zeros_like(out1)
            return torch.nn.functional.binary_cross_entropy_with_logits(out1, labels)

