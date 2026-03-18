import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x, update):
        return self.norm(x + update)


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_ff)
        self.out_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.out_proj(F.relu(self.in_proj(x)))


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div)
        pe[:, 1::2] = torch.cos(positions * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds positional encoding limit {self.pe.size(1)}."
            )
        return x + self.pe[:, :x.size(1), :]
