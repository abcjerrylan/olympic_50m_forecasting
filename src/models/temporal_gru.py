from __future__ import annotations

import torch
from torch import nn


class TemporalGRU(nn.Module):
    def __init__(self, input_size: int = 64, hidden_size: int = 48):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)

    def forward(self, edition_sequence: torch.Tensor) -> torch.Tensor:
        if edition_sequence.ndim != 2 or edition_sequence.shape[0] == 0:
            raise ValueError("edition_sequence must have shape [editions, 64] and cannot be empty")
        _, hidden = self.gru(edition_sequence.unsqueeze(0))
        return hidden[-1, 0]

