from __future__ import annotations

import torch
from torch import nn


class QueryDecoder(nn.Module):
    def __init__(self, query_dim: int = 12, dropout: float = 0.10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(48 + query_dim, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, context: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        expanded = context.unsqueeze(0).expand(queries.shape[0], -1)
        return self.network(torch.cat([expanded, queries], dim=-1)).squeeze(-1)

