from __future__ import annotations

import torch
from torch import nn


def masked_statistics(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return concatenated mean/max/std while excluding padded tokens."""
    if values.ndim != 2:
        raise ValueError(f"Expected [tokens, features], received {tuple(values.shape)}")
    if values.shape[0] == 0:
        return values.new_zeros(values.shape[1] * 3)
    if mask is None:
        selected = values
    else:
        selected = values[mask.to(dtype=torch.bool)]
    if selected.shape[0] == 0:
        return values.new_zeros(values.shape[1] * 3)
    mean = selected.mean(dim=0)
    maximum = selected.max(dim=0).values
    std = selected.std(dim=0, unbiased=False)
    return torch.cat([mean, maximum, std], dim=-1)


class StageDeepSetEncoder(nn.Module):
    def __init__(self, token_dim: int = 5, dropout: float = 0.10, pooling: str = "mean_max_std"):
        super().__init__()
        self.pooling = pooling
        self.token_mlp = nn.Sequential(
            nn.Linear(token_dim, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU()
        )
        pooled_dim = 32 if pooling == "mean" else 96
        self.compress = nn.Sequential(nn.Linear(pooled_dim, 48), nn.GELU(), nn.Dropout(dropout))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.shape[0] == 0:
            pooled_dim = 32 if self.pooling == "mean" else 96
            pooled = tokens.new_zeros(pooled_dim)
        else:
            encoded = self.token_mlp(tokens)
            if self.pooling == "mean":
                selected = encoded if mask is None else encoded[mask.to(dtype=torch.bool)]
                pooled = selected.mean(dim=0) if len(selected) else encoded.new_zeros(32)
            else:
                pooled = masked_statistics(encoded, mask)
        return self.compress(pooled)

