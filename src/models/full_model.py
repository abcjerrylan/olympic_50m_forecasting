from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .deepsets import StageDeepSetEncoder
from .query_decoder import QueryDecoder
from .temporal_gru import TemporalGRU


PHASES = ["HEATS", "SEMIFINALS", "FINALS"]


class EditionEncoder(nn.Module):
    def __init__(self, dropout: float = 0.10, count_dim: int = 25, sex_dim: int = 4):
        super().__init__()
        input_dim = 48 * 3 + 3 + count_dim + sex_dim + 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, 96), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, 64), nn.GELU(),
        )

    def forward(
        self,
        stage_embeddings: list[torch.Tensor],
        presence: torch.Tensor,
        count_features: torch.Tensor,
        sex_embedding: torch.Tensor,
        year_features: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(stage_embeddings + [presence, count_features, sex_embedding, year_features])
        return self.network(features)


class OlympicForecastModel(nn.Module):
    def __init__(
        self,
        token_dim: int = 5,
        dropout: float = 0.10,
        use_gru: bool = True,
        pooling: str = "mean_max_std",
        use_count_features: bool = True,
    ):
        super().__init__()
        self.use_gru = use_gru
        self.use_count_features = use_count_features
        self.stage_encoders = nn.ModuleDict({
            phase: StageDeepSetEncoder(token_dim, dropout, pooling) for phase in PHASES
        })
        self.sex_embedding = nn.Embedding(2, 4)
        self.phase_embedding = nn.Embedding(3, 4)
        self.edition_encoder = EditionEncoder(dropout)
        self.temporal = TemporalGRU(64, 48)
        self.no_gru_projection = nn.Linear(64, 48)
        self.decoder = QueryDecoder(12, dropout)

    def encode_context(self, editions: list[dict[str, Any]]) -> torch.Tensor:
        encoded_editions: list[torch.Tensor] = []
        for edition in editions:
            stage_embeddings = [
                self.stage_encoders[phase](edition["stages"][phase]) for phase in PHASES
            ]
            sex = self.sex_embedding(edition["sex_id"].reshape(()))
            counts = edition["count_features"]
            if not self.use_count_features:
                counts = torch.zeros_like(counts)
            encoded_editions.append(self.edition_encoder(
                stage_embeddings, edition["presence"], counts, sex, edition["year_features"]
            ))
        sequence = torch.stack(encoded_editions)
        if self.use_gru:
            return self.temporal(sequence)
        return self.no_gru_projection(sequence[-1])

    def forward(
        self,
        editions: list[dict[str, Any]],
        phase_rank: torch.Tensor,
        target_year_normalized: float | torch.Tensor,
        target_year_gap: float | torch.Tensor,
        sex_id: int | torch.Tensor,
    ) -> torch.Tensor:
        context = self.encode_context(editions)
        device = phase_rank.device
        phase_ids = phase_rank[:, 0].long()
        ranks = phase_rank[:, 1]
        phase_embed = self.phase_embedding(phase_ids)
        sex_tensor = torch.as_tensor(sex_id, dtype=torch.long, device=device).reshape(())
        sex_embed = self.sex_embedding(sex_tensor).unsqueeze(0).expand(len(phase_rank), -1)
        target_year = torch.as_tensor(target_year_normalized, dtype=torch.float32, device=device)
        year_gap = torch.as_tensor(target_year_gap, dtype=torch.float32, device=device)
        query = torch.cat([
            target_year.expand(len(phase_rank), 1),
            year_gap.expand(len(phase_rank), 1),
            sex_embed,
            phase_embed,
            (ranks / 100.0).unsqueeze(1),
            (torch.log1p(ranks) / 5.0).unsqueeze(1),
        ], dim=1)
        return self.decoder(context, query)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

