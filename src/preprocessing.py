from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data_loader import RAW_PHASES


PHASES = ["HEATS", "SEMIFINALS", "FINALS"]
PHASE_TO_ID = {phase: index for index, phase in enumerate(PHASES)}
SEX_TO_ID = {"Male": 0, "Female": 1}
COUNT_TYPES = ["DNS", "DSQ", "DNF", "missing_time", "valid_result"]
IDENTITY_COLUMNS = {"Name", "NAT", "Lane", "Official_event_page", "Official_API_source"}


@dataclass
class FoldScaler:
    base_year: int
    time_stats: dict[str, dict[str, float]]
    max_field_size: dict[str, int]
    training_years: list[int]

    @staticmethod
    def key(sex: str, phase: str) -> str:
        return f"{sex}|{phase}"

    def normalize_time(self, values: np.ndarray | float, sex: str, phase: str):
        stats = self.time_stats[self.key(sex, phase)]
        return (values - stats["mean"]) / stats["std"]

    def inverse_time(self, values: np.ndarray | float, sex: str, phase: str):
        stats = self.time_stats[self.key(sex, phase)]
        return values * stats["std"] + stats["mean"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FoldScaler":
        return cls(**values)


def fit_fold_scaler(world: pd.DataFrame, training_years: Iterable[int]) -> FoldScaler:
    years = sorted({int(year) for year in training_years})
    if not years:
        raise ValueError("training_years cannot be empty")
    training = world[world["Year"].isin(years)].copy()
    if training.empty:
        raise ValueError("No World Championships rows match training_years")
    stats: dict[str, dict[str, float]] = {}
    max_sizes: dict[str, int] = {}
    for sex in SEX_TO_ID:
        for phase in PHASES:
            subset = training[(training["Sex"] == sex) & (training["phase_main"] == phase) & training["valid_time"]]
            if subset.empty:
                raise ValueError(f"No valid training times for {sex} {phase}")
            values = subset["Time_seconds"].to_numpy(dtype=float)
            std = float(np.std(values, ddof=0))
            stats[FoldScaler.key(sex, phase)] = {
                "mean": float(np.mean(values)),
                "std": std if std > 1e-8 else 1.0,
            }
            group_sizes = subset.groupby(["Year", "Sex"]).size()
            max_sizes[FoldScaler.key(sex, phase)] = max(1, int(group_sizes.max()))
    return FoldScaler(min(years), stats, max_sizes, years)


def _competition_ranks(values: pd.Series) -> np.ndarray:
    return values.rank(method="min", ascending=True).to_numpy(dtype=float)


def _count_features(edition: pd.DataFrame) -> np.ndarray:
    features: list[float] = []
    for raw_phase in RAW_PHASES:
        phase_rows = edition[edition["Phase"] == raw_phase]
        for count_type in COUNT_TYPES:
            if count_type == "missing_time":
                value = int(((phase_rows["Status"] == "OK") & phase_rows["Time_seconds"].isna()).sum())
            elif count_type == "valid_result":
                value = int(phase_rows["valid_time"].sum())
            else:
                value = int((phase_rows["Status"] == count_type).sum())
            features.append(float(np.log1p(value) / 6.0))
    return np.asarray(features, dtype=np.float32)


def _stage_tokens(rows: pd.DataFrame, sex: str, phase: str, scaler: FoldScaler) -> np.ndarray:
    valid = rows[(rows["phase_main"] == phase) & rows["valid_time"]].copy()
    if valid.empty:
        return np.zeros((0, 5), dtype=np.float32)
    valid = valid.sort_values(["Time_seconds", "Phase", "Name"], kind="stable")
    ranks = _competition_ranks(valid["Time_seconds"])
    field_size = len(valid)
    rank_norm = (ranks - 1.0) / max(field_size - 1, 1)
    field_norm = np.full(field_size, field_size / scaler.max_field_size[FoldScaler.key(sex, phase)])
    normalized_time = scaler.normalize_time(valid["Time_seconds"].to_numpy(dtype=float), sex, phase)
    return np.column_stack([
        normalized_time,
        rank_norm,
        field_norm,
        valid["subphase_id"].to_numpy(dtype=float),
        valid["is_swim_off"].astype(float).to_numpy(),
    ]).astype(np.float32)


def build_context(
    world: pd.DataFrame,
    sex: str,
    target_year: int,
    scaler: FoldScaler,
    max_context_editions: int | None = None,
) -> list[dict[str, Any]]:
    years = sorted(int(year) for year in world.loc[world["Year"] < target_year, "Year"].unique())
    if any(year >= target_year for year in years):
        raise AssertionError("Future leakage: context contains target or later edition")
    if max_context_editions is not None:
        years = years[-max_context_editions:]
    editions: list[dict[str, Any]] = []
    previous_year: int | None = None
    for year in years:
        rows = world[(world["Year"] == year) & (world["Sex"] == sex)]
        if rows.empty:
            continue
        gap = 0.0 if previous_year is None else (year - previous_year) / 10.0
        stages = {phase: _stage_tokens(rows, sex, phase, scaler) for phase in PHASES}
        editions.append({
            "year": year,
            "sex_id": SEX_TO_ID[sex],
            "stages": stages,
            "presence": np.asarray([float(len(stages[p]) > 0) for p in PHASES], dtype=np.float32),
            "count_features": _count_features(rows),
            "year_features": np.asarray([(year - scaler.base_year) / 10.0, gap], dtype=np.float32),
        })
        previous_year = year
    return editions


def build_queries(target_rows: pd.DataFrame, sex: str, scaler: FoldScaler) -> dict[str, np.ndarray]:
    query_parts: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    group_ids: list[np.ndarray] = []
    raw_times: list[np.ndarray] = []
    ranks_all: list[np.ndarray] = []
    for phase in PHASES:
        rows = target_rows[(target_rows["phase_main"] == phase) & target_rows["valid_time"]].copy()
        if rows.empty:
            continue
        rows = rows.sort_values("Time_seconds", kind="stable")
        ranks = _competition_ranks(rows["Time_seconds"])
        times = rows["Time_seconds"].to_numpy(dtype=float)
        phase_ids = np.full(len(rows), PHASE_TO_ID[phase], dtype=np.float32)
        query_parts.append(np.column_stack([phase_ids, ranks]).astype(np.float32))
        labels.append(np.asarray(scaler.normalize_time(times, sex, phase), dtype=np.float32))
        phase_weights = np.ones(len(rows), dtype=np.float32)
        if phase == "HEATS":
            phase_weights[ranks <= 16] = 2.0
            phase_weights[ranks == 16] += 1.0
        elif phase == "SEMIFINALS":
            phase_weights[ranks <= 8] = 2.5
            phase_weights[ranks == 8] += 1.0
        else:
            phase_weights[ranks <= 3] = 3.0
            phase_weights[ranks == 3] += 1.0
        weights.append(phase_weights)
        group_ids.append(np.full(len(rows), PHASE_TO_ID[phase], dtype=np.int64))
        raw_times.append(times.astype(np.float32))
        ranks_all.append(ranks.astype(np.float32))
    if not query_parts:
        raise ValueError("Target edition has no valid queries")
    return {
        "phase_rank": np.concatenate(query_parts),
        "labels": np.concatenate(labels),
        "weights": np.concatenate(weights),
        "group_ids": np.concatenate(group_ids),
        "raw_times": np.concatenate(raw_times),
        "ranks": np.concatenate(ranks_all),
    }


def make_training_examples(
    world: pd.DataFrame,
    scaler: FoldScaler,
    target_years: Iterable[int],
    min_context_editions: int = 3,
    max_context_editions: int | None = None,
) -> list[dict[str, Any]]:
    if set(world["source"].unique()) != {"world_championships"}:
        raise ValueError("Training examples may contain World Championships rows only")
    examples: list[dict[str, Any]] = []
    for target_year in sorted({int(year) for year in target_years}):
        for sex in SEX_TO_ID:
            target = world[(world["Year"] == target_year) & (world["Sex"] == sex)]
            if target.empty:
                continue
            context = build_context(world, sex, target_year, scaler, max_context_editions)
            if len(context) < min_context_editions:
                continue
            examples.append({
                "target_year": target_year,
                "sex": sex,
                "sex_id": SEX_TO_ID[sex],
                "context": context,
                "queries": build_queries(target, sex, scaler),
                "target_year_normalized": (target_year - scaler.base_year) / 10.0,
                "target_year_gap": (target_year - context[-1]["year"]) / 10.0,
            })
    return examples


def tensor_feature_names() -> list[str]:
    return [
        "normalized_time", "phase_rank_normalized", "field_size_normalized",
        "subphase_id", "is_swim_off",
    ]


def assert_no_identity_features(feature_names: Iterable[str]) -> None:
    overlap = IDENTITY_COLUMNS.intersection(feature_names)
    if overlap:
        raise AssertionError(f"Identity fields are forbidden in model tensors: {sorted(overlap)}")

