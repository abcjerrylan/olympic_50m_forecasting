from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression

from .datasets import tensorize_context, tensorize_example
from .losses import weighted_huber_order_loss
from .preprocessing import PHASES, PHASE_TO_ID, SEX_TO_ID, FoldScaler, build_context


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _example_loss(model, example: dict[str, Any], config: dict[str, Any]):
    queries = example["queries"]
    predictions = model(
        example["context"], queries["phase_rank"], example["target_year_normalized"],
        example["target_year_gap"], example["sex_id"],
    )
    return weighted_huber_order_loss(
        predictions, queries["labels"], queries["weights"], queries["group_ids"], queries["ranks"],
        lambda_order=float(config.get("lambda_order", 2.0)),
        delta=float(config.get("huber_delta", 1.0)),
    )


def evaluate_examples(model, examples, device, config) -> float:
    if not examples:
        return float("nan")
    model.eval()
    losses = []
    with torch.no_grad():
        for raw in examples:
            example = tensorize_example(raw, device)
            loss, _ = _example_loss(model, example, config)
            losses.append(float(loss))
    return float(np.mean(losses))


def train_model(
    model: torch.nn.Module,
    train_examples: list[dict[str, Any]],
    validation_examples: list[dict[str, Any]],
    scaler: FoldScaler,
    config: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    if not train_examples:
        raise ValueError("No training examples were constructed")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 42))
    set_deterministic_seed(seed)
    device = resolve_device(str(config.get("device", "auto")))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    max_epochs = int(config.get("max_epochs", 500))
    patience = int(config.get("early_stopping_patience", 40))
    clip = float(config.get("gradient_clip_norm", 1.0))
    rng = random.Random(seed)
    best_score = float("inf")
    best_epoch = 0
    history: list[dict[str, float]] = []
    remaining_patience = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        ordered = list(train_examples)
        rng.shuffle(ordered)
        epoch_losses = []
        for raw in ordered:
            example = tensorize_example(raw, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _example_loss(model, example, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        train_loss = float(np.mean(epoch_losses))
        validation_loss = evaluate_examples(model, validation_examples, device, config)
        score = validation_loss if np.isfinite(validation_loss) else train_loss
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch
            remaining_patience = patience
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler": scaler.to_dict(),
                "config": config,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "training_targets": sorted({int(e["target_year"]) for e in train_examples}),
                "validation_targets": sorted({int(e["target_year"]) for e in validation_examples}),
            }, output_dir / "best_model.pt")
        else:
            remaining_patience -= 1
        if validation_examples and remaining_patience <= 0:
            break

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "best_epoch": best_epoch, "best_score": best_score, "epochs_ran": len(history),
            "parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "device": str(device), "seed": seed,
        }, handle, indent=2)
    checkpoint = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return {**checkpoint, "history": history, "device": str(device)}


def rank_queries() -> np.ndarray:
    parts = []
    for phase, count in (("HEATS", 16), ("SEMIFINALS", 8), ("FINALS", 3)):
        parts.append(np.column_stack([
            np.full(count, PHASE_TO_ID[phase], dtype=np.float32),
            np.arange(1, count + 1, dtype=np.float32),
        ]))
    return np.concatenate(parts).astype(np.float32)


def predict_olympics(
    model,
    world: pd.DataFrame,
    scaler: FoldScaler,
    target_year: int,
    device: torch.device,
    max_context_editions: int | None = None,
) -> pd.DataFrame:
    model.eval()
    query_array = rank_queries()
    rows = []
    for sex, sex_id in SEX_TO_ID.items():
        context_raw = build_context(world, sex, target_year, scaler, max_context_editions)
        if len(context_raw) < 3:
            raise ValueError(f"{target_year} {sex}: fewer than three context editions")
        if context_raw[-1]["year"] >= target_year:
            raise AssertionError("Future leakage detected during prediction")
        context = tensorize_context(context_raw, device)
        queries = torch.as_tensor(query_array, dtype=torch.float32, device=device)
        with torch.no_grad():
            normalized = model(
                context, queries, (target_year - scaler.base_year) / 10.0,
                (target_year - context_raw[-1]["year"]) / 10.0, sex_id,
            ).cpu().numpy()
        offset = 0
        for phase, count in (("HEATS", 16), ("SEMIFINALS", 8), ("FINALS", 3)):
            raw = np.asarray(scaler.inverse_time(normalized[offset:offset + count], sex, phase), dtype=float)
            ranks = np.arange(1, count + 1)
            projected = IsotonicRegression(increasing=True).fit_transform(ranks, raw)
            for rank, raw_value, projected_value in zip(ranks, raw, projected):
                rows.append({
                    "target_year": target_year, "sex": sex, "target_group": phase,
                    "rank": int(rank), "raw_predicted_time_seconds": float(raw_value),
                    "predicted_time_seconds": float(projected_value),
                    "training_year_max": int(max(scaler.training_years)),
                })
            offset += count
    return pd.DataFrame(rows)


def latest_world_baseline(world: pd.DataFrame, target_year: int) -> pd.DataFrame:
    eligible = world[(world["Year"] < target_year) & world["valid_time"]]
    latest_year = int(eligible["Year"].max())
    rows = []
    for sex in SEX_TO_ID:
        for phase, count in (("HEATS", 16), ("SEMIFINALS", 8), ("FINALS", 3)):
            values = np.sort(eligible[(eligible["Year"] == latest_year) & (eligible["Sex"] == sex) & (eligible["phase_main"] == phase)]["Time_seconds"].to_numpy(dtype=float))
            if len(values) < count:
                raise ValueError(f"Baseline lacks {sex} {phase} rank {count} in {latest_year}")
            for rank in range(1, count + 1):
                rows.append({
                    "target_year": target_year, "sex": sex, "target_group": phase,
                    "rank": rank, "predicted_time_seconds": float(values[rank - 1]),
                    "training_year_max": latest_year,
                })
    return pd.DataFrame(rows)

