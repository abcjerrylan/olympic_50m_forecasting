from __future__ import annotations

from typing import Any

import numpy as np
import torch


def tensorize_context(context: list[dict[str, Any]], device: torch.device) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for edition in context:
        output.append({
            "year": edition["year"],
            "sex_id": torch.tensor(edition["sex_id"], dtype=torch.long, device=device),
            "stages": {
                phase: torch.as_tensor(tokens, dtype=torch.float32, device=device)
                for phase, tokens in edition["stages"].items()
            },
            "presence": torch.as_tensor(edition["presence"], dtype=torch.float32, device=device),
            "count_features": torch.as_tensor(edition["count_features"], dtype=torch.float32, device=device),
            "year_features": torch.as_tensor(edition["year_features"], dtype=torch.float32, device=device),
        })
    return output


def tensorize_example(example: dict[str, Any], device: torch.device) -> dict[str, Any]:
    queries = example["queries"]
    return {
        **{key: value for key, value in example.items() if key not in {"context", "queries"}},
        "context": tensorize_context(example["context"], device),
        "queries": {
            key: torch.as_tensor(value, dtype=torch.long if key == "group_ids" else torch.float32, device=device)
            for key, value in queries.items()
        },
    }


def latest_top_only(context: list[dict[str, Any]], limits: dict[str, int] | None = None):
    limits = limits or {"HEATS": 16, "SEMIFINALS": 8, "FINALS": 3}
    copied: list[dict[str, Any]] = []
    for edition in context:
        item = {**edition, "stages": {}}
        for phase, tokens in edition["stages"].items():
            if isinstance(tokens, np.ndarray):
                item["stages"][phase] = tokens[: limits[phase]].copy()
            else:
                item["stages"][phase] = tokens[: limits[phase]].clone()
        copied.append(item)
    return copied

