from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_huber_order_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    group_ids: torch.Tensor,
    ranks: torch.Tensor,
    lambda_order: float = 2.0,
    delta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    element_loss = F.huber_loss(predictions, targets, reduction="none", delta=delta)
    base = (element_loss * weights).sum() / weights.sum().clamp_min(1.0)
    penalties: list[torch.Tensor] = []
    for group in torch.unique(group_ids):
        selected = group_ids == group
        group_predictions = predictions[selected]
        group_ranks = ranks[selected]
        order = torch.argsort(group_ranks)
        ordered = group_predictions[order]
        if len(ordered) > 1:
            penalties.append(torch.relu(ordered[:-1] - ordered[1:]).mean())
    order_loss = torch.stack(penalties).mean() if penalties else predictions.new_zeros(())
    total = base + lambda_order * order_loss
    return total, {"huber": float(base.detach()), "order": float(order_loss.detach())}

