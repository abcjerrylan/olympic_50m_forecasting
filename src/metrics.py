from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = np.asarray(predicted) - np.asarray(actual)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "bias": float(np.mean(errors)),
    }


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in predictions.dropna(subset=["actual_time_seconds"]).groupby(["target_year", "sex", "target_group"], dropna=False):
        values = regression_metrics(group["actual_time_seconds"].to_numpy(), group["predicted_time_seconds"].to_numpy())
        rows.append({"target_year": keys[0], "sex": keys[1], "target_group": keys[2], **values})
    return pd.DataFrame(rows)


def required_metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.dropna(subset=["actual_time_seconds"]).copy()
    scopes = [(sex, scored[scored["sex"] == sex]) for sex in sorted(scored["sex"].unique())]
    scopes.append(("Combined", scored))
    rows = []
    for scope, group in scopes:
        row: dict[str, float | str] = {"scope": scope}
        for phase, label in (("HEATS", "Heats16"), ("SEMIFINALS", "Semifinals8"), ("FINALS", "Finals3")):
            phase_rows = group[group["target_group"] == phase]
            values = regression_metrics(
                phase_rows["actual_time_seconds"].to_numpy(), phase_rows["predicted_time_seconds"].to_numpy()
            )
            row[f"MAE_{label}"] = values["mae"]
            row[f"RMSE_{label}"] = values["rmse"]
            row[f"Bias_{label}"] = values["bias"]
        overall = regression_metrics(group["actual_time_seconds"].to_numpy(), group["predicted_time_seconds"].to_numpy())
        row["MAE_All27"] = overall["mae"]
        row["RMSE_All27"] = overall["rmse"]
        row["Bias_All27"] = overall["bias"]
        rows.append(row)
    return pd.DataFrame(rows)


def cutoff_metric_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.dropna(subset=["actual_time_seconds"]).copy()
    scopes = [(sex, scored[scored["sex"] == sex]) for sex in sorted(scored["sex"].unique())]
    scopes.append(("Combined", scored))
    cutoffs = {
        "Heats_rank16_absolute_error": ("HEATS", 16),
        "Semifinals_rank8_absolute_error": ("SEMIFINALS", 8),
        "Finals_rank3_absolute_error": ("FINALS", 3),
        "Finals_rank1_absolute_error": ("FINALS", 1),
    }
    rows = []
    for scope, group in scopes:
        row: dict[str, float | str] = {"scope": scope}
        for name, (phase, rank) in cutoffs.items():
            selected = group[(group["target_group"] == phase) & (group["rank"] == rank)]
            row[name] = float(
                (selected["predicted_time_seconds"] - selected["actual_time_seconds"]).abs().mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)
