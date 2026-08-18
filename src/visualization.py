from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _finish(path: Path, title: str, invert_y: bool = False):
    plt.title(title)
    plt.xlabel("Competition year")
    plt.ylabel("Time (seconds)")
    if invert_y:
        plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_world_distributions(world: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = world[world["valid_time"]]
    for sex in ("Male", "Female"):
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=valid[valid["Sex"] == sex], x="Year", y="Time_seconds", hue="phase_main")
        plt.xticks(rotation=45)
        _finish(output_dir / f"world_distributions_{sex.lower()}.png", f"{sex} World Championships distributions")


def plot_prediction_outputs(predictions: pd.DataFrame, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = predictions.dropna(subset=["actual_time_seconds"]).copy()
    if scored.empty:
        return
    for sex in ("Male", "Female"):
        subset = scored[scored["sex"] == sex]
        for phase in ("HEATS", "SEMIFINALS", "FINALS"):
            phase_data = subset[subset["target_group"] == phase]
            plt.figure(figsize=(10, 6))
            for rank, rank_data in phase_data.groupby("rank"):
                plt.plot(rank_data["target_year"], rank_data["actual_time_seconds"], marker="o", markersize=4, alpha=0.45, color="black")
                plt.plot(rank_data["target_year"], rank_data["predicted_time_seconds"], marker="x", markersize=5, alpha=0.55, color="tab:blue")
            years = sorted(phase_data["target_year"].unique())
            if len(years) == 1:
                plt.xlim(years[0] - 1, years[0] + 1)
                plt.xticks(years)
            _finish(output_dir / f"actual_vs_predicted_{sex.lower()}_{phase.lower()}.png", f"{sex} {phase}: actual (black) vs predicted (blue)")
        plt.figure(figsize=(7, 7))
        plt.scatter(subset["actual_time_seconds"], subset["predicted_time_seconds"], alpha=0.7)
        low = min(subset["actual_time_seconds"].min(), subset["predicted_time_seconds"].min())
        high = max(subset["actual_time_seconds"].max(), subset["predicted_time_seconds"].max())
        plt.plot([low, high], [low, high], "k--")
        plt.xlabel("Actual time (seconds)")
        plt.ylabel("Predicted time (seconds)")
        plt.title(f"{sex}: predicted versus actual")
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_{sex.lower()}.png", dpi=160)
        plt.close()

        cutoffs = subset[((subset["target_group"] == "HEATS") & (subset["rank"] == 16)) |
                         ((subset["target_group"] == "SEMIFINALS") & (subset["rank"] == 8)) |
                         ((subset["target_group"] == "FINALS") & (subset["rank"] == 3))]
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=cutoffs, x="target_year", y="actual_time_seconds", hue="target_group", marker="o")
        sns.lineplot(data=cutoffs, x="target_year", y="predicted_time_seconds", hue="target_group", marker="x", legend=False, linestyle="--")
        cutoff_years = sorted(cutoffs["target_year"].unique())
        if len(cutoff_years) == 1:
            plt.xlim(cutoff_years[0] - 1, cutoff_years[0] + 1)
            plt.xticks(cutoff_years)
        _finish(output_dir / f"cutoff_trends_{sex.lower()}.png", f"{sex}: cutoff trends")

    scored["absolute_error"] = (scored["predicted_time_seconds"] - scored["actual_time_seconds"]).abs()
    mae = scored.groupby(["target_year", "sex"], as_index=False)["absolute_error"].mean()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=mae, x="target_year", y="absolute_error", hue="sex", marker="o")
    mae_years = sorted(mae["target_year"].unique())
    if len(mae_years) == 1:
        plt.xlim(mae_years[0] - 1, mae_years[0] + 1)
        plt.xticks(mae_years)
    plt.title("Per-Olympics MAE")
    plt.xlabel("Competition year")
    plt.ylabel("MAE (seconds)")
    plt.tight_layout()
    plt.savefig(output_dir / "mae_over_time.png", dpi=160)
    plt.close()
