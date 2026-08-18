from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import default_data_paths, load_results_workbook
from src.metrics import cutoff_metric_summary, required_metric_summary, summarize_predictions
from src.models import OlympicForecastModel
from src.preprocessing import fit_fold_scaler, make_training_examples
from src.target_builder import build_olympic_targets
from src.train_utils import latest_world_baseline, predict_olympics, resolve_device, train_model
from src.visualization import plot_prediction_outputs, plot_world_distributions


def parse_args():
    default_world, default_olympics = default_data_paths(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--world-championships", type=Path, default=default_world)
    parser.add_argument("--olympics", type=Path, default=default_olympics)
    parser.add_argument("--years", type=int, nargs="+", default=[2000, 2004, 2008, 2012, 2016, 2020, 2024])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "runs" / "walk_forward")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config["early_stopping_patience"] = args.patience
    world, _ = load_results_workbook(args.world_championships, "world_championships")
    olympics, _ = load_results_workbook(args.olympics, "olympics")
    targets, target_quality = build_olympic_targets(olympics)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(str(config.get("device", "auto")))
    prediction_frames = []
    baseline_frames = []

    for target_year in args.years:
        available = sorted(int(y) for y in world.loc[world["Year"] < target_year, "Year"].unique())
        if len(available) < 4:
            raise ValueError(f"{target_year}: insufficient World Championships history")
        if len(available) == 4:
            scaler_years = available
            train_target_years = available
            validation_target_years = []
        else:
            validation_year = available[-1]
            scaler_years = available[:-1]
            train_target_years = available[:-1]
            validation_target_years = [validation_year]
        scaler = fit_fold_scaler(world, scaler_years)
        train_examples = make_training_examples(
            world, scaler, train_target_years, int(config["min_context_editions"]), config.get("max_context_editions")
        )
        validation_examples = make_training_examples(
            world, scaler, validation_target_years, int(config["min_context_editions"]), config.get("max_context_editions")
        )
        model = OlympicForecastModel(dropout=float(config["dropout"]))
        fold_dir = args.output_dir / str(target_year)
        train_model(model, train_examples, validation_examples, scaler, config, fold_dir)
        model.to(device)
        prediction = predict_olympics(model, world, scaler, target_year, device, config.get("max_context_editions"))
        actual = targets[targets["target_year"] == target_year]
        prediction = prediction.merge(actual, on=["target_year", "sex", "target_group", "rank"], how="left")
        prediction["absolute_error_seconds"] = (prediction["predicted_time_seconds"] - prediction["actual_time_seconds"]).abs()
        prediction["signed_error_seconds"] = prediction["predicted_time_seconds"] - prediction["actual_time_seconds"]
        prediction["is_cutoff"] = (((prediction["target_group"] == "HEATS") & (prediction["rank"] == 16)) |
                                   ((prediction["target_group"] == "SEMIFINALS") & (prediction["rank"] == 8)) |
                                   ((prediction["target_group"] == "FINALS") & (prediction["rank"].isin([1, 3]))))
        prediction["run_id"] = f"walk_forward_{target_year}"
        prediction_frames.append(prediction)

        baseline = latest_world_baseline(world, target_year).merge(
            actual, on=["target_year", "sex", "target_group", "rank"], how="left"
        )
        baseline_frames.append(baseline)
        print(f"Completed Olympic fold {target_year}; context through {available[-1]}")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    baselines = pd.concat(baseline_frames, ignore_index=True)
    predictions.to_csv(args.output_dir / "olympic_predictions.csv", index=False)
    baselines.to_csv(args.output_dir / "latest_world_baseline.csv", index=False)
    summarize_predictions(predictions).to_csv(args.output_dir / "metrics.csv", index=False)
    summarize_predictions(baselines).to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    required_metric_summary(predictions).to_csv(args.output_dir / "required_metrics.csv", index=False)
    required_metric_summary(baselines).to_csv(args.output_dir / "baseline_required_metrics.csv", index=False)
    cutoff_metric_summary(predictions).to_csv(args.output_dir / "cutoff_metrics.csv", index=False)
    target_quality[target_quality["target_year"].isin(args.years)].to_csv(args.output_dir / "target_quality.csv", index=False)
    plot_world_distributions(world, args.output_dir / "figures")
    plot_prediction_outputs(predictions, args.output_dir / "figures")
    with (args.output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"years": args.years, "rows": len(predictions)}, handle, indent=2)
    print(f"Evaluation outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
