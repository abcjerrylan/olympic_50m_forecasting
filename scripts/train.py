from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import default_data_paths, load_results_workbook
from src.models import OlympicForecastModel
from src.preprocessing import assert_no_identity_features, fit_fold_scaler, make_training_examples, tensor_feature_names
from src.train_utils import train_model


def parse_args():
    default_world, _ = default_data_paths(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.yaml")
    parser.add_argument("--world-championships", type=Path, default=default_world)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "runs" / "main")
    parser.add_argument("--validation-year", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--no-gru", action="store_true")
    parser.add_argument("--mean-pooling", action="store_true")
    parser.add_argument("--no-count-features", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.patience is not None:
        config["early_stopping_patience"] = args.patience
    validation_year = args.validation_year or int(config["validation_year"])
    config["validation_year"] = validation_year
    world, quality = load_results_workbook(
        args.world_championships, "world_championships",
        plausible_min=float(config["plausible_time_min"]),
        plausible_max=float(config["plausible_time_max"]),
        recover_time_from_raw=bool(config.get("recover_time_from_raw", False)),
    )
    assert_no_identity_features(tensor_feature_names())
    all_years = sorted(int(year) for year in world["Year"].unique())
    training_years = [year for year in all_years if year < validation_year]
    if validation_year not in all_years:
        raise ValueError(f"Validation year {validation_year} is not in the World Championships data")
    scaler = fit_fold_scaler(world, training_years)
    train_examples = make_training_examples(
        world, scaler, training_years, int(config["min_context_editions"]), config.get("max_context_editions")
    )
    validation_examples = make_training_examples(
        world, scaler, [validation_year], int(config["min_context_editions"]), config.get("max_context_editions")
    )
    model = OlympicForecastModel(
        dropout=float(config["dropout"]), use_gru=not args.no_gru,
        pooling="mean" if args.mean_pooling else "mean_max_std",
        use_count_features=not args.no_count_features,
    )
    count = model.parameter_count()
    if not 20_000 <= count <= 80_000:
        raise AssertionError(f"Model parameter count {count:,} is outside [20,000, 80,000]")
    print(f"Trainable parameters: {count:,}")
    print(f"Training examples: {len(train_examples)}; validation examples: {len(validation_examples)}")
    result = train_model(model, train_examples, validation_examples, scaler, config, args.output_dir)
    with (args.output_dir / "data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump(quality, handle, indent=2)
    print(f"Best epoch: {result['best_epoch']}; best score: {result['best_score']:.6f}")
    print(f"Checkpoint: {(args.output_dir / 'best_model.pt').resolve()}")


if __name__ == "__main__":
    main()

