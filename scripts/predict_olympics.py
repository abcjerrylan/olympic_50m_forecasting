from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import default_data_paths, load_results_workbook
from src.models import OlympicForecastModel
from src.preprocessing import FoldScaler
from src.train_utils import predict_olympics, resolve_device


def parse_args():
    default_world, _ = default_data_paths(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--world-championships", type=Path, default=default_world)
    parser.add_argument("--target-year", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    scaler = FoldScaler.from_dict(checkpoint["scaler"])
    config = checkpoint["config"]
    model = OlympicForecastModel(dropout=float(config["dropout"]))
    model.load_state_dict(checkpoint["model_state"])
    device = resolve_device(str(config.get("device", "auto")))
    model.to(device)
    world, _ = load_results_workbook(args.world_championships, "world_championships")
    predictions = predict_olympics(model, world, scaler, args.target_year, device, config.get("max_context_editions"))
    predictions["actual_time_seconds"] = None
    predictions["absolute_error_seconds"] = None
    predictions["signed_error_seconds"] = None
    predictions["is_cutoff"] = (((predictions["target_group"] == "HEATS") & (predictions["rank"] == 16)) |
                                ((predictions["target_group"] == "SEMIFINALS") & (predictions["rank"] == 8)) |
                                ((predictions["target_group"] == "FINALS") & (predictions["rank"].isin([1, 3]))))
    predictions["run_id"] = f"future_{args.target_year}"
    predictions["provisional"] = bool(world.loc[world["Year"] < args.target_year, "Year"].max() < args.target_year - 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)
    print(f"Predictions: {args.output.resolve()}")


if __name__ == "__main__":
    main()

