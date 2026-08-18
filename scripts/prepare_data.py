from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import default_data_paths, load_results_workbook
from src.target_builder import build_olympic_targets


def parse_args():
    default_world, default_olympics = default_data_paths(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-championships", type=Path, default=default_world)
    parser.add_argument("--olympics", type=Path, default=default_olympics)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--recover-time-from-raw", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    world, world_quality = load_results_workbook(
        args.world_championships, "world_championships", recover_time_from_raw=args.recover_time_from_raw
    )
    olympics, olympic_quality = load_results_workbook(
        args.olympics, "olympics", recover_time_from_raw=args.recover_time_from_raw
    )
    targets, target_quality = build_olympic_targets(olympics)
    world.to_pickle(args.output_dir / "world_championships.pkl")
    olympics.to_pickle(args.output_dir / "olympics.pkl")
    targets.to_csv(args.output_dir / "olympic_targets.csv", index=False)
    target_quality.to_csv(args.output_dir / "olympic_target_quality.csv", index=False)
    with (args.output_dir / "data_quality.json").open("w", encoding="utf-8") as handle:
        json.dump({"world_championships": world_quality, "olympics": olympic_quality}, handle, indent=2)
    print(f"World rows: {len(world):,}; valid times: {int(world['valid_time'].sum()):,}")
    print(f"Olympic rows: {len(olympics):,}; targets: {len(targets):,}")
    print(f"Processed data: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

