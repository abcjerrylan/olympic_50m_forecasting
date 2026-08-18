from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "Year", "Edition", "Name", "Sex", "Phase", "Time_seconds",
    "Time_raw", "Status", "NAT", "Heat", "Lane", "Rank",
    "Official_event_page", "Official_API_source",
]

RAW_PHASES = [
    "Heats", "Heat Swim-off", "Semifinals", "Semifinal Swim-off", "Finals"
]

PHASE_MAIN = {
    "Heats": "HEATS",
    "Heat Swim-off": "HEATS",
    "Semifinals": "SEMIFINALS",
    "Semifinal Swim-off": "SEMIFINALS",
    "Finals": "FINALS",
}


def _parse_time_raw(value: Any) -> float:
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip()
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return float(minutes) * 60.0 + float(seconds)
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def load_results_workbook(
    path: str | Path,
    source: str,
    plausible_min: float = 18.0,
    plausible_max: float = 90.0,
    recover_time_from_raw: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and validate one source workbook without modifying it."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    sheets = pd.ExcelFile(path).sheet_names
    if "All_Data" not in sheets:
        raise ValueError(f"{path.name}: required worksheet All_Data is missing")
    frame = pd.read_excel(path, sheet_name="All_Data")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    frame = frame[REQUIRED_COLUMNS].copy()
    frame["Year"] = pd.to_numeric(frame["Year"], errors="raise").astype(int)
    frame["Sex"] = frame["Sex"].astype(str).str.strip()
    frame["Phase"] = frame["Phase"].astype(str).str.strip()
    frame["Status"] = frame["Status"].astype(str).str.strip().str.upper()
    if not set(frame["Sex"]).issubset({"Male", "Female"}):
        raise ValueError(f"{path.name}: unsupported Sex values {sorted(set(frame['Sex']))}")
    unknown_phases = sorted(set(frame["Phase"]) - set(RAW_PHASES))
    if unknown_phases:
        raise ValueError(f"{path.name}: unsupported Phase values {unknown_phases}")

    frame["Time_seconds"] = pd.to_numeric(frame["Time_seconds"], errors="coerce")
    source_missing_time = frame["Time_seconds"].isna()
    recovered_count = 0
    if recover_time_from_raw:
        recovered = frame.loc[source_missing_time, "Time_raw"].map(_parse_time_raw)
        valid_recovery = recovered.notna()
        recovered_count = int(valid_recovery.sum())
        frame.loc[recovered.index[valid_recovery], "Time_seconds"] = recovered[valid_recovery]

    finite = np.isfinite(frame["Time_seconds"].fillna(np.nan).to_numpy(dtype=float))
    plausible = frame["Time_seconds"].between(plausible_min, plausible_max, inclusive="both")
    frame["valid_time"] = (frame["Status"] == "OK") & finite & plausible
    frame["phase_main"] = frame["Phase"].map(PHASE_MAIN)
    frame["is_swim_off"] = frame["Phase"].str.contains("Swim-off", regex=False)
    frame["subphase_id"] = frame["is_swim_off"].astype(int)
    frame["source"] = source

    duplicate_key = ["Year", "Sex", "Phase", "Name", "Time_raw", "Heat"]
    duplicate_mask = frame.duplicated(duplicate_key, keep="first")
    duplicates = frame.loc[duplicate_mask, duplicate_key].copy()
    if duplicate_mask.any():
        frame = frame.loc[~duplicate_mask].copy()

    quality = {
        "path": str(path),
        "source": source,
        "rows_loaded": int(len(frame) + duplicate_mask.sum()),
        "rows_retained": int(len(frame)),
        "duplicate_rows_removed": int(duplicate_mask.sum()),
        "source_missing_time_count": int(source_missing_time.sum()),
        "recovered_time_count": recovered_count,
        "valid_time_count": int(frame["valid_time"].sum()),
        "status_counts": {str(k): int(v) for k, v in frame["Status"].value_counts().items()},
        "years": sorted(int(year) for year in frame["Year"].unique()),
        "duplicates": duplicates.to_dict(orient="records"),
    }
    return frame.reset_index(drop=True), quality


def default_data_paths(project_root: str | Path) -> tuple[Path, Path]:
    project_root = Path(project_root).resolve()
    data_dir = project_root / "data"
    world = data_dir / "World_Aquatics_Championships_50m_Freestyle_Complete_Results_1986-2025.xlsx"
    olympics = data_dir / "Olympic_Games_50m_Freestyle_Complete_Results_1988-2024.xlsx"
    return world, olympics
