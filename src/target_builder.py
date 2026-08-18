from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _name_key(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold()


def _fastest(rows: pd.DataFrame, count: int) -> np.ndarray:
    return np.sort(rows.loc[rows["valid_time"], "Time_seconds"].to_numpy(dtype=float))[:count]


def build_olympic_targets(olympics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build H16/S8/F3 targets; names are used only inside this function."""
    if set(olympics["source"].unique()) != {"olympics"}:
        raise ValueError("Olympic target builder accepts Olympic rows only")
    output: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for (year, sex), group in olympics.groupby(["Year", "Sex"], sort=True):
        group = group.copy()
        group["_name_key"] = _name_key(group["Name"])
        heats = group[(group["Phase"] == "Heats") & group["valid_time"]]
        semis = group[(group["Phase"] == "Semifinals") & group["valid_time"]]
        finals = group[(group["Phase"] == "Finals") & group["valid_time"]]
        semifinal_names = set(semis["_name_key"])
        final_names = set(finals["_name_key"])

        joined_heats = heats[heats["_name_key"].isin(semifinal_names)]
        heat_fallback = len(joined_heats) != 16 and int(year) >= 2000
        heat_values = _fastest(joined_heats if not heat_fallback else heats, 16) if int(year) >= 2000 else np.array([])

        joined_semis = semis[semis["_name_key"].isin(final_names)]
        semi_fallback = len(joined_semis) != 8 and int(year) >= 2000
        semi_values = _fastest(joined_semis if not semi_fallback else semis, 8) if int(year) >= 2000 else np.array([])

        if int(year) < 2000:
            medal_pool = finals[finals["Heat"].astype(str).str.strip().str.casefold() == "final a"]
        else:
            medal_pool = finals
        numeric_rank = pd.to_numeric(medal_pool["Rank"], errors="coerce")
        medal_pool = medal_pool.assign(_rank=numeric_rank)
        official_medalists = medal_pool[medal_pool["_rank"].isin([1, 2, 3])].sort_values(
            ["_rank", "Time_seconds", "Name"], kind="stable"
        )
        medal_values = official_medalists["Time_seconds"].astype(float).head(3).tolist()
        medal_official_ranks = official_medalists["_rank"].astype(int).head(3).tolist()
        medal_fallback = len(medal_values) < 3
        if medal_fallback:
            medal_values = _fastest(medal_pool, 3).tolist()
            medal_official_ranks = [None] * len(medal_values)

        for target_group, values in (("HEATS", heat_values), ("SEMIFINALS", semi_values), ("FINALS", medal_values)):
            for rank, value in enumerate(values, start=1):
                output.append({
                    "target_year": int(year), "sex": sex, "target_group": target_group,
                    "rank": rank, "official_rank": (
                        medal_official_ranks[rank - 1] if target_group == "FINALS" else None
                    ),
                    "actual_time_seconds": float(value),
                })
        quality.append({
            "target_year": int(year), "sex": sex,
            "complete_27": bool(len(heat_values) == 16 and len(semi_values) == 8 and len(medal_values) == 3),
            "heats_joined": int(len(joined_heats)), "semifinals_joined": int(len(joined_semis)),
            "heat_fallback": bool(heat_fallback), "semifinal_fallback": bool(semi_fallback),
            "medal_fallback": bool(medal_fallback),
        })
    targets = pd.DataFrame(output).sort_values(["target_year", "sex", "target_group", "rank"])
    return targets.reset_index(drop=True), pd.DataFrame(quality)
