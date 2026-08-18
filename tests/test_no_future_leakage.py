import pandas as pd

from src.preprocessing import build_context, fit_fold_scaler


def _world_frame():
    rows = []
    for year in [2001, 2003, 2005, 2007]:
        for sex in ["Male", "Female"]:
            for phase, main in [("Heats", "HEATS"), ("Semifinals", "SEMIFINALS"), ("Finals", "FINALS")]:
                rows.append({
                    "Year": year, "Sex": sex, "Phase": phase, "phase_main": main,
                    "Time_seconds": 20.0 + year % 10, "valid_time": True,
                    "subphase_id": 0, "is_swim_off": False, "Status": "OK", "Name": "x",
                })
    return pd.DataFrame(rows)


def test_context_is_strictly_before_target_year():
    world = _world_frame()
    scaler = fit_fold_scaler(world, [2001, 2003, 2005])
    context = build_context(world, "Male", 2005, scaler)
    assert [edition["year"] for edition in context] == [2001, 2003]
    assert max(edition["year"] for edition in context) < 2005

