from pathlib import Path

from src.data_loader import load_results_workbook
from src.target_builder import build_olympic_targets


def test_real_olympic_progression_targets_are_complete_from_2000():
    path = Path(__file__).resolve().parents[1] / "data" / "Olympic_Games_50m_Freestyle_Complete_Results_1988-2024.xlsx"
    olympics, _ = load_results_workbook(path, "olympics")
    targets, quality = build_olympic_targets(olympics)
    standard = quality[quality["target_year"] >= 2000]
    assert standard["complete_27"].all()
    assert not standard[["heat_fallback", "semifinal_fallback", "medal_fallback"]].any().any()
    counts = targets[targets["target_year"] >= 2000].groupby(["target_year", "sex"]).size()
    assert (counts == 27).all()


def test_early_olympics_have_masked_missing_stages():
    path = Path(__file__).resolve().parents[1] / "data" / "Olympic_Games_50m_Freestyle_Complete_Results_1988-2024.xlsx"
    olympics, _ = load_results_workbook(path, "olympics")
    targets, quality = build_olympic_targets(olympics)
    early = targets[targets["target_year"] < 2000]
    assert set(early["target_group"]) == {"FINALS"}
    assert not quality[quality["target_year"] < 2000]["complete_27"].any()
