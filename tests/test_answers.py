"""Tests for the fixed-question answers: they report what the analysis found, nothing else."""

import pandas as pd
import pytest

from src.analysis import analyze_all, detect_anomalies
from src.answers import (
    Z_THRESHOLD,
    anomaly_answer,
    dip_answer,
    effect_word,
    format_p,
    hour_range,
    trend_answer,
)
from src.data_loader import load_sales_data


def test_helpers():
    assert format_p(0.0000001) == "p < 0.001"
    assert format_p(0.0659) == "p = 0.066"
    assert [effect_word(d) for d in (0.18, 0.5, 0.79, 0.8)] == ["small", "medium", "medium", "large"]
    assert hour_range([14, 15, 16]) == "14:00 to 17:00"  # the 16:00 hour runs until 17:00


def test_the_cut_off_in_the_text_is_the_one_the_rule_uses():
    import inspect

    assert Z_THRESHOLD == inspect.signature(detect_anomalies).parameters["z_threshold"].default


CI_95 = {"confidence": 0.95, "n_resamples": 2000, "n_effective": 2000}

DIP = {
    "detected": True,
    "sample_size": 92,
    "effect_size": 0.176,
    "effect_size_ci": {"ci_low": 0.119, "ci_high": 0.235, **CI_95},
    "p_value": 1.3e-7,
    "p_value_corrected": 2.7e-7,
    "family_size": 2,
    "dip_hours": [14, 15, 16],
    "dip_avg_revenue": 78.12,
    "overall_avg_revenue": 84.94,
}


def test_dip_found():
    answer = dip_answer(DIP)
    assert answer.title == "Yes, sales dip from 14:00 to 17:00"
    assert "£78" in answer.finding and "£85" in answer.finding and "8% lower" in answer.finding
    assert "not why" in answer.finding  # the rules can't explain a dip, and the text says so
    assert "92 days" in answer.evidence and "p < 0.001" in answer.evidence
    assert "0.18" in answer.evidence and "small" in answer.evidence
    assert "95% CI 0.12 to 0.23" in answer.evidence  # the effect size is reported with an interval
    assert "correcting for the 2 tests run" in answer.evidence
    assert "quiet opening and closing hours included" in answer.evidence  # the assumption is stated


def test_dip_not_found_says_so_with_the_numbers():
    busy_afternoon = {**DIP, "detected": False, "dip_avg_revenue": 100.0, "overall_avg_revenue": 80.0}
    answer = dip_answer(busy_afternoon)
    assert answer.title == "No afternoon dip"
    assert "25% higher" in answer.finding


def test_dip_without_data():
    empty = {**DIP, "detected": False, "sample_size": 0, "dip_avg_revenue": 0.0}
    assert dip_answer(empty).title == "Not enough data to tell"


TREND = {
    "detected": False,
    "slope": 1.98,
    "slope_ci": {"ci_low": -0.03, "ci_high": 3.85, **CI_95},
    "slope_pct": 0.156,
    "p_value": 0.0536,           # weekday-adjusted fit, raw
    "p_value_corrected": 0.0536,
    "family_size": 2,
    "r_squared": 0.037,
    "slope_unadjusted": 1.98,
    "p_value_unadjusted": 0.0659,
    "r_squared_unadjusted": 0.037,
    "sample_size": 92,
    "trend_direction": "upward",
}


def test_a_trend_that_is_not_significant_is_not_called_a_trend():
    answer = trend_answer(TREND)
    assert answer.title == "No clear trend"
    assert "£1.98 a day" in answer.finding and "rises" in answer.finding
    assert "4%" in answer.evidence
    assert "only reported below p = 0.05" in answer.evidence
    # Both p-values are shown, and each is labelled: the weekday-adjusted one
    # the rule acts on, and the unadjusted one it replaced.
    assert "p = 0.054 after correcting for the 2 tests run" in answer.evidence
    assert "Without taking the weekly pattern out" in answer.evidence
    assert "p = 0.066" in answer.evidence
    assert "-£0.03 to £3.85" in answer.evidence  # the slope interval straddles zero


@pytest.mark.parametrize("slope,word", [(4.2, "rising"), (-4.2, "falling")])
def test_a_significant_trend_gives_its_direction_and_size(slope, word):
    answer = trend_answer({**TREND, "detected": True, "slope": slope, "p_value": 0.002})
    assert answer.title == f"Revenue is {word}"
    assert "£4.20 a day" in answer.finding


def test_trend_without_data():
    short = {**TREND, "trend_direction": "none", "sample_size": 2}
    assert trend_answer(short).title == "Not enough data to tell"


ANOMALIES = {
    "detected": True,
    "anomaly_count": 2,
    "anomaly_ratio": 2 / 92,
    "mean_z_score": 0.8,
    "sample_size": 92,
    "anomalous_dates": ["2026-04-25", "2026-05-09"],
}


def test_anomalies_are_listed_by_date():
    answer = anomaly_answer(ANOMALIES)
    assert answer.title == "2 of 92 days stand out"
    assert "2026-04-25, 2026-05-09" in answer.finding
    assert "2.5 standard deviations" in answer.evidence

    one = {**ANOMALIES, "anomaly_count": 1, "anomalous_dates": ["2026-04-25"]}
    assert anomaly_answer(one).title == "1 of 92 days stands out"


def test_no_anomalies_and_too_little_data():
    none = {**ANOMALIES, "detected": False, "anomaly_count": 0, "anomalous_dates": []}
    assert anomaly_answer(none).title == "No day stands out"
    assert anomaly_answer({**none, "sample_size": 5}).title == "Not enough data to tell"


def test_answers_agree_with_the_insights_on_the_sample_data():
    """On the shipped sample, each answer says 'found' exactly when the rule detected it."""
    analysis = analyze_all(load_sales_data("data/sample_sales.csv"))
    assert dip_answer(analysis["afternoon_dip"]).title.startswith("Yes") == bool(
        analysis["afternoon_dip"]["detected"]
    )
    assert (trend_answer(analysis["trend"]).title != "No clear trend") == bool(
        analysis["trend"]["detected"]
    )
    assert (anomaly_answer(analysis["anomalies"]).title != "No day stands out") == bool(
        analysis["anomalies"]["detected"]
    )
    assert isinstance(analysis["trend"]["slope"], float) and pd.notna(analysis["trend"]["slope"])


def test_the_app_reports_nothing_found_gracefully_when_correction_removes_the_dip():
    """
    A legitimate outcome: the family correction pushes the only finding above
    0.05 and the app has nothing to report. Nothing must crash, and every
    answer must still carry its numbers so the reader can see why.

    Built from the real sample analysis with the raw p-values replaced, so the
    rest of each result dict is exactly what the detectors produce.
    """
    from src.analysis import apply_family_correction


    df = load_sales_data("data/sample_sales.csv")
    analysis = analyze_all(df)
    analysis["afternoon_dip"]["p_value"] = 0.04
    analysis["trend"]["p_value"] = 0.90
    apply_family_correction(analysis)

    assert analysis["afternoon_dip"]["detected"] is False  # 0.04 * 2/1 = 0.08

    dip = dip_answer(analysis["afternoon_dip"])
    assert dip.title == "No afternoon dip"
    assert "£78" in dip.finding and "£85" in dip.finding       # the numbers are still shown
    assert "p = 0.080 after correcting for the 2 tests run (raw p = 0.040)" in dip.evidence

    assert trend_answer(analysis["trend"]).title == "No clear trend"
    assert anomaly_answer(analysis["anomalies"]).title == "No day stands out"

    # No finding means no insight ticket: the dashboard renders its
    # "Nothing to report" panel from an empty list.
    from src.insights import generate_afternoon_dip_insight, generate_trend_insight

    assert generate_afternoon_dip_insight(analysis, 92) is None
    assert generate_trend_insight(analysis, 92) is None
