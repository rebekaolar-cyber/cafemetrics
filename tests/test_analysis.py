"""Tests for analysis module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.analysis import (
    detect_afternoon_dip,
    detect_trend,
    detect_anomalies,
    analyze_all,
)


@pytest.fixture
def dip_data():
    """Create data with clear afternoon dip."""
    records = []
    for day in range(30):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        for hour in range(10, 20):
            if 14 <= hour <= 15:  # dip window
                revenue = 5.0 + np.random.normal(0, 0.5)  # low
            else:
                revenue = 15.0 + np.random.normal(0, 1.0)  # high
            records.append({
                "date": date,
                "hour": hour,
                "day_of_week": date.weekday(),
                "category": "Coffee",
                "product": "Espresso",
                "quantity": 1,
                "unit_price": 2.50,
                "revenue": max(0.5, revenue),
            })
    return pd.DataFrame(records)


@pytest.fixture
def trend_data():
    """Create data with clear upward trend."""
    records = []
    for day in range(30):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        daily_revenue = 500 + (day * 10) + np.random.normal(0, 20)
        records.append({
            "date": date,
            "hour": 12,
            "day_of_week": date.weekday(),
            "category": "Coffee",
            "product": "Espresso",
            "quantity": 1,
            "unit_price": 2.50,
            "revenue": daily_revenue,
        })
    return pd.DataFrame(records)


@pytest.fixture
def anomaly_data():
    """Create data with clear anomalies."""
    records = []
    for day in range(30):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        if day == 10:  # Day with large anomaly
            revenue = 10.0  # Very low
        elif day == 20:  # Another large anomaly
            revenue = 500.0  # Very high
        else:
            revenue = 100.0 + np.random.normal(0, 5.0)
        records.append({
            "date": date,
            "hour": 12,
            "day_of_week": date.weekday(),
            "category": "Coffee",
            "product": "Espresso",
            "quantity": 1,
            "unit_price": 2.50,
            "revenue": max(0.5, revenue),
        })
    return pd.DataFrame(records)


def test_detect_afternoon_dip(dip_data):
    """Test afternoon dip detection."""
    result = detect_afternoon_dip(dip_data)

    assert result["detected"] == True
    assert result["sample_size"] > 0
    assert result["dip_avg_revenue"] < result["overall_avg_revenue"]
    assert result["effect_size"] > 0
    assert isinstance(result["p_value"], float)


def test_detect_afternoon_dip_no_dip():
    """Test when there is no afternoon dip."""
    records = []
    for day in range(20):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        for hour in range(10, 20):
            records.append({
                "date": date,
                "hour": hour,
                "day_of_week": date.weekday(),
                "category": "Coffee",
                "product": "Espresso",
                "quantity": 1,
                "unit_price": 2.50,
                "revenue": 15.0,  # Constant
            })
    df = pd.DataFrame(records)
    result = detect_afternoon_dip(df)

    assert result["detected"] == False
    assert result["dip_avg_revenue"] == result["overall_avg_revenue"]


def test_detect_afternoon_dip_empty():
    """Test afternoon dip detection with empty data."""
    df = pd.DataFrame({"hour": [], "date": [], "revenue": []})
    result = detect_afternoon_dip(df)

    assert result["detected"] == False
    assert result["sample_size"] == 0


def test_detect_afternoon_dip_keys():
    """Test that all expected keys are in result."""
    df = pd.DataFrame({"hour": [10], "date": [datetime(2026, 4, 1)], "revenue": [10.0]})
    result = detect_afternoon_dip(df)

    expected_keys = {
        "detected",
        "sample_size",
        "effect_size",
        "effect_size_ci",      # bootstrap interval for the effect size
        "days_in_data",        # days present in the file
        "days_excluded",       # days with nothing to pair against
        "exclusion_note",      # why they were left out, in words
        "p_value",             # raw, from the paired t-test
        "p_value_corrected",   # Benjamini-Hochberg, across the family actually run
        "family_size",         # how many tests that family held
        "dip_hours",
        "dip_avg_revenue",
        "overall_avg_revenue",
    }
    assert set(result.keys()) == expected_keys


def test_detect_trend(trend_data):
    """Test trend detection."""
    result = detect_trend(trend_data)

    assert result["detected"] == True  # Clear upward trend
    assert result["slope"] > 0  # Positive slope
    assert result["sample_size"] == 30
    assert result["trend_direction"] == "upward"
    assert 0 <= result["r_squared"] <= 1


def test_detect_trend_no_trend():
    """Test when there is no trend.

    The noise is seeded. Unseeded, this test was flaky by construction: on pure
    noise the p-value is uniform on [0, 1], so about 1 run in 20 crossed 0.05
    and failed, whichever trend model was in use (measured: 4.3% with the old
    unadjusted fit, 5.1% with the weekday-adjusted one -- both just the false-
    positive rate a 0.05 test is defined to have). Seeding makes the assertion
    deterministic without loosening it; the rate itself is checked over many
    draws in test_statistics_support.test_pure_noise_is_not_called_a_trend.
    """
    rng = np.random.default_rng(20260401)
    records = []
    for day in range(20):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        records.append({
            "date": date,
            "hour": 12,
            "day_of_week": date.weekday(),
            "category": "Coffee",
            "product": "Espresso",
            "quantity": 1,
            "unit_price": 2.50,
            "revenue": 50.0 + rng.normal(0, 1.0),  # Random noise, no trend
        })
    df = pd.DataFrame(records)
    result = detect_trend(df)

    # With small noise, slope should be near zero
    assert abs(result["slope"]) < 5.0
    assert result["p_value"] > 0.05 or np.isnan(result["p_value"])
    assert result["detected"] is False


def test_detect_trend_downward():
    """Test downward trend detection."""
    records = []
    for day in range(20):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        records.append({
            "date": date,
            "hour": 12,
            "day_of_week": date.weekday(),
            "category": "Coffee",
            "product": "Espresso",
            "quantity": 1,
            "unit_price": 2.50,
            "revenue": 500 - (day * 10),  # Decreasing
        })
    df = pd.DataFrame(records)
    result = detect_trend(df)

    assert result["slope"] < 0
    assert result["trend_direction"] == "downward"


def test_detect_trend_small_sample():
    """Test trend detection with very small sample."""
    df = pd.DataFrame({
        "date": [datetime(2026, 4, 1), datetime(2026, 4, 2)],
        "hour": [12, 12],
        "day_of_week": [0, 1],
        "category": ["Coffee", "Coffee"],
        "product": ["Espresso", "Espresso"],
        "quantity": [1, 1],
        "unit_price": [2.50, 2.50],
        "revenue": [50.0, 55.0],
    })
    result = detect_trend(df)

    assert result["sample_size"] == 2
    assert result["detected"] == False  # Too small for significance


def test_detect_anomalies(anomaly_data):
    """Test anomaly detection."""
    result = detect_anomalies(anomaly_data, z_threshold=2.0)  # Slightly lower threshold

    assert result["detected"] == True
    assert result["anomaly_count"] > 0
    assert 0 <= result["anomaly_ratio"] <= 1
    assert result["sample_size"] == 30


def test_detect_anomalies_no_anomalies():
    """Test when there are no anomalies.

    The noise is seeded, and has to be. This test asserts the ABSENCE of a rare
    event on freshly drawn random data, which makes it the one fixture in this
    file whose outcome sits on a threshold: the scan flags about 0.96% of
    ordinary days by design, 13 of these 20 days are scanned, so an unseeded
    version fails whenever any of the 13 draws lands in the tail — measured at
    13.2% of runs over 3000 trials, against 1-(1-0.0096)^13 = 11.8% predicted
    from the detector's own documented rate. (The small excess is the overlap
    between neighbouring windows, which makes consecutive z-scores correlated,
    so the independent-trials formula is only an approximation.)

    Two things worth recording about that. It is NOT numerical instability: the
    failure rate is identical at a noise sd of 0.5 and of 5.0, where a
    floating-point effect would move. And this test used to pass unconditionally
    for a different reason — the default threshold was unreachable, so
    anomaly_count was always 0 whatever the data, and the assertion below was
    vacuous. Making the scan work is what exposed the flake. The assertion is
    unchanged; days_scanned is now checked too, so it cannot go vacuous again.

    The other unseeded fixtures in this file (dip_data, trend_data,
    anomaly_data) test large planted effects far from any threshold and do not
    need seeding: 0 failures in 150 redraws each.
    """
    rng = np.random.default_rng(20260401)
    records = []
    for day in range(20):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        records.append({
            "date": date,
            "hour": 12,
            "day_of_week": date.weekday(),
            "category": "Coffee",
            "product": "Espresso",
            "quantity": 1,
            "unit_price": 2.50,
            "revenue": 50.0 + rng.normal(0, 0.5),  # Small variation
        })
    df = pd.DataFrame(records)
    result = detect_anomalies(df)

    assert result["anomaly_count"] == 0
    assert result["days_scanned"] == 13, "quiet days must actually be examined, not skipped"


def test_detect_anomalies_custom_threshold():
    """Test anomaly detection with custom z-score threshold.

    The baseline carries a little seeded variation on purpose. It used to be a
    flat 50.0 every day, which only worked because the old code divided by
    ``rolling_std + 1e-8``: with no variation in the reference week there is no
    scale to judge a day against, and that 1e-8 turned any deviation at all into
    an astronomical z-score. Those days are now reported as unscannable instead,
    so a fixture meant to test threshold *sensitivity* has to contain something
    to be sensitive to.
    """
    rng = np.random.default_rng(20260401)
    revenue = list(50.0 + rng.normal(0, 2.0, 19)) + [200.0]  # last day is anomalous
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=20),
        "hour": [12] * 20,
        "day_of_week": [0] * 20,
        "category": ["Coffee"] * 20,
        "product": ["Espresso"] * 20,
        "quantity": [1] * 20,
        "unit_price": [2.50] * 20,
        "revenue": revenue,
    })

    # High threshold: less sensitive
    result_high = detect_anomalies(df, z_threshold=50.0)
    # Low threshold: more sensitive
    result_low = detect_anomalies(df, z_threshold=1.0)

    assert result_high["anomaly_count"] < result_low["anomaly_count"]
    assert result_low["anomaly_count"] >= 1
    # The planted spike is found at the shipped default, and it is the last day.
    assert detect_anomalies(df)["anomalous_dates"] == ["2026-04-20"]


def test_detect_anomalies_small_sample():
    """Test anomaly detection with small sample."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=5),
        "hour": [12] * 5,
        "day_of_week": [0] * 5,
        "category": ["Coffee"] * 5,
        "product": ["Espresso"] * 5,
        "quantity": [1] * 5,
        "unit_price": [2.50] * 5,
        "revenue": [50.0] * 5,
    })
    result = detect_anomalies(df)

    assert result["detected"] == False
    assert result["sample_size"] == 5


def test_analyze_all(dip_data):
    """Test combined analysis."""
    result = analyze_all(dip_data)

    assert "afternoon_dip" in result
    assert "trend" in result
    assert "anomalies" in result

    assert isinstance(result["afternoon_dip"], dict)
    assert isinstance(result["trend"], dict)
    assert isinstance(result["anomalies"], dict)


# ---------------------------------------------------------------------------
# Family-wise correction, intervals, and the weekday-adjusted trend
# ---------------------------------------------------------------------------

from scipy import stats

from src.analysis import (
    ANOMALY_MIN_DAYS,
    ANOMALY_WINDOW,
    DEFAULT_Z_THRESHOLD,
    FAMILY_ALPHA,
    P_VALUE_TESTS,
    apply_family_correction,
)
from src.statistics_support import benjamini_hochberg


def test_a_detector_called_on_its_own_is_a_family_of_one(dip_data):
    """Standalone, there is nothing to correct for, and the result says so."""
    dip = detect_afternoon_dip(dip_data)
    assert dip["family_size"] == 1
    assert dip["p_value_corrected"] == dip["p_value"]

    trend = detect_trend(dip_data)
    assert trend["family_size"] == 1
    assert trend["p_value_corrected"] == trend["p_value"]


def test_analyze_all_corrects_across_the_family_it_ran(dip_data):
    """The corrected p-values are exactly BH over the raw ones, and the raw ones survive."""
    result = analyze_all(dip_data)

    raw = [result[name]["p_value"] for name in P_VALUE_TESTS]
    expected = benjamini_hochberg(raw)
    for name, want in zip(P_VALUE_TESTS, expected):
        assert result[name]["p_value_corrected"] == pytest.approx(want)
        assert result[name]["family_size"] == len(P_VALUE_TESTS)

    # The correction can only make a p-value larger, never smaller.
    for name in P_VALUE_TESTS:
        assert result[name]["p_value_corrected"] >= result[name]["p_value"]


def test_the_anomaly_scan_is_not_in_the_family(dip_data):
    """It is a thresholding rule with no p-value, so it is not corrected."""
    result = analyze_all(dip_data)
    assert "anomalies" not in P_VALUE_TESTS
    assert "p_value" not in result["anomalies"]
    assert "p_value_corrected" not in result["anomalies"]


def test_detected_is_decided_by_the_corrected_p_value():
    """
    A dip whose raw p-value clears 0.05 but whose corrected one does not must
    not be reported. Built by hand, so the decision rule is tested on its own
    rather than through whatever the sample data happens to give.
    """
    results = {
        "afternoon_dip": {
            "detected": True, "sample_size": 40,
            "p_value": 0.04, "p_value_corrected": 0.04, "family_size": 1,
            "dip_avg_revenue": 50.0, "overall_avg_revenue": 60.0,
        },
        "trend": {
            "detected": False, "trend_direction": "upward",
            "p_value": 0.90, "p_value_corrected": 0.90, "family_size": 1,
        },
    }
    apply_family_correction(results)

    # m = 2, sorted raw [0.04, 0.90]: 0.04 * 2/1 = 0.08, 0.90 * 2/2 = 0.90.
    assert results["afternoon_dip"]["p_value_corrected"] == pytest.approx(0.08)
    assert results["afternoon_dip"]["p_value"] == 0.04          # raw value untouched
    assert results["afternoon_dip"]["family_size"] == 2
    assert results["afternoon_dip"]["detected"] is False        # 0.08 >= 0.05, so not reported

    # The trend's own corrected p-value is unchanged by the correction here, and
    # it was never significant, so it stays unreported too.
    assert results["trend"]["p_value_corrected"] == pytest.approx(0.90)
    assert results["trend"]["detected"] is False


def test_a_dip_that_is_lower_but_not_significant_is_not_reported():
    """The window being lower is not enough on its own; it never was meant to be."""
    results = {
        "afternoon_dip": {
            "detected": True, "sample_size": 40,
            "p_value": 0.40, "p_value_corrected": 0.40, "family_size": 1,
            "dip_avg_revenue": 59.0, "overall_avg_revenue": 60.0,
        },
    }
    apply_family_correction(results)
    assert results["afternoon_dip"]["detected"] is False


def test_the_dip_effect_size_comes_with_an_interval(dip_data):
    dip = detect_afternoon_dip(dip_data)
    ci = dip["effect_size_ci"]
    assert ci["ci_low"] < dip["effect_size"] < ci["ci_high"]
    assert ci["confidence"] == 0.95 and ci["n_effective"] > 0


def test_the_trend_slope_comes_with_an_interval(trend_data):
    trend = detect_trend(trend_data)
    ci = trend["slope_ci"]
    assert ci["ci_low"] < trend["slope"] < ci["ci_high"]
    assert ci["ci_low"] > 0  # a planted upward trend: the interval clears zero


def test_the_intervals_are_the_same_on_every_run(dip_data):
    """Fixed seed: a reported interval must not move between runs."""
    assert detect_afternoon_dip(dip_data)["effect_size_ci"] == \
        detect_afternoon_dip(dip_data)["effect_size_ci"]
    assert detect_trend(dip_data)["slope_ci"] == detect_trend(dip_data)["slope_ci"]


def test_an_empty_dip_result_still_carries_the_new_fields():
    dip = detect_afternoon_dip(pd.DataFrame({"hour": [], "date": [], "revenue": []}))
    assert dip["family_size"] == 1 and dip["p_value_corrected"] == 1.0
    assert np.isnan(dip["effect_size_ci"]["ci_low"])


def test_a_dip_with_no_variation_does_not_produce_a_nan_p_value():
    """ttest_rel on identical series gives NaN; a NaN would break the correction."""
    records = [
        {"date": datetime(2026, 4, 1) + timedelta(days=day), "hour": hour,
         "day_of_week": 0, "category": "Coffee", "product": "Espresso",
         "quantity": 1, "unit_price": 2.5, "revenue": 15.0}
        for day in range(20) for hour in range(10, 20)
    ]
    result = analyze_all(pd.DataFrame(records))
    assert result["afternoon_dip"]["p_value"] == 1.0
    assert result["afternoon_dip"]["p_value_corrected"] == 1.0
    assert result["afternoon_dip"]["detected"] is False


def test_the_trend_keeps_the_unadjusted_fit_beside_the_adjusted_one(trend_data):
    trend = detect_trend(trend_data)
    for key in ("slope_unadjusted", "p_value_unadjusted", "r_squared_unadjusted"):
        assert key in trend
    # Same data, two models: the unadjusted fit is the old linregress.
    daily = trend_data.groupby("date")["revenue"].sum()
    plain = stats.linregress(np.arange(len(daily)), daily.values)
    assert trend["p_value_unadjusted"] == pytest.approx(plain.pvalue)
    assert trend["slope_unadjusted"] == pytest.approx(plain.slope)


def test_a_purely_weekly_pattern_is_not_reported_as_a_trend():
    """
    The reason the weekday effect comes out first. 29 days -- four whole weeks
    plus a Monday -- with revenue set only by the weekday. There is no trend,
    but the unbalanced window tilts a weekday-blind straight line.
    """
    by_weekday = {0: 300.0, 1: 320.0, 2: 340.0, 3: 380.0, 4: 520.0, 5: 700.0, 6: 600.0}
    records = []
    for day in range(29):
        date = datetime(2026, 4, 6) + timedelta(days=day)  # a Monday
        records.append({
            "date": date, "hour": 12, "day_of_week": date.weekday(),
            "category": "Coffee", "product": "Espresso", "quantity": 1,
            "unit_price": 2.5, "revenue": by_weekday[date.weekday()],
        })
    df = pd.DataFrame(records)

    daily = df.groupby("date")["revenue"].sum()
    plain = stats.linregress(np.arange(len(daily)), daily.values)
    assert abs(plain.slope) > 1.0  # the weekday-blind fit does slope

    trend = detect_trend(df)
    assert trend["slope_unadjusted"] == pytest.approx(plain.slope)
    assert trend["slope"] == 0.0           # the weekly rhythm is all there was
    assert trend["detected"] is False


def test_family_alpha_is_the_cut_off_the_detectors_use():
    assert FAMILY_ALPHA == 0.05


def test_the_default_threshold_is_reachable():
    """
    REGRESSION GUARD. The default threshold must be able to fire.

    For most of this repo's life it could not: each day was scored against a
    rolling window that *included itself*, which bounds the score at
    (k - 1) / sqrt(k) = 2.268 for k = 7, below the then-default of 2.5. The
    detector returned zero anomalies for every possible dataset. If anyone ever
    puts the day back into its own reference window, or raises the default above
    the bound, this test fails.
    """
    rng = np.random.default_rng(7)
    revenue = list(100.0 + rng.normal(0, 5.0, 30))
    revenue[20] = 1000.0          # one unmistakable spike, with a full week before it
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=30),
        "hour": [12] * 30, "day_of_week": [0] * 30,
        "category": ["Coffee"] * 30, "product": ["Espresso"] * 30,
        "quantity": [1] * 30, "unit_price": [2.5] * 30, "revenue": revenue,
    })

    result = detect_anomalies(df)                       # the shipped default
    assert result["anomaly_count"] >= 1
    assert "2026-04-21" in result["anomalous_dates"]    # the spike, index 20
    assert result["detected"] is True


def test_why_the_day_is_excluded_from_its_own_window():
    """
    The algebra that made the old self-inclusive window unable to fire, kept as
    a guard against anyone 'simplifying' the shift(1) away.

    A value inside the sample it is measured against has a bounded standardised
    deviation: |x - mean| / sd <= (k - 1) / sqrt(k). No spike, however large,
    can exceed it. The shipped default must stay above that bound so the
    self-inclusive formula could never satisfy it.
    """
    for k in range(2, 8):
        spike = np.array([1.0] * (k - 1) + [1e12])      # as extreme as floats allow
        self_inclusive_z = abs(spike[-1] - spike.mean()) / spike.std(ddof=1)
        assert self_inclusive_z == pytest.approx((k - 1) / np.sqrt(k))

    assert (ANOMALY_WINDOW - 1) / np.sqrt(ANOMALY_WINDOW) == pytest.approx(2.2678, abs=1e-4)
    assert DEFAULT_Z_THRESHOLD > (ANOMALY_WINDOW - 1) / np.sqrt(ANOMALY_WINDOW)


# ---------------------------------------------------------------------------
# Robustness: ragged opening hours, short files, degenerate revenue
# ---------------------------------------------------------------------------

def _frame(rows):
    """Build a sales frame from (date, hour, revenue) triples."""
    df = pd.DataFrame(rows, columns=["date", "hour", "revenue"])
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_week"] = df["date"].dt.weekday
    df["category"] = "Coffee"
    df["product"] = "Espresso"
    df["quantity"] = 1
    df["unit_price"] = 2.5
    return df


def _full_day(date, revenue=10.0, hours=range(10, 20)):
    return [(date, hour, revenue) for hour in hours]


def test_uneven_opening_hours_do_not_raise():
    """
    The crash this guard exists for: scipy's "unequal length arrays".

    Three full days plus one day that sold nothing but afternoon coffee gives
    four dip-days against three non-dip days.
    """
    rows = _full_day("2026-04-01") + _full_day("2026-04-02") + _full_day("2026-04-03")
    rows += [("2026-04-04", hour, 10.0) for hour in (14, 15, 16)]

    result = detect_afternoon_dip(_frame(rows))
    assert result["sample_size"] == 3        # the three fully-traded days
    assert result["days_in_data"] == 4
    assert result["days_excluded"] == 1
    assert "1 of 4 days" in result["exclusion_note"]


def test_a_day_that_closes_before_the_afternoon_is_excluded():
    rows = _full_day("2026-04-01") + _full_day("2026-04-02") + _full_day("2026-04-03")
    rows += [("2026-04-04", hour, 10.0) for hour in range(10, 14)]  # closes at 14:00

    result = detect_afternoon_dip(_frame(rows))
    assert result["sample_size"] == 3
    assert result["days_excluded"] == 1


def test_days_are_paired_by_date_not_by_position():
    """
    The subtler half of the same bug, and the dangerous one.

    One full day, one morning-only day and one afternoon-only day give two
    dip-days and two non-dip days -- equal lengths, so no exception -- but they
    describe different dates. Pairing by position would compare 7 April's
    afternoon with 6 April's morning. Only the one fully-traded day is a pair.
    """
    rows = _full_day("2026-04-05")
    rows += [("2026-04-06", hour, 5.0) for hour in range(10, 14)]    # morning only
    rows += [("2026-04-07", hour, 99.0) for hour in (14, 15, 16)]    # afternoon only

    result = detect_afternoon_dip(_frame(rows))
    assert result["sample_size"] == 1        # not 2: only 5 April is a genuine pair
    assert result["days_excluded"] == 2
    # One pair cannot support a paired test, and the result says so rather than
    # reporting a p-value computed across mismatched dates.
    assert result["p_value"] == 1.0
    assert result["detected"] is False


def test_only_one_day_is_shared_between_the_two_series():
    rows = [("2026-04-01", hour, 10.0) for hour in range(10, 14)]
    rows += [("2026-04-02", hour, 10.0) for hour in (14, 15, 16)]
    rows += _full_day("2026-04-03")

    result = detect_afternoon_dip(_frame(rows))
    assert result["sample_size"] == 1
    assert result["detected"] is False


def test_a_one_day_file_does_not_raise_anywhere():
    df = _frame(_full_day("2026-04-01"))
    assert detect_afternoon_dip(df)["sample_size"] == 1   # one day cannot be paired
    assert detect_trend(df)["trend_direction"] == "none"
    assert detect_anomalies(df)["detected"] is False
    analyze_all(df)  # must not raise


def test_a_day_with_a_single_transaction_does_not_raise():
    rows = _full_day("2026-04-01") + _full_day("2026-04-02")
    rows += [("2026-04-03", 14, 7.0)]          # one sale, inside the window only
    rows += _full_day("2026-04-04")
    result = analyze_all(_frame(rows))
    assert result["afternoon_dip"]["days_excluded"] == 1


def test_the_excluded_days_feed_through_to_the_confidence_score():
    """
    sample_size must be the days compared, not the days uploaded, or the score
    claims evidence the test never had.
    """
    from src.insights import generate_afternoon_dip_insight

    rows = []
    for day in range(1, 31):                     # 30 fully-traded days, clear dip
        date = f"2026-04-{day:02d}"
        for hour in range(10, 20):
            rows.append((date, hour, 5.0 if 14 <= hour <= 16 else 15.0))
    for day in range(1, 21):                     # 20 afternoon-only days on top
        rows.append((f"2026-05-{day:02d}", 15, 5.0))

    analysis = analyze_all(_frame(rows))
    dip = analysis["afternoon_dip"]
    assert dip["days_in_data"] == 50 and dip["sample_size"] == 30 and dip["days_excluded"] == 20

    insight = generate_afternoon_dip_insight(analysis, calendar_days=50)
    # Consistency is 30/50, not 50/50: the 20 unpaired days earn nothing.
    assert insight.evidence["sample_size"] == 30
    assert insight.evidence["consistency_ratio"] == pytest.approx(30 / 50)


def test_an_empty_frame_does_not_raise_in_any_detector():
    """
    detect_trend used to derive the weekday before its length check, which
    raised AttributeError on an object-dtype empty 'date' column -- exactly the
    frame test_detect_afternoon_dip_empty already passes in.
    """
    for df in (
        pd.DataFrame({"hour": [], "date": [], "revenue": []}),
        pd.DataFrame({"date": pd.to_datetime([]), "hour": [], "revenue": []}),
    ):
        assert detect_afternoon_dip(df)["sample_size"] == 0
        assert detect_trend(df)["trend_direction"] == "none"
        assert detect_anomalies(df)["detected"] is False
        analyze_all(df)  # must not raise


def test_all_identical_revenue_does_not_raise_or_invent_a_finding():
    df = _frame([(f"2026-04-{day:02d}", hour, 10.0)
                 for day in range(1, 15) for hour in range(10, 20)])
    result = analyze_all(df)
    assert result["afternoon_dip"]["detected"] is False
    assert result["afternoon_dip"]["p_value"] == 1.0      # NaN from scipy, cleaned to 1.0
    assert result["trend"]["detected"] is False
    assert result["trend"]["slope"] == 0.0
    assert result["anomalies"]["anomaly_count"] == 0


def test_the_dip_averages_describe_only_the_compared_days():
    """
    An excluded day's revenue must not leak into the averages shown beside the
    test, or the numbers describe a different set of days than the p-value does.
    """
    rows = _full_day("2026-04-01", revenue=10.0) + _full_day("2026-04-02", revenue=10.0)
    with_outlier = rows + [("2026-04-03", hour, 1000.0) for hour in (14, 15, 16)]

    base = detect_afternoon_dip(_frame(rows))
    with_excluded = detect_afternoon_dip(_frame(with_outlier))
    assert with_excluded["dip_avg_revenue"] == pytest.approx(base["dip_avg_revenue"])
    assert with_excluded["overall_avg_revenue"] == pytest.approx(base["overall_avg_revenue"])


def test_ragged_opening_hours_do_not_emit_a_wall_of_warnings():
    """
    Paired days can still trade different hours, so a bootstrap resample may
    draw no day that traded some hour. nanmean warns and returns NaN, which the
    detector drops; the warning is suppressed narrowly so a real till export
    does not print thousands of them.
    """
    import warnings as _warnings

    rows = [("2026-04-01", 9, 3.0)] + _full_day("2026-04-01")
    for date in ("2026-04-02", "2026-04-03", "2026-04-04"):
        rows += _full_day(date)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = detect_afternoon_dip(_frame(rows))

    assert [w for w in caught if "empty slice" in str(w.message)] == []
    assert result["sample_size"] == 4 and result["days_excluded"] == 0


def _daily(revenue, start="2026-04-01"):
    n = len(revenue)
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n),
        "hour": [12] * n, "day_of_week": [0] * n,
        "category": ["Coffee"] * n, "product": ["Espresso"] * n,
        "quantity": [1] * n, "unit_price": [2.5] * n, "revenue": revenue,
    })


def test_a_day_is_scored_against_the_week_before_it_only():
    """
    The shift: the reference window must hold the 7 days BEFORE the day, and
    never the day itself. Checked by computing the expected z by hand.
    """
    rng = np.random.default_rng(5)
    revenue = list(100.0 + rng.normal(0, 5.0, 20))
    revenue[15] = 500.0
    df = _daily(revenue)

    result = detect_anomalies(df)
    prior_week = np.array(revenue[8:15])          # the 7 days before index 15
    expected_z = (revenue[15] - prior_week.mean()) / prior_week.std(ddof=1)

    assert abs(expected_z) > DEFAULT_Z_THRESHOLD
    assert "2026-04-16" in result["anomalous_dates"]   # index 15
    # The spike is NOT in its own reference window, so it cannot damp its own score.
    assert expected_z > (ANOMALY_WINDOW - 1) / np.sqrt(ANOMALY_WINDOW)


def test_the_first_seven_days_are_reported_as_unscanned_not_as_clean():
    rng = np.random.default_rng(6)
    df = _daily(list(100.0 + rng.normal(0, 5.0, 40)))

    result = detect_anomalies(df)
    assert result["sample_size"] == 40
    assert result["days_scanned"] == 40 - ANOMALY_WINDOW
    assert result["days_unscored"] == ANOMALY_WINDOW
    assert result["days_unscored_early"] == ANOMALY_WINDOW
    assert result["days_unscored_flat"] == 0
    assert "7 of 40 days were not scanned" in result["unscored_note"]
    assert "no full week of sales before them" in result["unscored_note"]


def test_a_flat_reference_week_makes_a_day_unscorable_not_infinitely_odd():
    """
    The 1e-8 fudge is gone. Seven identical days then a different one used to
    divide by 1e-8 and flag the day as astronomically anomalous; with no
    variation in the reference week there is no scale, so the day is unscored.
    """
    # Ten identical days, then a 50% jump. Every day with a reference window
    # here has a flat one, so nothing can be scored.
    df = _daily([100.0] * 10 + [150.0])

    result = detect_anomalies(df)
    assert result["anomaly_count"] == 0
    assert result["anomalous_dates"] == []
    assert result["days_scanned"] == 0
    assert result["days_unscored_flat"] == len(df) - ANOMALY_WINDOW
    assert "identical revenue every day" in result["unscored_note"]
    assert "no scale to measure against" in result["unscored_note"]

    # Not an anomaly at ANY threshold, because there is no scale at all.
    assert detect_anomalies(df, z_threshold=0.5)["anomaly_count"] == 0

    # What the old `rolling_std + 1e-8` denominator did with the same day: a
    # z-score of around 5e9, i.e. an automatic anomaly on a penny of movement.
    prior_week = np.array([100.0] * 7)
    old_style_z = (150.0 - prior_week.mean()) / (prior_week.std(ddof=1) + 1e-8)
    assert old_style_z > 1e9

    # Once the reference week has any real variation, scoring resumes.
    rng = np.random.default_rng(12)
    varied = _daily(list(100.0 + rng.normal(0, 5.0, 10)) + [150.0])
    assert detect_anomalies(varied)["days_scanned"] == 4


def test_the_ratio_and_mean_z_use_the_scanned_days_as_the_denominator():
    rng = np.random.default_rng(8)
    revenue = list(100.0 + rng.normal(0, 5.0, 20))
    revenue[14] = 600.0
    result = detect_anomalies(_daily(revenue))

    assert result["days_scanned"] == 13
    assert result["anomaly_ratio"] == pytest.approx(result["anomaly_count"] / 13)
    assert result["mean_z_score"] > 0


def test_a_file_too_short_to_scan_says_so():
    for n in (1, 5, 7):
        result = detect_anomalies(_daily([100.0 + i for i in range(n)]))
        assert result["sample_size"] == n
        assert result["days_scanned"] == 0
        assert result["detected"] is False
        assert f"at least {ANOMALY_MIN_DAYS} days" in result["unscored_note"]

    # Eight days is the minimum that can scan anything: exactly one day.
    rng = np.random.default_rng(9)
    assert detect_anomalies(_daily(list(100.0 + rng.normal(0, 5.0, 8))))["days_scanned"] == 1


def test_the_threshold_default_and_its_rate_are_reported_together():
    from src.statistics_support import expected_false_flag_rate

    rng = np.random.default_rng(10)
    result = detect_anomalies(_daily(list(100.0 + rng.normal(0, 5.0, 30))))
    assert result["expected_false_flag_rate"] == pytest.approx(
        expected_false_flag_rate(DEFAULT_Z_THRESHOLD, ANOMALY_WINDOW)
    )
    # A passed threshold is reflected in the reported rate, not just the count.
    looser = detect_anomalies(_daily(list(100.0 + rng.normal(0, 5.0, 30))), z_threshold=2.5)
    assert looser["expected_false_flag_rate"] > result["expected_false_flag_rate"]


def test_the_sample_data_now_surfaces_two_anomalies():
    """
    Pins the newly visible finding on the shipped sample, which reported nothing
    for the whole life of the repo because the threshold was unreachable.

    2026-05-02 is the day the generator's planted 6-day dip (offsets 25-30, at
    0.7x) ends: its reference week is depressed, it is not, hence z = +5.9.
    2026-06-14 (z = +4.9) is a genuine excursion -- and over 85 scanned days at
    ~0.96% a false flag is expected about 0.8 times, so one of the two could be
    noise. That is what expected_false_flag_rate is in the output for.
    """
    from src.data_loader import load_sales_data

    result = detect_anomalies(load_sales_data("data/sample_sales.csv"))
    assert result["anomalous_dates"] == ["2026-05-02", "2026-06-14"]
    assert result["days_scanned"] == 85 and result["days_unscored"] == 7
    assert result["detected"] is True
