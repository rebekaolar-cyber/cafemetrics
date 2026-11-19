"""Tests for the shared statistical machinery.

The Benjamini-Hochberg cases are checked against values worked out by hand from
the definition, written out in each test, so a reviewer can verify the maths
without running anything. The bootstrap cases pin determinism under the fixed
seed, because a confidence interval that moves between runs is not reportable.
"""

import numpy as np
import pytest
from scipy import stats

from src.statistics_support import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FAMILY_ALPHA,
    benjamini_hochberg,
    bootstrap_ci,
    correct_family,
    dip_effect_size,
    format_interval,
    resample_indices,
    weekday_adjusted_trend,
    weekday_centre,
)


# --------------------------------------------------------------------------
# Benjamini-Hochberg, against hand-computed values
# --------------------------------------------------------------------------

def test_bh_family_of_one_changes_nothing():
    """m = 1: q_(1) = p_(1) * 1/1 = p. One test needs no correction."""
    assert benjamini_hochberg([0.03]) == [0.03]


def test_bh_two_tests_is_the_project_case():
    """
    The family analyze_all() actually runs: the dip test and the trend test.

    m = 2, sorted p = [0.01, 0.04].
      rank 1: 0.01 * 2/1 = 0.02
      rank 2: 0.04 * 2/2 = 0.04
    Monotone from the top: [min(0.02, 0.04), 0.04] = [0.02, 0.04].
    """
    assert benjamini_hochberg([0.01, 0.04]) == pytest.approx([0.02, 0.04])


def test_bh_three_tests_with_the_monotonicity_step_biting():
    """
    m = 3, sorted p = [0.01, 0.03, 0.04].
      rank 1: 0.01 * 3/1 = 0.03
      rank 2: 0.03 * 3/2 = 0.045
      rank 3: 0.04 * 3/3 = 0.04
    Rank 2's 0.045 exceeds rank 3's 0.04, so the running minimum from the top
    pulls it down to 0.04 -- without that step a smaller raw p would come out
    with a larger adjusted one.
    Sorted q = [0.03, 0.04, 0.04]; input order was [0.01, 0.04, 0.03], so the
    answer is [0.03, 0.04, 0.04].
    """
    assert benjamini_hochberg([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.04, 0.04])


def test_bh_is_capped_at_one():
    """
    m = 3, sorted p = [0.5, 0.6, 0.9].
      rank 1: 0.5 * 3/1 = 1.5 -> capped at 1.0
      rank 2: 0.6 * 3/2 = 0.9
      rank 3: 0.9 * 3/3 = 0.9
    Monotone from the top gives [0.9, 0.9, 0.9], then the cap keeps them <= 1.
    """
    assert benjamini_hochberg([0.5, 0.6, 0.9]) == pytest.approx([0.9, 0.9, 0.9])


def test_bh_all_equal_p_values():
    """
    m = 4, all p = 0.01.
      ranks 1..4: 0.04, 0.02, 0.013333, 0.01
    Monotone from the top makes every one 0.01, because the largest rank's
    0.01 caps all of them. Ties therefore come out unchanged here.
    """
    assert benjamini_hochberg([0.01] * 4) == pytest.approx([0.01] * 4)


def test_bh_never_lowers_a_p_value_and_preserves_order():
    rng = np.random.default_rng(7)
    raw = rng.random(25)
    adjusted = np.array(benjamini_hochberg(raw))
    assert (adjusted >= raw - 1e-12).all()      # correction can only be conservative
    assert (adjusted <= 1.0).all()
    # Monotone: a smaller raw p-value never comes out with a larger adjusted one.
    # (Equal adjusted values are expected -- the capping step creates ties.)
    for i in range(raw.size):
        for j in range(raw.size):
            if raw[i] < raw[j]:
                assert adjusted[i] <= adjusted[j] + 1e-12


def test_bh_the_sample_data_family():
    """The two sample-data p-values, worked by hand: m = 2.

    dip 1.3296796954601355e-07 * 2/1 = 2.659359390920271e-07
    trend 0.05358471618083346 * 2/2 = 0.05358471618083346
    """
    dip, trend = 1.3296796954601355e-07, 0.05358471618083346
    assert benjamini_hochberg([dip, trend]) == pytest.approx(
        [2.659359390920271e-07, 0.05358471618083346]
    )


def test_bh_empty_family():
    assert benjamini_hochberg([]) == []


@pytest.mark.parametrize("bad", [[-0.1], [1.1], [float("nan")], [0.5, np.nan]])
def test_bh_rejects_impossible_p_values(bad):
    with pytest.raises(ValueError):
        benjamini_hochberg(bad)


def test_correct_family_keeps_the_names():
    out = correct_family({"trend": 0.04, "afternoon_dip": 0.01})
    assert set(out) == {"trend", "afternoon_dip"}
    assert out["afternoon_dip"] == pytest.approx(0.02)
    assert out["trend"] == pytest.approx(0.04)


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------

def _mean_of_resample(data):
    def statistic(rng):
        return float(np.mean(data[resample_indices(rng, data.shape[0])]))
    return statistic


def test_bootstrap_is_deterministic_under_the_fixed_seed():
    """Two identical calls must give byte-identical bounds, or nothing is reportable."""
    data = np.arange(50, dtype=float)
    first = bootstrap_ci(_mean_of_resample(data))
    second = bootstrap_ci(_mean_of_resample(data))
    assert first == second
    assert first["ci_low"] == second["ci_low"] and first["ci_high"] == second["ci_high"]


def test_bootstrap_default_seed_and_size_are_the_documented_ones():
    data = np.arange(50, dtype=float)
    assert bootstrap_ci(_mean_of_resample(data)) == bootstrap_ci(
        _mean_of_resample(data), n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED
    )


def test_bootstrap_a_different_seed_gives_a_different_interval():
    data = np.arange(50, dtype=float)
    fixed = bootstrap_ci(_mean_of_resample(data))
    other = bootstrap_ci(_mean_of_resample(data), seed=BOOTSTRAP_SEED + 1)
    assert fixed["ci_low"] != other["ci_low"]


def test_bootstrap_brackets_the_true_mean_of_a_known_sample():
    """Mean of 0..99 is 49.5; a 95% interval on 100 points should contain it."""
    data = np.arange(100, dtype=float)
    ci = bootstrap_ci(_mean_of_resample(data))
    assert ci["ci_low"] < 49.5 < ci["ci_high"]
    assert ci["confidence"] == BOOTSTRAP_CONFIDENCE
    assert ci["n_effective"] == ci["n_resamples"] == BOOTSTRAP_RESAMPLES


def test_bootstrap_of_a_constant_statistic_is_a_point():
    ci = bootstrap_ci(lambda rng: 4.0, n_resamples=10)
    assert ci["ci_low"] == ci["ci_high"] == 4.0


def test_bootstrap_discards_unusable_draws():
    calls = {"n": 0}

    def half_nan(rng):
        calls["n"] += 1
        return float("nan") if calls["n"] % 2 else 1.0

    ci = bootstrap_ci(half_nan, n_resamples=10)
    assert ci["n_resamples"] == 10
    assert ci["n_effective"] == 5
    assert ci["ci_low"] == ci["ci_high"] == 1.0


def test_bootstrap_with_no_usable_draw_reports_no_interval():
    ci = bootstrap_ci(lambda rng: float("nan"), n_resamples=5)
    assert np.isnan(ci["ci_low"]) and np.isnan(ci["ci_high"])
    assert ci["n_effective"] == 0
    assert format_interval(ci) == "no interval available"


def test_bootstrap_rejects_impossible_arguments():
    with pytest.raises(ValueError):
        bootstrap_ci(lambda rng: 1.0, n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci(lambda rng: 1.0, confidence=1.0)


def test_a_wider_interval_is_requested_and_delivered():
    data = np.arange(100, dtype=float)
    narrow = bootstrap_ci(_mean_of_resample(data), confidence=0.80)
    wide = bootstrap_ci(_mean_of_resample(data), confidence=0.99)
    assert wide["ci_high"] - wide["ci_low"] > narrow["ci_high"] - narrow["ci_low"]


def test_resample_indices_draws_with_replacement_inside_the_range():
    picks = resample_indices(np.random.default_rng(1), 20)
    assert picks.shape == (20,)
    assert picks.min() >= 0 and picks.max() < 20


def test_format_interval():
    ci = {"ci_low": 0.1185, "ci_high": 0.2349, "confidence": 0.95}
    assert format_interval(ci) == "95% CI 0.12 to 0.23"
    assert format_interval(ci, unit="£") == "95% CI £0.12 to £0.23"
    assert format_interval(ci, decimals=3) == "95% CI 0.118 to 0.235"  # 0.1185 rounds down in binary
    # A negative bound takes its sign outside the unit.
    straddles = {"ci_low": -0.034, "ci_high": 3.85, "confidence": 0.95}
    assert format_interval(straddles, unit="£") == "95% CI -£0.03 to £3.85"


# --------------------------------------------------------------------------
# Weekday decomposition
# --------------------------------------------------------------------------

def test_weekday_centre_removes_the_weekday_differences():
    # Two weeks: Mondays average 10, Tuesdays 20. After centring both are 15.
    values = [10.0, 20.0, 10.0, 20.0]
    weekdays = [0, 1, 0, 1]
    assert weekday_centre(values, weekdays) == pytest.approx([15.0, 15.0, 15.0, 15.0])


def test_weekday_centre_keeps_the_overall_level_and_within_weekday_spread():
    values = np.array([10.0, 30.0, 12.0, 26.0])
    weekdays = [0, 1, 0, 1]
    out = weekday_centre(values, weekdays)
    assert out.mean() == pytest.approx(values.mean())           # level preserved
    assert out[2] - out[0] == pytest.approx(values[2] - values[0])  # within-weekday gaps kept


def test_weekday_centre_length_mismatch():
    with pytest.raises(ValueError):
        weekday_centre([1.0, 2.0], [0])


def test_weekday_adjusted_trend_recovers_a_planted_slope():
    """Revenue = 100 + 2*t plus a fixed weekday offset: the slope must come back as 2."""
    n = 56
    t = np.arange(n)
    weekdays = t % 7
    offsets = np.array([0.0, 5.0, -5.0, 10.0, 20.0, 40.0, 30.0])
    y = 100 + 2 * t + offsets[weekdays]

    fit = weekday_adjusted_trend(y, weekdays)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["p_value"] < 1e-9            # noiseless, so overwhelmingly significant
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["df"] == n - 8               # intercept + slope + 6 weekday dummies


def test_weekday_pattern_alone_is_not_reported_as_a_trend():
    """
    The failure mode this function exists for.

    Four weeks plus one extra day, so the window holds five Mondays and four of
    every other weekday. Revenue depends only on the weekday -- there is no
    trend whatever -- but the unbalanced window makes a plain straight-line fit
    slope anyway. The adjusted fit must find nothing.
    """
    weekdays = np.arange(29) % 7
    offsets = np.array([0.0, 10.0, 20.0, 30.0, 60.0, 90.0, 70.0])
    y = 100 + offsets[weekdays]

    plain = stats.linregress(np.arange(29), y)
    assert abs(plain.slope) > 0.3  # the calendar artefact is real and sizeable

    fit = weekday_adjusted_trend(y, weekdays)
    assert fit["slope"] == 0.0
    assert fit["p_value"] == 1.0
    assert fit["r_squared"] == 0.0

    # The same thing with noise on top, so the fit is not numerically exact and
    # the ordinary t-test path is the one being exercised.
    noisy = y + np.random.default_rng(5).normal(0, 5, 29)
    noisy_fit = weekday_adjusted_trend(noisy, weekdays)
    assert noisy_fit["p_value"] > 0.05


def test_pure_noise_is_not_called_a_trend():
    """
    The test's false-positive rate is the one it claims: about 5%.

    400 seeded draws of 20 days of pure noise. If the degrees of freedom were
    wrong -- n - 2 rather than n - 8 with six weekday dummies in the model --
    the rate would run visibly above 0.05 and the detector would report trends
    that are not there. Deterministic, because the generator is seeded.
    """
    rng = np.random.default_rng(1234)
    weekdays = np.arange(20) % 7
    hits = sum(
        weekday_adjusted_trend(50 + rng.normal(0, 1.0, 20), weekdays)["p_value"] < 0.05
        for _ in range(400)
    )
    assert 0.01 <= hits / 400 <= 0.10


def test_weekday_adjusted_trend_charges_the_right_degrees_of_freedom():
    """
    The slope's p-value must be computed against n - 8 df once six weekday
    dummies are in the model, not the n - 2 a two-variable fit would assume.
    Checked by recomputing the p-value from the reported t statistic.
    """
    rng = np.random.default_rng(3)
    n = 60
    t = np.arange(n)
    weekdays = t % 7
    y = 100 + 0.5 * t + rng.normal(0, 10, n)

    fit = weekday_adjusted_trend(y, weekdays)
    assert fit["df"] == n - 8
    expected = 2 * stats.t.sf(abs(fit["t_stat"]), n - 8)
    assert fit["p_value"] == pytest.approx(expected)
    # The same t against the looser n - 2 df would give a smaller p-value.
    assert 2 * stats.t.sf(abs(fit["t_stat"]), n - 2) < fit["p_value"]


def test_weekday_adjusted_trend_uses_the_day_index_it_is_given():
    """
    A bootstrap resample hands the days over out of order, each carrying its own
    day index. The fit must use those indices, not the row positions, or the
    trend it measures is the order the rows happened to be drawn in.
    """
    n = 28
    t = np.arange(n)
    weekdays = t % 7
    y = 100 + 3.0 * t

    shuffle = np.random.default_rng(11).permutation(n)
    with_index = weekday_adjusted_trend(y[shuffle], weekdays[shuffle], t[shuffle])
    assert with_index["slope"] == pytest.approx(3.0)

    without_index = weekday_adjusted_trend(y[shuffle], weekdays[shuffle])
    assert without_index["slope"] != pytest.approx(3.0)


def test_weekday_adjusted_trend_handles_degenerate_input():
    assert weekday_adjusted_trend([1.0, 2.0], [0, 1])["p_value"] == 1.0   # n < 3
    flat = weekday_adjusted_trend([5.0] * 14, list(np.arange(14) % 7))
    assert flat["slope"] == pytest.approx(0.0) and flat["r_squared"] == 0.0
    # Not enough days to identify a slope plus six dummies: 8 parameters, 8 days.
    short = weekday_adjusted_trend(list(range(8)), list(np.arange(8) % 7))
    assert short["df"] == 0 and short["p_value"] == 1.0


def test_weekday_adjusted_trend_length_mismatch():
    with pytest.raises(ValueError):
        weekday_adjusted_trend([1.0, 2.0, 3.0], [0, 1])
    with pytest.raises(ValueError):
        weekday_adjusted_trend([1.0, 2.0, 3.0], [0, 1, 2], [0, 1])


def test_weekday_adjusted_trend_with_a_single_weekday():
    """Only Mondays in the data: no dummies to add, so it is a plain fit."""
    y = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
    fit = weekday_adjusted_trend(y, [0] * 5)
    assert fit["slope"] == pytest.approx(2.0)
    assert fit["df"] == 3  # n - intercept - slope


# --------------------------------------------------------------------------
# The dip effect size, shared by the point estimate and the bootstrap
# --------------------------------------------------------------------------

def test_dip_effect_size_matches_its_definition():
    hour_means = [10.0, 12.0, 14.0, 16.0]   # mean 13, sd(ddof=1) = 2.5819889
    dip_means = [8.0, 10.0]                 # mean 9
    expected = abs(13.0 - 9.0) / np.std(hour_means, ddof=1)
    assert dip_effect_size(hour_means, dip_means) == pytest.approx(expected)


def test_dip_effect_size_is_zero_when_there_is_no_spread_to_divide_by():
    assert dip_effect_size([5.0, 5.0, 5.0], [1.0]) == 0.0
    assert dip_effect_size([5.0], [1.0]) == 0.0      # sd undefined on one hour
    assert dip_effect_size([5.0, 7.0], []) == 0.0


def test_family_alpha_is_the_conventional_cut_off():
    assert FAMILY_ALPHA == 0.05
