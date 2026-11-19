"""Detect recurring patterns: dips, trends, anomalies.

Statistical conventions used throughout this module
---------------------------------------------------
**Family-wise correction.** ``analyze_all()`` runs more than one hypothesis test
on the same dataset, so a raw p-value overstates how surprising the smallest of
them is. Every detector therefore reports two p-values: ``p_value``, the raw one
from its own test, and ``p_value_corrected``, the Benjamini–Hochberg adjusted
one computed across the family of tests actually run. The *corrected* value
drives both the ``detected`` flag and the confidence score; the raw value is
kept and displayed so nothing the UI used to show silently changes meaning.
A detector called on its own is a family of one, so the two are then equal and
``family_size`` says so.

**Intervals.** Point estimates (the dip effect size, the trend slope) come with
a 95% percentile bootstrap interval, seeded from
``statistics_support.BOOTSTRAP_SEED`` so the published numbers are reproducible.

**The anomaly scan is not in the family.** ``detect_anomalies()`` is a
thresholding rule on rolling z-scores, not a hypothesis test, and produces no
p-value. Inventing one so it could be corrected would be worse than leaving it
out, so the family is the tests that genuinely have p-values.
"""

import warnings

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple

from src.statistics_support import (
    BOOTSTRAP_CONFIDENCE,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FAMILY_ALPHA,
    bootstrap_ci,
    correct_family,
    dip_effect_size,
    expected_false_flag_rate,
    resample_indices,
    weekday_adjusted_trend,
)

# Detectors that produce a p-value, and so take part in the BH correction.
P_VALUE_TESTS: Tuple[str, ...] = ("afternoon_dip", "trend")

# The anomaly scan compares each day against the ANOMALY_WINDOW days before it,
# and needs all of them, so the earliest ANOMALY_WINDOW days cannot be scanned
# and a dataset shorter than ANOMALY_MIN_DAYS cannot be scanned at all.
ANOMALY_WINDOW: int = 7
ANOMALY_MIN_DAYS: int = ANOMALY_WINDOW + 1
# 4.0 flags about 0.96% of ordinary days — roughly one in 104. See
# detect_anomalies.__doc__ for the derivation and for why the previous 2.5 was
# not defensible once the threshold became reachable at all.
DEFAULT_Z_THRESHOLD: float = 4.0


def _clean_p(p_value: Any) -> float:
    """A p-value fit to correct and report: NaN or missing becomes 1.0."""
    if p_value is None:
        return 1.0
    value = float(p_value)
    return 1.0 if not np.isfinite(value) else min(1.0, max(0.0, value))


def detect_afternoon_dip(df: pd.DataFrame, dip_window: Tuple[int, int] = (14, 16)) -> Dict[str, Any]:
    """
    Detect if there's a recurring afternoon dip (lower sales in specific hours).

    WHAT IS COMPARED, AND THE ASSUMPTION THAT CARRIES
    -------------------------------------------------
    The test is a paired t-test over trading days. For each day it takes the
    mean hourly revenue *inside* the window and the mean hourly revenue
    *outside* it, and tests whether the two differ.

    "Outside it" means **every other hour the café is open**, opening and
    closing hours included. That is an assumption, and it is not neutral: the
    first and last trading hours of a café are usually quiet, so including them
    pulls the comparison average down and makes the afternoon window look
    *better* than it would against the middle of the day alone. The dip this
    detector reports is therefore a dip relative to a typical open hour, not a
    dip relative to the café's busy core, and the effect size is conservative
    in that direction. It is left this way on purpose: any narrower comparison
    window (11–17? trading hours above the daily median? the two hours either
    side?) would be a judgement call with nothing in the data to anchor it, and
    a documented assumption beats an arbitrary one. A café with real opening
    hours should set ``dip_window`` and the comparison window from its own
    trading pattern.

    ``effect_size`` is ``|mean(hour means) - mean(dip means)| / sd(hour means)``:
    a standardised difference on the scale of the *between-hour* spread. It is
    read on the Cohen's d scale by ``src/confidence.py``, but note it is not
    Cohen's dz for the paired test above, whose denominator would be the spread
    of the per-day paired differences. The between-hour spread is much the
    larger denominator on data like the sample — 38.70 against 14.28 — so this
    statistic is the smaller and more cautious of the two: 0.18 ("small") where
    dz would give 0.60 ("medium") and lift the dip's confidence score from 79 to
    87, from MEDIUM to HIGH. The cautious one is kept deliberately. Switching
    denominators would promote the only finding in the sample data, which is a
    reason for suspicion, not for switching.

    WHICH DAYS ARE COMPARED
    -----------------------
    Only days with sales both inside and outside the window can be paired, so
    the two series are inner-joined on the date before the test. A day that
    closed before 14:00, or that sold nothing outside the afternoon, is dropped.
    The count of dropped days is reported in ``days_excluded`` and spelled out
    in ``exclusion_note`` rather than being swallowed, and ``sample_size`` is
    the number of days actually compared — so the confidence score rests on the
    days the test used, not on the days the file happened to contain. Every
    average in the result is taken over the paired days too.

    Args:
        df: Sales data with 'date', 'hour' and 'revenue' columns.
        dip_window: Tuple of (start_hour, end_hour) to check for dip (inclusive).

    Returns:
        Dict with keys:
        - detected: bool, True if the window is both lower and significant after
          family correction (p_value_corrected < 0.05)
        - sample_size: number of days actually compared (i.e. paired days)
        - days_in_data: distinct days present in the data
        - days_excluded: days_in_data - sample_size, i.e. days with nothing to
          pair against
        - exclusion_note: a sentence naming how many days were left out and why,
          or "" when none were
        - effect_size: see above (>= 0)
        - effect_size_ci: 95% bootstrap interval for effect_size, from
          bootstrap_ci() resampling whole days
        - p_value: raw p-value from the paired t-test
        - p_value_corrected: Benjamini–Hochberg adjusted p-value across the
          family of tests run; equal to p_value when called on its own
        - family_size: how many tests the correction was computed over
        - dip_hours: list of hours checked
        - dip_avg_revenue: average hourly revenue during dip
        - overall_avg_revenue: average hourly revenue overall
    """
    hourly = df.groupby(["date", "hour"])["revenue"].sum().reset_index()

    dip_start, dip_end = dip_window
    dip_hours = list(range(dip_start, dip_end + 1))
    dip_by_day = hourly[hourly["hour"].isin(dip_hours)].groupby("date")["revenue"].mean()
    non_dip_by_day = hourly[~hourly["hour"].isin(dip_hours)].groupby("date")["revenue"].mean()

    # Only days with sales BOTH inside and outside the window can be paired. A
    # day that closed before 14:00, or sold nothing but afternoon coffee, has
    # nothing to pair with and is dropped. Without this inner join the two
    # series can differ in length (scipy then raises "unequal length arrays")
    # or, worse, be the same length while describing different dates, in which
    # case the paired test silently pairs one day's afternoon with another
    # day's morning. Dropping unpaired days is also the statistically correct
    # move, not merely the safe one: a paired test is defined only on pairs.
    paired_dates = dip_by_day.index.intersection(non_dip_by_day.index).sort_values()
    dip_data = dip_by_day.loc[paired_dates]
    non_dip_data = non_dip_by_day.loc[paired_dates]

    days_in_data = int(hourly["date"].nunique())
    days_excluded = days_in_data - len(paired_dates)
    exclusion_note = (
        ""
        if days_excluded == 0
        else (
            f"{days_excluded} of {days_in_data} days had sales only inside or only outside "
            f"{min(dip_hours):02d}:00–{max(dip_hours) + 1:02d}:00, so there was nothing to "
            f"compare them against; they are left out of the test."
        )
    )

    no_interval = {
        "ci_low": float("nan"), "ci_high": float("nan"),
        "confidence": BOOTSTRAP_CONFIDENCE, "n_resamples": 0, "n_effective": 0,
    }
    # A paired test needs at least two pairs; with one, the t-statistic has no
    # degrees of freedom and scipy returns NaN.
    if len(paired_dates) < 2:
        return {
            "detected": False,
            "sample_size": len(paired_dates),
            "days_in_data": days_in_data,
            "days_excluded": days_excluded,
            "exclusion_note": exclusion_note,
            "effect_size": 0.0,
            "effect_size_ci": no_interval,
            "p_value": 1.0,
            "p_value_corrected": 1.0,
            "family_size": 1,
            "dip_hours": dip_hours,
            "dip_avg_revenue": 0.0,
            "overall_avg_revenue": 0.0,
        }

    # Every average below is taken over the paired days only, so the figures the
    # dashboard shows describe exactly the days the test rested on.
    compared = hourly[hourly["date"].isin(paired_dates)]

    dip_avg = dip_data.mean()
    hour_means = compared.groupby("hour")["revenue"].mean()
    overall_avg = hour_means.mean()

    effect_size = dip_effect_size(hour_means.values, dip_data.values)

    # Paired t-test over the days present in both groups, in the same date order.
    t_stat, p_value = stats.ttest_rel(dip_data.values, non_dip_data.values)
    p_value = _clean_p(p_value)

    # Bootstrap: resample whole trading days, because the day is the unit the
    # paired test pairs on. Each resample rebuilds both the per-hour means and
    # the per-day dip means, so the interval reflects every part of the statistic.
    day_hour = compared.pivot_table(index="date", columns="hour", values="revenue", aggfunc="sum")
    dip_columns = [hour for hour in day_hour.columns if hour in dip_hours]
    matrix = day_hour.to_numpy(dtype=float)
    dip_matrix = day_hour[dip_columns].to_numpy(dtype=float) if dip_columns else None

    def one_resample(rng: np.random.Generator) -> float:
        if dip_matrix is None or matrix.shape[0] < 2:
            return float("nan")
        picks = resample_indices(rng, matrix.shape[0])
        with np.errstate(invalid="ignore"):
            hours = np.nanmean(matrix[picks], axis=0)
            dips = np.nanmean(dip_matrix[picks], axis=1)
        hours = hours[np.isfinite(hours)]
        dips = dips[np.isfinite(dips)]
        if hours.size < 2 or dips.size == 0:
            return float("nan")
        return dip_effect_size(hours, dips)

    with warnings.catch_warnings():
        # Days can trade different hours, so a resample may draw no day at all
        # that traded some hour; nanmean warns "Mean of empty slice" and returns
        # NaN, which the lines above then drop. The behaviour is wanted, the
        # warning is noise, and a real till export would emit thousands of them.
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        effect_ci = bootstrap_ci(one_resample)

    return {
        "detected": bool(dip_avg < overall_avg and p_value < FAMILY_ALPHA),
        "sample_size": len(dip_data),
        "days_in_data": days_in_data,
        "days_excluded": days_excluded,
        "exclusion_note": exclusion_note,
        "effect_size": effect_size,
        "effect_size_ci": effect_ci,
        "p_value": p_value,
        "p_value_corrected": p_value,
        "family_size": 1,
        "dip_hours": dip_hours,
        "dip_avg_revenue": float(dip_avg),
        "overall_avg_revenue": float(overall_avg),
    }


def detect_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect an upward or downward trend in daily revenue, with the weekly pattern removed.

    WHY THE WEEKDAY EFFECT COMES OUT FIRST
    --------------------------------------
    A café's revenue depends strongly on the day of the week. A window of days
    almost never contains exactly equal numbers of each weekday — a 92-day
    window has 14 of one weekday and 13 of the rest — so a plain regression of
    daily revenue on a day index can slope simply because of which weekdays the
    window begins and ends on. That is a calendar artefact, not a change in the
    business.

    So the model fitted here is one OLS with an intercept, a day index and
    day-of-week dummies (see ``statistics_support.weekday_adjusted_trend``).
    ``slope`` and ``p_value`` come from that model, with the degrees of freedom
    the weekday terms consume charged against the standard error. ``r_squared``
    is the day index's *partial* R²: the share of the variation left over once
    the weekly pattern is accounted for.

    The older, unadjusted straight-line fit is kept alongside as
    ``slope_unadjusted`` / ``p_value_unadjusted`` / ``r_squared_unadjusted``, so
    the effect of the adjustment is visible rather than silent.

    Args:
        df: Sales data with 'date' and 'revenue' columns. 'day_of_week' is used
            when present; otherwise the weekday is taken from 'date'.

    Returns:
        Dict with keys:
        - detected: bool, True if the weekday-adjusted slope is significant after
          family correction (p_value_corrected < 0.05)
        - slope: revenue change per day (pounds/day), weekday-adjusted
        - slope_ci: 95% bootstrap interval for slope, resampling whole days
        - slope_pct: slope as a percentage of average daily revenue
        - p_value: raw p-value of the weekday-adjusted slope
        - p_value_corrected: Benjamini–Hochberg adjusted p-value across the
          family; equal to p_value when called on its own
        - family_size: how many tests the correction was computed over
        - r_squared: partial R² of the day index after the weekday effect
        - slope_unadjusted, p_value_unadjusted, r_squared_unadjusted: the plain
          linregress fit, for comparison
        - sample_size: number of days
        - trend_direction: "upward", "downward", or "none"
    """
    daily = df.groupby("date", as_index=False)["revenue"].sum()

    no_interval = {
        "ci_low": float("nan"), "ci_high": float("nan"),
        "confidence": BOOTSTRAP_CONFIDENCE, "n_resamples": 0, "n_effective": 0,
    }
    # This check comes before the weekday lookup on purpose: an empty frame has
    # no usable 'date' dtype, and deriving the weekday from it first raised
    # AttributeError on exactly the empty frames the callers already guard for.
    if len(daily) < 3:
        return {
            "detected": False,
            "slope": 0.0,
            "slope_ci": no_interval,
            "slope_pct": 0.0,
            "p_value": 1.0,
            "p_value_corrected": 1.0,
            "family_size": 1,
            "r_squared": 0.0,
            "slope_unadjusted": 0.0,
            "p_value_unadjusted": 1.0,
            "r_squared_unadjusted": 0.0,
            "sample_size": len(daily),
            "trend_direction": "none",
        }

    if "day_of_week" in df.columns:
        weekday_lookup = df.groupby("date")["day_of_week"].first()
        weekdays = daily["date"].map(weekday_lookup).to_numpy()
    else:
        weekdays = daily["date"].dt.weekday.to_numpy()

    y = daily["revenue"].to_numpy(dtype=float)
    x = np.arange(len(daily), dtype=float)

    # The old, weekday-blind fit, kept for comparison.
    plain = stats.linregress(x, y)

    fit = weekday_adjusted_trend(y, weekdays)
    p_value = _clean_p(fit["p_value"])

    avg_revenue = y.mean()
    slope_pct = (fit["slope"] / avg_revenue * 100) if avg_revenue > 0 else 0.0

    # Bootstrap: resample whole days (day index, revenue, weekday together) and
    # refit the same weekday-adjusted model on each resample.
    def one_resample(rng: np.random.Generator) -> float:
        picks = resample_indices(rng, y.shape[0])
        # Each drawn day keeps its own day index, so the time variable survives
        # the resample; dropping it would measure a trend across draw order.
        drawn = weekday_adjusted_trend(y[picks], weekdays[picks], x[picks])
        # A resample can miss a weekday or lose the fit; those draws are dropped.
        if drawn["df"] < 1 or not np.isfinite(drawn["stderr"]):
            return float("nan")
        return drawn["slope"]

    slope_ci = bootstrap_ci(one_resample)

    return {
        "detected": bool(p_value < FAMILY_ALPHA),
        "slope": float(fit["slope"]),
        "slope_ci": slope_ci,
        "slope_pct": float(slope_pct),
        "p_value": p_value,
        "p_value_corrected": p_value,
        "family_size": 1,
        "r_squared": float(fit["r_squared"]),
        "slope_unadjusted": float(plain.slope),
        "p_value_unadjusted": _clean_p(plain.pvalue),
        "r_squared_unadjusted": float(plain.rvalue ** 2),
        "sample_size": len(daily),
        "trend_direction": "upward" if fit["slope"] > 0 else "downward",
    }


def detect_anomalies(df: pd.DataFrame, z_threshold: float = DEFAULT_Z_THRESHOLD) -> Dict[str, Any]:
    """
    Detect anomalous days: days far from the revenue of the week *before* them.

    A day is anomalous if ``|z| > z_threshold``, where ``z`` is the day's revenue
    measured against the mean and standard deviation of the
    ``ANOMALY_WINDOW`` (7) days immediately preceding it. The day itself is not
    in its own reference window.

    WHY THE DAY IS EXCLUDED FROM ITS OWN WINDOW
    -------------------------------------------
    It used to be included (a trailing ``rolling(7, min_periods=1)``), and that
    made the detector incapable of firing. When a value sits inside the sample it
    is measured against, its standardised deviation is algebraically bounded: for
    ``k`` values the largest possible ``|x - mean| / sd`` is ``(k - 1) / sqrt(k)``,
    which for ``k = 7`` is 2.268. The default threshold was 2.5, above that bound,
    so **no dataset could produce an anomaly** — a single day of a billion pounds
    among days of one pound still scored 2.268. ``src/answers.py`` had always told
    the reader this function used "the average of the 7 days up to it", which is
    the behaviour implemented here; the self-inclusive window was a bug, not a
    deliberately conservative choice.

    WHY THE FIRST SEVEN DAYS ARE NOT SCANNED
    ----------------------------------------
    ``min_periods`` is the full window: a day is scored only if all 7 preceding
    days are present. A shorter window would be worse than it looks. The standard
    deviation of a 1-day window is undefined, a 2-day window gives an estimate on
    one degree of freedom, and — the deciding reason — a threshold only means one
    thing if every day is judged against the same amount of evidence. With a
    variable window the same ``z_threshold`` would imply a different false-flag
    rate on every day, and the number on screen would not be interpretable. So
    the first 7 days are reported as unscanned rather than scanned leniently, and
    ``days_unscored`` and ``unscored_note`` say so: a reader can see that the
    early days were not examined instead of inferring that they were clean.

    WHAT THE THRESHOLD MEANS, AND WHY THE DEFAULT IS 4.0
    ----------------------------------------------------
    Because the day is now outside its reference window, ``z`` is no longer
    bounded — but neither is it a standard normal deviate. If daily revenue were
    normal and independent, ``x - mean_7`` has variance ``sigma^2 (1 + 1/7)``
    while ``sd_7`` estimates ``sigma`` on 6 degrees of freedom, so

        z  ~  t(6) * sqrt(8 / 7)

    whose tails are far fatter than a normal's. The old default of 2.5 would flag
    **5.8% of perfectly ordinary days** — about one day in 17, or five false flags
    in a 92-day period — not the ~1.2% that reading "2.5 standard deviations" as
    a normal deviate suggests. A list of "unusual days" that long is noise, so the
    default is now 4.0, which flags about **0.96% of ordinary days, one in 104**.
    The rate implied by whatever threshold is passed is returned in
    ``expected_false_flag_rate`` so the number is auditable rather than folklore;
    it is a nominal rate resting on the normality and independence assumptions
    above, which real café revenue only roughly satisfies, so treat it as an
    order of magnitude and not a guarantee.

    A FLAT REFERENCE WEEK MAKES A DAY UNSCORABLE
    --------------------------------------------
    If the 7 preceding days have identical revenue, their standard deviation is
    exactly zero and there is no scale against which to judge today. The previous
    code divided by ``sd + 1e-8``, which turned any deviation at all — a penny —
    into an astronomical z-score and an automatic anomaly. That fudge is gone: a
    day whose reference week shows no variation is counted in ``days_unscored``,
    because "the week before gave me no way to measure this" is the honest
    answer, not "this is infinitely unusual".

    This is a thresholding rule, not a hypothesis test: it yields no p-value and
    therefore takes no part in the family-wise correction applied in
    ``analyze_all()``, and the insight built from it is scored without a
    significance component (see ``src/confidence.py``).

    Args:
        df: Sales data with 'date' and 'revenue' columns.
        z_threshold: Z-score threshold for flagging anomalies. See above for what
            a given value implies about the false-flag rate.

    Returns:
        Dict with keys:
        - detected: bool, True if anomalies found
        - anomaly_count: number of anomalous days
        - anomaly_ratio: anomalous days as a fraction of the days *scanned*
        - mean_z_score: average absolute z-score over the days scanned
        - sample_size: number of days in the data
        - days_scanned: days that could be scored (sample_size - days_unscored)
        - days_unscored: days that could not be scored at all
        - days_unscored_early: of those, how many lacked a full week before them
        - days_unscored_flat: of those, how many had a reference week with no
          variation to measure against
        - unscored_note: a sentence naming how many days were not scanned and
          why, or "" when every day was scanned
        - expected_false_flag_rate: the share of ordinary days this threshold
          would flag under the assumptions above
        - anomalous_dates: list of anomalous dates
    """
    daily = df.groupby("date")["revenue"].sum().reset_index()
    total_days = len(daily)
    false_flag_rate = expected_false_flag_rate(z_threshold, ANOMALY_WINDOW)

    if total_days < ANOMALY_MIN_DAYS:
        return {
            "detected": False,
            "anomaly_count": 0,
            "anomaly_ratio": 0.0,
            "mean_z_score": 0.0,
            "sample_size": total_days,
            "days_scanned": 0,
            "days_unscored": total_days,
            "days_unscored_early": total_days,
            "days_unscored_flat": 0,
            "unscored_note": (
                f"No day could be scanned: each one needs the {ANOMALY_WINDOW} days before it, "
                f"so at least {ANOMALY_MIN_DAYS} days of sales are required and this data has "
                f"{total_days}."
            ),
            "expected_false_flag_rate": false_flag_rate,
            "anomalous_dates": [],
        }

    revenue = daily["revenue"]
    # shift(1) first, so the window holds the ANOMALY_WINDOW days BEFORE each day
    # and never the day itself.
    prior = revenue.shift(1)
    window = prior.rolling(window=ANOMALY_WINDOW, min_periods=ANOMALY_WINDOW)
    reference_mean = window.mean()
    reference_std = window.std()

    has_window = reference_mean.notna() & reference_std.notna()
    flat_window = has_window & (reference_std <= 0)
    scorable = has_window & (reference_std > 0)

    z_scores = pd.Series(np.nan, index=daily.index, dtype=float)
    z_scores[scorable] = (
        (revenue[scorable] - reference_mean[scorable]) / reference_std[scorable]
    )

    is_anomalous = scorable & (z_scores.abs() > z_threshold)
    anomaly_count = int(is_anomalous.sum())

    days_scanned = int(scorable.sum())
    days_unscored_early = int((~has_window).sum())
    days_unscored_flat = int(flat_window.sum())
    days_unscored = days_unscored_early + days_unscored_flat

    reasons = []
    if days_unscored_early:
        reasons.append(
            f"{days_unscored_early} had no full week of sales before them"
        )
    if days_unscored_flat:
        reasons.append(
            f"{days_unscored_flat} had a reference week with identical revenue every day, "
            f"leaving no scale to measure against"
        )
    unscored_note = (
        ""
        if days_unscored == 0
        else (
            f"{days_unscored} of {total_days} days were not scanned: "
            + "; ".join(reasons)
            + "."
        )
    )

    return {
        "detected": anomaly_count > 0,
        "anomaly_count": anomaly_count,
        "anomaly_ratio": float(anomaly_count / days_scanned) if days_scanned else 0.0,
        "mean_z_score": float(z_scores.abs().mean()) if days_scanned else 0.0,
        "sample_size": total_days,
        "days_scanned": days_scanned,
        "days_unscored": days_unscored,
        "days_unscored_early": days_unscored_early,
        "days_unscored_flat": days_unscored_flat,
        "unscored_note": unscored_note,
        "expected_false_flag_rate": false_flag_rate,
        "anomalous_dates": daily.loc[is_anomalous, "date"].dt.strftime("%Y-%m-%d").tolist(),
    }


def apply_family_correction(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Correct the p-values of every test in a set of detector results, together.

    Benjamini–Hochberg across the tests in ``P_VALUE_TESTS`` that are present,
    then re-derive each ``detected`` flag from the *corrected* p-value. The raw
    ``p_value`` of each detector is left untouched, so both numbers are
    available to the UI and to ``src/answers.py``.

    The dicts are modified in place and also returned, for convenience.

    Args:
        results: Mapping of detector name to its result dict, as built by
            ``analyze_all()``.

    Returns:
        The same mapping, with ``p_value_corrected`` and ``family_size`` filled
        in and ``detected`` re-derived for every corrected test.
    """
    family = {
        name: _clean_p(results[name].get("p_value", 1.0))
        for name in P_VALUE_TESTS
        if name in results
    }
    if not family:
        return results

    corrected = correct_family(family)
    for name, adjusted in corrected.items():
        result = results[name]
        result["p_value_corrected"] = float(adjusted)
        result["family_size"] = len(family)
        significant = adjusted < FAMILY_ALPHA
        if name == "afternoon_dip":
            lower = result["dip_avg_revenue"] < result["overall_avg_revenue"]
            result["detected"] = bool(result["sample_size"] > 0 and lower and significant)
        elif name == "trend":
            result["detected"] = bool(result["trend_direction"] != "none" and significant)
    return results


def analyze_all(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all analyses and return results for confidence scoring.

    The three detectors are run on one dataset, so their p-values are corrected
    together by ``apply_family_correction()`` before anything is reported: the
    corrected p-value is what decides ``detected`` and what the confidence score
    is computed from, while the raw p-value stays in the result beside it.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Dict with keys: afternoon_dip, trend, anomalies.
    """
    results = {
        "afternoon_dip": detect_afternoon_dip(df),
        "trend": detect_trend(df),
        "anomalies": detect_anomalies(df),
    }
    return apply_family_correction(results)
