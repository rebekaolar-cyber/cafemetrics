"""Shared statistical machinery for the detectors in :mod:`src.analysis`.

Three things live here, kept out of ``analysis.py`` so each has its own tests:

1. :func:`benjamini_hochberg` — false-discovery-rate correction, because
   ``analyze_all()`` runs more than one hypothesis test on a single dataset.
2. :func:`bootstrap_ci` — a percentile bootstrap with a fixed, documented seed,
   so every point estimate the app reports can be reported with an interval.
3. :func:`weekday_adjusted_trend` — a trend test that removes the day-of-week
   effect before asking whether revenue is drifting, because a dataset that
   merely starts on a Monday and ends on a Sunday can otherwise show a slope.

Nothing here touches pandas indexing conventions or the shape of the detector
dicts; it all works on plain arrays so the maths can be checked in isolation.
"""

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

# --- Reproducibility -------------------------------------------------------
# Every bootstrap in this project draws from a generator seeded with this
# constant, so two runs on the same data give byte-identical intervals and a
# reviewer can reproduce any published number. Changing it changes published
# intervals, so treat it as part of the released configuration.
BOOTSTRAP_SEED: int = 20260401
# 2000 resamples: the Monte-Carlo error on a 95% percentile bound is well under
# a percentage point of the interval width at this size, and the whole suite
# still runs in about a second. Raise it, don't lower it.
BOOTSTRAP_RESAMPLES: int = 2000
# Nominal coverage of every interval this module reports.
BOOTSTRAP_CONFIDENCE: float = 0.95
# The false-discovery rate the project controls at. One constant, so the cut-off
# used by the detectors, the confidence score and the written answers cannot
# fall out of step.
FAMILY_ALPHA: float = 0.05


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini–Hochberg adjusted p-values (FDR-controlling) for a family of tests.

    Running several tests on one dataset and reporting whichever comes out below
    0.05 inflates the chance of a false finding. BH rescales each p-value so that
    comparing the *adjusted* value against 0.05 controls the expected proportion
    of false discoveries among the findings reported, rather than the chance of
    any single false positive.

    The adjustment, for a family of ``m`` p-values sorted ascending with ranks
    ``i = 1..m``::

        q_(i) = min over j >= i of  min(1, p_(j) * m / j)

    The outer minimum enforces monotonicity, so a smaller raw p-value can never
    come out with a larger adjusted one.

    Args:
        p_values: The family's raw p-values, each in [0, 1]. Order is preserved
            in the result. An empty family returns an empty list.

    Returns:
        Adjusted p-values in the same order as the input, each in [0, 1].

    Raises:
        ValueError: If any p-value is NaN or outside [0, 1].

    Example:
        >>> [round(q, 4) for q in benjamini_hochberg([0.01, 0.04, 0.03])]
        [0.03, 0.04, 0.04]
    """
    raw = np.asarray(p_values, dtype=float)
    if raw.size == 0:
        return []
    if np.isnan(raw).any() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("p_values must all be numbers in [0, 1]")

    m = raw.size
    order = np.argsort(raw, kind="stable")  # ascending
    ranked = raw[order] * m / np.arange(1, m + 1)
    # Walk down from the largest rank so each value is capped by the one above it.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.minimum(ranked, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = ranked
    return [float(value) for value in adjusted]


def correct_family(named_p_values: Dict[str, float]) -> Dict[str, float]:
    """
    Apply :func:`benjamini_hochberg` to a named family of tests.

    Convenience wrapper so callers can keep track of which test each adjusted
    p-value belongs to.

    Args:
        named_p_values: Mapping of test name to raw p-value.

    Returns:
        Mapping of the same names to adjusted p-values.
    """
    names = list(named_p_values)
    adjusted = benjamini_hochberg([named_p_values[name] for name in names])
    return dict(zip(names, adjusted))


def bootstrap_ci(
    statistic: Callable[[np.random.Generator], float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CONFIDENCE,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, float]:
    """
    Percentile bootstrap interval for any statistic, with a fixed seed.

    The caller supplies a function that draws one resample and returns the
    statistic computed on it; this function only handles the seeding, the
    repetition and the percentiles. That keeps the resampling *unit* (days,
    rows, pairs) a decision of the detector that owns the data.

    Args:
        statistic: Called once per resample with a seeded ``numpy`` generator;
            must return a float, or NaN for a resample it cannot evaluate
            (a degenerate draw, say). NaN draws are discarded.
        n_resamples: Number of resamples. Default :data:`BOOTSTRAP_RESAMPLES`.
        confidence: Nominal coverage, e.g. 0.95 for a 95% interval.
        seed: Seed for the generator. Default :data:`BOOTSTRAP_SEED`. With the
            same data, statistic, seed and resample count, the interval is
            identical between runs — the tests assert this.

    Returns:
        Dict with keys:
        - ci_low, ci_high: the interval bounds (NaN if nothing could be computed)
        - confidence: the nominal coverage used
        - n_resamples: the number of resamples requested
        - n_effective: how many resamples produced a usable value

    Raises:
        ValueError: If n_resamples < 1 or confidence is not in (0, 1).
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")

    rng = np.random.default_rng(seed)
    draws = np.array([statistic(rng) for _ in range(n_resamples)], dtype=float)
    usable = draws[~np.isnan(draws)]

    if usable.size == 0:
        low = high = float("nan")
    else:
        tail = (1 - confidence) / 2 * 100
        low, high = np.percentile(usable, [tail, 100 - tail])

    return {
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": float(confidence),
        "n_resamples": int(n_resamples),
        "n_effective": int(usable.size),
    }


def weekday_centre(values: Sequence[float], weekdays: Sequence[int]) -> np.ndarray:
    """
    Remove the day-of-week effect by centring each day on its own weekday mean.

    Each value has its weekday's mean subtracted and the grand mean added back,
    so the series keeps its original level but loses the weekly rhythm::

        residual_i = y_i - mean(y over days with the same weekday) + mean(y)

    Args:
        values: One number per day (e.g. daily revenue).
        weekdays: The weekday of each day, same length as values. Any labels
            that compare equal will do; 0–6 is the convention here.

    Returns:
        The weekday-centred values, as a float array.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    y = np.asarray(values, dtype=float)
    w = np.asarray(weekdays)
    if y.shape[0] != w.shape[0]:
        raise ValueError("values and weekdays must have the same length")

    residual = y.astype(float).copy()
    for label in np.unique(w):
        mask = w == label
        residual[mask] = y[mask] - y[mask].mean()
    return residual + y.mean()


def weekday_adjusted_trend(
    values: Sequence[float],
    weekdays: Sequence[int],
    day_index: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """
    Test for a linear trend in a daily series after removing the weekly pattern.

    Fits one ordinary least-squares model with an intercept, a day index and
    day-of-week dummy variables (one weekday dropped as the reference level)::

        y_t = a + b*t + sum_k c_k * [weekday_t == k] + e_t

    and reports the slope ``b``. Two things follow from doing it in one model
    rather than regressing on a centred series:

    - the slope's standard error is computed with the degrees of freedom the
      weekday terms actually consume (``n - 2 - number_of_weekday_dummies``),
      so the p-value is not optimistic; and
    - ``r_squared`` is the *partial* R² of the day index: the share of the
      variation still left once the weekly pattern is accounted for. That is the
      number worth quoting, because the weekly pattern is not the trend.

    Why bother: a 92-day window that starts on a Wednesday and ends on a Tuesday
    contains unequal counts of each weekday, so a strong weekly rhythm alone can
    tilt an unadjusted regression line. Removing the weekday effect first means a
    reported trend is a drift in the level of business, not an artefact of which
    weekdays the window happens to begin and end on.

    Args:
        values: Daily totals, one per day, in time order.
        weekdays: Weekday label (0–6) for each day, same length and order.
        day_index: The time variable ``t``, one per day. Defaults to
            ``0..n-1``, i.e. the position of each day in the series. Pass it
            explicitly when the rows are not in time order — a bootstrap
            resample, for instance, must carry each day's original index with
            it, or the trend being measured is destroyed.

    Returns:
        Dict with keys slope, p_value, r_squared (partial, see above), stderr,
        t_stat, df, n. If the fit is not identified (too few days, a constant
        series, or no residual degrees of freedom) the slope is 0.0, the p-value
        1.0 and r_squared 0.0. A series the model fits exactly, with no residual
        left, has no usable t-statistic; it is reported as p = 0.0 if its slope
        is of a meaningful size and p = 1.0 if the slope is rounding error.

    Raises:
        ValueError: If the sequences have different lengths.
    """
    y = np.asarray(values, dtype=float)
    w = np.asarray(weekdays)
    if y.shape[0] != w.shape[0]:
        raise ValueError("values and weekdays must have the same length")

    n = y.shape[0]
    if day_index is None:
        t = np.arange(n, dtype=float)
    else:
        t = np.asarray(day_index, dtype=float)
        if t.shape[0] != n:
            raise ValueError("day_index must have the same length as values")
    empty = {
        "slope": 0.0, "p_value": 1.0, "r_squared": 0.0,
        "stderr": float("nan"), "t_stat": float("nan"), "df": 0, "n": n,
    }
    if n < 3:
        return empty

    levels = np.unique(w)
    # Drop the first weekday as the reference level to keep the design full rank.
    dummies = np.column_stack([(w == level).astype(float) for level in levels[1:]]) \
        if levels.size > 1 else np.empty((n, 0))

    full = np.column_stack([np.ones(n), t, dummies])
    reduced = np.column_stack([np.ones(n), dummies])  # weekday effect only

    df = n - full.shape[1]
    if df < 1:
        return empty

    beta, *_ = np.linalg.lstsq(full, y, rcond=None)
    residual = y - full @ beta
    rss_full = float(residual @ residual)

    beta_reduced, *_ = np.linalg.lstsq(reduced, y, rcond=None)
    residual_reduced = y - reduced @ beta_reduced
    rss_reduced = float(residual_reduced @ residual_reduced)

    # A numerically exact fit (no residual at all) makes the t-statistic 0/0:
    # both the slope and its standard error are then at rounding-error scale and
    # their ratio is meaningless, which can turn a slope of 1e-10 into p < 0.001.
    # Real sales data never does this; a synthetic or perfectly regular series
    # does. Decide such a case on the size of the slope itself instead.
    if rss_full <= 1e-20 * max(1.0, float(y @ y)):
        negligible = abs(float(beta[1])) <= 1e-9 * max(1.0, float(np.abs(y).mean()))
        return {
            "slope": 0.0 if negligible else float(beta[1]),
            "p_value": 1.0 if negligible else 0.0,
            "r_squared": 0.0 if negligible else 1.0,
            "stderr": 0.0,
            "t_stat": 0.0 if negligible else float("inf"),
            "df": int(df),
            "n": int(n),
        }

    sigma2 = rss_full / df
    try:
        cov_unscaled = np.linalg.inv(full.T @ full)
    except np.linalg.LinAlgError:  # pragma: no cover - guarded by the rank checks
        return empty
    stderr = float(np.sqrt(sigma2 * cov_unscaled[1, 1]))

    slope = float(beta[1])
    if not np.isfinite(stderr) or stderr == 0:
        return {**empty, "slope": slope}

    t_stat = slope / stderr
    p_value = float(2 * stats.t.sf(abs(t_stat), df))
    partial_r2 = (rss_reduced - rss_full) / rss_reduced if rss_reduced > 0 else 0.0

    return {
        "slope": slope,
        "p_value": p_value,
        "r_squared": float(max(0.0, min(1.0, partial_r2))),
        "stderr": stderr,
        "t_stat": float(t_stat),
        "df": int(df),
        "n": int(n),
    }


def resample_indices(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    One bootstrap resample of ``n`` row positions, drawn with replacement.

    A one-liner, but naming it keeps every detector's resampling unit visible
    and identical, and keeps the draw order (and so the seed's meaning) in one
    place.

    Args:
        rng: The generator supplied by :func:`bootstrap_ci`.
        n: Number of positions to draw from, i.e. 0..n-1.

    Returns:
        Array of n integer positions.
    """
    return rng.integers(0, n, size=n)


def format_interval(ci: Dict[str, float], unit: str = "", decimals: int = 2) -> str:
    """
    A bootstrap interval written out for display, e.g. '95% CI £0.21 to £3.76'.

    Args:
        ci: A dict from :func:`bootstrap_ci`.
        unit: Prefix for each bound, e.g. "£". Empty for a unitless statistic.
        decimals: Decimal places for each bound.

    Returns:
        The interval as a short phrase, or 'no interval available' if the
        bootstrap could not produce one.
    """
    low, high = ci.get("ci_low"), ci.get("ci_high")
    if low is None or high is None or not (np.isfinite(low) and np.isfinite(high)):
        return "no interval available"
    pct = round(ci.get("confidence", BOOTSTRAP_CONFIDENCE) * 100)

    def bound(value: float) -> str:
        # Sign outside the unit, so a negative pound bound reads '-£0.03'.
        sign = "-" if value < 0 else ""
        return f"{sign}{unit}{abs(value):.{decimals}f}"

    return f"{pct}% CI {bound(low)} to {bound(high)}"


def expected_false_flag_rate(z_threshold: float, window: int) -> float:
    """
    The share of ordinary days a rolling-window z-score rule would flag.

    For a day scored against the ``window`` days *before* it, the difference
    ``x - mean(window)`` has variance ``sigma^2 (1 + 1/window)`` while the
    window's standard deviation estimates ``sigma`` on ``window - 1`` degrees of
    freedom. So under normal, independent daily revenue::

        z  ~  t(window - 1) * sqrt((window + 1) / window)

    and the two-sided tail area beyond ``z_threshold`` is the rate at which the
    rule flags days that are not actually unusual. Those fatter-than-normal
    tails matter a great deal at small windows: with ``window = 7`` a threshold
    of 2.5 flags 5.8% of ordinary days, not the 1.2% that treating it as a
    normal deviate would suggest.

    This is a nominal rate under assumptions real café revenue only roughly
    meets — weekly seasonality alone breaks independence — so it is an order of
    magnitude for interpreting a threshold, not a guarantee.

    Args:
        z_threshold: The cut-off on |z|.
        window: Number of prior days in the reference window (>= 2).

    Returns:
        The two-sided tail probability, in [0, 1]. Returns 1.0 for a
        non-positive threshold and 0.0 if the window is too small to give the
        standard deviation any degrees of freedom.
    """
    if window < 2:
        return 0.0
    if z_threshold <= 0:
        return 1.0
    scale = np.sqrt((window + 1) / window)
    return float(2 * stats.t.sf(z_threshold / scale, window - 1))


def dip_effect_size(hour_means_by_day: Sequence[float], dip_means_by_day: Sequence[float]) -> float:
    """
    The afternoon-dip effect size, exactly as :func:`src.analysis.detect_afternoon_dip` defines it.

    Factored out so the bootstrap and the point estimate cannot drift apart:
    both call this. See that function's docstring for what the statistic does
    and does not mean.

    Args:
        hour_means_by_day: Mean revenue of each (day, hour) cell, grouped to one
            mean per hour-of-day across all days.
        dip_means_by_day: Mean revenue per day within the dip window.

    Returns:
        ``|mean(hour_means) - mean(dip_means)| / sd(hour_means)``, or 0.0 when
        the standard deviation is zero or undefined.
    """
    hour_means = np.asarray(hour_means_by_day, dtype=float)
    dip_means = np.asarray(dip_means_by_day, dtype=float)
    if hour_means.size < 2 or dip_means.size == 0:
        return 0.0
    spread = float(hour_means.std(ddof=1))
    if not np.isfinite(spread) or spread <= 0:
        return 0.0
    return float(abs(hour_means.mean() - dip_means.mean()) / spread)


