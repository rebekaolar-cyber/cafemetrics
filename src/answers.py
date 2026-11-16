"""Answers to the dashboard's three fixed questions, written from the analysis results.

Each answer is a template filled with numbers from analyze_all(), so it can only say what the
data shows: a dip, a trend or outlier days are reported when the rules find them, and
"nothing found" is reported, with the numbers, when they don't.
"""

import inspect
from typing import Any, Dict, List, NamedTuple

from src.analysis import detect_anomalies

# The cut-offs the rules use, read from where they are defined so the text can't drift.
Z_THRESHOLD: float = inspect.signature(detect_anomalies).parameters["z_threshold"].default
P_CUTOFF = 0.05


class Answer(NamedTuple):
    """One answer: the verdict in a few words, the finding, and the evidence behind it."""

    title: str
    finding: str
    evidence: str


def format_p(p_value: float) -> str:
    """A p-value as it is usually written: 'p < 0.001' or 'p = 0.066'."""
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def effect_word(effect_size: float) -> str:
    """Cohen's rule of thumb for an effect size: small, medium or large."""
    if effect_size >= 0.8:
        return "large"
    if effect_size >= 0.5:
        return "medium"
    return "small"


def hour_range(hours: List[int]) -> str:
    """The clock time a run of whole hours covers: [14, 15, 16] -> '14:00 to 17:00'."""
    return f"{min(hours):02d}:00 to {max(hours) + 1:02d}:00"


def dip_answer(dip: Dict[str, Any]) -> Answer:
    """Is there an afternoon dip? From detect_afternoon_dip()."""
    window = hour_range(dip["dip_hours"])
    if dip["sample_size"] == 0:
        return Answer(
            "Not enough data to tell",
            f"There are no sales in the {window} window to compare with the rest of the day.",
            "Needs at least one day with sales both inside and outside the window.",
        )
    dip_avg, overall = dip["dip_avg_revenue"], dip["overall_avg_revenue"]
    gap = abs(overall - dip_avg) / overall * 100 if overall else 0.0
    evidence = (
        f"Compared on {dip['sample_size']} days with a paired t-test, {format_p(dip['p_value'])}. "
        f"Effect size {dip['effect_size']:.2f}, which counts as {effect_word(dip['effect_size'])}."
    )
    if not dip["detected"]:
        return Answer(
            "No afternoon dip",
            f"An average hour from {window} takes £{dip_avg:,.0f}, against £{overall:,.0f} for "
            f"an average hour across the day. That is {gap:.0f}% higher, not lower.",
            evidence,
        )
    return Answer(
        f"Yes, sales dip from {window}",
        f"An average hour in that window takes £{dip_avg:,.0f}, against £{overall:,.0f} for an "
        f"average hour across the day: {gap:.0f}% lower. The data shows that it happens, "
        "not why.",
        evidence,
    )


def trend_answer(trend: Dict[str, Any]) -> Answer:
    """Is revenue rising or falling? From detect_trend()."""
    days = trend["sample_size"]
    if trend["trend_direction"] == "none":
        return Answer(
            "Not enough data to tell",
            f"A trend needs at least 3 days of sales; this data has {days}.",
            "No line was fitted.",
        )
    rising = trend["slope"] > 0
    per_day = f"£{abs(trend['slope']):,.2f} a day"
    share = f"{abs(trend['slope_pct']):.2f}% of an average day"
    evidence = (
        f"Straight-line fit over {days} days, {format_p(trend['p_value'])}. "
        f"The line explains {trend['r_squared']:.0%} of the day-to-day variation."
    )
    if not trend["detected"]:
        return Answer(
            "No clear trend",
            f"A straight line through daily revenue {'rises' if rising else 'falls'} by {per_day} "
            f"({share}), but a change that small could be ordinary day-to-day variation.",
            evidence + f" A trend is only reported below p = {P_CUTOFF}.",
        )
    return Answer(
        f"Revenue is {'rising' if rising else 'falling'}",
        f"Daily revenue {'rises' if rising else 'falls'} by about {per_day} ({share}).",
        evidence,
    )


def anomaly_answer(anomalies: Dict[str, Any]) -> Answer:
    """Did any day stand out? From detect_anomalies()."""
    days, count = anomalies["sample_size"], anomalies["anomaly_count"]
    rule = (
        f"A day stands out when its revenue is more than {Z_THRESHOLD:g} standard deviations "
        "from the average of the 7 days up to it."
    )
    if days < 7:
        return Answer(
            "Not enough data to tell",
            f"Spotting unusual days needs at least 7 days of sales; this data has {days}.",
            rule,
        )
    if count == 0:
        return Answer(
            "No day stands out",
            f"None of the {days} days is far enough from its recent average to count as unusual.",
            rule,
        )
    dates = ", ".join(anomalies["anomalous_dates"])
    return Answer(
        f"{count} of {days} days stand{'s' if count == 1 else ''} out",
        f"Unusually high or low revenue on: {dates}. Check what happened on those dates "
        "(an event, the weather, staffing, a promotion).",
        rule,
    )
