"""Answers to the dashboard's three fixed questions, written from the analysis results.

Each answer is a template filled with numbers from analyze_all(), so it can only say what the
data shows: a dip, a trend or outlier days are reported when the rules find them, and
"nothing found" is reported, with the numbers, when they don't.
"""

import inspect
from typing import Any, Dict, List, NamedTuple

from src.analysis import ANOMALY_MIN_DAYS, ANOMALY_WINDOW, detect_anomalies
from src.statistics_support import FAMILY_ALPHA, format_interval

# The cut-offs the rules use, read from where they are defined so the text can't drift.
Z_THRESHOLD: float = inspect.signature(detect_anomalies).parameters["z_threshold"].default
P_CUTOFF = FAMILY_ALPHA


class Answer(NamedTuple):
    """One answer: the verdict in a few words, the finding, and the evidence behind it."""

    title: str
    finding: str
    evidence: str


def format_p(p_value: float) -> str:
    """A p-value as it is usually written: 'p < 0.001' or 'p = 0.066'."""
    return "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"


def format_corrected_p(result: Dict[str, Any]) -> str:
    """
    The raw p-value and the family-corrected one, side by side.

    The corrected value is the one the rules act on, so it is named first; the
    raw one is kept visible because it is the number the test itself produced
    and the number earlier versions of this app displayed.

    Args:
        result: A detector result holding 'p_value', 'p_value_corrected' and
            'family_size'.

    Returns:
        E.g. "p < 0.001 after correcting for the 2 tests run (raw p < 0.001)",
        or just the raw p-value when only one test was run.
    """
    raw, corrected = result["p_value"], result["p_value_corrected"]
    family = result.get("family_size", 1)
    if family <= 1:
        return format_p(raw)
    return (
        f"{format_p(corrected)} after correcting for the {family} tests run "
        f"(raw {format_p(raw)})"
    )


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
    if dip["sample_size"] < 2:
        return Answer(
            "Not enough data to tell",
            f"There are not two days with sales both inside and outside the {window} window, "
            f"so there is nothing to compare. {dip.get('exclusion_note', '')}".strip(),
            "A paired comparison needs at least two days with sales on both sides of the window.",
        )
    dip_avg, overall = dip["dip_avg_revenue"], dip["overall_avg_revenue"]
    gap = abs(overall - dip_avg) / overall * 100 if overall else 0.0
    evidence = (
        f"Compared on {dip['sample_size']} days with a paired t-test, "
        f"{format_corrected_p(dip)}. "
        f"Effect size {dip['effect_size']:.2f} "
        f"({format_interval(dip['effect_size_ci'])}), "
        f"which counts as {effect_word(dip['effect_size'])}. "
        f"The window is compared against every other hour the café is open, "
        f"quiet opening and closing hours included."
    )
    note = dip.get("exclusion_note", "")
    if note:
        evidence = f"{evidence} {note}"
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
        f"Straight-line fit over {days} days with the day-of-week pattern taken out first, "
        f"{format_corrected_p(trend)}. "
        f"Slope {format_interval(trend['slope_ci'], unit='£')} a day. "
        f"The line explains {trend['r_squared']:.0%} of the variation left once the weekly "
        f"pattern is accounted for. Without taking the weekly pattern out the same fit gives "
        f"{format_p(trend['p_value_unadjusted'])}."
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
    scanned = anomalies.get("days_scanned", days)
    rule = (
        f"A day stands out when its revenue is more than {Z_THRESHOLD:g} standard deviations "
        f"from the average of the {ANOMALY_WINDOW} days before it. That day is not counted in "
        f"its own average, so the comparison is with the week leading up to it. On steady "
        f"takings this rule flags about "
        f"{anomalies.get('expected_false_flag_rate', 0.0):.1%} of ordinary days anyway, so "
        f"expect roughly one false alarm every "
        f"{round(1 / anomalies['expected_false_flag_rate']) if anomalies.get('expected_false_flag_rate') else 0} "
        f"days."
    )
    note = anomalies.get("unscored_note", "")
    if note:
        rule = f"{rule} {note}"

    if days < ANOMALY_MIN_DAYS:
        return Answer(
            "Not enough data to tell",
            f"Spotting unusual days needs at least {ANOMALY_MIN_DAYS} days of sales, because "
            f"each day is compared with the {ANOMALY_WINDOW} before it; this data has {days}.",
            rule,
        )
    if scanned == 0:
        return Answer(
            "Not enough data to tell",
            f"None of the {days} days could be checked: {note or 'no day had a usable week before it.'}",
            rule,
        )
    if count == 0:
        return Answer(
            "No day stands out",
            f"None of the {scanned} days checked is far enough from the week before it to count "
            f"as unusual.",
            rule,
        )
    dates = ", ".join(anomalies["anomalous_dates"])
    return Answer(
        f"{count} of {scanned} days stand{'s' if count == 1 else ''} out",
        f"Unusually high or low revenue on: {dates}. Check what happened on those dates "
        "(an event, the weather, staffing, a promotion).",
        rule,
    )
