"""Generate templated insights from analysis and confidence scores."""

from typing import Dict, List, Any
from src.analysis import analyze_all
from src.confidence import compute_confidence, score_to_band, confidence_basis


class Insight:
    """A single insight with text, confidence score, and metadata."""

    def __init__(
        self,
        title: str,
        text: str,
        confidence_score: int,
        basis: str,
        suggested_action: str,
        pattern_type: str,
    ):
        """
        Args:
            title: Brief insight title (e.g., "Afternoon Dip Detected").
            text: Human-readable insight text.
            confidence_score: 0–100 confidence.
            basis: Explanation of the confidence (e.g., "6 weeks of data...").
            suggested_action: What the café should consider doing.
            pattern_type: Type of pattern (e.g., "afternoon_dip", "trend", "anomaly").
        """
        self.title = title
        self.text = text
        self.confidence_score = confidence_score
        self.confidence_band = score_to_band(confidence_score)
        self.basis = basis
        self.suggested_action = suggested_action
        self.pattern_type = pattern_type
        self.status = "pending"  # pending, verified, dismissed

    def to_dict(self) -> Dict[str, Any]:
        """Return insight as dictionary."""
        return {
            "title": self.title,
            "text": self.text,
            "confidence_score": self.confidence_score,
            "confidence_band": self.confidence_band,
            "basis": self.basis,
            "suggested_action": self.suggested_action,
            "pattern_type": self.pattern_type,
            "status": self.status,
        }


def generate_afternoon_dip_insight(
    analysis: Dict[str, Any],
    calendar_days: int,
) -> Insight:
    """
    Generate insight from afternoon dip analysis.

    Args:
        analysis: Output from analyze_all().
        calendar_days: Total calendar days in dataset.

    Returns:
        Insight object.
    """
    dip = analysis["afternoon_dip"]

    if not dip["detected"]:
        return None  # No dip detected

    # Template-based phrasing selected by confidence band
    pct_lower = round(
        (dip["overall_avg_revenue"] - dip["dip_avg_revenue"]) /
        dip["overall_avg_revenue"] * 100
    )

    score = compute_confidence(
        sample_size=dip["sample_size"],
        consistency_ratio=dip["sample_size"] / calendar_days,
        effect_size=dip["effect_size"],
        p_value=dip["p_value"],
    )

    band = score_to_band(score)
    basis = confidence_basis(dip["sample_size"], dip["sample_size"] / calendar_days, calendar_days)

    if band == "HIGH":
        text = (
            f"Strong afternoon dip detected (14:00–16:00). Sales are ~{pct_lower}% lower "
            f"during these hours vs. the daily average. This pattern is consistent and statistically significant."
        )
        action = (
            "Consider reducing staff during 14:00–16:00, or running a targeted promotion "
            "(e.g., afternoon specials) to boost sales."
        )
    elif band == "MEDIUM":
        text = (
            f"Afternoon dip detected (14:00–16:00). Sales drop ~{pct_lower}% below daily average. "
            f"Pattern is present but less pronounced."
        )
        action = (
            "Monitor this pattern closely. A small promotion or adjustment might help, "
            "but gather more data before major staffing changes."
        )
    else:  # LOW
        text = (
            f"Possible afternoon dip (14:00–16:00) observed (~{pct_lower}% below average), "
            "but the pattern is weak or inconsistent."
        )
        action = "Collect more data before taking action."

    return Insight(
        title="Afternoon Dip",
        text=text,
        confidence_score=score,
        basis=basis,
        suggested_action=action,
        pattern_type="afternoon_dip",
    )


def generate_trend_insight(
    analysis: Dict[str, Any],
    calendar_days: int,
) -> Insight:
    """
    Generate insight from trend analysis.

    Args:
        analysis: Output from analyze_all().
        calendar_days: Total calendar days in dataset.

    Returns:
        Insight object or None if no significant trend.
    """
    trend = analysis["trend"]

    if not trend["detected"]:
        return None

    direction = trend["trend_direction"]

    score = compute_confidence(
        sample_size=trend["sample_size"],
        consistency_ratio=1.0,  # Trend is continuous if detected
        effect_size=abs(trend["slope_pct"]) / 10,  # Scale percentage to effect-like metric
        p_value=trend["p_value"],
    )

    basis = confidence_basis(trend["sample_size"], 1.0, calendar_days)

    if direction == "upward":
        text = (
            f"Revenue is trending upward at ~£{trend['slope']:.2f} per day "
            f"({trend['slope_pct']:.2f}% growth). This suggests improving sales or seasonal growth."
        )
        action = "Capitalize on momentum. Ensure you're meeting demand (inventory, staffing)."
    else:
        text = (
            f"Revenue is trending downward at ~£{trend['slope']:.2f} per day "
            f"({trend['slope_pct']:.2f}% decline). This may indicate declining interest or external factors."
        )
        action = "Investigate the cause (menu, pricing, competition, events). Consider promotions."

    return Insight(
        title=f"{direction.capitalize()} Revenue Trend",
        text=text,
        confidence_score=score,
        basis=basis,
        suggested_action=action,
        pattern_type="trend",
    )


def generate_anomaly_insights(
    analysis: Dict[str, Any],
    calendar_days: int,
) -> List[Insight]:
    """
    Generate insights from anomaly analysis.

    Args:
        analysis: Output from analyze_all().
        calendar_days: Total calendar days in dataset.

    Returns:
        List of Insight objects (may be empty if no anomalies).
    """
    anomalies = analysis["anomalies"]

    if not anomalies["detected"] or anomalies["anomaly_count"] == 0:
        return []

    score = compute_confidence(
        sample_size=calendar_days,
        consistency_ratio=anomalies["anomaly_count"] / calendar_days,
        effect_size=anomalies["mean_z_score"] * 0.3,
        p_value=0.001 if anomalies["anomaly_count"] > 0 else 1.0,
    )

    basis = confidence_basis(
        calendar_days,
        anomalies["anomaly_count"] / calendar_days,
        calendar_days,
    )

    text = (
        f"{anomalies['anomaly_count']} anomalous days detected out of {calendar_days} "
        f"({anomalies['anomaly_ratio']:.1%}). These days had unusually high or low sales. "
        f"Dates: {', '.join(anomalies['anomalous_dates'][:3])}"
        f"{' ...' if len(anomalies['anomalous_dates']) > 3 else ''}."
    )

    action = (
        "Review what happened on these dates (events, weather, staffing changes, "
        "promos). Identify patterns to prevent future dips or understand what drives spikes."
    )

    return [Insight(
        title="Sales Anomalies",
        text=text,
        confidence_score=score,
        basis=basis,
        suggested_action=action,
        pattern_type="anomalies",
    )]


def generate_all_insights(
    df: Any,  # DataFrame
    calendar_days: int = None,
) -> List[Insight]:
    """
    Run analysis and generate all available insights.

    Args:
        df: Sales data from load_sales_data().
        calendar_days: Total calendar days (if None, computed from df).

    Returns:
        List of Insight objects, filtered to only meaningful ones.
    """
    if calendar_days is None:
        calendar_days = (df["date"].max() - df["date"].min()).days + 1

    analysis = analyze_all(df)
    insights = []

    # Afternoon dip
    dip_insight = generate_afternoon_dip_insight(analysis, calendar_days)
    if dip_insight is not None:
        insights.append(dip_insight)

    # Trend
    trend_insight = generate_trend_insight(analysis, calendar_days)
    if trend_insight is not None:
        insights.append(trend_insight)

    # Anomalies
    anomaly_insights = generate_anomaly_insights(analysis, calendar_days)
    insights.extend(anomaly_insights)

    return insights
