"""Transparent confidence scoring for insights based on statistical rigor.

This module implements a rule-based confidence formula that combines sample size,
consistency, effect size, and statistical significance into a 0–100 confidence score.
Every score is auditable and explainable—no black-box models.
"""

from typing import Tuple


def compute_confidence(
    sample_size: int,
    consistency_ratio: float,
    effect_size: float,
    p_value: float,
) -> int:
    """
    Compute insight confidence score (0–100) from statistical evidence.

    The formula combines four components:
    1. Sample size (more data reduces uncertainty)
    2. Consistency (pattern observed frequently)
    3. Effect size (magnitude of the effect)
    4. Statistical significance (p-value)

    FORMULA:
    --------
    score = (sample_weight × size_score) +
            (consistency_weight × consistency_score) +
            (effect_weight × effect_score) +
            (pvalue_weight × p_value_score)

    Where each component is normalized to [0, 100]:

    - size_score: min(100, (sample_size / 30) × 100)
      → Plateaus at 30 samples (≥1 month of data)

    - consistency_score: consistency_ratio × 100
      → 0–100 based on fraction of periods the pattern was observed

    - effect_score: min(100, effect_size × 80)
      → Cohen's d interpretation: small=0.2, medium=0.5, large=0.8
      → Plateaus at effect_size ≥ 1.25

    - p_value_score: 0 if p ≥ 0.05, else (1 – p) × 100
      → Only non-zero if statistically significant (p < 0.05)

    Weights:
    - sample_weight: 0.25 (ensures sample size matters but doesn't dominate)
    - consistency_weight: 0.35 (human-readable: "seen on X% of days")
    - effect_weight: 0.25 (practical significance)
    - pvalue_weight: 0.15 (statistical rigor, but not dominant)

    Args:
        sample_size: Number of days or observations (≥1).
        consistency_ratio: Fraction of periods pattern was observed (0–1).
        effect_size: Magnitude of effect (e.g., Cohen's d; ≥0).
        p_value: Statistical significance (0–1; typically from t-test or linregress).

    Returns:
        Confidence score 0–100 (int).

    Raises:
        ValueError: If inputs are out of expected ranges.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be ≥1")
    if not (0 <= consistency_ratio <= 1):
        raise ValueError("consistency_ratio must be in [0, 1]")
    if effect_size < 0:
        raise ValueError("effect_size must be ≥0")
    if not (0 <= p_value <= 1):
        raise ValueError("p_value must be in [0, 1]")

    # Component 1: Sample Size
    # 30 days of data = full marks; scale sub-linearly before that
    size_score = min(100, (sample_size / 30) * 100)

    # Component 2: Consistency
    # How often was the pattern observed (0–100%)
    consistency_score = consistency_ratio * 100

    # Component 3: Effect Size
    # Cohen's d scale: 0.2 (small), 0.5 (medium), 0.8 (large)
    # Cap at 1.25 for a full 100 points (e.g., >1.25 is "very large")
    effect_score = min(100, effect_size * 80)

    # Component 4: P-value (Statistical Significance)
    # Only award points if p < 0.05 (conventional threshold)
    if p_value >= 0.05:
        p_value_score = 0
    else:
        p_value_score = (1 - p_value) * 100

    # Weighted combination
    sample_weight = 0.25
    consistency_weight = 0.35
    effect_weight = 0.25
    pvalue_weight = 0.15

    score = (
        sample_weight * size_score +
        consistency_weight * consistency_score +
        effect_weight * effect_score +
        pvalue_weight * p_value_score
    )

    return int(round(score))


def score_to_band(score: int) -> str:
    """
    Map confidence score to a qualitative band.

    - HIGH: ≥85 (strong evidence, actionable)
    - MEDIUM: 60–84 (reasonable evidence, caution advised)
    - LOW: <60 (weak evidence, hold for verification)

    Args:
        score: Confidence score (0–100).

    Returns:
        Band name: "HIGH", "MEDIUM", or "LOW".
    """
    if score >= 85:
        return "HIGH"
    elif score >= 60:
        return "MEDIUM"
    else:
        return "LOW"


def confidence_basis(
    sample_size: int,
    consistency_ratio: float,
    calendar_days: int = None,
) -> str:
    """
    Generate a human-readable explanation of the confidence score basis.

    Args:
        sample_size: Number of trading days with observations.
        consistency_ratio: Fraction of days the pattern was observed (0–1).
        calendar_days: Optionally, total calendar days in dataset (for context).

    Returns:
        A one-sentence explanation, e.g.:
        "6 weeks of data, consistent across 28 of 30 weekdays."
    """
    weeks = sample_size / 7
    consistent_days = int(consistency_ratio * sample_size)

    if calendar_days is None:
        return f"{weeks:.1f} weeks of data, consistent across {consistent_days}/{sample_size} days"
    else:
        return (
            f"{weeks:.1f} weeks of data ({calendar_days} calendar days), "
            f"consistent across {consistent_days}/{sample_size} trading days"
        )


# Example use in testing and documentation
if __name__ == "__main__":
    # Example: Afternoon dip detected with moderate confidence
    score1 = compute_confidence(
        sample_size=30,
        consistency_ratio=0.90,
        effect_size=0.25,
        p_value=0.001,
    )
    print(f"Example 1 (Afternoon dip): {score1} ({score_to_band(score1)})")
    print(f"  Basis: {confidence_basis(30, 0.90, 92)}")
    print()

    # Example: Weak trend with low sample size
    score2 = compute_confidence(
        sample_size=10,
        consistency_ratio=0.60,
        effect_size=0.10,
        p_value=0.10,
    )
    print(f"Example 2 (Weak trend): {score2} ({score_to_band(score2)})")
    print(f"  Basis: {confidence_basis(10, 0.60)}")
    print()

    # Example: Strong anomaly pattern
    score3 = compute_confidence(
        sample_size=60,
        consistency_ratio=1.0,
        effect_size=2.0,
        p_value=0.001,
    )
    print(f"Example 3 (Strong anomaly): {score3} ({score_to_band(score3)})")
    print(f"  Basis: {confidence_basis(60, 1.0, 92)}")
