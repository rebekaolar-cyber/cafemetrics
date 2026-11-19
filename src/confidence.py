"""Transparent confidence scoring for insights based on statistical rigor.

This module implements a rule-based confidence formula that combines sample size,
consistency, effect size, and statistical significance into a 0–100 confidence score.
Every score is auditable and explainable—no black-box models.
"""

from typing import Dict, List, Optional, Tuple

# The four parts of the score and how many of the 100 points each can give.
WEIGHTS: Dict[str, float] = {
    "Sample size": 0.25,
    "Consistency": 0.35,
    "Effect size": 0.25,
    "Significance": 0.15,
}

# The part that only a genuine hypothesis test can earn. A rule that performs no
# test scores on the other three, rescaled — see active_weights().
SIGNIFICANCE = "Significance"


def active_weights(tested: bool) -> Dict[str, float]:
    """
    The weights in play for one finding, depending on whether a test was run.

    A detector that performs a hypothesis test is scored on all four parts, with
    the weights in WEIGHTS. A detector that performs no test — the anomaly scan
    is a thresholding rule, not a test — cannot earn the Significance part, so
    that part is removed and the remaining three are rescaled to still total
    100. Rescaling rather than simply capping such a finding at 85 keeps the
    0–100 scale meaning the same thing on every ticket: a score is always the
    share of the evidence available *for that kind of finding*.

    Args:
        tested: True if a p-value from a real test is available.

    Returns:
        Mapping of part name to weight, summing to 1.0.
    """
    if tested:
        return dict(WEIGHTS)
    kept = {name: weight for name, weight in WEIGHTS.items() if name != SIGNIFICANCE}
    total = sum(kept.values())
    return {name: weight / total for name, weight in kept.items()}


def confidence_breakdown(
    sample_size: int,
    consistency_ratio: float,
    effect_size: float,
    p_value: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    The confidence score, itemised: how many points each part contributed.

    Uses exactly the formula documented in compute_confidence(), which sums these parts.

    Args:
        sample_size: Number of days or observations (≥1).
        consistency_ratio: Fraction of periods pattern was observed (0–1).
        effect_size: Magnitude of effect (≥0).
        p_value: The p-value of the test behind the finding, or **None when the
            detector performs no test**. None drops the Significance part and
            rescales the other three to total 100 (see active_weights()); it
            does not score zero for significance, because "no test was run" is
            not the same claim as "the test found nothing".

    Returns:
        One dict per part, in the order of WEIGHTS, with keys:
        - name: "Sample size", "Consistency", "Effect size" or "Significance"
        - points: points earned (weight × component score), not rounded
        - max_points: the most this part can give — 25, 35, 25, 15 with a
          p-value; 29.4, 41.2, 29.4 and no Significance row without one

    Raises:
        ValueError: If inputs are out of expected ranges.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be ≥1")
    if not (0 <= consistency_ratio <= 1):
        raise ValueError("consistency_ratio must be in [0, 1]")
    if effect_size < 0:
        raise ValueError("effect_size must be ≥0")
    if p_value is not None and not (0 <= p_value <= 1):
        raise ValueError("p_value must be in [0, 1], or None if no test was run")

    # 30 days of data = full marks
    size_score = min(100, (sample_size / 30) * 100)
    # How often the pattern was observed (0–100%)
    consistency_score = consistency_ratio * 100
    # Cohen's d scale, capped at 1.25 for a full 100 points
    effect_score = min(100, effect_size * 80)

    scores = {
        "Sample size": size_score,
        "Consistency": consistency_score,
        "Effect size": effect_score,
    }
    if p_value is not None:
        # Only awarded if p < 0.05 (conventional threshold)
        scores[SIGNIFICANCE] = 0 if p_value >= 0.05 else (1 - p_value) * 100

    weights = active_weights(tested=p_value is not None)
    return [
        {"name": name, "points": weight * scores[name], "max_points": weight * 100}
        for name, weight in weights.items()
    ]


def compute_confidence(
    sample_size: int,
    consistency_ratio: float,
    effect_size: float,
    p_value: Optional[float] = None,
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

    FINDINGS WITH NO TEST BEHIND THEM
    ---------------------------------
    Pass p_value=None when the detector performs no hypothesis test, and the
    Significance component is dropped and the other three rescaled to total 100
    (0.294 / 0.412 / 0.294). The anomaly scan is the case in this project: it
    compares rolling z-scores against a threshold, which yields no p-value. The
    alternative of handing such a finding a nominal p-value would be inventing
    evidence, and scoring it p = 1.0 would be claiming a test had been run and
    had failed. Neither is true, so neither is reported.

    Args:
        sample_size: Number of days or observations (≥1).
        consistency_ratio: Fraction of periods pattern was observed (0–1).
        effect_size: Magnitude of effect (e.g., Cohen's d; ≥0).
        p_value: Statistical significance (0–1; typically from t-test or
            linregress), or None if the finding rests on no test at all.

    Returns:
        Confidence score 0–100 (int).

    Raises:
        ValueError: If inputs are out of expected ranges.
    """
    parts = confidence_breakdown(sample_size, consistency_ratio, effect_size, p_value)
    return int(round(sum(part["points"] for part in parts)))


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
