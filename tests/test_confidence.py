"""Tests for confidence scoring module."""

import pytest
from src.confidence import (
    compute_confidence,
    score_to_band,
    confidence_basis,
)


class TestComputeConfidence:
    """Tests for the core confidence scoring function."""

    def test_high_confidence_case(self):
        """Test a case that should yield HIGH confidence."""
        score = compute_confidence(
            sample_size=30,
            consistency_ratio=0.90,
            effect_size=0.80,  # Larger effect for HIGH confidence
            p_value=0.001,
        )
        assert 85 <= score <= 100
        assert score_to_band(score) == "HIGH"

    def test_medium_confidence_case(self):
        """Test a case that should yield MEDIUM confidence."""
        score = compute_confidence(
            sample_size=25,
            consistency_ratio=0.80,
            effect_size=0.45,
            p_value=0.01,
        )
        assert 60 <= score < 85

    def test_low_confidence_small_sample(self):
        """Test that small samples yield LOW confidence."""
        score = compute_confidence(
            sample_size=3,
            consistency_ratio=0.50,
            effect_size=0.10,
            p_value=0.50,
        )
        assert score < 60
        assert score_to_band(score) == "LOW"

    def test_low_confidence_weak_effect(self):
        """Test that weak effects yield LOW confidence."""
        score = compute_confidence(
            sample_size=30,
            consistency_ratio=0.50,
            effect_size=0.05,
            p_value=0.50,
        )
        assert score < 60

    def test_low_confidence_insignificant(self):
        """Test that non-significant results (p≥0.05) score lower."""
        score_insig = compute_confidence(
            sample_size=30,
            consistency_ratio=0.70,
            effect_size=0.40,
            p_value=0.10,  # Not significant
        )
        score_sig = compute_confidence(
            sample_size=30,
            consistency_ratio=0.70,
            effect_size=0.40,
            p_value=0.01,  # Significant
        )
        assert score_insig < score_sig

    def test_zero_effect_size(self):
        """Test that zero effect size yields low score."""
        score = compute_confidence(
            sample_size=30,
            consistency_ratio=1.0,
            effect_size=0.0,
            p_value=0.001,
        )
        assert score < 80  # Large sample and consistency but no effect

    def test_perfect_inputs(self):
        """Test with ideal inputs."""
        score = compute_confidence(
            sample_size=90,
            consistency_ratio=1.0,
            effect_size=2.0,
            p_value=0.001,
        )
        assert score == 100

    def test_all_zero_except_pvalue(self):
        """Test with zero sample and effect but good p-value."""
        score = compute_confidence(
            sample_size=1,
            consistency_ratio=0.0,
            effect_size=0.0,
            p_value=0.001,
        )
        assert score < 50  # Mostly p-value, which is low weight

    def test_sample_size_plateau(self):
        """Test that sample size plateaus after 30 days."""
        score_30 = compute_confidence(
            sample_size=30,
            consistency_ratio=0.50,
            effect_size=0.30,
            p_value=0.05,
        )
        score_60 = compute_confidence(
            sample_size=60,
            consistency_ratio=0.50,
            effect_size=0.30,
            p_value=0.05,
        )
        # Both should have similar score (sample size only contributes 25×25%=6.25 points)
        assert score_30 == score_60

    def test_effect_size_plateau(self):
        """Test that effect size plateaus after 1.25."""
        score_125 = compute_confidence(
            sample_size=30,
            consistency_ratio=0.70,
            effect_size=1.25,
            p_value=0.05,
        )
        score_250 = compute_confidence(
            sample_size=30,
            consistency_ratio=0.70,
            effect_size=2.50,
            p_value=0.05,
        )
        assert score_125 == score_250

    def test_input_validation_sample_size(self):
        """Test that sample_size < 1 raises error."""
        with pytest.raises(ValueError, match="sample_size must be"):
            compute_confidence(0, 0.5, 0.3, 0.05)

    def test_input_validation_consistency_ratio(self):
        """Test that consistency_ratio outside [0,1] raises error."""
        with pytest.raises(ValueError, match="consistency_ratio"):
            compute_confidence(30, 1.5, 0.3, 0.05)

        with pytest.raises(ValueError, match="consistency_ratio"):
            compute_confidence(30, -0.1, 0.3, 0.05)

    def test_input_validation_effect_size(self):
        """Test that negative effect_size raises error."""
        with pytest.raises(ValueError, match="effect_size must be"):
            compute_confidence(30, 0.5, -0.1, 0.05)

    def test_input_validation_p_value(self):
        """Test that p_value outside [0,1] raises error."""
        with pytest.raises(ValueError, match="p_value must be"):
            compute_confidence(30, 0.5, 0.3, 1.5)

        with pytest.raises(ValueError, match="p_value must be"):
            compute_confidence(30, 0.5, 0.3, -0.05)


class TestScoreToBand:
    """Tests for score-to-band mapping."""

    def test_high_band(self):
        """Test HIGH band (≥85)."""
        assert score_to_band(85) == "HIGH"
        assert score_to_band(90) == "HIGH"
        assert score_to_band(100) == "HIGH"

    def test_medium_band(self):
        """Test MEDIUM band (60–84)."""
        assert score_to_band(60) == "MEDIUM"
        assert score_to_band(70) == "MEDIUM"
        assert score_to_band(84) == "MEDIUM"

    def test_low_band(self):
        """Test LOW band (<60)."""
        assert score_to_band(0) == "LOW"
        assert score_to_band(30) == "LOW"
        assert score_to_band(59) == "LOW"

    def test_band_boundaries(self):
        """Test exact boundaries."""
        assert score_to_band(84) == "MEDIUM"
        assert score_to_band(85) == "HIGH"
        assert score_to_band(59) == "LOW"
        assert score_to_band(60) == "MEDIUM"


class TestConfidenceBasis:
    """Tests for human-readable basis strings."""

    def test_basis_with_calendar_days(self):
        """Test basis string with calendar context."""
        basis = confidence_basis(30, 0.90, calendar_days=92)
        assert "4.3 weeks" in basis or "4.2 weeks" in basis
        assert "27/30" in basis or "28/30" in basis
        assert "92 calendar days" in basis

    def test_basis_without_calendar_days(self):
        """Test basis string without calendar context."""
        basis = confidence_basis(30, 0.90)
        assert "4.3 weeks" in basis or "4.2 weeks" in basis
        assert "27/30" in basis or "28/30" in basis
        assert "calendar days" not in basis

    def test_basis_single_week(self):
        """Test with single week of data."""
        basis = confidence_basis(7, 1.0)
        assert "1.0 weeks" in basis
        assert "7/7" in basis

    def test_basis_small_sample(self):
        """Test with small sample."""
        basis = confidence_basis(3, 0.33)
        assert "0.4 weeks" in basis or "0.3 weeks" in basis  # Rounding variation


class TestConfidenceExamples:
    """Integration-style tests using realistic scenarios."""

    def test_afternoon_dip_example(self):
        """Test a realistic afternoon dip detection scenario."""
        # 4 weeks of data, dip observed 28/30 weekdays, small-to-medium effect
        score = compute_confidence(
            sample_size=30,
            consistency_ratio=28/30,
            effect_size=0.25,
            p_value=0.0001,
        )
        assert score_to_band(score) == "MEDIUM"
        assert score >= 65  # Should be in the 65–84 range

    def test_trend_example(self):
        """Test a realistic trend detection scenario."""
        # 8 weeks of data, trend consistent, small effect, barely significant
        score = compute_confidence(
            sample_size=56,
            consistency_ratio=1.0,  # Trend is continuous
            effect_size=0.15,
            p_value=0.045,
        )
        assert score_to_band(score) == "MEDIUM"

    def test_anomaly_example(self):
        """Test a realistic anomaly detection scenario."""
        # 3 months, 3 major anomalies (3/92 days), large effect
        anomaly_count = 3
        total_days = 92
        score = compute_confidence(
            sample_size=total_days,
            consistency_ratio=anomaly_count / total_days,
            effect_size=3.0,
            p_value=0.001,
        )
        # Low consistency (3%), but large effect and good p-value
        assert score_to_band(score) == "MEDIUM"


class TestConfidenceBreakdown:
    """The itemised score must add up to the score itself."""

    def test_parts_sum_to_the_score(self):
        from src.confidence import confidence_breakdown

        for inputs in (
            dict(sample_size=92, consistency_ratio=1.0, effect_size=0.176, p_value=1.3e-7),
            dict(sample_size=10, consistency_ratio=0.6, effect_size=0.1, p_value=0.10),
            dict(sample_size=60, consistency_ratio=1.0, effect_size=2.0, p_value=0.001),
            dict(sample_size=1, consistency_ratio=0.0, effect_size=0.0, p_value=1.0),
        ):
            parts = confidence_breakdown(**inputs)
            assert int(round(sum(part["points"] for part in parts))) == compute_confidence(**inputs)

    def test_parts_are_named_and_capped_at_their_weight(self):
        from src.confidence import WEIGHTS, confidence_breakdown

        parts = confidence_breakdown(sample_size=500, consistency_ratio=1.0, effect_size=9.0, p_value=0.0)
        assert [part["name"] for part in parts] == list(WEIGHTS)
        assert [part["max_points"] for part in parts] == [25, 35, 25, 15]
        assert all(part["points"] == pytest.approx(part["max_points"]) for part in parts)

    def test_the_sample_dip_loses_its_points_on_effect_size(self):
        """The dashboard's example: 79/100, full marks everywhere except the small effect."""
        from src.confidence import confidence_breakdown

        parts = confidence_breakdown(sample_size=92, consistency_ratio=1.0, effect_size=0.176, p_value=1.3e-7)
        points = {part["name"]: part["points"] for part in parts}
        assert points["Sample size"] == 25 and points["Consistency"] == 35
        assert points["Effect size"] == pytest.approx(3.52)
        assert points["Significance"] == pytest.approx(15, abs=0.001)

    def test_invalid_inputs_are_rejected(self):
        from src.confidence import confidence_breakdown

        with pytest.raises(ValueError):
            confidence_breakdown(sample_size=0, consistency_ratio=0.5, effect_size=0.5, p_value=0.5)


class TestFindingsWithNoTestBehindThem:
    """p_value=None: the Significance part is dropped, the rest rescaled to 100.

    The anomaly scan performs no hypothesis test. It used to be handed a
    hard-coded p_value of 0.001 whenever any day was flagged, which bought it
    all 15 significance points on no evidence at all.
    """

    def test_the_three_remaining_weights_still_total_one_hundred(self):
        from src.confidence import active_weights

        full = active_weights(tested=True)
        assert sum(full.values()) == pytest.approx(1.0)
        assert full == {"Sample size": 0.25, "Consistency": 0.35,
                        "Effect size": 0.25, "Significance": 0.15}

        rescaled = active_weights(tested=False)
        assert "Significance" not in rescaled
        assert sum(rescaled.values()) == pytest.approx(1.0)
        # 0.25/0.85, 0.35/0.85, 0.25/0.85
        assert rescaled["Sample size"] == pytest.approx(0.25 / 0.85)
        assert rescaled["Consistency"] == pytest.approx(0.35 / 0.85)
        assert rescaled["Effect size"] == pytest.approx(0.25 / 0.85)

    def test_the_breakdown_has_three_rows_and_no_significance_row(self):
        from src.confidence import confidence_breakdown

        parts = confidence_breakdown(sample_size=30, consistency_ratio=0.5,
                                     effect_size=0.4, p_value=None)
        assert [part["name"] for part in parts] == ["Sample size", "Consistency", "Effect size"]
        assert sum(part["max_points"] for part in parts) == pytest.approx(100.0)

    def test_a_perfect_untested_finding_can_still_reach_one_hundred(self):
        """Rescaling, not capping: the 0-100 scale keeps meaning the same thing."""
        from src.confidence import compute_confidence

        assert compute_confidence(sample_size=100, consistency_ratio=1.0,
                                  effect_size=2.0, p_value=None) == 100

    def test_no_test_is_not_the_same_claim_as_a_failed_test(self):
        """
        p_value=None must not be treated as p = 1.0. A failed test scores zero
        for significance out of 15; no test at all removes the category.
        """
        from src.confidence import compute_confidence

        untested = compute_confidence(30, 0.5, 0.4, p_value=None)
        failed_test = compute_confidence(30, 0.5, 0.4, p_value=1.0)
        assert untested > failed_test

    def test_the_parts_still_sum_to_the_score_without_a_p_value(self):
        from src.confidence import compute_confidence, confidence_breakdown

        inputs = dict(sample_size=47, consistency_ratio=0.3, effect_size=0.9, p_value=None)
        parts = confidence_breakdown(**inputs)
        assert round(sum(part["points"] for part in parts)) == compute_confidence(**inputs)

    def test_the_anomaly_score_before_and_after_dropping_the_fake_p_value(self):
        """
        Pins the one number this change moves, on a dataset that has anomalies:
        2 flagged days out of 30, mean |z| 0.3128.

        Before: 25.00 + 2.33 + 1.88 + 14.98 (fabricated) = 44
        After:  29.41 + 2.75 + 2.21                      = 34
        """
        from src.confidence import compute_confidence

        evidence = {"sample_size": 30, "consistency_ratio": 2 / 30,
                    "effect_size": 0.31276 * 0.3}
        assert compute_confidence(**evidence, p_value=0.001) == 44   # the old, fabricated score
        assert compute_confidence(**evidence, p_value=None) == 34    # the honest one

    def test_an_out_of_range_p_value_is_still_rejected(self):
        from src.confidence import compute_confidence

        with pytest.raises(ValueError):
            compute_confidence(30, 0.5, 0.5, p_value=1.5)
