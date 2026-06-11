"""Tests for insights generation module."""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.insights import (
    Insight,
    generate_afternoon_dip_insight,
    generate_trend_insight,
    generate_anomaly_insights,
    generate_all_insights,
)


@pytest.fixture
def dip_data():
    """Create data with clear afternoon dip."""
    records = []
    for day in range(30):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        for hour in range(10, 20):
            if 14 <= hour <= 15:  # dip window
                revenue = 5.0 + hour * 0.1
            else:
                revenue = 15.0 + hour * 0.2
            records.append({
                "date": date,
                "hour": hour,
                "day_of_week": date.weekday(),
                "category": "Coffee",
                "product": "Espresso",
                "quantity": 1,
                "unit_price": 2.50,
                "revenue": revenue,
            })
    return pd.DataFrame(records)


@pytest.fixture
def trend_data():
    """Create data with upward trend."""
    records = []
    for day in range(30):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        records.append({
            "date": date,
            "hour": 12,
            "day_of_week": date.weekday(),
            "category": "Coffee",
            "product": "Espresso",
            "quantity": 1,
            "unit_price": 2.50,
            "revenue": 500 + (day * 10),
        })
    return pd.DataFrame(records)


class TestInsightClass:
    """Tests for the Insight class."""

    def test_insight_creation(self):
        """Test creating an Insight object."""
        insight = Insight(
            title="Test Insight",
            text="This is a test insight.",
            confidence_score=75,
            basis="10 days of data",
            suggested_action="Do something",
            pattern_type="test",
        )
        assert insight.title == "Test Insight"
        assert insight.confidence_score == 75
        assert insight.confidence_band == "MEDIUM"
        assert insight.status == "pending"

    def test_insight_to_dict(self):
        """Test converting Insight to dictionary."""
        insight = Insight(
            title="Test",
            text="Test text",
            confidence_score=80,
            basis="Basis",
            suggested_action="Action",
            pattern_type="test",
        )
        d = insight.to_dict()
        assert d["title"] == "Test"
        assert d["confidence_score"] == 80
        assert d["confidence_band"] == "MEDIUM"
        assert d["status"] == "pending"

    def test_insight_high_confidence(self):
        """Test HIGH confidence band."""
        insight = Insight(
            title="Test",
            text="Test",
            confidence_score=90,
            basis="Basis",
            suggested_action="Action",
            pattern_type="test",
        )
        assert insight.confidence_band == "HIGH"

    def test_insight_low_confidence(self):
        """Test LOW confidence band."""
        insight = Insight(
            title="Test",
            text="Test",
            confidence_score=50,
            basis="Basis",
            suggested_action="Action",
            pattern_type="test",
        )
        assert insight.confidence_band == "LOW"


class TestAfternoonDipInsight:
    """Tests for afternoon dip insight generation."""

    def test_dip_insight_generated(self, dip_data):
        """Test that afternoon dip insight is generated."""
        from src.analysis import analyze_all
        analysis = analyze_all(dip_data)
        insight = generate_afternoon_dip_insight(analysis, 30)

        assert insight is not None
        assert insight.pattern_type == "afternoon_dip"
        assert insight.title == "Afternoon Dip"
        assert "afternoon" in insight.text.lower()

    def test_dip_insight_confidence(self, dip_data):
        """Test that dip insight has reasonable confidence."""
        from src.analysis import analyze_all
        analysis = analyze_all(dip_data)
        insight = generate_afternoon_dip_insight(analysis, 30)

        # Should have at least MEDIUM confidence
        assert insight.confidence_score >= 60

    def test_dip_insight_has_action(self, dip_data):
        """Test that dip insight has a suggested action."""
        from src.analysis import analyze_all
        analysis = analyze_all(dip_data)
        insight = generate_afternoon_dip_insight(analysis, 30)

        assert len(insight.suggested_action) > 0
        assert insight.suggested_action != ""

    def test_dip_insight_no_dip(self):
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
                    "revenue": 15.0,
                })
        df = pd.DataFrame(records)

        from src.analysis import analyze_all
        analysis = analyze_all(df)
        insight = generate_afternoon_dip_insight(analysis, 20)

        assert insight is None


class TestTrendInsight:
    """Tests for trend insight generation."""

    def test_trend_insight_generated(self, trend_data):
        """Test that trend insight is generated."""
        from src.analysis import analyze_all
        analysis = analyze_all(trend_data)
        insight = generate_trend_insight(analysis, 30)

        assert insight is not None
        assert insight.pattern_type == "trend"
        assert "upward" in insight.title.lower()

    def test_trend_insight_upward(self, trend_data):
        """Test upward trend is identified."""
        from src.analysis import analyze_all
        analysis = analyze_all(trend_data)
        insight = generate_trend_insight(analysis, 30)

        assert "upward" in insight.text.lower()
        assert "momentum" in insight.suggested_action.lower()

    def test_trend_insight_no_trend(self):
        """Test when there is no trend."""
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
                "revenue": 50.0,
            })
        df = pd.DataFrame(records)

        from src.analysis import analyze_all
        analysis = analyze_all(df)
        insight = generate_trend_insight(analysis, 20)

        assert insight is None


class TestAnomalyInsights:
    """Tests for anomaly insight generation."""

    def test_anomaly_insights_structure(self):
        """Test that anomaly insights return a list."""
        records = []
        for day in range(30):
            date = datetime(2026, 4, 1) + timedelta(days=day)
            if day == 10:
                revenue = 5.0
            elif day == 20:
                revenue = 500.0
            else:
                revenue = 100.0
            records.append({
                "date": date,
                "hour": 12,
                "day_of_week": date.weekday(),
                "category": "Coffee",
                "product": "Espresso",
                "quantity": 1,
                "unit_price": 2.50,
                "revenue": revenue,
            })
        df = pd.DataFrame(records)

        from src.analysis import analyze_all
        analysis = analyze_all(df)
        insights = generate_anomaly_insights(analysis, 30)

        assert isinstance(insights, list)

    def test_anomaly_insights_no_anomalies(self):
        """Test when there are no anomalies."""
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
                "revenue": 50.0,
            })
        df = pd.DataFrame(records)

        from src.analysis import analyze_all
        analysis = analyze_all(df)
        insights = generate_anomaly_insights(analysis, 20)

        assert len(insights) == 0


class TestGenerateAllInsights:
    """Tests for the combined insight generation function."""

    def test_all_insights_returns_list(self, dip_data):
        """Test that generate_all_insights returns a list."""
        insights = generate_all_insights(dip_data, 30)
        assert isinstance(insights, list)

    def test_all_insights_structure(self, dip_data):
        """Test that each insight has required fields."""
        insights = generate_all_insights(dip_data, 30)

        for insight in insights:
            assert hasattr(insight, "title")
            assert hasattr(insight, "text")
            assert hasattr(insight, "confidence_score")
            assert hasattr(insight, "basis")
            assert hasattr(insight, "suggested_action")
            assert hasattr(insight, "pattern_type")

    def test_all_insights_multiple_patterns(self):
        """Test generation with multiple patterns."""
        records = []
        for day in range(40):
            date = datetime(2026, 4, 1) + timedelta(days=day)
            for hour in range(10, 20):
                if 14 <= hour <= 15:
                    revenue = 5.0
                else:
                    revenue = 15.0 + (day * 0.5)  # Trend + dip
                records.append({
                    "date": date,
                    "hour": hour,
                    "day_of_week": date.weekday(),
                    "category": "Coffee",
                    "product": "Espresso",
                    "quantity": 1,
                    "unit_price": 2.50,
                    "revenue": revenue,
                })
        df = pd.DataFrame(records)
        insights = generate_all_insights(df, 40)

        # Should have at least 1 insight (dip or trend)
        assert len(insights) >= 1
        assert all(isinstance(i, Insight) for i in insights)
