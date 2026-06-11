"""Tests for analysis module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.analysis import (
    detect_afternoon_dip,
    detect_trend,
    detect_anomalies,
    analyze_all,
)


@pytest.fixture
def dip_data():
    """Create data with clear afternoon dip."""
    records = []
    for day in range(30):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        for hour in range(10, 20):
            if 14 <= hour <= 15:  # dip window
                revenue = 5.0 + np.random.normal(0, 0.5)  # low
            else:
                revenue = 15.0 + np.random.normal(0, 1.0)  # high
            records.append({
                "date": date,
                "hour": hour,
                "day_of_week": date.weekday(),
                "category": "Coffee",
                "product": "Espresso",
                "quantity": 1,
                "unit_price": 2.50,
                "revenue": max(0.5, revenue),
            })
    return pd.DataFrame(records)


@pytest.fixture
def trend_data():
    """Create data with clear upward trend."""
    records = []
    for day in range(30):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        daily_revenue = 500 + (day * 10) + np.random.normal(0, 20)
        records.append({
            "date": date,
            "hour": 12,
            "day_of_week": date.weekday(),
            "category": "Coffee",
            "product": "Espresso",
            "quantity": 1,
            "unit_price": 2.50,
            "revenue": daily_revenue,
        })
    return pd.DataFrame(records)


@pytest.fixture
def anomaly_data():
    """Create data with clear anomalies."""
    records = []
    for day in range(30):
        date = datetime(2026, 4, 1) + timedelta(days=day)
        if day == 10:  # Day with large anomaly
            revenue = 10.0  # Very low
        elif day == 20:  # Another large anomaly
            revenue = 500.0  # Very high
        else:
            revenue = 100.0 + np.random.normal(0, 5.0)
        records.append({
            "date": date,
            "hour": 12,
            "day_of_week": date.weekday(),
            "category": "Coffee",
            "product": "Espresso",
            "quantity": 1,
            "unit_price": 2.50,
            "revenue": max(0.5, revenue),
        })
    return pd.DataFrame(records)


def test_detect_afternoon_dip(dip_data):
    """Test afternoon dip detection."""
    result = detect_afternoon_dip(dip_data)

    assert result["detected"] == True
    assert result["sample_size"] > 0
    assert result["dip_avg_revenue"] < result["overall_avg_revenue"]
    assert result["effect_size"] > 0
    assert isinstance(result["p_value"], float)


def test_detect_afternoon_dip_no_dip():
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
                "revenue": 15.0,  # Constant
            })
    df = pd.DataFrame(records)
    result = detect_afternoon_dip(df)

    assert result["detected"] == False
    assert result["dip_avg_revenue"] == result["overall_avg_revenue"]


def test_detect_afternoon_dip_empty():
    """Test afternoon dip detection with empty data."""
    df = pd.DataFrame({"hour": [], "date": [], "revenue": []})
    result = detect_afternoon_dip(df)

    assert result["detected"] == False
    assert result["sample_size"] == 0


def test_detect_afternoon_dip_keys():
    """Test that all expected keys are in result."""
    df = pd.DataFrame({"hour": [10], "date": [datetime(2026, 4, 1)], "revenue": [10.0]})
    result = detect_afternoon_dip(df)

    expected_keys = {
        "detected",
        "sample_size",
        "effect_size",
        "p_value",
        "dip_hours",
        "dip_avg_revenue",
        "overall_avg_revenue",
    }
    assert set(result.keys()) == expected_keys


def test_detect_trend(trend_data):
    """Test trend detection."""
    result = detect_trend(trend_data)

    assert result["detected"] == True  # Clear upward trend
    assert result["slope"] > 0  # Positive slope
    assert result["sample_size"] == 30
    assert result["trend_direction"] == "upward"
    assert 0 <= result["r_squared"] <= 1


def test_detect_trend_no_trend():
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
            "revenue": 50.0 + np.random.normal(0, 1.0),  # Random noise, no trend
        })
    df = pd.DataFrame(records)
    result = detect_trend(df)

    # With small noise, slope should be near zero
    assert abs(result["slope"]) < 5.0
    assert result["p_value"] > 0.05 or np.isnan(result["p_value"])


def test_detect_trend_downward():
    """Test downward trend detection."""
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
            "revenue": 500 - (day * 10),  # Decreasing
        })
    df = pd.DataFrame(records)
    result = detect_trend(df)

    assert result["slope"] < 0
    assert result["trend_direction"] == "downward"


def test_detect_trend_small_sample():
    """Test trend detection with very small sample."""
    df = pd.DataFrame({
        "date": [datetime(2026, 4, 1), datetime(2026, 4, 2)],
        "hour": [12, 12],
        "day_of_week": [0, 1],
        "category": ["Coffee", "Coffee"],
        "product": ["Espresso", "Espresso"],
        "quantity": [1, 1],
        "unit_price": [2.50, 2.50],
        "revenue": [50.0, 55.0],
    })
    result = detect_trend(df)

    assert result["sample_size"] == 2
    assert result["detected"] == False  # Too small for significance


def test_detect_anomalies(anomaly_data):
    """Test anomaly detection."""
    result = detect_anomalies(anomaly_data, z_threshold=2.0)  # Slightly lower threshold

    assert result["detected"] == True
    assert result["anomaly_count"] > 0
    assert 0 <= result["anomaly_ratio"] <= 1
    assert result["sample_size"] == 30


def test_detect_anomalies_no_anomalies():
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
            "revenue": 50.0 + np.random.normal(0, 0.5),  # Small variation
        })
    df = pd.DataFrame(records)
    result = detect_anomalies(df)

    assert result["anomaly_count"] == 0


def test_detect_anomalies_custom_threshold():
    """Test anomaly detection with custom z-score threshold."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=20),
        "hour": [12] * 20,
        "day_of_week": [0] * 20,
        "category": ["Coffee"] * 20,
        "product": ["Espresso"] * 20,
        "quantity": [1] * 20,
        "unit_price": [2.50] * 20,
        "revenue": [50.0] * 19 + [200.0],  # Last day is anomalous
    })

    # High threshold: less sensitive
    result_high = detect_anomalies(df, z_threshold=5.0)
    assert result_high["anomaly_count"] <= 1

    # Low threshold: more sensitive
    result_low = detect_anomalies(df, z_threshold=1.0)
    assert result_low["anomaly_count"] >= 1


def test_detect_anomalies_small_sample():
    """Test anomaly detection with small sample."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=5),
        "hour": [12] * 5,
        "day_of_week": [0] * 5,
        "category": ["Coffee"] * 5,
        "product": ["Espresso"] * 5,
        "quantity": [1] * 5,
        "unit_price": [2.50] * 5,
        "revenue": [50.0] * 5,
    })
    result = detect_anomalies(df)

    assert result["detected"] == False
    assert result["sample_size"] == 5


def test_analyze_all(dip_data):
    """Test combined analysis."""
    result = analyze_all(dip_data)

    assert "afternoon_dip" in result
    assert "trend" in result
    assert "anomalies" in result

    assert isinstance(result["afternoon_dip"], dict)
    assert isinstance(result["trend"], dict)
    assert isinstance(result["anomalies"], dict)
