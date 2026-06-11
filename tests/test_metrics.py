"""Tests for metrics module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.metrics import (
    total_revenue,
    avg_daily_revenue,
    avg_transaction_value,
    total_transactions,
    peak_hour,
    peak_day,
    best_product,
    summary_metrics,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "date": [
            pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-01"),
            pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-02"),
            pd.Timestamp("2026-04-02"),
        ],
        "hour": [10, 12, 14, 10, 12],
        "day_of_week": [0, 0, 0, 1, 1],
        "category": ["Coffee", "Coffee", "Food", "Coffee", "Food"],
        "product": ["Espresso", "Latte", "Sandwich", "Espresso", "Sandwich"],
        "quantity": [1, 1, 2, 1, 1],
        "unit_price": [2.50, 3.50, 6.50, 2.50, 6.50],
        "revenue": [2.50, 3.50, 13.00, 2.50, 6.50],
    })


def test_total_revenue(sample_df):
    """Test total revenue calculation."""
    result = total_revenue(sample_df)
    expected = 2.50 + 3.50 + 13.00 + 2.50 + 6.50
    assert result == expected


def test_total_revenue_empty():
    """Test total revenue with empty DataFrame."""
    df = pd.DataFrame({"revenue": []})
    assert total_revenue(df) == 0.0


def test_avg_daily_revenue(sample_df):
    """Test average daily revenue."""
    result = avg_daily_revenue(sample_df)
    # Day 1: 2.50 + 3.50 + 13.00 = 19.00
    # Day 2: 2.50 + 6.50 = 9.00
    # Average: (19.00 + 9.00) / 2 = 14.00
    assert result == 14.00


def test_avg_daily_revenue_single_day(sample_df):
    """Test average daily revenue with single day."""
    df = sample_df[sample_df["date"] == pd.Timestamp("2026-04-01")]
    result = avg_daily_revenue(df)
    expected = 2.50 + 3.50 + 13.00
    assert result == expected


def test_avg_transaction_value(sample_df):
    """Test average transaction value."""
    result = avg_transaction_value(sample_df)
    total = 2.50 + 3.50 + 13.00 + 2.50 + 6.50
    expected = total / 5
    assert result == expected


def test_total_transactions(sample_df):
    """Test total transaction count."""
    result = total_transactions(sample_df)
    assert result == 5


def test_total_transactions_empty():
    """Test transaction count with empty DataFrame."""
    df = pd.DataFrame({"revenue": []})
    assert total_transactions(df) == 0


def test_peak_hour(sample_df):
    """Test peak hour detection."""
    # Hour 10: 2.50 + 2.50 = 5.00
    # Hour 12: 3.50 + 6.50 = 10.00
    # Hour 14: 13.00
    # Peak is hour 14 with 13.00
    hour, revenue = peak_hour(sample_df)
    assert hour == 14
    assert revenue == 13.00


def test_peak_hour_empty():
    """Test peak hour with empty DataFrame."""
    df = pd.DataFrame({"hour": [], "revenue": []})
    hour, revenue = peak_hour(df)
    assert hour == 0
    assert revenue == 0.0


def test_peak_hour_tie():
    """Test peak hour when multiple hours have same revenue."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-01")] * 4,
        "hour": [10, 10, 12, 12],
        "day_of_week": [0, 0, 0, 0],
        "category": ["Coffee"] * 4,
        "product": ["Espresso"] * 4,
        "quantity": [1] * 4,
        "unit_price": [2.50, 2.50, 2.50, 2.50],
        "revenue": [2.50, 2.50, 2.50, 2.50],
    })
    hour, revenue = peak_hour(df)
    # idxmax() returns the first occurrence
    assert hour in [10, 12]
    assert revenue == 5.00


def test_peak_day(sample_df):
    """Test peak day of week."""
    # Monday (0): 2.50 + 3.50 + 13.00 = 19.00 → avg 19.00 (1 day)
    # Tuesday (1): 2.50 + 6.50 = 9.00 → avg 9.00 (1 day)
    # Peak is Monday
    day, revenue = peak_day(sample_df)
    assert day == "Monday"
    assert revenue == 19.00


def test_peak_day_empty():
    """Test peak day with empty DataFrame."""
    df = pd.DataFrame({"day_of_week": [], "date": [], "revenue": []})
    day, revenue = peak_day(df)
    assert day == ""
    assert revenue == 0.0


def test_best_product(sample_df):
    """Test best product by revenue."""
    # Espresso: 2.50 + 2.50 = 5.00
    # Latte: 3.50
    # Sandwich: 13.00 + 6.50 = 19.50
    # Best is Sandwich
    product, revenue = best_product(sample_df)
    assert product == "Sandwich"
    assert revenue == 19.50


def test_best_product_empty():
    """Test best product with empty DataFrame."""
    df = pd.DataFrame({"product": [], "revenue": []})
    product, revenue = best_product(df)
    assert product == ""
    assert revenue == 0.0


def test_summary_metrics(sample_df):
    """Test summary metrics dictionary."""
    metrics = summary_metrics(sample_df)

    assert metrics["total_revenue"] == (2.50 + 3.50 + 13.00 + 2.50 + 6.50)
    assert metrics["avg_daily_revenue"] == 14.00
    assert metrics["total_transactions"] == 5
    assert metrics["peak_hour"] == 14
    assert metrics["peak_day"] == "Monday"
    assert metrics["best_product"] == "Sandwich"

    # Verify all expected keys are present
    expected_keys = {
        "total_revenue",
        "avg_daily_revenue",
        "avg_transaction_value",
        "total_transactions",
        "peak_hour",
        "peak_hour_revenue",
        "peak_day",
        "peak_day_revenue",
        "best_product",
        "best_product_revenue",
    }
    assert set(metrics.keys()) == expected_keys


def test_summary_metrics_empty():
    """Test summary metrics with empty data."""
    df = pd.DataFrame({
        "date": [],
        "hour": [],
        "day_of_week": [],
        "category": [],
        "product": [],
        "quantity": [],
        "unit_price": [],
        "revenue": [],
    })
    metrics = summary_metrics(df)
    assert metrics["total_revenue"] == 0.0
    assert metrics["total_transactions"] == 0
