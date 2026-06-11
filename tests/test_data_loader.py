"""Tests for data_loader module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from pathlib import Path

from src.data_loader import (
    load_sales_data,
    aggregate_by_hour,
    aggregate_by_day,
    get_data_span,
)


@pytest.fixture
def valid_csv(tmp_path):
    """Create a temporary valid CSV file."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=10),
        "hour": [10] * 10,
        "day_of_week": [0] * 10,
        "category": ["Coffee"] * 10,
        "product": ["Espresso"] * 10,
        "quantity": [1] * 10,
        "unit_price": [2.50] * 10,
        "revenue": [2.50] * 10,
    })
    csv_path = tmp_path / "sales.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_load_sales_data_valid(valid_csv):
    """Test loading valid CSV."""
    df = load_sales_data(valid_csv)
    assert len(df) == 10
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["hour"].min() >= 0 and df["hour"].max() <= 23


def test_load_sales_data_file_not_found():
    """Test error when file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_sales_data("/nonexistent/path.csv")


def test_load_sales_data_missing_columns(tmp_path):
    """Test error when required columns are missing."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=5),
        "hour": [10] * 5,
        # Missing other required columns
    })
    csv_path = tmp_path / "incomplete.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_sales_data(str(csv_path))


def test_load_sales_data_invalid_hour(tmp_path):
    """Test error when hour is out of range."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=5),
        "hour": [25] * 5,  # Invalid
        "day_of_week": [0] * 5,
        "category": ["Coffee"] * 5,
        "product": ["Espresso"] * 5,
        "quantity": [1] * 5,
        "unit_price": [2.50] * 5,
        "revenue": [2.50] * 5,
    })
    csv_path = tmp_path / "invalid_hour.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="hour must be integer 0-23"):
        load_sales_data(str(csv_path))


def test_load_sales_data_revenue_mismatch(tmp_path):
    """Test error when revenue ≠ quantity * unit_price."""
    df = pd.DataFrame({
        "date": pd.date_range("2026-04-01", periods=5),
        "hour": [10] * 5,
        "day_of_week": [0] * 5,
        "category": ["Coffee"] * 5,
        "product": ["Espresso"] * 5,
        "quantity": [1] * 5,
        "unit_price": [2.50] * 5,
        "revenue": [5.00] * 5,  # Wrong: should be 2.50
    })
    csv_path = tmp_path / "revenue_mismatch.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="revenue does not match"):
        load_sales_data(str(csv_path))


def test_aggregate_by_hour(valid_csv):
    """Test hourly aggregation."""
    df = load_sales_data(valid_csv)
    hourly = aggregate_by_hour(df)

    assert "transaction_count" in hourly.columns
    assert "total_revenue" in hourly.columns
    assert "avg_transaction_value" in hourly.columns
    assert all(hourly["transaction_count"] > 0)
    assert all(hourly["total_revenue"] > 0)


def test_aggregate_by_day(valid_csv):
    """Test daily aggregation."""
    df = load_sales_data(valid_csv)
    daily = aggregate_by_day(df)

    assert "transaction_count" in daily.columns
    assert "total_revenue" in daily.columns
    assert "avg_transaction_value" in daily.columns
    assert len(daily) == 10  # 10 unique days


def test_aggregate_by_day_multiple_hours(tmp_path):
    """Test daily aggregation with multiple hours per day."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-01")] * 3 + [pd.Timestamp("2026-04-02")] * 3,
        "hour": [10, 12, 14, 10, 12, 14],
        "day_of_week": [0, 0, 0, 1, 1, 1],
        "category": ["Coffee"] * 6,
        "product": ["Espresso"] * 6,
        "quantity": [1] * 6,
        "unit_price": [2.50] * 6,
        "revenue": [2.50] * 6,
    })
    csv_path = tmp_path / "multi_hour.csv"
    df.to_csv(csv_path, index=False)

    df_loaded = load_sales_data(str(csv_path))
    daily = aggregate_by_day(df_loaded)

    assert len(daily) == 2  # 2 unique days
    assert daily.iloc[0]["transaction_count"] == 3  # First day has 3 transactions


def test_get_data_span(valid_csv):
    """Test data span calculation."""
    df = load_sales_data(valid_csv)
    calendar_days, trading_days = get_data_span(df)

    assert calendar_days == 10
    assert trading_days == 10


def test_get_data_span_gaps(tmp_path):
    """Test data span with gaps (missing days)."""
    df = pd.DataFrame({
        "date": [pd.Timestamp("2026-04-01"), pd.Timestamp("2026-04-03"),
                 pd.Timestamp("2026-04-05")],
        "hour": [10, 10, 10],
        "day_of_week": [0, 2, 4],
        "category": ["Coffee"] * 3,
        "product": ["Espresso"] * 3,
        "quantity": [1] * 3,
        "unit_price": [2.50] * 3,
        "revenue": [2.50] * 3,
    })
    csv_path = tmp_path / "gaps.csv"
    df.to_csv(csv_path, index=False)

    df_loaded = load_sales_data(str(csv_path))
    calendar_days, trading_days = get_data_span(df_loaded)

    assert calendar_days == 5  # Apr 1 to Apr 5 = 5 days
    assert trading_days == 3   # Only 3 days with sales
