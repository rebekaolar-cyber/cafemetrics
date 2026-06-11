"""Load and validate café sales data."""

import pandas as pd
from pathlib import Path
from typing import Tuple


def load_sales_data(csv_path: str) -> pd.DataFrame:
    """
    Load sales data from CSV and perform basic validation.

    Args:
        csv_path: Path to CSV file with columns: date, hour, day_of_week,
                  category, product, quantity, unit_price, revenue.

    Returns:
        DataFrame with validated sales data. date column is datetime.

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If required columns are missing or data is invalid.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["date"])

    required_columns = {"date", "hour", "day_of_week", "category", "product",
                        "quantity", "unit_price", "revenue"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")

    # Validate data types and ranges
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("date column must be datetime")

    if not (df["hour"].between(0, 23).all() and df["hour"].dtype in ["int64", "int32"]):
        raise ValueError("hour must be integer 0-23")

    if not (df["day_of_week"].between(0, 6).all() and df["day_of_week"].dtype in ["int64", "int32"]):
        raise ValueError("day_of_week must be integer 0-6")

    if not (df["quantity"] > 0).all():
        raise ValueError("quantity must be positive")

    if not (df["unit_price"] > 0).all():
        raise ValueError("unit_price must be positive")

    if not (df["revenue"] > 0).all():
        raise ValueError("revenue must be positive")

    # Verify revenue = quantity * unit_price (within floating-point tolerance)
    expected_revenue = df["quantity"] * df["unit_price"]
    if not (df["revenue"].sub(expected_revenue).abs() < 0.01).all():
        raise ValueError("revenue does not match quantity * unit_price")

    return df.sort_values("date").reset_index(drop=True)


def aggregate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales by date and hour.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        DataFrame with columns: date, hour, day_of_week, transaction_count,
        total_revenue, avg_transaction_value.
    """
    hourly = df.groupby(["date", "hour", "day_of_week"]).agg(
        transaction_count=("revenue", "count"),
        total_revenue=("revenue", "sum"),
        avg_transaction_value=("revenue", "mean"),
    ).reset_index()

    return hourly


def aggregate_by_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales by date.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        DataFrame with columns: date, day_of_week, transaction_count,
        total_revenue, avg_transaction_value.
    """
    daily = df.groupby(["date", "day_of_week"]).agg(
        transaction_count=("revenue", "count"),
        total_revenue=("revenue", "sum"),
        avg_transaction_value=("revenue", "mean"),
    ).reset_index()

    return daily


def get_data_span(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Get the number of calendar days and number of actual trading days.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Tuple of (calendar_days, trading_days).
    """
    calendar_days = (df["date"].max() - df["date"].min()).days + 1
    trading_days = df["date"].nunique()
    return calendar_days, trading_days
