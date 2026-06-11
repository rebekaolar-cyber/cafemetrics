"""Calculate basic café sales metrics."""

import pandas as pd
from typing import Dict, Any, Tuple


def total_revenue(df: pd.DataFrame) -> float:
    """
    Calculate total revenue across all transactions.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Total revenue in pounds.
    """
    if len(df) == 0:
        return 0.0
    return float(df["revenue"].sum())


def avg_daily_revenue(df: pd.DataFrame) -> float:
    """
    Calculate average revenue per trading day.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Average daily revenue in pounds.
    """
    daily = df.groupby("date")["revenue"].sum()
    if len(daily) == 0:
        return 0.0
    return float(daily.mean())


def avg_transaction_value(df: pd.DataFrame) -> float:
    """
    Calculate average revenue per transaction.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Average transaction value in pounds.
    """
    if len(df) == 0:
        return 0.0
    return float(df["revenue"].mean())


def total_transactions(df: pd.DataFrame) -> int:
    """
    Count total number of transactions.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Total number of transactions.
    """
    return len(df)


def peak_hour(df: pd.DataFrame) -> Tuple[int, float]:
    """
    Find the hour with highest total revenue.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Tuple of (hour_of_day, total_revenue_in_that_hour).
        Returns (0, 0.0) if data is empty.
    """
    if len(df) == 0:
        return 0, 0.0

    hourly = df.groupby("hour")["revenue"].sum()
    peak_h = hourly.idxmax()
    return int(peak_h), float(hourly.max())


def peak_day(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Find the day of week with highest average revenue.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Tuple of (day_name, average_revenue_on_that_day).
        Returns ("", 0.0) if data is empty.
    """
    if len(df) == 0:
        return "", 0.0

    # Group by day_of_week and calculate avg revenue per day
    daily_by_dow = df.groupby("date").agg({"day_of_week": "first", "revenue": "sum"})
    dow_avg = daily_by_dow.groupby("day_of_week")["revenue"].mean()

    peak_dow = dow_avg.idxmax()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    return day_names[peak_dow], float(dow_avg.max())


def best_product(df: pd.DataFrame) -> Tuple[str, float]:
    """
    Find the product with highest total revenue.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Tuple of (product_name, total_revenue).
        Returns ("", 0.0) if data is empty.
    """
    if len(df) == 0:
        return "", 0.0

    product_rev = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
    top_product = product_rev.index[0]
    return str(top_product), float(product_rev.iloc[0])


def summary_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all metrics and return as a dictionary.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Dictionary with keys: total_revenue, avg_daily_revenue, avg_transaction_value,
        total_transactions, peak_hour, peak_hour_revenue, peak_day, peak_day_revenue,
        best_product, best_product_revenue.
    """
    peak_h, peak_h_rev = peak_hour(df)
    peak_d, peak_d_rev = peak_day(df)
    best_prod, best_prod_rev = best_product(df)

    return {
        "total_revenue": total_revenue(df),
        "avg_daily_revenue": avg_daily_revenue(df),
        "avg_transaction_value": avg_transaction_value(df),
        "total_transactions": total_transactions(df),
        "peak_hour": peak_h,
        "peak_hour_revenue": peak_h_rev,
        "peak_day": peak_d,
        "peak_day_revenue": peak_d_rev,
        "best_product": best_prod,
        "best_product_revenue": best_prod_rev,
    }
