"""Detect recurring patterns: dips, trends, anomalies."""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple


def detect_afternoon_dip(df: pd.DataFrame, dip_window: Tuple[int, int] = (14, 16)) -> Dict[str, Any]:
    """
    Detect if there's a recurring afternoon dip (lower sales in specific hours).

    Compares average revenue in dip_window to overall mean by hour.
    Returns stats (sample_size, effect_size, p_value) for confidence scoring.

    Args:
        df: Sales data with 'hour' and 'revenue' columns.
        dip_window: Tuple of (start_hour, end_hour) to check for dip (inclusive).

    Returns:
        Dict with keys:
        - detected: bool, True if dip exists
        - sample_size: number of days with dips observed
        - effect_size: (dip_avg - overall_avg) / overall_std
        - p_value: from paired t-test (dip hours vs non-dip hours)
        - dip_hours: list of hours checked
        - dip_avg_revenue: average hourly revenue during dip
        - overall_avg_revenue: average hourly revenue overall
    """
    hourly = df.groupby(["date", "hour"])["revenue"].sum().reset_index()

    dip_start, dip_end = dip_window
    dip_hours = list(range(dip_start, dip_end + 1))
    dip_data = hourly[hourly["hour"].isin(dip_hours)].groupby("date")["revenue"].mean()
    non_dip_data = hourly[~hourly["hour"].isin(dip_hours)].groupby("date")["revenue"].mean()

    if len(dip_data) == 0 or len(non_dip_data) == 0:
        return {
            "detected": False,
            "sample_size": 0,
            "effect_size": 0.0,
            "p_value": 1.0,
            "dip_hours": dip_hours,
            "dip_avg_revenue": 0.0,
            "overall_avg_revenue": 0.0,
        }

    dip_avg = dip_data.mean()
    overall_avg = hourly.groupby("hour")["revenue"].mean().mean()
    overall_std = hourly.groupby("hour")["revenue"].mean().std()

    effect_size = (overall_avg - dip_avg) / overall_std if overall_std > 0 else 0.0

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(dip_data, non_dip_data)

    return {
        "detected": dip_avg < overall_avg,
        "sample_size": len(dip_data),
        "effect_size": abs(effect_size),
        "p_value": p_value,
        "dip_hours": dip_hours,
        "dip_avg_revenue": float(dip_avg),
        "overall_avg_revenue": float(overall_avg),
    }


def detect_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect upward or downward trend using linear regression.

    Uses scipy.stats.linregress to fit daily revenue vs time.
    Returns slope, p-value, and R² for confidence scoring.

    Args:
        df: Sales data with 'date' and 'revenue' columns.

    Returns:
        Dict with keys:
        - detected: bool, True if statistically significant trend
        - slope: revenue change per day (pounds/day)
        - slope_pct: percentage change per day
        - p_value: significance of the slope
        - r_squared: goodness of fit
        - sample_size: number of days
        - trend_direction: "upward", "downward", or "none"
    """
    daily = df.groupby("date")["revenue"].sum().reset_index()
    if len(daily) < 3:
        return {
            "detected": False,
            "slope": 0.0,
            "slope_pct": 0.0,
            "p_value": 1.0,
            "r_squared": 0.0,
            "sample_size": len(daily),
            "trend_direction": "none",
        }

    x = np.arange(len(daily))
    y = daily["revenue"].values

    result = stats.linregress(x, y)

    avg_revenue = y.mean()
    slope_pct = (result.slope / avg_revenue * 100) if avg_revenue > 0 else 0.0

    is_significant = result.pvalue < 0.05
    trend_direction = "upward" if result.slope > 0 else "downward"

    return {
        "detected": is_significant,
        "slope": float(result.slope),
        "slope_pct": float(slope_pct),
        "p_value": float(result.pvalue),
        "r_squared": float(result.rvalue ** 2),
        "sample_size": len(daily),
        "trend_direction": trend_direction,
    }


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 2.5) -> Dict[str, Any]:
    """
    Detect anomalous days using Z-score (deviation from rolling mean).

    A day is anomalous if |z_score| > z_threshold.
    Returns proportion of anomalous days and details.

    Args:
        df: Sales data with 'date' and 'revenue' columns.
        z_threshold: Z-score threshold for flagging anomalies.

    Returns:
        Dict with keys:
        - detected: bool, True if anomalies found
        - anomaly_count: number of anomalous days
        - anomaly_ratio: fraction of days that are anomalous
        - mean_z_score: average absolute z-score
        - sample_size: number of days
        - anomalous_dates: list of anomalous dates
    """
    daily = df.groupby("date")["revenue"].sum().reset_index()
    if len(daily) < 7:
        return {
            "detected": False,
            "anomaly_count": 0,
            "anomaly_ratio": 0.0,
            "mean_z_score": 0.0,
            "sample_size": len(daily),
            "anomalous_dates": [],
        }

    rolling_mean = daily["revenue"].rolling(window=7, min_periods=1).mean()
    rolling_std = daily["revenue"].rolling(window=7, min_periods=1).std()

    z_scores = (daily["revenue"] - rolling_mean) / (rolling_std + 1e-8)
    is_anomalous = np.abs(z_scores) > z_threshold

    anomaly_count = is_anomalous.sum()
    anomaly_ratio = anomaly_count / len(daily)

    anomalous_dates = daily.loc[is_anomalous, "date"].dt.strftime("%Y-%m-%d").tolist()

    return {
        "detected": anomaly_count > 0,
        "anomaly_count": int(anomaly_count),
        "anomaly_ratio": float(anomaly_ratio),
        "mean_z_score": float(np.abs(z_scores).mean()),
        "sample_size": len(daily),
        "anomalous_dates": anomalous_dates,
    }


def analyze_all(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run all analyses and return results for confidence scoring.

    Args:
        df: Sales data from load_sales_data().

    Returns:
        Dict with keys: afternoon_dip, trend, anomalies.
    """
    return {
        "afternoon_dip": detect_afternoon_dip(df),
        "trend": detect_trend(df),
        "anomalies": detect_anomalies(df),
    }
