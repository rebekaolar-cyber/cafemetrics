"""Load and validate café sales data.

TODO: CSV Upload Feature (future enhancement)
============================================
If adding user CSV uploads via Dash, implement these safeguards:

1. FILE SIZE LIMIT:
   max_size = 10 * 1024 * 1024  # 10 MB
   if len(file_contents) > max_size:
       raise ValueError(f"File too large. Max {max_size / 1024 / 1024:.0f} MB")

2. EXTENSION VALIDATION:
   allowed_extensions = {'.csv'}
   if not filename.lower().endswith(tuple(allowed_extensions)):
       raise ValueError("Only .csv files are allowed")

3. COLUMN VALIDATION (already implemented in load_sales_data):
   - Validate required columns exist
   - Validate data types match expected (date, int, float, etc.)
   - Validate data ranges (hour 0-23, revenue > 0)

4. NEVER EXECUTE FILE CONTENTS:
   ✗ Bad:   eval(df.values)  or exec(file_contents)
   ✓ Safe:  Only parse CSV as data, never as code

5. CONSIDER SCANNING:
   - For CSV injection (formulas starting with =, +, @, -)
   - Check cell values don't start with '=' (Excel formula injection)
   - Sanitize any cells that look like code

6. RATE LIMITING:
   - Limit uploads per user per minute (prevent DoS)
   - Store upload audit trail (who, when, file hash)

7. ISOLATION:
   - Process uploads in a temporary directory
   - Delete after validation/loading
   - Never store raw uploads with sensitive data

The current app uses static data (generated_sample_data.py) so none
of this is needed yet. Add these when CSV upload is implemented.
"""

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
        raise FileNotFoundError(
            f"❌ Data file not found: {csv_path}\n"
            f"   Run: python data/generate_sample_data.py"
        )

    # Try to read the CSV file, catching parse errors gracefully
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except pd.errors.ParserError as e:
        raise ValueError(
            f"❌ CSV file is malformed (cannot parse): {csv_path}\n"
            f"   Error: {str(e)[:100]}...\n"
            f"   Check that the file is valid CSV format"
        ) from e
    except Exception as e:
        raise ValueError(
            f"❌ Failed to read CSV file: {csv_path}\n"
            f"   Error: {type(e).__name__}: {str(e)[:100]}"
        ) from e

    required_columns = {"date", "hour", "day_of_week", "category", "product",
                        "quantity", "unit_price", "revenue"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        found = sorted(df.columns)
        raise ValueError(
            f"❌ CSV file is missing required columns: {sorted(missing)}\n"
            f"   Found columns: {found}\n"
            f"   Required: {sorted(required_columns)}\n"
            f"   Check the CSV header and regenerate if needed"
        )

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
