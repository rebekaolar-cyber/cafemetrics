"""
01_clean.py — Café Sales Data Quality Check
Junior Data Analyst | April 2026
"""

import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/raw_sales.csv"   # update this path if your file lives elsewhere
ROUNDING_TOLERANCE = 0.01     # max acceptable difference: revenue vs units*price

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Café Sales — Data Quality Report")
print("=" * 55)

df = pd.read_csv(CSV_PATH)
issues = []  # collects every problem found

print(f"\n[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns\n")

# ── 1. CONVERT DATE ───────────────────────────────────────────────────────────
print("── 1. Date conversion ──────────────────────────────────")
try:
    df["date"] = pd.to_datetime(df["date"])
    print(f"  ✓  'date' converted to datetime64  "
          f"(range: {df['date'].min().date()} → {df['date'].max().date()})")
except Exception as e:
    issues.append(f"Date conversion failed: {e}")
    print(f"  ✗  Date conversion FAILED: {e}")

# ── 2. NULL CHECK ─────────────────────────────────────────────────────────────
print("\n── 2. Null / missing values ────────────────────────────")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("  ✓  No nulls found in any column")
else:
    for col, n in null_counts[null_counts > 0].items():
        msg = f"Column '{col}' has {n} null(s)"
        issues.append(msg)
        print(f"  ✗  {msg}")

# ── 3. REVENUE INTEGRITY ──────────────────────────────────────────────────────
print("\n── 3. Revenue = units_sold × unit_price ────────────────")
df["_expected_revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
df["_diff"] = (df["revenue"] - df["_expected_revenue"]).abs()
bad_revenue = df[df["_diff"] > ROUNDING_TOLERANCE]

if bad_revenue.empty:
    print(f"  ✓  All revenue values match (tolerance ±{ROUNDING_TOLERANCE})")
else:
    msg = f"{len(bad_revenue)} row(s) have revenue mismatches"
    issues.append(msg)
    print(f"  ✗  {msg}:")
    display_cols = ["date", "product", "units_sold", "unit_price",
                    "revenue", "_expected_revenue", "_diff"]
    print(bad_revenue[display_cols].to_string(index=False))

# ── 4. BONUS CHECKS ───────────────────────────────────────────────────────────
print("\n── 4. Additional sanity checks ─────────────────────────")

# Negative or zero numeric values
for col in ["units_sold", "unit_price", "revenue"]:
    bad = df[df[col] <= 0]
    if not bad.empty:
        msg = f"Column '{col}' has {len(bad)} row(s) with zero/negative values"
        issues.append(msg)
        print(f"  ✗  {msg}")
    else:
        print(f"  ✓  '{col}' — all values positive")

# Duplicate rows
dupes = df.duplicated(subset=["date", "product"]).sum()
if dupes:
    msg = f"{dupes} duplicate (date, product) pair(s) detected"
    issues.append(msg)
    print(f"  ✗  {msg}")
else:
    print("  ✓  No duplicate (date, product) pairs")

# Unknown categories
known_cats = {"Coffee", "Food", "Drinks"}
unknown_cats = set(df["category"].unique()) - known_cats
if unknown_cats:
    msg = f"Unexpected categories: {unknown_cats}"
    issues.append(msg)
    print(f"  ✗  {msg}")
else:
    print(f"  ✓  All categories recognised: {sorted(df['category'].unique())}")

# day_of_week consistency
df["_computed_dow"] = df["date"].dt.day_name()
dow_mismatch = df[df["day_of_week"] != df["_computed_dow"]]
if not dow_mismatch.empty:
    msg = f"{len(dow_mismatch)} row(s) with wrong day_of_week label"
    issues.append(msg)
    print(f"  ✗  {msg}")
else:
    print("  ✓  'day_of_week' labels match actual dates")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if issues:
    print(f"  RESULT: {len(issues)} issue(s) found — action required\n")
    for i, issue in enumerate(issues, 1):
        print(f"  [{i}] {issue}")
else:
    print("  RESULT: All checks passed — data looks clean ✓")
print("=" * 55)

# Drop helper columns before returning clean df
df.drop(columns=["_expected_revenue", "_diff", "_computed_dow"],
        inplace=True, errors="ignore")

"""
01_clean.py — Café Sales Data Quality Check
Junior Data Analyst | April 2026
"""

import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/raw_sales.csv"   # update this path if your file lives elsewhere
ROUNDING_TOLERANCE = 0.01     # max acceptable difference: revenue vs units*price

# ── LOAD ──────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  Café Sales — Data Quality Report")
print("=" * 55)

df = pd.read_csv(CSV_PATH)
issues = []  # collects every problem found

print(f"\n[INFO] Loaded {len(df):,} rows × {len(df.columns)} columns\n")

# ── 1. CONVERT DATE ───────────────────────────────────────────────────────────
print("── 1. Date conversion ──────────────────────────────────")
try:
    df["date"] = pd.to_datetime(df["date"])
    print(f"  ✓  'date' converted to datetime64  "
          f"(range: {df['date'].min().date()} → {df['date'].max().date()})")
except Exception as e:
    issues.append(f"Date conversion failed: {e}")
    print(f"  ✗  Date conversion FAILED: {e}")

# ── 2. NULL CHECK ─────────────────────────────────────────────────────────────
print("\n── 2. Null / missing values ────────────────────────────")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("  ✓  No nulls found in any column")
else:
    for col, n in null_counts[null_counts > 0].items():
        msg = f"Column '{col}' has {n} null(s)"
        issues.append(msg)
        print(f"  ✗  {msg}")

# ── 3. REVENUE INTEGRITY ──────────────────────────────────────────────────────
print("\n── 3. Revenue = units_sold × unit_price ────────────────")
df["_expected_revenue"] = (df["units_sold"] * df["unit_price"]).round(2)
df["_diff"] = (df["revenue"] - df["_expected_revenue"]).abs()
bad_revenue = df[df["_diff"] > ROUNDING_TOLERANCE]

if bad_revenue.empty:
    print(f"  ✓  All revenue values match (tolerance ±{ROUNDING_TOLERANCE})")
else:
    msg = f"{len(bad_revenue)} row(s) have revenue mismatches"
    issues.append(msg)
    print(f"  ✗  {msg}:")
    display_cols = ["date", "product", "units_sold", "unit_price",
                    "revenue", "_expected_revenue", "_diff"]
    print(bad_revenue[display_cols].to_string(index=False))

# ── 4. BONUS CHECKS ───────────────────────────────────────────────────────────
print("\n── 4. Additional sanity checks ─────────────────────────")

# Negative or zero numeric values
for col in ["units_sold", "unit_price", "revenue"]:
    bad = df[df[col] <= 0]
    if not bad.empty:
        msg = f"Column '{col}' has {len(bad)} row(s) with zero/negative values"
        issues.append(msg)
        print(f"  ✗  {msg}")
    else:
        print(f"  ✓  '{col}' — all values positive")

# Duplicate rows
dupes = df.duplicated(subset=["date", "product"]).sum()
if dupes:
    msg = f"{dupes} duplicate (date, product) pair(s) detected"
    issues.append(msg)
    print(f"  ✗  {msg}")
else:
    print("  ✓  No duplicate (date, product) pairs")

# Unknown categories
known_cats = {"Coffee", "Food", "Drinks"}
unknown_cats = set(df["category"].unique()) - known_cats
if unknown_cats:
    msg = f"Unexpected categories: {unknown_cats}"
    issues.append(msg)
    print(f"  ✗  {msg}")
else:
    print(f"  ✓  All categories recognised: {sorted(df['category'].unique())}")

# day_of_week consistency
df["_computed_dow"] = df["date"].dt.day_name()
dow_mismatch = df[df["day_of_week"] != df["_computed_dow"]]
if not dow_mismatch.empty:
    msg = f"{len(dow_mismatch)} row(s) with wrong day_of_week label"
    issues.append(msg)
    print(f"  ✗  {msg}")
else:
    print("  ✓  'day_of_week' labels match actual dates")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if issues:
    print(f"  RESULT: {len(issues)} issue(s) found — action required\n")
    for i, issue in enumerate(issues, 1):
        print(f"  [{i}] {issue}")
else:
    print("  RESULT: All checks passed — data looks clean ✓")
print("=" * 55)

# Drop helper columns before returning clean df
df.drop(columns=["_expected_revenue", "_diff", "_computed_dow"],
        inplace=True, errors="ignore")
