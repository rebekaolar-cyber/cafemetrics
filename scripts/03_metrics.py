import pandas as pd

df = pd.read_csv("data/cleaned_sales.csv", parse_dates=["date"])

# ── 1. Total Revenue ──────────────────────────────────────────────────────────
total_revenue = df["revenue"].sum()

# ── 2. Average Daily Revenue ──────────────────────────────────────────────────
n_days = df["date"].nunique()
avg_daily_revenue = total_revenue / n_days

# ── 3. Best Selling Product (units) ──────────────────────────────────────────
units_by_product = df.groupby("product")["units_sold"].sum()
best_selling_product = units_by_product.idxmax()
best_selling_units = units_by_product.max()

# ── 4. Top Revenue Product ────────────────────────────────────────────────────
revenue_by_product = df.groupby("product")["revenue"].sum()
top_revenue_product = revenue_by_product.idxmax()
top_revenue_product_value = revenue_by_product.max()

# ── 5. Revenue by Category as % of total ─────────────────────────────────────
revenue_by_category = df.groupby("category")["revenue"].sum()
category_pct = (revenue_by_category / total_revenue * 100).round(2)

# ── 6. Best Day of Week by average daily revenue ─────────────────────────────
daily_revenue = df.groupby(["date", "day_of_week"])["revenue"].sum().reset_index()
avg_by_dow = daily_revenue.groupby("day_of_week")["revenue"].mean()
best_dow = avg_by_dow.idxmax()
best_dow_value = avg_by_dow.max()

# ── 7. Weekly Revenue Trend ───────────────────────────────────────────────────
weekly_revenue = df.groupby("week")["revenue"].sum().sort_index()

# ── Print ─────────────────────────────────────────────────────────────────────
sep = "─" * 52

print(sep)
print(f"  1. Total Revenue:              £{total_revenue:>10,.2f}")
print(f"  2. Average Daily Revenue:      £{avg_daily_revenue:>10,.2f}")
print(f"  3. Best Selling Product:        {best_selling_product} ({best_selling_units:,} units)")
print(f"  4. Top Revenue Product:         {top_revenue_product} (£{top_revenue_product_value:,.2f})")
print(sep)
print("  5. Revenue by Category:")
for cat, pct in category_pct.items():
    print(f"       {cat:<10}  £{revenue_by_category[cat]:>10,.2f}  ({pct:.2f}%)")
print(sep)
print(f"  6. Best Day of Week:            {best_dow} (avg £{best_dow_value:,.2f}/day)")
print(sep)
print("  7. Weekly Revenue Trend:")
for week, rev in weekly_revenue.items():
    print(f"       Week {week:>2}   £{rev:>10,.2f}")
print(sep)

# ── Save summary CSV ──────────────────────────────────────────────────────────
rows = [
    ("total_revenue",            f"{total_revenue:.2f}"),
    ("avg_daily_revenue",        f"{avg_daily_revenue:.2f}"),
    ("best_selling_product",     f"{best_selling_product} ({best_selling_units} units)"),
    ("top_revenue_product",      f"{top_revenue_product} (£{top_revenue_product_value:.2f})"),
]
for cat, pct in category_pct.items():
    rows.append((f"revenue_pct_{cat.lower()}", f"{pct:.2f}%"))
rows.append(("best_day_of_week", f"{best_dow} (avg £{best_dow_value:.2f})"))
for week, rev in weekly_revenue.items():
    rows.append((f"weekly_revenue_week_{week}", f"{rev:.2f}"))

summary = pd.DataFrame(rows, columns=["metric", "value"])
summary.to_csv("data/metrics_summary.csv", index=False)
print(f"\n  Saved {len(summary)} rows → data/metrics_summary.csv")
