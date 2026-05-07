import os
import pandas as pd
import plotly.express as px

os.makedirs("charts", exist_ok=True)

df = pd.read_csv("data/cleaned_sales.csv", parse_dates=["date"])

PALETTE = ["#01696f", "#da7101", "#006494"]

LAYOUT = dict(
    font_family="Arial",
    paper_bgcolor="white",
    plot_bgcolor="white",
    title_font_size=18,
    title_x=0.5,
    margin=dict(t=70, b=60, l=80, r=40),
)

# ── 1. Weekly Revenue Trend (line) ────────────────────────────────────────────
weekly = df.groupby("week")["revenue"].sum().reset_index()

fig1 = px.line(
    weekly,
    x="week",
    y="revenue",
    title="Weekly Revenue Trend",
    markers=True,
    color_discrete_sequence=[PALETTE[0]],
    labels={"week": "Week Number", "revenue": "Total Revenue (£)"},
)
fig1.update_traces(line_width=2.5, marker_size=7)
fig1.update_xaxes(tickmode="linear", dtick=1, showgrid=False)
fig1.update_yaxes(showgrid=True, gridcolor="#e8e8e8", tickprefix="£")
fig1.update_layout(**LAYOUT)
fig1.write_image("charts/01_weekly_revenue.png", width=900, height=480, scale=2)
print("Saved charts/01_weekly_revenue.png")

# ── 2. Revenue by Product (horizontal bar, sorted high→low) ──────────────────
product_rev = (
    df.groupby("product")["revenue"]
    .sum()
    .reset_index()
    .sort_values("revenue", ascending=True)   # ascending so highest is at top
)

fig2 = px.bar(
    product_rev,
    x="revenue",
    y="product",
    orientation="h",
    title="Revenue by Product",
    color_discrete_sequence=[PALETTE[2]],
    labels={"product": "", "revenue": "Total Revenue (£)"},
    text_auto=",.0f",
)
fig2.update_traces(textposition="outside", textfont_size=11)
fig2.update_xaxes(showgrid=True, gridcolor="#e8e8e8", tickprefix="£")
fig2.update_yaxes(showgrid=False)
fig2.update_layout(**{**LAYOUT, "margin": dict(t=70, b=60, l=100, r=80)})
fig2.write_image("charts/02_product_revenue.png", width=900, height=480, scale=2)
print("Saved charts/02_product_revenue.png")

# ── 3. Revenue by Category (donut) ───────────────────────────────────────────
cat_rev = df.groupby("category")["revenue"].sum().reset_index()

fig3 = px.pie(
    cat_rev,
    names="category",
    values="revenue",
    title="Revenue by Category",
    hole=0.45,
    color_discrete_sequence=PALETTE,
)
fig3.update_traces(
    textposition="outside",
    textinfo="label+percent",
    textfont_size=13,
    pull=[0.03, 0.03, 0.03],
)
fig3.update_layout(**LAYOUT, showlegend=False)
fig3.write_image("charts/03_category_split.png", width=700, height=520, scale=2)
print("Saved charts/03_category_split.png")

# ── 4. Avg Revenue by Day of Week (bar, Mon→Sun) ─────────────────────────────
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

daily = df.groupby(["date", "day_of_week"])["revenue"].sum().reset_index()
avg_dow = (
    daily.groupby("day_of_week")["revenue"]
    .mean()
    .reindex(DOW_ORDER)
    .reset_index()
)
avg_dow.columns = ["day_of_week", "avg_revenue"]

fig4 = px.bar(
    avg_dow,
    x="day_of_week",
    y="avg_revenue",
    title="Avg Revenue by Day of Week",
    color_discrete_sequence=[PALETTE[1]],
    labels={"day_of_week": "Day of Week", "avg_revenue": "Avg Daily Revenue (£)"},
    text_auto=",.0f",
)
fig4.update_traces(textposition="outside", textfont_size=11)
fig4.update_xaxes(showgrid=False)
fig4.update_yaxes(showgrid=True, gridcolor="#e8e8e8", tickprefix="£")
fig4.update_layout(**LAYOUT)
fig4.write_image("charts/04_day_of_week.png", width=900, height=480, scale=2)
print("Saved charts/04_day_of_week.png")

print("\nAll 4 charts saved to charts/")
