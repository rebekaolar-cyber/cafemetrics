# CaféMetrics — Claude Prompt Log

A record of the prompts used to build this project end-to-end using Claude.

---

## Step 1 — Data Enrichment (`scripts/02_enrich.py`)

**Prompt:**
> Read data/raw_sales.csv. Then write and run scripts/02_enrich.py that:
> - Loads data/raw_sales.csv
> - Converts date to datetime
> - Adds a week column (ISO week number) and a month_name column
> - Saves the result to data/cleaned_sales.csv
> - Prints the shape and first 3 rows

**Produced:**
A Python script that loaded the raw CSV, parsed dates with pandas, derived ISO week numbers via `.dt.isocalendar().week` and month names via `.dt.strftime("%B")`, saved `data/cleaned_sales.csv` (720 rows × 9 columns), and printed the shape and a preview.

**Why it worked well:**
Listing each transformation as a bullet with an explicit output file and a print requirement gave Claude a clear checklist to verify against, leaving no ambiguity about what "done" looked like.

---

## Step 2 — KPI Metrics (`scripts/03_metrics.py`)

**Prompt:**
> Read data/cleaned_sales.csv and write scripts/03_metrics.py that computes these KPIs and prints them clearly labelled:
> 1. Total Revenue
> 2. Average Daily Revenue
> 3. Best Selling Product by total units_sold
> 4. Top Revenue Product by total revenue
> 5. Revenue by Category as % of total
> 6. Best Day of Week by average daily revenue
> 7. Weekly Revenue Trend — print week number + total revenue for each week
>
> Save a summary CSV to data/metrics_summary.csv with two columns: metric, value. Run the script after writing it.

**Produced:**
A 60-line script computing all 7 KPIs using pandas groupby operations, printing results in a formatted table with `£` prefixes, and saving a 22-row `metrics_summary.csv` for downstream use by the dashboard and report.

**Why it worked well:**
Numbering each KPI explicitly and specifying the exact output schema (`metric, value`) meant the summary CSV came out in a reusable format without needing a follow-up correction.

---

## Step 3 — Charts (`scripts/04_charts.py`)

**Prompt:**
> Read data/cleaned_sales.csv and write scripts/04_charts.py that creates and saves 4 Plotly charts to a folder called charts/:
> 1. charts/01_weekly_revenue.png — Line chart of weekly revenue trend. Title: "Weekly Revenue Trend"
> 2. charts/02_product_revenue.png — Horizontal bar chart sorted highest to lowest. Title: "Revenue by Product"
> 3. charts/03_category_split.png — Donut chart of revenue % by category. Title: "Revenue by Category"
> 4. charts/04_category_split.png — Bar chart of average daily revenue by day of week, sorted Mon→Sun. Title: "Avg Revenue by Day of Week"
>
> Use plotly.express. Save each as PNG using fig.write_image().
> Color scheme: use a clean palette (#01696f, #da7101, #006494).
> Make sure the charts/ folder is created if it doesn't exist. Run the script after writing it.

**Produced:**
Four publication-quality PNG charts (1800×960px at 2× scale) using a shared layout base dict for consistency. Each chart used the specified hex palette and included axis labels, tick prefixes, and value annotations.

**Why it worked well:**
Specifying exact filenames, chart types, sort orders, and the hex colour palette in one prompt eliminated the most common reasons charts need to be regenerated, producing all four correctly on the first run.

---

## Step 4 — Interactive Dashboard (`dashboard/app.py`)

**Prompt:**
> Read data/cleaned_sales.csv and data/metrics_summary.csv. Write dashboard/app.py — a Plotly Dash app that shows:
> 1. A header: "☕ CaféMetrics Dashboard"
> 2. Four KPI cards in a row: Total Revenue, Avg Daily Revenue, Best Product, Best Day
> 3. A dropdown to filter by category (All, Coffee, Food, Drinks)
> 4. Two charts side by side that update when dropdown changes:
>    - Left: Weekly revenue line chart
>    - Right: Revenue by product bar chart (horizontal)
> 5. A footer: "Data: Apr–Jun 2026 | Built with Plotly Dash + Claude"
>
> Use a clean white background, teal (#01696f) as the accent color.
> Keep it under 120 lines. Run it after writing.

**Produced:**
A 114-line Dash app with inline CSS styling, KPI cards pulling live values from `metrics_summary.csv`, a reactive dropdown wired to a single `@app.callback` updating both charts simultaneously, and a consistent teal accent throughout.

**Why it worked well:**
The line-count constraint forced a lean, single-file implementation without unnecessary abstraction, and specifying which data source each element should draw from prevented the app from hardcoding values.

---

## Step 5 — Business Insights Report (`report/insights_report.md`)

**Prompt:**
> Read data/cleaned_sales.csv and data/metrics_summary.csv.
> Write a professional business insights report and save it to report/insights_report.md. The report should include:
> 1. Executive Summary (3-4 sentences, plain English)
> 2. Key Metrics snapshot (use the actual numbers from metrics_summary.csv)
> 3. Revenue Trends — what the weekly data shows, any patterns
> 4. Product Performance — winners and underperformers
> 5. Category Analysis — Coffee vs Food vs Drinks breakdown
> 6. Day of Week Patterns — best and worst days
> 7. Three Actionable Recommendations for the café owner (specific, practical)
> 8. Data Notes — mention week 14 and 27 are partial weeks
>
> Write it as if you are a junior data analyst presenting to a café manager.
> Use markdown formatting with headers. Keep it under 600 words.

**Produced:**
An 8-section markdown report grounded in real numbers from the metrics CSV, identifying Food as the dominant category (48%), flagging the Week 18 revenue dip, and delivering three specific low-cost recommendations (weekend push, Wednesday promotion, food/drinks menu expansion).

**Why it worked well:**
Defining the audience ("junior data analyst presenting to a café manager") set the tone precisely, while the instruction to use actual numbers from the CSV prevented vague commentary and kept every claim anchored to the data.

---

## Step 6 — Prompt Documentation (`prompts/prompt_log.md`)

**Prompt:**
> Create prompts/prompt_log.md that documents all the Claude prompts used to build this project. Include 6 entries with these fields for each:
> - Step number and name
> - The prompt used
> - What it produced
> - Why this prompt worked well (1 sentence)

**Produced:**
This file — a structured log of all six prompts used across the project, with context on outputs and prompt design rationale.

**Why it worked well:**
Asking for a fixed schema (step, prompt, output, rationale) with an explicit field count made the documentation format consistent and scannable rather than a wall of prose.
