import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

df = pd.read_csv("data/cleaned_sales.csv", parse_dates=["date"])
kpi = pd.read_csv("data/metrics_summary.csv").set_index("metric")["value"]

TEAL = "#01696f"
CARD = {"background": "#f7fafa", "border": f"1px solid {TEAL}",
        "borderRadius": "10px", "padding": "18px 24px", "flex": "1", "textAlign": "center"}
LABEL = {"fontSize": 12, "color": "#777", "marginBottom": 4, "textTransform": "uppercase", "letterSpacing": 1}
VALUE = {"fontSize": 22, "fontWeight": "700", "color": TEAL}

app = Dash(__name__)
app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "background": "white",
                              "maxWidth": 1100, "margin": "0 auto", "padding": "32px 24px"}, children=[

    html.H1("☕ CaféMetrics Dashboard",
            style={"textAlign": "center", "color": TEAL, "marginBottom": 28, "letterSpacing": 1}),

    # ── KPI cards ──────────────────────────────────────────────────────────────
    html.Div(style={"display": "flex", "gap": 16, "marginBottom": 32}, children=[
        html.Div([html.P("Total Revenue",        style=LABEL), html.P(f"£{float(kpi['total_revenue']):,.0f}",  style=VALUE)], style=CARD),
        html.Div([html.P("Avg Daily Revenue",    style=LABEL), html.P(f"£{float(kpi['avg_daily_revenue']):,.0f}", style=VALUE)], style=CARD),
        html.Div([html.P("Best Selling Product", style=LABEL), html.P(kpi["best_selling_product"],             style={**VALUE, "fontSize": 16})], style=CARD),
        html.Div([html.P("Best Day",             style=LABEL), html.P(kpi["best_day_of_week"],                 style={**VALUE, "fontSize": 16})], style=CARD),
    ]),

    # ── Dropdown ───────────────────────────────────────────────────────────────
    html.Div([
        html.Label("Filter by Category", style={**LABEL, "marginBottom": 6, "display": "block"}),
        dcc.Dropdown(id="cat-filter",
                     options=[{"label": c, "value": c} for c in ["All", "Coffee", "Food", "Drinks"]],
                     value="All", clearable=False,
                     style={"width": 220, "borderColor": TEAL}),
    ], style={"marginBottom": 24}),

    # ── Charts ─────────────────────────────────────────────────────────────────
    html.Div(style={"display": "flex", "gap": 24}, children=[
        dcc.Graph(id="weekly-chart", style={"flex": 1}),
        dcc.Graph(id="product-chart", style={"flex": 1}),
    ]),

    html.P("Data: Apr–Jun 2026 | Built with Plotly Dash + Claude",
           style={"textAlign": "center", "color": "#aaa", "marginTop": 32, "fontSize": 12}),
])

LAYOUT_BASE = dict(paper_bgcolor="white", plot_bgcolor="white", font_family="Arial",
                   title_font_color=TEAL, title_font_size=15, title_x=0.5,
                   margin=dict(t=50, b=40, l=60, r=20))

@app.callback(
    Output("weekly-chart", "figure"),
    Output("product-chart", "figure"),
    Input("cat-filter", "value"),
)
def update_charts(category):
    filtered = df if category == "All" else df[df["category"] == category]

    weekly = filtered.groupby("week")["revenue"].sum().reset_index()
    fig1 = px.line(weekly, x="week", y="revenue", markers=True,
                   title="Weekly Revenue Trend",
                   color_discrete_sequence=[TEAL],
                   labels={"week": "Week", "revenue": "Revenue (£)"})
    fig1.update_traces(line_width=2.5, marker_size=7)
    fig1.update_xaxes(showgrid=False, tickmode="linear", dtick=1)
    fig1.update_yaxes(gridcolor="#ececec", tickprefix="£")
    fig1.update_layout(**LAYOUT_BASE)

    prod = (filtered.groupby("product")["revenue"].sum()
            .reset_index().sort_values("revenue", ascending=True))
    fig2 = px.bar(prod, x="revenue", y="product", orientation="h",
                  title="Revenue by Product",
                  color_discrete_sequence=[TEAL],
                  labels={"product": "", "revenue": "Revenue (£)"},
                  text_auto=",.0f")
    fig2.update_traces(textposition="outside", textfont_size=10)
    fig2.update_xaxes(gridcolor="#ececec", tickprefix="£")
    fig2.update_yaxes(showgrid=False)
    fig2.update_layout(**{**LAYOUT_BASE, "margin": dict(t=50, b=40, l=90, r=60)})

    return fig1, fig2

if __name__ == "__main__":
    app.run(debug=True, port=8050)

server = app.server
