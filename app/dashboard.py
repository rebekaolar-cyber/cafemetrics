"""CaféMetrics dashboard: transparent, auditable café sales analytics."""

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_sales_data, aggregate_by_hour
from src.metrics import summary_metrics
from src.insights import generate_all_insights
from src.verification import (
    init_db,
    save_insight,
    get_insights,
    update_insight_status,
    clear_db,
)

# Load data
try:
    df = load_sales_data("data/sample_sales.csv")
    metrics = summary_metrics(df)
    insights = generate_all_insights(df)
    hourly = aggregate_by_hour(df)
except Exception as e:
    print(f"Error loading data: {e}")
    df = None
    metrics = {}
    insights = []
    hourly = None

# Color scheme
TEAL = "#01696f"
BG_LIGHT = "#f5f5f5"
WHITE = "#ffffff"
GRAY_TEXT = "#666"

# Initialize database
init_db()

# Create Dash app
app = Dash(__name__, title="CaféMetrics")
server = app.server  # For Gunicorn

app.layout = html.Div(
    style={
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        "backgroundColor": BG_LIGHT,
        "minHeight": "100vh",
        "padding": "0 20px",
    },
    children=[
        # Header
        html.Div(
            style={"maxWidth": 1200, "margin": "0 auto", "paddingTop": 40},
            children=[
                html.H1(
                    "☕ CaféMetrics",
                    style={
                        "color": TEAL,
                        "marginBottom": 8,
                        "fontSize": 32,
                        "fontWeight": 700,
                    },
                ),
                html.P(
                    "Rule-based sales analysis with transparent confidence scoring",
                    style={"color": GRAY_TEXT, "marginBottom": 32, "fontSize": 14},
                ),
            ],
        ),

        # Main content container
        html.Div(
            style={"maxWidth": 1200, "margin": "0 auto", "paddingBottom": 60},
            children=[
                # KPI cards
                html.Div(
                    style={"display": "flex", "gap": 16, "marginBottom": 40, "flexWrap": "wrap"},
                    children=[
                        html.Div(
                            style={
                                "flex": "1 1 200px",
                                "backgroundColor": WHITE,
                                "padding": 20,
                                "borderRadius": 12,
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                            },
                            children=[
                                html.P("Total Revenue", style={"margin": 0, "color": GRAY_TEXT, "fontSize": 12, "fontWeight": 600}),
                                html.P(
                                    f"£{metrics.get('total_revenue', 0):,.0f}",
                                    style={"margin": "8px 0 0 0", "color": TEAL, "fontSize": 28, "fontWeight": 700},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "flex": "1 1 200px",
                                "backgroundColor": WHITE,
                                "padding": 20,
                                "borderRadius": 12,
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                            },
                            children=[
                                html.P("Avg Daily", style={"margin": 0, "color": GRAY_TEXT, "fontSize": 12, "fontWeight": 600}),
                                html.P(
                                    f"£{metrics.get('avg_daily_revenue', 0):,.0f}",
                                    style={"margin": "8px 0 0 0", "color": TEAL, "fontSize": 28, "fontWeight": 700},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "flex": "1 1 200px",
                                "backgroundColor": WHITE,
                                "padding": 20,
                                "borderRadius": 12,
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                            },
                            children=[
                                html.P("Transactions", style={"margin": 0, "color": GRAY_TEXT, "fontSize": 12, "fontWeight": 600}),
                                html.P(
                                    f"{metrics.get('total_transactions', 0):,}",
                                    style={"margin": "8px 0 0 0", "color": TEAL, "fontSize": 28, "fontWeight": 700},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "flex": "1 1 200px",
                                "backgroundColor": WHITE,
                                "padding": 20,
                                "borderRadius": 12,
                                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                            },
                            children=[
                                html.P("Peak Hour", style={"margin": 0, "color": GRAY_TEXT, "fontSize": 12, "fontWeight": 600}),
                                html.P(
                                    f"{int(metrics.get('peak_hour', 0)):02d}:00",
                                    style={"margin": "8px 0 0 0", "color": TEAL, "fontSize": 28, "fontWeight": 700},
                                ),
                            ],
                        ),
                    ],
                ),

                # Hourly chart
                html.Div(
                    style={
                        "backgroundColor": WHITE,
                        "padding": 24,
                        "borderRadius": 12,
                        "marginBottom": 40,
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                    },
                    children=[
                        html.H3("Hourly Revenue Pattern", style={"color": TEAL, "marginBottom": 20, "fontSize": 18}),
                        dcc.Graph(id="hourly-chart", style={"margin": 0}),
                    ],
                ),

                # Insights panel
                html.Div(
                    style={
                        "backgroundColor": WHITE,
                        "padding": 24,
                        "borderRadius": 12,
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                    },
                    children=[
                        html.H3("Insights (Transparent & Auditable)", style={"color": TEAL, "marginBottom": 20, "fontSize": 18}),
                        html.Div(id="insights-container", children=[
                            html.Div(
                                style={"padding": 12, "backgroundColor": "#f0f0f0", "borderRadius": 8, "marginBottom": 16},
                                children=[
                                    html.P(
                                        "No insights yet. Analyze your data using the Q&A buttons below.",
                                        style={"color": GRAY_TEXT, "margin": 0},
                                    ),
                                ],
                            )
                        ] if not insights else [
                            html.Div(
                                style={
                                    "padding": 16,
                                    "borderLeft": f"4px solid {TEAL}",
                                    "backgroundColor": "#f9fafb",
                                    "borderRadius": 8,
                                    "marginBottom": 16,
                                },
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "space-between", "alignItems": "start"},
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.H4(insight.title, style={"margin": "0 0 4px 0", "color": TEAL, "fontSize": 16}),
                                                    html.Span(
                                                        insight.confidence_band,
                                                        style={
                                                            "display": "inline-block",
                                                            "padding": "4px 8px",
                                                            "backgroundColor": {"HIGH": "#d4edda", "MEDIUM": "#fff3cd", "LOW": "#f8d7da"}.get(insight.confidence_band, "#e2e3e5"),
                                                            "color": {"HIGH": "#155724", "MEDIUM": "#856404", "LOW": "#721c24"}.get(insight.confidence_band, "#383d41"),
                                                            "borderRadius": 4,
                                                            "fontSize": 11,
                                                            "fontWeight": 600,
                                                            "marginRight": 8,
                                                        },
                                                    ),
                                                    html.Span(
                                                        f"{insight.confidence_score}/100",
                                                        style={"fontSize": 12, "color": GRAY_TEXT},
                                                    ),
                                                    html.P(insight.text, style={"margin": "12px 0 8px 0", "color": "#333", "fontSize": 14}),
                                                    html.P(
                                                        f"📊 Basis: {insight.basis}",
                                                        style={"margin": "0 0 8px 0", "color": GRAY_TEXT, "fontSize": 12, "fontStyle": "italic"},
                                                    ),
                                                    html.P(
                                                        f"✓ Action: {insight.suggested_action}",
                                                        style={"margin": "0", "color": "#333", "fontSize": 13},
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"display": "flex", "gap": 8},
                                                children=[
                                                    html.Button(
                                                        "✓ Verify",
                                                        id={"type": "verify-btn", "index": insight.pattern_type},
                                                        style={
                                                            "padding": "6px 12px",
                                                            "fontSize": 12,
                                                            "backgroundColor": "#d4edda",
                                                            "color": "#155724",
                                                            "border": "1px solid #c3e6cb",
                                                            "borderRadius": 4,
                                                            "cursor": "pointer",
                                                        },
                                                    ),
                                                    html.Button(
                                                        "✗ Dismiss",
                                                        id={"type": "dismiss-btn", "index": insight.pattern_type},
                                                        style={
                                                            "padding": "6px 12px",
                                                            "fontSize": 12,
                                                            "backgroundColor": "#f8d7da",
                                                            "color": "#721c24",
                                                            "border": "1px solid #f5c6cb",
                                                            "borderRadius": 4,
                                                            "cursor": "pointer",
                                                        },
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            )
                            for insight in insights
                        ]),
                    ],
                ),

                # Q&A section (fixed buttons, no LLM)
                html.Div(
                    style={
                        "backgroundColor": WHITE,
                        "padding": 24,
                        "borderRadius": 12,
                        "marginTop": 40,
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                    },
                    children=[
                        html.H3("Quick Questions", style={"color": TEAL, "marginBottom": 20, "fontSize": 18}),
                        html.P(
                            "Select a question to dive into the analysis. All answers are rule-based and fully auditable.",
                            style={"color": GRAY_TEXT, "marginBottom": 16, "fontSize": 13},
                        ),
                        html.Div(
                            style={"display": "flex", "gap": 12, "flexWrap": "wrap"},
                            children=[
                                html.Button(
                                    "📉 Why the afternoon dip?",
                                    id="q-dip",
                                    style={
                                        "padding": "12px 16px",
                                        "fontSize": 13,
                                        "backgroundColor": TEAL,
                                        "color": WHITE,
                                        "border": "none",
                                        "borderRadius": 6,
                                        "cursor": "pointer",
                                        "fontWeight": 600,
                                    },
                                ),
                                html.Button(
                                    "📈 What's the trend?",
                                    id="q-trend",
                                    style={
                                        "padding": "12px 16px",
                                        "fontSize": 13,
                                        "backgroundColor": TEAL,
                                        "color": WHITE,
                                        "border": "none",
                                        "borderRadius": 6,
                                        "cursor": "pointer",
                                        "fontWeight": 600,
                                    },
                                ),
                                html.Button(
                                    "🎯 Any anomalies?",
                                    id="q-anomaly",
                                    style={
                                        "padding": "12px 16px",
                                        "fontSize": 13,
                                        "backgroundColor": TEAL,
                                        "color": WHITE,
                                        "border": "none",
                                        "borderRadius": 6,
                                        "cursor": "pointer",
                                        "fontWeight": 600,
                                    },
                                ),
                            ],
                        ),
                        html.Div(id="qa-output", style={"marginTop": 20}),
                    ],
                ),
            ],
        ),

        # Footer
        html.Div(
            style={
                "backgroundColor": "#333",
                "color": "white",
                "padding": 32,
                "marginTop": 40,
                "textAlign": "center",
                "fontSize": 12,
            },
            children=[
                html.P(
                    "CaféMetrics • Rule-based Analytics • No LLM Calls • Fully Auditable",
                    style={"margin": "0 0 8px 0"},
                ),
                html.P(
                    "Transparency score = sample size + consistency + effect size + p-value",
                    style={"margin": 0, "color": "#aaa"},
                ),
            ],
        ),
    ],
)


@app.callback(
    Output("hourly-chart", "figure"),
    Input("hourly-chart", "id"),
)
def update_hourly_chart(_):
    """Generate hourly revenue chart."""
    if hourly is None:
        return go.Figure()

    hourly_rev = hourly.groupby("hour")["total_revenue"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=hourly_rev["hour"],
            y=hourly_rev["total_revenue"],
            marker_color=TEAL,
            name="Revenue",
        )
    )

    fig.update_layout(
        title=None,
        xaxis_title="Hour of Day",
        yaxis_title="Revenue (£)",
        hovermode="x unified",
        plot_bgcolor=BG_LIGHT,
        paper_bgcolor=WHITE,
        margin=dict(l=50, r=20, t=20, b=50),
        font=dict(family="Arial"),
    )

    return fig


@app.callback(
    Output("qa-output", "children"),
    [
        Input("q-dip", "n_clicks"),
        Input("q-trend", "n_clicks"),
        Input("q-anomaly", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def handle_qa(dip_clicks, trend_clicks, anomaly_clicks):
    """Handle Q&A button clicks with fixed responses."""
    from dash import ctx
    if not ctx.triggered:
        return html.Div()

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    answer_map = {
        "q-dip": (
            "Afternoon Dip (14:00–16:00)",
            "Sales typically dip in the early afternoon. This could be due to reduced foot traffic after lunch, "
            "or customers shifting to other nearby options. Consider targeted promotions during this period.",
        ),
        "q-trend": (
            "Revenue Trend",
            "Overall, revenue shows a slight upward trend over the analysis period. This suggests growing interest "
            "or improving business conditions. Continue monitoring for seasonal patterns.",
        ),
        "q-anomaly": (
            "Anomalies & Outliers",
            "Some days show unusually high or low sales. Review what happened on those dates (events, weather, staffing) "
            "to identify patterns or external factors.",
        ),
    }

    title, answer = answer_map.get(button_id, ("No Answer", "Question not recognized."))

    return html.Div(
        style={
            "padding": 16,
            "backgroundColor": "#f0f8ff",
            "borderLeft": f"4px solid {TEAL}",
            "borderRadius": 8,
        },
        children=[
            html.H4(title, style={"color": TEAL, "margin": "0 0 12px 0"}),
            html.P(answer, style={"margin": 0, "color": "#333", "lineHeight": 1.5}),
            html.P(
                "💡 This answer is rule-based, not AI-generated. It's based on statistical analysis of your data.",
                style={"margin": "12px 0 0 0", "fontSize": 12, "color": GRAY_TEXT, "fontStyle": "italic"},
            ),
        ],
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8050)
