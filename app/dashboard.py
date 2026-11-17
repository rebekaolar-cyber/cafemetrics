"""CaféMetrics dashboard: transparent, auditable café sales analytics.

The page, top to bottom:
- masthead: what this is and which data it reads
- the till strip: four totals
- revenue by hour of the day, with the afternoon window the dip rule checks marked
- the findings, one ticket each, with the confidence score itemised like a receipt
- three fixed questions, answered from the analysis (src/answers.py), never from canned text

How it looks is in assets/style.css. This file is structure and callbacks.
"""

import os
import sys
from pathlib import Path

import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, ctx, dcc, html
from dash.exceptions import PreventUpdate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import analyze_all
from src.answers import Answer, anomaly_answer, dip_answer, trend_answer
from src.confidence import WEIGHTS, confidence_breakdown
from src.data_loader import aggregate_by_hour, get_data_span, load_sales_data
from src.insights import generate_all_insights
from src.metrics import summary_metrics
from src.verification import get_insights, init_db, save_insight, update_insight_status

REPO = "https://github.com/rebekaolar-cyber/cafemetrics"
DATA_FILE = "data/sample_sales.csv"
BAR, HOT, INK, INK2, RULE = "#0a6c71", "#e96a24", "#14302f", "#45605e", "#c5d1ce"
FONT = "Recursive, system-ui, sans-serif"

# Load data with clear error reporting
DATA_LOAD_ERROR = None
df = None
metrics = {}
insights = []
hourly = None
analysis = None
trading_days = 0
try:
    df = load_sales_data(DATA_FILE)
    metrics = summary_metrics(df)
    insights = generate_all_insights(df)
    hourly = aggregate_by_hour(df)
    analysis = analyze_all(df)
    _, trading_days = get_data_span(df)
except (FileNotFoundError, ValueError) as e:
    DATA_LOAD_ERROR = str(e)
    print(f"Data loading failed: {DATA_LOAD_ERROR}")
except Exception as e:
    # Unexpected error: log it, but don't show a full traceback to the user
    DATA_LOAD_ERROR = (
        f"Unexpected error loading data. Check logs for details.\n"
        f"Error type: {type(e).__name__}"
    )
    print(f"Data loading failed: {DATA_LOAD_ERROR}")
    print(f"   Full error: {e}")

# Initialize database and manage insights
init_db()

# Save each insight to the database, or pick up the status it already has there
insight_db_cache = {}
if insights:
    for insight in insights:
        existing = get_insights()
        existing_insight = next(
            (i for i in existing if i.get("pattern_type") == insight.pattern_type),
            None,
        )
        if existing_insight:
            insight_db_cache[insight.pattern_type] = existing_insight["id"]
            insight.status = existing_insight["status"]
        else:
            db_id = save_insight(
                title=insight.title,
                pattern_type=insight.pattern_type,
                confidence_score=insight.confidence_score,
                status="pending",
            )
            insight_db_cache[insight.pattern_type] = db_id
            insight.status = "pending"


# ---------------------------------------------------------------- building blocks ------------

BAND_WORDS = {"HIGH": "High confidence", "MEDIUM": "Medium confidence", "LOW": "Low confidence"}
WHY_IT_COUNTS = {
    "Sample size": "more days, less chance; full marks from 30 days",
    "Consistency": "the share of days the pattern shows up",
    "Effect size": "how big the difference is; full marks from 1.25",
    "Significance": "only counts when p is below 0.05",
}


def figure(value, label):
    """One total on the till strip."""
    return html.Div(className="figure", children=[html.B(value), html.Span(label)])


def till_strip():
    return html.Section(
        className="paper strip figures",
        children=[
            figure(f"£{metrics.get('total_revenue', 0):,.0f}", f"taken in {trading_days} days"),
            figure(f"£{metrics.get('avg_daily_revenue', 0):,.0f}", "on an average day"),
            figure(f"{metrics.get('total_transactions', 0):,}", "sales"),
            figure(f"{int(metrics.get('peak_hour', 0)):02d}:00", "the busiest hour"),
        ],
    )


def dip_hours():
    """The hours the afternoon-dip rule checks, if it found a dip."""
    dip = analysis["afternoon_dip"] if analysis else None
    return dip["dip_hours"] if dip and dip["detected"] else []


def hourly_figure():
    """Revenue by hour of the day, summed over the whole period."""
    if hourly is None:
        return go.Figure()

    by_hour = hourly.groupby("hour")["total_revenue"].sum()
    marked = dip_hours()
    peak = by_hour.idxmax()

    fig = go.Figure(
        go.Bar(
            x=by_hour.index,
            y=by_hour.values,
            marker_color=[HOT if hour in marked else BAR for hour in by_hour.index],
            width=0.62,
            text=[f"£{value / 1000:.1f}k" if hour == peak else "" for hour, value in by_hour.items()],
            textposition="outside",
            textfont=dict(color=INK, size=14),
            constraintext="none",
            customdata=[f"{hour:02d}:00 to {hour + 1:02d}:00" for hour in by_hour.index],
            hovertemplate="%{customdata}<br>£%{y:,.0f}<extra></extra>",
            cliponaxis=False,
        )
    )
    fig.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=28, b=8),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK2),
        showlegend=False,
        hoverlabel=dict(font=dict(family=FONT, size=13), bgcolor="#fffdf7", bordercolor=RULE),
        bargap=0.3,
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(by_hour.index),
        ticktext=[f"{hour:02d}" for hour in by_hour.index],
        title_text="hour of the day",
        title_font=dict(size=12),
        showgrid=False,
        linecolor=RULE,
        ticks="",
        fixedrange=True,
    )
    fig.update_yaxes(
        tickprefix="£", tickformat="~s", gridcolor="#e3eae8", zeroline=False, ticks="",
        fixedrange=True, rangemode="tozero",
    )
    return fig


def chart_ticket():
    marked = dip_hours()
    key = [html.Span([html.I(), "revenue in that hour, all days added up"])]
    if marked:
        window = f"{min(marked):02d}:00 to {max(marked) + 1:02d}:00"
        key.append(html.Span([html.I(className="hot"), f"{window}, the window the dip rule checks"]))
    return html.Section(
        className="paper",
        children=[
            html.Div(className="key", children=key),
            dcc.Graph(id="hourly-chart", figure=hourly_figure(), config={"displayModeBar": False}),
        ],
    )


def receipt(insight):
    """The confidence score itemised: points earned by each of its four parts."""
    if not insight.evidence:
        return None
    rows = []
    for part in confidence_breakdown(**insight.evidence):
        share = part["points"] / part["max_points"] * 100
        rows.append(
            html.Div(
                className="row",
                children=[
                    html.Span(part["name"], className="name"),
                    html.Span(f"{part['points']:.1f} of {part['max_points']:.0f}", className="points"),
                    html.Div(html.I(style={"width": f"{share:.0f}%"}), className="meter"),
                ],
            )
        )
    rows.append(
        html.Div(
            className="row total",
            children=[
                html.Span("Confidence", className="name"),
                html.Span(f"{insight.confidence_score} of 100", className="points"),
            ],
        )
    )
    return html.Div(className="receipt", children=rows)


def ticket_foot(insight):
    """Verify and Dismiss while the finding is open; the stamp once it has been judged."""
    if insight.status == "verified":
        return [html.Span("Verified", className="stamp")]
    if insight.status == "dismissed":
        return [html.Span("Dismissed", className="stamp no")]
    return [
        html.Span("Does this match what you see in the café?", className="ask"),
        html.Button(
            "Verify",
            id={"type": "verify-btn", "index": insight.pattern_type},
            className="btn primary",
        ),
        html.Button(
            "Dismiss",
            id={"type": "dismiss-btn", "index": insight.pattern_type},
            className="btn",
        ),
    ]


def ticket_body(insight):
    """Everything printed on one finding's ticket."""
    return [
        html.Div(
            className="ticket-head",
            children=[
                html.Div([
                    html.H3(insight.title),
                    html.P(BAND_WORDS.get(insight.confidence_band, insight.confidence_band), className="band"),
                ]),
                html.Div(className="score", children=[html.B(str(insight.confidence_score)), " of 100"]),
            ],
        ),
        html.P(insight.text, className="finding"),
        receipt(insight),
        html.P(f"Based on {insight.basis}.", className="basis"),
        html.P([html.B("What to do"), insight.suggested_action], className="action"),
        html.Div(className="ticket-foot", children=ticket_foot(insight)),
    ]


def ticket_class(insight):
    return f"paper ticket is-{insight.status}"


def scoring_ticket():
    """How every confidence score on the page is put together."""
    rows = [
        html.Div(
            className="row",
            children=[
                html.Span(name, className="name"),
                html.Span(f"up to {weight * 100:.0f}", className="points"),
                html.Span(WHY_IT_COUNTS[name], className="why"),
            ],
        )
        for name, weight in WEIGHTS.items()
    ]
    rows.append(
        html.Div(
            className="row total",
            children=[html.Span("Confidence", className="name"), html.Span("up to 100", className="points")],
        )
    )
    return html.Aside(
        className="paper ticket how",
        children=[
            html.H3("How a score is added up"),
            html.P("Every finding is scored out of 100 by the same four rules. No model, no AI."),
            html.Div(className="receipt", children=rows),
            html.P("85 or more is high, 60 to 84 medium, under 60 low.", className="basis"),
        ],
    )


def findings():
    if not insights:
        note = html.Div(
            className="paper ticket",
            children=[
                html.H3("Nothing to report"),
                html.P(
                    "The rules found no dip, trend or unusual day in this data. "
                    "The questions below show the numbers behind that.",
                    className="finding",
                ),
            ],
        )
        return [note, scoring_ticket()]
    tickets = [
        html.Article(
            id={"type": "insight-card", "index": insight.pattern_type},
            className=ticket_class(insight),
            children=ticket_body(insight),
        )
        for insight in insights
    ]
    return tickets + [scoring_ticket()]


def answer_block(answer: Answer):
    return html.Div(
        className="paper answer",
        children=[
            html.H3(answer.title),
            html.P(answer.finding, className="finding"),
            html.P(answer.evidence, className="evidence"),
            html.P("Filled in from this data by a fixed template. No AI wrote it.", className="source"),
        ],
    )


def period():
    if df is None:
        return ""
    first, last = df["date"].min(), df["date"].max()
    return f"{first.day} {first:%B} to {last.day} {last:%B %Y}"


# ---------------------------------------------------------------- page ----------------------

app = Dash(
    __name__,
    title="CaféMetrics",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # For Gunicorn

def page():
    """The whole page. Built on every visit, so a reload shows each finding's current stamp."""
    return html.Div(
        className="page",
        children=[
            # Hidden store for insight metadata (pattern_type -> db_id mapping)
            dcc.Store(id="insight-store", data=insight_db_cache),

            html.Header(
                className="mast",
                children=[
                    html.Div(
                        className="mast-top",
                        children=[
                            html.B("CaféMetrics"),
                            html.Span(f"Sample data, generated: {period()}") if df is not None else None,
                            html.A("Code on GitHub", href=REPO, target="_blank", rel="noopener",
                                   className="code-link"),
                        ],
                    ),
                    html.H1("What do the sales say?"),
                    html.P(
                        "Café sales read by fixed rules, not by AI. Every finding shows how sure it "
                        "is, and exactly where that number comes from.",
                        className="lede",
                    ),
                ],
            ),

            # Error banner (shown if data loading failed)
            (html.Div(
                className="error",
                children=[html.H2("The sales file could not be loaded"), html.P(DATA_LOAD_ERROR)],
            ) if DATA_LOAD_ERROR else None),

            html.Main(
                children=[
                    till_strip(),

                    html.H2("The day, hour by hour", className="section-title"),
                    chart_ticket(),

                    html.H2("What the rules found", className="section-title"),
                    html.Div(id="insights-container", className="tickets", children=findings()),

                    html.H2("Ask the data", className="section-title"),
                    html.P(
                        "Three fixed questions. Each answer is worked out from the sales file as you ask.",
                        className="section-note",
                    ),
                    html.Div(
                        className="questions",
                        children=[
                            html.Button("Is there an afternoon dip?", id="q-dip", className="btn"),
                            html.Button("Is revenue rising or falling?", id="q-trend", className="btn"),
                            html.Button("Did any day stand out?", id="q-anomaly", className="btn"),
                        ],
                    ),
                    html.Div(id="qa-output", **{"aria-live": "polite"}),
                ],
            ),

            html.Footer(
                className="foot",
                children=[
                    "Built by Rebeka Olar. Rule-based: no AI is called, and every number on this page "
                    "can be recomputed from the sales file. ",
                    html.A("Code and tests on GitHub", href=REPO),
                    ".",
                ],
            ),
        ],
    )


app.layout = page


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
    """Answer the clicked question from the analysis of the loaded data."""
    if analysis is None:
        raise PreventUpdate

    answers = {
        "q-dip": lambda: dip_answer(analysis["afternoon_dip"]),
        "q-trend": lambda: trend_answer(analysis["trend"]),
        "q-anomaly": lambda: anomaly_answer(analysis["anomalies"]),
    }
    if ctx.triggered_id not in answers:
        raise PreventUpdate
    return answer_block(answers[ctx.triggered_id]())


@app.callback(
    Output({"type": "insight-card", "index": ALL}, "children"),
    Output({"type": "insight-card", "index": ALL}, "className"),
    [
        Input({"type": "verify-btn", "index": ALL}, "n_clicks"),
        Input({"type": "dismiss-btn", "index": ALL}, "n_clicks"),
    ],
    prevent_initial_call=True,
)
def handle_insight_action(verify_clicks, dismiss_clicks):
    """
    Handle Verify/Dismiss button clicks for insights.

    - Input: clicks on any button matching {"type": "verify-btn" / "dismiss-btn"}
    - Output: every insight ticket matching {"type": "insight-card"}, redrawn
    - ctx.triggered_id: which button was clicked, e.g.
      {"type": "verify-btn", "index": "afternoon_dip"}
    """
    button = ctx.triggered_id
    # Redrawing the tickets creates new buttons, which also fires this callback (with no click).
    if not button or not ctx.triggered[0]["value"]:
        raise PreventUpdate

    pattern_type = button["index"]
    new_status = "verified" if button["type"] == "verify-btn" else "dismissed"

    for insight in insights:
        if insight.pattern_type == pattern_type:
            db_id = insight_db_cache.get(pattern_type)
            if db_id:
                update_insight_status(db_id, new_status)
            insight.status = new_status  # so the buttons give way to the stamp
            break

    return [ticket_body(insight) for insight in insights], [ticket_class(insight) for insight in insights]


if __name__ == "__main__":
    # Configuration from environment variables (safe defaults for production)
    # DEBUG: NEVER True in production. Flask/Dash debug mode exposes an interactive
    # console (Werkzeug debugger) that can execute arbitrary Python code on the server.
    # Only enable locally for development. Default is False.
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    # PORT: Server port. Defaults to 8050 (Dash convention).
    # Production deployments often run behind a reverse proxy on port 80/443.
    port = int(os.environ.get("PORT", 8050))

    # HOST: Bind to all interfaces (0.0.0.0) to allow external connections.
    # Behind a reverse proxy, this is safe. Direct exposure requires firewall.
    host = os.environ.get("HOST", "127.0.0.1")

    if debug:
        print("WARNING: Debug mode is ENABLED. Never use in production.")

    app.run(debug=debug, host=host, port=port)
