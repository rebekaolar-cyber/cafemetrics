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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import analyze_all
from src.answers import Answer, anomaly_answer, dip_answer, trend_answer
from src.confidence import WEIGHTS, confidence_breakdown
from src.data_loader import aggregate_by_hour, get_data_span, load_sales_data
from src.ingest import MAX_UPLOAD_BYTES, LoadReport, read_upload
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


# ---------------------------------------------------------------- the dataset on show -------
#
# The page can show two things: the sample data, loaded above and always
# available so the live demo works on a cold start, or a file somebody has just
# uploaded. Both are a Dataset. The module-level `df`, `metrics`, `insights`,
# `hourly`, `analysis` and `trading_days` stay bound to the sample for anything
# that reads them directly.
#
# CURRENT is process-wide: one visitor's upload changes what the next request
# renders. That is wrong for a multi-user service and fine for a one-instance
# demo with one dyno and no login, which is what this is. Uploaded findings are
# never written to the audit database and carry no Verify/Dismiss buttons — the
# audit trail is about this café's sample, not a stranger's file.


@dataclass
class Dataset:
    """Everything the page needs about one set of sales."""

    df: object
    metrics: dict
    insights: list
    hourly: object
    analysis: Optional[dict]
    trading_days: int
    label: str
    verifiable: bool = False
    report: Optional[LoadReport] = None


SAMPLE = Dataset(
    df=df, metrics=metrics, insights=insights, hourly=hourly, analysis=analysis,
    trading_days=trading_days, label="Sample data", verifiable=True,
)
CURRENT = SAMPLE


def build_dataset(frame, label: str, report: LoadReport) -> Dataset:
    """Run the whole analysis over an uploaded frame.

    The guard below is a backstop, not a workaround for a known bug. It was
    written for one: a café that shut at two o'clock gave the afternoon-dip
    t-test two series of different lengths and scipy raised "unequal length
    arrays". That is fixed in src/analysis.py — the two series are inner-joined
    on the date, and days with nothing to pair against are counted and reported
    rather than crashing the comparison — and no input has been found since that
    reaches this except clause, including the uneven files it was written for.

    It stays anyway. This is the one path in the app that runs arbitrary data
    from a stranger through four statistical modules, and a traceback on the
    page is a much worse failure than a missing finding. The cost is a few lines;
    the thing it defends against is the rule nobody has thought of yet. If it
    ever does trip, the totals and the chart still render and the report says
    the findings could not be worked out, naming the exception type.
    """
    _, days = get_data_span(frame)
    analysis_result = None
    found = []
    try:
        analysis_result = analyze_all(frame)
        found = generate_all_insights(frame)
    except Exception as error:  # noqa: BLE001 - anything at all, from any rule
        report.warnings.append(
            "The totals and the chart below are from your file, but the pattern rules "
            f"could not be run on it ({type(error).__name__}). That usually means the "
            "data is too thin or too uneven for the statistics — too few days, or hours "
            "that appear on some days and not others."
        )
    return Dataset(
        df=frame,
        metrics=summary_metrics(frame),
        insights=found,
        hourly=aggregate_by_hour(frame),
        analysis=analysis_result,
        trading_days=days,
        label=label,
        verifiable=False,
        report=report,
    )


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


def till_strip(data: Optional[Dataset] = None):
    data = data or CURRENT
    figures = data.metrics
    return html.Section(
        className="paper strip figures",
        children=[
            figure(f"£{figures.get('total_revenue', 0):,.0f}", f"taken in {data.trading_days} days"),
            figure(f"£{figures.get('avg_daily_revenue', 0):,.0f}", "on an average day"),
            figure(f"{figures.get('total_transactions', 0):,}", "sales"),
            figure(f"{int(figures.get('peak_hour', 0)):02d}:00", "the busiest hour"),
        ],
    )


def dip_hours(data: Optional[Dataset] = None):
    """The hours the afternoon-dip rule checks, if it found a dip."""
    data = data or CURRENT
    dip = data.analysis["afternoon_dip"] if data.analysis else None
    return dip["dip_hours"] if dip and dip["detected"] else []


def hourly_figure(data: Optional[Dataset] = None):
    """Revenue by hour of the day, summed over the whole period."""
    data = data or CURRENT
    if data.hourly is None:
        return go.Figure()

    by_hour = data.hourly.groupby("hour")["total_revenue"].sum()
    marked = dip_hours(data)
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


def chart_ticket(data: Optional[Dataset] = None):
    data = data or CURRENT
    marked = dip_hours(data)
    key = [html.Span([html.I(), "revenue in that hour, all days added up"])]
    if marked:
        window = f"{min(marked):02d}:00 to {max(marked) + 1:02d}:00"
        key.append(html.Span([html.I(className="hot"), f"{window}, the window the dip rule checks"]))
    return html.Section(
        className="paper",
        children=[
            html.Div(className="key", children=key),
            dcc.Graph(id="hourly-chart", figure=hourly_figure(data), config={"displayModeBar": False}),
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


def ticket_foot(insight, verifiable: bool = True):
    """Verify and Dismiss while the finding is open; the stamp once it has been judged."""
    if not verifiable:
        # An uploaded file is not this cafe's own data: nothing to confirm against,
        # and nothing that belongs in the audit database.
        return [html.Span("From the file you uploaded, not saved.", className="ask")]
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


def ticket_body(insight, verifiable: bool = True):
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
        html.Div(className="ticket-foot", children=ticket_foot(insight, verifiable)),
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
            html.P(
                "Every finding is scored out of 100 by the same four rules. No model, no AI. "
                "A finding that rests on no statistical test, like the unusual-day scan, is "
                "scored on the first three, rescaled to total 100."
            ),
            html.Div(className="receipt", children=rows),
            html.P("85 or more is high, 60 to 84 medium, under 60 low.", className="basis"),
        ],
    )


def findings(data: Optional[Dataset] = None):
    data = data or CURRENT
    if not data.insights:
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
            id=({"type": "insight-card", "index": insight.pattern_type}
                if data.verifiable else f"upload-card-{insight.pattern_type}"),
            className=ticket_class(insight),
            children=ticket_body(insight, data.verifiable),
        )
        for insight in data.insights
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


def period(data: Optional[Dataset] = None):
    data = data or CURRENT
    if data.df is None:
        return ""
    first, last = data.df["date"].min(), data.df["date"].max()
    return f"{first.day} {first:%B} to {last.day} {last:%B %Y}"



# ---------------------------------------------------------------- upload --------------------

MAX_UPLOAD_MB = MAX_UPLOAD_BYTES / 1024 / 1024


def upload_card():
    """Drop your own till export here. Nothing is written to disk."""
    return html.Section(
        className="paper upload",
        children=[
            html.H3("Try it on your own sales"),
            html.P(
                "Drop a CSV export from your till. English or Dutch headers, commas or "
                "semicolons, comma decimals — it works all of those out and tells you what "
                "it did. Nothing is uploaded to a database or kept: the file is read in "
                f"memory and forgotten. Up to {MAX_UPLOAD_MB:.0f} MB.",
                className="section-note",
            ),
            dcc.Upload(
                id="sales-upload",
                className="dropzone",
                accept=".csv,text/csv",
                multiple=False,
                max_size=MAX_UPLOAD_BYTES,
                children=html.Div([
                    html.B("Choose a CSV"),
                    html.Span(" or drag it here"),
                ]),
            ),
            html.Div(id="load-report", **{"aria-live": "polite"}),
            html.Button("Back to the sample data", id="use-sample", className="btn"),
        ],
    )


def report_panel(report: LoadReport, label: str):
    """The validation report, printed like everything else on this page."""
    rows = []

    def line(text, css="report-line"):
        rows.append(html.P(text, className=css))

    head = "What was read" if report.ok else "That file could not be used"
    line(report.summary, "finding")
    if report.how_it_was_read():
        line(report.how_it_was_read())
    for error in report.errors:
        line(error, "report-line bad")
    for warning in report.warnings:
        line(warning, "report-line warn")

    if report.columns:
        rows.append(html.H4("Columns"))
        rows.append(html.Ul([html.Li(str(note)) for note in report.columns]))
    if report.ignored:
        rows.append(html.H4("Ignored"))
        rows.append(html.Ul([html.Li(name) for name in report.ignored]))
    if report.drops:
        rows.append(html.H4(f"Rows dropped ({report.rows_dropped})"))
        rows.append(html.Ul([html.Li(str(drop)) for drop in report.drops]))

    return html.Div(
        className=f"report {'ok' if report.ok else 'bad'}",
        children=[html.H3(head), html.P(label, className="basis")] + rows,
    )


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
                    upload_card(),

                    html.Div(id="source-note", className="source-note",
                             children=f"Showing: {CURRENT.label}"),

                    html.Div(id="till-container", children=till_strip()),

                    html.H2("The day, hour by hour", className="section-title"),
                    html.Div(id="chart-container", children=chart_ticket()),

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
    current = CURRENT.analysis
    if current is None:
        raise PreventUpdate

    answers = {
        "q-dip": lambda: dip_answer(current["afternoon_dip"]),
        "q-trend": lambda: trend_answer(current["trend"]),
        "q-anomaly": lambda: anomaly_answer(current["anomalies"]),
    }
    if ctx.triggered_id not in answers:
        raise PreventUpdate
    return answer_block(answers[ctx.triggered_id]())


@app.callback(
    Output("source-note", "children"),
    Output("till-container", "children"),
    Output("chart-container", "children"),
    Output("insights-container", "children"),
    Output("load-report", "children"),
    Output("qa-output", "children", allow_duplicate=True),
    Input("sales-upload", "contents"),
    Input("use-sample", "n_clicks"),
    State("sales-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, sample_clicks, filename):
    """Read an uploaded CSV from memory and redraw the page from it.

    Nothing here touches the filesystem: Dash hands over the file as a base64
    data URL and it is parsed straight out of memory, so data/sample_sales.csv
    can never be overwritten by an upload.

    The work is in show_sample() and apply_upload() so both can be exercised
    without a live callback context.
    """
    if ctx.triggered_id == "use-sample":
        return show_sample()
    if not contents:
        raise PreventUpdate
    return apply_upload(contents, filename)


def rendered(data: Dataset, report_children=None, note: Optional[str] = None):
    """The five pieces of page the upload callback replaces, plus a cleared answer."""
    return (
        note if note is not None else f"Showing: {data.label}",
        till_strip(data),
        chart_ticket(data),
        findings(data),
        report_children,
        None,
    )


def show_sample():
    """Put the sample data back on the page."""
    global CURRENT
    CURRENT = SAMPLE
    return rendered(SAMPLE)


def apply_upload(contents, filename: Optional[str] = None):
    """Parse an upload and return the page pieces. Never raises on bad input."""
    global CURRENT
    frame, report = read_upload(contents, filename)
    label = filename or "your file"
    if frame is None:
        # Whatever was on the page stays there; only the report changes.
        return rendered(CURRENT, report_panel(report, label))
    # build_dataset can add to the report, so the panel is drawn after it, not before.
    CURRENT = build_dataset(frame, label, report)
    return rendered(
        CURRENT, report_panel(report, label), f"Showing: {CURRENT.label} — {period(CURRENT)}"
    )


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

    for insight in SAMPLE.insights:
        if insight.pattern_type == pattern_type:
            db_id = insight_db_cache.get(pattern_type)
            if db_id:
                update_insight_status(db_id, new_status)
            insight.status = new_status  # so the buttons give way to the stamp
            break

    return ([ticket_body(insight) for insight in SAMPLE.insights],
            [ticket_class(insight) for insight in SAMPLE.insights])


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
