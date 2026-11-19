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
import re
import secrets
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from flask import has_request_context, jsonify, request

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis import analyze_all
from src.answers import Answer, anomaly_answer, dip_answer, trend_answer
from src.confidence import WEIGHTS, confidence_breakdown
from src.data_loader import aggregate_by_hour, get_data_span, load_sales_data
from src.ingest import MAX_UPLOAD_BYTES, LoadReport, read_upload
from src.insights import generate_all_insights
from src.metrics import summary_metrics
from src.session_state import SESSION_TTL_SECONDS, SessionStore
from src.verification import get_insights, init_db, save_insight

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

# ---------------------------------------------------------------- the audit database --------
#
# The SQLite audit trail is a local-use feature: it records this café's own
# findings so a run can be compared with the last one. The web app does not
# depend on it. Nothing below is required for the page to render, and nothing a
# visitor does writes to it.
#
# It used to be the one import-time path with no handling. init_db() opens a
# file in the working directory, and on a read-only filesystem sqlite3 raises
# "unable to open database file" while the module is still being imported: the
# gunicorn worker exits 3 and the service dies with "Worker failed to boot"
# instead of degrading. A deploy audit reproduced exactly that. Render's disk is
# writable today, so it was latent, but a latent crash that takes the whole
# service down is still a crash. It gets the same treatment as the data load:
# catch everything, say so on the page, carry on.

DB_INIT_ERROR = None
insight_db_cache = {}
try:
    init_db()
    # Save each insight to the database, or pick up the status it already has there
    if insights:
        existing = get_insights()
        for insight in insights:
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
except Exception as e:  # noqa: BLE001 - sqlite3, OSError, anything: never at import
    DB_INIT_ERROR = (
        "The local audit database could not be opened "
        f"({type(e).__name__}). Everything on this page still works; findings "
        "are not being recorded to it."
    )
    insight_db_cache = {}
    print(f"Audit database unavailable: {DB_INIT_ERROR}")
    print(f"   Full error: {e}")


# ---------------------------------------------------------------- the dataset on show -------
#
# The page can show two things: the sample data, loaded above and always
# available so the live demo works on a cold start, or a file somebody has just
# uploaded. Both are a Dataset. The module-level `df`, `metrics`, `insights`,
# `hourly`, `analysis` and `trading_days` stay bound to the sample for anything
# that reads them directly.
#
# What a visitor is looking at is per-visitor, held in STATE below and keyed by
# an opaque cookie. There is no process-wide "current dataset" any more: an
# upload changes that visitor's page and nobody else's. Uploaded findings are
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


# ---------------------------------------------------------------- who is asking -------------
#
# Session identity is an opaque 256-bit token in an HttpOnly cookie, minted
# server-side on the first request that needs one (see _session_id below).
#
# Why a cookie and not a dcc.Store: a Store lives in the browser and is written
# by page JavaScript, so it is a value the client chooses. The cookie is set by
# the server, is HttpOnly so page scripts cannot read or write it, and carries
# secrets.token_urlsafe(32) — 256 bits from the OS CSPRNG. A visitor cannot
# guess another visitor's id, and ids are never derived from anything a client
# sends (no IP, no User-Agent, no header). A client can of course send some
# other value in its own cookie jar, but without the token it can only name a
# session that does not exist, which reads as "fresh visitor": the sample.
#
# Two tabs in the same browser share the cookie, so they share one session:
# uploading in one tab and reloading the other shows the same file. That is the
# choice made here — one browser is one person, and a per-tab id would mean the
# same person's second tab silently showed the sample instead of their data.
# Two different browsers, two different devices, or a private window are
# separate sessions and cannot see each other.

SESSION_COOKIE = "cafemetrics_sid"
SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9_-]{22,128}\Z")
# Where the resolved id is stashed for the life of one request. request.environ
# is per-request by construction; flask.g is not (see _session_id).
SESSION_ENVIRON_KEY = "cafemetrics.session_id"
SESSION_NEW_ENVIRON_KEY = "cafemetrics.session_id_is_new"

STATE = SessionStore()

# The store is for web requests. Tests, scripts and anything else that calls
# these functions outside a request context get this single off-request slot
# instead, so a direct apply_upload() still behaves like a session of one. It is
# never reachable from a browser: every path to it is guarded by
# has_request_context().


@dataclass
class _OffRequest:
    dataset: Dataset
    statuses: Dict[str, str] = field(default_factory=dict)


OFF_REQUEST = _OffRequest(dataset=SAMPLE)
# Local use keeps the audit database's stamps; a visitor's page never does.
OFF_REQUEST.statuses = {
    insight.pattern_type: insight.status for insight in (insights or [])
}

DROPPED_NOTE = (
    "the file you uploaded is no longer held in memory — nothing is saved to "
    "disk, and an idle visit is forgotten after "
    f"{int(SESSION_TTL_SECONDS // 60)} minutes. Upload it again to see it."
)


def _session_id() -> Optional[str]:
    """This visitor's session id, minted if they have none. None off-request.

    The id is stashed in ``request.environ``, which the WSGI server builds
    fresh for every request and throws away with it. It is deliberately *not*
    on ``flask.g``: g belongs to the application context, which outlives a
    request whenever one is pushed around several (``with app.app_context():``
    around two requests, some test harnesses, some CLI paths). A stash on g
    would survive into the second request and hand visitor B visitor A's
    session, with every isolation test still green — a guarantee that happens
    to hold rather than one that cannot break.

    The cookie is attached to the response by _attach_session_cookie below, so
    a request that mints an id and then fails still leaves the browser with a
    usable session.
    """
    if not has_request_context():
        return None
    environ = request.environ
    minted = environ.get(SESSION_ENVIRON_KEY)
    if minted:
        return minted
    try:
        sent = request.cookies.get(SESSION_COOKIE)
    except Exception:  # noqa: BLE001 - a malformed cookie header must not 500
        sent = None
    if isinstance(sent, str) and SESSION_ID_RE.match(sent):
        environ[SESSION_ENVIRON_KEY] = sent
        return sent
    fresh = secrets.token_urlsafe(32)
    environ[SESSION_ENVIRON_KEY] = fresh
    environ[SESSION_NEW_ENVIRON_KEY] = True
    return fresh


def session_view() -> Tuple[Dataset, Optional[str]]:
    """The dataset this visitor should see, and anything they should be told.

    Falling back is always safe: a session that has been evicted, has expired,
    or never existed gets the sample data. Never a traceback, never a blank
    page, and never another visitor's file.
    """
    session_id = _session_id()
    if session_id is None:
        return OFF_REQUEST.dataset, None
    try:
        held = STATE.dataset(session_id)
        if held is not None:
            return held, None
        if STATE.was_dropped(session_id):
            return SAMPLE, DROPPED_NOTE
    except Exception as error:  # noqa: BLE001 - the page renders whatever happens
        print(f"Session lookup failed ({type(error).__name__}): {error}")
    return SAMPLE, None


def session_dataset() -> Dataset:
    """Just the dataset, for the many render helpers that only need that."""
    return session_view()[0]


def session_statuses() -> Dict[str, str]:
    """This visitor's own Verify/Dismiss stamps. Never another visitor's."""
    session_id = _session_id()
    if session_id is None:
        return dict(OFF_REQUEST.statuses)
    try:
        return STATE.statuses(session_id)
    except Exception as error:  # noqa: BLE001
        print(f"Session stamps unavailable ({type(error).__name__}): {error}")
        return {}


def _dataset_bytes(data: Dataset) -> int:
    """What holding this dataset costs, measured rather than guessed.

    pandas reports its own deep usage for the two frames; the analysis dict and
    the findings are small fixed structures, charged a flat 256 KiB so they are
    never free.
    """
    total = 256 * 1024
    for frame in (data.df, data.hourly):
        try:
            total += int(frame.memory_usage(deep=True).sum())
        except Exception:  # noqa: BLE001 - not a frame, or a frame that cannot say
            total += 1024 * 1024
    return total


def remember_dataset(data: Dataset) -> bool:
    """Hold this dataset for whoever is asking. False if it was not kept."""
    session_id = _session_id()
    if session_id is None:
        OFF_REQUEST.dataset = data
        return True
    try:
        return STATE.put_dataset(session_id, data, _dataset_bytes(data))
    except Exception as error:  # noqa: BLE001
        print(f"Session store refused a dataset ({type(error).__name__}): {error}")
        return False


def forget_dataset() -> None:
    """Put the sample back for whoever is asking, keeping their stamps."""
    session_id = _session_id()
    if session_id is None:
        OFF_REQUEST.dataset = SAMPLE
        return
    try:
        STATE.clear_dataset(session_id)
    except Exception as error:  # noqa: BLE001
        print(f"Session clear failed ({type(error).__name__}): {error}")


def remember_status(pattern_type: str, status: str) -> None:
    """Record one stamp against the asking visitor alone."""
    session_id = _session_id()
    if session_id is None:
        OFF_REQUEST.statuses[pattern_type] = status
        return
    try:
        STATE.set_status(session_id, pattern_type, status)
    except Exception as error:  # noqa: BLE001
        print(f"Session stamp not kept ({type(error).__name__}): {error}")


def __getattr__(name):
    """`CURRENT` is no longer a variable; it is whatever the caller can see.

    Kept as a module attribute because it is the name every reader of this app
    knows, and because it keeps "what is on show" a read-only question with one
    answer. Three things to know about it:

    * this hook fires only for attribute access from outside
      (``dashboard.CURRENT``); a bare ``CURRENT`` inside this module raises
      NameError, which is what should happen to any future code reaching for a
      process-wide dataset again;
    * assigning to it is refused, because a real module global would shadow
      this hook permanently and silently restore the shared-state bug (see
      _Module below);
    * ``from app.dashboard import CURRENT`` still works and binds the *current
      caller's* dataset once, at import time, which is then stale forever.
      There is no way to intercept that, so: don't. Call session_dataset().
    """
    if name == "CURRENT":
        return session_dataset()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _Module(types.ModuleType):
    """This module's own type, so `dashboard.CURRENT = x` can be refused.

    Without this, one assignment anywhere — a test being helpful, a future
    callback "caching" the dataset — creates a module global that wins over
    __getattr__ for the life of the process and is shared by every visitor
    again, with no error and nothing to notice.
    """

    def __setattr__(self, name, value):
        if name == "CURRENT":
            raise AttributeError(
                "CURRENT is read-only: it is per-visitor state, not a global. "
                "Use remember_dataset(dataset) to hold one for the asking "
                "visitor, or forget_dataset() to put the sample back."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name == "CURRENT":
            raise AttributeError("CURRENT is read-only and is not a variable to delete.")
        super().__delattr__(name)


_this_module = sys.modules.get(__name__)
if _this_module is not None:  # a copy loaded by spec alone is not registered
    _this_module.__class__ = _Module


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
    data = data or session_dataset()
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
    data = data or session_dataset()
    dip = data.analysis["afternoon_dip"] if data.analysis else None
    return dip["dip_hours"] if dip and dip["detected"] else []


def hourly_figure(data: Optional[Dataset] = None):
    """Revenue by hour of the day, summed over the whole period."""
    data = data or session_dataset()
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
    data = data or session_dataset()
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


def stamp_of(insight, status: Optional[str] = None) -> str:
    """The stamp to print: the caller's, or the finding's own if none is given.

    A visitor's stamps come from their session and nowhere else. `status=None`
    means "no session in play" — a direct call from a test or a script — and
    falls back to the finding's own attribute.
    """
    return status if status is not None else insight.status


def ticket_foot(insight, verifiable: bool = True, status: Optional[str] = None):
    """Verify and Dismiss while the finding is open; the stamp once it has been judged."""
    if not verifiable:
        # An uploaded file is not this cafe's own data: nothing to confirm against,
        # and nothing that belongs in the audit database.
        return [html.Span("From the file you uploaded, not saved.", className="ask")]
    stamp = stamp_of(insight, status)
    if stamp == "verified":
        return [html.Span("Verified", className="stamp")]
    if stamp == "dismissed":
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


def ticket_body(insight, verifiable: bool = True, status: Optional[str] = None):
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
        html.Div(className="ticket-foot", children=ticket_foot(insight, verifiable, status)),
    ]


def ticket_class(insight, status: Optional[str] = None):
    return f"paper ticket is-{stamp_of(insight, status)}"


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


def findings(data: Optional[Dataset] = None, statuses: Optional[Dict[str, str]] = None):
    """The findings tickets, stamped with the asking visitor's own judgements.

    Stamps are read from the session, never from the shared Insight objects and
    never from the audit database: two strangers on the live site stamp the same
    sample findings without seeing each other's stamps. A fresh visitor starts
    with every finding pending, which is what an unjudged finding is.
    """
    data = data or session_dataset()
    if statuses is None:
        statuses = session_statuses() if data.verifiable else {}
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
            className=ticket_class(insight, statuses.get(insight.pattern_type, "pending")),
            children=ticket_body(
                insight, data.verifiable, statuses.get(insight.pattern_type, "pending")
            ),
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
    data = data or session_dataset()
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


def source_note(data: Dataset, notice: Optional[str] = None, head: Optional[str] = None):
    """The "Showing: ..." line, with anything the visitor needs telling after it."""
    line = head if head is not None else f"Showing: {data.label}"
    if not notice:
        return line
    return [line, f" — {notice}"]


# ---------------------------------------------------------------- page ----------------------

app = Dash(
    __name__,
    title="CaféMetrics",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # For Gunicorn

# ---------------------------------------------------------------- the request body ----------
#
# Nothing below this app reads a request body until Flask has already buffered
# it, JSON-parsed it and base64-decoded it. The 5 MB upload cap in src/ingest.py
# is checked before the pandas parse, which is the right place for it, but by
# then the bytes have been spent: a direct POST to /_dash-update-component with
# an oversized body was measured taking the worker from ~170 MB to 835 MB on an
# 80 MiB body and 1185 MB on 200 MiB, and inside a real 512 MB cgroup a 60 MB
# body — or a 50 MB body with two ordinary sessions already held — gets the
# worker SIGKILLed. The master respawns in about a second, so it self-heals and
# looks like a random outage, taking every held session with it.
#
# MAX_CONTENT_LENGTH makes Werkzeug refuse an oversized body on the
# Content-Length header, before any of it is read: measured 0.00 s and no RSS
# growth at all for 50 MB and 100 MB posts. 8 MiB is the ceiling because a
# legitimate 5 MB CSV arrives base64-encoded inside a JSON envelope and
# measured 6.7 MiB on the wire; browser visitors never meet the limit anyway,
# because dcc.Upload(max_size=...) refuses the file client-side first, so the
# only callers it affects are direct posts.
MAX_REQUEST_BYTES = 8 * 1024 * 1024
server.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES


@server.errorhandler(413)
def _body_too_large(error):
    """Say what happened in words, for whoever (or whatever) sent the body.

    Werkzeug may well have closed the connection before this is read, which is
    why the browser's own 5 MB check is the message a real visitor sees. This
    exists so the refusal is never a bare stack trace, and so a client that does
    read the response is told the number it has to come under.
    """
    print(f"Request body refused: over {MAX_REQUEST_BYTES} bytes")
    return (
        jsonify({
            "error": "request too large",
            "message": (
                f"That request body is over the "
                f"{MAX_REQUEST_BYTES / 1024 / 1024:.0f} MB limit, so it was "
                f"refused without being read. The sales file itself may be up to "
                f"{MAX_UPLOAD_MB:.0f} MB; a CSV arrives base64-encoded, which is "
                "about a third larger than the file."
            ),
        }),
        413,
    )


@server.after_request
def _attach_session_cookie(response):
    """Give a visitor who has no session id one, server-side.

    HttpOnly so no page script can read or write it; SameSite=Lax so another
    site cannot ride it; Secure whenever the request arrived over TLS, which on
    Render means reading the proxy's X-Forwarded-Proto rather than the socket
    (setting Secure on plain http would make the browser drop the cookie and
    every local visit would look like a new session).

    Any response that carries a Set-Cookie, and the layout response whose
    content is the visitor's own data, is marked no-store and Vary: Cookie.
    Render does not cache app responses, but a CDN in front of this that cached
    one Set-Cookie would hand every visitor the same session and recreate the
    exact bug this module exists to fix.
    """
    try:
        environ = request.environ
        session_id = environ.get(SESSION_ENVIRON_KEY)
        per_visitor = request.path == "/_dash-layout"
        if session_id and environ.get(SESSION_NEW_ENVIRON_KEY):
            forwarded = request.headers.get("X-Forwarded-Proto", "").split(",")[0].strip()
            response.set_cookie(
                SESSION_COOKIE,
                session_id,
                max_age=int(SESSION_TTL_SECONDS),
                httponly=True,
                samesite="Lax",
                secure=(forwarded == "https") or bool(request.is_secure),
                path="/",
            )
            per_visitor = True
        if per_visitor:
            response.headers["Cache-Control"] = "no-store"
            response.headers["Vary"] = "Cookie"
    except Exception as error:  # noqa: BLE001 - never turn a response into a 500
        print(f"Session cookie not set ({type(error).__name__}): {error}")
    return response

def page():
    """The whole page, or a plain message if it cannot be built.

    Dash evaluates this once at import (to validate the layout) and then on
    every visit. Both are places where an exception is expensive: at import it
    takes the gunicorn worker down before it can serve anything, and on a visit
    it is a 500 with a traceback. So the build is wrapped, like the data load
    and the audit database are, and a failure degrades to a readable page.
    """
    try:
        return _page()
    except Exception as error:  # noqa: BLE001 - a page is better than a stack trace
        print(f"Page could not be built ({type(error).__name__}): {error}")
        return html.Div(
            className="page",
            children=[
                html.Header(className="mast", children=[html.H1("CaféMetrics")]),
                html.Div(
                    className="error",
                    children=[
                        html.H2("This page could not be built"),
                        html.P(
                            "Something went wrong assembling the dashboard "
                            f"({type(error).__name__}). The server is up; try reloading. "
                            "Nothing you uploaded was saved anywhere."
                        ),
                        html.A("Code on GitHub", href=REPO),
                    ],
                ),
            ],
        )


def _page():
    """The whole page, built on every visit from the asking visitor's own state.

    Dash calls this per request, inside a Flask request context, so the dataset
    and the stamps below are this visitor's. A visitor with nothing of their own
    gets the sample, which is why a cold start always demos.
    """
    data, notice = session_view()
    statuses = session_statuses() if data.verifiable else {}
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
                            html.Span(f"Sample data, generated: {period(SAMPLE)}") if df is not None else None,
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

            # The audit database is optional; say so plainly if it is missing.
            (html.P(DB_INIT_ERROR, className="report-line warn") if DB_INIT_ERROR else None),

            html.Main(
                children=[
                    upload_card(),

                    html.Div(id="source-note", className="source-note",
                             children=source_note(data, notice)),

                    html.Div(id="till-container", children=till_strip(data)),

                    html.H2("The day, hour by hour", className="section-title"),
                    html.Div(id="chart-container", children=chart_ticket(data)),

                    html.H2("What the rules found", className="section-title"),
                    html.Div(id="insights-container", className="tickets",
                             children=findings(data, statuses)),

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
    """Answer the clicked question from the analysis of the asking visitor's data."""
    data, notice = session_view()
    current = data.analysis
    if current is None:
        raise PreventUpdate

    answers = {
        "q-dip": lambda: dip_answer(current["afternoon_dip"]),
        "q-trend": lambda: trend_answer(current["trend"]),
        "q-anomaly": lambda: anomaly_answer(current["anomalies"]),
    }
    if ctx.triggered_id not in answers:
        raise PreventUpdate
    block = answer_block(answers[ctx.triggered_id]())
    if not notice:
        return block
    # The visitor's file has gone; the answer is from the sample, and says so.
    return html.Div([html.P(f"Answered from the sample data: {notice}",
                            className="report-line warn"), block])


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


def rendered(
    data: Dataset,
    report_children=None,
    note: Optional[str] = None,
    notice: Optional[str] = None,
    statuses: Optional[Dict[str, str]] = None,
):
    """The five pieces of page the upload callback replaces, plus a cleared answer."""
    return (
        source_note(data, notice, note),
        till_strip(data),
        chart_ticket(data),
        findings(data, statuses),
        report_children,
        None,
    )


def show_sample():
    """Put the sample data back on the page, for this visitor only."""
    forget_dataset()
    return rendered(SAMPLE)


def apply_upload(contents, filename: Optional[str] = None):
    """Parse an upload and return the page pieces. Never raises on bad input.

    The dataset is held against the asking visitor's session, so it changes
    their page and nobody else's. If the store will not keep it — a file bigger
    than the whole cache allows — the page is still drawn from it once and the
    report says it was not kept, rather than pretending or crashing.
    """
    frame, report = read_upload(contents, filename)
    label = filename or "your file"
    if frame is None:
        # Whatever was on the page stays there; only the report changes.
        data, notice = session_view()
        return rendered(data, report_panel(report, label), notice=notice)
    # build_dataset can add to the report, so the panel is drawn after it, not before.
    data = build_dataset(frame, label, report)
    if not remember_dataset(data):
        report.warnings.append(
            "This page was drawn from your file, but the file was too large to "
            "keep in memory, so the next thing you click will show the sample "
            "data again. Nothing is written to disk either way."
        )
    return rendered(
        data, report_panel(report, label), f"Showing: {data.label} — {period(data)}"
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

    The stamp is recorded against the clicking visitor's session and nowhere
    else. It is not written to the audit database — no web request writes to
    disk — and it is not written onto the shared Insight objects, so one
    stranger's Verify is invisible to the next visitor, who still sees the
    finding as pending and can judge it themselves.
    """
    button = ctx.triggered_id
    # Redrawing the tickets creates new buttons, which also fires this callback (with no click).
    if not button or not ctx.triggered[0]["value"]:
        raise PreventUpdate

    pattern_type = button.get("index") if isinstance(button, dict) else None
    kind = button.get("type") if isinstance(button, dict) else None
    known = {insight.pattern_type for insight in SAMPLE.insights}
    if pattern_type not in known or kind not in ("verify-btn", "dismiss-btn"):
        # A crafted request naming something that is not a finding on this page.
        raise PreventUpdate

    remember_status(pattern_type, "verified" if kind == "verify-btn" else "dismissed")
    statuses = session_statuses()

    return ([ticket_body(insight, True, statuses.get(insight.pattern_type, "pending"))
             for insight in SAMPLE.insights],
            [ticket_class(insight, statuses.get(insight.pattern_type, "pending"))
             for insight in SAMPLE.insights])


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
