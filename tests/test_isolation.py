"""Two strangers on the live site must not be able to see each other.

The dashboard used to keep the dataset on show in a module-level ``CURRENT``,
and Verify/Dismiss stamps in one SQLite table. One process, no login: whoever
uploaded last decided what the next visitor saw, label, totals and findings
included, and one stranger's Verify showed up on another's screen. These tests
are the proof that neither can happen now.

A "visitor" here is a Flask request context carrying a session cookie, which is
exactly what the server sees from a browser: session identity is read from that
cookie and from nothing else, so two cookies are two visitors.
"""

import base64
import importlib
from contextlib import contextmanager

import pytest
from dash._callback_context import context_value
from dash._utils import AttributeDict

from src import verification
from src.session_state import SessionStore
from tests.test_dashboard import texts, walk
from tests.test_ingest import DUTCH_EXPORT, canonical, full_days


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    """Import the app with its audit database redirected to a temp file."""
    original = verification.DB_PATH
    verification.DB_PATH = str(tmp_path_factory.mktemp("db") / "insights.db")
    try:
        yield importlib.import_module("app.dashboard")
    finally:
        verification.DB_PATH = original


@pytest.fixture(autouse=True)
def clean_state(dashboard):
    """No test may inherit another's sessions, stamps or off-request dataset."""
    dashboard.STATE.clear()
    saved = dict(dashboard.OFF_REQUEST.statuses)
    yield
    dashboard.STATE.clear()
    dashboard.OFF_REQUEST.dataset = dashboard.SAMPLE
    dashboard.OFF_REQUEST.statuses.clear()
    dashboard.OFF_REQUEST.statuses.update(saved)
    for insight in dashboard.SAMPLE.insights:
        insight.status = "pending"


@contextmanager
def visitor(dashboard, session_id=None, proto=None):
    """A request the way the server sees one: with or without a session cookie."""
    headers = {}
    if session_id:
        headers["Cookie"] = f"{dashboard.SESSION_COOKIE}={session_id}"
    if proto:
        headers["X-Forwarded-Proto"] = proto
    with dashboard.server.test_request_context("/", headers=headers):
        yield


@contextmanager
def clicking(prop_id):
    """A faked Dash callback context, so the real callback body can be run."""
    token = context_value.set(
        AttributeDict(triggered_inputs=[{"prop_id": prop_id, "value": 1}])
    )
    try:
        yield
    finally:
        context_value.reset(token)


@contextmanager
def swapped_store(dashboard, **kwargs):
    """Run against a store with tighter caps, then put the real one back."""
    real = dashboard.STATE
    dashboard.STATE = SessionStore(**kwargs)
    try:
        yield dashboard.STATE
    finally:
        dashboard.STATE = real


def upload(dashboard, data: bytes, filename="kassa.csv"):
    payload = base64.b64encode(data).decode()
    return dashboard.apply_upload(f"data:text/csv;base64,{payload}", filename)


def shown(dashboard):
    """What the asking visitor's page says it is showing, and the till totals."""
    page = dashboard.page()
    note = next(n for n in walk(page) if getattr(n, "id", None) == "source-note")
    till = next(n for n in walk(page) if getattr(n, "id", None) == "till-container")
    return " ".join(str(t) for t in texts(note)), " ".join(texts(till))


A = "sessionAAAAAAAAAAAAAAAAAAAAAA"
B = "sessionBBBBBBBBBBBBBBBBBBBBBB"
C = "sessionCCCCCCCCCCCCCCCCCCCCCC"


# ---------------------------------------------------------------- the data leak ---


def test_two_visitors_uploading_different_files_see_different_data(dashboard):
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "aaa-kassa.csv")
    with visitor(dashboard, B):
        upload(dashboard, full_days(30), "bbb-till.csv")

    with visitor(dashboard, A):
        note, till = shown(dashboard)
        assert "aaa-kassa.csv" in note
        assert "bbb-till.csv" not in note
        assert dashboard.CURRENT.df["revenue"].sum() == pytest.approx(25.50)
        a_till = till
    with visitor(dashboard, B):
        note, till = shown(dashboard)
        assert "bbb-till.csv" in note
        assert "aaa-kassa.csv" not in note
        assert dashboard.CURRENT.df["revenue"].sum() > 1000
        assert till != a_till, "the till strip is the visitor's own totals"


def test_a_fresh_visitor_sees_the_sample_not_whatever_was_uploaded_last(dashboard):
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "private-sales.csv")

    with visitor(dashboard, C):
        note, till = shown(dashboard)
        assert "Sample data" in note
        assert "private-sales.csv" not in note
        assert dashboard.CURRENT is dashboard.SAMPLE
        assert "£117,212" in till, "the cold start still demos"


def test_a_visitor_with_no_cookie_at_all_sees_the_sample(dashboard):
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "private-sales.csv")

    with visitor(dashboard):  # no cookie: a first-time arrival
        note, _ = shown(dashboard)
        assert "Sample data" in note
        assert dashboard.CURRENT is dashboard.SAMPLE


def test_an_unknown_but_well_formed_session_id_is_treated_as_a_new_visitor(dashboard):
    """Sending someone else's-shaped id gets you the sample, not their file."""
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "private-sales.csv")

    for guess in ("sessionAAAAAAAAAAAAAAAAAAAAA", "a" * 32, "x" * 128):
        with visitor(dashboard, guess):
            assert dashboard.CURRENT is dashboard.SAMPLE


def test_one_visitor_going_back_to_the_sample_leaves_the_other_alone(dashboard):
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "aaa.csv")
    with visitor(dashboard, B):
        upload(dashboard, full_days(30), "bbb.csv")

    with visitor(dashboard, A):
        note = dashboard.show_sample()[0]
        assert "Sample data" in note
        assert dashboard.CURRENT is dashboard.SAMPLE

    with visitor(dashboard, B):
        assert dashboard.CURRENT is not dashboard.SAMPLE
        assert "bbb.csv" in shown(dashboard)[0]


def test_the_module_no_longer_holds_a_process_wide_dataset(dashboard):
    """CURRENT is a question about the asker, not a variable anyone can set."""
    assert "CURRENT" not in vars(dashboard), "a module global would be shared again"
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "aaa.csv")
        assert dashboard.CURRENT.label == "aaa.csv"
    # Off-request callers (tests, scripts) get their own slot, still the sample.
    assert dashboard.CURRENT is dashboard.SAMPLE

    with pytest.raises(AttributeError):
        dashboard.NOT_A_THING


def test_two_tabs_in_one_browser_are_one_visitor_and_share_their_upload(dashboard):
    """The documented choice: one browser is one person, so both tabs agree."""
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "aaa.csv")
    with visitor(dashboard, A):  # the same cookie, a second tab
        assert "aaa.csv" in shown(dashboard)[0]


# ---------------------------------------------------------------- the stamps ---


def test_one_visitors_verify_is_invisible_to_another(dashboard):
    pattern = dashboard.SAMPLE.insights[0].pattern_type
    click = f'{{"index":"{pattern}","type":"verify-btn"}}.n_clicks'

    with visitor(dashboard, A), clicking(click):
        bodies, classes = dashboard.handle_insight_action([1], [None])
        assert "paper ticket is-verified" in classes
        assert "Verified" in " ".join(texts(dashboard.html.Div(bodies[0])))

    with visitor(dashboard, B):
        printed = texts(dashboard.html.Div(dashboard.findings()))
        assert "Verified" not in printed
        assert "Verify" in printed and "Dismiss" in printed

    # Nothing was written onto the shared findings themselves.
    assert [i.status for i in dashboard.SAMPLE.insights] == ["pending"] * len(
        dashboard.SAMPLE.insights
    )


def test_a_dismiss_is_the_dismissing_visitors_alone(dashboard):
    pattern = dashboard.SAMPLE.insights[0].pattern_type
    click = f'{{"index":"{pattern}","type":"dismiss-btn"}}.n_clicks'

    with visitor(dashboard, A), clicking(click):
        _, classes = dashboard.handle_insight_action([None], [1])
        assert classes[0] == "paper ticket is-dismissed"

    with visitor(dashboard, A):
        assert dashboard.STATE.statuses(A) == {pattern: "dismissed"}
        assert "Dismissed" in texts(dashboard.html.Div(dashboard.findings()))
    with visitor(dashboard, C):
        assert dashboard.STATE.statuses(C) == {}
        assert "Dismissed" not in texts(dashboard.html.Div(dashboard.findings()))


def test_a_stamp_survives_going_back_to_the_sample_but_not_another_visitor(dashboard):
    pattern = dashboard.SAMPLE.insights[0].pattern_type
    with visitor(dashboard, A):
        dashboard.remember_status(pattern, "verified")
        upload(dashboard, DUTCH_EXPORT, "aaa.csv")
        dashboard.show_sample()
        assert dashboard.session_statuses() == {pattern: "verified"}
    with visitor(dashboard, B):
        assert dashboard.session_statuses() == {}


def test_a_crafted_click_on_something_that_is_not_a_finding_is_ignored(dashboard):
    from dash.exceptions import PreventUpdate

    for prop_id in (
        '{"index":"../../etc/passwd","type":"verify-btn"}.n_clicks',
        '{"index":"afternoon_dip","type":"made-up-btn"}.n_clicks',
    ):
        with visitor(dashboard, A), clicking(prop_id):
            with pytest.raises(PreventUpdate):
                dashboard.handle_insight_action([1], [None])
        assert dashboard.STATE.statuses(A) == {}


def test_an_uploaded_dataset_never_carries_stamps_from_the_session(dashboard):
    pattern = dashboard.SAMPLE.insights[0].pattern_type
    with visitor(dashboard, A):
        dashboard.remember_status(pattern, "verified")
        upload(dashboard, full_days(30), "aaa.csv")
        printed = texts(dashboard.html.Div(dashboard.findings()))
        assert "Verified" not in printed, "an upload is not this cafe's own data"
        assert "From the file you uploaded, not saved." in printed


# ---------------------------------------------------------------- losing a session ---


def test_an_evicted_session_degrades_to_the_sample_with_a_message(dashboard):
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "aaa.csv")
        assert "aaa.csv" in shown(dashboard)[0]

    # Enough other visitors arrive to push A out of a cache this size.
    for i in range(dashboard.STATE.max_sessions + 2):
        with visitor(dashboard, f"crowd{i:026d}"):
            upload(dashboard, DUTCH_EXPORT, f"crowd{i}.csv")

    with visitor(dashboard, A):
        note, till = shown(dashboard)
        assert "Sample data" in note
        assert "no longer held in memory" in note
        assert "crowd" not in note, "never another visitor's file"
        assert "£117,212" in till
        assert "Traceback" not in note
        assert dashboard.CURRENT is dashboard.SAMPLE


def test_an_expired_session_degrades_to_the_sample_with_a_message(dashboard):
    """Age, not pressure: the store is re-pointed at a hand-wound clock."""
    real = dashboard.STATE
    now = [1000.0]
    dashboard.STATE = SessionStore(ttl_seconds=60, clock=lambda: now[0])
    try:
        with visitor(dashboard, A):
            upload(dashboard, DUTCH_EXPORT, "aaa.csv")
            assert "aaa.csv" in shown(dashboard)[0]
        now[0] += 3600
        with visitor(dashboard, A):
            note, till = shown(dashboard)
            assert "Sample data" in note and "no longer held in memory" in note
            assert "£117,212" in till
    finally:
        dashboard.STATE = real


def test_a_question_asked_after_eviction_is_answered_from_the_sample_and_says_so(dashboard):
    with swapped_store(dashboard, max_sessions=1):
        with visitor(dashboard, A):
            upload(dashboard, full_days(30), "aaa.csv")
        with visitor(dashboard, B):
            upload(dashboard, DUTCH_EXPORT, "bbb.csv")  # pushes A out

        with visitor(dashboard, A), clicking("q-trend.n_clicks"):
            printed = " ".join(texts(dashboard.handle_qa(1, 1, 1)))
            assert "Answered from the sample data" in printed
            assert "no longer held in memory" in printed
            assert "bbb.csv" not in printed


def test_a_question_is_answered_from_the_askers_own_file(dashboard):
    with visitor(dashboard, A):
        upload(dashboard, full_days(30), "aaa.csv")
    with visitor(dashboard, A), clicking("q-dip.n_clicks"):
        mine = " ".join(texts(dashboard.handle_qa(1, None, None)))
    with visitor(dashboard, B), clicking("q-dip.n_clicks"):
        theirs = " ".join(texts(dashboard.handle_qa(1, None, None)))

    assert mine != theirs, "the same question, answered from two different files"
    assert "Answered from the sample data" not in mine


def test_eviction_under_pressure_does_not_raise_and_stays_within_its_caps(dashboard):
    """Forty visitors through a store that holds three, each served their own file."""
    with swapped_store(dashboard, max_sessions=3, max_bytes=2 * 1024 * 1024) as state:
        for i in range(40):
            with visitor(dashboard, f"load{i:026d}"):
                pieces = upload(dashboard, DUTCH_EXPORT, f"load{i}.csv")
                assert f"load{i}.csv" in pieces[0]
                assert "£" in " ".join(texts(pieces[1]))
                stats = state.stats()
                assert stats["sessions"] <= 3
                assert 0 <= stats["bytes"] <= 2 * 1024 * 1024


def test_a_dataset_too_big_to_keep_is_drawn_once_and_the_visitor_is_told(dashboard):
    real = dashboard.STATE
    dashboard.STATE = SessionStore(max_bytes=1)  # nothing can be retained
    try:
        with visitor(dashboard, A):
            pieces = upload(dashboard, canonical(rows=40), "huge.csv")
            assert "huge.csv" in pieces[0], "their page is still drawn from their file"
            assert "£" in " ".join(texts(pieces[1]))
            report = " ".join(texts(pieces[4]))
            assert "too large to keep in memory" in report
            assert "Traceback" not in report
            # And the next request degrades cleanly rather than half-showing it.
            note, till = shown(dashboard)
            assert "Sample data" in note and "£117,212" in till
    finally:
        dashboard.STATE = real


def test_a_broken_session_store_still_renders_a_page(dashboard):
    """If the store itself fails, the visitor gets the sample, not a 500."""
    class Broken:
        max_sessions = 1

        def __getattr__(self, name):
            def boom(*args, **kwargs):
                raise RuntimeError("store is on fire")
            return boom

    real = dashboard.STATE
    dashboard.STATE = Broken()
    try:
        with visitor(dashboard, A):
            note, till = shown(dashboard)
            assert "Sample data" in note
            assert "£117,212" in till
            assert dashboard.remember_dataset(dashboard.SAMPLE) is False
            assert dashboard.session_statuses() == {}
            dashboard.forget_dataset()  # must not raise
    finally:
        dashboard.STATE = real


# ---------------------------------------------------------------- session identity ---


def test_the_server_mints_an_unguessable_httponly_cookie_for_a_new_visitor(dashboard):
    client = dashboard.server.test_client()
    first = client.get("/_dash-layout")
    assert first.status_code == 200

    cookies = [h for key, h in first.headers if key == "Set-Cookie"]
    mine = [c for c in cookies if c.startswith(f"{dashboard.SESSION_COOKIE}=")]
    assert len(mine) == 1, cookies
    assert "HttpOnly" in mine[0], "page scripts must not be able to read it"
    assert "SameSite=Lax" in mine[0]

    value = mine[0].split(";")[0].split("=", 1)[1]
    assert dashboard.SESSION_ID_RE.match(value)
    assert len(value) >= 32, "256 bits of CSPRNG, not a counter"

    # A client that already has one is not re-issued, and two clients differ.
    again = client.get("/_dash-layout")
    assert not [h for key, h in again.headers if key == "Set-Cookie"]
    other = dashboard.server.test_client().get("/_dash-layout")
    other_value = [h for key, h in other.headers if key == "Set-Cookie"][0]
    assert value not in other_value


def test_the_cookie_is_marked_secure_only_behind_https(dashboard):
    with visitor(dashboard, proto="https"):
        pass  # the header is read when the response is built, below

    client = dashboard.server.test_client()
    plain = client.get("/_dash-layout")
    assert "Secure" not in [h for key, h in plain.headers if key == "Set-Cookie"][0]

    secure = dashboard.server.test_client().get(
        "/_dash-layout", headers={"X-Forwarded-Proto": "https"}
    )
    assert "Secure" in [h for key, h in secure.headers if key == "Set-Cookie"][0]


def test_the_session_id_does_not_outlive_its_request(dashboard):
    """The stash must be request-scoped, not application-context scoped.

    `flask.g` reads like the obvious place for this and is the wrong scope: an
    app context can be pushed around more than one request, and a stash on g
    then survives into the second one, so visitor B silently inherits visitor
    A's session — with every other test in this file still passing. The id
    lives in request.environ, which the server builds per request and discards
    with it. This test is the shape of that bug.
    """
    with dashboard.server.app_context():
        with dashboard.server.test_request_context("/"):
            first = dashboard._session_id()
            upload(dashboard, DUTCH_EXPORT, "alice.csv")
            assert dashboard.CURRENT.label == "alice.csv"

        with dashboard.server.test_request_context("/"):  # a second visitor, no cookie
            second = dashboard._session_id()
            assert second != first, "a new request must mint a new id"
            assert dashboard.CURRENT is dashboard.SAMPLE
            assert dashboard.session_statuses() == {}

        # And the id is not left lying around on the application context.
        from flask import g

        assert not [name for name in vars(g) if "cafemetrics" in name]


def test_one_request_resolves_to_one_id_however_often_it_is_asked(dashboard):
    with visitor(dashboard):
        seen = {dashboard._session_id() for _ in range(5)}
        assert len(seen) == 1
        assert dashboard.SESSION_ENVIRON_KEY in dashboard.request.environ


def test_a_malformed_cookie_is_replaced_rather_than_trusted(dashboard):
    for junk in ("", "short", "has spaces in it", "x" * 500, "../../etc/passwd"):
        with visitor(dashboard, junk):
            session_id = dashboard._session_id()
            assert session_id != junk
            assert dashboard.SESSION_ID_RE.match(session_id)
            assert dashboard.CURRENT is dashboard.SAMPLE


def test_two_live_clients_over_http_cannot_see_each_others_label(dashboard):
    """End to end over the real WSGI app, cookies handled by the client."""
    one, two = dashboard.server.test_client(), dashboard.server.test_client()
    first = one.get("/_dash-layout")
    two.get("/_dash-layout")

    # Give client one an upload, keyed by the cookie the server just issued it.
    header = next(h for key, h in first.headers
                  if key == "Set-Cookie" and h.startswith(dashboard.SESSION_COOKIE))
    sid = header.split(";")[0].split("=", 1)[1]
    with visitor(dashboard, sid):
        upload(dashboard, DUTCH_EXPORT, "one-private.csv")

    assert b"one-private.csv" in one.get("/_dash-layout").data
    assert b"one-private.csv" not in two.get("/_dash-layout").data
    assert b"Sample data" in two.get("/_dash-layout").data


def test_the_layout_response_is_never_cacheable(dashboard):
    """A CDN that cached one Set-Cookie would hand every visitor one session."""
    client = dashboard.server.test_client()
    first = client.get("/_dash-layout")
    assert first.headers.get("Cache-Control") == "no-store"
    assert first.headers.get("Vary") == "Cookie"

    # Still no-store on the next visit, when no cookie is being issued.
    again = client.get("/_dash-layout")
    assert not [h for key, h in again.headers if key == "Set-Cookie"]
    assert again.headers.get("Cache-Control") == "no-store"
    assert again.headers.get("Vary") == "Cookie"

    # Static assets are left alone: they are the same for everybody.
    css = client.get("/assets/style.css")
    assert css.status_code == 200
    assert css.headers.get("Cache-Control") != "no-store"


# ---------------------------------------------------------------- the request body ---


def test_an_oversized_request_body_is_refused_without_being_read(dashboard):
    """The cheapest way to kill the worker, before this: one big POST.

    The 5 MB cap in src/ingest.py is checked before the pandas parse, but the
    body has already been buffered, JSON-parsed and base64-decoded by then. A
    60 MB post was measured getting the worker SIGKILLed inside a real 512 MB
    cgroup. MAX_CONTENT_LENGTH refuses it on the Content-Length header, so none
    of it is read.
    """
    assert dashboard.server.config["MAX_CONTENT_LENGTH"] == 8 * 1024 * 1024
    client = dashboard.server.test_client()

    response = client.post(
        "/_dash-update-component",
        data=b"x" * (9 * 1024 * 1024),
        content_type="application/json",
    )
    assert response.status_code == 413
    body = response.get_json()
    assert "limit" in body["message"] and "refused" in body["message"]
    assert "Traceback" not in response.get_data(as_text=True)

    # The worker is unharmed and still serving the sample.
    after = client.get("/_dash-layout")
    assert after.status_code == 200 and b"117,212" in after.data


def test_the_limit_leaves_room_for_a_legitimate_upload_of_the_documented_size(dashboard):
    """A 5 MB CSV is about 6.7 MiB as a base64 JSON body: inside 8 MiB."""
    from src.ingest import MAX_UPLOAD_BYTES

    wire = base64.b64encode(b"x" * MAX_UPLOAD_BYTES)
    assert len(wire) + 2048 < dashboard.server.config["MAX_CONTENT_LENGTH"]
    assert MAX_UPLOAD_BYTES < dashboard.server.config["MAX_CONTENT_LENGTH"]


# ---------------------------------------------------------------- CURRENT is read-only ---


def test_current_cannot_be_turned_back_into_a_shared_global(dashboard):
    """One assignment would shadow __getattr__ for the life of the process."""
    with pytest.raises(AttributeError, match="read-only"):
        dashboard.CURRENT = dashboard.SAMPLE
    with pytest.raises(AttributeError, match="read-only"):
        del dashboard.CURRENT

    assert "CURRENT" not in vars(dashboard)
    with visitor(dashboard, A):
        upload(dashboard, DUTCH_EXPORT, "aaa.csv")
        assert dashboard.CURRENT.label == "aaa.csv"
    assert dashboard.CURRENT is dashboard.SAMPLE, "still per-asker, not a global"

    # Everything else on the module is still writable (tests monkeypatch it).
    dashboard.STATE = dashboard.STATE


# ---------------------------------------------------------------- the disk ---


def test_no_web_request_writes_anything_to_disk(dashboard, tmp_path):
    """Uploads, stamps and questions: all in memory, nothing on the filesystem."""
    import os

    watched = [".", "data", "app", "src"]
    before = {where: sorted(os.listdir(where)) for where in watched}
    db_before = len(verification.get_insights())
    db_file = os.path.getmtime(verification.DB_PATH)
    sample_before = open("data/sample_sales.csv", "rb").read()

    pattern = dashboard.SAMPLE.insights[0].pattern_type
    click = f'{{"index":"{pattern}","type":"verify-btn"}}.n_clicks'
    with visitor(dashboard, A):
        upload(dashboard, full_days(30), "aaa.csv")
        with clicking(click):
            dashboard.handle_insight_action([1], [None])
        with clicking("q-dip.n_clicks"):
            dashboard.handle_qa(1, None, None)
        dashboard.show_sample()

    assert {where: sorted(os.listdir(where)) for where in watched} == before
    assert open("data/sample_sales.csv", "rb").read() == sample_before
    assert len(verification.get_insights()) == db_before, "no stamp reached the audit db"
    assert os.path.getmtime(verification.DB_PATH) == db_file


def test_the_app_imports_and_serves_when_the_audit_database_cannot_be_opened(
    dashboard, tmp_path, monkeypatch
):
    """init_db() raising at import used to kill the gunicorn worker, exit 3.

    A second, separate copy of the module is imported under its own name (so
    the shared app.dashboard is left alone) with DB_PATH pointing inside a
    directory that does not exist — which is what sqlite3 does on a read-only
    filesystem: OperationalError("unable to open database file"). The import
    must survive it, say so on the page, and still serve the sample.
    """
    import importlib.util

    monkeypatch.setattr(verification, "DB_PATH", str(tmp_path / "nope" / "x.db"))
    spec = importlib.util.spec_from_file_location("dashboard_readonly_db", "app/dashboard.py")
    degraded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(degraded)  # the import that used to kill the worker

    assert degraded.DB_INIT_ERROR is not None
    assert "OperationalError" in degraded.DB_INIT_ERROR
    assert degraded.insight_db_cache == {}
    assert degraded.DATA_LOAD_ERROR is None

    printed = texts(degraded.page())
    assert any("audit database could not be opened" in str(t) for t in printed)
    assert "What do the sales say?" in printed
    ids = [getattr(n, "id", None) for n in walk(degraded.page())]
    assert "hourly-chart" in ids and "till-container" in ids


def test_a_page_that_cannot_be_built_degrades_instead_of_raising(dashboard, monkeypatch):
    monkeypatch.setattr(dashboard, "_page", lambda: 1 / 0)
    printed = texts(dashboard.page())
    assert any("could not be built" in str(t) for t in printed)
    assert any("ZeroDivisionError" in str(t) for t in printed)
