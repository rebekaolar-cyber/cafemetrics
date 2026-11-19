"""The dashboard builds from the sample data: page structure, the itemised score, the stamps."""

import importlib

import pytest

from src import verification


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    """Import the app with its audit database redirected to a temp file."""
    original = verification.DB_PATH
    verification.DB_PATH = str(tmp_path_factory.mktemp("db") / "insights.db")
    try:
        yield importlib.import_module("app.dashboard")
    finally:
        verification.DB_PATH = original


def walk(component):
    """Every component in a Dash layout tree, depth first."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from walk(child)


def texts(component):
    """Every piece of text printed inside a component."""
    found = []
    for node in walk(component):
        children = getattr(node, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if isinstance(child, str):
                found.append(child)
    return found


def test_the_sample_data_loads_without_an_error(dashboard):
    assert dashboard.DATA_LOAD_ERROR is None
    assert dashboard.trading_days == 92


def test_the_page_has_every_part_the_callbacks_need(dashboard):
    ids = [getattr(node, "id", None) for node in walk(dashboard.page())]
    for needed in ("hourly-chart", "insights-container", "q-dip", "q-trend", "q-anomaly", "qa-output"):
        assert needed in ids
    # One ticket and one pair of buttons per finding, matched by the callback's pattern ids.
    for insight in dashboard.insights:
        assert {"type": "insight-card", "index": insight.pattern_type} in ids


def test_the_chart_marks_the_dip_window_and_only_that(dashboard):
    bars = dashboard.hourly_figure().data[0]
    marked = [hour for hour, colour in zip(bars.x, bars.marker.color) if colour == dashboard.HOT]
    assert marked == dashboard.analysis["afternoon_dip"]["dip_hours"]


def test_each_ticket_itemises_its_score(dashboard):
    assert dashboard.insights, "the sample data should produce at least one finding"
    for insight in dashboard.insights:
        printed = " ".join(texts(dashboard.receipt(insight)))
        for part in ("Sample size", "Consistency", "Effect size"):
            assert part in printed
        # Significance is printed exactly when a real test stands behind the
        # finding. The anomaly scan runs no test, so its ticket has three rows
        # and the other three weights are rescaled to still total 100.
        tested = insight.evidence.get("p_value") is not None
        assert ("Significance" in printed) is tested
        assert f"{insight.confidence_score} of 100" in printed

    # The biconditional above is derived from the same dict it checks, so on its
    # own it would still pass if a bug made every insight untested. Pin which
    # findings must carry a real p-value, so that cannot hide here.
    by_type = {insight.pattern_type: insight for insight in dashboard.insights}
    for pattern in ("afternoon_dip", "trend"):
        if pattern in by_type:
            p_value = by_type[pattern].evidence["p_value"]
            assert isinstance(p_value, float) and 0 < p_value <= 1, (
                f"{pattern} rests on a hypothesis test and must report its p-value"
            )
    if "anomalies" in by_type:
        assert by_type["anomalies"].evidence["p_value"] is None


def test_a_judged_finding_shows_a_stamp_instead_of_buttons(dashboard):
    insight = dashboard.insights[0]
    original = insight.status
    try:
        insight.status = "pending"
        assert {"Verify", "Dismiss"} <= set(texts(dashboard.html.Div(dashboard.ticket_foot(insight))))
        insight.status = "verified"
        assert texts(dashboard.html.Div(dashboard.ticket_foot(insight))) == ["Verified"]
        insight.status = "dismissed"
        assert texts(dashboard.html.Div(dashboard.ticket_foot(insight))) == ["Dismissed"]
        assert dashboard.ticket_class(insight) == "paper ticket is-dismissed"
    finally:
        insight.status = original


def test_answers_are_printed_with_their_evidence(dashboard):
    answer = dashboard.trend_answer(dashboard.analysis["trend"])
    printed = texts(dashboard.answer_block(answer))
    assert answer.title in printed and answer.finding in printed and answer.evidence in printed


def test_the_header_has_a_button_to_the_code(dashboard):
    links = [c for c in walk(dashboard.page()) if getattr(c, "className", None) == "code-link"]
    assert len(links) == 1
    assert links[0].href == dashboard.REPO
    assert links[0].target == "_blank"
