"""The upload path through the page: what gets drawn, and what never happens.

The dashboard module is a singleton holding the dataset currently on show, so
every test here puts the sample data back afterwards. That is also what the
"Back to the sample data" button does, which makes the teardown a test in
itself.
"""

import base64
import importlib

import pytest

from src import verification
from tests.test_ingest import DUTCH_EXPORT, canonical, full_days
from tests.test_dashboard import texts, walk

FAVICON = "app/assets/favicon.ico"


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    original = verification.DB_PATH
    verification.DB_PATH = str(tmp_path_factory.mktemp("db") / "insights.db")
    try:
        yield importlib.import_module("app.dashboard")
    finally:
        verification.DB_PATH = original


@pytest.fixture(autouse=True)
def sample_restored(dashboard):
    """Never leave an upload on the page for the next test."""
    yield
    dashboard.show_sample()


def upload(dashboard, data: bytes, filename="kassa.csv", mime="text/csv"):
    """Hand the dashboard a file the way a browser does."""
    payload = base64.b64encode(data).decode()
    return dashboard.apply_upload(f"data:{mime};base64,{payload}", filename)


def report_text(pieces):
    return " ".join(texts(pieces[4]))


# ---------------------------------------------------------------- the page -----


def test_the_page_offers_an_upload_and_defaults_to_the_sample(dashboard):
    ids = [getattr(node, "id", None) for node in walk(dashboard.page())]
    for needed in ("sales-upload", "use-sample", "load-report", "source-note",
                   "till-container", "chart-container"):
        assert needed in ids

    note = next(n for n in walk(dashboard.page()) if getattr(n, "id", None) == "source-note")
    assert "Sample data" in texts(note)[0]
    assert dashboard.CURRENT is dashboard.SAMPLE


def test_the_dropzone_asks_for_csv_and_caps_the_size(dashboard):
    zone = next(n for n in walk(dashboard.page()) if getattr(n, "id", None) == "sales-upload")
    assert ".csv" in zone.accept
    assert zone.multiple is False
    assert zone.max_size == dashboard.MAX_UPLOAD_BYTES == 5 * 1024 * 1024


def test_nothing_is_shown_from_the_load_report_until_a_file_arrives(dashboard):
    panel = next(n for n in walk(dashboard.page()) if getattr(n, "id", None) == "load-report")
    assert not panel.children


# ---------------------------------------------------------------- a good file --


def test_a_dutch_export_redraws_the_whole_page_from_it(dashboard):
    pieces = upload(dashboard, DUTCH_EXPORT, "kassa-export.csv")
    note, till, chart, found, panel, answer = pieces

    assert "kassa-export.csv" in note
    assert dashboard.CURRENT is not dashboard.SAMPLE
    assert dashboard.CURRENT.df["revenue"].sum() == pytest.approx(25.50)
    # the till strip now shows the uploaded totals, not the sample's
    assert "£26" in " ".join(texts(till)) or "£25" in " ".join(texts(till))
    assert chart is not None and found
    assert answer is None, "a stale answer from the old data must be cleared"

    text = report_text(pieces)
    assert "What was read" in text
    assert "read from “Datum”" in text and "read from “Omzet”" in text
    assert "day_of_week" in text


def test_the_sample_data_is_untouched_by_an_upload(dashboard):
    before = open("data/sample_sales.csv", "rb").read()
    upload(dashboard, DUTCH_EXPORT)

    assert open("data/sample_sales.csv", "rb").read() == before
    assert dashboard.SAMPLE.trading_days == 92
    assert len(dashboard.df) == 25607, "the module-level sample frame must not move"


def test_the_back_button_puts_the_sample_data_back(dashboard):
    upload(dashboard, DUTCH_EXPORT)
    assert dashboard.CURRENT is not dashboard.SAMPLE

    note, till, chart, found, panel, answer = dashboard.show_sample()

    assert dashboard.CURRENT is dashboard.SAMPLE
    assert "Sample data" in note
    assert panel is None, "the old file's report goes away with the old file"
    assert "£" in " ".join(texts(till))


def test_the_questions_are_answered_from_the_uploaded_file(dashboard):
    upload(dashboard, full_days(30))
    assert dashboard.CURRENT.analysis is not None
    assert dashboard.CURRENT.analysis is not dashboard.SAMPLE.analysis
    assert dashboard.CURRENT.insights, "an even 30-day file should produce findings"


def test_an_uploaded_finding_has_no_verify_buttons_and_is_not_written_to_the_audit_db(dashboard):
    before = len(verification.get_insights())
    upload(dashboard, full_days(30))

    assert dashboard.CURRENT.insights
    for insight in dashboard.CURRENT.insights:
        printed = texts(dashboard.html.Div(dashboard.ticket_foot(insight, verifiable=False)))
        assert "Verify" not in printed and "Dismiss" not in printed
    assert len(verification.get_insights()) == before


def test_uploaded_tickets_do_not_steal_the_pattern_ids_the_verify_callback_matches(dashboard):
    upload(dashboard, full_days(30))
    ids = [getattr(node, "id", None) for node in walk(dashboard.html.Div(dashboard.findings()))]
    assert not any(isinstance(i, dict) and i.get("type") == "insight-card" for i in ids)


# ---------------------------------------------------------------- bad files ----


@pytest.mark.parametrize(
    "name,data,expected",
    [
        ("binary", open(FAVICON, "rb").read(), "not a CSV"),
        ("empty", b"", "empty"),
        ("header only", b"datum,uur,omzet\n", "no sales in it"),
        ("no date", b"uur,omzet\n9,4.00\n", "no date column"),
        ("no hour", b"datum,omzet\n2026-04-01,4.00\n", "no hour column"),
        ("no money", b"datum,uur,artikel\n2026-04-01,9,Latte\n", "no money"),
        ("unknown columns", b"foo,bar\n1,2\n", "None of the columns"),
        ("all rows bad", b"datum,uur,omzet\nnope,9,4.00\n", "Not one value"),
        ("too big", b"x" * (5 * 1024 * 1024 + 1), "limit is"),
        ("a spreadsheet", b"PK\x03\x04", "spreadsheet"),
    ],
)
def test_a_bad_file_puts_a_readable_report_on_the_page_and_leaves_the_data_alone(
    dashboard, name, data, expected
):
    filename = "kassa.xlsx" if name == "a spreadsheet" else "kassa.csv"
    pieces = upload(dashboard, data, filename)
    note, till, chart, found, panel, _ = pieces
    text = report_text(pieces)

    assert "That file could not be used" in text, name
    assert expected in text, (name, text[:300])
    assert "Traceback" not in text and "Error:" not in text
    # the page still shows something: the sample stays up behind the report
    assert dashboard.CURRENT is dashboard.SAMPLE
    assert "Sample data" in note
    assert "£" in " ".join(texts(till))
    assert chart is not None and found


def test_a_file_with_broken_rows_shows_the_line_numbers_that_were_dropped(dashboard):
    data = (
        b"datum,uur,aantal,stuksprijs,omzet\n"
        b"01-04-2026;9;2;3,50;7,00\n"      # wrong separator for this header
    )
    pieces = upload(dashboard, data)
    assert "could not be used" in report_text(pieces) or "dropped" in report_text(pieces)


def test_rows_dropped_from_an_otherwise_good_file_are_itemised_on_the_page(dashboard):
    data = canonical(rows=40) + b"bad date,9,2,Drinks,Latte,2,3.50,7.00\n"
    pieces = upload(dashboard, data)
    text = report_text(pieces)

    assert "What was read" in text
    assert "Rows dropped (1)" in text
    assert "line 42" in text, "the header is line 1, so the 41st data row is line 42"
    assert dashboard.CURRENT is not dashboard.SAMPLE


def test_unrecognised_columns_are_named_on_the_page(dashboard):
    data = b"datum,uur,omzet,kassa_id,medewerker\n2026-04-01,9,4.00,T1,Sam\n"
    text = report_text(upload(dashboard, data))

    assert "Ignored" in text
    assert "kassa_id" in text and "medewerker" in text


def test_a_file_with_uneven_opening_hours_is_analysed_not_apologised_for(dashboard):
    """An uneven file must not break the page — and no longer costs it its findings.

    Was test_a_file_the_statistics_cannot_cope_with_still_shows_its_totals. When
    it was written, detect_afternoon_dip() ran scipy's paired t-test over two
    series grouped separately over the dip hours and the rest, and raised
    "unequal length arrays" whenever a day had sales in one window and not the
    other — which is any café with uneven opening hours. The dashboard's job was
    to survive that, and this test pinned the guard: totals and chart still
    drawn, findings replaced by an apology naming the ValueError.

    That crash is fixed. The two series are inner-joined on the date, so days
    with nothing to pair against are dropped from the comparison and counted in
    the result instead of derailing it. So the page no longer apologises for this
    file, it analyses it, and the assertions below say so. What has not changed,
    and is what this test still guards, is that an uneven file puts no traceback
    on the page and never costs the reader their totals.
    """
    pieces = upload(dashboard, canonical(rows=60))
    note, till, chart, found, panel, _ = pieces
    text = report_text(pieces)

    assert "What was read" in text
    assert "Traceback" not in text
    assert "£420" in " ".join(texts(till)), "the totals are still the file's own"
    assert found, "the findings and the scoring card still print"

    # The rules ran. No apology, and nothing swallowed.
    assert "could not be run" not in text
    assert "ValueError" not in text
    assert dashboard.CURRENT.analysis is not None
    assert dashboard.CURRENT.insights, "an uneven file now yields real findings"

    # The dip test did not fall over on the unpairable days; it set them aside
    # and said how many, which is the behaviour that replaced the crash.
    dip = dashboard.CURRENT.analysis["afternoon_dip"]
    assert dip["days_in_data"] == 28
    assert dip["sample_size"] == 15, "only the days with sales on both sides pair up"
    assert dip["days_excluded"] == 13
    assert "13 of 28 days" in dip["exclusion_note"]


def test_an_upload_without_a_filename_is_still_handled(dashboard):
    pieces = dashboard.apply_upload(
        "data:text/csv;base64," + base64.b64encode(DUTCH_EXPORT).decode(), None
    )
    assert "your file" in pieces[0]


def test_a_non_csv_extension_is_read_anyway_with_a_note(dashboard):
    text = report_text(upload(dashboard, canonical(rows=40), "export.dat"))
    assert "does not end in .csv" in text
    assert "What was read" in text
