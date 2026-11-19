"""A stranger's export, in every broken shape it arrives in.

The one rule every test here enforces: load_sales_frame returns a report and
never raises. If any of these ever raise, a café owner sees a traceback instead
of an answer, which is the whole failure this module exists to prevent.

Files are built as bytes on purpose. A BOM, a semicolon separator and a comma
decimal are facts about bytes, and writing them through pandas would hide
exactly the thing under test.
"""

import pandas as pd
import pytest

from src.data_loader import (
    aggregate_by_day,
    aggregate_by_hour,
    get_data_span,
    load_sales_data,
)
from src.ingest import (
    CANONICAL_COLUMNS,
    COLUMN_ALIASES,
    MAX_UPLOAD_BYTES,
    _to_number,
    load_sales_frame,
    normalise_header,
    read_upload,
)
from src.insights import generate_all_insights
from src.metrics import summary_metrics

SAMPLE = "data/sample_sales.csv"
FAVICON = "app/assets/favicon.ico"

CANONICAL_HEADER = b"date,hour,day_of_week,category,product,quantity,unit_price,revenue\n"


def canonical(rows=30, start_day=1):
    """A small clean file in the exact columns load_sales_data demands."""
    lines = [CANONICAL_HEADER]
    for i in range(rows):
        day = start_day + i % 28
        hour = 8 + i % 11
        dow = (day - 1) % 7
        lines.append(
            f"2026-04-{day:02d},{hour},{dow},Drinks,Latte,2,3.50,7.00\n".encode()
        )
    return b"".join(lines)


def full_days(days=30, hours=range(8, 20)):
    """A file with every open hour on every day.

    Only a file this even survives src/analysis.py: the afternoon-dip rule runs a
    paired t-test between the dip hours and the rest, and raises when some day has
    sales in one of those windows and not the other. See
    tests/test_upload_ui.py::test_a_file_the_statistics_cannot_cope_with_still_shows_its_totals.
    """
    lines = [CANONICAL_HEADER]
    for day in range(1, days + 1):
        for hour in hours:
            dow = (day - 1) % 7
            # quiet between two and four, so the dip rule has something to find
            quantity = 1 if 14 <= hour <= 16 else 3
            lines.append(
                f"2026-04-{day:02d},{hour},{dow},Drinks,Latte,"
                f"{quantity},3.50,{quantity * 3.5:.2f}\n".encode()
            )
    return b"".join(lines)


def load(data, **kwargs):
    """Load and assert the no-exception contract in one place."""
    frame, report = load_sales_frame(data, **kwargs)
    assert isinstance(report.lines(), list)
    assert all(isinstance(line, str) for line in report.lines())
    return frame, report


# ---------------------------------------------------------------- the table ----


def test_the_alias_table_covers_every_canonical_column_and_itself():
    assert set(COLUMN_ALIASES) == set(CANONICAL_COLUMNS)
    for canonical_name, aliases in COLUMN_ALIASES.items():
        assert canonical_name in aliases, f"{canonical_name} must accept its own name"
        for alias in aliases:
            assert normalise_header(alias) == alias, f"{alias} is not in normalised form"


def test_the_dutch_headers_a_till_export_produces_are_all_mapped():
    expected = {
        "datum": "date", "uur": "hour", "tijd": "hour", "aantal": "quantity",
        "prijs": "unit_price", "stuksprijs": "unit_price", "omzet": "revenue",
        "totaal": "revenue", "product": "product", "artikel": "product",
        "categorie": "category", "dag": "day_of_week",
    }
    for dutch, canonical_name in expected.items():
        assert dutch in COLUMN_ALIASES[canonical_name], f"{dutch} should map to {canonical_name}"


@pytest.mark.parametrize(
    "header,expected",
    [
        ("Unit Price", "unit_price"), ("  DATUM  ", "datum"), ("unit-price", "unit_price"),
        ("Stuksprijs (€)", "stuksprijs"), ("product.1", "product"), ("day of week", "day_of_week"),
    ],
)
def test_headers_are_normalised_before_they_are_looked_up(header, expected):
    assert normalise_header(header) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3.50", 3.50), ("3,50", 3.50), ("€ 3,50", 3.50), ("£1,234.50", 1234.50),
        ("1.234,50", 1234.50), ("1,234", 1234.0), ("(2.00)", -2.0), ("", float("nan")),
        ("n/a", float("nan")), ("not a number", float("nan")), ("-1,5", -1.5),
    ],
)
def test_money_is_read_in_both_european_and_british_notation(text, expected):
    got = _to_number(text)
    if pd.isna(expected):
        assert pd.isna(got)
    else:
        assert got == pytest.approx(expected)


# ---------------------------------------------------------------- regression ---


def test_the_sample_file_comes_back_byte_for_byte_what_the_strict_loader_gives():
    """The whole point of an additive API: today's happy path must not move."""
    strict = load_sales_data(SAMPLE)
    frame, report = load(SAMPLE)

    assert report.ok
    assert frame.equals(strict)
    assert list(frame.dtypes) == list(strict.dtypes)
    assert summary_metrics(frame) == summary_metrics(strict)
    assert aggregate_by_hour(frame).equals(aggregate_by_hour(strict))
    assert aggregate_by_day(frame).equals(aggregate_by_day(strict))
    assert get_data_span(frame) == get_data_span(strict)
    assert [i.title for i in generate_all_insights(frame)] == [
        i.title for i in generate_all_insights(strict)
    ]


def test_a_clean_canonical_file_is_read_with_nothing_derived_and_nothing_dropped():
    frame, report = load(canonical())

    assert report.ok and report.rows_kept == report.rows_read == 30
    assert report.drops == [] and report.ignored == []
    assert set(report.mapped_columns) == set(CANONICAL_COLUMNS)
    assert report.derived_columns == []
    assert list(frame.columns) == list(CANONICAL_COLUMNS)


def test_an_uploaded_frame_is_always_shaped_like_the_strict_loaders_output():
    strict = load_sales_data(SAMPLE)
    for data in (canonical(), DUTCH_EXPORT, b"datum,uur,omzet\n2026-04-01,9,4.00\n"):
        frame, report = load(data)
        assert report.ok
        assert list(frame.columns) == list(strict.columns)
        assert list(frame.dtypes) == list(strict.dtypes), report.summary
        assert frame["date"].is_monotonic_increasing
        assert list(frame.index) == list(range(len(frame)))


# ---------------------------------------------------------------- dutch --------

#: What a Dutch café's Excel actually writes: BOM, semicolons, comma decimals,
#: day-first dates and not one English header.
DUTCH_EXPORT = (
    "﻿Datum;Uur;Categorie;Artikel;Aantal;Stuksprijs;Omzet\r\n"
    "01-04-2026;9;Dranken;Cappuccino;2;3,50;7,00\r\n"
    "01-04-2026;15;Eten;Tosti;1;5,25;5,25\r\n"
    "02-04-2026;9;Dranken;Cappuccino;3;3,50;10,50\r\n"
    "02-04-2026;16;Dranken;Thee;1;2,75;2,75\r\n"
).encode("utf-8")


def test_a_dutch_excel_export_is_read_in_full():
    frame, report = load(DUTCH_EXPORT, filename="kassa-export.csv")

    assert report.ok, report.lines()
    assert report.delimiter == ";"
    assert report.had_bom and report.decimal_comma
    assert report.rows_kept == 4 and report.drops == []
    # headers, including the one that carries the BOM
    assert set(report.mapped_columns) == {
        "date", "hour", "category", "product", "quantity", "unit_price", "revenue"
    }
    assert report.derived_columns == ["day_of_week"]
    # 01-04-2026 is day-first: 1 April, not 4 January
    assert frame["date"].min() == pd.Timestamp("2026-04-01")
    assert frame["unit_price"].tolist() == [3.50, 5.25, 3.50, 2.75]
    assert frame["revenue"].sum() == pytest.approx(25.50)
    assert frame["product"].tolist() == ["Cappuccino", "Tosti", "Cappuccino", "Thee"]
    assert "semicolons" in report.how_it_was_read()
    assert "decimal commas" in report.how_it_was_read()


def test_a_semicolon_file_without_a_bom_is_still_read():
    frame, report = load(DUTCH_EXPORT.replace(b"\xef\xbb\xbf", b""))
    assert report.ok and not report.had_bom and report.rows_kept == 4


def test_a_bom_on_a_canonical_file_does_not_hide_the_date_column():
    frame, report = load(b"\xef\xbb\xbf" + canonical())
    assert report.ok and report.had_bom
    assert "date" in report.mapped_columns


def test_a_tab_separated_export_is_read_too():
    data = (
        "datum\tuur\tartikel\tomzet\n"
        "2026-04-01\t9\tLatte\t3.50\n"
        "2026-04-02\t9\tLatte\t3.50\n"
    ).encode()
    frame, report = load(data)
    assert report.ok and report.delimiter == "\t" and report.rows_kept == 2


# ---------------------------------------------------------------- derivation ---


def test_day_of_week_is_derived_from_the_date_when_the_column_is_missing():
    data = b"date,hour,product,quantity,unit_price\n2026-04-01,9,Latte,2,3.50\n"
    frame, report = load(data)

    assert report.ok
    assert "day_of_week" in report.derived_columns
    assert frame["day_of_week"].iloc[0] == 2  # 1 April 2026 is a Wednesday
    assert "worked out from the date" in str(
        next(c for c in report.columns if c.column == "day_of_week")
    )


def test_revenue_is_derived_from_quantity_times_unit_price():
    data = b"datum,uur,aantal,prijs\n2026-04-01,9,3,2.50\n"
    frame, report = load(data)

    assert report.ok and "revenue" in report.derived_columns
    assert frame["revenue"].iloc[0] == pytest.approx(7.50)


def test_unit_price_is_derived_from_revenue_over_quantity():
    data = b"datum,uur,aantal,omzet\n2026-04-01,9,4,10.00\n"
    frame, report = load(data)

    assert report.ok and "unit_price" in report.derived_columns
    assert frame["unit_price"].iloc[0] == pytest.approx(2.50)


def test_quantity_is_filled_with_one_when_the_file_only_has_line_totals():
    data = b"datum,uur,totaal\n2026-04-01,9,4.00\n"
    frame, report = load(data)

    assert report.ok
    assert frame["quantity"].iloc[0] == 1
    assert frame["unit_price"].iloc[0] == pytest.approx(4.00)


def test_the_hour_is_derived_from_a_timestamp_column():
    data = (
        b"timestamp,product,aantal,prijs\n"
        b"2026-04-01 09:15:00,Latte,2,3.00\n"
        b"2026-04-01 16:45:00,Latte,1,3.00\n"
    )
    frame, report = load(data)

    assert report.ok and "hour" in report.derived_columns
    assert frame["hour"].tolist() == [9, 16]
    assert frame["date"].nunique() == 1, "the time of day must not leak into the date"


@pytest.mark.parametrize("value,expected", [("14", 14), ("14:30", 14), ("2:30 PM", 14), ("9.15", 9)])
def test_an_hour_column_holding_a_clock_time_is_understood(value, expected):
    data = f"datum,tijd,omzet\n2026-04-01,{value},4.00\n".encode()
    frame, report = load(data)
    assert report.ok, report.lines()
    assert frame["hour"].iloc[0] == expected


def test_category_and_product_are_filled_so_the_product_metric_still_runs():
    data = b"datum,uur,omzet\n2026-04-01,9,4.00\n2026-04-02,9,5.00\n"
    frame, report = load(data)

    assert report.ok
    assert frame["category"].unique().tolist() == ["Unknown"]
    assert summary_metrics(frame)["best_product"] == "Unknown"


def test_a_day_of_week_column_that_disagrees_with_its_date_is_kept_and_flagged():
    data = b"date,hour,dag,revenue\n2026-04-01,9,5,4.00\n"
    frame, report = load(data)

    assert report.ok
    assert frame["day_of_week"].iloc[0] == 5
    assert any("does not match" in w for w in report.warnings)


def test_a_day_of_week_column_full_of_names_is_replaced_by_the_derived_one():
    data = b"date,hour,dag,revenue\n2026-04-01,9,woensdag,4.00\n"
    frame, report = load(data)

    assert report.ok
    assert frame["day_of_week"].iloc[0] == 2
    assert "day_of_week" in report.derived_columns
    assert any("dag" in name for name in report.ignored)


# ---------------------------------------------------------------- refusals -----


def test_a_file_with_no_date_column_is_refused_with_the_columns_it_did_have():
    frame, report = load(b"hour,product,revenue\n9,Latte,3.00\n")

    assert frame is None and not report.ok
    assert "no date column" in report.errors[0]
    assert "hour" in report.errors[0] and "product" in report.errors[0]


def test_a_file_with_no_hour_and_no_timestamp_is_refused_and_says_why():
    frame, report = load(b"date,product,revenue\n2026-04-01,Latte,3.00\n")

    assert frame is None and not report.ok
    assert "no hour column" in report.errors[0]
    assert "uur" in report.errors[0], "the message should name the Dutch header too"


def test_a_file_with_no_money_at_all_is_refused():
    frame, report = load(b"date,hour,product\n2026-04-01,9,Latte\n")

    assert frame is None and not report.ok
    assert "no money" in report.errors[0]
    assert "omzet" in report.errors[0] or "revenue" in report.errors[0]


def test_quantity_alone_is_not_enough_money_to_work_with():
    frame, report = load(b"date,hour,aantal\n2026-04-01,9,2\n")
    assert frame is None and "unit_price" in report.errors[0]


def test_a_file_whose_columns_mean_nothing_to_us_is_refused_not_guessed_at():
    frame, report = load(b"foo,bar,baz\n1,2,3\n")

    assert frame is None and not report.ok
    assert "None of the columns" in report.errors[0]
    assert all(col in report.errors[0] for col in ("foo", "bar", "baz"))


# ---------------------------------------------------------------- bad rows -----


def test_rows_whose_date_cannot_be_read_are_dropped_with_their_line_numbers():
    data = (
        CANONICAL_HEADER
        + b"2026-04-01,9,2,Drinks,Latte,2,3.50,7.00\n"
        + b"not a date,9,2,Drinks,Latte,2,3.50,7.00\n"
        + b"2026-04-03,9,4,Drinks,Latte,2,3.50,7.00\n"
    )
    frame, report = load(data)

    assert report.ok and report.rows_kept == 2
    drop = next(d for d in report.drops if "date" in d.reason)
    assert drop.rows == [3], "the header is line 1, so the second data row is line 3"
    assert "line 3" in str(drop)


def test_mixed_date_formats_in_one_column_are_all_parsed():
    data = (
        b"datum,uur,omzet\n"
        b"2026-04-01,9,4.00\n"
        b"02-04-2026,9,4.00\n"
        b"2026/04/03,9,4.00\n"
        b"4 April 2026,9,4.00\n"
    )
    frame, report = load(data)

    assert report.ok and report.rows_kept == 4, report.lines()
    assert frame["date"].dt.day.tolist() == [1, 2, 3, 4]


def test_a_date_column_nothing_can_be_read_from_is_a_refusal_not_a_silent_empty():
    frame, report = load(b"date,hour,revenue\nnope,9,4.00\nalso nope,9,4.00\n")

    assert frame is None and not report.ok
    assert "Not one value in the date column" in report.errors[0]
    assert "nope" in report.errors[0], "show the owner what we choked on"


def test_zero_and_negative_quantities_are_dropped_as_voids_and_refunds():
    data = (
        CANONICAL_HEADER
        + b"2026-04-01,9,2,Drinks,Latte,2,3.50,7.00\n"
        + b"2026-04-02,9,3,Drinks,Latte,0,3.50,0.00\n"
        + b"2026-04-03,9,4,Drinks,Latte,-1,3.50,-3.50\n"
    )
    frame, report = load(data)

    assert report.ok and report.rows_kept == 1
    reasons = " ".join(d.reason for d in report.drops)
    assert "zero or negative" in reasons
    assert sorted(r for d in report.drops for r in d.rows) == [3, 4]


def test_non_numeric_values_in_numeric_columns_drop_only_their_own_rows():
    data = (
        b"datum,uur,aantal,prijs,omzet\n"
        b"2026-04-01,9,2,3.50,7.00\n"
        b"2026-04-02,9,twee,3.50,7.00\n"
        b"2026-04-03,9,2,n/a,7.00\n"
        b"2026-04-04,9,2,3.50,onbekend\n"
    )
    frame, report = load(data)

    assert report.ok and report.rows_kept == 2, report.lines()
    assert set(frame["date"].dt.day) == {1, 3}, "a bad unit price is survivable, a bad revenue is not"
    assert report.rows_dropped == 2


def test_an_hour_outside_the_day_is_dropped():
    data = b"datum,uur,omzet\n2026-04-01,9,4.00\n2026-04-02,25,4.00\n2026-04-03,-1,4.00\n"
    frame, report = load(data)

    assert report.ok and report.rows_kept == 1
    assert any("between 0 and 23" in d.reason for d in report.drops)


def test_a_revenue_that_is_not_quantity_times_price_is_flagged_and_kept():
    data = (
        CANONICAL_HEADER
        + b"2026-04-01,9,2,Drinks,Latte,1,2.50,5.00\n"
        + b"2026-04-02,9,3,Drinks,Latte,1,2.50,2.50\n"
    )
    frame, report = load(data)

    assert report.ok and report.rows_kept == 2
    assert frame["revenue"].sum() == pytest.approx(7.50), "the till's own total is kept"
    assert any("not quantity × unit_price" in w for w in report.warnings)
    assert report.drops == [], "a bad price is worth saying, not worth throwing a sale away"


def test_a_file_where_every_row_is_unusable_reports_that_instead_of_returning_nothing():
    data = CANONICAL_HEADER + b"2026-04-01,9,2,Drinks,Latte,0,3.50,0.00\n" * 3
    frame, report = load(data)

    assert frame is None and not report.ok
    assert "had to be dropped" in report.summary
    assert report.rows_dropped == 3 and report.drops


# ---------------------------------------------------------------- junk files ---


def test_an_empty_file_says_it_is_empty():
    frame, report = load(b"")
    assert frame is None and not report.ok
    assert "empty" in report.errors[0]


def test_a_file_of_only_whitespace_says_it_is_empty():
    frame, report = load(b"\n\n   \n")
    assert frame is None and "empty" in report.errors[0]


def test_a_header_only_file_names_the_headers_it_understood():
    frame, report = load(b"datum,uur,omzet\n")

    assert frame is None and not report.ok
    assert report.rows_read == 0
    assert "no sales in it" in report.summary
    assert set(report.mapped_columns) == {"date", "hour", "revenue"}


def test_a_one_row_file_is_analysed_and_warned_about():
    frame, report = load(b"datum,uur,omzet\n2026-04-01,9,4.00\n")

    assert report.ok and len(frame) == 1
    assert any("trading day" in w for w in report.warnings)
    assert summary_metrics(frame)["total_revenue"] == pytest.approx(4.00)


def test_a_binary_file_is_told_it_is_not_a_csv():
    frame, report = load(open(FAVICON, "rb").read(), filename="favicon.ico")

    assert frame is None and not report.ok
    assert "not a CSV" in report.errors[0]
    assert "binary" in report.errors[0]
    assert not any("Traceback" in line for line in report.lines())


def test_an_image_through_the_upload_path_is_refused_the_same_way():
    import base64

    payload = base64.b64encode(open(FAVICON, "rb").read()).decode()
    frame, report = read_upload(f"data:image/x-icon;base64,{payload}", "favicon.ico")

    assert frame is None and not report.ok
    assert "not a CSV" in " ".join(report.errors)


def test_a_file_too_big_for_a_free_instance_is_refused_before_it_is_parsed():
    frame, report = load(b"x" * (MAX_UPLOAD_BYTES + 1))

    assert frame is None and not report.ok
    assert "limit is" in report.errors[0]
    assert "5 MB" in report.errors[0]


def test_a_ragged_or_unquoted_file_is_reported_not_raised():
    frame, report = load(b'date,hour,revenue\n"unclosed,9,3.00\n2026-04-01,9,3.00\n')

    assert frame is None and not report.ok
    assert "could not be parsed" in report.errors[0]


def test_a_missing_path_is_reported_not_raised():
    frame, report = load("tests/no-such-file.csv")
    assert frame is None and "No such file" in report.errors[0]


def test_a_spreadsheet_is_refused_with_the_excel_instructions_not_an_engine_error():
    frame, report = read_upload("data:application/vnd.ms-excel;base64,UEsDBA==", "kassa.xlsx")

    assert frame is None and not report.ok
    assert "spreadsheet" in report.errors[0] and "CSV" in report.errors[0]


def test_an_upload_with_no_contents_is_reported_not_raised():
    frame, report = read_upload(None, "kassa.csv")
    assert frame is None and "No file was uploaded" in report.errors[0]


def test_an_upload_whose_base64_is_corrupt_is_reported_not_raised():
    frame, report = read_upload("data:text/csv;base64,!!!!not base64!!!!", "kassa.csv")
    assert frame is None and not report.ok


# ---------------------------------------------------------------- extras -------


def test_columns_we_do_not_know_are_listed_as_ignored_never_guessed_at():
    data = (
        b"date,hour,revenue,till_id,cashier,vat_rate,loyalty_card\n"
        b"2026-04-01,9,4.00,T1,Sam,0.09,none\n"
    )
    frame, report = load(data)

    assert report.ok
    assert list(frame.columns) == list(CANONICAL_COLUMNS)
    ignored = " ".join(report.ignored)
    for unknown in ("till_id", "cashier", "vat_rate", "loyalty_card"):
        assert unknown in ignored


def test_a_repeated_header_name_is_read_once_and_the_second_one_is_reported():
    data = (
        b"datum,uur,omzet,artikel,artikel\n"
        b"2026-04-01,9,4.00,Latte,Tosti\n"
    )
    frame, report = load(data)

    assert report.ok
    assert frame["product"].iloc[0] == "Latte", "the first of the pair wins"
    assert any("second" in name for name in report.ignored)


def test_two_different_headers_that_mean_the_same_thing_do_not_both_win():
    data = b"datum,uur,omzet,totaal\n2026-04-01,9,4.00,99.00\n"
    frame, report = load(data)

    assert report.ok
    assert frame["revenue"].iloc[0] == pytest.approx(4.00)
    assert any("revenue" in name for name in report.ignored)


def test_identical_rows_are_kept_because_two_identical_sales_are_normal():
    row = b"2026-04-01,9,2,Drinks,Latte,2,3.50,7.00\n"
    frame, report = load(CANONICAL_HEADER + row * 3)

    assert report.ok and report.rows_kept == 3
    assert report.duplicate_rows == 2
    assert report.drops == []


def test_a_file_that_is_its_own_second_half_is_called_out_as_a_double_export():
    body = canonical(20)
    doubled = body + body.split(CANONICAL_HEADER, 1)[1]
    frame, report = load(doubled)

    assert report.ok and report.rows_kept == 40
    assert any("pasted twice" in w for w in report.warnings)


def test_the_sample_file_is_not_flagged_as_a_double_export():
    _, report = load(SAMPLE)
    assert not any("pasted twice" in w for w in report.warnings)
    assert report.duplicate_rows > 0, "the sample really does repeat rows; that is fine"


def test_a_short_file_is_warned_about_before_anyone_trusts_its_confidence_scores():
    _, report = load(canonical(rows=10))
    assert any("confidence scores will be low" in w for w in report.warnings)


def test_a_long_enough_file_is_not_warned_about_its_length():
    _, report = load(canonical(rows=28 * 3))
    assert not any("confidence scores" in w for w in report.warnings)


def test_loading_from_a_path_and_from_the_same_bytes_gives_the_same_frame(tmp_path):
    path = tmp_path / "kassa.csv"
    path.write_bytes(DUTCH_EXPORT)

    from_path, path_report = load(str(path))
    from_bytes, bytes_report = load(DUTCH_EXPORT, filename="kassa.csv")

    assert from_path.equals(from_bytes)
    assert path_report.summary == bytes_report.summary


def test_an_upload_never_writes_anything_to_disk(tmp_path, monkeypatch):
    """The sample file must be untouchable from the upload path."""
    import base64

    before = open(SAMPLE, "rb").read()
    payload = base64.b64encode(DUTCH_EXPORT).decode()

    def refuse(*args, **kwargs):
        raise AssertionError("the ingest path must not open a file for writing")

    monkeypatch.setattr("pandas.DataFrame.to_csv", refuse)
    frame, report = read_upload(f"data:text/csv;base64,{payload}", "kassa.csv")

    assert report.ok
    assert open(SAMPLE, "rb").read() == before
