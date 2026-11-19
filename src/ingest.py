"""Read a stranger's sales export and say exactly what happened to it.

`src/data_loader.load_sales_data` is strict on purpose: it is the contract for
the sample file, it demands eight exact column names, and it raises on the first
thing it dislikes. That is right for a file we generate and wrong for a file
somebody uploads. A café owner's till export arrives with Dutch headers,
semicolons, comma decimals, a BOM, a blank trailing row and three columns we
have never heard of, and the useful answer is never a traceback.

This module is the forgiving path. `load_sales_frame` returns

    (DataFrame | None, LoadReport)

and never raises for bad input. The DataFrame, when there is one, has exactly
the columns and dtypes `load_sales_data` returns, so everything downstream
(`summary_metrics`, `analyze_all`, `generate_all_insights`, the aggregations)
works on it unchanged. The LoadReport says what each column was mapped from,
what was derived, what was ignored, which rows were dropped and why.

What is genuinely mandatory
---------------------------
- a date (from `date`, `datum`, `timestamp`, ... ) — nothing can stand in for it
- an hour (from `hour`/`uur`/`tijd`, or read off a date column that carries a
  time) — the whole dashboard is hour-of-day
- money: either `revenue`, or `quantity` and `unit_price` to multiply

Everything else is derived or filled:
- `day_of_week`  <- date.dayofweek
- `revenue`      <- quantity * unit_price
- `unit_price`   <- revenue / quantity
- `quantity`     <- 1 when only a line total is exported
- `category`,
  `product`      <- "Unknown" (kept so the top-product metric still runs)
"""

from __future__ import annotations

import base64
import binascii
import csv
import io
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# A free Render instance has 512 MB of RAM and pandas needs several times the
# file size while parsing, so the cap is about the host, not about CSV.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

CANONICAL_COLUMNS = (
    "date", "hour", "day_of_week", "category", "product",
    "quantity", "unit_price", "revenue",
)

#: Header variants we accept, canonical name -> the spellings seen in the wild.
#: Matching is on the *normalised* header (lowercased, stripped, punctuation and
#: whitespace collapsed to a single underscore), so "Unit Price", "unit-price"
#: and " UNIT_PRICE " all arrive here as "unit_price". Explicit table, no fuzzy
#: guessing: a header we do not list is reported as ignored rather than guessed.
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "date": (
        "date", "datum", "day", "dag_datum", "sale_date", "salesdate",
        "transaction_date", "order_date", "datetime", "date_time",
        "timestamp", "time_stamp", "tijdstip", "verkoopdatum", "boekdatum",
    ),
    "hour": (
        "hour", "uur", "tijd", "time", "hour_of_day", "uur_van_de_dag",
        "hr", "tijdstip_uur",
    ),
    "day_of_week": (
        "day_of_week", "dayofweek", "dag", "weekdag", "dag_van_de_week",
        "weekday", "dow",
    ),
    "category": (
        "category", "categorie", "cat", "product_category", "productgroep",
        "groep", "artikelgroep", "department", "afdeling",
    ),
    "product": (
        "product", "artikel", "item", "product_name", "productnaam",
        "artikelnaam", "description", "omschrijving", "naam", "sku",
    ),
    "quantity": (
        "quantity", "aantal", "qty", "amount_sold", "count", "units",
        "stuks", "hoeveelheid", "number", "num", "aantal_stuks",
    ),
    "unit_price": (
        "unit_price", "unitprice", "prijs", "stuksprijs", "price",
        "price_per_unit", "eenheidsprijs", "prijs_per_stuk", "verkoopprijs",
        "item_price",
    ),
    "revenue": (
        "revenue", "omzet", "totaal", "total", "line_total", "total_price",
        "sales", "turnover", "totaalbedrag", "bedrag", "amount", "subtotaal",
        "regeltotaal",
    ),
}

#: Reverse lookup, built once. Later canonical names never shadow earlier ones.
_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for _canonical, _aliases in COLUMN_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL.setdefault(_alias, _canonical)

_DELIMITERS = (",", ";", "\t", "|")
_CURRENCY = re.compile(r"[£$€ \s]")
_BLANKS = {"", "-", "--", "n/a", "na", "nan", "none", "null", "nvt", "n.v.t."}
_TIME_IN_STRING = re.compile(r"\b([01]?\d|2[0-3])[:.]([0-5]\d)")


# ---------------------------------------------------------------- report ------


@dataclass
class ColumnNote:
    """One canonical column and where it came from."""

    column: str
    source: Optional[str]          # the header in the uploaded file, if any
    how: str                       # "mapped" | "derived" | "filled"
    detail: str = ""

    def __str__(self) -> str:
        if self.how == "mapped":
            same = self.source == self.column
            return f"{self.column} — read from “{self.source}”" if not same else f"{self.column} — read as-is"
        return f"{self.column} — {self.detail}"


@dataclass
class DropNote:
    """A group of rows dropped for one reason, with their line numbers."""

    reason: str
    rows: List[int] = field(default_factory=list)   # 1-based line number in the file

    @property
    def count(self) -> int:
        return len(self.rows)

    def __str__(self) -> str:
        shown = ", ".join(str(r) for r in self.rows[:10])
        more = f" and {self.count - 10} more" if self.count > 10 else ""
        return f"{self.count} row{'s' if self.count != 1 else ''} dropped — {self.reason} (line {shown}{more})"


@dataclass
class LoadReport:
    """What happened to the file. Safe to render straight into the page."""

    ok: bool = False
    source: str = ""
    summary: str = ""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    columns: List[ColumnNote] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)
    drops: List[DropNote] = field(default_factory=list)
    rows_read: int = 0
    rows_kept: int = 0
    duplicate_rows: int = 0
    delimiter: Optional[str] = None
    encoding: Optional[str] = None
    decimal_comma: bool = False
    had_bom: bool = False

    # -- small helpers the loader uses while it works -----------------------
    def fail(self, message: str) -> "LoadReport":
        self.ok = False
        self.errors.append(message)
        if not self.summary:
            self.summary = message
        return self

    def note(self, column: str, source: Optional[str], how: str, detail: str = "") -> None:
        self.columns.append(ColumnNote(column, source, how, detail))

    def drop(self, reason: str, line_numbers: Sequence[int]) -> None:
        lines = [int(n) for n in line_numbers]
        if lines:
            self.drops.append(DropNote(reason, sorted(lines)))

    @property
    def rows_dropped(self) -> int:
        return sum(d.count for d in self.drops)

    @property
    def derived_columns(self) -> List[str]:
        return [c.column for c in self.columns if c.how in ("derived", "filled")]

    @property
    def mapped_columns(self) -> List[str]:
        return [c.column for c in self.columns if c.how == "mapped"]

    def how_it_was_read(self) -> str:
        """One sentence on the file's shape, for the "we coped with this" note."""
        if not self.delimiter:
            return ""
        names = {",": "commas", ";": "semicolons", "\t": "tabs", "|": "pipes"}
        bits = [f"separated by {names.get(self.delimiter, repr(self.delimiter))}"]
        if self.encoding and self.encoding != "utf-8-sig":
            bits.append(f"{self.encoding} text")
        if self.had_bom:
            bits.append("with a UTF-8 byte-order mark")
        if self.decimal_comma:
            bits.append("decimal commas")
        return "Read as " + ", ".join(bits) + "."

    def lines(self) -> List[str]:
        """The whole report as plain sentences, for a log or a test."""
        out = [self.summary] if self.summary else []
        if self.how_it_was_read():
            out.append(self.how_it_was_read())
        out += [f"error: {e}" for e in self.errors]
        out += [f"warning: {w}" for w in self.warnings]
        out += [str(c) for c in self.columns]
        if self.ignored:
            out.append("ignored columns: " + ", ".join(self.ignored))
        out += [str(d) for d in self.drops]
        return out


# ---------------------------------------------------------------- parsing -----


def normalise_header(name: object) -> str:
    """"  Unit Price (€) " -> "unit_price". Also drops pandas' .1 de-duplication."""
    text = str(name).replace("﻿", "").strip().lower()
    text = re.sub(r"\.\d+$", "", text)            # pandas renames a repeat to "product.1"
    text = re.sub(r"\(.*?\)", " ", text)          # units and currencies in brackets
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _header_list(columns, limit: int = 12) -> str:
    """Column names as they appear in the file, shortened so an error stays readable."""
    names = [re.sub(r"\s+", " ", str(c)).strip()[:40] for c in list(columns)[:limit]]
    more = f" and {len(list(columns)) - limit} more" if len(list(columns)) > limit else ""
    return ", ".join(f"\u201c{n}\u201d" for n in names) + more


def _decode(raw: bytes, report: LoadReport) -> Optional[str]:
    """Bytes to text, or a readable error. Rejects things that are not text."""
    if raw.startswith(b"\xef\xbb\xbf"):
        report.had_bom = True
    head = raw[:8192]
    if b"\x00" in head:
        report.fail(
            "That file is not a CSV — it contains binary data (null bytes). "
            "Export the sales as CSV from the till system and upload that."
        )
        return None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        report.encoding = encoding
        # latin-1 decodes anything, so sanity-check that it looks like text.
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in text[:4096])
        if printable / max(len(text[:4096]), 1) < 0.9:
            report.fail(
                "That file is not a CSV — it does not look like text at all. "
                "Export the sales as CSV and upload that."
            )
            return None
        return text
    report.fail("That file could not be decoded as text, so it cannot be a CSV.")
    return None


def _sniff_delimiter(text: str) -> str:
    """Pick the separator from the header line. Dutch Excel writes ';'."""
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(_DELIMITERS)).delimiter
    except csv.Error:
        pass
    header = next((line for line in text.splitlines() if line.strip()), "")
    counts = {d: header.count(d) for d in _DELIMITERS}
    best = max(counts, key=counts.get)
    return best if counts[best] else ","


def _to_number(value: object, decimal_comma: Optional[bool] = None) -> float:
    """"€ 1.234,50" / "1,234.50" / "3,50" / "(2.00)" -> float, else NaN.

    When the thousands/decimal role of a lone comma is ambiguous ("1,234"), a
    comma followed by exactly three digits is read as a thousands separator and
    anything else as a decimal comma. Café prices are "3,50", not "3,500".
    """
    if value is None or isinstance(value, float) and pd.isna(value):
        return float("nan")
    text = _CURRENCY.sub("", str(value)).strip()
    if text.lower() in _BLANKS:
        return float("nan")
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    if text.startswith("+"):
        text = text[1:]
    has_dot, has_comma = "." in text, "," in text
    if has_dot and has_comma:
        # whichever comes last is the decimal separator
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif has_comma:
        tail = text.rsplit(",", 1)[1]
        thousands = len(tail) == 3 and text.count(",") >= 1 and decimal_comma is not True
        text = text.replace(",", "") if thousands else text.replace(",", ".")
    try:
        number = float(text)
    except ValueError:
        return float("nan")
    return -number if negative else number


def _numeric(series: pd.Series, decimal_comma: Optional[bool] = None) -> pd.Series:
    return series.map(lambda v: _to_number(v, decimal_comma)).astype(float)


def _looks_decimal_comma(frame: pd.DataFrame, columns: Sequence[str]) -> bool:
    """True when the money columns use "3,50" rather than "3.50"."""
    commas = dots = 0
    for column in columns:
        for value in frame[column].head(200):
            text = str(value)
            if re.search(r",\d{1,2}\b", text):
                commas += 1
            if re.search(r"\.\d{1,2}\b", text):
                dots += 1
    return commas > dots


def _to_dates(series: pd.Series) -> pd.Series:
    """Parse a date column as forgivingly as pandas allows, mixed formats and all."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.where(~cleaned.str.lower().isin(_BLANKS), None)
    for kwargs in ({"format": "mixed", "dayfirst": True}, {"dayfirst": True}, {}):
        try:
            with warnings.catch_warnings():
                # "Could not infer format" is the normal case here, not news.
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(cleaned, errors="coerce", **kwargs)
        except (ValueError, TypeError):
            continue
        if parsed.notna().any() or cleaned.isna().all():
            return parsed
    return pd.to_datetime(pd.Series([None] * len(series), index=series.index))


def _hours_from_text(series: pd.Series) -> pd.Series:
    """An hour column that holds "14", "14:30", "2:30 PM" or "14.30"."""

    def one(value: object) -> float:
        text = str(value).strip()
        if text.lower() in _BLANKS:
            return float("nan")
        pm = bool(re.search(r"\bp\.?m\.?\b", text, re.I))
        am = bool(re.search(r"\ba\.?m\.?\b", text, re.I))
        match = _TIME_IN_STRING.search(text)
        hour = float(match.group(1)) if match else _to_number(text)
        if pd.isna(hour):
            return float("nan")
        if pm and hour < 12:
            hour += 12
        if am and hour == 12:
            hour = 0
        return hour

    return series.map(one).astype(float)


# ---------------------------------------------------------------- loader ------


Source = Union[str, Path, bytes, bytearray]


def load_sales_frame(
    source: Source,
    filename: Optional[str] = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Tuple[Optional[pd.DataFrame], LoadReport]:
    """Read sales from a path or from raw bytes, and report on it.

    Args:
        source: a path to a CSV, or the file's bytes (what Dash hands you).
        filename: the name to show in the report, when `source` is bytes.
        max_bytes: refuse anything larger. Default 5 MB; see MAX_UPLOAD_BYTES.

    Returns:
        (frame, report). `frame` is None when nothing usable could be read; the
        report then says why. When `frame` is not None it has exactly the
        columns and dtypes `load_sales_data` returns, sorted by date.

    Raises:
        Nothing. Every failure comes back inside the report.
    """
    report = LoadReport(source=filename or (str(source) if not isinstance(source, (bytes, bytearray)) else "uploaded file"))

    # -- bytes ------------------------------------------------------------
    if isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    else:
        path = Path(source)
        if not path.exists():
            return None, report.fail(f"No such file: {path}")
        try:
            raw = path.read_bytes()
        except OSError as error:
            return None, report.fail(f"That file could not be read: {error.strerror or error}")

    if len(raw) > max_bytes:
        return None, report.fail(
            f"That file is {len(raw) / 1024 / 1024:.1f} MB and the limit is "
            f"{max_bytes / 1024 / 1024:.0f} MB. Split it by month, or send a date range."
        )
    if not raw.strip():
        return None, report.fail("That file is empty — there is nothing in it to read.")

    text = _decode(raw, report)
    if text is None:
        return None, report

    # -- table ------------------------------------------------------------
    report.delimiter = _sniff_delimiter(text)
    try:
        frame = pd.read_csv(
            io.StringIO(text),
            sep=report.delimiter,
            dtype=str,
            keep_default_na=False,
            skip_blank_lines=True,
            engine="python",
        )
    except (pd.errors.EmptyDataError, StopIteration):
        return None, report.fail("That file has no header row — the first line is blank or missing.")
    except pd.errors.ParserError as error:
        return None, report.fail(
            "That file could not be parsed as a table: "
            f"{str(error).splitlines()[0][:160]}. Check for a stray quote or a ragged row."
        )
    except Exception as error:   # a CSV can be broken in ways pandas names badly
        return None, report.fail(f"That file could not be parsed as a table ({type(error).__name__}).")

    if frame.shape[1] == 0:
        return None, report.fail("That file has no columns.")
    report.rows_read = len(frame)
    # Original line number of each row: header is line 1, so row 0 is line 2.
    frame = frame.reset_index(drop=True)
    line_numbers = pd.Series(frame.index + 2, index=frame.index)

    # -- headers ----------------------------------------------------------
    chosen: Dict[str, str] = {}      # canonical -> original header
    for original in frame.columns:
        canonical = _ALIAS_TO_CANONICAL.get(normalise_header(original))
        if canonical is None:
            report.ignored.append(f"{original} (not a column this dashboard knows)")
        elif canonical in chosen:
            report.ignored.append(f"{original} (a second “{canonical}” column; the first one was used)")
        else:
            chosen[canonical] = original

    if not chosen:
        return None, report.fail(
            "None of the columns in that file could be recognised. Found: "
            f"{_header_list(frame.columns)}. A sales export needs at least a date, "
            "an hour and a revenue or price column."
        )

    if len(frame) == 0:
        report.ok = False
        for canonical, original in chosen.items():
            report.note(canonical, str(original), "mapped")
        report.summary = (
            "That file has a header row and no sales in it. The headers were read "
            f"({', '.join(sorted(chosen))}), so the export probably covered an empty date range."
        )
        report.errors.append("No data rows found.")
        return None, report

    source_frame = {canonical: frame[original] for canonical, original in chosen.items()}
    money_columns = [c for c in ("unit_price", "revenue", "quantity") if c in source_frame]
    report.decimal_comma = bool(money_columns) and _looks_decimal_comma(
        pd.DataFrame({c: source_frame[c] for c in money_columns}), money_columns
    )
    decimal_hint = True if report.decimal_comma else None

    out = pd.DataFrame(index=frame.index)

    # -- date (mandatory) -------------------------------------------------
    if "date" not in source_frame:
        return None, report.fail(
            "That file has no date column, and a date cannot be worked out from "
            "anything else. Add a column called date, datum or timestamp. "
            "Columns found: " + ", ".join(str(c) for c in frame.columns[:12]) + "."
        )
    parsed_dates = _to_dates(source_frame["date"])
    if parsed_dates.notna().sum() == 0:
        sample = ", ".join(repr(v) for v in source_frame["date"].head(3))
        return None, report.fail(
            f"Not one value in the date column “{chosen['date']}” could be read as a date "
            f"(for example {sample}). Export dates as 2026-04-01 or 01-04-2026."
        )
    has_time = bool((parsed_dates.dt.hour.fillna(0) != 0).any() or (parsed_dates.dt.minute.fillna(0) != 0).any())
    out["date"] = parsed_dates.dt.normalize()
    report.note("date", str(chosen["date"]), "mapped")

    # -- hour (mandatory, but a timestamp counts) -------------------------
    if "hour" in source_frame:
        out["hour"] = _hours_from_text(source_frame["hour"])
        report.note("hour", str(chosen["hour"]), "mapped")
    elif has_time:
        out["hour"] = parsed_dates.dt.hour.astype(float)
        report.note("hour", str(chosen["date"]), "derived",
                    f"taken from the time of day in “{chosen['date']}”")
    else:
        return None, report.fail(
            "That file has no hour column and its dates carry no time of day, so "
            "there is no way to tell when in the day each sale happened — and this "
            "dashboard is entirely about the shape of the day. Add an hour, uur or "
            "tijd column, or export the date as a full timestamp."
        )

    # -- money ------------------------------------------------------------
    quantity = _numeric(source_frame["quantity"], decimal_hint) if "quantity" in source_frame else None
    unit_price = _numeric(source_frame["unit_price"], decimal_hint) if "unit_price" in source_frame else None
    revenue = _numeric(source_frame["revenue"], decimal_hint) if "revenue" in source_frame else None

    if revenue is None and (quantity is None or unit_price is None):
        missing = [c for c in ("revenue", "quantity", "unit_price") if c not in source_frame]
        return None, report.fail(
            "That file has no money in it that can be used: there is no revenue "
            "column, and quantity × unit_price cannot be worked out either "
            f"(missing {', '.join(missing)}). Add a revenue/omzet/totaal column, "
            "or both an aantal and a prijs column."
        )

    if quantity is None:
        quantity = pd.Series(1.0, index=frame.index)
        report.note("quantity", None, "filled",
                    "set to 1 — the file exports line totals, not counts")
    else:
        report.note("quantity", str(chosen["quantity"]), "mapped")

    if revenue is None:
        revenue = quantity * unit_price
        report.note("revenue", None, "derived",
                    f"quantity × unit_price (from “{chosen['quantity']}” and “{chosen['unit_price']}”)")
    else:
        report.note("revenue", str(chosen["revenue"]), "mapped")

    if unit_price is None:
        divisor = quantity.where(quantity != 0)
        unit_price = (revenue / divisor).replace([float("inf"), float("-inf")], float("nan"))
        report.note("unit_price", None, "derived", "revenue ÷ quantity")
    else:
        report.note("unit_price", str(chosen["unit_price"]), "mapped")

    out["quantity"] = quantity
    out["unit_price"] = unit_price
    out["revenue"] = revenue

    # -- day_of_week ------------------------------------------------------
    if "day_of_week" in source_frame:
        given = _numeric(source_frame["day_of_week"], decimal_hint)
        derived = out["date"].dt.dayofweek.astype(float)
        if given.notna().any() and given.between(0, 6).all():
            out["day_of_week"] = given
            report.note("day_of_week", str(chosen["day_of_week"]), "mapped")
            disagree = int((given.notna() & (given != derived)).sum())
            if disagree:
                report.warnings.append(
                    f"{disagree} row(s) have a day_of_week that does not match their date. "
                    "The column was kept as exported."
                )
        else:
            out["day_of_week"] = derived
            report.note("day_of_week", None, "derived",
                        f"worked out from the date — “{chosen['day_of_week']}” was not a number 0-6 "
                        "(Monday is 0)")
            report.ignored.append(f"{chosen['day_of_week']} (not a number 0-6)")
    else:
        out["day_of_week"] = out["date"].dt.dayofweek.astype(float)
        report.note("day_of_week", None, "derived", "worked out from the date (Monday is 0)")

    # -- labels -----------------------------------------------------------
    for column in ("category", "product"):
        if column in source_frame:
            values = source_frame[column].astype(str).str.strip()
            out[column] = values.where(~values.str.lower().isin(_BLANKS), "Unknown")
            report.note(column, str(chosen[column]), "mapped")
        else:
            out[column] = "Unknown"
            report.note(column, None, "filled", "no such column in the file; set to “Unknown”")

    # -- row-level cleaning ----------------------------------------------
    keep = pd.Series(True, index=out.index)

    def reject(mask: pd.Series, reason: str) -> None:
        mask = mask.fillna(False) & keep
        if mask.any():
            report.drop(reason, line_numbers[mask].tolist())
            keep.loc[mask] = False

    reject(out["date"].isna(), "the date could not be read")
    reject(out["hour"].isna(), "the hour could not be read")
    reject(~out["hour"].between(0, 23), "the hour is not between 0 and 23")
    reject(out["revenue"].isna(), "revenue was not a number")
    reject(out["quantity"].isna(), "quantity was not a number")
    reject(out["quantity"] <= 0, "quantity was zero or negative (a refund or a void, not a sale)")
    reject(out["revenue"] < 0, "revenue was negative (a refund, not a sale)")
    # A price we cannot read is repairable from a revenue we can: the sale is real
    # either way, and revenue is what almost every number on the page is made of.
    repairable = out["unit_price"].isna() & out["revenue"].notna() & (out["quantity"] > 0) & keep
    if repairable.any():
        out.loc[repairable, "unit_price"] = (
            out.loc[repairable, "revenue"] / out.loc[repairable, "quantity"]
        )
        report.warnings.append(
            f"{int(repairable.sum())} row(s) had a unit price that could not be read. "
            "It was worked out from revenue ÷ quantity for those rows; the sales themselves "
            "were kept."
        )
    reject(out["unit_price"].isna(), "the unit price could not be read or worked out")
    reject(out["unit_price"] < 0, "the unit price was negative")

    clean = out[keep].copy()
    report.rows_kept = len(clean)

    if clean.empty:
        report.ok = False
        report.summary = (
            f"All {report.rows_read} row(s) in that file had to be dropped, so there is "
            "nothing to analyse. The reasons are listed below."
        )
        report.errors.append("No usable rows left after cleaning.")
        return None, report

    # -- things worth saying but not worth dropping a row for -------------
    mismatch = (clean["revenue"] - clean["quantity"] * clean["unit_price"]).abs() > 0.01
    if mismatch.any():
        worst = (clean["revenue"] - clean["quantity"] * clean["unit_price"]).abs().max()
        report.warnings.append(
            f"{int(mismatch.sum())} row(s) have a revenue that is not quantity × unit_price "
            f"(largest gap {worst:,.2f}). The exported revenue was kept, since that is what "
            "the till rang up, but the prices in those rows are not reliable."
        )

    # Identical rows are kept: two flat whites in the same hour are not an error,
    # and in the sample file 45% of rows are exact duplicates of another. The
    # failure that does matter is the same export pasted in twice, so look for
    # that shape instead of counting duplicate rows.
    report.duplicate_rows = int(clean.duplicated(keep="first").sum())
    half = len(clean) // 2
    if half and len(clean) % 2 == 0:
        top = clean.iloc[:half].reset_index(drop=True)
        bottom = clean.iloc[half:].reset_index(drop=True)
        if top.equals(bottom):
            report.warnings.append(
                "The second half of this file is an exact copy of the first half. "
                "The export looks like it was run or pasted twice, so every total "
                "on this page is double what the café actually took."
            )

    if clean["date"].nunique() < 14:
        report.warnings.append(
            f"Only {clean['date'].nunique()} trading day(s) of data. The confidence scores "
            "will be low; the sample-size rule gives full marks from 30 days."
        )

    # -- dtypes that match load_sales_data -------------------------------
    clean["hour"] = clean["hour"].round().astype("int64")
    clean["day_of_week"] = clean["day_of_week"].round().astype("int64")
    if (clean["quantity"] % 1 == 0).all():
        clean["quantity"] = clean["quantity"].astype("int64")
    clean["unit_price"] = clean["unit_price"].astype(float)
    clean["revenue"] = clean["revenue"].astype(float)
    clean = clean[list(CANONICAL_COLUMNS)].sort_values("date").reset_index(drop=True)

    report.ok = True
    first, last = clean["date"].min(), clean["date"].max()
    span = f"{first.day} {first:%B %Y} to {last.day} {last:%B %Y}"
    dropped = f" ({report.rows_dropped:,} dropped)" if report.rows_dropped else ""
    report.summary = (
        f"Read {report.rows_kept:,} of {report.rows_read:,} rows{dropped}, "
        f"{clean['date'].nunique():,} trading day(s) from {span}."
    )
    return clean, report


def read_upload(
    contents: Optional[str],
    filename: Optional[str] = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
) -> Tuple[Optional[pd.DataFrame], LoadReport]:
    """Read the base64 data URL `dcc.Upload` puts in its `contents` property.

    The bytes never reach the filesystem: they are decoded in memory and handed
    straight to `load_sales_frame`, so an upload cannot overwrite the sample
    data or leave anything behind on the server.

    Args:
        contents: "data:text/csv;base64,....", as Dash provides it.
        filename: the browser's name for the file, used in the report only.
        max_bytes: the size cap, applied to the decoded bytes.

    Returns:
        (frame, report), exactly as `load_sales_frame`. Never raises.
    """
    report = LoadReport(source=filename or "uploaded file")
    if not contents:
        return None, report.fail("No file was uploaded.")

    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls", ".xlsm", ".ods")):
        return None, report.fail(
            f"“{filename}” is a spreadsheet, and this app reads CSV only. In Excel: "
            "File → Save As → CSV, then upload that. A semicolon-separated Dutch "
            "export is fine."
        )
    if name and not name.endswith(".csv") and not name.endswith(".txt"):
        report.warnings.append(
            f"“{filename}” does not end in .csv. It was read as CSV anyway; if the "
            "report below looks wrong, that is probably why."
        )

    _, _, payload = contents.partition(",")
    try:
        raw = base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        return None, report.fail("That upload could not be decoded. Try choosing the file again.")

    frame, parse_report = load_sales_frame(raw, filename=filename, max_bytes=max_bytes)
    # keep the extension warning raised above
    parse_report.warnings = report.warnings + parse_report.warnings
    return frame, parse_report
