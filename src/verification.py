"""Track insight verification status (pending, verified, dismissed) in SQLite.

SQL SAFETY NOTE:
All queries in this module use parameterized statements (? placeholders with tuple args).
This prevents SQL injection by separating query structure from data.

✓ SAFE:   c.execute("SELECT * FROM insights WHERE id = ?", (insight_id,))
✗ UNSAFE: c.execute(f"SELECT * FROM insights WHERE id = {insight_id}")

Parameterized queries ensure user input is always treated as data, never as code.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


DB_PATH = "insights.db"


def init_db() -> None:
    """Initialize the insights verification database."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS insights (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            confidence_score INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'pending',
            verified_at TIMESTAMP,
            dismissed_at TIMESTAMP,
            notes TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_insight(
    title: str,
    pattern_type: str,
    confidence_score: int,
    status: str = "pending",
) -> int:
    """
    Save an insight to the database.

    Args:
        title: Insight title.
        pattern_type: Type of pattern (e.g., "afternoon_dip").
        confidence_score: Confidence score (0–100).
        status: Initial status ("pending", "verified", "dismissed").

    Returns:
        ID of the inserted insight.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute(
        """
        INSERT INTO insights (title, pattern_type, confidence_score, status)
        VALUES (?, ?, ?, ?)
        """,
        (title, pattern_type, confidence_score, status),
    )

    insight_id = c.lastrowid
    conn.commit()
    conn.close()

    return insight_id


def update_insight_status(insight_id: int, new_status: str, notes: str = None) -> None:
    """
    Update an insight's status (pending → verified or dismissed).

    Args:
        insight_id: ID of the insight.
        new_status: New status ("verified" or "dismissed").
        notes: Optional notes about the verification/dismissal.
    """
    if new_status not in ("pending", "verified", "dismissed"):
        raise ValueError(f"Invalid status: {new_status}")

    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if new_status == "verified":
        c.execute(
            """
            UPDATE insights
            SET status = ?, verified_at = CURRENT_TIMESTAMP, notes = ?
            WHERE id = ?
            """,
            (new_status, notes, insight_id),
        )
    elif new_status == "dismissed":
        c.execute(
            """
            UPDATE insights
            SET status = ?, dismissed_at = CURRENT_TIMESTAMP, notes = ?
            WHERE id = ?
            """,
            (new_status, notes, insight_id),
        )

    conn.commit()
    conn.close()


def get_insights(status: str = None) -> List[Dict]:
    """
    Retrieve insights, optionally filtered by status.

    Args:
        status: Filter by status ("pending", "verified", "dismissed"). None = all.

    Returns:
        List of insight dicts.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    if status is None:
        c.execute("SELECT * FROM insights ORDER BY created_at DESC")
    else:
        c.execute(
            "SELECT * FROM insights WHERE status = ? ORDER BY created_at DESC",
            (status,),
        )

    rows = c.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_insight_by_id(insight_id: int) -> Dict:
    """
    Retrieve a specific insight by ID.

    Args:
        insight_id: ID of the insight.

    Returns:
        Insight dict, or None if not found.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("SELECT * FROM insights WHERE id = ?", (insight_id,))
    row = c.fetchone()
    conn.close()

    return dict(row) if row else None


def clear_db() -> None:
    """Clear all insights from the database (for testing/reset)."""
    if Path(DB_PATH).exists():
        Path(DB_PATH).unlink()
