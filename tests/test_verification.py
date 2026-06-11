"""Tests for verification module."""

import pytest
from src.verification import (
    save_insight,
    update_insight_status,
    get_insights,
    get_insight_by_id,
    clear_db,
)


@pytest.fixture(autouse=True)
def cleanup():
    """Clear database before and after each test."""
    clear_db()
    yield
    clear_db()


def test_save_insight():
    """Test saving an insight."""
    insight_id = save_insight(
        title="Test Insight",
        pattern_type="test",
        confidence_score=75,
    )
    assert insight_id == 1


def test_get_insights_all():
    """Test retrieving all insights."""
    save_insight("Test 1", "test", 75)
    save_insight("Test 2", "test", 80)

    insights = get_insights()
    assert len(insights) == 2


def test_get_insights_by_status():
    """Test retrieving insights filtered by status."""
    id1 = save_insight("Test 1", "test", 75, status="pending")
    id2 = save_insight("Test 2", "test", 80, status="verified")

    pending = get_insights(status="pending")
    assert len(pending) == 1
    assert pending[0]["id"] == id1

    verified = get_insights(status="verified")
    assert len(verified) == 1
    assert verified[0]["id"] == id2


def test_update_insight_status():
    """Test updating insight status."""
    insight_id = save_insight("Test", "test", 75)

    update_insight_status(insight_id, "verified", notes="Confirmed pattern")

    insight = get_insight_by_id(insight_id)
    assert insight["status"] == "verified"
    assert insight["notes"] == "Confirmed pattern"
    assert insight["verified_at"] is not None


def test_update_insight_dismissed():
    """Test dismissing an insight."""
    insight_id = save_insight("Test", "test", 75)

    update_insight_status(insight_id, "dismissed", notes="False positive")

    insight = get_insight_by_id(insight_id)
    assert insight["status"] == "dismissed"
    assert insight["dismissed_at"] is not None


def test_get_insight_by_id():
    """Test retrieving a specific insight."""
    insight_id = save_insight("Test", "test", 75)
    insight = get_insight_by_id(insight_id)

    assert insight["title"] == "Test"
    assert insight["confidence_score"] == 75


def test_get_nonexistent_insight():
    """Test retrieving non-existent insight."""
    insight = get_insight_by_id(999)
    assert insight is None


def test_invalid_status():
    """Test that invalid status raises error."""
    insight_id = save_insight("Test", "test", 75)

    with pytest.raises(ValueError, match="Invalid status"):
        update_insight_status(insight_id, "invalid")
