"""The bounded per-visitor store: what it keeps, what it throws away, and when.

These tests are about the ceiling. A per-session cache with no cap is a memory
leak, and this one runs on a 512 MB instance, so the interesting cases are all
about pressure: too many sessions, too many bytes, too old, too big to keep at
all. Time is injected rather than slept on, so nothing here is timing-flaky.
"""

import threading

import pytest

from src.session_state import MAX_STATUSES_PER_SESSION, SessionStore


class Clock:
    """A hand-wound clock, so age can be tested without sleeping."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def tick(self, seconds):
        self.now += seconds


def store(**kwargs):
    kwargs.setdefault("clock", Clock())
    kwargs.setdefault("max_sessions", 3)
    kwargs.setdefault("max_bytes", 1000)
    kwargs.setdefault("ttl_seconds", 100)
    return SessionStore(**kwargs), kwargs["clock"]


# ---------------------------------------------------------------- the basics ---


def test_a_dataset_goes_in_and_comes_back_out_for_that_session_only():
    state, _ = store()
    assert state.put_dataset("a", "A's frame", 10) is True

    assert state.dataset("a") == "A's frame"
    assert state.dataset("b") is None, "another session must see nothing"
    assert state.dataset(None) is None
    assert state.stats()["bytes"] == 10


def test_stamps_are_per_session_and_never_leak():
    state, _ = store()
    state.set_status("a", "afternoon_dip", "verified")

    assert state.statuses("a") == {"afternoon_dip": "verified"}
    assert state.statuses("b") == {}

    # The returned dict is a copy: a caller cannot reach in and edit the store.
    state.statuses("a")["afternoon_dip"] = "dismissed"
    assert state.statuses("a") == {"afternoon_dip": "verified"}


def test_a_second_upload_replaces_the_first_without_double_counting_its_bytes():
    state, _ = store()
    state.put_dataset("a", "one", 400)
    state.put_dataset("a", "two", 300)

    assert state.dataset("a") == "two"
    assert state.stats()["bytes"] == 300
    assert state.stats()["sessions"] == 1


def test_nothing_about_a_junk_session_id_raises():
    state, _ = store()
    for junk in (None, "", 0, 17, b"bytes", [], {}, object()):
        assert state.put_dataset(junk, "frame", 10) is False
        assert state.dataset(junk) is None
        assert state.statuses(junk) == {}
        assert state.was_dropped(junk) is False
        assert state.set_status(junk, "k", "v") is False
        state.clear_dataset(junk)
        state.drop(junk)
    assert state.stats()["sessions"] == 0


def test_a_dataset_with_no_measurable_size_is_charged_nothing_but_still_bounded():
    state, _ = store()
    assert state.put_dataset("a", "frame", "not a number") is True
    assert state.stats()["bytes"] == 0
    assert state.put_dataset("b", "frame", -5) is True
    assert state.stats()["bytes"] == 0


# ---------------------------------------------------------------- the caps -----


def test_the_count_cap_evicts_the_least_recently_used_session():
    state, _ = store(max_sessions=3, max_bytes=10_000)
    for name in ("a", "b", "c"):
        state.put_dataset(name, name, 10)

    state.dataset("a")  # a is used again, so b is now the oldest
    state.put_dataset("d", "d", 10)

    assert state.stats()["sessions"] == 3
    assert state.dataset("b") is None, "the least recently used went first"
    assert state.dataset("a") == "a"
    assert state.dataset("c") == "c"
    assert state.dataset("d") == "d"


def test_the_byte_cap_evicts_until_the_total_fits():
    state, _ = store(max_sessions=100, max_bytes=1000)
    state.put_dataset("a", "a", 400)
    state.put_dataset("b", "b", 400)
    state.put_dataset("c", "c", 400)

    stats = state.stats()
    assert stats["bytes"] <= 1000
    assert stats["with_dataset"] == 2
    assert state.dataset("a") is None, "the oldest paid for the newest"
    assert state.dataset("c") == "c"


def test_one_visitor_cannot_take_more_than_their_share_of_the_cache():
    """The per-entry cap: half the total by default, so one file cannot evict all.

    Without it a single legal upload can fill the whole cache: a three-column
    5 MB CSV accounts for 47 MiB of pandas against a 48 MiB total, so it would
    arrive, evict every other session, and sit there alone.
    """
    state, _ = store(max_sessions=100, max_bytes=1000)
    assert state.max_entry_bytes == 500

    state.put_dataset("a", "a", 300)
    assert state.put_dataset("hog", "most of the cache", 900) is False

    assert state.dataset("hog") is None
    assert state.dataset("a") == "a", "the other session is untouched"
    assert state.stats()["bytes"] == 300

    # Exactly at the cap is allowed; one byte over is not.
    assert state.put_dataset("edge", "fits", 500) is True
    assert state.put_dataset("over", "does not", 501) is False


def test_the_per_entry_cap_can_be_set_explicitly():
    state, _ = store(max_sessions=100, max_bytes=1000, max_entry_bytes=100)
    assert state.put_dataset("a", "a", 150) is False
    assert state.put_dataset("b", "b", 100) is True


def test_a_dataset_larger_than_the_whole_cache_is_refused_and_evicts_nobody():
    state, _ = store(max_sessions=100, max_bytes=1000)
    state.put_dataset("a", "a", 500)
    state.set_status("big", "trend", "verified")

    assert state.put_dataset("big", "enormous", 5000) is False

    assert state.dataset("big") is None
    assert state.dataset("a") == "a", "a well-behaved session is not evicted for it"
    assert state.statuses("big") == {"trend": "verified"}, "its stamps survive"
    assert state.stats()["bytes"] == 500


def test_an_entry_untouched_for_the_ttl_is_dropped():
    state, clock = store(ttl_seconds=100)
    state.put_dataset("a", "a", 10)

    clock.tick(99)
    assert state.dataset("a") == "a", "still inside the window, and now touched again"

    clock.tick(101)
    assert state.dataset("a") is None
    assert state.was_dropped("a") is True
    assert state.stats()["bytes"] == 0


def test_a_session_that_keeps_being_used_does_not_expire():
    state, clock = store(ttl_seconds=100)
    state.put_dataset("a", "a", 10)
    for _ in range(10):
        clock.tick(50)
        assert state.dataset("a") == "a"


# ---------------------------------------------------------------- what the visitor is told ---


def test_an_evicted_session_is_remembered_so_the_visitor_can_be_told():
    state, _ = store(max_sessions=1, max_bytes=10_000)
    state.put_dataset("a", "a", 10)
    state.put_dataset("b", "b", 10)

    assert state.dataset("a") is None
    assert state.was_dropped("a") is True
    assert state.was_dropped("b") is False


def test_going_back_to_the_sample_on_purpose_is_not_reported_as_a_loss():
    state, _ = store()
    state.put_dataset("a", "a", 10)
    state.set_status("a", "trend", "dismissed")

    state.clear_dataset("a")

    assert state.dataset("a") is None
    assert state.was_dropped("a") is False, "the visitor asked for this"
    assert state.statuses("a") == {"trend": "dismissed"}, "stamps are not the file"
    assert state.stats()["bytes"] == 0


def test_uploading_again_clears_the_dropped_mark():
    state, _ = store(max_sessions=1)
    state.put_dataset("a", "a", 10)
    state.put_dataset("b", "b", 10)
    assert state.was_dropped("a") is True

    state.put_dataset("a", "a again", 10)
    assert state.was_dropped("a") is False
    assert state.dataset("a") == "a again"


def test_the_memory_of_dropped_sessions_is_itself_bounded():
    state, _ = store(max_sessions=1, max_bytes=10_000, eviction_memory=5)
    for i in range(50):
        state.put_dataset(f"s{i}", "frame", 10)

    stats = state.stats()
    assert stats["dropped_remembered"] == 5, "a bounded ring, not a growing list"
    assert stats["sessions"] == 1


def test_stamps_per_session_are_capped():
    state, _ = store()
    for i in range(MAX_STATUSES_PER_SESSION + 10):
        state.set_status("a", f"pattern_{i}", "verified")

    assert len(state.statuses("a")) == MAX_STATUSES_PER_SESSION
    # An existing key can still be updated after the cap is reached.
    assert state.set_status("a", "pattern_0", "dismissed") is True
    assert state.statuses("a")["pattern_0"] == "dismissed"


# ---------------------------------------------------------------- pressure -----


def test_eviction_under_sustained_pressure_never_raises_and_stays_bounded():
    state, clock = store(max_sessions=5, max_bytes=1000, ttl_seconds=100)
    for i in range(500):
        session = f"visitor-{i}"
        assert state.put_dataset(session, f"frame {i}", 300) is True
        state.set_status(session, "trend", "verified")
        if i % 7 == 0:
            clock.tick(30)
        stats = state.stats()
        assert stats["sessions"] <= 5
        assert stats["bytes"] <= 1000
        assert stats["bytes"] >= 0

    assert state.dataset("visitor-499") == "frame 499", "the latest visitor is served"


def test_many_threads_hammering_the_store_leave_it_consistent():
    state = SessionStore(max_sessions=8, max_bytes=4000, ttl_seconds=1000)
    errors = []

    def work(worker):
        try:
            for i in range(200):
                session = f"w{worker}-{i % 5}"
                state.put_dataset(session, i, 500)
                state.set_status(session, "trend", "verified")
                state.dataset(session)
                state.statuses(session)
                if i % 10 == 0:
                    state.clear_dataset(session)
        except Exception as error:  # noqa: BLE001 - the point of the test
            errors.append(error)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    stats = state.stats()
    assert stats["sessions"] <= 8
    assert 0 <= stats["bytes"] <= 4000
    # The byte count must still match what is actually held, not have drifted.
    assert stats["bytes"] == sum(
        entry.nbytes for entry in state._entries.values()  # noqa: SLF001 - accounting check
    )


def test_drop_and_clear_free_everything():
    state, _ = store()
    state.put_dataset("a", "a", 100)
    state.set_status("a", "trend", "verified")

    state.drop("a")
    assert state.dataset("a") is None and state.statuses("a") == {}
    assert state.stats()["bytes"] == 0

    state.put_dataset("b", "b", 100)
    state.clear()
    assert state.stats() == {
        "sessions": 0, "with_dataset": 0, "bytes": 0, "dropped_remembered": 0,
        "max_sessions": 3, "max_bytes": 1000, "max_entry_bytes": 500,
    }


def test_the_real_defaults_are_sane_for_a_512_mb_instance():
    from src import session_state

    assert session_state.MAX_SESSIONS >= 1
    # A 5 MB CSV (the upload limit) accounts for 17.8 MiB of pandas when it is a
    # wide export and 47.0 MiB when it is three narrow columns, both measured.
    # The total must hold two per-entry caps and stay far below 512 MB, and the
    # per-entry cap must still admit an ordinary full-size export.
    assert 24 * 1024 * 1024 <= session_state.MAX_TOTAL_BYTES <= 128 * 1024 * 1024
    assert session_state.MAX_ENTRY_BYTES * 2 <= session_state.MAX_TOTAL_BYTES
    assert session_state.MAX_ENTRY_BYTES >= 18 * 1024 * 1024
    assert 60 <= session_state.SESSION_TTL_SECONDS <= 4 * 3600
    assert SessionStore().max_entry_bytes == session_state.MAX_ENTRY_BYTES


@pytest.mark.parametrize("max_sessions", [0, -1])
def test_a_nonsense_cap_still_leaves_a_usable_store(max_sessions):
    state = SessionStore(max_sessions=max_sessions, max_bytes=100, ttl_seconds=100)
    assert state.put_dataset("a", "a", 10) is True
    assert state.dataset("a") == "a"
    assert state.stats()["sessions"] == 1
