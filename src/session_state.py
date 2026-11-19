"""Per-visitor state for the live dashboard, bounded in count, bytes and age.

Why this exists
---------------
The dashboard used to keep the dataset on show in a module-level ``CURRENT``.
One process, one global: whoever uploaded last decided what every other
visitor saw on their next request, label and totals included. On a public
instance with no login that is a data leak, not just a wrong screen.

This module is the replacement: a small in-memory store, keyed by an opaque
session id, holding one visitor's parsed dataset and their own Verify/Dismiss
stamps. Nothing here touches the disk.

The caps are the point
----------------------
A per-session cache with no ceiling is a memory leak with extra steps, and the
free instance this runs on has 512 MB. So the store is bounded four ways:

* **count** — at most ``MAX_SESSIONS`` entries exist at once;
* **bytes** — the retained datasets add up to at most ``MAX_TOTAL_BYTES``;
* **one visitor's share** — a single dataset over ``MAX_ENTRY_BYTES`` is never
  retained, so no one upload can evict the whole cache for itself;
* **age** — an entry untouched for ``SESSION_TTL_SECONDS`` is dropped.

Over the count or the total, the least-recently-used entry goes first. A
dataset over the per-entry cap is refused outright (``put_dataset`` returns
False) and evicts nobody; the caller is expected to render it once and tell the
visitor it was not kept.

When an entry that held a dataset is dropped, its id is remembered in a small
bounded ring (``EVICTION_MEMORY`` ids, no data) so the visitor can be told
their file is gone instead of silently seeing the sample again. That covers the
two in-process paths, LRU eviction and age expiry. It cannot cover a worker
restart: the ring lives in the same process as the datasets it explains, so
after a recycle the visitor's cookie is still valid, their file is gone, and
they are shown the sample with no note. Telling them that would need state that
outlives the process, which is exactly what this design does not have.

Thread safety: every public method takes one lock, so the accounting cannot
drift and a reader cannot see a half-written entry.

**Do not add ``--threads`` to the gunicorn command without reading this.**
Today ``render.yaml`` runs one sync worker with no threads, so requests
serialise: the cost of ten simultaneous uploads is latency, not memory (six
concurrent 5 MB posts measured 369.9 MB peak RSS and a ~1.9 s queue per
request, because a waiting request's body sits in kernel socket buffers, not in
the worker). That single sync worker is accidental protection, not a design.
Adding threads converts the queue into concurrent parses: driving the same
parse path concurrently in-process measured 591 MB for two and 770 MB for
three, against a 512 MB limit. If threads are ever wanted, first put a
semaphore of one around the parse-and-analyse path in app/dashboard.py, or drop
the upload cap far enough that N concurrent parses fit — and measure it inside
a real 512 MB cgroup, because this is the failure that gets the worker
SIGKILLed rather than the one that shows up as an error.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional

# How many visitors can hold state at once. Each entry is a dict and a few
# strings unless it holds a dataset, so the real ceiling is the byte cap below;
# this one stops an unbounded number of tiny entries accumulating.
MAX_SESSIONS = 24

# Total retained dataset bytes, across every session.
#
# What a 5 MB CSV (the upload limit) costs depends entirely on its shape, and
# the accounted figures are measured, not assumed: a wide eight-column export at
# the limit is 17.8 MiB of pandas, a three-column one at the same 5 MB is 47.0
# MiB, because the thinner the rows the more of them fit. So "a worst-case
# upload" has no single size, and a total cap alone would let one thin file
# evict every other session for itself.
#
# Hence the pair: nothing over MAX_ENTRY_BYTES is ever retained (it is still
# rendered once for the visitor who sent it, with a note that it was not kept),
# and what is retained adds up to at most MAX_TOTAL_BYTES — two full-sized
# entries. Against a worker that boots at ~170 MB inside 512 MB, that is ~48 MB
# of cache on top of a measured 333 MB sequential worst-case peak, leaving
# roughly 150 MB of headroom.
MAX_TOTAL_BYTES = 48 * 1024 * 1024

# No single visitor's dataset may take more than half the cache.
MAX_ENTRY_BYTES = MAX_TOTAL_BYTES // 2

# An idle visitor's file is forgotten after half an hour.
SESSION_TTL_SECONDS = 30 * 60

# Ids remembered after their dataset was dropped, so the next request can say so.
EVICTION_MEMORY = 256

# A visitor cannot accumulate more stamps than there are findings; this is a
# generous ceiling so a crafted request cannot grow one entry without bound.
MAX_STATUSES_PER_SESSION = 64


class Entry:
    """One visitor's state: their dataset, if any, and their own stamps."""

    __slots__ = ("dataset", "nbytes", "statuses", "last_used")

    def __init__(self, last_used: float = 0.0):
        self.dataset: Any = None
        self.nbytes: int = 0
        self.statuses: Dict[str, str] = {}
        self.last_used: float = last_used


class SessionStore:
    """A bounded LRU/TTL store of per-visitor state. Never raises on bad input."""

    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        max_bytes: int = MAX_TOTAL_BYTES,
        ttl_seconds: float = SESSION_TTL_SECONDS,
        eviction_memory: int = EVICTION_MEMORY,
        max_entry_bytes: Optional[int] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.max_sessions = max(1, int(max_sessions))
        self.max_bytes = max(0, int(max_bytes))
        # Half the cache by default, so one visitor cannot evict all the others.
        self.max_entry_bytes = (
            self.max_bytes // 2 if max_entry_bytes is None else max(0, int(max_entry_bytes))
        )
        self.ttl_seconds = float(ttl_seconds)
        self.eviction_memory = max(0, int(eviction_memory))
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, Entry]" = OrderedDict()
        self._dropped: "OrderedDict[str, bool]" = OrderedDict()
        self._bytes = 0

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _usable(session_id: Any) -> bool:
        return isinstance(session_id, str) and bool(session_id)

    def _remember_dropped(self, session_id: str) -> None:
        if not self.eviction_memory:
            return
        self._dropped.pop(session_id, None)
        self._dropped[session_id] = True
        while len(self._dropped) > self.eviction_memory:
            self._dropped.popitem(last=False)

    def _release(self, entry: Entry) -> None:
        self._bytes -= entry.nbytes
        if self._bytes < 0:  # defensive: the count must never go negative
            self._bytes = 0
        entry.dataset = None
        entry.nbytes = 0

    def _expire_locked(self) -> None:
        if self.ttl_seconds <= 0:
            return
        cutoff = self._clock() - self.ttl_seconds
        for session_id, entry in list(self._entries.items()):
            if entry.last_used <= cutoff:
                had_data = entry.dataset is not None
                self._entries.pop(session_id, None)
                self._release(entry)
                if had_data:
                    self._remember_dropped(session_id)

    def _evict_locked(self, protect: Optional[str] = None) -> None:
        while self._entries and (
            len(self._entries) > self.max_sessions or self._bytes > self.max_bytes
        ):
            victim = next(
                (key for key in self._entries if key != protect),
                None,
            )
            if victim is None:
                return
            entry = self._entries.pop(victim)
            had_data = entry.dataset is not None
            self._release(entry)
            if had_data:
                self._remember_dropped(victim)

    # -- reading ----------------------------------------------------------

    def get(self, session_id: Any) -> Optional[Entry]:
        """The visitor's entry, or None. Expired entries are dropped first."""
        if not self._usable(session_id):
            return None
        with self._lock:
            self._expire_locked()
            entry = self._entries.get(session_id)
            if entry is None:
                return None
            entry.last_used = self._clock()
            self._entries.move_to_end(session_id)
            return entry

    def dataset(self, session_id: Any) -> Any:
        """The visitor's dataset, or None if they have none (or it is gone)."""
        entry = self.get(session_id)
        return entry.dataset if entry is not None else None

    def statuses(self, session_id: Any) -> Dict[str, str]:
        """A copy of the visitor's own Verify/Dismiss stamps."""
        entry = self.get(session_id)
        return dict(entry.statuses) if entry is not None else {}

    def was_dropped(self, session_id: Any) -> bool:
        """True if this visitor once held a dataset that has since been dropped."""
        if not self._usable(session_id):
            return False
        with self._lock:
            return session_id in self._dropped

    def stats(self) -> Dict[str, int]:
        """Counters, for diagnostics and tests."""
        with self._lock:
            return {
                "sessions": len(self._entries),
                "with_dataset": sum(1 for e in self._entries.values() if e.dataset is not None),
                "bytes": self._bytes,
                "dropped_remembered": len(self._dropped),
                "max_sessions": self.max_sessions,
                "max_bytes": self.max_bytes,
                "max_entry_bytes": self.max_entry_bytes,
            }

    # -- writing ----------------------------------------------------------

    def put_dataset(self, session_id: Any, dataset: Any, nbytes: int) -> bool:
        """Hold ``dataset`` for this visitor. False if it could not be kept.

        A False return is not an error: the caller has the dataset in hand and
        can render it once. It means the store refused to retain it, either
        because the id is unusable or because it is over the per-entry cap.
        """
        if not self._usable(session_id) or dataset is None:
            return False
        try:
            nbytes = max(0, int(nbytes))
        except (TypeError, ValueError):
            nbytes = 0
        with self._lock:
            self._expire_locked()
            entry = self._entries.pop(session_id, None) or Entry(self._clock())
            self._release(entry)  # a new upload replaces the old one's bytes
            entry.last_used = self._clock()
            self._entries[session_id] = entry
            self._dropped.pop(session_id, None)
            if nbytes > self.max_entry_bytes:
                # More than one visitor's share. Keep the (data-less) entry so
                # their stamps survive, and evict nobody on its behalf.
                self._evict_locked(protect=session_id)
                return False
            entry.dataset = dataset
            entry.nbytes = nbytes
            self._bytes += nbytes
            self._evict_locked(protect=session_id)
            return self._entries.get(session_id) is entry and entry.dataset is not None

    def clear_dataset(self, session_id: Any) -> None:
        """Forget this visitor's dataset but keep their stamps.

        Used by "Back to the sample data": the visitor asked for this, so the
        id is *not* remembered as dropped and they get no "your file is gone"
        message.
        """
        if not self._usable(session_id):
            return
        with self._lock:
            entry = self._entries.get(session_id)
            if entry is None:
                return
            self._release(entry)
            entry.last_used = self._clock()
            self._entries.move_to_end(session_id)
            self._dropped.pop(session_id, None)

    def set_status(self, session_id: Any, key: str, value: str) -> bool:
        """Record one Verify/Dismiss stamp for this visitor only."""
        if not self._usable(session_id) or not isinstance(key, str):
            return False
        with self._lock:
            self._expire_locked()
            entry = self._entries.get(session_id)
            if entry is None:
                entry = Entry(self._clock())
                self._entries[session_id] = entry
            if key not in entry.statuses and len(entry.statuses) >= MAX_STATUSES_PER_SESSION:
                return False
            entry.statuses[key] = value
            entry.last_used = self._clock()
            self._entries.move_to_end(session_id)
            self._evict_locked(protect=session_id)
            return session_id in self._entries

    def drop(self, session_id: Any) -> None:
        """Forget a visitor completely, dataset and stamps."""
        if not self._usable(session_id):
            return
        with self._lock:
            entry = self._entries.pop(session_id, None)
            if entry is not None:
                self._release(entry)

    def clear(self) -> None:
        """Forget everything, including which ids were dropped. For tests."""
        with self._lock:
            self._entries.clear()
            self._dropped.clear()
            self._bytes = 0
