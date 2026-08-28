"""Cross-instance runtime state on the KV store: circuit breakers and
monthly quotas. Cloud Run runs many instances; a block seen by one must
stop the others, and SerpAPI's free tier (250/key/month) has to be
counted in one place.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .cache import kv

_NS = "runtime"


class Breaker:
    """Shared circuit breaker with a short local cache to avoid a KV read
    on every request."""

    def __init__(self, name: str, cooldown_s: float = 15 * 60, local_ttl_s: float = 30):
        self.name = name
        self.cooldown_s = cooldown_s
        self.local_ttl_s = local_ttl_s
        self._local_until = 0.0
        self._local_checked = 0.0

    def is_open(self) -> bool:
        now = time.time()
        if now - self._local_checked > self.local_ttl_s:
            doc = kv().get(_NS, f"breaker:{self.name}") or {}
            self._local_until = float(doc.get("blocked_until", 0))
            self._local_checked = now
        return now < self._local_until

    def trip(self, reason: str = ""):
        until = time.time() + self.cooldown_s
        self._local_until, self._local_checked = until, time.time()
        doc = kv().get(_NS, f"breaker:{self.name}") or {}
        kv().put(_NS, f"breaker:{self.name}", {
            "blocked_until": until, "reason": reason[:200],
            "trips": int(doc.get("trips", 0)) + 1, "last_trip": time.time()})


def month_key(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def week_key(now: datetime | None = None) -> str:
    """ISO week, for quotas the provider counts per week (USPTO ODP)."""
    y, w, _ = (now or datetime.now(timezone.utc)).isocalendar()
    return f"{y}-W{w:02d}"


class MonthlyQuota:
    """Per-(name, month) counter, e.g. SerpAPI calls per key."""

    def __init__(self, name: str, cap: int):
        self.name, self.cap = name, cap

    def _key(self) -> str:
        return f"quota:{self.name}:{month_key()}"

    def used(self) -> int:
        return int((kv().get(_NS, self._key()) or {}).get("n", 0))

    def remaining(self) -> int:
        return max(0, self.cap - self.used())

    def take(self, n: int = 1) -> bool:
        """Reserve n units; False (and no increment) if it would exceed cap."""
        if self.used() + n > self.cap:
            return False
        kv().incr(_NS, self._key(), by=n)
        return True

    def release(self, n: int = 1):
        kv().incr(_NS, self._key(), by=-n)

    def set_used(self, n: int):
        """Overwrite the counter with the provider's own figure."""
        kv().put(_NS, self._key(), {"n": int(n)})

    def exhaust(self):
        """Mark the whole month as spent (provider said 401/429)."""
        gap = self.cap - self.used()
        if gap > 0:
            kv().incr(_NS, self._key(), by=gap)


class PeriodQuota(MonthlyQuota):
    """MonthlyQuota with a caller-chosen period key (e.g. week_key for USPTO ODP)."""

    def __init__(self, name: str, cap: int, period=week_key):
        super().__init__(name, cap)
        self.period = period

    def _key(self) -> str:
        return f"quota:{self.name}:{self.period()}"


class MinuteGate:
    """Cross-process request smoothing: at most `per_minute` takes per wall-clock
    minute for `name`, counted in the shared KV (sqlite locally, Firestore on
    Cloud Run). Vertex DSQ docs: "Avoid sending requests in sharp, second-level
    spikes ... Distributing your API calls more evenly helps the system manage
    your load predictably." Callers `await gate.wait()` before each request."""

    def __init__(self, name: str, per_minute: int):
        self.name, self.per_minute = name, per_minute

    def _key(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        return f"gate:{self.name}:{now.strftime('%Y%m%d%H%M')}"

    def take(self) -> float:
        """Reserve one slot now; returns seconds to sleep first (0 = go)."""
        if self.per_minute <= 0:
            return 0.0
        now = datetime.now(timezone.utc)
        n = kv().incr(_NS, self._key(now))
        if n <= self.per_minute:
            return 0.0
        kv().incr(_NS, self._key(now), by=-1)
        return 60.0 - now.second - now.microsecond / 1e6 + 0.05

    async def wait(self) -> float:
        import asyncio
        waited = 0.0
        while True:
            s = self.take()
            if s <= 0:
                return waited
            await asyncio.sleep(s)
            waited += s


class SerialLock:
    """Cross-process mutex + cooldown for a source whose limit is "1 request
    per second across all endpoints" (Semantic Scholar API key terms): a
    request holds the lock from send to response; after release no process
    may take it again for `cooldown_s`. flock on a file in LOCK_DIR, the
    release time kept next to it. Use: `async with SerialLock("s2", 1.0):`."""

    def __init__(self, name: str, cooldown_s: float = 1.0):
        import os
        from pathlib import Path
        self.name, self.cooldown_s = name, cooldown_s
        d = Path(os.environ.get("LOCK_DIR", "/tmp/amie_locks"))
        d.mkdir(parents=True, exist_ok=True)
        self._lock_path, self._ts_path = d / f"{name}.lock", d / f"{name}.last"
        self._fh = None

    def _acquire_blocking(self) -> float:
        import fcntl
        import time
        self._fh = open(self._lock_path, "a+")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        try:
            last = float(self._ts_path.read_text() or 0)
        except Exception:
            last = 0.0
        wait = self.cooldown_s - (time.time() - last)
        if wait > 0:
            time.sleep(wait)
        return max(wait, 0.0)

    def _release_blocking(self) -> None:
        import fcntl
        import time
        try:
            self._ts_path.write_text(repr(time.time()))
        finally:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None

    async def __aenter__(self):
        import asyncio
        self.waited = await asyncio.to_thread(self._acquire_blocking)
        return self

    async def __aexit__(self, *exc):
        import asyncio
        await asyncio.to_thread(self._release_blocking)
        return False
