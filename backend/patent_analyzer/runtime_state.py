"""Cross-instance runtime state: circuit breakers, quotas and rate gates.

Everything that has to survive a request lives in `cloud_state` — one JSON
object per key in GCS, written with a generation precondition. It used to live
in `cache.kv()`, which on Cloud Run is a SQLite file inside the container: the
service runs up to three instances, so every counter here was three counters,
each certain it was the only one, and a deploy reset all three (Harry,
2026-09-20: "不能有 local counter，必须云端").

The one thing that is NOT shared is the pacing of a rate gate, and that is a
deliberate, stated compromise rather than an oversight — see MinuteGate.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from . import cloud_state

_NS = "runtime"


def instance_share() -> int:
    """How many instances may be running, so a per-instance rate can be set to
    the share of the limit this one is entitled to.

    A rate gate is not a counter: it decides whether to sleep for a fraction of
    a second, and asking GCS that question on every request would cost more
    latency than the gate saves. Dividing the allowance instead is conservative
    — with fewer instances live we simply go slower than we are allowed — and
    it is correct in the direction that matters: the provider never sees more
    than its limit.

    MAX_INSTANCES mirrors the service's autoscaling.knative.dev/maxScale (3 at
    the time of writing). Locally there is one process and the default is 1.
    """
    try:
        n = int(os.environ.get("MAX_INSTANCES", "3" if os.environ.get("K_SERVICE") else "1"))
    except ValueError:
        n = 1
    return max(1, n)


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
            doc, _, _ = cloud_state.read(f"breaker:{self.name}")
            self._local_until = float(doc.get("blocked_until", 0))
            self._local_checked = now
        return now < self._local_until

    def trip(self, reason: str = ""):
        until = time.time() + self.cooldown_s
        self._local_until, self._local_checked = until, time.time()

        def _f(doc):
            doc["blocked_until"] = max(float(doc.get("blocked_until", 0)), until)
            doc["reason"] = reason[:200]
            doc["trips"] = int(doc.get("trips", 0)) + 1
            doc["last_trip"] = time.time()
        cloud_state.update(f"breaker:{self.name}", _f)


def month_key(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def week_key(now: datetime | None = None) -> str:
    """ISO week, for quotas the provider counts per week (USPTO ODP)."""
    y, w, _ = (now or datetime.now(timezone.utc)).isocalendar()
    return f"{y}-W{w:02d}"


def day_key(now: datetime | None = None) -> str:
    """UTC calendar day, for quotas a provider states per day (Unpaywall)."""
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d")


class MonthlyQuota:
    """Per-(name, month) counter, e.g. SerpAPI calls per key."""

    def __init__(self, name: str, cap: int):
        self.name, self.cap = name, cap

    def _key(self) -> str:
        return f"quota:{self.name}:{month_key()}"

    def used(self) -> int:
        return int(cloud_state.get(self._key()) or 0)

    def remaining(self) -> int:
        return max(0, self.cap - self.used())

    def take(self, n: int = 1) -> bool:
        """Reserve n units; False (and no increment) if it would exceed cap.

        The test and the increment are one conditional write, so two instances
        cannot both see room and both take the last slot. Read-then-increment
        is where an over-spend comes from.
        """
        taken = {"ok": False}

        def _f(doc):
            cur = int(doc.get("n", 0))
            if cur + n > self.cap:
                taken["ok"] = False
                return
            doc["n"] = cur + n
            taken["ok"] = True
        cloud_state.update(self._key(), _f)
        return taken["ok"]

    def release(self, n: int = 1):
        cloud_state.incr(self._key(), by=-n)

    def set_used(self, n: int):
        """Overwrite the counter with the provider's own figure."""
        cloud_state.update(self._key(), lambda d: d.__setitem__("n", int(n)))

    def exhaust(self):
        """Mark the whole period as spent (provider said 401/429)."""
        cloud_state.update(self._key(), lambda d: d.__setitem__("n", int(self.cap)))


class PeriodQuota(MonthlyQuota):
    """MonthlyQuota with a caller-chosen period key (e.g. week_key for USPTO ODP)."""

    def __init__(self, name: str, cap: int, period=week_key):
        super().__init__(name, cap)
        self.period = period

    def _key(self) -> str:
        return f"quota:{self.name}:{self.period()}"


class MinuteGate:
    """Request smoothing: at most `per_minute / instance_share()` takes per
    wall-clock minute for `name`, counted in this process.

    Vertex DSQ docs: "Avoid sending requests in sharp, second-level spikes ...
    Distributing your API calls more evenly helps the system manage your load
    predictably." Callers `await gate.wait()` before each request.

    This is the one piece of state here that is NOT shared, and the reason is
    in `instance_share`: it is a rate, not an allowance, and a gate that asks
    GCS before every request costs more than it saves. Running fewer instances
    than MAX_INSTANCES only makes us slower than we are entitled to be."""

    def __init__(self, name: str, per_minute: int):
        self.name = name
        # This instance's share of the allowance. The count itself stays in
        # process memory on purpose: a gate answers "sleep or go" in the
        # milliseconds before a request, and a GCS round trip per request would
        # cost more than the gate saves. Dividing the allowance keeps the
        # aggregate under the provider's limit without asking anyone, and errs
        # towards going slower than permitted rather than faster.
        self.per_minute = max(1, int(per_minute // instance_share())) if per_minute > 0 else 0
        self.allowance = per_minute
        self._counts: dict[str, int] = {}

    def _key(self, now: datetime | None = None) -> str:
        now = now or datetime.now(timezone.utc)
        return f"gate:{self.name}:{now.strftime('%Y%m%d%H%M')}"

    def take(self) -> float:
        """Reserve one slot now; returns seconds to sleep first (0 = go)."""
        if self.per_minute <= 0:
            return 0.0
        now = datetime.now(timezone.utc)
        k = self._key(now)
        if len(self._counts) > 4:                      # keep only the live minutes
            for old in sorted(self._counts)[:-2]:
                self._counts.pop(old, None)
        n = self._counts.get(k, 0) + 1
        if n <= self.per_minute:
            self._counts[k] = n
            return 0.0
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
