"""Multi-channel prior art recall.

Each channel module exposes async functions returning
(list[Candidate], error: str | None) tuples. The pool layer merges and
deduplicates results from all channels into a single ranked candidate set.
"""

from .pool import Candidate, pool_and_dedupe

__all__ = ["Candidate", "pool_and_dedupe"]
