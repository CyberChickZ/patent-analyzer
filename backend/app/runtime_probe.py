"""Does this container actually get CPU when no request is in flight?

Cloud Run throttles CPU outside a request unless the service is deployed with
`--no-cpu-throttling`. This service runs its pipeline as a background task, so
with throttling on the pipeline gets CPU only while some request happens to be
being handled. Job ea70d51a: the extract gate finished at 18:51:02Z and the
search node logged nothing for the next seven minutes. The same service took
68 minutes on a job that takes 25 locally.

Nothing reported it. The run was not failing — it was being paused, for free,
by a deployment flag, and every timing number the job produced was a number
about the flag rather than about the work.

So the container asks, once, at startup, and `/health` carries the answer.
Three states and no fourth:

  off      cpuIdle is false — CPU is always allocated, background work runs
  on       cpuIdle is true — THIS IS THE PROBLEM, and /health says so
  unknown  the Admin API would not answer; the reason is carried with it
  n/a      not on Cloud Run

`unknown` is deliberately not `off`. A probe that cannot see the answer and
reports the good one is worse than no probe.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

TIMEOUT = 5.0
_cache: dict | None = None

METADATA_TOKEN = ("http://metadata.google.internal/computeMetadata/v1/instance/"
                  "service-accounts/default/token")
METADATA_REGION = "http://metadata.google.internal/computeMetadata/v1/instance/region"


def _meta(url: str) -> str:
    req = urllib.request.Request(url, headers={"Metadata-Flavor": "Google"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read().decode()


def _token() -> str:
    return json.loads(_meta(METADATA_TOKEN)).get("access_token", "")


def _region() -> str:
    # ".../regions/us-west1"
    return _meta(METADATA_REGION).rsplit("/", 1)[-1]


def probe(force: bool = False) -> dict:
    """{"state": ..., "detail": ...}. Cached: it cannot change without a new
    revision, and a new revision is a new container."""
    global _cache
    if _cache is not None and not force:
        return _cache
    _cache = _probe()
    return _cache


def _probe() -> dict:
    service = os.environ.get("K_SERVICE", "")
    revision = os.environ.get("K_REVISION", "")
    project = os.environ.get("GC_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    if not service:
        return {"state": "n/a", "detail": "not running on Cloud Run"}
    if not (revision and project):
        return {"state": "unknown",
                "detail": f"K_REVISION={revision!r} GC_PROJECT={project!r}; cannot name the revision"}
    try:
        region = _region()
        url = (f"https://run.googleapis.com/v2/projects/{project}/locations/{region}"
               f"/services/{service}/revisions/{revision}")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            doc = json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        return {"state": "unknown",
                "detail": f"Cloud Run Admin API said HTTP {exc.code}; the service account "
                          f"probably lacks run.viewer on this service"}
    except Exception as exc:
        return {"state": "unknown", "detail": f"{type(exc).__name__}: {exc}"[:200]}

    containers = doc.get("containers") or []
    if not containers:
        return {"state": "unknown",
                "detail": "the API did not return a revision with containers; shape changed"}
    res = containers[0].get("resources") or {}
    # proto3 JSON omits a false boolean, so an ABSENT cpuIdle means CPU is always
    # allocated — the good case. Reading absence as "unknown" would have made the
    # fixed deployment (00087, which has no cpuIdle) report a problem it does not
    # have. `containers` is what tells us we got a revision at all.
    idle = bool(res.get("cpuIdle", False))
    return {"state": "on" if idle else "off",
            "detail": ("CPU is throttled outside a request: the background pipeline only runs "
                       "while some request is being handled. Deploy with --no-cpu-throttling."
                       if idle else "CPU is always allocated; background work runs")}


def warn_line(p: dict | None = None) -> str:
    """The one line worth putting in the log at startup, or "" when there is
    nothing wrong. Only `on` is wrong; `unknown` is reported but not shouted,
    because a probe that cries wolf gets ignored and then the real one is too."""
    p = p or probe()
    if p.get("state") == "on":
        return ("CPU THROTTLING IS ON: background pipeline work only gets CPU while a request "
                "is in flight, so jobs stall between requests and every duration this service "
                "reports is wrong. Redeploy with --no-cpu-throttling.")
    return ""
