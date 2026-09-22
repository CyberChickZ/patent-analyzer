"""A deployment can be broken while looking healthy."""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import runtime_probe as rp


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(rp, "_cache", None)
    for k in ("K_SERVICE", "K_REVISION", "GC_PROJECT", "GOOGLE_CLOUD_PROJECT"):
        monkeypatch.delenv(k, raising=False)


def test_off_cloud_run_it_says_so_and_asks_nobody():
    def boom(*a, **k):
        raise AssertionError("must not touch the metadata server")
    assert rp.probe()["state"] == "n/a"


def test_cpu_idle_true_is_the_problem_and_health_says_degraded(monkeypatch):
    monkeypatch.setattr(rp, "_probe", lambda: {"state": "on", "detail": "throttled"})
    monkeypatch.setattr(rp, "_cache", None)
    assert "CPU THROTTLING IS ON" in rp.warn_line()
    import app.main as m
    monkeypatch.setattr(rp, "_cache", {"state": "on", "detail": "throttled"})
    body = TestClient(m.app).get("/health").json()
    assert body["status"] == "degraded"
    assert body["cpu_throttling"]["state"] == "on"
    assert "no-cpu-throttling" in body["warning"]


def test_cpu_always_allocated_is_quiet(monkeypatch):
    monkeypatch.setattr(rp, "_cache", {"state": "off", "detail": "always allocated"})
    import app.main as m
    body = TestClient(m.app).get("/health").json()
    assert body["status"] == "ok" and "warning" not in body
    assert body["cpu_throttling"]["state"] == "off"


def test_unknown_is_never_reported_as_fine(monkeypatch):
    """A probe that cannot see the answer and reports the good one is worse
    than no probe."""
    monkeypatch.setattr(rp, "_cache", {"state": "unknown", "detail": "HTTP 403"})
    assert rp.warn_line() == "", "unknown is reported, not shouted"
    import app.main as m
    body = TestClient(m.app).get("/health").json()
    assert body["cpu_throttling"] == {"state": "unknown", "detail": "HTTP 403"}
    assert body["status"] == "ok", "we do not know it is broken; we do know we cannot tell"


def test_a_revision_we_cannot_name_is_unknown_not_off(monkeypatch):
    monkeypatch.setenv("K_SERVICE", "patent-analyzer")
    out = rp._probe()
    assert out["state"] == "unknown" and "K_REVISION" in out["detail"]


def test_it_reads_the_field_that_is_true_not_the_one_that_reads_well(monkeypatch):
    """Measured on patent-analyzer-00086-jvm, 2026-09-22, and the two views of
    the same service disagree:

        gcloud run services describe ... spec.template.metadata.annotations
            run.googleapis.com/cpu-throttling=false      <- reads like "fine"
        run.googleapis.com/v2 ... /revisions/<rev>
            "resources": {"cpuIdle": true}               <- actually throttled

    Anyone checking the annotation concludes the deployment is fine. The probe
    reads the v2 revision's cpuIdle, which is the field the runtime obeys.
    """
    real = {"containers": [{"resources": {"limits": {"cpu": "2", "memory": "4Gi"},
                                          "cpuIdle": True, "startupCpuBoost": True}}],
            "maxInstanceRequestConcurrency": 1,
            "scaling": {"minInstanceCount": 1, "maxInstanceCount": 3}}
    monkeypatch.setenv("K_SERVICE", "patent-analyzer")
    monkeypatch.setenv("K_REVISION", "patent-analyzer-00086-jvm")
    monkeypatch.setenv("GC_PROJECT", "aime-hello-world")
    monkeypatch.setattr(rp, "_region", lambda: "us-west1")
    monkeypatch.setattr(rp, "_token", lambda: "t")

    class _R:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            import json
            return json.dumps(real).encode()
    monkeypatch.setattr(rp.urllib.request, "urlopen", lambda *a, **k: _R())
    out = rp._probe()
    assert out["state"] == "on" and "no-cpu-throttling" in out["detail"]

    real["containers"][0]["resources"]["cpuIdle"] = False
    assert rp._probe()["state"] == "off"

    del real["containers"][0]["resources"]["cpuIdle"]
    out = rp._probe()
    assert out["state"] == "unknown" and "API shape changed" in out["detail"]
