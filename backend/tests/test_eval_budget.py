import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "evals"))

from evals.common import Budget, BudgetExceeded, add_budget_arg


def test_a_run_that_cannot_fit_is_refused_before_it_spends_anything():
    """The estimate uses the last MEASURED per-item cost, so this is an answer
    and not a guess: 8 papers at $2.81 is $22.48, and a $10 budget is a no."""
    b = Budget(10.0, 8, 2.81, "E4")
    with pytest.raises(BudgetExceeded) as e:
        b.start(out=lambda *a: None)
    assert "$22.48" in str(e.value) and "$10.00" in str(e.value)
    assert "cut the item count to 3" in str(e.value), "a refusal should say what would fit"
    assert b.spent == 0.0 and b.done == 0


def test_a_run_that_fits_starts():
    b = Budget(30.0, 8, 2.81, "E4")
    b.start(out=lambda *a: None)
    assert b.estimate == 22.48


def test_going_over_stops_where_it_stands_and_says_where():
    said = []
    b = Budget(10.0, 8, 1.0, "E4")
    b.start(out=said.append)
    for _ in range(8):
        try:
            b.charge(4.5, out=said.append)
        except BudgetExceeded:
            break
    assert b.done == 3 and b.spent == 13.5
    stop = [s for s in said if "STOPPED" in s]
    assert stop and "item 3 of 8" in stop[0] and "5 items were NOT run" in stop[0]


def test_observe_takes_a_total_so_concurrent_items_cannot_double_count():
    b = Budget(10.0, 4, 1.0)
    b.start(out=lambda *a: None)
    b.observe(3.0, out=lambda *a: None)
    b.observe(3.0, out=lambda *a: None)      # the meter has not moved
    assert b.spent == 3.0 and b.done == 2


def test_the_flag_is_required_on_anything_that_calls_a_model():
    import argparse
    ap = argparse.ArgumentParser()
    add_budget_arg(ap)
    with pytest.raises(SystemExit):
        ap.parse_args([])
    assert ap.parse_args(["--budget-usd", "12.5"]).budget_usd == 12.5


def test_the_ceiling_lives_under_every_model_call(monkeypatch):
    """Wiring a budget into each script's loop guards only the loops somebody
    remembered to wire, and the calls that cost money are several frames deep
    inside the pipeline. The check sits in the one place both call paths pass
    through."""
    import asyncio

    from app import llm
    from patent_analyzer import metering

    monkeypatch.setenv("EVAL_BUDGET_USD", "5")
    monkeypatch.setattr(metering, "totals", lambda *a, **k: {"cost_usd": 4.99})
    asyncio.run(llm._smooth("gemini-3.8-flash"))          # under: proceeds

    monkeypatch.setattr(metering, "totals", lambda *a, **k: {"cost_usd": 5.01})
    with pytest.raises(llm.EvalBudgetExceeded) as e:
        asyncio.run(llm._smooth("gemini-3.8-flash"))
    assert "$5.01" in str(e.value) and "treat the output as partial" in str(e.value)


def test_the_server_is_not_affected_when_the_variable_is_unset(monkeypatch):
    """The deployment has its own ceiling, and it is a different rule: it
    refuses new jobs and never interrupts a running one."""
    import asyncio

    from app import llm
    from patent_analyzer import metering

    monkeypatch.delenv("EVAL_BUDGET_USD", raising=False)
    monkeypatch.setattr(metering, "totals", lambda *a, **k: {"cost_usd": 10_000.0})
    asyncio.run(llm._smooth("gemini-3.8-flash"))


def test_every_script_that_calls_a_model_takes_the_flag():
    """A list, so a new eval script that spends money and forgets the flag
    fails here instead of on the bill."""
    import re
    from pathlib import Path
    evals = Path(__file__).parent.parent / "evals"
    missing = []
    for p in sorted(evals.glob("*.py")):
        src = p.read_text()
        spends = any(t in src for t in ("app.llm", "call_llm", "evaluate_batch", "from nodes"))
        if not spends or p.name in ("common.py", "llm_cache.py"):
            continue
        if not re.search(r"parse_args\(\)", src):
            continue          # no CLI to hang a flag on
        if "add_budget_arg" not in src:
            missing.append(p.name)
    assert not missing, f"these call a model and take no --budget-usd: {missing}"
