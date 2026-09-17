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
