import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import prompts


@pytest.fixture(autouse=True)
def _tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(prompts, "PROMPT_DIR", tmp_path)
    monkeypatch.setattr(prompts, "_store", None)
    prompts._cache.clear()
    prompts.register_default("t.hello", "Hello {name}, braces stay {{literal}}")
    prompts.set_overrides(None)
    yield


def test_default_is_version_zero_and_renders():
    text, v = prompts.get("t.hello")
    assert v == 0 and prompts.render("t.hello", name="Ann") == "Hello Ann, braces stay {literal}"
    assert prompts.used_versions() == {"t.hello": 0}


def test_put_creates_versions_and_current_moves():
    assert prompts.put("t.hello", "Hi {name}", by="harry") == 1
    assert prompts.put("t.hello", "Yo {name}", by="harry", make_current=False) == 2
    text, v = prompts.get("t.hello")
    assert (text, v) == ("Hi {name}", 1)
    prompts.set_current("t.hello", 2)
    assert prompts.get("t.hello") == ("Yo {name}", 2)
    prompts.set_current("t.hello", 0)
    assert prompts.get("t.hello")[1] == 0
    assert [x["v"] for x in prompts.describe("t.hello")["versions"]] == [1, 2]


def test_job_override_pins_version_or_inline_text():
    prompts.put("t.hello", "Hi {name}")
    prompts.set_overrides({"t.hello": 0})
    assert prompts.get("t.hello")[1] == 0
    prompts.set_overrides({"t.hello": "Inline {name}!"})
    assert prompts.render("t.hello", name="B") == "Inline B!" and prompts.used_versions() == {"t.hello": "inline"}


def test_unknown_name_rejected():
    with pytest.raises(KeyError):
        prompts.get("nope")
