import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import job_title, short_title


def test_a_title_is_a_name_not_a_paragraph():
    assert short_title("A method of decentralized swarm coordination for a plurality of agents in a field") == \
        "A method of decentralized swarm coordination for a…"
    assert short_title("Short one") == "Short one"
    assert short_title("   ") == "" and short_title(None) == ""


def test_old_jobs_fall_back_to_the_file_name_then_the_id():
    assert job_title({"id": "abc12345", "title": "Swarm coordination", "filename": "x.pdf"}) == "Swarm coordination"
    assert job_title({"id": "abc12345", "filename": "decentralized_swarm.pdf"}) == "decentralized_swarm.pdf"
    assert job_title({"id": "abc12345", "filename": ""}) == "abc12345"


def test_the_list_carries_the_title_so_the_page_does_not_show_a_hash():
    """A job id is an address, not a name. The list endpoint has to hand the
    page something a person can scan (Harry, 2026-09-20)."""
    import inspect

    from app import main as m
    src = inspect.getsource(m.list_jobs)
    assert '"title": job_title(j)' in src
