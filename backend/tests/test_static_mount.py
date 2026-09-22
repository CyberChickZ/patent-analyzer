"""The app mounts app/static at import. git cannot store an empty directory, so
a checkout that has never had one must still be able to start — revision
patent-analyzer-00080-9c4 (2026-09-19) died on exactly this."""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BACKEND = Path(__file__).parent.parent


def test_importing_the_app_creates_the_static_dir_if_a_checkout_lacks_it():
    static = BACKEND / "app" / "static"
    had = static.exists()
    contents = sorted(p.name for p in static.iterdir()) if had else []
    assert contents == [], "static/ now holds files — this test must stop deleting it"
    if had:
        static.rmdir()                       # a fresh checkout looks like this
    try:
        assert not static.exists()
        out = subprocess.run(
            [sys.executable, "-c", "import sys; sys.path.insert(0, '.'); import app.main"],
            cwd=BACKEND, capture_output=True, text=True, timeout=180)
        assert out.returncode == 0, out.stderr[-1500:]
        assert static.is_dir()
    finally:
        static.mkdir(parents=True, exist_ok=True)


def test_static_is_empty_and_untracked_which_is_why_the_mkdir_is_needed():
    """If someone commits a file under static/, the mkdir becomes belt-and-braces
    rather than load-bearing — and this test should be revisited, not deleted."""
    if shutil.which("git") is None:
        return
    tracked = subprocess.run(["git", "ls-files", "app/static"], cwd=BACKEND,
                             capture_output=True, text=True).stdout.strip()
    assert tracked == "", f"static/ is tracked now ({tracked!r}); re-read the comment in main.py"
