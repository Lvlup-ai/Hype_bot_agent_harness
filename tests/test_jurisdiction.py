"""Write jurisdiction: an agent may write only where its role allows.

Every test writes to a temporary run tree, runs something in the guard, and
checks what survived. Nothing here depends on an LLM.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_harness.jurisdiction import (
    Guard,
    JurisdictionError,
    Matrix,
    capture_snapshot,
    enforce_from_snapshot,
    is_writable,
)

MATRIX = Matrix(roles={
    "proposer": ("items/{item}/proposal.md", "items/{item}/code.py", "scratch/*"),
    "auditor": ("journal.md", "items/{item}/audit.md"),
})


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    root = tmp_path / "run"
    (root / "items" / "b_001").mkdir(parents=True)
    (root / "journal.md").write_text("day 1\n")
    (root / "state.json").write_text('{"phase": "B"}')
    (root / "items" / "b_001" / "proposal.md").write_text("old\n")
    return root


# ── Matching ─────────────────────────────────────────────────────────────────

def test_star_crosses_directories() -> None:
    assert is_writable("scratch/deep/er/file.txt", ("scratch/*",))


def test_slots_are_filled_from_context() -> None:
    globs = MATRIX.globs("proposer", {"item": "b_001"})
    assert "items/b_001/proposal.md" in globs
    assert not is_writable("items/b_002/proposal.md", globs)


def test_unknown_role_may_write_nothing() -> None:
    assert MATRIX.globs("typo", {"item": "b_001"}) == ()


def test_a_missing_slot_never_matches() -> None:
    globs = MATRIX.globs("proposer")
    assert not is_writable("items/b_001/proposal.md", globs)


# ── In-process guard ─────────────────────────────────────────────────────────

def test_writing_inside_jurisdiction_is_fine(run_root: Path) -> None:
    with Guard(run_root, MATRIX, "proposer", {"item": "b_001"}) as g:
        (run_root / "items" / "b_001" / "proposal.md").write_text("new\n")
        (run_root / "items" / "b_001" / "code.py").write_text("x = 1\n")
    assert g.result is not None and g.result.ok
    assert (run_root / "items" / "b_001" / "proposal.md").read_text() == "new\n"


def test_modified_file_outside_jurisdiction_is_restored(run_root: Path) -> None:
    with Guard(run_root, MATRIX, "proposer", {"item": "b_001"}) as g:
        (run_root / "state.json").write_text('{"phase": "HACKED"}')
    assert not g.result.ok
    assert g.result.violations == ("state.json",)
    assert g.result.restored == ("state.json",)
    assert (run_root / "state.json").read_text() == '{"phase": "B"}'


def test_created_file_outside_jurisdiction_is_removed(run_root: Path) -> None:
    with Guard(run_root, MATRIX, "proposer", {"item": "b_001"}) as g:
        (run_root / "items" / "b_002").mkdir()
        (run_root / "items" / "b_002" / "proposal.md").write_text("sneaky\n")
    assert g.result.violations == ("items/b_002/proposal.md",)
    assert not (run_root / "items" / "b_002" / "proposal.md").exists()


def test_deleted_file_outside_jurisdiction_is_recreated(run_root: Path) -> None:
    with Guard(run_root, MATRIX, "auditor", {"item": "b_001"}) as g:
        (run_root / "items" / "b_001" / "proposal.md").unlink()
    assert g.result.violations == ("items/b_001/proposal.md",)
    assert (run_root / "items" / "b_001" / "proposal.md").read_text() == "old\n"


def test_the_auditor_cannot_touch_the_proposal(run_root: Path) -> None:
    with Guard(run_root, MATRIX, "auditor", {"item": "b_001"}) as g:
        (run_root / "journal.md").write_text("day 2\n")                        # allowed
        (run_root / "items" / "b_001" / "proposal.md").write_text("edited\n")  # not
    assert g.result.violations == ("items/b_001/proposal.md",)
    assert (run_root / "journal.md").read_text() == "day 2\n"
    assert (run_root / "items" / "b_001" / "proposal.md").read_text() == "old\n"


def test_protected_paths_are_never_writable(tmp_path: Path, run_root: Path) -> None:
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "core.py").write_text("def f(): return 1\n")
    matrix = Matrix(roles=MATRIX.roles, protected=(harness,))
    with Guard(run_root, matrix, "proposer", {"item": "b_001"}) as g:
        (harness / "core.py").write_text("def f(): return 2\n")
        (harness / "evil.py").write_text("import os\n")
    assert g.result.violations == ("harness/core.py", "harness/evil.py")
    assert (harness / "core.py").read_text() == "def f(): return 1\n"
    assert not (harness / "evil.py").exists()


def test_a_protected_path_that_does_not_exist_is_ignored(tmp_path: Path, run_root: Path) -> None:
    matrix = Matrix(roles=MATRIX.roles, protected=(tmp_path / "absent",))
    with Guard(run_root, matrix, "proposer", {"item": "b_001"}) as g:
        pass
    assert g.result.ok


def test_the_agents_exception_is_not_swallowed(run_root: Path) -> None:
    with pytest.raises(RuntimeError, match="agent crashed"):
        with Guard(run_root, MATRIX, "proposer", {"item": "b_001"}) as g:
            (run_root / "state.json").write_text("partial")
            raise RuntimeError("agent crashed")
    # ...and the restoration still happened on the way out.
    assert (run_root / "state.json").read_text() == '{"phase": "B"}'
    assert g.result is not None and not g.result.ok


def test_raise_on_violation(run_root: Path) -> None:
    with Guard(run_root, MATRIX, "proposer", {"item": "b_001"}) as g:
        (run_root / "journal.md").write_text("tampered\n")
    with pytest.raises(JurisdictionError, match="journal.md"):
        g.raise_on_violation()


def test_no_change_at_all_is_ok(run_root: Path) -> None:
    with Guard(run_root, MATRIX, "proposer", {"item": "b_001"}) as g:
        pass
    assert g.result == g.result.__class__(ok=True)


# ── Cross-process variant ────────────────────────────────────────────────────

def test_disk_snapshot_lives_outside_the_run_root(run_root: Path) -> None:
    snap = capture_snapshot(run_root)
    try:
        assert run_root.resolve() not in snap.resolve().parents
        manifest = json.loads((snap / "manifest.json").read_text())
        assert len(manifest["files"]) == 3
    finally:
        enforce_from_snapshot(run_root, snap, ())


def test_enforce_from_snapshot_restores_like_the_guard(run_root: Path) -> None:
    snap = capture_snapshot(run_root)
    (run_root / "state.json").write_text("bad")
    (run_root / "items" / "b_001" / "proposal.md").write_text("fine\n")
    (run_root / "new.txt").write_text("illegal")
    result = enforce_from_snapshot(run_root, snap, MATRIX.globs("proposer", {"item": "b_001"}))
    assert result.violations == ("new.txt", "state.json")
    assert (run_root / "state.json").read_text() == '{"phase": "B"}'
    assert (run_root / "items" / "b_001" / "proposal.md").read_text() == "fine\n"
    assert not (run_root / "new.txt").exists()
    assert not snap.exists(), "the snapshot is cleaned up by default"


def test_cli_round_trip(tmp_path: Path, run_root: Path) -> None:
    matrix_file = tmp_path / "matrix.json"
    matrix_file.write_text(json.dumps({"roles": {r: list(g) for r, g in MATRIX.roles.items()}}))
    py = sys.executable
    snap = subprocess.run(
        [py, "-m", "agent_harness.jurisdiction", "capture",
         "--run-root", str(run_root), "--matrix", str(matrix_file)],
        check=True, capture_output=True, text=True).stdout.strip()

    (run_root / "journal.md").write_text("tampered by proposer\n")
    (run_root / "items" / "b_001" / "code.py").write_text("ok\n")

    out = subprocess.run(
        [py, "-m", "agent_harness.jurisdiction", "enforce",
         "--run-root", str(run_root), "--matrix", str(matrix_file),
         "--snapshot-dir", snap, "--role", "proposer", "--slot", "item=b_001"],
        check=True, capture_output=True, text=True).stdout
    result = json.loads(out)
    assert result == {"ok": False, "violations": ["journal.md"], "restored": ["journal.md"]}
    assert (run_root / "journal.md").read_text() == "day 1\n"
    assert (run_root / "items" / "b_001" / "code.py").read_text() == "ok\n"
