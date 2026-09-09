"""The ledger appends, never edits, and only accepts its own vocabulary."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent_harness.ledger import Ledger, UnknownEvent

EVENTS = ("gate", "run", "report", "retry")


def fixed_clock():
    return datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc)


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledgers" / "fleet.md", EVENTS, clock=fixed_clock)


def test_first_record_creates_the_file_with_its_header(ledger: Ledger) -> None:
    line = ledger.record("abc123", "gate", "first look")
    text = ledger.path.read_text()
    assert text.startswith("# Ledger (append-only)")
    assert "| Date (UTC) | Subject | Event | Note |" in text
    assert line == "| 2026-01-02 03:04 UTC | `abc123` | gate | first look |\n"
    assert text.endswith(line)


def test_records_are_appended_in_order_and_nothing_is_rewritten(ledger: Ledger) -> None:
    ledger.record("s1", "gate")
    before = ledger.path.read_text()
    ledger.record("s1", "run", "pass 1")
    after = ledger.path.read_text()
    assert after.startswith(before)
    assert after.count("\n") == before.count("\n") + 1


def test_an_unknown_event_is_refused_and_nothing_is_written(ledger: Ledger) -> None:
    with pytest.raises(UnknownEvent, match="'verdict' is not in"):
        ledger.record("s1", "verdict")
    assert not ledger.path.exists()


def test_an_empty_vocabulary_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Ledger(tmp_path / "l.md", ())


def test_counts_by_event_and_by_subject(ledger: Ledger) -> None:
    for subject, event in (("s1", "gate"), ("s1", "run"), ("s1", "retry"),
                           ("s2", "gate"), ("s2", "run"), ("s2", "run")):
        ledger.record(subject, event)
    assert ledger.counts() == {"gate": 2, "run": 3, "report": 0, "retry": 1}
    assert ledger.counts("s2") == {"gate": 1, "run": 2, "report": 0, "retry": 0}
    assert ledger.total_lines() == 6 and ledger.total_lines("s1") == 3
    assert ledger.subjects() == ("s1", "s2")


def test_total_lines_counts_events_outside_the_current_vocabulary(tmp_path: Path) -> None:
    """The file may carry names that were dropped from the list since. They still count."""
    older = Ledger(tmp_path / "l.md", ("gate", "iteration_no_go"), clock=fixed_clock)
    older.record("s1", "gate")
    older.record("s1", "iteration_no_go", "written before the vocabulary was frozen")
    current = Ledger(tmp_path / "l.md", ("gate", "run"), clock=fixed_clock)
    assert sum(current.counts().values()) == 1
    assert current.total_lines() == 2


def test_a_note_cannot_break_the_table(ledger: Ledger) -> None:
    ledger.record("s1", "run", "a | pipe\nand a newline")
    rows = [l for l in ledger.path.read_text().splitlines() if l.startswith("| 2026")]
    assert len(rows) == 1
    assert rows[0].count(" | ") == 3


def test_reading_a_missing_ledger_is_empty_not_an_error(ledger: Ledger) -> None:
    assert ledger.counts() == {e: 0 for e in EVENTS}
    assert ledger.total_lines() == 0
    assert ledger.subjects() == ()


def test_cli_record_and_counts(tmp_path: Path) -> None:
    path = str(tmp_path / "l.md")
    base = [sys.executable, "-m", "agent_harness.ledger"]
    opts = ["--path", path, "--events", *EVENTS]
    ok = subprocess.run([*base, "record", *opts, "--subject", "s1", "--event", "run", "--note", "n"],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and json.loads(ok.stdout) == {"ok": True, "total_lines": 1}
    bad = subprocess.run([*base, "record", *opts, "--subject", "s1", "--event", "verdict"],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and "not in" in json.loads(bad.stdout)["error"]
    counts = subprocess.run([*base, "counts", *opts], capture_output=True, text=True)
    assert json.loads(counts.stdout)["counts"]["run"] == 1
