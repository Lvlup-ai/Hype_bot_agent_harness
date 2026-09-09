"""PASS, RETRY or ESCALATE, under a cap the reviewer cannot exceed."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_harness.budget import Budget
from agent_harness.ledger import Ledger
from agent_harness.review_loop import (
    BoundaryState,
    Decision,
    Direction,
    Finding,
    Review,
    ReviewError,
)

AGAINST = Finding(Direction.AGAINST, "n is 12", "the file has 9 rows")
FOR = Finding(Direction.FOR, "median is 3", "recomputed: 4")
NEUTRAL = Finding(Direction.NEUTRAL, "unit missing", "table header")


def review(max_retries: int = 3, **kw) -> Review:
    return Review(BoundaryState(boundary="A→B"), max_retries, **kw)


# ── Findings ─────────────────────────────────────────────────────────────────

def test_a_finding_needs_a_direction_and_evidence() -> None:
    with pytest.raises(ReviewError, match="unknown direction"):
        Finding("MAYBE", "claim", "evidence")  # type: ignore[arg-type]
    with pytest.raises(ReviewError, match="claim and its evidence"):
        Finding(Direction.AGAINST, "claim", "  ")
    assert Finding("AGAINST", "c", "e").direction is Direction.AGAINST


# ── PASS ─────────────────────────────────────────────────────────────────────

def test_pass_with_nothing_to_report() -> None:
    out = review().decide(Decision.PASS)
    assert out.decision is Decision.PASS and not out.converted
    assert out.reason == "no finding alters the reading"


def test_pass_with_neutral_or_favourable_findings() -> None:
    out = review().decide("PASS", [FOR, NEUTRAL])
    assert out.decision is Decision.PASS and len(out.findings) == 2


def test_pass_with_an_against_finding_needs_a_reason() -> None:
    with pytest.raises(ReviewError, match="must say why"):
        review().decide(Decision.PASS, [AGAINST])
    out = review().decide(Decision.PASS, [AGAINST], reason="wording only, the count is in a note")
    assert out.decision is Decision.PASS


# ── RETRY ────────────────────────────────────────────────────────────────────

def test_retry_needs_an_against_finding() -> None:
    with pytest.raises(ReviewError, match="at least one AGAINST"):
        review().decide(Decision.RETRY, [NEUTRAL, FOR], redo=["recount"])


def test_retry_needs_something_to_redo() -> None:
    with pytest.raises(ReviewError, match="what to redo"):
        review().decide(Decision.RETRY, [AGAINST], redo=["  "])


def test_retry_counts_and_numbers_itself() -> None:
    r = review(max_retries=3)
    out = r.decide(Decision.RETRY, [AGAINST], redo=["recount the rows"])
    assert out.decision is Decision.RETRY and out.retry_number == 1
    assert r.state.retries == 1
    assert out.reason == "retry 1/3"


def test_beyond_the_cap_a_retry_becomes_an_escalate() -> None:
    r = review(max_retries=2)
    r.decide(Decision.RETRY, [AGAINST], redo=["x"])
    r.decide(Decision.RETRY, [AGAINST], redo=["x"])
    out = r.decide(Decision.RETRY, [AGAINST], redo=["x"])
    assert out.decision is Decision.ESCALATE and out.requested is Decision.RETRY
    assert out.converted and "retry cap reached (2/2)" in out.reason
    assert r.state.retries == 2, "the refused retry is not counted"


def test_a_cap_of_zero_never_retries() -> None:
    out = review(max_retries=0).decide(Decision.RETRY, [AGAINST], redo=["x"])
    assert out.decision is Decision.ESCALATE


def test_a_retry_is_refused_when_the_block_has_no_budget_left() -> None:
    b = Budget(total=1)
    b.consume("only one", "in_scope")
    out = review(budget=b).decide(Decision.RETRY, [AGAINST], redo=["x"])
    assert out.decision is Decision.ESCALATE and "budget is exhausted" in out.reason


def test_a_retry_with_budget_left_goes_through_without_spending_it() -> None:
    b = Budget(total=5)
    out = review(budget=b).decide(Decision.RETRY, [AGAINST], redo=["x"])
    assert out.decision is Decision.RETRY
    assert b.spent_this_run() == 0, "the retried block spends its own trials"


def test_every_retry_is_one_ledger_line(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.md", ("gate", "retry"))
    r = review(max_retries=3, ledger=ledger, subject="subj")
    r.decide(Decision.RETRY, [AGAINST], redo=["recount", "restate n"])
    r.decide(Decision.RETRY, [AGAINST], redo=["again"])
    assert ledger.counts("subj")["retry"] == 2
    assert "A→B: retry 1/3 — recount; restate n" in ledger.path.read_text()


def test_a_converted_retry_leaves_no_ledger_line(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "l.md", ("retry",))
    r = review(max_retries=0, ledger=ledger)
    r.decide(Decision.RETRY, [AGAINST], redo=["x"])
    assert ledger.total_lines() == 0


# ── ESCALATE ─────────────────────────────────────────────────────────────────

def test_escalate_needs_a_reason() -> None:
    with pytest.raises(ReviewError, match="needs a human decision"):
        review().decide(Decision.ESCALATE)
    out = review().decide(Decision.ESCALATE, reason="the threshold itself is wrong")
    assert out.decision is Decision.ESCALATE and out.retry_number is None


# ── Vocabulary and trail ─────────────────────────────────────────────────────

def test_an_invented_decision_is_refused_and_leaves_no_trace() -> None:
    r = review()
    with pytest.raises(ReviewError, match="unknown decision 'RELAUNCH'"):
        r.decide("RELAUNCH", [AGAINST], redo=["x"])
    assert r.state.history == [] and r.state.retries == 0


def test_a_malformed_retry_leaves_no_trace() -> None:
    r = review()
    with pytest.raises(ReviewError):
        r.decide(Decision.RETRY, [NEUTRAL], redo=["x"])
    assert r.state.history == []


def test_the_state_keeps_the_whole_trail() -> None:
    r = review()
    r.decide(Decision.PASS)
    r.decide(Decision.RETRY, [AGAINST], redo=["x"])
    assert [h["decision"] for h in r.state.history] == ["PASS", "RETRY"]
    assert r.state.history[1]["findings"][0]["direction"] == "AGAINST"


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli(*args: str) -> tuple[int, dict]:
    proc = subprocess.run([sys.executable, "-m", "agent_harness.review_loop", *args],
                          capture_output=True, text=True)
    return proc.returncode, json.loads(proc.stdout)


def test_cli_persists_the_boundary_state(tmp_path: Path) -> None:
    state = str(tmp_path / "boundary.json")
    base = ("--state-file", state, "--boundary", "A→B", "--max-retries", "1")
    code, out = _cli(*base, "--decision", "RETRY",
                     "--finding", "AGAINST::n is 12::9 rows", "--redo", "recount")
    assert code == 0 and out["decision"] == "RETRY" and out["retries_spent"] == 1
    code, out = _cli(*base, "--decision", "RETRY",
                     "--finding", "AGAINST::n is 12::still 9 rows", "--redo", "recount")
    assert out["decision"] == "ESCALATE" and out["converted"] is True
    saved = json.loads(Path(state).read_text())
    assert saved["retries"] == 1 and len(saved["history"]) == 2


def test_cli_refuses_a_malformed_decision_and_saves_nothing(tmp_path: Path) -> None:
    state = tmp_path / "b.json"
    code, out = _cli("--state-file", str(state), "--boundary", "A→B", "--max-retries", "3",
                     "--decision", "RETRY", "--finding", "NEUTRAL::x::y", "--redo", "z")
    assert code == 2 and "AGAINST" in out["error"]
    assert not state.exists()
    code, out = _cli("--state-file", str(state), "--boundary", "A→B", "--max-retries", "3",
                     "--decision", "RETRY", "--finding", "bad finding", "--redo", "z")
    assert code == 2 and "DIRECTION::claim::evidence" in out["error"]


def test_cli_refuses_a_state_file_from_another_boundary(tmp_path: Path) -> None:
    state = str(tmp_path / "b.json")
    _cli("--state-file", state, "--boundary", "A→B", "--max-retries", "3", "--decision", "PASS")
    code, out = _cli("--state-file", state, "--boundary", "B→C", "--max-retries", "3",
                     "--decision", "PASS")
    assert code == 2 and "belongs to boundary" in out["error"]
