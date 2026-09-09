"""The orchestrator is a table: allowed transitions produce directives, the rest raise."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agent_harness.state_machine import (
    TRANSITIONS,
    Audit,
    Directive,
    Event,
    FailureKind,
    LoopConfig,
    PhaseMachine,
    PhaseState,
    RunConfig,
    RunExistsError,
    RunNotFoundError,
    Step,
    TransitionError,
    Verdict,
    has_converged,
    init_run,
    load_run,
    save_state,
)


def machine(**overrides) -> PhaseMachine:
    cfg = LoopConfig(**{"max_iterations": 3, "max_failures": 2,
                        "convergence_window": 2, "convergence_threshold_pct": 1.0,
                        **overrides})
    return PhaseMachine(PhaseState(phase="B"), cfg)


# ── The happy path ───────────────────────────────────────────────────────────

def test_one_full_iteration() -> None:
    m = machine()
    assert m.next_action() is Directive.PROPOSE
    assert m.state.step is Step.PROPOSING and m.state.iteration == 1
    assert m.record_audit(Audit.GO) is Directive.EXECUTE
    assert m.state.step is Step.EXECUTING
    assert m.record_result(accepted=True, score=10.0, result_id="B_001") is Directive.NEXT_ITERATION
    assert m.state.best == 10.0 and m.state.best_id == "B_001"
    assert m.state.step is Step.IDLE


def test_a_rejected_result_never_becomes_the_best() -> None:
    m = machine()
    m.next_action(); m.record_audit(Audit.GO)
    m.record_result(accepted=False, score=99.0, result_id="B_001")
    assert m.state.best is None and m.state.best_id is None
    assert m.state.rejected_ids == ["B_001"]


def test_the_best_only_moves_up() -> None:
    m = machine(max_iterations=5)
    for score, rid in ((5.0, "a"), (3.0, "b"), (7.0, "c")):
        m.next_action(); m.record_audit(Audit.GO)
        m.record_result(accepted=True, score=score, result_id=rid)
    assert m.state.best_history == [5.0, 5.0, 7.0]
    assert m.state.best_id == "c"


# ── Forbidden transitions ────────────────────────────────────────────────────

def test_recording_a_result_before_an_audit_is_forbidden() -> None:
    m = machine()
    m.next_action()
    with pytest.raises(TransitionError, match="'result' is not allowed in step 'proposing'"):
        m.record_result(accepted=True, score=1.0, result_id="x")


def test_recording_an_audit_without_a_proposal_is_forbidden() -> None:
    m = machine()
    with pytest.raises(TransitionError, match="'audit' is not allowed in step 'idle'"):
        m.record_audit(Audit.GO)


def test_asking_for_the_next_action_twice_is_forbidden() -> None:
    m = machine()
    m.next_action()
    with pytest.raises(TransitionError):
        m.next_action()


def test_a_forbidden_event_leaves_the_state_untouched() -> None:
    m = machine()
    m.next_action()
    before = m.state.model_copy(deep=True)
    with pytest.raises(TransitionError):
        m.record_result(accepted=True, score=1.0, result_id="x")
    assert m.state == before


def test_the_table_is_the_whole_protocol() -> None:
    """Every (step, event) pair not in the table must raise; every pair in it must not."""
    for step in Step:
        for event in Event:
            m = machine()
            m.state.step = step
            if step is Step.DONE:
                m.state.verdict = Verdict.MAX_ITERATIONS
            call = {
                Event.NEXT_ACTION: m.next_action,
                Event.AUDIT: lambda: m.record_audit(Audit.GO),
                Event.FAILURE: lambda: m.record_failure(FailureKind.RUNTIME_ERROR),
                Event.RESULT: lambda: m.record_result(True, 1.0, "x"),
            }[event]
            if (step, event) in TRANSITIONS:
                call()
            else:
                with pytest.raises(TransitionError):
                    call()


# ── Failures and the retry cap ───────────────────────────────────────────────

def test_no_go_means_retry_until_the_cap_then_abandon() -> None:
    m = machine(max_failures=2)
    m.next_action()
    assert m.record_audit(Audit.NO_GO) is Directive.PROPOSE
    assert m.state.step is Step.PROPOSING
    assert m.record_audit(Audit.NO_GO) is Directive.NEXT_ITERATION
    assert m.state.step is Step.IDLE
    assert m.state.iteration == 1, "the abandoned iteration is still counted"


@pytest.mark.parametrize("kind", list(FailureKind))
def test_every_failure_kind_counts_like_a_no_go(kind: FailureKind) -> None:
    m = machine(max_failures=2)
    m.next_action()
    assert m.record_failure(kind) is Directive.PROPOSE
    assert m.record_failure(kind) is Directive.NEXT_ITERATION


def test_a_failure_during_execution_is_accepted_too() -> None:
    m = machine(max_failures=3)
    m.next_action(); m.record_audit(Audit.GO)
    assert m.state.step is Step.EXECUTING
    assert m.record_failure(FailureKind.RUNTIME_ERROR) is Directive.PROPOSE
    assert m.state.step is Step.PROPOSING


def test_the_failure_counter_resets_each_iteration() -> None:
    m = machine(max_failures=2)
    m.next_action(); m.record_audit(Audit.NO_GO)
    m.record_audit(Audit.GO); m.record_result(True, 1.0, "a")
    m.next_action()
    assert m.state.consecutive_failures == 0


# ── Verdicts ─────────────────────────────────────────────────────────────────

def test_the_iteration_cap_ends_the_phase_with_max_iterations() -> None:
    m = machine(max_iterations=2, convergence_window=10)
    for rid in ("a", "b"):
        m.next_action(); m.record_audit(Audit.GO); m.record_result(True, 1.0, rid)
    assert m.next_action() is Directive.PHASE_DONE
    assert m.state.verdict is Verdict.MAX_ITERATIONS
    assert not m.state.inconclusive


def test_a_phase_without_an_acceptable_result_is_inconclusive() -> None:
    m = machine(max_iterations=2)
    for _ in range(2):
        m.next_action(); m.record_audit(Audit.GO); m.record_result(False, 5.0, "x")
    assert m.next_action() is Directive.PHASE_DONE
    assert m.state.verdict is Verdict.NO_VIABLE_RESULT
    assert m.state.inconclusive


def test_a_negative_best_at_the_cap_is_inconclusive_too() -> None:
    m = machine(max_iterations=1)
    m.next_action(); m.record_audit(Audit.GO); m.record_result(True, -3.0, "x")
    m.next_action()
    assert m.state.verdict is Verdict.NO_VIABLE_RESULT


def test_convergence_stops_the_phase_early() -> None:
    m = machine(max_iterations=10, convergence_window=2, convergence_threshold_pct=1.0)
    scores = (10.0, 10.05, 10.08)   # +0.8 % over the window: converged
    directives = []
    for i, s in enumerate(scores):
        m.next_action(); m.record_audit(Audit.GO)
        directives.append(m.record_result(True, s, f"r{i}"))
    assert directives[-1] is Directive.PHASE_DONE
    assert m.state.verdict is Verdict.CONVERGED
    assert m.next_action() is Directive.PHASE_DONE, "done stays done"


def test_no_early_stop_while_the_best_is_negative() -> None:
    m = machine(max_iterations=10, convergence_window=2, convergence_threshold_pct=1.0)
    for i, s in enumerate((-5.0, -5.0, -5.0, -5.0)):
        m.next_action(); m.record_audit(Audit.GO)
        d = m.record_result(True, s, f"r{i}")
    assert d is Directive.NEXT_ITERATION
    assert m.state.verdict is None


def test_early_stop_on_a_negative_best_when_allowed() -> None:
    m = machine(max_iterations=10, convergence_window=2, convergence_threshold_pct=1.0,
                stop_only_when_best_nonnegative=False)
    for i in range(3):
        m.next_action(); m.record_audit(Audit.GO)
        d = m.record_result(True, -5.0, f"r{i}")
    assert d is Directive.PHASE_DONE and m.state.verdict is Verdict.CONVERGED


def test_verdicts_are_a_closed_vocabulary() -> None:
    with pytest.raises(ValueError):
        Verdict("looks_good_enough")


# ── has_converged ────────────────────────────────────────────────────────────

def test_convergence_needs_window_plus_one_points() -> None:
    assert not has_converged([1.0, 1.0], window=2, threshold_pct=0.0)
    assert has_converged([1.0, 1.0, 1.0], window=2, threshold_pct=0.0)


def test_convergence_gain_is_relative_to_the_older_value() -> None:
    assert has_converged([100.0, 101.0, 101.5], window=2, threshold_pct=2.0)
    assert not has_converged([100.0, 101.0, 103.0], window=2, threshold_pct=2.0)
    assert has_converged([-100.0, -100.0, -100.0], window=2, threshold_pct=0.0)


# ── Persistence ──────────────────────────────────────────────────────────────

def test_init_creates_config_and_state_and_refuses_to_overwrite(tmp_path: Path) -> None:
    cfg = RunConfig(phases=("B", "D"), loop=LoopConfig(max_iterations=2))
    ctx = init_run(cfg, tmp_path, run_id="r1")
    assert ctx.config_path.exists() and ctx.state_path.exists()
    with pytest.raises(RunExistsError):
        init_run(cfg, tmp_path, run_id="r1")


def test_state_round_trips_through_disk(tmp_path: Path) -> None:
    ctx = init_run(RunConfig(phases=("B",)), tmp_path, run_id="r1")
    m = PhaseMachine(ctx.state.phase("B"), ctx.config.loop)
    m.next_action(); m.record_audit(Audit.GO); m.record_result(True, 4.0, "B_001")
    save_state(ctx)
    again = load_run("r1", tmp_path)
    assert again.state.phase("B").best_id == "B_001"
    assert again.state.phase("B").step is Step.IDLE
    assert again.config.loop == ctx.config.loop


def test_loading_an_unknown_run_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError):
        load_run("nope", tmp_path)


def test_no_tmp_file_survives_a_save(tmp_path: Path) -> None:
    ctx = init_run(RunConfig(phases=("B",)), tmp_path, run_id="r1")
    save_state(ctx)
    assert sorted(p.name for p in ctx.root.iterdir()) == ["config.json", "state.json"]


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli(*args: str) -> tuple[int, dict]:
    proc = subprocess.run([sys.executable, "-m", "agent_harness.state_machine", *args],
                          capture_output=True, text=True)
    return proc.returncode, json.loads(proc.stdout)


def test_cli_drives_a_phase_to_its_verdict(tmp_path: Path) -> None:
    root = str(tmp_path)
    code, out = _cli("init", "--runs-root", root, "--run-id", "r1", "--phases", "B",
                     "--max-iterations", "1", "--convergence-window", "5")
    assert code == 0 and out["run_id"] == "r1"
    common = ("--runs-root", root, "--run-id", "r1", "--phase", "B")

    code, out = _cli("next-action", *common)
    assert (code, out["action"], out["iteration"]) == (0, "propose", 1)
    code, out = _cli("record-audit", *common, "--verdict", "GO")
    assert out["action"] == "execute"
    code, out = _cli("record-result", *common, "--accepted", "true", "--score", "2.5", "--id", "B_001")
    assert out["action"] == "next_iteration" and out["best"] == 2.5
    code, out = _cli("next-action", *common)
    assert out["action"] == "phase_done" and out["verdict"] == "max_iterations"
    assert out["run_finished"] is True


def test_cli_refuses_a_forbidden_transition_and_saves_nothing(tmp_path: Path) -> None:
    root = str(tmp_path)
    _cli("init", "--runs-root", root, "--run-id", "r1", "--phases", "B")
    common = ("--runs-root", root, "--run-id", "r1", "--phase", "B")
    code, out = _cli("record-result", *common, "--accepted", "true", "--score", "1", "--id", "x")
    assert code == 2 and "not allowed in step 'idle'" in out["error"]
    state = json.loads((tmp_path / "r1" / "state.json").read_text())
    assert state["phases"] == {}, "the refused event left no trace"


def test_cli_refuses_an_undeclared_phase(tmp_path: Path) -> None:
    _cli("init", "--runs-root", str(tmp_path), "--run-id", "r1", "--phases", "B")
    code, out = _cli("next-action", "--runs-root", str(tmp_path), "--run-id", "r1", "--phase", "Z")
    assert code == 2 and "not declared" in out["error"]


def test_cli_rejects_an_invented_verdict_or_kind(tmp_path: Path) -> None:
    _cli("init", "--runs-root", str(tmp_path), "--run-id", "r1", "--phases", "B")
    common = ("--runs-root", str(tmp_path), "--run-id", "r1", "--phase", "B")
    _cli("next-action", *common)
    proc = subprocess.run([sys.executable, "-m", "agent_harness.state_machine",
                           "record-audit", *common, "--verdict", "MAYBE"],
                          capture_output=True, text=True)
    assert proc.returncode == 2 and "invalid choice" in proc.stderr
