"""Trial budgets: distance costs, in-scope minimum, persistence, reasons, compartments."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_harness.budget import Budget, BudgetExhausted, DriftRefused


# ── Cost by distance ─────────────────────────────────────────────────────────

def test_each_distance_has_its_cost() -> None:
    b = Budget(total=10)
    b.consume("a", "in_scope")
    b.consume("b", "adjacent")
    b.consume("c", "off_topic")
    assert [t.cost for t in b.trials] == [1, 2, 3]
    assert b.spent_this_run() == 6 and b.remaining() == 4
    assert [t.n for t in b.trials] == [1, 2, 3]


def test_an_unknown_distance_is_refused() -> None:
    b = Budget(total=10)
    ok, why = b.check("sideways")
    assert not ok and "unknown distance" in why
    with pytest.raises(DriftRefused):
        b.consume("x", "sideways")


def test_costs_are_configurable() -> None:
    b = Budget(total=10, costs={"in_scope": 2, "off_topic": 5})
    b.consume("a", "in_scope")
    assert b.remaining() == 8


def test_a_budget_below_one_is_refused() -> None:
    with pytest.raises(ValueError):
        Budget(total=0)


# ── Exhaustion ───────────────────────────────────────────────────────────────

def test_the_budget_runs_out_and_says_so() -> None:
    b = Budget(total=3)
    b.consume("a", "in_scope"); b.consume("b", "in_scope"); b.consume("c", "in_scope")
    ok, why = b.check("in_scope")
    assert not ok and "budget exhausted" in why
    with pytest.raises(BudgetExhausted):
        b.consume("d", "in_scope")


def test_an_expensive_trial_is_refused_before_the_budget_is_zero() -> None:
    b = Budget(total=2)
    ok, why = b.check("off_topic")
    assert not ok and "2 left for a off_topic trial costing 3" in why


# ── Minimum in scope ─────────────────────────────────────────────────────────

def test_every_declared_track_needs_a_trial_before_drifting() -> None:
    b = Budget(total=20, tracks=("mechanism_a", "mechanism_b"))
    b.consume("a1", "in_scope", track="mechanism_a")
    ok, why = b.check("off_topic")
    assert not ok and "mechanism_b" in why and "has not been answered" in why
    with pytest.raises(DriftRefused):
        b.consume("x", "off_topic")
    b.consume("b1", "in_scope", track="mechanism_b")
    assert b.check("off_topic")[0]


def test_an_adjacent_trial_does_not_count_for_a_track() -> None:
    b = Budget(total=20, tracks=("t",))
    b.consume("near", "adjacent", track="t")
    assert b.tracks_without_trial() == ("t",)


def test_without_tracks_a_fixed_minimum_applies() -> None:
    b = Budget(total=20, min_in_scope=2)
    b.consume("a", "in_scope")
    ok, why = b.check("off_topic")
    assert not ok and "only 1 in-scope trial(s) of 2" in why
    b.consume("b", "in_scope")
    assert b.check("off_topic")[0]


def test_adjacent_trials_are_never_blocked_by_the_minimum() -> None:
    b = Budget(total=20, tracks=("t",))
    assert b.check("adjacent")[0]


# ── Persistence and reasons ──────────────────────────────────────────────────

def test_spending_is_persisted_per_subject_and_reloaded(tmp_path: Path) -> None:
    store = tmp_path / "budget"
    run1 = Budget.load(store, "subj", "first_pass", total=10)
    run1.consume("a", "in_scope"); run1.consume("b", "adjacent")
    assert run1.commit(store)
    saved = json.loads((store / "subj.json").read_text())
    assert saved == {"subject": "subj", "spent": {"_": 3}, "trials_total": 2}

    run2 = Budget.load(store, "subj", "retry", total=10)
    assert run2.spent_before() == 3 and run2.remaining() == 7
    assert run2.prior_trials == 2
    run2.consume("c", "in_scope")
    assert run2.cumulative() == 4 and run2.trials_total() == 3
    run2.commit(store)
    assert json.loads((store / "subj.json").read_text())["spent"] == {"_": 4}


def test_a_rerun_reads_the_cumulative_cost_but_adds_nothing(tmp_path: Path) -> None:
    store = tmp_path / "budget"
    first = Budget.load(store, "subj", "first_pass", total=10)
    first.consume("a", "in_scope"); first.commit(store)

    rerun = Budget.load(store, "subj", "rerun", total=10)
    rerun.consume("a", "in_scope")          # the same hypothesis, re-measured
    assert not rerun.consumes
    assert rerun.spent_this_run() == 1      # visible...
    assert rerun.cumulative() == 1          # ...but not added to the subject
    assert rerun.trials_total() == 1
    assert not rerun.commit(store)
    assert json.loads((store / "subj.json").read_text())["spent"] == {"_": 1}


def test_a_second_run_does_not_get_a_fresh_budget(tmp_path: Path) -> None:
    store = tmp_path / "budget"
    first = Budget.load(store, "subj", "first_pass", total=3)
    for n in "abc":
        first.consume(n, "in_scope")
    first.commit(store)
    second = Budget.load(store, "subj", "retry", total=3)
    assert second.remaining() == 0
    with pytest.raises(BudgetExhausted):
        second.consume("d", "in_scope")


def test_an_unknown_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="reason 'because'"):
        Budget(total=5, reason="because")


def test_a_budget_without_subject_is_not_persisted(tmp_path: Path) -> None:
    b = Budget(total=5)
    b.consume("a", "in_scope")
    assert not b.commit(tmp_path)
    assert not any(tmp_path.iterdir())


# ── Compartments ─────────────────────────────────────────────────────────────

def test_compartments_have_their_own_caps() -> None:
    b = Budget(total=6, compartments={"regime": 2, "entry": 4})
    b.consume("r1", "in_scope", compartment="regime")
    b.consume("r2", "in_scope", compartment="regime")
    ok, why = b.check("in_scope", compartment="regime")
    assert not ok and "exhausted in 'regime'" in why
    assert b.check("in_scope", compartment="entry")[0]
    assert b.remaining("entry") == 4


def test_compartments_never_sum_past_the_global_cap() -> None:
    b = Budget(total=3, compartments={"a": 3, "b": 3})
    b.consume("a1", "in_scope", compartment="a"); b.consume("a2", "in_scope", compartment="a")
    assert b.remaining("b") == 1, "the global cap binds"


def test_an_unknown_compartment_is_refused_when_compartments_exist() -> None:
    b = Budget(total=6, compartments={"regime": 2})
    ok, why = b.check("in_scope", compartment="exit")
    assert not ok and "unknown compartment" in why


def test_compartment_spending_is_persisted_separately(tmp_path: Path) -> None:
    store = tmp_path / "budget"
    b = Budget.load(store, "subj", "first_pass", total=6, compartments={"a": 3, "b": 3})
    b.consume("x", "in_scope", compartment="a"); b.commit(store)
    again = Budget.load(store, "subj", "retry", total=6, compartments={"a": 3, "b": 3})
    assert again.spent_before("a") == 1 and again.spent_before("b") == 0
    assert again.remaining("a") == 2


# ── Reporting ────────────────────────────────────────────────────────────────

def test_summary_shows_cost_this_run_and_across_runs(tmp_path: Path) -> None:
    store = tmp_path / "budget"
    first = Budget.load(store, "subj", "first_pass", total=10)
    first.consume("a", "in_scope"); first.consume("b", "off_topic"); first.commit(store)
    second = Budget.load(store, "subj", "retry", total=10)
    second.consume("c", "adjacent")
    s = second.summary()
    assert s.startswith("1 trials measured, 2/10 of budget spent this run")
    assert "in_scope 0, adjacent 1, off_topic 0" in s
    assert "4 left" in s
    assert "6/10 on subject `subj` across runs" in s


def test_summary_marks_a_rerun(tmp_path: Path) -> None:
    b = Budget.load(tmp_path, "subj", "rerun", total=10)
    b.consume("a", "in_scope")
    assert b.summary().endswith("[rerun: nothing added]")


def test_summary_lists_compartments() -> None:
    b = Budget(total=6, compartments={"a": 3, "b": 3})
    b.consume("x", "in_scope", compartment="a")
    assert b.summary().endswith("· a 1/3, b 0/3")
