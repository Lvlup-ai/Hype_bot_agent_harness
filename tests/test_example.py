"""The end-to-end example runs, and every mechanism fires exactly as documented."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_harness.prompt_tests import run_checks
from examples.dataset_audit import run as example
from examples.dataset_audit.data import N_ANOMALIES, rows

ROOT = Path(__file__).resolve().parent.parent


def test_the_table_has_nine_labelled_anomalies() -> None:
    assert N_ANOMALIES == 9
    assert len(rows()) == 40
    assert len({r["id"] for r in rows()}) == 40


def test_the_run_matches_its_documentation(tmp_path: Path) -> None:
    s = example.run(tmp_path, run_id="t", quiet=True)

    assert s["verdict"] == "max_iterations" and s["iterations"] == 7
    assert s["accepted"] == ["rule_01", "rule_02", "rule_04", "rule_07"]
    assert s["rejected"] == ["rule_06"]
    assert s["best_id"] == "rule_07" and s["best"] == 0.6154

    # iteration 3: refused before measurement, the track was not served yet
    assert s["refused"] == ["rule_03"]
    # iteration 5: the write outside the jurisdiction was restored
    assert s["violations"] == ["state.json"]
    state = json.loads((tmp_path / "t" / "state.json").read_text())
    assert state["phases"]["rules"]["verdict"] == "max_iterations", "state.json is the machine's"

    # budget: 1 + 1 + 1 + 3 + 2, five trials measured
    assert s["budget_spent"] == 8 and s["budget_trials"] == 5

    # boundary: one overstated claim, one retry, then a pass
    assert s["review_decisions"] == ["RETRY", "PASS"] and s["retries"] == 1
    assert s["ledger_counts"] == {"gate": 1, "run": 1, "report": 1, "retry": 1, "escalate": 0}
    assert s["ledger_lines"] == 4

    rationale = (tmp_path / "t" / "items" / "rule_07" / "rationale.md").read_text()
    assert "claimed_recall: 0.4444" in rationale, "the proposer restated its claim"


def test_the_run_is_deterministic(tmp_path: Path) -> None:
    a = example.run(tmp_path / "a", run_id="t", quiet=True)
    b = example.run(tmp_path / "b", run_id="t", quiet=True)
    assert a["trace"] == b["trace"]


def test_a_second_run_finds_the_budget_partly_spent(tmp_path: Path) -> None:
    example.run(tmp_path, run_id="first", quiet=True)
    saved = json.loads((tmp_path / "budget" / f"{example.dataset_id()}.json").read_text())
    assert saved["spent"] == {"_": 8}
    second = example.run(tmp_path, run_id="second", quiet=True)
    # 4 trials left: the off-topic and adjacent proposals are refused for budget, not drift
    assert second["refused"] == ["rule_03", "rule_06", "rule_07"]
    assert all("budget exhausted" in line for line in second["trace"] if "refused before" in line)
    assert second["budget_spent"] == 3
    assert any("11/12 on subject" in line and "across runs" in line for line in second["trace"])
    # the ledger persisted too: both runs are on it
    assert second["ledger_lines"] == 7 and second["ledger_counts"]["retry"] == 1


def test_the_briefs_tell_the_truth() -> None:
    spec = yaml.safe_load((ROOT / "examples" / "dataset_audit" / "prompt_checks.yaml").read_text())
    report = run_checks(spec, ROOT)
    assert report.failures == []
    assert report.checked >= 10
