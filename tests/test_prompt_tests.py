"""A brief that lies about the repository must fail, with the lie named."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent_harness.prompt_tests import (
    Brief,
    BriefDrift,
    flatten,
    must_carry_status,
    resolve_ref,
    run_checks,
)

METHODS = ("zscore", "rolling_mean", "rolling_std", "hurst")

BRIEF = """
# Proposer brief

You have **four** derived methods: `zscore`, `rolling_mean`,
`rolling_std` and `hurst`. The grid is **1000 ms**; the old
100 ms grid is gone. There is *no held-out data*: every number is in-sample.
"""


def brief(text: str = BRIEF) -> Brief:
    return Brief("agents/proposer.md", text=text)


# ── Flattening ───────────────────────────────────────────────────────────────

def test_flatten_removes_markdown_and_line_breaks() -> None:
    assert flatten("the **ten**\n   `methods`, *all* of __them__") == "the ten methods, all of them"


def test_flatten_keeps_identifiers_with_underscores() -> None:
    assert flatten("`rolling_mean` and snake_case") == "rolling_mean and snake_case"


# ── must_name_all ────────────────────────────────────────────────────────────

def test_name_all_passes_when_every_item_is_named() -> None:
    brief().must_name_all(METHODS, label="methods")


def test_name_all_lists_the_missing_ones() -> None:
    with pytest.raises(BriefDrift, match=r"2 of 6 methods never named: \['ou_half_life', 'combine'\]"):
        brief().must_name_all((*METHODS, "ou_half_life", "combine"), label="methods")


# ── must_state_count ─────────────────────────────────────────────────────────

def test_state_count_accepts_words_and_digits() -> None:
    brief().must_state_count(4, "derived methods")
    brief("There are 4 derived methods.").must_state_count(4, "derived methods")
    brief("Il y a quatre méthodes dérivées.").must_state_count(4, "méthodes dérivées")


def test_state_count_fails_on_a_wrong_count() -> None:
    with pytest.raises(BriefDrift, match="states 4 derived methods for 5 in the repository"):
        brief().must_state_count(5, "derived methods")


def test_state_count_fails_when_no_count_is_stated() -> None:
    with pytest.raises(BriefDrift, match="expected one of: 3 methods / three methods / trois methods"):
        brief("Some methods exist.").must_state_count(3, "methods")


def test_state_count_ignores_numbers_that_are_part_of_a_word() -> None:
    brief("v10 methods are listed: there are two methods.").must_state_count(2, "methods")


# ── must_quote ───────────────────────────────────────────────────────────────

def test_quote_finds_the_current_value_with_its_unit() -> None:
    brief().must_quote(1000, suffix=" ms")
    brief().must_quote(1000.0, suffix=" ms")


def test_quote_fails_on_a_stale_value() -> None:
    with pytest.raises(BriefDrift, match="must quote the current value '500 ms'"):
        brief().must_quote(500, suffix=" ms")


# ── must_not_contain / must_contain ──────────────────────────────────────────

def test_stale_phrases_are_caught_through_formatting() -> None:
    with pytest.raises(BriefDrift, match=r"stale phrase\(s\) present: \['100 ms grid'\]"):
        brief().must_not_contain("100 ms grid")
    brief().must_not_contain("50 ms grid")


def test_required_phrases_are_found_through_formatting() -> None:
    brief().must_contain("no held-out data", "in-sample")
    with pytest.raises(BriefDrift, match=r"required phrase\(s\) absent: \['walk-forward'\]"):
        brief().must_contain("in-sample", "walk-forward")


# ── must_carry_status ────────────────────────────────────────────────────────

def test_every_file_must_show_a_status_marker(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\nstatus: ✅ ready\n")
    (tmp_path / "b.md").write_text("# B\nno marker here\n")
    (tmp_path / "c.md").write_text("# C\n🔧 needs work\n")
    must_carry_status([tmp_path / "a.md", tmp_path / "c.md"], ["✅", "🔧"])
    with pytest.raises(BriefDrift, match=r"without a status marker \['✅', '🔧'\]: \['b.md'\]"):
        must_carry_status(sorted(tmp_path.glob("*.md")), ["✅", "🔧"])


# ── Missing brief ────────────────────────────────────────────────────────────

def test_a_missing_brief_is_a_drift(tmp_path: Path) -> None:
    with pytest.raises(BriefDrift, match="brief not found"):
        Brief(tmp_path / "nope.md")


# ── References ───────────────────────────────────────────────────────────────

@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "core.py").write_text("METHODS = ('a', 'b', 'c')\nclass Cfg:\n    GRID_MS = 1000\n")
    (tmp_path / "config.yaml").write_text("measure:\n  grid_ms: 1000\n  horizons: [1, 6, 24]\n")
    (tmp_path / "agents").mkdir()
    (tmp_path / "agents" / "p.md").write_text(
        "Three methods: `a`, `b`, `c`. Grid **1000 ms**. Horizons 1, 6 and 24 h. No held-out data.\n")
    (tmp_path / "library").mkdir()
    (tmp_path / "library" / "x.md").write_text("🟢 usable\n")
    (tmp_path / "library" / "y.md").write_text("no status\n")
    (tmp_path / "library" / "index.md").write_text("index, no status by design\n")
    return tmp_path


def test_resolve_python_and_yaml_references(repo: Path) -> None:
    assert resolve_ref("core:METHODS", repo) == ("a", "b", "c")
    assert resolve_ref("core:Cfg.GRID_MS", repo) == 1000
    assert resolve_ref("config.yaml#measure.grid_ms", repo) == 1000
    assert resolve_ref("config.yaml#measure.horizons.2", repo) == 24
    assert resolve_ref(42, repo) == 42 and resolve_ref("plain text", repo) == "plain text"


def test_run_checks_end_to_end(repo: Path) -> None:
    spec = {
        "briefs": {"agents/p.md": {
            "name_all": [{"items": "core:METHODS", "label": "methods"}],
            "state_count": [{"value": "core:METHODS", "noun": "methods"},
                            {"value": 3, "noun": "methods"}],
            "quote": [{"value": "config.yaml#measure.grid_ms", "suffix": " ms"}],
            "forbid": ["100 ms grid"],
            "require": ["no held-out data"],
        }},
        "status": [{"glob": "library/*.md", "markers": ["🟢", "🔧"], "except": ["library/index.md"]}],
    }
    report = run_checks(spec, repo)
    assert report.checked == 7
    assert report.failures == ["files without a status marker ['🟢', '🔧']: ['y.md']"]


def test_run_checks_reports_a_missing_brief(repo: Path) -> None:
    report = run_checks({"briefs": {"agents/missing.md": {"require": ["x"]}}}, repo)
    assert report.checked == 1 and "brief not found" in report.failures[0]


def test_cli_exit_code_follows_the_report(repo: Path) -> None:
    spec = repo / "prompt_checks.yaml"
    spec.write_text(textwrap.dedent("""
        briefs:
          agents/p.md:
            state_count:
              - value: core:METHODS
                noun: methods
            require: ["no held-out data"]
    """))
    proc = subprocess.run([sys.executable, "-m", "agent_harness.prompt_tests", str(spec)],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and json.loads(proc.stdout) == {"ok": True, "checked": 2, "failures": []}

    (repo / "core.py").write_text("METHODS = ('a', 'b', 'c', 'd')\n")
    proc = subprocess.run([sys.executable, "-m", "agent_harness.prompt_tests", str(spec)],
                          capture_output=True, text=True)
    out = json.loads(proc.stdout)
    assert proc.returncode == 1
    assert out["failures"] == ["p.md: states 3 methods for 4 in the repository"]
