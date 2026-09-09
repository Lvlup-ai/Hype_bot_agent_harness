"""The loader refuses what a reader could not trust, and the round trip loses nothing."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from agent_harness.contracts import (
    ContractError,
    Deliverable,
    DeliverableMissing,
    DeliverableUnreadable,
    FloatOrNaN,
    InconsistentDeliverable,
    ProposalItem,
    ProposalReport,
    Registry,
    SchemaViolation,
    UnknownContract,
    example_registry,
)


def report() -> ProposalReport:
    return ProposalReport.build([
        ProposalItem(id="a", score=1.5, accepted=True, scores_by_horizon={1: 0.5, 6: 1.5}),
        ProposalItem(id="b", score=math.nan, accepted=False),
        ProposalItem(id="c", score=-2.0, accepted=False, scores_by_horizon={1: math.nan}),
    ])


@pytest.fixture
def registry() -> Registry:
    return example_registry()


# ── Round trip ───────────────────────────────────────────────────────────────

def test_what_the_writer_dumps_the_reader_loads(registry: Registry, tmp_path: Path) -> None:
    path = registry.dump(report(), tmp_path / "out" / "report.json")
    again = registry.load(path)
    assert again.summary.count == 3 and again.summary.accepted == 1
    assert again.summary.best_score == 1.5
    assert again.items[0].id == "a"


def test_nan_survives_as_null_and_comes_back_as_nan(registry: Registry, tmp_path: Path) -> None:
    path = registry.dump(report(), tmp_path / "r.json")
    raw = json.loads(path.read_text())
    assert raw["items"][1]["score"] is None, "strict JSON: null, never NaN"
    again = registry.load(path)
    assert math.isnan(again.items[1].score)
    assert math.isnan(again.items[2].scores_by_horizon[1])


def test_integer_keys_survive(registry: Registry, tmp_path: Path) -> None:
    path = registry.dump(report(), tmp_path / "r.json")
    raw = json.loads(path.read_text())
    assert list(raw["items"][0]["scores_by_horizon"]) == ["1", "6"], "JSON keys are strings"
    again = registry.load(path)
    assert again.items[0].scores_by_horizon == {1: 0.5, 6: 1.5}


def test_dumps_is_strict_json(registry: Registry) -> None:
    text = registry.dumps(report())
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)


def test_a_writer_cannot_emit_what_a_reader_would_refuse(registry: Registry) -> None:
    bad = report().model_copy(update={"summary": report().summary.model_copy(update={"count": 7})})
    with pytest.raises(InconsistentDeliverable, match="summary.count is 7 but there are 3"):
        registry.dumps(bad)


# ── Refusals ─────────────────────────────────────────────────────────────────

def test_missing_file(registry: Registry, tmp_path: Path) -> None:
    with pytest.raises(DeliverableMissing):
        registry.load(tmp_path / "nope.json")


def test_truncated_json_is_refused_with_its_position(registry: Registry, tmp_path: Path) -> None:
    text = registry.dumps(report())
    path = tmp_path / "cut.json"
    path.write_text(text[: len(text) // 2])
    with pytest.raises(DeliverableUnreadable, match=r"invalid JSON at line \d+ column \d+"):
        registry.load(path)


def test_missing_contract_field(registry: Registry) -> None:
    with pytest.raises(UnknownContract, match="missing or malformed `contract`"):
        registry.loads('{"items": []}')


def test_unknown_contract(registry: Registry) -> None:
    with pytest.raises(UnknownContract, match="'proposal-report/2' is not supported"):
        registry.loads('{"contract": "proposal-report/2", "items": []}')


def test_a_typo_in_a_field_name_is_an_error(registry: Registry) -> None:
    raw = json.loads(registry.dumps(report()))
    raw["itemz"] = raw.pop("items")
    with pytest.raises(SchemaViolation):
        registry.loads(json.dumps(raw))


def test_a_missing_piece_is_an_error(registry: Registry) -> None:
    raw = json.loads(registry.dumps(report()))
    del raw["summary"]
    with pytest.raises(SchemaViolation, match="summary"):
        registry.loads(json.dumps(raw))


def test_a_summary_that_contradicts_its_rows_is_refused(registry: Registry) -> None:
    raw = json.loads(registry.dumps(report()))
    raw["summary"]["accepted"] = 3
    with pytest.raises(InconsistentDeliverable, match="summary.accepted is 3 but 1 items"):
        registry.loads(json.dumps(raw))
    raw = json.loads(registry.dumps(report()))
    raw["summary"]["best_score"] = 9.0
    with pytest.raises(InconsistentDeliverable, match="best_score is 9.0 but the best item scores 1.5"):
        registry.loads(json.dumps(raw))


def test_a_deliverable_is_an_object(registry: Registry) -> None:
    with pytest.raises(SchemaViolation, match="is a JSON object"):
        registry.loads("[1, 2]")


def test_every_refusal_is_a_contract_error() -> None:
    for exc in (DeliverableMissing, DeliverableUnreadable, UnknownContract,
                SchemaViolation, InconsistentDeliverable):
        assert issubclass(exc, ContractError)


# ── Registry ─────────────────────────────────────────────────────────────────

class Note(Deliverable):
    text: str
    weight: FloatOrNaN = math.nan


def test_registering_and_loading_a_custom_contract() -> None:
    r = Registry()
    r.register("note/1", Note, checks=[lambda d: "empty note" if not d.text else None])
    assert r.contracts == ("note/1",)
    n = r.loads('{"contract": "note/1", "text": "hi"}')
    assert isinstance(n, Note) and math.isnan(n.weight)
    with pytest.raises(InconsistentDeliverable, match="empty note"):
        r.loads('{"contract": "note/1", "text": ""}')


def test_register_rejects_bad_inputs() -> None:
    r = Registry()
    with pytest.raises(ValueError):
        r.register("note", Note)
    with pytest.raises(TypeError):
        r.register("note/1", dict)  # type: ignore[arg-type]


def test_dumping_an_unregistered_contract_is_refused() -> None:
    r = Registry()
    with pytest.raises(UnknownContract):
        r.dumps(Note(contract="note/1", text="x"))


# ── CLI ──────────────────────────────────────────────────────────────────────

def test_cli_validates_a_file(registry: Registry, tmp_path: Path) -> None:
    good = registry.dump(report(), tmp_path / "good.json")
    proc = subprocess.run([sys.executable, "-m", "agent_harness.contracts", str(good)],
                          capture_output=True, text=True)
    assert proc.returncode == 0 and json.loads(proc.stdout)["contract"] == "proposal-report/1"

    raw = json.loads(good.read_text())
    raw["summary"]["count"] = 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))
    proc = subprocess.run([sys.executable, "-m", "agent_harness.contracts", str(bad)],
                          capture_output=True, text=True)
    out = json.loads(proc.stdout)
    assert proc.returncode == 2 and out["error"] == "InconsistentDeliverable"
