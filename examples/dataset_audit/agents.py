"""The three agents, scripted. Each function is where a model call would go.

They are deterministic on purpose: the example must run in CI without a key,
and the harness is exactly the part that does not need a model. A real
deployment replaces the body of ``propose``, ``audit`` and ``review`` with a
call to a model that reads the matching brief in ``briefs/`` and returns the
same structure.

The proposer's script is written to make every mechanism of the harness fire
once: a drift refused by the budget, a write outside its jurisdiction, an
off-topic rule that measures badly, and a rationale that overstates a result
and gets caught at the boundary.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from agent_harness.contracts import ProposalReport
from agent_harness.review_loop import Decision, Direction, Finding

from .data import COLUMNS
from .rules import OPERATORS, Rule, RuleError


@dataclass(frozen=True)
class Proposal:
    rule: Rule
    distance: str          # in_scope | adjacent | off_topic, declared by the proposer
    track: str | None      # which declared track this serves, if in scope
    claims: dict[str, float]   # what the proposer expects: checked at the boundary


# ── Proposer ─────────────────────────────────────────────────────────────────
#
# One proposal per iteration, in this order. Replace with a model call that
# reads briefs/proposer.md and returns a Proposal.

_SCRIPT: tuple[Proposal, ...] = (
    Proposal(Rule("age", "range", [0, 120]), "in_scope", "age", {"precision": 1.0, "recall": 0.33}),
    Proposal(Rule("email", "contains", "@"), "in_scope", "email", {"precision": 1.0, "recall": 0.33}),
    Proposal(Rule("score", "range", [0, 100]), "off_topic", None, {"precision": 0.5}),   # too early
    Proposal(Rule("signup_date", "range", ["2000-01-01", "2026-01-01"]), "in_scope", "signup_date",
             {"precision": 1.0, "recall": 0.22}),
    Proposal(Rule("country", "in", ["FR", "DE", "ES", "IT", "PT"]), "adjacent", None, {"precision": 0.5}),
    Proposal(Rule("score", "range", [0, 100]), "off_topic", None, {"precision": 0.5}),   # now allowed
    Proposal(Rule("age", "range", [1, 120]), "adjacent", "age", {"precision": 1.0, "recall": 0.8}),  # overstated
)

_TAMPERS_STATE_AT = 5   # iteration at which the scripted proposer misbehaves


def propose(iteration: int, run_root: Path, item_dir: Path) -> Proposal:
    """Write ``proposal.json`` and ``rationale.md`` into the item directory.

    At iteration 5 the script also writes into ``state.json``, which is not in
    the proposer's jurisdiction. The guard restores it; the harness records a
    jurisdiction failure. A model would do this by accident; the script does
    it on purpose so the example shows the restoration.
    """
    p = _SCRIPT[(iteration - 1) % len(_SCRIPT)]
    item_dir.mkdir(parents=True, exist_ok=True)
    (item_dir / "proposal.json").write_text(json.dumps({
        "rule": p.rule.to_dict(), "distance": p.distance, "track": p.track}, indent=2))
    (item_dir / "rationale.md").write_text(_rationale(p))
    if iteration == _TAMPERS_STATE_AT:
        (run_root / "state.json").write_text('{"phase": "rules", "verdict": "converged"}')
    return p


def _rationale(p: Proposal, claims: dict[str, float] | None = None) -> str:
    claims = claims if claims is not None else p.claims
    lines = [f"# Rule: {p.rule.column} {p.rule.op} {p.rule.value}", "",
             f"Distance from the question: {p.distance}.", ""]
    lines += [f"claimed_{k}: {v}" for k, v in claims.items()]
    return "\n".join(lines) + "\n"


def fix_rationale(item_dir: Path, measured: dict[str, float]) -> None:
    """The proposer's answer to a RETRY: restate the claims as measured."""
    raw = json.loads((item_dir / "proposal.json").read_text())
    p = Proposal(Rule.from_dict(raw["rule"]), raw["distance"], raw.get("track"), {})
    (item_dir / "rationale.md").write_text(
        _rationale(p, {k: measured[k] for k in ("precision", "recall")}))


# ── Auditor ──────────────────────────────────────────────────────────────────

def audit(item_dir: Path) -> tuple[str, str]:
    """GO when the rule is well formed against the known columns and operators.

    Writes ``audit.md``. Replace with a model call that reads briefs/auditor.md.
    """
    raw = json.loads((item_dir / "proposal.json").read_text())
    try:
        Rule.from_dict(raw["rule"]).validate(COLUMNS)
        verdict, why = "GO", f"well-formed rule over {raw['rule']['column']}"
    except RuleError as exc:
        verdict, why = "NO_GO", str(exc)
    (item_dir / "audit.md").write_text(f"# Audit\n\nVerdict: {verdict}\n\n{why}\n")
    return verdict, why


# ── Reviewer ─────────────────────────────────────────────────────────────────

_CLAIM = re.compile(r"^claimed_(\w+):\s*([0-9.]+)\s*$", re.M)
TOLERANCE = 0.05


def review(report: ProposalReport, run_root: Path) -> tuple[Decision, list[Finding], list[str], str]:
    """Compare each accepted rule's claims with what was measured.

    A claim that exceeds the measurement by more than the tolerance is a
    finding AGAINST the deliverable, and the reviewer asks for a RETRY that
    restates it. Replace with a model call that reads briefs/reviewer.md.
    """
    findings: list[Finding] = []
    redo: list[str] = []
    for item in report.items:
        if not item.accepted:
            continue
        item_dir = run_root / "items" / item.id
        measured = json.loads((item_dir / "measure.json").read_text())
        for metric, claimed in _CLAIM.findall((item_dir / "rationale.md").read_text()):
            claimed, actual = float(claimed), measured.get(metric)
            if actual is None:
                continue
            if claimed > actual + TOLERANCE:
                findings.append(Finding(Direction.AGAINST,
                                        f"{item.id} claims {metric} {claimed}",
                                        f"measure.json says {actual}"))
                redo.append(f"restate the claimed {metric} of {item.id}")
            elif claimed < actual - TOLERANCE:
                findings.append(Finding(Direction.FOR,
                                        f"{item.id} claims {metric} {claimed}",
                                        f"measure.json says {actual}: better than claimed"))
    if redo:
        return Decision.RETRY, findings, redo, "a claim overstates a measurement"
    return Decision.PASS, findings, [], "claims match measurements"


__all__ = ["Proposal", "propose", "fix_rationale", "audit", "review", "OPERATORS"]
