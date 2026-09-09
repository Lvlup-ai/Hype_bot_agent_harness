"""The bounded adversarial loop: PASS, RETRY or ESCALATE, at most N retries.

Why this module exists
----------------------
At the boundary between two blocks, a reviewing agent checks the deliverable
against its sources and decides whether the producing block must redo its
work. The original pipeline wrote the rules in the reviewer's mandate: three
decisions, a cap of N retries declared before the run, a retry only on a
finding that goes against the result, and a line in the ledger each time.
None of it was enforced. An audit later found the cap had no real cost and
the ledger held one line for two retries.

This module enforces the rules:

* three decisions, nothing else;
* every finding carries a direction (``FOR``, ``AGAINST``, ``NEUTRAL``) and
  evidence, because a finding without a direction cannot be acted on and a
  favourable correction is worth as much as an unfavourable one;
* a ``RETRY`` needs at least one ``AGAINST`` finding and says *what* to redo,
  never how. Spending a retry on cosmetics impoverishes the research twice;
* a ``PASS`` that carries ``AGAINST`` findings must say why they do not
  change the reading;
* beyond ``max_retries``, a ``RETRY`` becomes an ``ESCALATE``: the reviewer
  cannot retry a fourth time, a human decides;
* a retry is refused when the block's budget could not measure anything more
  (the block, not the reviewer, spends the trials on its re-run);
* every retry is one line in the ledger.

The loop is a boundary, not a phase: the orchestrator applies the outcome
(open the next phase, reopen the previous one, or stop and report).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from agent_harness.budget import Budget
from agent_harness.ledger import Ledger

__all__ = [
    "BoundaryState",
    "Decision",
    "Direction",
    "Finding",
    "Review",
    "ReviewError",
    "ReviewOutcome",
]


class Decision(str, Enum):
    PASS = "PASS"
    RETRY = "RETRY"
    ESCALATE = "ESCALATE"


class Direction(str, Enum):
    FOR = "FOR"            # reality is better than what the deliverable claims
    AGAINST = "AGAINST"    # reality is worse
    NEUTRAL = "NEUTRAL"    # label, unit, wording; the conclusion is unchanged


class ReviewError(ValueError):
    """The decision is malformed. Nothing was recorded."""


@dataclass(frozen=True)
class Finding:
    """One contested claim, with its direction and the evidence for the disagreement."""

    direction: Direction
    claim: str
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.direction, Direction):
            try:
                object.__setattr__(self, "direction", Direction(self.direction))
            except ValueError as exc:
                raise ReviewError(f"unknown direction {self.direction!r}") from exc
        if not self.claim.strip() or not self.evidence.strip():
            raise ReviewError("a finding needs both a claim and its evidence")

    def to_dict(self) -> dict:
        return {"direction": self.direction.value, "claim": self.claim, "evidence": self.evidence}


class BoundaryState(BaseModel):
    """Persistent state of one boundary: how many retries were spent, and the trail."""

    boundary: str
    retries: int = 0
    history: list[dict] = Field(default_factory=list)


@dataclass(frozen=True)
class ReviewOutcome:
    decision: Decision            # what the orchestrator must apply
    requested: Decision           # what the reviewer asked for
    retry_number: int | None      # 1-based, when decision is RETRY
    max_retries: int
    reason: str
    findings: tuple[Finding, ...] = ()
    redo: tuple[str, ...] = ()

    @property
    def converted(self) -> bool:
        """True when a RETRY was turned into an ESCALATE by the harness."""
        return self.requested is not self.decision

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value, "requested": self.requested.value,
            "converted": self.converted, "retry_number": self.retry_number,
            "max_retries": self.max_retries, "reason": self.reason,
            "findings": [f.to_dict() for f in self.findings], "redo": list(self.redo),
        }


class Review:
    """Applies the rules to one reviewer decision at one boundary."""

    def __init__(self, state: BoundaryState, max_retries: int,
                 budget: Budget | None = None, ledger: Ledger | None = None,
                 subject: str = "", retry_event: str = "retry") -> None:
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        self.state = state
        self.max_retries = max_retries
        self.budget = budget
        self.ledger = ledger
        self.subject = subject
        self.retry_event = retry_event

    def decide(self, decision: Decision | str, findings: Sequence[Finding] = (),
               redo: Sequence[str] = (), reason: str = "") -> ReviewOutcome:
        try:
            requested = Decision(decision)
        except ValueError as exc:
            raise ReviewError(
                f"unknown decision {decision!r}; allowed: {[d.value for d in Decision]}") from exc
        findings = tuple(findings)
        redo = tuple(r for r in redo if r.strip())
        against = [f for f in findings if f.direction is Direction.AGAINST]

        if requested is Decision.PASS:
            if against and not reason.strip():
                raise ReviewError(
                    "a PASS with AGAINST findings must say why they do not change the reading")
            outcome = ReviewOutcome(Decision.PASS, requested, None, self.max_retries,
                                    reason or "no finding alters the reading", findings, ())

        elif requested is Decision.ESCALATE:
            if not reason.strip():
                raise ReviewError("an ESCALATE must say what needs a human decision")
            outcome = ReviewOutcome(Decision.ESCALATE, requested, None, self.max_retries,
                                    reason, findings, redo)

        else:  # RETRY
            if not against:
                raise ReviewError(
                    "a RETRY needs at least one AGAINST finding; "
                    "cosmetic or favourable findings do not justify redoing the work")
            if not redo:
                raise ReviewError("a RETRY must list what to redo (never how)")
            outcome = self._retry(requested, findings, redo, reason)

        self.state.history.append(outcome.to_dict())
        return outcome

    def _retry(self, requested: Decision, findings: tuple[Finding, ...],
               redo: tuple[str, ...], reason: str) -> ReviewOutcome:
        if self.state.retries >= self.max_retries:
            return ReviewOutcome(
                Decision.ESCALATE, requested, None, self.max_retries,
                f"retry cap reached ({self.state.retries}/{self.max_retries}): a human decides",
                findings, redo)
        if self.budget is not None and self.budget.remaining() < 1:
            return ReviewOutcome(
                Decision.ESCALATE, requested, None, self.max_retries,
                "the block's budget is exhausted: a retry could not measure anything",
                findings, redo)
        self.state.retries += 1
        n = self.state.retries
        if self.ledger is not None:
            self.ledger.record(self.subject or self.state.boundary, self.retry_event,
                               f"{self.state.boundary}: retry {n}/{self.max_retries} — "
                               + "; ".join(redo))
        return ReviewOutcome(Decision.RETRY, requested, n, self.max_retries,
                             reason or f"retry {n}/{self.max_retries}", findings, redo)


# ── CLI ──────────────────────────────────────────────────────────────────────
#
# For an orchestrating LLM: the boundary state lives in a JSON file it never
# edits by hand. One call = one decision. Exit 2 on a malformed decision.

def _parse_finding(text: str) -> Finding:
    parts = text.split("::", 2)
    if len(parts) != 3:
        raise ReviewError(f"a finding is DIRECTION::claim::evidence, got {text!r}")
    return Finding(Direction(parts[0]) if parts[0] in Direction.__members__ else parts[0],
                   parts[1], parts[2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_harness.review_loop",
        description="Record a reviewer decision at a boundary, under the retry cap.")
    parser.add_argument("--state-file", required=True, type=Path)
    parser.add_argument("--boundary", required=True)
    parser.add_argument("--max-retries", required=True, type=int)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--finding", action="append", default=[], metavar="DIRECTION::claim::evidence")
    parser.add_argument("--redo", action="append", default=[])
    parser.add_argument("--reason", default="")
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--ledger-events", nargs="*", default=["retry"])
    parser.add_argument("--subject", default="")
    args = parser.parse_args(argv)

    if args.state_file.exists():
        state = BoundaryState.model_validate_json(args.state_file.read_text(encoding="utf-8"))
        if state.boundary != args.boundary:
            print(json.dumps({"error": f"state file belongs to boundary {state.boundary!r}"}))
            return 2
    else:
        state = BoundaryState(boundary=args.boundary)

    ledger = Ledger(args.ledger, tuple(args.ledger_events)) if args.ledger else None
    review = Review(state, args.max_retries, ledger=ledger, subject=args.subject)
    try:
        findings = [_parse_finding(f) for f in args.finding]
        outcome = review.decide(args.decision, findings, args.redo, args.reason)
    except ReviewError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2  # nothing saved

    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.state_file.with_suffix(".json.tmp")
    tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(args.state_file)
    print(json.dumps({**outcome.to_dict(), "retries_spent": state.retries}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
