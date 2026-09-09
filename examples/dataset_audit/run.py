"""The orchestrator: wires the harness around the three agents and prints a trace.

Run it from the repository root::

    python -m examples.dataset_audit.run

This is what an orchestrating LLM would do step by step through the CLIs
(``next-action``, ``record-*``, ``capture``/``enforce``). Here it is a script,
so the whole run is reproducible and the test suite can assert every line
of the trace.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from agent_harness.budget import Budget, BudgetExhausted, DriftRefused
from agent_harness.contracts import ProposalItem, ProposalReport, example_registry
from agent_harness.jurisdiction import Guard, Matrix
from agent_harness.ledger import Ledger
from agent_harness.review_loop import BoundaryState, Decision, Review
from agent_harness.state_machine import (
    Audit,
    Directive,
    FailureKind,
    LoopConfig,
    PhaseMachine,
    RunConfig,
    init_run,
    save_state,
)

from . import agents
from .data import dataset_id, rows
from .rules import Rule, measure

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.yaml"
PHASE = "rules"
BOUNDARY = "rules→report"

MATRIX = Matrix(
    roles={
        "proposer": ("items/{item}/proposal.json", "items/{item}/rationale.md"),
        "auditor": ("items/{item}/audit.md", "journal.md"),
        "reviewer": ("review/*",),
    },
    protected=(HERE / "briefs",),
)


def _say(trace: list[str], line: str, quiet: bool) -> None:
    trace.append(line)
    if not quiet:
        print(line)


def run(runs_root: Path, run_id: str = "example", quiet: bool = False) -> dict:
    """One full run. Returns a summary the tests assert on."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    trace: list[str] = []
    subject = dataset_id()
    table = rows()

    # -- pre-declared everything -------------------------------------------
    ledger = Ledger(runs_root / "ledger.md", tuple(cfg["ledger_events"]))
    ledger.record(subject, "gate", f"{len(table)} rows, {sum(r['anomaly'] for r in table)} labelled anomalies")
    budget = Budget.load(runs_root / "budget", subject, "first_pass",
                         total=cfg["budget"]["total"], costs=cfg["budget"]["costs"],
                         tracks=tuple(cfg["budget"]["tracks"]))
    ctx = init_run(RunConfig(phases=tuple(cfg["phases"]), loop=LoopConfig(**cfg["loop"])),
                   runs_root, run_id)
    machine = PhaseMachine(ctx.state.phase(PHASE), ctx.config.loop)
    ledger.record(subject, "run", f"{run_id}: budget {budget.total}, "
                                  f"max {ctx.config.loop.max_iterations} iterations")
    acceptance = cfg["acceptance"]
    violations: list[str] = []
    refused: list[str] = []
    measured: dict[str, dict] = {}

    # -- the phase ------------------------------------------------------------
    while True:
        directive = machine.next_action()
        save_state(ctx)
        if directive is Directive.PHASE_DONE:
            _say(trace, f"phase done: {machine.state.verdict.value}, best {machine.state.best_id} "
                        f"({machine.state.best})", quiet)
            break
        n = machine.state.iteration
        item = f"rule_{n:02d}"
        item_dir = ctx.root / "items" / item

        # proposer, under guard
        with Guard(ctx.root, MATRIX, "proposer", {"item": item}) as g:
            proposal = agents.propose(n, ctx.root, item_dir)
        if not g.result.ok:
            violations.extend(g.result.violations)
            _say(trace, f"[{n}] {item}: wrote outside its jurisdiction "
                        f"{g.result.violations}; restored, failure recorded", quiet)
            machine.record_failure(FailureKind.JURISDICTION)
            save_state(ctx)
            continue

        # budget arbitration, before any measurement
        ok, why = budget.check(proposal.distance)
        if not ok:
            refused.append(item)
            _say(trace, f"[{n}] {item}: refused before measurement — {why}", quiet)
            machine.record_failure(FailureKind.REFUSED)
            save_state(ctx)
            continue

        # auditor, under guard
        with Guard(ctx.root, MATRIX, "auditor", {"item": item}) as g:
            verdict, why = agents.audit(item_dir)
        if not g.result.ok:
            machine.record_failure(FailureKind.JURISDICTION)
            save_state(ctx)
            continue
        directive = machine.record_audit(Audit(verdict))
        save_state(ctx)
        if directive is not Directive.EXECUTE:
            _say(trace, f"[{n}] {item}: audit NO_GO — {why}", quiet)
            continue

        # deterministic measurement, outside any guard
        m = measure(proposal.rule, table)
        accepted = m.precision >= acceptance["precision_min"] and m.flagged >= acceptance["flagged_min"]
        (item_dir / "measure.json").write_text(json.dumps(m.to_dict(), indent=2))
        measured[item] = m.to_dict()
        try:
            trial = budget.consume(item, proposal.distance, track=proposal.track, accepted=accepted)
        except (BudgetExhausted, DriftRefused) as exc:      # cannot happen after check(), kept honest
            machine.record_failure(FailureKind.REFUSED)
            save_state(ctx)
            _say(trace, f"[{n}] {item}: {exc}", quiet)
            continue
        machine.record_result(accepted, m.f1, item)
        save_state(ctx)
        _say(trace, f"[{n}] {item}: {proposal.rule.column} {proposal.rule.op} {proposal.rule.value} "
                    f"({proposal.distance}, cost {trial.cost}) → flagged {m.flagged}, "
                    f"precision {m.precision}, recall {m.recall}, F1 {m.f1} → "
                    f"{'accepted' if accepted else 'rejected'}", quiet)

    budget.commit(runs_root / "budget")
    _say(trace, "budget: " + budget.summary(), quiet)

    # -- the deliverable, written by the harness under contract -----------------
    registry = example_registry()
    report = ProposalReport.build([
        ProposalItem(id=i, score=measured[i]["f1"], accepted=i in machine.state.accepted_ids)
        for i in measured])
    report_path = registry.dump(report, ctx.root / "report.json")
    ledger.record(subject, "report", f"{len(report.items)} rules measured, "
                                     f"{report.summary.accepted} accepted")

    # -- the boundary: reviewer, retry cap, ledger -----------------------------
    review = Review(BoundaryState(boundary=BOUNDARY), cfg["review"]["max_retries"],
                    budget=budget, ledger=ledger, subject=subject)
    decisions: list[str] = []
    while True:
        loaded = registry.load(report_path)                 # the reviewer reads through the contract
        with Guard(ctx.root, MATRIX, "reviewer") as g:
            decision, findings, redo, reason = agents.review(loaded, ctx.root)
        outcome = review.decide(decision, findings, redo, reason)
        decisions.append(outcome.decision.value)
        _say(trace, f"review {BOUNDARY}: {outcome.decision.value}"
                    + (f" (retry {outcome.retry_number}/{outcome.max_retries})"
                       if outcome.retry_number else "")
                    + f" — {outcome.reason}; findings: "
                    + (", ".join(f"{f.direction.value} {f.claim}" for f in findings) or "none"),
             quiet)
        if outcome.decision is not Decision.RETRY:
            break
        # the retry: the proposer redoes what was asked, under guard
        for f in findings:
            item = f.claim.split()[0]
            with Guard(ctx.root, MATRIX, "proposer", {"item": item}):
                agents.fix_rationale(ctx.root / "items" / item, measured[item])

    (ctx.root / "review").mkdir(exist_ok=True)
    (ctx.root / "review" / "boundary.json").write_text(review.state.model_dump_json(indent=2))
    _say(trace, f"ledger: {ledger.total_lines()} lines, {ledger.counts(subject)}", quiet)

    return {
        "run_root": str(ctx.root),
        "verdict": machine.state.verdict.value,
        "best_id": machine.state.best_id,
        "best": machine.state.best,
        "iterations": machine.state.iteration,
        "accepted": list(machine.state.accepted_ids),
        "rejected": list(machine.state.rejected_ids),
        "violations": violations,
        "refused": refused,
        "budget_spent": budget.spent_this_run(),
        "budget_trials": len(budget.trials),
        "review_decisions": decisions,
        "retries": review.state.retries,
        "ledger_lines": ledger.total_lines(),
        "ledger_counts": ledger.counts(subject),
        "trace": trace,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs-root", type=Path, default=HERE / "runs")
    parser.add_argument("--run-id", default=None, help="default: a UTC timestamp")
    args = parser.parse_args(argv)
    from agent_harness.state_machine import new_run_id
    summary = run(args.runs_root, args.run_id or new_run_id())
    print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
