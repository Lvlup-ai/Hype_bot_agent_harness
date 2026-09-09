"""The deterministic orchestrator: an explicit state machine per research phase.

Why this module exists
----------------------
In the pipeline this harness comes from, the flow of a run lived in a runbook
prompt: "retry at most 3 times", "stop when the best result stops improving",
"only these verdicts are valid". Every one of those rules was broken at least
once, silently, by an orchestrating LLM that summarised its context, chose a
convenient verdict, or shortened a run to save tokens. The rules were right;
they just were not enforced by anything.

Here the flow is a table. The orchestrating LLM asks for the next action,
performs it (invokes an agent), and records what happened. It cannot skip a
step, record an event the machine is not waiting for, or set a verdict: those
are ``TransitionError``s, and the state file is written by this module alone.

The loop, as driven by an orchestrator (LLM or script)::

    machine = PhaseMachine(phase_state, config)
    while True:
        d = machine.next_action()
        if d is Directive.PHASE_DONE: break
        # d is PROPOSE: invoke the proposer (under a jurisdiction guard)
        d = machine.record_audit(Audit.GO)      # or NO_GO, or record_failure(...)
        if d is Directive.EXECUTE:
            # run the deterministic measurement
            d = machine.record_result(accepted=True, score=..., result_id=...)
        # d is NEXT_ITERATION or PHASE_DONE; loop

Directives are what the orchestrator must do next; events are what it reports
back. The allowed (state, event) pairs are in ``TRANSITIONS``. Everything else
is forbidden.

Verdicts are a closed vocabulary (``Verdict``) and are set only by the
machine: convergence, the iteration cap, or a phase that ended without an
acceptable result.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "Audit",
    "Directive",
    "FailureKind",
    "LoopConfig",
    "PhaseMachine",
    "PhaseState",
    "RunConfig",
    "RunContext",
    "RunExistsError",
    "RunNotFoundError",
    "RunState",
    "Step",
    "TRANSITIONS",
    "TransitionError",
    "Verdict",
    "has_converged",
    "init_run",
    "load_run",
    "save_state",
]


# ── Vocabulary ───────────────────────────────────────────────────────────────

class Directive(str, Enum):
    """What the orchestrator must do next."""

    PROPOSE = "propose"                # invoke the proposing agent (new attempt or retry)
    EXECUTE = "execute"                # the proposal passed audit: measure it
    NEXT_ITERATION = "next_iteration"  # this iteration is over, ask again
    PHASE_DONE = "phase_done"          # the phase has a verdict; stop asking


class Audit(str, Enum):
    GO = "GO"
    NO_GO = "NO_GO"


class FailureKind(str, Enum):
    RUNTIME_ERROR = "runtime_error"    # the proposal crashed when measured
    JURISDICTION = "jurisdiction"      # the agent wrote outside its jurisdiction
    INVALID_OUTPUT = "invalid_output"  # the deliverable failed its contract


class Verdict(str, Enum):
    """How a phase ended. Nothing else is a verdict."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    NO_VIABLE_RESULT = "no_viable_result"


class Step(str, Enum):
    """Where the machine is, i.e. which events it will accept."""

    IDLE = "idle"              # between iterations: waiting for next_action
    PROPOSING = "proposing"    # a proposer is at work: waiting for audit or failure
    EXECUTING = "executing"    # a measurement is running: waiting for result or failure
    DONE = "done"              # verdict set: accepts nothing but next_action


class Event(str, Enum):
    NEXT_ACTION = "next_action"
    AUDIT = "audit"
    FAILURE = "failure"
    RESULT = "result"


# The whole protocol, in one table. A pair absent from it is a forbidden
# transition. The target step after AUDIT depends on the verdict, so the
# table lists the step the event is *accepted in*; handlers pick the target.
TRANSITIONS: frozenset[tuple[Step, Event]] = frozenset({
    (Step.IDLE, Event.NEXT_ACTION),
    (Step.DONE, Event.NEXT_ACTION),
    (Step.PROPOSING, Event.AUDIT),
    (Step.PROPOSING, Event.FAILURE),
    (Step.EXECUTING, Event.RESULT),
    (Step.EXECUTING, Event.FAILURE),
})


class TransitionError(RuntimeError):
    """The orchestrator reported an event the machine was not waiting for."""


# ── State ────────────────────────────────────────────────────────────────────

class LoopConfig(BaseModel):
    """Pre-declared limits of a phase. Read from the run's config, never changed mid-run."""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(20, ge=1)
    max_failures: int = Field(5, ge=1, description="consecutive NO_GO/failures before the iteration is abandoned")
    convergence_window: int = Field(10, ge=1)
    convergence_threshold_pct: float = Field(2.0, ge=0.0)
    # Early stop on convergence only when the best is >= 0: a run whose best is
    # still negative has not found anything yet, so "not improving" means
    # "keep looking", not "done".
    stop_only_when_best_nonnegative: bool = True


class PhaseState(BaseModel):
    """Mutable state of one phase. Written only through ``PhaseMachine``."""

    phase: str
    step: Step = Step.IDLE
    iteration: int = 0
    consecutive_failures: int = 0
    best_history: list[float] = Field(default_factory=list)
    best_id: str | None = None
    verdict: Verdict | None = None
    accepted_ids: list[str] = Field(default_factory=list)
    rejected_ids: list[str] = Field(default_factory=list)

    @property
    def best(self) -> float | None:
        return self.best_history[-1] if self.best_history else None

    @property
    def inconclusive(self) -> bool:
        return self.verdict is Verdict.NO_VIABLE_RESULT


# ── Convergence ──────────────────────────────────────────────────────────────

def _relative_gain_pct(ref: float, cur: float) -> float:
    if ref == 0.0:
        return 0.0 if cur == ref else math.copysign(math.inf, cur - ref)
    return (cur - ref) / abs(ref) * 100.0


def has_converged(best_history: list[float], window: int, threshold_pct: float) -> bool:
    """The running best gained at most ``threshold_pct`` over the last ``window`` iterations.

    Needs ``window + 1`` points: with fewer, nothing can be concluded and the
    answer is "not converged".
    """
    if len(best_history) < window + 1:
        return False
    return _relative_gain_pct(best_history[-(window + 1)], best_history[-1]) <= threshold_pct


# ── Machine ──────────────────────────────────────────────────────────────────

class PhaseMachine:
    """Applies events to a ``PhaseState`` under the transition table."""

    def __init__(self, state: PhaseState, config: LoopConfig) -> None:
        self.state = state
        self.config = config

    # -- guard ---------------------------------------------------------------

    def _accept(self, event: Event) -> None:
        if (self.state.step, event) not in TRANSITIONS:
            raise TransitionError(
                f"phase {self.state.phase!r}: event {event.value!r} is not allowed "
                f"in step {self.state.step.value!r}")

    # -- events --------------------------------------------------------------

    def next_action(self) -> Directive:
        """Open an iteration, or say the phase is done."""
        self._accept(Event.NEXT_ACTION)
        if self.state.verdict is not None:
            return Directive.PHASE_DONE
        if self.state.iteration >= self.config.max_iterations:
            self._finish_at_cap()
            return Directive.PHASE_DONE
        self.state.iteration += 1
        self.state.consecutive_failures = 0
        self.state.step = Step.PROPOSING
        return Directive.PROPOSE

    def record_audit(self, verdict: Audit) -> Directive:
        """GO → measure; NO_GO → retry, or abandon the iteration at the cap."""
        self._accept(Event.AUDIT)
        if verdict is Audit.GO:
            self.state.step = Step.EXECUTING
            return Directive.EXECUTE
        return self._register_failure()

    def record_failure(self, kind: FailureKind) -> Directive:
        """A crash, a jurisdiction violation or an invalid deliverable counts like a NO_GO."""
        self._accept(Event.FAILURE)
        return self._register_failure()

    def record_result(self, accepted: bool, score: float, result_id: str) -> Directive:
        """A clean measurement. Only an *accepted* result can become the best.

        ``accepted`` is decided upstream by a deterministic gate, never by the
        orchestrator's judgement; ``score`` is the phase's ranking metric.
        """
        self._accept(Event.RESULT)
        st = self.state
        st.consecutive_failures = 0
        prev = st.best
        if accepted:
            st.accepted_ids.append(result_id)
            if prev is None or score > prev:
                st.best_id = result_id
                prev = score
        else:
            st.rejected_ids.append(result_id)
        if prev is not None:
            st.best_history.append(prev)

        cfg = self.config
        may_stop = prev is not None and (prev >= 0.0 or not cfg.stop_only_when_best_nonnegative)
        if may_stop and has_converged(st.best_history, cfg.convergence_window,
                                      cfg.convergence_threshold_pct):
            st.verdict = Verdict.CONVERGED
            st.step = Step.DONE
            return Directive.PHASE_DONE
        st.step = Step.IDLE
        return Directive.NEXT_ITERATION

    # -- internals -----------------------------------------------------------

    def _register_failure(self) -> Directive:
        st = self.state
        st.consecutive_failures += 1
        if st.consecutive_failures >= self.config.max_failures:
            st.step = Step.IDLE
            return Directive.NEXT_ITERATION
        st.step = Step.PROPOSING
        return Directive.PROPOSE

    def _finish_at_cap(self) -> None:
        st = self.state
        st.verdict = (Verdict.MAX_ITERATIONS if st.best is not None and st.best >= 0.0
                      else Verdict.NO_VIABLE_RESULT)
        st.step = Step.DONE


# ── Run persistence ──────────────────────────────────────────────────────────

class RunExistsError(RuntimeError):
    """A run with this id already exists; nothing is ever overwritten."""


class RunNotFoundError(RuntimeError):
    """No run with this id under the runs root."""


class RunConfig(BaseModel):
    """Everything pre-declared for a run. Frozen: a limit is not renegotiated mid-run."""

    model_config = ConfigDict(frozen=True)

    phases: tuple[str, ...]
    loop: LoopConfig = LoopConfig()
    extra: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RunState(BaseModel):
    run_id: str
    phases: dict[str, PhaseState] = Field(default_factory=dict)
    current_phase: str | None = None
    finished: bool = False
    last_save_utc: datetime | None = None

    def phase(self, name: str) -> PhaseState:
        ps = self.phases.get(name)
        if ps is None:
            ps = PhaseState(phase=name)
            self.phases[name] = ps
        return ps


class RunContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: Path
    config: RunConfig
    state: RunState

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"


def new_run_id(now: datetime | None = None) -> str:
    moment = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return moment.strftime("%Y-%m-%dT%H-%M-%S")


def init_run(config: RunConfig, runs_root: Path, run_id: str | None = None) -> RunContext:
    rid = run_id or new_run_id()
    root = runs_root / rid
    if root.exists():
        raise RunExistsError(f"run {rid!r} already exists at {root}")
    root.mkdir(parents=True)
    ctx = RunContext(root=root, config=config, state=RunState(run_id=rid))
    ctx.config_path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    save_state(ctx)
    return ctx


def load_run(run_id: str, runs_root: Path) -> RunContext:
    root = runs_root / run_id
    cfg, st = root / "config.json", root / "state.json"
    if not (cfg.exists() and st.exists()):
        raise RunNotFoundError(f"run {run_id!r} not found or incomplete at {root}")
    return RunContext(root=root,
                      config=RunConfig.model_validate_json(cfg.read_text(encoding="utf-8")),
                      state=RunState.model_validate_json(st.read_text(encoding="utf-8")))


def save_state(ctx: RunContext, now: datetime | None = None) -> None:
    """Atomic write: the state file is never seen half-written."""
    ctx.state.last_save_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    tmp = ctx.state_path.with_suffix(".json.tmp")
    tmp.write_text(ctx.state.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(ctx.state_path)


# ── CLI ──────────────────────────────────────────────────────────────────────
#
# One command = one event. Each loads the run, applies the event, saves the
# state atomically and prints a JSON payload. This is the only way the state
# file is meant to change once a run exists.

def _payload(ctx: RunContext, machine: PhaseMachine, directive: Directive) -> dict:
    st = machine.state
    return {
        "action": directive.value,
        "phase": st.phase,
        "iteration": st.iteration,
        "consecutive_failures": st.consecutive_failures,
        "best": st.best,
        "best_id": st.best_id,
        "verdict": st.verdict.value if st.verdict else None,
        "inconclusive": st.inconclusive,
        "run_finished": ctx.state.finished,
    }


def _bool(s: str) -> bool:
    if s.lower() in ("true", "yes", "1"):
        return True
    if s.lower() in ("false", "no", "0"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {s!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_harness.state_machine",
        description="Drive a run one event at a time. Prints JSON; exit 2 on a forbidden transition.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="create a run and print its id")
    p_init.add_argument("--runs-root", type=Path, default=Path("runs"))
    p_init.add_argument("--run-id", default=None)
    p_init.add_argument("--phases", nargs="+", required=True)
    p_init.add_argument("--max-iterations", type=int, default=20)
    p_init.add_argument("--max-failures", type=int, default=5)
    p_init.add_argument("--convergence-window", type=int, default=10)
    p_init.add_argument("--convergence-threshold-pct", type=float, default=2.0)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--runs-root", type=Path, default=Path("runs"))
    common.add_argument("--run-id", required=True)
    common.add_argument("--phase", required=True)

    sub.add_parser("next-action", parents=[common])
    p_aud = sub.add_parser("record-audit", parents=[common])
    p_aud.add_argument("--verdict", required=True, choices=[a.value for a in Audit])
    p_fail = sub.add_parser("record-failure", parents=[common])
    p_fail.add_argument("--kind", required=True, choices=[k.value for k in FailureKind])
    p_res = sub.add_parser("record-result", parents=[common])
    p_res.add_argument("--accepted", required=True, type=_bool)
    p_res.add_argument("--score", required=True, type=float)
    p_res.add_argument("--id", required=True, dest="result_id")

    args = parser.parse_args(argv)

    if args.cmd == "init":
        cfg = RunConfig(phases=tuple(args.phases), loop=LoopConfig(
            max_iterations=args.max_iterations, max_failures=args.max_failures,
            convergence_window=args.convergence_window,
            convergence_threshold_pct=args.convergence_threshold_pct))
        ctx = init_run(cfg, args.runs_root, args.run_id)
        print(json.dumps({"run_id": ctx.state.run_id, "root": str(ctx.root)}))
        return 0

    ctx = load_run(args.run_id, args.runs_root)
    if args.phase not in ctx.config.phases:
        print(json.dumps({"error": f"phase {args.phase!r} is not declared in this run"}))
        return 2
    machine = PhaseMachine(ctx.state.phase(args.phase), ctx.config.loop)
    try:
        if args.cmd == "next-action":
            ctx.state.current_phase = args.phase
            directive = machine.next_action()
        elif args.cmd == "record-audit":
            directive = machine.record_audit(Audit(args.verdict))
        elif args.cmd == "record-failure":
            directive = machine.record_failure(FailureKind(args.kind))
        else:
            directive = machine.record_result(args.accepted, args.score, args.result_id)
    except TransitionError as exc:
        print(json.dumps({"error": str(exc), "step": machine.state.step.value}))
        return 2  # nothing was saved: a forbidden event leaves no trace in the state

    if machine.state.verdict is not None and all(
            ctx.state.phases.get(p) is not None and ctx.state.phases[p].verdict is not None
            for p in ctx.config.phases):
        ctx.state.finished = True
    save_state(ctx)
    print(json.dumps(_payload(ctx, machine, directive)))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
