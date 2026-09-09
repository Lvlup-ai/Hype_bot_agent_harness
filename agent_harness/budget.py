"""Persistent trial budgets: every measured hypothesis costs something.

Why this module exists
----------------------
When no data is hidden from the agents, the only thing that bounds a search
is what it costs. Each hypothesis actually measured consumes budget; when the
budget is gone, the harness refuses. Four mechanisms, each born from an
incident in the original pipeline:

1. **Cost by distance.** A trial inside the question asked costs 1, a
   neighbouring one 2, an off-topic one 3. Drifting away from the brief is
   allowed, never free.
2. **Minimum in scope.** Before any off-topic trial, every declared track
   must have received at least one in-scope trial. Without this, an agent
   could leave the question on its first trial without ever trying it.
3. **Persistence per subject.** The cumulative cost is written to disk and
   reloaded on the next run. The pipeline's counter used to restart at zero
   on every execution, so "a retry costs trials" cost nothing at all, and an
   audit escalated it.
4. **The reason for the pass.** ``first_pass`` and ``retry`` consume;
   ``rerun`` does not. Strict persistence was rejected on a measurement: four
   technical re-runs in one night, one of them an import error, would have
   consumed 72 trials out of 36. A tooling error must not cost research budget.

Compartments
------------
A budget may be split into named compartments (one per family of ideas, say),
each with its own cap. Without compartments there is one global cap. Distance
costs apply in both cases.

What is left out on purpose: any statistic that corrects results for the
number of trials. ``trials_total()`` exposes the number so you can plug in
your own.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Budget",
    "BudgetExhausted",
    "DEFAULT_COSTS",
    "DriftRefused",
    "Trial",
    "CONSUMING_REASONS",
    "REASONS",
]

DEFAULT_COSTS: dict[str, int] = {"in_scope": 1, "adjacent": 2, "off_topic": 3}
REASONS: tuple[str, ...] = ("first_pass", "retry", "rerun")
CONSUMING_REASONS: frozenset[str] = frozenset({"first_pass", "retry"})
_NO_COMPARTMENT = "_"


class BudgetExhausted(RuntimeError):
    """The budget is spent: the search stops here."""


class DriftRefused(RuntimeError):
    """The harness refuses this distance from the question, and says why."""


@dataclass(frozen=True)
class Trial:
    """One measured hypothesis: the trace that makes the counter auditable."""

    n: int
    name: str
    distance: str
    cost: int
    compartment: str | None = None
    track: str | None = None
    accepted: bool | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Budget:
    """A trial budget for one subject, held by the orchestrator."""

    total: int
    costs: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    in_scope: str = "in_scope"
    off_topic: str = "off_topic"
    min_in_scope: int = 0
    tracks: tuple[str, ...] = ()
    compartments: Mapping[str, int] = field(default_factory=dict)
    subject: str = ""
    reason: str = "first_pass"
    # Spent on this subject before this run, per compartment ("_" without one).
    prior: dict[str, int] = field(default_factory=dict)
    trials: list[Trial] = field(default_factory=list)
    # Hypotheses drawn on this subject in earlier runs (a count, not a cost).
    prior_trials: int = 0

    def __post_init__(self) -> None:
        if self.total < 1:
            raise ValueError("a budget below 1 trial would mean no research at all")
        if self.reason not in REASONS:
            raise ValueError(f"reason {self.reason!r} is not one of {REASONS}")
        if self.in_scope not in self.costs or self.off_topic not in self.costs:
            raise ValueError("costs must define the in-scope and off-topic distances")
        for c, cap in self.compartments.items():
            if cap < 1:
                raise ValueError(f"compartment {c!r} has a cap below 1")

    # -- persistence ---------------------------------------------------------

    @staticmethod
    def _file(store: Path, subject: str) -> Path:
        return store / f"{subject}.json"

    @classmethod
    def load(cls, store: Path, subject: str, reason: str, **kwargs) -> "Budget":
        """A budget that knows what this subject already cost.

        The prior is loaded whatever the reason: a ``rerun`` must be able to
        report the cumulative cost even though it adds nothing to it.
        """
        b = cls(subject=subject, reason=reason, **kwargs)
        f = cls._file(store, subject)
        if f.exists():
            raw = json.loads(f.read_text(encoding="utf-8"))
            b.prior = {str(k): int(v) for k, v in raw.get("spent", {}).items()}
            b.prior_trials = int(raw.get("trials_total", 0))
        return b

    def commit(self, store: Path) -> bool:
        """Add this run's spending to the subject's file. No effect on a ``rerun``."""
        if not (self.subject and self.consumes):
            return False
        spent = dict(self.prior)
        for t in self.trials:
            key = t.compartment or _NO_COMPARTMENT
            spent[key] = spent.get(key, 0) + t.cost
        store.mkdir(parents=True, exist_ok=True)
        self._file(store, self.subject).write_text(
            json.dumps({"subject": self.subject, "spent": spent,
                        "trials_total": self.trials_total()}, indent=2, sort_keys=True),
            encoding="utf-8")
        return True

    # -- reading -------------------------------------------------------------

    @property
    def consumes(self) -> bool:
        """Does this pass draw new hypotheses?"""
        return self.reason in CONSUMING_REASONS

    def cap(self, compartment: str | None = None) -> int:
        if compartment is None:
            return self.total
        return self.compartments.get(compartment, self.total)

    def spent_this_run(self, compartment: str | None = None) -> int:
        return sum(t.cost for t in self.trials
                   if compartment is None or t.compartment == compartment)

    def spent_before(self, compartment: str | None = None) -> int:
        if compartment is None:
            return sum(self.prior.values())
        return self.prior.get(compartment, 0)

    def cumulative(self, compartment: str | None = None) -> int:
        """What the subject has cost so far, this run included when it consumes."""
        this = self.spent_this_run(compartment) if (self.consumes or not self.subject) else 0
        return self.spent_before(compartment) + this

    def remaining(self, compartment: str | None = None) -> int:
        """Prior spending counts: a second run on a subject does not get a fresh budget."""
        cap = self.cap(compartment)
        used = self.spent_before(compartment) + self.spent_this_run(compartment)
        if compartment is not None:
            # The global cap binds as well: compartments never sum past it.
            return max(0, min(cap - used, self.total - self.spent_before() - self.spent_this_run()))
        return max(0, cap - used)

    def trials_total(self) -> int:
        """Trials drawn on this subject so far, for whoever corrects for them.

        Prior *cost* is not converted into a count: the file records both, and
        a ``rerun`` re-measures hypotheses that were already counted.
        """
        return self.prior_trials + (len(self.trials) if self.consumes else 0)

    def tracks_without_trial(self) -> tuple[str, ...]:
        served = {t.track for t in self.trials if t.distance == self.in_scope and t.track}
        return tuple(tr for tr in self.tracks if tr not in served)

    def n_in_scope(self) -> int:
        return sum(1 for t in self.trials if t.distance == self.in_scope)

    # -- arbitration ---------------------------------------------------------

    def check(self, distance: str, compartment: str | None = None) -> tuple[bool, str]:
        """May the next trial be spent? Always returns a reason, either way."""
        if distance not in self.costs:
            return False, f"unknown distance {distance!r}; known: {tuple(self.costs)}"
        if compartment is not None and self.compartments and compartment not in self.compartments:
            return False, f"unknown compartment {compartment!r}; known: {tuple(self.compartments)}"
        cost = self.costs[distance]
        left = self.remaining(compartment)
        if left < cost:
            where = f" in {compartment!r}" if compartment else ""
            return False, f"budget exhausted{where}: {left} left for a {distance} trial costing {cost}"
        if distance == self.off_topic:
            missing = self.tracks_without_trial()
            if missing:
                return False, (f"{len(missing)} declared track(s) have no in-scope trial yet: "
                               f"{', '.join(missing)}. The question asked has not been answered")
            if not self.tracks and self.n_in_scope() < self.min_in_scope:
                return False, (f"only {self.n_in_scope()} in-scope trial(s) of {self.min_in_scope} "
                               "required before drifting off topic")
        return True, f"allowed: cost {cost}, {left - cost} left after"

    def consume(self, name: str, distance: str, compartment: str | None = None,
                track: str | None = None, accepted: bool | None = None,
                note: str = "") -> Trial:
        """Record a measured trial. Raises if the arbitration refuses it."""
        ok, why = self.check(distance, compartment)
        if not ok:
            cost = self.costs.get(distance, 0)
            exhausted = distance in self.costs and self.remaining(compartment) < cost
            raise (BudgetExhausted if exhausted else DriftRefused)(why)
        t = Trial(n=len(self.trials) + 1, name=name, distance=distance,
                  cost=self.costs[distance], compartment=compartment, track=track,
                  accepted=accepted, note=note)
        self.trials.append(t)
        return t

    # -- reporting -----------------------------------------------------------

    def summary(self) -> str:
        """The one line a report must carry: what the search cost, visibly."""
        by_distance = {d: sum(1 for t in self.trials if t.distance == d) for d in self.costs}
        dist = ", ".join(f"{d} {n}" for d, n in by_distance.items())
        line = (f"{len(self.trials)} trials measured, {self.spent_this_run()}/{self.total} of "
                f"budget spent this run ({dist}), {self.remaining()} left")
        if self.subject and self.spent_before():
            line += f"; {self.cumulative()}/{self.total} on subject `{self.subject}` across runs"
        if not self.consumes:
            line += " [rerun: nothing added]"
        if self.compartments:
            parts = []
            for c in self.compartments:
                parts.append(f"{c} {self.spent_before(c) + self.spent_this_run(c)}/{self.cap(c)}")
            line += " · " + ", ".join(parts)
        return line

