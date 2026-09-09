"""Versioned deliverable contracts with a validating loader.

Why this module exists
----------------------
One agent writes a file, another reads it. The loader is the interface
between the two, and it must verify what it reads. In the original pipeline
a specification said for a month that block B "reads the report of block A",
and no component could actually read it: no loader, no schema, no boundary
test. When one was written, it caught truncated files, reports whose summary
contradicted their own rows, and a JSON round trip that had turned NaN into
null and integer keys into strings.

What the loader refuses, in order
---------------------------------
1. A missing or unreadable file (bad JSON, with the position of the error).
2. A deliverable with no ``contract`` field, or one the registry does not know.
3. A schema violation. Models are strict: an unknown field is an error, never
   a silence (a typo in a field name must not pass).
4. An internal inconsistency: a registered check recomputes something from
   the rows and compares it with what the summary announces.

How to use it
-------------
Declare a model per contract, register it with its checks, then load::

    class Report(Deliverable):
        items: tuple[Item, ...]
        summary: Summary

    registry = Registry()
    registry.register("my-report/1", Report, checks=[count_matches_items])
    report = registry.load(path)          # raises ContractError on any refusal

``FloatOrNaN`` is a float that survives JSON: ``NaN`` is written as ``null``
and read back as ``NaN``. Integer dictionary keys survive as well.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, ValidationError

__all__ = [
    "Check",
    "ContractError",
    "Deliverable",
    "DeliverableMissing",
    "DeliverableUnreadable",
    "FloatOrNaN",
    "InconsistentDeliverable",
    "Registry",
    "SchemaViolation",
    "UnknownContract",
    "ProposalReport",
    "example_registry",
]


# ── Errors ───────────────────────────────────────────────────────────────────

class ContractError(ValueError):
    """The deliverable was refused. The message says why."""


class DeliverableMissing(ContractError):
    pass


class DeliverableUnreadable(ContractError):
    pass


class UnknownContract(ContractError):
    pass


class SchemaViolation(ContractError):
    pass


class InconsistentDeliverable(ContractError):
    pass


# ── Base model and helpers ───────────────────────────────────────────────────

def _null_to_nan(v: Any) -> Any:
    return math.nan if v is None else v


FloatOrNaN = Annotated[float, BeforeValidator(_null_to_nan)]
"""A float whose NaN survives the JSON round trip (written as null)."""


class Deliverable(BaseModel):
    """Base of every contract model. Strict and immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: str


Check = Callable[[Deliverable], str | None]
"""A consistency check: returns a message when the deliverable contradicts itself."""


def _contract_field(raw: Any) -> str:
    if not isinstance(raw, dict):
        raise SchemaViolation(f"a deliverable is a JSON object, got {type(raw).__name__}")
    contract = raw.get("contract")
    if not isinstance(contract, str) or "/" not in contract:
        raise UnknownContract(
            "missing or malformed `contract` field; expected \"name/version\"")
    return contract


def _finite_or_null(obj: Any) -> Any:
    """Prepare a structure for strict JSON: non-finite floats become null."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _finite_or_null(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_null(v) for v in obj]
    return obj


# ── Registry ─────────────────────────────────────────────────────────────────

class Registry:
    """The contracts a reader supports, each with its model and its checks."""

    def __init__(self) -> None:
        self._models: dict[str, type[Deliverable]] = {}
        self._checks: dict[str, tuple[Check, ...]] = {}

    def register(self, contract: str, model: type[Deliverable],
                 checks: Sequence[Check] = ()) -> None:
        if "/" not in contract:
            raise ValueError(f"contract name {contract!r} must be \"name/version\"")
        if not (isinstance(model, type) and issubclass(model, Deliverable)):
            raise TypeError("model must subclass Deliverable")
        self._models[contract] = model
        self._checks[contract] = tuple(checks)

    @property
    def contracts(self) -> tuple[str, ...]:
        return tuple(self._models)

    # -- reading -------------------------------------------------------------

    def loads(self, text: str, origin: str = "<text>") -> Deliverable:
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise DeliverableUnreadable(
                f"{origin}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}"
            ) from exc
        contract = _contract_field(raw)
        model = self._models.get(contract)
        if model is None:
            raise UnknownContract(
                f"{origin}: contract {contract!r} is not supported; known: {self.contracts}")
        try:
            obj = model.model_validate(raw)
        except ValidationError as exc:
            raise SchemaViolation(f"{origin}: {contract}: {exc}") from exc
        for check in self._checks[contract]:
            problem = check(obj)
            if problem:
                raise InconsistentDeliverable(f"{origin}: {contract}: {problem}")
        return obj

    def load(self, path: Path) -> Deliverable:
        if not path.exists():
            raise DeliverableMissing(f"deliverable not found: {path}")
        return self.loads(path.read_text(encoding="utf-8"), origin=str(path))

    # -- writing -------------------------------------------------------------

    def dumps(self, obj: Deliverable) -> str:
        """Strict JSON: NaN becomes null, nothing non-standard is written.

        The object is validated against its own contract's checks first, so a
        writer cannot emit what a reader would refuse.
        """
        if obj.contract not in self._models:
            raise UnknownContract(f"contract {obj.contract!r} is not registered")
        for check in self._checks[obj.contract]:
            problem = check(obj)
            if problem:
                raise InconsistentDeliverable(f"{obj.contract}: {problem}")
        payload = _finite_or_null(obj.model_dump(mode="json"))
        return json.dumps(payload, indent=2, allow_nan=False, ensure_ascii=False)

    def dump(self, obj: Deliverable, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.dumps(obj) + "\n", encoding="utf-8")
        return path


# ── A worked contract ────────────────────────────────────────────────────────
#
# Enough to show the pattern: rows, a summary derived from the rows, and a
# check that refuses a summary the rows do not support.

class ProposalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    score: FloatOrNaN
    accepted: bool
    scores_by_horizon: dict[int, FloatOrNaN] = {}


class ProposalSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    accepted: int
    best_score: FloatOrNaN


class ProposalReport(Deliverable):
    """``proposal-report/1``: the items a proposer produced and what became of them."""

    items: tuple[ProposalItem, ...]
    summary: ProposalSummary
    in_sample: bool = True   # the reader must be told the numbers are not held out

    @classmethod
    def build(cls, items: Sequence[ProposalItem]) -> "ProposalReport":
        finite = [i.score for i in items if math.isfinite(i.score)]
        return cls(contract="proposal-report/1", items=tuple(items),
                   summary=ProposalSummary(count=len(items),
                                           accepted=sum(1 for i in items if i.accepted),
                                           best_score=max(finite) if finite else math.nan))


def _count_matches_items(d: Deliverable) -> str | None:
    assert isinstance(d, ProposalReport)
    if d.summary.count != len(d.items):
        return f"summary.count is {d.summary.count} but there are {len(d.items)} items"
    n = sum(1 for i in d.items if i.accepted)
    if d.summary.accepted != n:
        return f"summary.accepted is {d.summary.accepted} but {n} items are accepted"
    return None


def _best_score_is_the_max(d: Deliverable) -> str | None:
    assert isinstance(d, ProposalReport)
    finite = [i.score for i in d.items if math.isfinite(i.score)]
    expected = max(finite) if finite else math.nan
    got = d.summary.best_score
    same = (math.isnan(expected) and math.isnan(got)) or expected == got
    return None if same else f"summary.best_score is {got} but the best item scores {expected}"


def example_registry() -> Registry:
    r = Registry()
    r.register("proposal-report/1", ProposalReport,
               checks=[_count_matches_items, _best_score_is_the_max])
    return r


# ── CLI ──────────────────────────────────────────────────────────────────────

def _import_registry(spec: str) -> Registry:
    """``package.module:attribute`` naming a Registry or a callable returning one."""
    mod_name, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(mod_name), attr or "registry")
    reg = obj() if callable(obj) and not isinstance(obj, Registry) else obj
    if not isinstance(reg, Registry):
        raise TypeError(f"{spec} is not a Registry")
    return reg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_harness.contracts",
        description="Validate a deliverable against a registry. Prints JSON; exit 2 on refusal.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--registry", default="agent_harness.contracts:example_registry",
                        help="module:attribute of a Registry or a factory")
    args = parser.parse_args(argv)
    registry = _import_registry(args.registry)
    try:
        obj = registry.load(args.path)
    except ContractError as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "detail": str(exc)}))
        return 2
    print(json.dumps({"ok": True, "contract": obj.contract}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    # `python -m` loads this file as `__main__`; a registry imported by name
    # would then carry a second copy of the classes above, and `except
    # ContractError` would miss it. Route through the canonical module.
    from agent_harness.contracts import main as _main

    sys.exit(_main())
