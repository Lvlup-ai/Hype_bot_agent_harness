"""A tiny rule language and its deterministic measurement.

A rule names one column, one operator and one value, and flags the rows the
condition rejects. Measured against the labelled anomalies it yields
precision, recall and F1. This is the "engine": it measures, it never judges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OPERATORS: tuple[str, ...] = ("range", "contains", "not_empty", "in", "not_in")


class RuleError(ValueError):
    """The rule is malformed: unknown column, unknown operator, or a value of the wrong shape."""


@dataclass(frozen=True)
class Rule:
    column: str
    op: str
    value: Any = None

    def validate(self, columns: tuple[str, ...]) -> None:
        if self.column not in columns:
            raise RuleError(f"unknown column {self.column!r}; known: {columns}")
        if self.op not in OPERATORS:
            raise RuleError(f"unknown operator {self.op!r}; known: {OPERATORS}")
        if self.op == "range" and not (isinstance(self.value, (list, tuple)) and len(self.value) == 2):
            raise RuleError("range needs a [low, high] pair")
        if self.op in ("in", "not_in") and not isinstance(self.value, (list, tuple, set)):
            raise RuleError(f"{self.op} needs a list of allowed values")
        if self.op == "contains" and not isinstance(self.value, str):
            raise RuleError("contains needs a string")

    def rejects(self, row: dict) -> bool:
        """True when the row breaks the rule, i.e. is flagged."""
        v = row[self.column]
        if self.op == "range":
            low, high = self.value
            return not (low <= v <= high)
        if self.op == "contains":
            return self.value not in str(v)
        if self.op == "not_empty":
            return v is None or str(v).strip() == ""
        if self.op == "in":
            return v not in self.value
        if self.op == "not_in":
            return v in self.value
        raise RuleError(f"unknown operator {self.op!r}")

    def to_dict(self) -> dict:
        return {"column": self.column, "op": self.op,
                "value": list(self.value) if isinstance(self.value, (tuple, set)) else self.value}

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        return cls(column=str(d["column"]), op=str(d["op"]), value=d.get("value"))


@dataclass(frozen=True)
class Measure:
    flagged: int
    true_positives: int
    precision: float
    recall: float
    f1: float
    flagged_ids: tuple[int, ...]

    def to_dict(self) -> dict:
        return {"flagged": self.flagged, "true_positives": self.true_positives,
                "precision": self.precision, "recall": self.recall, "f1": self.f1,
                "flagged_ids": list(self.flagged_ids)}


def measure(rule: Rule, rows: list[dict]) -> Measure:
    """Apply the rule and score it against the ``anomaly`` labels."""
    flagged = [r for r in rows if rule.rejects(r)]
    tp = sum(1 for r in flagged if r["anomaly"])
    n_anomalies = sum(1 for r in rows if r["anomaly"])
    precision = tp / len(flagged) if flagged else 0.0
    recall = tp / n_anomalies if n_anomalies else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return Measure(len(flagged), tp, round(precision, 4), round(recall, 4), round(f1, 4),
                   tuple(r["id"] for r in flagged))
