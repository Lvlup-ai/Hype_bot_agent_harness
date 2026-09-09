"""Append-only event ledger with a closed vocabulary.

Why this module exists
----------------------
A run measures many hypotheses and corrects for that. But a *fleet* of runs,
on many subjects, selects "the best of N" in a way no per-run statistic can
see. The ledger is the counter that makes this visible: one line per event,
never edited, never reordered, readable by a human without any tool.

Two lessons from the original pipeline shaped the reading side:

* the vocabulary must be closed, otherwise agents invent event names and the
  counts stop meaning anything;
* counting only the *known* events under-counts the ledger. A report once
  announced 19 lines for a file that held 32, because older entries used
  names that had since been dropped from the list. ``counts()`` and
  ``total_lines()`` are therefore two different questions.

Format
------
A Markdown table, one row per event::

    | Date (UTC) | Subject | Event | Note |
    |---|---|---|---|
    | 2026-01-01 10:00 UTC | `abc123` | run | first pass |
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["Ledger", "UnknownEvent"]

_HEADER = (
    "# Ledger (append-only)\n\n"
    "> Every subject that goes through the pipeline is one draw. Selecting the\n"
    "> best of N subjects is a bias that no per-run statistic can see: read every\n"
    "> result in the light of the number of lines below.\n\n"
    "| Date (UTC) | Subject | Event | Note |\n|---|---|---|---|\n"
)


class UnknownEvent(ValueError):
    """The event is not in the ledger's vocabulary. Nothing was written."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Ledger:
    """A ledger file and the events it accepts.

    ``events`` is the closed vocabulary. It is the caller's declaration, made
    once per project; the ledger never learns a new name from a line.
    """

    path: Path
    events: tuple[str, ...]
    clock: Callable[[], datetime] = field(default=_utc_now, repr=False)

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("a ledger needs at least one event name")
        self.events = tuple(self.events)

    # -- writing -------------------------------------------------------------

    def record(self, subject: str, event: str, note: str = "") -> str:
        """Append one line. Returns the line written. Never rewrites anything."""
        if event not in self.events:
            raise UnknownEvent(f"event {event!r} is not in {self.events}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(_HEADER, encoding="utf-8")
        stamp = self.clock().astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        cell = (note or "—").replace("|", "\\|").replace("\n", " ")
        line = f"| {stamp} | `{subject}` | {event} | {cell} |\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)
        return line

    # -- reading -------------------------------------------------------------

    def _rows(self) -> list[str]:
        if not self.path.exists():
            return []
        return [l for l in self.path.read_text(encoding="utf-8").splitlines()
                if l.startswith("| ") and " UTC |" in l]

    def counts(self, subject: str | None = None) -> dict[str, int]:
        """Lines per *known* event, optionally for one subject.

        This is not the number of lines in the file: see ``total_lines()``.
        """
        out = {e: 0 for e in self.events}
        for row in self._rows():
            if subject is not None and f"| `{subject}` |" not in row:
                continue
            for e in self.events:
                if f"| {e} |" in row:
                    out[e] += 1
                    break
        return out

    def total_lines(self, subject: str | None = None) -> int:
        """Every event line, known vocabulary or not. The honest size of the ledger."""
        rows = self._rows()
        if subject is not None:
            rows = [r for r in rows if f"| `{subject}` |" in r]
        return len(rows)

    def subjects(self) -> tuple[str, ...]:
        """Distinct subjects, in order of first appearance."""
        seen: list[str] = []
        for row in self._rows():
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            if len(cells) >= 2:
                s = cells[1].strip("`")
                if s not in seen:
                    seen.append(s)
        return tuple(seen)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_harness.ledger",
        description="Append to, or read, an append-only event ledger.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--path", required=True, type=Path)
    common.add_argument("--events", nargs="+", required=True, help="the closed vocabulary")

    rec = sub.add_parser("record", parents=[common])
    rec.add_argument("--subject", required=True)
    rec.add_argument("--event", required=True)
    rec.add_argument("--note", default="")

    cnt = sub.add_parser("counts", parents=[common])
    cnt.add_argument("--subject", default=None)

    args = parser.parse_args(argv)
    ledger = Ledger(args.path, tuple(args.events))
    if args.cmd == "record":
        try:
            ledger.record(args.subject, args.event, args.note)
        except UnknownEvent as exc:
            print(json.dumps({"error": str(exc)}))
            return 2
        print(json.dumps({"ok": True, "total_lines": ledger.total_lines()}))
        return 0
    print(json.dumps({"counts": ledger.counts(args.subject),
                      "total_lines": ledger.total_lines(args.subject)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
