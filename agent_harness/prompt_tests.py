"""Tests that check an agent brief tells the truth about the repository.

Why this module exists
----------------------
A prompt is a Markdown document that nothing compiles and nothing tests. It
describes the repository: constants, counts, thresholds, forbidden inputs,
field names. The repository moves; the prompt stays. In the original
pipeline a brief announced eight derived methods when the code had ten, and
the two missing ones were exactly the instruments the research question
called central. An agent following its brief to the letter could not measure
what it was asked to. Nobody saw it for a month.

Six checks cover the class of drift that was observed:

1. ``must_name_all``   every element of a code collection is named in the brief;
2. ``must_state_count`` a count in the brief ("ten methods") equals a value in the code,
   and no wrong count is stated for the same noun;
3. ``must_quote``      a configuration value is quoted as it currently is ("1000 ms");
4. ``must_not_contain`` a phrase a fix made false never comes back;
5. ``must_contain``    an invariant the brief must carry is present;
6. ``must_carry_status`` every file of a set shows one of the known status markers.

Text is flattened before comparison (bold, inline code and line breaks removed,
case ignored): "the **ten**\\n methods" is the same sentence as "the ten
methods". A literal assertion would fail on formatting instead of substance.

Use it from pytest::

    brief = Brief("agents/proposer.md")
    brief.must_name_all(METHODS, label="methods")
    brief.must_state_count(len(METHODS), noun="methods")
    brief.must_quote(config["grid_ms"], suffix=" ms")

or from a YAML file run by ``python -m agent_harness.prompt_tests checks.yaml``.

What it does not do: judge whether a prompt is good. It checks that the prompt
is accurate about the repository. The rest is reading.
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Brief", "BriefDrift", "Report", "flatten", "must_carry_status",
           "resolve_ref", "run_checks"]


class BriefDrift(AssertionError):
    """The brief says something the repository contradicts."""


# ── Text normalisation ───────────────────────────────────────────────────────

def flatten(text: str) -> str:
    """The text without its Markdown dressing, whitespace collapsed, lower-cased."""
    text = text.replace("**", "").replace("`", "").replace("__", "")
    text = re.sub(r"(?<!\w)[*_](?=\w)|(?<=\w)[*_](?!\w)", "", text)  # *emphasis* / _emphasis_
    return re.sub(r"\s+", " ", text).strip().lower()


_EN = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
       8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
       14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
       19: "nineteen", 20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
       70: "seventy", 80: "eighty", 90: "ninety", 100: "hundred"}
_FR = {0: "zéro", 1: "un", 2: "deux", 3: "trois", 4: "quatre", 5: "cinq", 6: "six", 7: "sept",
       8: "huit", 9: "neuf", 10: "dix", 11: "onze", 12: "douze", 13: "treize", 14: "quatorze",
       15: "quinze", 16: "seize", 17: "dix-sept", 18: "dix-huit", 19: "dix-neuf", 20: "vingt",
       30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante", 70: "soixante-dix",
       80: "quatre-vingts", 90: "quatre-vingt-dix", 100: "cent"}
_WORD_TO_INT: dict[str, int] = {w: n for table in (_EN, _FR) for n, w in table.items()}
_WORD_TO_INT["une"] = 1


def _number_forms(n: int) -> list[str]:
    forms = [str(n)]
    for table in (_EN, _FR):
        if n in table:
            forms.append(table[n])
    if n == 1:
        forms.append("une")
    return forms


_NUMBER_TOKEN = r"(\d+|" + "|".join(re.escape(w) for w in sorted(_WORD_TO_INT, key=len, reverse=True)) + r")"


def _stated_counts(flat: str, noun: str) -> list[int]:
    """Every number, in digits or words, that directly precedes ``noun`` in the text."""
    pattern = re.compile(r"(?<![\w-])" + _NUMBER_TOKEN + r" " + re.escape(noun.lower()) + r"(?!\w)")
    out = []
    for m in pattern.finditer(flat):
        tok = m.group(1)
        out.append(int(tok) if tok.isdigit() else _WORD_TO_INT[tok])
    return out


# ── The brief ────────────────────────────────────────────────────────────────

@dataclass
class Brief:
    """One agent brief, read once, checked many times."""

    path: Path
    raw: str = field(init=False, repr=False)
    flat: str = field(init=False, repr=False)

    def __init__(self, path: Path | str, text: str | None = None) -> None:
        self.path = Path(path)
        if text is None:
            if not self.path.exists():
                raise BriefDrift(f"{self.path}: brief not found")
            text = self.path.read_text(encoding="utf-8")
        self.raw = text
        self.flat = flatten(text)

    @property
    def name(self) -> str:
        return self.path.name

    def _fail(self, msg: str) -> None:
        raise BriefDrift(f"{self.name}: {msg}")

    # 1
    def must_name_all(self, items: Iterable[str], label: str = "items") -> None:
        """Every element is named in the brief; the missing ones are listed."""
        wanted = [str(i) for i in items]
        missing = [i for i in wanted if i.lower() not in self.flat]
        if missing:
            self._fail(f"{len(missing)} of {len(wanted)} {label} never named: {missing}. "
                       "An agent cannot use a tool its brief does not mention")

    # 2
    def must_state_count(self, n: int, noun: str) -> None:
        """The brief states ``n`` of ``noun`` and no other number for that noun."""
        stated = _stated_counts(self.flat, noun)
        wrong = sorted({s for s in stated if s != n})
        if wrong:
            self._fail(f"states {wrong[0]} {noun} for {n} in the repository")
        if n not in stated:
            forms = " / ".join(f"{f} {noun}" for f in _number_forms(n))
            self._fail(f"must state the count — expected one of: {forms}")

    # 3
    def must_quote(self, value: Any, suffix: str = "", prefix: str = "") -> None:
        """A configuration value appears verbatim, with its unit if any."""
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        needle = f"{prefix}{value}{suffix}".lower()
        if flatten(needle) not in self.flat:
            self._fail(f"must quote the current value {needle!r}; a value cited from memory "
                       "goes stale the day the configuration changes")

    # 4
    def must_not_contain(self, *phrases: str) -> None:
        """A stale phrase must not be back."""
        present = [p for p in phrases if flatten(p) in self.flat]
        if present:
            self._fail(f"stale phrase(s) present: {present}")

    # 5
    def must_contain(self, *phrases: str) -> None:
        """An invariant the brief must carry, because it lives nowhere else."""
        absent = [p for p in phrases if flatten(p) not in self.flat]
        if absent:
            self._fail(f"required phrase(s) absent: {absent}")


# 6
def must_carry_status(paths: Iterable[Path], markers: Sequence[str],
                      head_chars: int = 2000) -> None:
    """Every file shows one of ``markers`` in its first ``head_chars`` characters.

    A file without a status is invisible to any count made on statuses, and
    therefore to the agent that reads those counts.
    """
    without = [p.name for p in paths
               if not any(m in p.read_text(encoding="utf-8")[:head_chars] for m in markers)]
    if without:
        raise BriefDrift(f"files without a status marker {list(markers)}: {without}")


# ── References to the repository (for the YAML mode) ─────────────────────────

def resolve_ref(ref: Any, base: Path) -> Any:
    """``module:attr.sub`` imports a Python value; ``file.yaml#a.b`` reads a YAML key.

    Anything else is returned as is (a literal).
    """
    if not isinstance(ref, str):
        return ref
    if "#" in ref and not ref.startswith("#"):
        file, _, keypath = ref.partition("#")
        import yaml
        data = yaml.safe_load((base / file).read_text(encoding="utf-8"))
        for key in keypath.split("."):
            data = data[int(key)] if isinstance(data, list) else data[key]
        return data
    if ":" in ref and not ref.startswith(":"):
        mod, _, attr = ref.partition(":")
        if str(base) not in sys.path:
            sys.path.insert(0, str(base))
        obj: Any = importlib.import_module(mod)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj
    return ref


def _as_count(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple, set, frozenset, dict, str)):
        return len(value)
    raise TypeError(f"cannot take a count from {type(value).__name__}")


# ── YAML mode ────────────────────────────────────────────────────────────────

@dataclass
class Report:
    checked: int = 0
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checked": self.checked, "failures": list(self.failures)}


def _run_one(report: Report, fn, *args, **kwargs) -> None:
    report.checked += 1
    try:
        fn(*args, **kwargs)
    except BriefDrift as exc:
        report.failures.append(str(exc))


def run_checks(spec: dict, base: Path) -> Report:
    """Run the checks described by a YAML/JSON structure. Paths are relative to ``base``."""
    report = Report()
    for rel, checks in (spec.get("briefs") or {}).items():
        try:
            brief = Brief(base / rel)
        except BriefDrift as exc:
            report.checked += 1
            report.failures.append(str(exc))
            continue
        for c in checks.get("name_all") or []:
            _run_one(report, brief.must_name_all,
                     resolve_ref(c["items"], base), c.get("label", "items"))
        for c in checks.get("state_count") or []:
            _run_one(report, brief.must_state_count,
                     _as_count(resolve_ref(c["value"], base)), c["noun"])
        for c in checks.get("quote") or []:
            _run_one(report, brief.must_quote, resolve_ref(c["value"], base),
                     c.get("suffix", ""), c.get("prefix", ""))
        if checks.get("forbid"):
            _run_one(report, brief.must_not_contain, *checks["forbid"])
        if checks.get("require"):
            _run_one(report, brief.must_contain, *checks["require"])
    for s in spec.get("status") or []:
        excluded = set(s.get("except") or [])
        files = [p for p in sorted(base.glob(s["glob"]))
                 if p.is_file() and not any(fnmatch.fnmatch(p.relative_to(base).as_posix(), e)
                                            for e in excluded)]
        _run_one(report, must_carry_status, files, list(s["markers"]), s.get("head_chars", 2000))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_harness.prompt_tests",
        description="Check that agent briefs are accurate about the repository.")
    parser.add_argument("spec", type=Path, help="YAML or JSON file describing the checks")
    parser.add_argument("--base", type=Path, default=None,
                        help="root for relative paths (default: the spec's directory)")
    args = parser.parse_args(argv)
    text = args.spec.read_text(encoding="utf-8")
    if args.spec.suffix in (".yaml", ".yml"):
        import yaml
        spec = yaml.safe_load(text)
    else:
        spec = json.loads(text)
    report = run_checks(spec or {}, (args.base or args.spec.parent).resolve())
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    from agent_harness.prompt_tests import main as _main

    sys.exit(_main())
