"""Write jurisdiction: each agent role may write only where it is allowed to.

Why this module exists
----------------------
An LLM agent invoked with file tools can write anywhere the process can. Tool
allow-lists are global to the session and cannot be scoped per agent, so they
cannot protect the harness, the tests or another agent's files from a single
invocation. The only reliable barrier is downstream: snapshot the watched
tree before the agent runs, compare after, and **restore** anything that was
touched outside the agent's jurisdiction.

Two ways to use it
------------------
* In-process, as a context manager::

      with Guard(run_root, matrix, role="proposer", slots={"item": "b_003"}) as g:
          invoke_agent(...)
      g.result.ok, g.result.violations, g.result.restored

* Across processes, when the agent is invoked by something you cannot wrap
  (a tool call from an orchestrating LLM)::

      SNAP=$(python -m agent_harness.jurisdiction capture --run-root runs/x)
      ...agent runs...
      python -m agent_harness.jurisdiction enforce --run-root runs/x \\
          --snapshot-dir "$SNAP" --role proposer --slot item=b_003

  The snapshot lives in a system temp directory, outside the run root, so the
  agent cannot tamper with it.

Semantics
---------
* The **run root** is where agents work. Inside it, a path is writable by a
  role if it matches one of the role's globs. Globs are relative to the run
  root, POSIX style; ``*`` crosses directory separators (so ``items/*`` covers
  the whole subtree). ``{slot}`` placeholders are filled from ``slots``.
* **Protected paths** are watched too, and nothing under them is writable by
  any agent. Use them for the harness code, the tests, the prompts.
* Restoration is exact: a modified or deleted file gets its original bytes
  back; an illegally created file is removed.
* The guard never swallows an exception raised by the agent.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import sys
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Guard",
    "GuardResult",
    "JurisdictionError",
    "Matrix",
    "capture_snapshot",
    "enforce_from_snapshot",
    "is_writable",
    "snapshot",
]


class JurisdictionError(RuntimeError):
    """An agent wrote outside its jurisdiction (raised only on request)."""


@dataclass(frozen=True)
class Matrix:
    """Who may write where. Roles map to globs relative to the run root.

    ``protected`` lists extra directories or files, anywhere on disk, that are
    watched and never writable. Only the ones that exist are watched.
    """

    roles: Mapping[str, tuple[str, ...]]
    protected: tuple[Path, ...] = ()

    def globs(self, role: str, slots: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """The role's globs with ``{slot}`` placeholders filled in.

        An unknown role has no jurisdiction at all: it may write nothing. That is
        the safe default; a typo in a role name must not open the whole tree.
        """
        raw = self.roles.get(role, ())
        if not slots:
            return tuple(raw)
        return tuple(g.format_map(_Slots(slots)) for g in raw)


class _Slots(dict):
    """Leave ``{unknown}`` untouched instead of raising, so a glob that names a
    slot the caller did not provide simply never matches."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def is_writable(relpath: str, globs: tuple[str, ...]) -> bool:
    """Whether a run-root-relative POSIX path matches one of the globs."""
    return any(fnmatch.fnmatchcase(relpath, g) for g in globs)


@dataclass(frozen=True)
class GuardResult:
    """What one guarded invocation did outside its jurisdiction."""

    ok: bool
    violations: tuple[str, ...] = ()  # display paths, sorted
    restored: tuple[str, ...] = ()    # display paths actually restored/removed

    def to_dict(self) -> dict:
        return {"ok": self.ok, "violations": list(self.violations), "restored": list(self.restored)}


# ── File walking ─────────────────────────────────────────────────────────────

def _files_under(base: Path) -> Iterator[Path]:
    if base.is_file():
        yield base
    elif base.is_dir():
        for p in sorted(base.rglob("*")):
            if p.is_file():
                yield p


def _watched_roots(run_root: Path, protected: tuple[Path, ...]) -> list[Path]:
    return [run_root, *[p for p in protected if p.exists()]]


def _under(path: Path, root: Path) -> bool:
    r = root.resolve()
    p = path.resolve()
    return p == r or r in p.parents


def snapshot(run_root: Path, protected: tuple[Path, ...] = ()) -> dict[str, bytes]:
    """In-memory snapshot: absolute resolved path → file bytes."""
    out: dict[str, bytes] = {}
    for base in _watched_roots(run_root, protected):
        for f in _files_under(base):
            out[str(f.resolve())] = f.read_bytes()
    return out


def _is_violation(abs_path: Path, run_root: Path, globs: tuple[str, ...]) -> bool:
    """Under the run root → check the globs; anywhere else watched → always."""
    if _under(abs_path, run_root):
        rel = abs_path.resolve().relative_to(run_root.resolve()).as_posix()
        return not is_writable(rel, globs)
    return True


def _display(abs_path: Path, run_root: Path, protected: tuple[Path, ...]) -> str:
    p = abs_path.resolve()
    if _under(abs_path, run_root):
        return p.relative_to(run_root.resolve()).as_posix()
    for base in protected:
        if _under(abs_path, base):
            return p.relative_to(base.resolve().parent).as_posix()
    return p.as_posix()


def _restore(changed: set[str], before: Mapping[str, bytes | None],
             run_root: Path, globs: tuple[str, ...],
             protected: tuple[Path, ...]) -> GuardResult:
    """Shared by the in-memory and on-disk variants.

    ``before[path]`` is the original content, or ``None`` when the file did not
    exist before (so a change means it was created).
    """
    violations: list[str] = []
    restored: list[str] = []
    for abs_str in sorted(changed):
        abs_path = Path(abs_str)
        if not _is_violation(abs_path, run_root, globs):
            continue
        shown = _display(abs_path, run_root, protected)
        violations.append(shown)
        original = before.get(abs_str)
        if original is not None:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(original)
            restored.append(shown)
        elif abs_path.exists():
            abs_path.unlink()
            restored.append(shown)
    return GuardResult(ok=not violations, violations=tuple(violations), restored=tuple(restored))


# ── In-process guard ─────────────────────────────────────────────────────────

@dataclass
class Guard:
    """Context manager around one agent invocation.

    Only the invocation goes inside the ``with`` block. The orchestrator's own
    writes (state files, directories, archiving) belong outside it.
    """

    run_root: Path
    matrix: Matrix
    role: str
    slots: Mapping[str, str] = field(default_factory=dict)
    result: GuardResult | None = field(default=None, init=False)
    _before: dict[str, bytes] = field(default_factory=dict, init=False, repr=False)

    def __enter__(self) -> "Guard":
        self._before = snapshot(self.run_root, self.matrix.protected)
        return self

    def __exit__(self, *exc: object) -> bool:
        after = snapshot(self.run_root, self.matrix.protected)
        changed = {k for k in set(self._before) | set(after)
                   if self._before.get(k) != after.get(k)}
        self.result = _restore(changed, self._before, self.run_root,
                               self.matrix.globs(self.role, self.slots),
                               self.matrix.protected)
        return False  # never hide the agent's exception

    def raise_on_violation(self) -> None:
        if self.result is not None and not self.result.ok:
            raise JurisdictionError(
                f"role {self.role!r} wrote outside its jurisdiction: "
                + ", ".join(self.result.violations))


# ── Cross-process variant ────────────────────────────────────────────────────

def capture_snapshot(run_root: Path, protected: tuple[Path, ...] = ()) -> Path:
    """Copy every watched file into a fresh system temp directory.

    Returns the snapshot directory. It is outside the run root on purpose: an
    agent that could rewrite the snapshot could erase its own violation.
    """
    snap = Path(tempfile.mkdtemp(prefix="jurisdiction_"))
    blobs = snap / "blobs"
    blobs.mkdir()
    files: dict[str, str] = {}
    for i, f in enumerate(f for base in _watched_roots(run_root, protected)
                          for f in _files_under(base)):
        name = f"{i:06d}"
        shutil.copy2(f, blobs / name)
        files[str(f.resolve())] = name
    (snap / "manifest.json").write_text(json.dumps({
        "run_root": str(run_root.resolve()),
        "protected": [str(p.resolve()) for p in protected],
        "files": files,
    }), encoding="utf-8")
    return snap


def enforce_from_snapshot(run_root: Path, snapshot_dir: Path,
                          globs: tuple[str, ...],
                          protected: tuple[Path, ...] = (),
                          cleanup: bool = True) -> GuardResult:
    """Compare the tree with a captured snapshot and restore violations."""
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    blobs = snapshot_dir / "blobs"
    recorded: dict[str, str] = manifest["files"]

    before: dict[str, bytes | None] = {}
    changed: set[str] = set()
    for abs_str, blob in recorded.items():
        original = (blobs / blob).read_bytes()
        before[abs_str] = original
        p = Path(abs_str)
        current = p.read_bytes() if p.is_file() else None
        if current != original:
            changed.add(abs_str)
    for base in _watched_roots(run_root, protected):
        for f in _files_under(base):
            key = str(f.resolve())
            if key not in recorded:
                before[key] = None
                changed.add(key)

    result = _restore(changed, before, run_root, globs, protected)
    if cleanup:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
    return result


# ── CLI ──────────────────────────────────────────────────────────────────────

def _load_matrix(path: Path) -> Matrix:
    """A matrix file is YAML or JSON: ``roles: {role: [globs]}``, ``protected: [paths]``."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    roles = {str(r): tuple(str(g) for g in gs) for r, gs in (raw.get("roles") or {}).items()}
    protected = tuple(Path(p) for p in (raw.get("protected") or []))
    return Matrix(roles=roles, protected=protected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent_harness.jurisdiction",
        description="Snapshot a run tree before an agent runs; enforce and restore after.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-root", required=True, type=Path)
    common.add_argument("--matrix", type=Path, default=None,
                        help="YAML/JSON file with `roles` and `protected`.")
    common.add_argument("--protected", nargs="*", type=Path, default=[],
                        help="Extra protected paths (added to the matrix's).")

    sub.add_parser("capture", parents=[common], help="print the snapshot directory")
    enf = sub.add_parser("enforce", parents=[common], help="print a JSON result")
    enf.add_argument("--snapshot-dir", required=True, type=Path)
    enf.add_argument("--role", required=True)
    enf.add_argument("--slot", action="append", default=[], metavar="NAME=VALUE")

    args = parser.parse_args(argv)
    matrix = _load_matrix(args.matrix) if args.matrix else Matrix(roles={})
    protected = tuple(matrix.protected) + tuple(args.protected)

    if args.cmd == "capture":
        print(capture_snapshot(args.run_root, protected))
        return 0

    slots = dict(s.split("=", 1) for s in args.slot)
    result = enforce_from_snapshot(args.run_root, args.snapshot_dir,
                                   matrix.globs(args.role, slots), protected)
    print(json.dumps(result.to_dict()))
    return 0  # the JSON carries the verdict; callers parse stdout, not the exit code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
