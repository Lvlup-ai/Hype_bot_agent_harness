# agent-harness

A deterministic harness for orchestrating LLM agents.

**The LLM judges, the harness computes.** Agents propose, audit and contest.
The harness decides what happens next, keeps count, and refuses what the
protocol forbids. No decision of flow is ever left to a prompt.

> Status: early extraction. This code comes from a private multi-agent research
> pipeline that ran for several months. It is being generalised here, module by
> module, with its tests. The state machine is the part the original pipeline
> should have had in code from the start: it lived in a runbook prompt, and the
> day an audit showed that a "bounded" retry loop had no enforced bound at all
> is the day this repository was decided.

## The problem

A pipeline of agents usually looks like a script that chains prompts. It works
until it doesn't, and when it doesn't, nothing tells you:

- an agent wrote outside the files it was allowed to touch;
- a "maximum of 3 retries" was exceeded because the counter lived in the model's
  context and the context was summarised;
- a report was truncated, or its summary statistics contradict its own rows;
- an agent invented a verdict that was not in the allowed vocabulary;
- a prompt described the repository as it was two months ago.

Every one of these happened in the pipeline this harness comes from. Each
module below exists because of one of them.

## What the harness provides

| Module | What it enforces |
|---|---|
| `state_machine` | An explicit transition table. A forbidden transition is an error, not a warning. Verdicts come from a closed vocabulary. |
| `jurisdiction` | Each agent role may write only inside its declared globs. Snapshot before, enforce after, restore what was touched illegally. |
| `budget` | Persistent trial quotas. A retry costs budget; a re-run of the same hypothesis does not. |
| `ledger` | Append-only event log with a closed vocabulary. The multiple-testing counter you can actually read back. |
| `contracts` | Versioned deliverable contracts with a validating loader: unknown version, truncated payload or self-contradicting statistics are refused. |
| `review_loop` | The bounded adversarial loop: RETRY / PASS / ESCALATE, at most N retries, then a human. |
| `prompt_tests` | Tests that check an agent brief tells the truth about the repository it describes. |

## How a run looks

```
            ┌────────────────────────── harness (deterministic) ──────────────────────────┐
            │                                                                              │
  human ──► │  next_action ──► [agent: propose] ──► record ──► next_action ──► ...         │
            │       ▲              (guarded by             │                               │
            │       │               jurisdiction)          ▼                               │
            │       │                                  [agent: audit] ──► record           │
            │       │                                                        │             │
            │       └──────────── RETRY (n < N) ◄──── [agent: contest] ◄─────┘             │
            │                     PASS ──► next phase                                      │
            │                     ESCALATE ──► human                                       │
            └──────────────────────────────────────────────────────────────────────────────┘
```

The orchestrating LLM does exactly three things: ask the harness for the next
action, invoke one agent under a jurisdiction guard, and record the outcome
through the harness CLI. It never writes state files directly.

## Quickstart

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

An end-to-end example in a neutral domain is part of the roadmap and will
live under `examples/`.

## Design rules

- Same inputs and same code give the same outputs. No randomness anywhere.
- Every threshold is declared before the run, in a file, and is never changed
  during the run.
- A prompt is a document that nothing compiles. Here, prompts are tested.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
