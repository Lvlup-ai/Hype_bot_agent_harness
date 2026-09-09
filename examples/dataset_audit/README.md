# Example: dataset quality audit

Three agents look for rules that flag the anomalous rows of a forty-row user
table. The agents are scripted Python functions, so the run needs no model and
no key; each function says where a model call would go. The harness around
them is the real one.

```bash
python -m examples.dataset_audit.run
```

## What you will see

Seven iterations, each exercising one mechanism:

| # | The proposer | The harness |
|---|---|---|
| 1 | `age range [0, 120]`, in scope | audited GO, measured, accepted, cost 1 |
| 2 | `email contains @`, in scope | accepted, same F1 as the best: the best does not move |
| 3 | `score range [0, 100]`, off topic | **refused before measurement**: the `signup_date` track has no rule yet |
| 4 | `signup_date range`, in scope | accepted |
| 5 | a valid rule, but the agent writes into `state.json` | the file is **restored**, a jurisdiction failure is recorded |
| 6 | `score range [0, 100]`, off topic, now allowed | measured at cost 3, precision 0, rejected |
| 7 | `age range [1, 120]`, adjacent | accepted, new best; its rationale **overstates** recall |

Then the phase ends on `max_iterations`, the harness writes `report.json`
under the contract `proposal-report/1`, and the reviewer reads it through
the contract loader. It compares each accepted rule's claims with
`measure.json`, finds the overstated recall, and returns `RETRY` with an
`AGAINST` finding. The proposer restates the claim; the reviewer returns
`PASS`. The ledger has one line for the retry, the retry counter reads 1 of 2.

Everything lands under `examples/dataset_audit/runs/<run id>/`: `state.json`
(written only by the state machine), one directory per rule with the
proposal, the audit and the measurement, `report.json`, and the boundary
state. The ledger and the budget file sit next to the run and persist across
runs: run the example twice with the same runs root and the second run finds
the budget already partly spent.

## The briefs are tested

`briefs/` holds the three prompts a model would receive. `prompt_checks.yaml`
states what they must say about the code: the count of operators comes from
`rules.py`, the costs and thresholds from `config.yaml`. Add an operator
without updating the proposer's brief and CI fails:

```bash
python -m agent_harness.prompt_tests examples/dataset_audit/prompt_checks.yaml --base .
```

## Plugging a model in

Replace the bodies of `propose`, `audit` and `review` in `agents.py` with
calls to your model, each reading the matching brief. Keep the return
shapes. Nothing else changes: the guard, the budget, the state machine, the
contract and the review cap apply to a model exactly as they apply to the
script.
