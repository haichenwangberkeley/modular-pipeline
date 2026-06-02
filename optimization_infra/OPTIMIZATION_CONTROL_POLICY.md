# Optimization Control Policy

Date: 2026-05-31

## Control Flow

```text
Establish validated baseline
        |
        v
Choose quantitative local scan
        |
        v
Plan minimal recomputation
        |
        v
Execute candidates and verify outputs
        |
        v
Extract structured observations
        |
        v
Summarize improvements, deteriorations, and trade-offs
        |
        +----> another local scan is justified
        |             |
        |             v
        |       continue quantitative loop
        |
        +----> a qualitatively different approach may help
        |             |
        |             v
        |       create decision packet
        |             |
        |             v
        |       human or reasoning agent chooses next branch
        |
        +----> no credible improvement remains
                      |
                      v
                  stop and report
```

## Continue A Local Loop When

- The parameter response is interpretable.
- A promising region remains underexplored.
- No verifier failure occurred.
- The proposed next scan stays within existing configuration capabilities.
- The likely mechanism of improvement is understood.

## Escalate To Human Or Reasoning Agent When

- A new analysis strategy is suggested.
- A new variable or category may help.
- Local optimization saturates.
- The response is non-monotonic or difficult to interpret.
- A suspicious improvement appears.
- An unexpected deterioration reveals a possible modeling issue.
- A durable-service extension may be needed.
- Scientific judgment is required.

## Stop When

- No credible improvement remains.
- Computational cost is no longer justified.
- Validation fails repeatedly.
- Unresolved scientific ambiguity blocks further work.
- A human decision is required before further valid progress.

## Bounded Agent Loop Mode

A bounded agent loop may run for a fixed number of rounds only when all of the following are declared:

- `max_rounds`.
- Objective metric.
- Allowed configuration surface.
- Parent run or baseline.
- Branch and strategy identifiers.
- Stop-on-failure and escalation behavior.

Each round must:

- Read the prior round report or scan summary before selecting the next round.
- Preserve a descriptive version anchor in `VERSION.yaml` and `VERSION.md`.
- Preserve compact evidence artifacts instead of overwriting them.
- Write `ROUND_REPORT.yaml` and `ROUND_REPORT.md`.
- Create or update a decision packet when the next step needs human or reasoning-agent judgment.

The loop must stop when `max_rounds` is reached, a verifier/invariant block occurs, a service change is required, or human approval is needed.

## Verifier Hooks

The optimization framework must plan or run hooks for:

- Configuration validation.
- Schema validation.
- Artifact provenance validation.
- Cut-flow sanity checks.
- Yield sanity checks.
- Histogram integrity checks.
- Fit-completion checks.
- Fit-quality checks.
- Missing-output checks.
- Baseline-comparability checks.
- Invariant checks.

Do not invent physics thresholds. When a threshold requires scientific judgment, record it as unresolved and escalate.
