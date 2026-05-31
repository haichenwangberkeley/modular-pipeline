# Branching Policy

Date: 2026-05-31

## Rules

- Every run has a `parent_run_id`, except the root baseline.
- Every run belongs to a `strategy_id` and `branch_id`.
- Local scans remain within a strategy branch.
- Qualitative strategy changes create a new branch.
- The reason for creating a branch must be recorded in the run registry and decision packet.
- Branches may share valid upstream artifacts only when provenance and invalidation rules allow it.
- Negative results remain searchable in observations and scan summaries so future agents do not repeat failed ideas blindly.
- Comparisons across branches must state evaluation conditions, including inputs, blinding state, service versions, and verification status.

## Branch Creation Metadata

Each new branch should record:

- Parent branch and parent run.
- Decision packet path.
- Decision-maker and response.
- Qualitative change being tested.
- Expected mechanism of improvement.
- Stages likely invalidated.
- Approval requirements.

## Cross-Branch Comparisons

Cross-branch comparisons are valid only when:

- Baseline and candidate use compatible input samples.
- Blinding policy is identical or approved.
- Metrics are computed by compatible service versions.
- Verifier status is passing.
- Any reused artifacts have complete provenance.
