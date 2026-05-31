# Artifact Reuse Policy

Date: 2026-05-31

This policy governs reuse of artifacts across baseline runs, candidate runs, scans, and qualitative branches.

## Safe Reuse

An artifact may be reused only if all of the following are true:

- Its verifier passed.
- Its schema is compatible with the consuming stage.
- Its producing service version remains compatible.
- Its relevant configuration subset has not changed.
- Its upstream artifacts remain valid.
- Its provenance is complete.
- No applicable scientific, interface, provenance, or change-control invariant has changed.
- The artifact is explicitly marked reusable.

## Required Invalidation

An artifact must be regenerated if any of the following are true:

- An upstream scientific input changed.
- A relevant configuration field changed.
- Its schema is incompatible.
- Its verifier failed, was not run, or is missing.
- Its producing service changed in a behaviorally relevant way.
- A physics-policy invariant changed.
- Required provenance is incomplete.
- The framework cannot establish safe reuse.

When uncertain, invalidate rather than silently reuse.

## Reuse Across Branches

Artifacts may be reused across qualitative strategy branches only when provenance and invalidation rules establish compatibility. Branch membership alone is not sufficient evidence of validity or invalidity.

## Blinding Guard

Artifacts produced with observed-data unblinding may not be reused in ordinary optimization branches unless explicit approval is recorded in provenance and the candidate run is also approved for the same observed-data access.

## Fit-Range And Histogram Coverage

Changing fit range may reuse upstream event caches when event-selection coverage is adequate. Histogram artifacts may be reused only when their bin edges and recorded coverage include the requested range. If coverage metadata is missing, regenerate histogram and downstream fit artifacts.
