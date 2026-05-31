# Compatibility Policy

Date: 2026-05-31

This policy classifies future changes to the analysis workspace. It applies before editing executable services or physics-sensitive configuration.

## Class A: Configuration-Only Change

Examples:

- Cut thresholds within approved policy bounds.
- Histogram binning.
- Enabled categories.
- Variable choices.
- Model settings.
- Scan ranges.
- Resource settings such as `max_events`.

Requirements:

- Preserve existing service code.
- Preserve physics-policy invariants.
- Record the configuration snapshot.
- Run verification checks.
- Compare baseline and candidate under comparable conditions.

## Class B: Backward-Compatible Service Extension

Examples:

- Optional new configuration field.
- New derived variable.
- New plotting mode.
- New output artifact.
- Additional service command that does not alter existing behavior.

Requirements:

- Update implementation.
- Add tests or documented verification.
- Update `ledger/EXECUTABLE_SERVICES.yaml`.
- Update `ledger/CHANGELOG.md`.
- Verify existing baseline configurations.
- Document the new extension point.

## Class C: New Executable Service

Examples:

- Reusable optimization scanner.
- Likelihood wrapper.
- New event-processing module.
- Reusable classifier-training service.
- Metadata-capture service.

Requirements:

- Define the interface.
- Add tests.
- Add a ledger entry.
- Define invariants.
- Document inputs and outputs.
- Document consumers.
- Create an example configuration.
- Provide a verifier or acceptance check.

## Class D: Backward-Incompatible Service Change

Examples:

- Changed meaning of a configuration field.
- Changed output schema.
- Removed entrypoint.
- Changed event-weight convention.
- Changed category semantics.

Requirements:

- Do not implement automatically.
- Write a migration proposal.
- Identify all affected consumers.
- Explain why compatibility cannot be preserved.
- Propose a version increment.
- Require explicit human approval.

## Class E: Physics-Policy Change

Examples:

- Modified normalization convention.
- Altered control-region meaning.
- Changed statistical interpretation.
- Changed blinding rule.
- Changed nominal-sample policy.

Requirements:

- Do not implement automatically.
- Record the proposal separately.
- Explain the scientific motivation and consequences.
- Require explicit human approval.

## Default Agent Operating Rule

Future optimization agents should first attempt Class A changes. If a desired strategy cannot be represented as a Class A change, the agent must classify the required change before editing code.
