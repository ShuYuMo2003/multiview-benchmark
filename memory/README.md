# Project memory

This directory is the persistent engineering memory for the benchmark. It is
part of the project, not an experiment artifact.

- `PROJECT_CHARTER.md`: project motivation, benchmark scope, non-goals, and
  success criteria. Edit this first when the purpose itself changes.
- `PROJECT_STATE.md`: current implementation status and next executable work.
- `DESIGN_DECISIONS.md`: decisions, alternatives, and reasons.
- `DATA_CONTRACT.md`: LeRobot-to-training schema and label conventions.
- `EVALUATION_PROTOCOL.md`: metric definitions and benchmark validation.
- `EXPERIMENT_LOG.md`: append-only experiment records and artifact locations.

Read `PROJECT_CHARTER.md` first. Update `PROJECT_STATE.md` whenever an
implementation milestone changes. Append to `EXPERIMENT_LOG.md` for every
non-trivial run; do not overwrite old results.
