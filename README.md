# Brief Product — Client Brief Extraction Workflow

A LangGraph-based prototype that turns client briefs into structured draft outputs, keeps per-client run state, records source and recipe hashes, and routes uncertain fields for human review. It is designed to support a consultant workflow; generated fields still require review before delivery.

## Current evaluation snapshot (2026-11-05)

- The project reports parity checks across 23 briefs and multiple model/task combinations. Most evaluated fields could be generated using one or two targeted brief fields within the stated tolerance.
- A keyword-output parsing correction increased valid-output rate from 52% to 91% and keyword F1 from 0.179 to 0.349 in the recorded evaluation.
- The saved quality report currently fails its quality gate: its sample is below the minimum, and grounding, citation completeness, unsupported-claim rate, entity resolution and evaluator agreement do not all meet thresholds. Do not describe it as a passing quality certification.
- Existing trajectory and mutation artifacts report 15/15 simulated trajectories and 5/5 injected-guard cases. These are prototype checks, not production acceptance.

## Validation boundary

The checks verify output structure and evidence presence/overlap; they do not establish that every cited passage semantically supports the generated claim. Human review remains part of the workflow.

The test suite could not be collected in the current audit environment because required LangGraph/LangChain imports were unavailable, including `langgraph.checkpoint.sqlite`. Install compatible project dependencies and rerun the suite before treating source tests as passing.