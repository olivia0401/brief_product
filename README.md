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

## Relationship to the extraction benchmark

This is a **LangGraph extraction-workflow prototype**, not a second independent
brand-brief study or evidence of a separate production deployment. The
23-brief comparisons and the 52% → 91% keyword valid-output correction overlap
with the brand-naming evaluation reported in
[`prompt-evaluation`](https://github.com/olivia0401/prompt-evaluation).
Do not count those results twice or describe them as two client rollouts.

In this repository, the terms are used consistently:

- **Client brief:** the original information supplied for an extraction task.
- **Extraction workflow:** the LangGraph prototype generating structured outputs.
- **Source check:** a check that quoted evidence comes from the brief; it does
  not by itself establish that the generated interpretation is correct.
- **Automated quality check:** additional checks that may route an uncertain
  extraction to **human review**; human review is not an automatic pass.

The detailed report in `evals/RESULTS.md` is dated **2026-10-11**, and the
README evaluation snapshot above is **2026-11-05**. Those are record dates,
not evidence that each change was delivered during any particular employment
contract or approved for production. The synthetic Acme demo is illustrative;
raw client briefs and per-item client-content reports are not included here.

**Current limit:** The saved quality report has **not passed** its overall
quality gate. The 15/15 simulated trajectories and 5/5 injected-guard checks
are narrower prototype tests, not substitutes for production acceptance. The
uncollected test suite and its dependency issue remain documented above.
