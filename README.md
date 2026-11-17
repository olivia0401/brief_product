# Brief Product — Client Brief Extraction Workflow

A LangGraph-based prototype that turns client briefs into structured draft outputs, keeps per-client run state, records source and recipe hashes, and routes uncertain fields for human review. It is designed to support a consultant workflow; generated fields still require review before delivery.

## Current evaluation snapshot (2026-11-05)

- The project reports parity checks across 23 briefs and multiple model/task combinations. Most evaluated fields could be generated using one or two targeted brief fields within the stated tolerance.
- A keyword-output parsing correction increased valid-output rate from 52% to 91% and keyword F1 from 0.179 to 0.349 in the recorded evaluation.
- The saved quality report currently fails its quality gate: its sample is below the minimum, and grounding, citation completeness, unsupported-claim rate, entity resolution and evaluator agreement do not all meet thresholds. Do not describe it as a passing quality certification.
- Existing trajectory and mutation artifacts report 15/15 simulated trajectories and 5/5 injected-guard cases. These are prototype checks, not production acceptance.

## Validation boundary

The checks verify output structure and evidence presence/overlap; they do not establish that every cited passage semantically supports the generated claim. Human review remains part of the workflow.

An earlier audit could not collect the test suite because LangGraph/LangChain dependencies (including `langgraph.checkpoint.sqlite`) were absent. This was addressed by adding `requirements-test.txt` and the [offline CI workflow](.github/workflows/ci.yml), which checks out a pinned version of the shared `prompt-evaluation` parser. The [2026-11-17 CI run](https://github.com/olivia0401/brief_product/actions/runs/43992337767) passed **33 pytest tests, 15/15 scripted trajectories, and 5/5 mutation/guardrail probes**. These use a mocked LLM and synthetic briefs: they verify offline control flow and tests, **not** live-provider quality, customer acceptance, or the overall evidence-quality gate.

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
- **Source check:** checks whether quoted evidence comes from the brief. The
  earlier routing logic also used a lexical-overlap screen; neither alone
  establishes whether the extracted interpretation is correct.
- **Automated quality check:** the revised routing uses the **existing**
  lexical-overlap score with a tighter acceptance threshold (0.30 → 0.42),
  referring more uncertain extractions to **human review**. This was a
  threshold/routing change, not the first introduction of word overlap.

The detailed report in `evals/RESULTS.md` is dated **2026-10-11**, and the
README evaluation snapshot above is **2026-11-05**. Those are record dates,
not evidence that each change was delivered during any particular employment
contract or approved for production. The synthetic Acme demo is illustrative;
raw client briefs and per-item client-content reports are not included here.

**Current limit:** The saved quality report has **not passed** its overall
quality gate. The 15/15 simulated trajectories and 5/5 injected-guard checks
are narrower prototype tests, not substitutes for production acceptance. The
uncollected test suite and its dependency issue remain documented above.
