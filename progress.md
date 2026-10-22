# vLLM Explainer — Progress

Last update: 2026-10-22 19:10
Current phase: **0 Plan** — base decisions approved; waiting for the user's plan changes.

This file is the memory of the project. A new session must be able to continue from it alone.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0a Survey | done | 4 surveys in `notes/survey/` (core-engine, plugins, model-execution, serving-config-distributed); spot-checked |
| 0b Plan approval | open | `plan.md` draft |
| 1 Foundation + samples | not started | |
| Full build (plan §12) | not started | |
| F Final pass | not started | |

## Done

- 2026-10-22: USER DECISION: Model Runner V2 only. Supervisor verified V2 is the default (config/vllm.py:713-748) and V1 is deprecated; V1 fallback = one cited note on concepts/model-runner.

- 2026-10-22: USER DECISION: full size (~100 pages); same base decisions as DeepInfer; git per stage, author Thomas Paul <contact@thomaspaul.dev>, no push. User wants to change the plan before Phase 1.

- 2026-10-22: USER INSTRUCTION: max 4 parallel subagents until 19:56; after that let running agents finish, then max 2.
- 2026-10-22: USER DECISION: basics already on DeepInfer Explained → one-line recap + link. Backlink domain https://explaindeepinfer.projects.thomaspaul.dev (`basics-covered.md`).
- 2026-10-22: USER INSTRUCTION: "I only care about the latest vllm. we dont need to study about legacy code. by end of going through the doc, one should become proficient in online llm serving."
- 2026-10-22: USER INSTRUCTION: no kernel details; cover vLLM basics, how plugins work, how vLLM actually works.
- 2026-10-22: DONE: 4 survey subagents (parallel). Supervisor spot-checked ~20 claims: all correct.
- 2026-10-22: Pinned commit 3e44c0db26f3530d674d255dafabaedcf0e3d742 (2026-10-22 22:21 +0800).

## Next step

Get user answers on plan §2 "Proposed" rows and the size option (§4 full vs §4.6 focused). Then Phase 1a.

## Questions for the user

See plan.md §2 (rows marked *Proposed*) and §4.6.

## Issues log

- 2026-10-22 (survey): docs/design/custom_op.md:11, :285 show `@CustomOp.register_oot("name")` → TypeError (custom_op.py:354-356). FAQ material.
- 2026-10-22 (survey): docs/design/plugin_system.md gaps (logits_processors group missing; OOT-wins rule missing). FAQ material.
- FACT-CHECK FOCUS: Model Runner V2 default conditions (config/vllm.py:713-748).

## Subagent log

| Date | Task | Result |
|---|---|---|
| 2026-10-22 | Survey: core engine | done, notes/survey/core-engine.md |
| 2026-10-22 | Survey: plugins | done, notes/survey/plugins.md |
| 2026-10-22 | Survey: model execution | done, notes/survey/model-execution.md |
| 2026-10-22 | Survey: serving/config/distributed | done, notes/survey/serving-config-distributed.md |
