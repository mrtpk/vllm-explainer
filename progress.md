# vLLM Explainer — Progress

Last update: 2026-10-22 19:10
Current phase: **Full build** (plan §12; max 2 agents). Phase 1 samples done except the r1 fact check. — plan approved ("go", 2026-10-22 19:13). Running (4/4): foundation agent, reference-generators agent, 1b basics samples (learn/ml-basics-recap, learn/serving-metrics), 1b Tour 1 (tours/life-of-a-chat-request + request_stepper visual). The two 1b writers started early, in parallel with 1a (they may need small fixes after STYLE.md lands).

This file is the memory of the project. A new session must be able to continue from it alone.

## Phase status

| Phase | Status | Notes |
|---|---|---|
| 0a Survey | done | 4 surveys in `notes/survey/` (core-engine, plugins, model-execution, serving-config-distributed); spot-checked |
| 0b Plan approval | done | approved 2026-10-22 ("go"); committed dd25917 |
| 1 Foundation + samples | done 20:35 (all samples written + fact-checked) | 1a foundation agent (STYLE, nav, core glossary, {{deepinfer}} tag, concepts/plugin-loading); generators agent (tools/gen_env_vars, gen_serve_flags, gen_config_classes, gen_metrics → notes/generated/*.json) |
| Full build (plan §12) | not started | |
| F Final pass | not started | |

## RESUME NOTES

(none — work is running normally. If a usage limit cuts agents off: list here the batch, the agent role, the partial files, and the resume order.)

## Done

- 2026-10-22 20:10: USER INSTRUCTION: quotas can run out; resume work after hitting quotas, about an hour later, unattended. Supervisor set an hourly session cron (job 3603c11b, minute :17, session-only, expires after 7 days) that reads this file, continues cut-off agents, keeps ≤2 agents running, and commits fact-checked batches.

- 2026-10-22 19:20: USER INSTRUCTION: "You dont need to stop for me. RUn unattented. This is okay because I already approve deepinfer explainer strcuture and coloring" → no Phase 1 stop; run the full build unattended; still no push.

- 2026-10-22 19:18: DECISION: start the basics and Tour 1 sample writers in parallel with 1a to use the 4-agent window before 19:56. Fixed all page ids (sent to the foundation agent). Risk: pages may need small STYLE.md fixes later.

- 2026-10-22 19:13: USER DECISION: plan approved ("go"). Starter kit copied, explainer.json filled, git init (author Thomas Paul, repo-local), first build 0 errors, commit dd25917.

- 2026-10-22: USER DECISION: Model Runner V2 only. Supervisor verified V2 is the default (config/vllm.py:713-748) and V1 is deprecated; V1 fallback = one cited note on concepts/model-runner.

- 2026-10-22: USER DECISION: full size (~100 pages); same base decisions as DeepInfer; git per stage, author Thomas Paul <contact@thomaspaul.dev>, no push. User wants to change the plan before Phase 1.

- 2026-10-22: USER INSTRUCTION: max 4 parallel subagents until 19:56; after that let running agents finish, then max 2.
- 2026-10-22: USER DECISION: basics already on DeepInfer Explained → one-line recap + link. Backlink domain https://explaindeepinfer.projects.thomaspaul.dev (`basics-covered.md`).
- 2026-10-22: USER INSTRUCTION: "I only care about the latest vllm. we dont need to study about legacy code. by end of going through the doc, one should become proficient in online llm serving."
- 2026-10-22: USER INSTRUCTION: no kernel details; cover vLLM basics, how plugins work, how vLLM actually works.
- 2026-10-22: DONE: 4 survey subagents (parallel). Supervisor spot-checked ~20 claims: all correct.
- 2026-10-22: Pinned commit 3e44c0db26f3530d674d255dafabaedcf0e3d742 (2026-10-22 22:21 +0800).

## Next step

Wait for the foundation + generators agents. Review, build, commit Phase 1a. Then Phase 1b samples (plan §11): max 4 parallel before 19:56, else max 2. After 1b: fact checks (1c), supervisor review, then continue straight into the full build (plan §12) — no user stop (user instruction 19:20).

## Questions for the user

See plan.md §2 (rows marked *Proposed*) and §4.6.

## Issues log

- FINAL-PASS TODO: reference/serve-flags must describe --quantization fp8 as deprecated (warn → fp8_per_tensor), not as an error. Later pages lora-serving, quantization-plugins, weight-loading: no FP8 raise, no unload open question, 92.4 MB/slot.

- 2026-10-22 22:23 (hourly check): 3 unknown 'fork' subagents found running (not started by the supervisor). Asked each for its task and files; flagged to the user. Not stopped yet (may be user-started). Watch for file conflicts with the B1b/B2b fact checkers.

- LESSON: fact checkers can also be wrong (B2a 'last-slice abort bug' was an indentation misread). Supervisor verifies any 'bug in vLLM' claim before it reaches a page or the FAQ.

- FINAL-PASS TODO: tours/life-of-a-chat-request ~537 errors table: validator errors (stream_options, tool_choice) go through the VLLMError handler, not RequestValidationError. Any page citing the `priority` docstring or saying continuous_usage_stats works alone. architecture-overview ~153: total 18 with CLI supervisor (B2a FC may fix).

- FINAL-PASS TODO: every page that says 'TP=1 → uni executor / worker inside engine core' must say TP = PP = DP = 1 (parallel.py:1055). Check: Tour 1, home map, plugin-loading, plugins-init, ml-basics-recap, processes-and-ipc, learn pages. Tour 1 deep dive 'What changes with DP > 1' LB score must use client_count × inflight (core_client.py:~1778).

- TODO B6b writer: concepts/compile-and-cuda-graphs must explain varlen-decode FULL graphs and eager_break_during_capture / breakable CUDA graph (attention.py:~767, compilation/breakable_cudagraph.py:~67). B10: reference/vllm-vs-deepinfer must match the table on learn/what-is-vllm.

- FINAL-PASS TODO (from r1 FC): learn/ml-basics-recap ~99-100 H100 quiz states 8192/1024 without the reported-memory assumption (≥70 and <160 GiB from the driver); learn/serving-metrics ~198 tier table "max" wording; any page saying kv_cache_dtype auto = model dtype; any page saying VllmConfig holds all configs; any page saying VLLM_USE_V2_MODEL_RUNNER always wins.
- Browser check: shot.js needs LD_LIBRARY_PATH=/tmp/claude-1000/vllm-foundation/libroot/usr/lib/x86_64-linux-gnu.

- FINAL-PASS TODO: tours/life-of-a-chat-request cites core_client.py:1338-1340 (abort_requests_async) — check it is the intended topic. Any page that says the engine core is separate because of VLLM_ENABLE_V1_MULTIPROCESSING, or 'worker process (one per GPU)' without the TP=1 uni case.


- FINAL-PASS TODO: no page may say VLLM_PLUGINS filters all groups, or that the registry subprocess is OOT-only/always runs (sent to home writer for plugins-init.html). Later pages: platform-plugins, oot-models, plugin-map, life-of-a-plugin.

- FINAL-PASS TODO: every page that states a default (chunked prefill, prefix caching, max_num_seqs, max_num_batched_tokens, cudagraph mode, async scheduling) must cite the EngineArgs/arg_utils override, not only the config dataclass. Source of truth: notes/facts.md "Processes and defaults".
- FINAL-PASS TODO: pages that say the engine loop is EngineCore.step() → default is step_with_batch_queue (async scheduling). Check Tour 1 (being fact-checked), home, engine-core-loop.
- FACT-CHECK FOCUS (all batches): b0 defect rate was ~41% (39/95). Brief every writer: check arg_utils overrides for defaults; FlashInfer sampler default; async scheduling default.

- DECISION 1a questions: site/ is committed (as DeepInfer) at stage ends; code/v1-attention-backend merges backend.py + selector.py (OK); nav titles OK.
- QUEUE (max 2 running): next writers B1b, B2b, B3a ... (plan §12 order), each followed by its fact check. Pattern: one writer + one fact checker at a time.

- FAQ material from b0: docs/design/metrics.md queue-time wording, arrival_time wording, TPOT formula differ from code (see notes/facts.md Conflicts).

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
| 2026-10-22 19:15 | 1a Foundation | done 19:44; committed b383713. Supervisor checked screenshot + diagram: good. Added 6 core terms (request, engine-step, prefill, decode, batch, scheduler). |
| 2026-10-22 19:15 | Reference generators (tools/gen_*.py) | done 19:48; committed 24a9b2d. 4 tools, --check OK; 10/10 spot checks each. |
| 2026-10-22 19:50 | Reference pages env-vars, serve-flags, config-classes, metrics (r1) | running |
| 2026-10-22 19:18 | 1b basics samples (b0) | done 19:34: learn/ml-basics-recap, learn/serving-metrics, diagram b0_serving_timeline, glossary b0.json (7 terms) |
| 2026-10-22 19:35 | Fact check b0 | done 19:47: ~95 claims, 39 defects fixed (14 citation, 6 value, 13 logic, 6 missing source). Main lesson: defaults cited to config dataclasses are overridden in arg_utils. |
| 2026-10-22 19:47 | 1b home page + code/plugins-init (c1) | done 20:04; supervisor checked map screenshot (good) |
| 2026-10-22 20:05 | Fact check home + plugins-init | done ~20:25: ~230 claims, 47 defects. Supervisor fixed the import chain on plugin-loading (lines 172-173). |
| 2026-10-22 19:50 | r1 reference pages | done ~20:15: 4 pages, generators filled, --check OK, build 0 errors |
| 2026-10-22 20:16 | Fact check r1 | done ~20:35: ~235 claims, ~40 defects; generators unchanged except gen_metrics † text; --check OK |
| 2026-10-22 20:26 | B1a writer (what-is-vllm, openai-api, processes-and-ipc, python-entry-points, pytorch-for-serving) | done ~20:50; 5 diagrams, b1a.json 11 terms |
| 2026-10-22 20:51 | B1a fact check | done ~21:20: ~250 claims, 38 defects (21 logic, 6 value, 7 citation, 4 missing) |
| 2026-10-22 21:21 | B1b writer | done ~21:50; 5 diagrams, b1b.json 14 terms |
| 2026-10-22 21:51 | B1b fact check | done ~22:30: ~230 claims, 24 defects (fp8 does not raise; LoRA unload answered; 92.4 MB/slot) |
| 2026-10-22 20:36 | B2a writer (architecture-overview, engine-client, data-contracts + 4 engine module pages) | done ~21:00; 4 diagrams, b2a.json 7 terms |
| 2026-10-22 21:01 | B2a fact check | done ~21:35: ~270 claims, 37 defects. Supervisor REJECTED its 'last-slice abort bug' (abort is inside the loop) and removed the note from 3 pages; supervisor fixed the msgspec claim on processes-and-ipc. |
| 2026-10-22 21:37 | B2b writer | done ~22:10; 5 diagrams, b2b.json 9 terms |
| 2026-10-22 22:11 | B2b fact check | running |
| 2026-10-22 19:18 | 1b Tour 1 (t1) | done 19:45: 16 stops, 3 diagrams, request_stepper.js, t1.json (12 terms); not browser-checked by writer |
| 2026-10-22 19:46 | Fact check t1 (+ browser check of stepper) | done ~20:00: ~230 claims, 38 defects (11 citation, 6 value, 9 logic, 3 default source, 9 other). Stepper OK at 1280/390, no overflow. |
| 2026-10-22 19:49 | Fact check concepts/plugin-loading (+ folder descriptions) | done ~19:58: ~80 claims, 30 defects fixed (8 citation, 11 value/condition, 4 logic, 3 missing source, 5 glossary/folder). Open question answered (platform before general plugins in every process). |
