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

(none — both cut-off writers were resumed at 01:01; B3b finished.)

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

- FYI (not blocking): POST /scale_elastic_ep is reachable without --api-key on every generative server (verified). Traced: on a default deployment it cannot rescale engines (500/400 or no-op); one Ray+dense path rewrites front-end port fields before failing; Ray+MoE+EPLB scale-down not traced. User may want to report the unauthenticated route upstream. Site states facts only.

See plan.md §2 (rows marked *Proposed*) and §4.6.

## Issues log

- FINAL-PASS TODO (B9 FC): openai-server and any page listing unguarded routes must add /generative_scoring and pooling routes; pages saying auto KV dtype = model dtype without the checkpoint scheme caveat; 'Engine 000:' is not DP-only.

- FINAL-PASS TODO (B7b FC): concepts/model-runner ~165 'drafts never go to the CPU' wrong for structured-output batches; learn/lora-adapters + frontend-plugins: no 'adapter loads inside the step' wording; slot layouts follow activation at load.

- FINAL-PASS TODO (B7b): compile-and-cuda-graphs should mention LoRA doubles CUDA graph captures by default; engine-hooks should mention custom logits processor add_request default True forces the FP32 copy.

- FINAL-PASS TODO (B8a, supervisor verified): pre_register_and_update runs 5× (not 4×, not 2×) in the serve front end: fix tours/life-of-a-plugin stop 5 + counts table + diagram b5t_plugin_life ("x 4"), concepts/platform-interface (~49, ~101-102, ~118), concepts/quantization-plugins ~136, diagram b4a_platform_hooks. check_and_update_config/apply_config_platform_defaults ALSO run in every engine core (core.py:1336-1337): fix tours/life-of-a-plugin ("runs only in the front end"; table "1 each"), concepts/platform-interface and plugin-map wherever they say front end only. reference/env-vars VLLM_ENGINE_READY_TIMEOUT_S row: only with >1 API server.

- FINAL-PASS TODO (B7a/B6t FC): learn/openai-api ~69, ~207 and reference/serve-flags --api-key row need the /invocations, /ping limitation; concepts/model-runner ~173 cite vllm.py:713-734 → 713-748; STE warning in learn/processes-and-ipc.

- FINAL-PASS TODO (B7a): learn/openai-api tool deep dive — required/named also need --enable-auto-tool-choice; Tour 1 stop 3 template resolution incomplete (add processor + built-in fallback or link renderers page). B9 must include the --api-key / /invocations limitation in deployment/production pages.

- FINAL-PASS TODO (B6b FC): reference/env-vars VLLM_LOG_BATCHSIZE_INTERVAL text ('after every forward' wrong); any page saying an mp worker always uses WorkerAsyncOutputCopy.

- FINAL-PASS TODO (B5t FC, supervisor verified): pre_register_and_update runs 4× in the vllm serve front end, not twice: fix concepts/platform-interface ~49, ~101-102, ~118; concepts/quantization-plugins ~136; diagram b4a_platform_hooks (used on platform-interface and code/platforms-interface ~25); also facts docstring note.

- FINAL-PASS TODO (B6a FC): learn/tp-pp-in-practice ~47 '64 all-reduces per step' → align with 65 (64 + embedding) on model-definition; any page saying the completeness check catches missing linear weights in BF16 models; TeleChat2 mapper-only; SupportsQuant remapping Llama names.

- FINAL-PASS TODO (B5 FC): any page saying sleep mode is CUDA/HIP only; front end imports the KV factory lazily; Llama prefix model.layers.N.qkv_proj (missing self_attn); platform-interface 'pre_register_and_update only before parsing'.

- FINAL-PASS TODO (B4b FC): 'O2 gives VLLM_COMPILE' needs the exceptions (breakable-CUDA-graph default archs vllm.py:75-108, 825-851; --enforce-eager; TORCH_COMPILE_DISABLE=1) on learn/what-is-vllm ~96, learn/pytorch-for-serving ~118, ~146, reference/serve-flags ~65.

- FINAL-PASS TODO (B3b FC): code/v1-core-kv-cache-manager ~132 'AsyncScheduler hashes it later when output of step 3 arrives' → the next step's allocate_slots hashes it; concepts/async-scheduling ~96 mechanism → num_tokens cap at allocation (kv_cache_manager.py:612-615).

- FINAL-PASS TODO (B4a FC): kv-connectors, frontend-plugins, reference/extension-points must say connector registries are also read in the front end (class only); attention-backend-selection: register_backend string duplicate raises; custom-ops: forward_oot only for enabled ops; quantization-plugins: workers get the quant config object via pickled VllmConfig; compile-and-cuda-graphs / config-system: check_and_update_config 'late in' __post_init__.

- FINAL-PASS TODO (B2b FC): 'slow load hits VLLM_ENGINE_READY_TIMEOUT_S' without the >1-API-server condition on concepts/architecture-overview ~161, learn/processes-and-ipc ~140, reference/env-vars ~86; architecture-overview 'uni only when TP=PP=DP=1' → add 'by default' and PCP; async-off lists missing the CPU case (reference pages, t1 glossary async-scheduling); 'engine core calls get_output()' without the mp worker-thread case (Tour 1); any 'sleep level 2 discards all GPU memory'.

- FINAL-PASS TODO: reference/serve-flags must describe --quantization fp8 as deprecated (warn → fp8_per_tensor), not as an error. Later pages lora-serving, quantization-plugins, weight-loading: no FP8 raise, no unload open question, 92.4 MB/slot.

- 2026-10-22 22:23 (hourly check): 3 unknown 'fork' subagents found running (not started by the supervisor). Asked each for its task and files; flagged to the user. Not stopped (may be user-started). By 22:31 all 3 had ended; git status showed no stray changes. CAUSE (22:45): the B2b fact checker forked them itself. Fix: both briefs now forbid sub-agents.

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
- QUEUE (max 2 running): B7b FC, B8a FC, then B8b, B9, B10, F (final pass: consistency + site check + README). Pattern: one writer + one fact checker at a time.

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
| 2026-10-22 22:11 | B2b fact check | done ~22:45: ~380 claims, ~73 defects. NOTE: it started 3 forks itself (the 'unknown agents'); briefs now forbid sub-agents. |
| 2026-10-22 22:32 | B3a writer (scheduler, kv-cache-manager + 3 modules + scheduler simulator) | done ~23:10; 4 diagrams, b3a.json 12 terms; core block-pool term reworded by supervisor |
| 2026-10-22 23:11 | B3a fact check | done ~23:40: ~260 claims, 25 defects; simulator verified |
| 2026-10-22 23:41 | B3b writer | cut off by usage limit ~00:05, resumed 01:01, done ~01:30 |
| 2026-10-23 01:31 | B3b fact check | done ~01:45: ~190 claims, 21 defects; explorer verified in 12 scenarios |
| 2026-10-23 01:46 | B5 writer | done ~02:35; 6 diagrams, chooser visual, b5.json 16 terms. Decision: DeepSeek-V3 MLA example kept (page notes ignored +quant_fp8). |
| 2026-10-23 02:36 | B5 fact check (+ 3 plugin-map row fixes) | done ~03:00: ~270 claims, 34 defects; chooser all 640 combos OK |
| 2026-10-23 03:01 | B5t writer (life-of-a-plugin, build-a-toy-plugin) | done ~03:35; 2 diagrams |
| 2026-10-23 03:36 | B5t fact check | done ~03:55: ~190 claims, 27 defects; pre_register_and_update ×4 (supervisor verified) |
| 2026-10-23 03:57 | B7a writer | done ~04:45; 6 diagrams, b7a.json 13 terms |
| 2026-10-23 04:46 | B7a fact check | done: ~330 claims, 26 defects + 2 extra fixes (openai-api, Tour 1 stop 3) |
| 2026-10-23 02:11 | B6a writer | done ~02:50; 3 diagrams, b6a.json 13 terms |
| 2026-10-23 02:51 | B6a fact check | done ~03:15: ~175 claims, 26 defects |
| 2026-10-23 03:16 | B6b writer | done ~04:00; 6 diagrams, b6b.json 18 terms |
| 2026-10-23 04:01 | B6b fact check | done ~04:30: ~330 claims, 47 defects |
| 2026-10-23 04:31 | B6t writer (tours/forward-pass) | done ~04:55; 2 diagrams |
| 2026-10-23 04:56 | B6t fact check | done: ~125 claims, 16 defects |
| 2026-10-23 06:03 | B7b writer | done ~06:55; 5 diagrams, b7b.json 19 terms |
| 2026-10-23 06:56 | B7b fact check | done ~07:20: ~190 claims, ~50 defects |
| 2026-10-23 06:03 | B8a writer | done ~06:40; 2 diagrams, b8a.json 7 terms |
| 2026-10-23 06:41 | B8a fact check | done ~07:30: ~330 claims, 22 defects |
| 2026-10-23 07:31 | B9 writer (operating) | done ~08:25; 5 diagrams, tuning playground, b9.json 10 terms |
| 2026-10-23 08:26 | B9 fact check | done ~08:50: ~310 claims, 32 defects; playground verified |
| 2026-10-23 07:21 | B8b writer | done ~08:15; 6 diagrams, b8b.json 16 terms |
| 2026-10-23 08:16 | B8b fact check | done ~08:40: ~260 ranges + ~60 claims, ~36 defects; traced /scale_elastic_ep |
| 2026-10-23 08:41 | B10 writer | running |
| 2026-10-22 22:47 | B4a writer (plugin-map, platform-plugins, platform-interface + 2 modules + plugin timeline) | done ~23:25; 3 diagrams, b4a.json 5 terms |
| 2026-10-22 23:26 | B4a fact check | done ~23:55: ~270 claims, 24 defects; OOT quant open question answered |
| 2026-10-22 23:56 | B4b writer | cut off ~00:05, resumed 01:01, done ~01:45; 3 diagrams, b4b.json 11 terms |
| 2026-10-23 01:46 | B4b fact check | done ~02:10: ~190 claims, 37 defects; CustomOp default exceptions |
| 2026-10-22 19:18 | 1b Tour 1 (t1) | done 19:45: 16 stops, 3 diagrams, request_stepper.js, t1.json (12 terms); not browser-checked by writer |
| 2026-10-22 19:46 | Fact check t1 (+ browser check of stepper) | done ~20:00: ~230 claims, 38 defects (11 citation, 6 value, 9 logic, 3 default source, 9 other). Stepper OK at 1280/390, no overflow. |
| 2026-10-22 19:49 | Fact check concepts/plugin-loading (+ folder descriptions) | done ~19:58: ~80 claims, 30 defects fixed (8 citation, 11 value/condition, 4 logic, 3 missing source, 5 glossary/folder). Open question answered (platform before general plugins in every process). |
