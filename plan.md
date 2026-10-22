# vLLM Explainer — Plan

Status: **DRAFT — base decisions approved 2026-10-22; user wants changes before Phase 1.**
Date: 2026-10-22
Source repo: `../vllm` at commit `3e44c0d` (https://github.com/vllm-project/vllm)
Skill: `skills/codebase-explainer/` (this folder). Sibling site: `../deepinfer-explainer`, published at
https://explaindeepinfer.projects.thomaspaul.dev.

---

## 1. Goal

A static study site that teaches how the **latest** vLLM works. When the reader finishes it, they are
**proficient in online LLM serving with vLLM**: they know what happens to a request inside `vllm serve`,
how to configure, tune, monitor and scale a deployment, and how vLLM is extended with plugins.

Three focus areas (user, 2026-10-22):
1. **vLLM basics** — only what DeepInfer Explained does not already teach.
2. **How vLLM actually works** — the online request path, engine, scheduler, KV cache, model execution.
3. **How plugins work** — every extension point, how plugins load in each process, how to write one.

## 2. Decisions

| Topic | Decision |
|---|---|
| Version | **Latest only** (user 2026-10-22). Pinned commit `3e44c0db26f3530d674d255dafabaedcf0e3d742` (2026-10-22). No V0 (only aliases remain), no Model Runner V1 (V2 is the default; V1 gets one sentence as "the fallback"). |
| Model runner | **Model Runner V2 only** (user 2026-10-22). Verified: default when `VLLM_USE_V2_MODEL_RUNNER` is unset, Triton is present and no unsupported feature is set (vllm/config/vllm.py:713-748); built in gpu_worker.py:567-583; V1 is deprecated (comment in `_get_v1_model_runner_unsupported_features`). V1 fallback triggers (no Triton, PP with external_launcher, spec methods suffix/custom_class, EAGLE parallel drafting, heterogeneous-vocab drafts, some DBO cases) get one cited note on concepts/model-runner. OOT platforms may bring their own worker/runner via `worker_cls`: explained in the plugin pages only as a hook. |
| Goal | Proficient in **online** serving (user 2026-10-22). Tour 1 = one HTTP chat request. Offline `LLM` appears only as a short side note. |
| Basics already taught | One-line recap + link to the DeepInfer page (user 2026-10-22). See `basics-covered.md`. Links use https://explaindeepinfer.projects.thomaspaul.dev/<id>.html. |
| Out of scope | Kernel internals (`csrc/`, Triton/CUDA/CuTe/Helion kernel code), legacy code, the experimental Rust frontend (one paragraph only), per-model special code in `vllm/models/*` (one paragraph only). Explain only how vLLM selects and calls kernels. |
| Subagents | **Max 4 at the same time until 19:56 (2026-10-22); after that, let running agents finish, then max 2** (user instruction). Fact checkers count toward the limit. |
| Reader | same as DeepInfer — knows Python, has read DeepInfer Explained (or knows the LLM basics). |
| Depth | beginner start + full "Deep dive" on every page. |
| Language | ASD-STE100, like DeepInfer. |
| Look | same look as DeepInfer Explained (light article look; dark only by toggle). Same starter kit. |
| Hosting | local first; published later like DeepInfer (CDN libraries OK). |
| Location | `test-harness/vllm-explainer/` (this folder). `../vllm` stays unchanged. |
| Name | "vLLM Explained". |
| Checkpoint | **Changed 2026-10-22: no stop.** User: "You dont need to stop for me. RUn unattented. This is okay because I already approve deepinfer explainer strcuture and coloring". Phase 1 samples are reviewed by the supervisor, then the full build continues without a user gate. Push/publish still only on request. |

User answers 2026-10-22: full size (~100 pages); same base decisions as DeepInfer (reader, depth, ASD-STE100, look, local first, stop after Phase 1); git per stage with author Thomas Paul <contact@thomaspaul.dev>, no push. Plan changes requested before Phase 1 (pending).
| Git | commit per stage, push only on request, author `Thomas Paul <contact@thomaspaul.dev>` (as in DeepInfer), repo-local. |

## 3. Facts about the codebase (from 4 surveys, `notes/survey/*.md`)

- Size: 8,357 tracked files. Python ~2.0M lines (`vllm/` ~1.13M in 2,907 files; `tests/` ~0.77M).
  Rust 152k (experimental frontend). C++/CUDA ~143k (out of scope). 23,594 commits; no submodules.
- Main surprises at this commit: Model Runner V2 is the default; `vllm/attention/` is gone (layer is
  in `model_executor/layers/attention/`); a second model tree `vllm/models/` (per family, nvidia/amd);
  `FusedMoE` is a factory function; a Rust OpenAI frontend in `rust/`; `vllm/ir` + `vllm/kernels`
  (an op IR with pluggable kernel providers).
- Process layout (serving): API server process(es) → ZMQ + msgpack → EngineCore process (one per DP
  rank: scheduler, KV cache manager) → shared-memory MessageQueue → one worker process per GPU.
  Single GPU: worker runs inside the EngineCore process.
- 20 extension points (`notes/survey/plugins.md` §1). 5 plugin entry-point groups + `vllm.logits_processors`.
- ~317 env vars (`vllm/envs.py`, a dict of lambdas); ~45 `@config` classes; CLI flags made from the
  config dataclasses → all three reference tables can be **generated**.
- Good docs: `docs/design/*.md` (32 design docs), `docs/serving/*`, `docs/configuration/*`.
  Reusable images in `docs/assets/design/`. Docs vs code differences already found (FAQ material),
  e.g. `docs/design/custom_op.md:11` shows `register_oot("name")`, which raises `TypeError`.

## 4. Site structure (about 100 pages — full option chosen by the user 2026-10-22)

```
vllm-explainer/site/
  index.html   Home: what vLLM is, system map (processes), "Start here" paths, code version
  learn/       PART 1 — vLLM basics (12)
  concepts/    PART 2 — Concepts (48, in 6 groups)
  tours/       PART 3 — Guided tours (7)
  code/        PART 4 — Folder tree + module pages (~28)
  reference/   PART 5 — Generated tables, extension points, glossary, FAQ (8)
```

### 4.1 Part 1 — Learn the basics (12)
1. `learn/ml-basics-recap` — hub: the 12 DeepInfer basics, one-line recap + link each.
2. `learn/what-is-vllm` — what vLLM is today: V1 engine, `vllm serve` vs `LLM`, hardware, what is new vs DeepInfer.
3. `learn/serving-metrics` — TTFT, TPOT/ITL, E2E latency, throughput, goodput, concurrency, SLOs; the latency/throughput trade-off.
4. `learn/openai-api` — chat/completions API, roles, chat templates, SSE streaming, tool calls, reasoning content.
5. `learn/processes-and-ipc` — why an engine uses several processes; ZMQ sockets, msgpack, shared memory.
6. `learn/python-entry-points` — packaging entry points and `importlib.metadata` (the base of plugins).
7. `learn/pytorch-for-serving` — `nn.Module`, custom ops (`torch.library`), FX graphs, `torch.compile` (Dynamo + Inductor).
8. `learn/prefix-caching` — shared prefixes in chat traffic; hash chains; why it matters for TTFT.
9. `learn/tp-pp-in-practice` — column/row split math with small numbers; TP vs PP vs DP for serving.
10. `learn/quantized-checkpoints` — weight-only vs weight+activation; FP8, AWQ, GPTQ, compressed-tensors checkpoints.
11. `learn/lora-adapters` — what a LoRA adapter is; serving many adapters on one base model.
12. `learn/hf-checkpoints` — `config.json` (`architectures`), safetensors, tokenizer and chat template files.

### 4.2 Part 2 — Concepts (48)
- **Architecture and engine (11):** architecture-overview · engine-client (AsyncLLM) · engine-core-loop ·
  data-contracts (EngineCoreRequest → SchedulerOutput → ModelRunnerOutput → EngineCoreOutputs) ·
  scheduler (one token budget, no prefill/decode phase) · kv-cache-manager (block pool, memory profile,
  `gpu_memory_utilization`) · prefix-caching (hash chain, LRU free queue) · hybrid-kv-cache (groups and
  specs: SWA, Mamba, MLA) · async-scheduling (batch queue, overlap) · executors-and-workers (uni / mp /
  ray, shared-memory MessageQueue) · model-runner (MRV2 persistent batch, inputs, CUDA graph dispatch)
- **Plugins and extensibility (12):** plugin-map (all 20 extension points) · plugin-loading (groups,
  `VLLM_PLUGINS`, once-per-process guard, which process loads what) · platform-plugins (`current_platform`,
  detection order, OOT wins) · platform-interface (config hooks, component pickers) · oot-models
  (ModelRegistry, lazy "mod:Cls", subprocess inspection) · model-interfaces (`Supports*` protocols) ·
  custom-ops (`CustomOp` dispatch, `register_oot`, `PluggableLayer`, `custom_ops` toggles) ·
  attention-backend-selection (priority lists, `register_backend`) · quantization-plugins
  (`QuantizationConfig` → `QuantizeMethodBase`, `register_quantization_config`) · kv-connectors
  (factory, scheduler/worker roles) · frontend-plugins (endpoint, IO processor, tool/reasoning parsers,
  LoRA resolvers, stat loggers) · engine-hooks (logits processors, `scheduler_cls`, `worker_extension_cls`,
  model loaders, sleep-mode backends)
- **Model execution (6):** model-definition (a model file, llama.py) · weight-loading (loaders,
  stacked params, `WeightsMapper`, TP shards) · forward-context (how the Attention layer finds its
  metadata and KV cache) · attention-backends (backend / builder / impl) · compile-and-cuda-graphs
  (`@support_torch_compile`, piecewise split, CUDA-graph modes, fusion passes) · tp-layers
  (Column/Row parallel linear, vocab parallel)
- **Serving features (9):** openai-server (routes, serving classes, streaming) · renderers-and-chat-templates ·
  tool-and-reasoning-parsers · structured-outputs · speculative-decoding (methods, rejection sampling) ·
  sampling-and-logits-processors · lora-serving · multimodal-serving (overview) · pooling-and-embeddings (overview)
- **Scale and deployment (5):** config-system (EngineArgs → VllmConfig, `@config`, env vars) ·
  distributed-groups (rank layout, GroupCoordinator) · data-parallel-serving (DP coordinator, LB modes,
  multiple API servers) · expert-parallel-moe (FusedMoE at the API level, EP, EPLB) ·
  disaggregated-prefill (KV connectors in practice)
- **Operating vLLM (5):** tuning (max_num_seqs, max_num_batched_tokens, chunked prefill, CUDA-graph
  sizes, memory) · metrics-and-observability (Prometheus metrics, stat loggers, tracing) ·
  benchmarking (`vllm bench serve`) · deployment-recipes (single GPU → TP → DP+EP; `recipes/`) ·
  troubleshooting (OOM, slow start-up, preemption, common errors)

### 4.3 Part 3 — Guided tours (7)
1. **Life of a chat request** — `POST /v1/chat/completions` (stream) → render → AsyncLLM → ZMQ →
   EngineCore → scheduler → worker → sample → detokenize → SSE `[DONE]` (~16 stops).
2. **Start-up of `vllm serve`** — CLI → plugins → platform → VllmConfig → engine + workers →
   weights → memory profile → KV cache → CUDA graphs → "ready" (~17 stops).
3. **One engine step** — schedule → execute_model → sample_tokens → update_from_output; async overlap.
4. **One forward pass** — Llama in MRV2: inputs, attention metadata, forward context, Attention → backend, logits, sampler.
5. **Life of a plugin** — `pip install` → entry points → loaded in front end, engine core, each worker → platform hooks → `CustomOp.__new__` swap.
6. **Build a toy plugin** — walk through vLLM's own test plugins (`tests/plugins/*`): general plugin + OOT model + platform + custom op + stat logger.
7. **One step across GPUs** — TP=2 × DP=2: API servers, DP coordinator, two engines, broadcast, all-reduce.

### 4.4 Part 4 — Code map (~28 module pages)
Folder tree of the whole repo (generated). Module pages (latest code only):
- engine: `v1/engine/async_llm.py`, `core.py`, `core_client.py`, `input_processor.py`, `output_processor.py`
- core: `v1/core/sched/scheduler.py`, `kv_cache_manager.py`, `block_pool.py`, `kv_cache_utils.py`
- executor/worker: `v1/executor/multiproc_executor.py`, `v1/worker/gpu_worker.py`, `v1/worker/gpu/model_runner.py`
- plugins/platforms: `plugins/__init__.py`, `platforms/__init__.py`, `platforms/interface.py`, `model_executor/custom_op.py`, `model_executor/models/registry.py`
- model execution: `forward_context.py`, `layers/attention/attention.py`, `v1/attention/backend.py` + `selector.py`, `models/llama.py`, `layers/linear.py`
- serving/config: `entrypoints/cli/serve.py`, `entrypoints/openai/chat_completion/serving.py`, `engine/arg_utils.py`, `config/vllm.py`, `envs.py`
- distributed: `distributed/parallel_state.py`, `device_communicators/shm_broadcast.py`, `kv_transfer/kv_connector/factory.py`

### 4.5 Part 5 — Reference (8)
- `reference/env-vars` — all `VLLM_*` env vars (generated from `vllm/envs.py`)
- `reference/serve-flags` — all `vllm serve` flags by group (generated from the arg parser)
- `reference/config-classes` — every `@config` class and field (generated)
- `reference/metrics` — Prometheus metric names (generated from the metrics code)
- `reference/extension-points` — the 20 extension points table, cited
- `reference/vllm-vs-deepinfer` — what DeepInfer kept, changed or dropped from vLLM (with links to both sites)
- Glossary (generated) · FAQ (incl. "Where do docs and code disagree?", "Known open questions")

### 4.6 Focused option (~70 pages)
Same Parts 1-3 and 5, but merge some concepts (hybrid-kv-cache into kv-cache-manager; multimodal +
pooling into one page; distributed-groups into data-parallel-serving) and keep only ~12 module pages.
Saves about a third of the writer and fact-check runs.

## 5. Interactive visuals (vanilla JS, 6)
- **Process map stepper** (home + Tour 1): a request moves through API server → EngineCore → workers and back.
- **Scheduler simulator**: requests, token budget, chunked prefill, preemption, spec tokens.
- **Prefix cache explorer**: type two prompts; see shared hashed blocks, ref counts, LRU eviction.
- **Plugin loading timeline**: which plugin group loads in which process, and when.
- **CustomOp / backend chooser**: pick platform + config; see which `forward_*` and attention backend runs.
- **Tuning playground**: change `max_num_seqs`, `max_num_batched_tokens`, `gpu_memory_utilization`; see
  KV blocks and the batch shape (simple model, clearly marked "Simplified").

## 6. Technology
The `codebase-explainer` starter kit (same as DeepInfer): static HTML, `build.py` (standard library),
`explainer.json`, SVG diagrams from Python with `svglib.py`, Prism from cdnjs, Inter from Google Fonts.
**One addition:** a build tag `{{deepinfer <page-id> | text}}` that links to the DeepInfer site (base URL in
`explainer.json`), so every recap link has one form and can be checked.

## 7. Writing rules
STYLE.md §6-7. Every claim about the code cites `file:line` at `3e44c0d`. No claim without a source.
Latest code only. Basics from DeepInfer: one-line recap + link.

## 8. How the supervisor works
Supervisor (main session) + subagents. **Max 4 parallel until 19:56 on 2026-10-22, then max 2.**
Writer → separate fact checker → commit per batch. `progress.md` updated after each event.
`../vllm` is never changed. No push without a request.

## 9. Definition of done
- All planned pages exist (or are deferred by the user); `python3 build.py` has 0 errors.
- Every concept and tour page has a diagram, code excerpts, a deep dive and "Check yourself".
- Every batch is fact-checked; the FINAL-PASS TODO list is empty.
- Site check at 1280 and 390 px: no overflow, no JS errors, no broken links (also DeepInfer links); every visual works.
- A reader can follow Home → Basics → Tour 1 → Operating vLLM with no dead end.

## 10. Risks
| Risk | What we do |
|---|---|
| vLLM is very large | Strict scope (§2); code-map tree complete, module pages only for central files |
| Fast-moving code, stale docs | Cite code, never docs, for behaviour; docs-vs-code list in the FAQ |
| Quota runs out | 4 → 2 parallel agents; batches small; resume notes in `progress.md` |
| Operating pages drift into opinion | Every tuning claim cites the code default or the doc it comes from; mark advice clearly |

## 11. Phase 1 (review package)
| Step | Work | Who |
|---|---|---|
| 1a | Starter kit, `explainer.json`, STYLE.md, nav with all pages, core glossary, `{{deepinfer}}` tag; sample concept page **concepts/plugin-loading** | 1 subagent |
| 1b | Samples (≤4 parallel): (i) home + process-map stepper + code tree + module page `plugins/__init__.py`; (ii) basics `learn/ml-basics-recap` + `learn/serving-metrics`; (iii) Tour 1 "Life of a chat request"; (iv) generated `reference/env-vars` | 4 subagents (or 2 after 19:56) |
| 1c | Fact check of all samples; site check | fact checkers + supervisor |
| 1d | ~~Stop for user review~~ — removed 2026-10-22 (user: run unattended). Supervisor reviews samples and continues. | supervisor |

## 12. Full build (after approval) — batches
| Batch | Pages |
|---|---|
| B1 Basics | rest of learn/ |
| B2 Engine | architecture-overview, engine-client, engine-core-loop, data-contracts, async-scheduling + module pages |
| B3 Scheduler + KV | scheduler, kv-cache-manager, prefix-caching, hybrid-kv-cache + modules + scheduler/prefix visuals; Tour 3 |
| B4 Plugins I | plugin-map, platform-plugins, platform-interface, oot-models, model-interfaces + modules; Tour 5 |
| B5 Plugins II | custom-ops, attention-backend-selection, quantization-plugins, kv-connectors, frontend-plugins, engine-hooks; Tour 6; chooser visual |
| B6 Model execution | model-definition, weight-loading, forward-context, attention-backends, compile-and-cuda-graphs, tp-layers + modules; Tour 4 |
| B7 Serving features | openai-server, renderers, parsers, structured-outputs, spec-decode, sampling, lora-serving, multimodal, pooling |
| B8 Scale | config-system, distributed-groups, data-parallel-serving, expert-parallel-moe, disaggregated-prefill; Tours 2, 7 |
| B9 Operating | tuning, metrics, benchmarking, deployment-recipes, troubleshooting; tuning visual |
| B10 Reference | serve-flags, config-classes, metrics, extension-points, vllm-vs-deepinfer, FAQ |
| F Final | consistency pass + site check + README |

Waves follow the agent limit: each writer is followed by its fact checker; never more running agents than the limit.

## 13. Deferred pages
(none)
