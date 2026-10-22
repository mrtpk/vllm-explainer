# Survey: core engine (V1) — vLLM @ 3e44c0db

Source: survey subagent, 2026-10-22. A guide, NOT a source: writers must open and cite the code.
Spot-checked by supervisor (all correct): scheduler.py:624-633 "no prefill/decode phase" note; envs.py:162
VLLM_ENABLE_V1_MULTIPROCESSING=True; parallel.py:1099-1100 world_size 1 → "uni"; worker/gpu/README.md:1
"[Experimental]"; cache.py:254-256 prefix caching on, sha256.

## Size
| Folder | Files | Lines | Role |
|---|---|---|---|
| vllm/v1/engine | 15 | 12,217 | front end (LLMEngine/AsyncLLM, input/output processors, detokenizer), EngineCore loop, ZMQ clients |
| vllm/v1/core | 18 | 13,780 | scheduler, KV cache manager, block pool, prefix-cache hashing, KV planning |
| vllm/v1/executor | 8 | 3,056 | uni / mp / ray / external_launcher |
| vllm/v1/worker | 124 | 45,201 | GPU worker + two model runners; `gpu/` (Model Runner V2) = 26,886 |
| single files | 4 | 3,274 | request.py 412, outputs.py 405, entrypoints/llm.py 910, sampling_params.py 1,547 |
KV specs: vllm/v1/kv_cache_interface.py (1,640).

## Flow LLM.generate() → text (offline)
1. LLM.__init__ → LLMEngine.from_engine_args (entrypoints/llm.py:341). Multiprocessing on by default → EngineCore in its own process (envs.py:162; llm_engine.py:186).
2. LLM.generate (llm.py:417) → _run_completion (entrypoints/common/offline.py:340; OfflineInferenceMixin).
3. _add_completion_requests (offline.py:304) renders/tokenizes; _add_request sets FINAL_ONLY (offline.py:568-585).
4. LLMEngine.add_request (v1/engine/llm_engine.py:229) → InputProcessor.process_inputs (input_processor.py:373) → EngineCoreRequest (msgspec Struct, v1/engine/__init__.py:113).
5. OutputProcessor.add_request registers RequestState + detokenizer (llm_engine.py:297; output_processor.py:663).
6. SyncMPClient.add_request (core_client.py:1058) → _send_input (core_client.py:1038): msgpack over ZMQ ROUTER.
7. EngineCore input thread process_input_sockets (core.py:1921); preprocess_add_request (core.py:1114) → Request.from_engine_core_request (request.py:244), block hashes.
8. run_busy_loop (core.py:1609): _process_input_queue, _process_engine_step (core.py:1666) → step_fn.
9. EngineCore.step (core.py:657) → Scheduler.schedule (sched/scheduler.py:622) → SchedulerOutput.
10. executor.execute_model(non_block=True) (core.py:668). MP: shm MessageQueue broadcast (multiproc_executor.py:342, :428); uni: in-process (uniproc_executor.py:119).
11. Worker.execute_model (gpu_worker.py:1325) → GPUModelRunner.execute_model (V2: worker/gpu/model_runner.py:1742): update states, prepare inputs + attn metadata, pick CUDA graph, forward; returns None (model_runner.py:2090-2111).
12. sample_tokens (core.py:676; gpu_worker.py:1318; model_runner.py:2114); async D2H via AsyncOutput (gpu/async_utils.py:115).
13. Scheduler.update_from_output (scheduler.py:2117): append tokens, check_stop (sched/utils.py:98, called scheduler.py:2603), free finished, build EngineCoreOutputs.
14. Output thread process_output_sockets (core.py:2030) → ZMQ PULL; client EngineCoreOutputQueueThread decodes (core_client.py:985-1021).
15. LLMEngine.step (llm_engine.py:319) → OutputProcessor.process_outputs (output_processor.py:728); detokenizer.update (output_processor.py:825) wraps tokenizers DecodeStream (detokenizer.py:179); stop-string abort (llm_engine.py:353).
16. _run_engine loops until done, sorts by request id (offline.py:589-642).

Online: AsyncLLM (async_llm.py:93), AsyncMPClient (core_client.py:1143, zmq.asyncio); add_request (async_llm.py:397) makes RequestOutputCollector (async_llm.py:529); generate (async_llm.py:726) yields; output_handler task (async_llm.py:876) chunks by VLLM_V1_OUTPUT_PROC_CHUNK_SIZE (async_llm.py:905-912). DP>1: DPAsyncMPClient / DPLBAsyncMPClient (core_client.py:1461, :1702). AsyncLLMEngine is an alias (vllm/engine/async_llm_engine.py:6).

## Architecture
- Processes: front end (LLM or API server: LLMEngine/AsyncLLM, renderer, Input/OutputProcessor, EngineCoreClient); EngineCore (EngineCoreProc core.py:1153; owns Scheduler, StructuredOutputManager, Executor core.py:147, :177); worker processes only when world_size>1 (else "uni", in EngineCore process; config/parallel.py:1099-1100). InprocClient (core_client.py:364) when MP off. docs/design/arch_overview.md:107-131 process-count table.
- Threads: EngineCoreProc input + output IO threads (core.py:1236-1265); sync client output thread (core_client.py:1016); MP worker async_output_busy_loop (multiproc_executor.py:693, :1024).
- IPC: ZMQ ROUTER (in) / PULL (out) (core_client.py:651-682), MsgpackEncoder/Decoder (core_client.py:731-732); tensors out-of-band (engine/tensor_ipc.py). EngineCore→workers: shm MessageQueue broadcast, replies from output_rank only (multiproc_executor.py:165, :376).
- Ownership: LLM → LLMEngine → {InputProcessor, OutputProcessor(RequestState, detokenizer), EngineCoreClient} (llm_engine.py:101-123). EngineCore → Executor → Worker(s) → GPUModelRunner; EngineCore → Scheduler → KVCacheManager → KVCacheCoordinator → BlockPool + SingleTypeKVCacheManager per group.

## Scheduler
- schedule() scheduler.py:622; token_budget = max_num_scheduled_tokens (:642).
- RUNNING pass (:715): num_new_tokens = num_tokens_with_spec + placeholders − num_computed_tokens, capped by long_prefill_token_threshold, budget, max_model_len (:769-786); allocate_slots (:834); requests that can't get tokens are skipped with `continue` (:808-825) → FCFS not strict.
- Preemption: running[-1] (FCFS) or max (priority, arrival) (:849-856); _preempt_request (:1658) frees blocks, num_computed_tokens=0 → recompute, no swap.
- WAITING pass (:956) only if no preemption this step; caps active requests (:979); get_computed_blocks for prefix hits (:1094); no chunked prefill → stop if prompt does not fit (:1223-1229).
- Spec tokens in the same token count (:924-940); drafts via update_draft_token_ids (:2647). AsyncScheduler (sched/async_scheduler.py:14).

## KV cache
- Sizing: EngineCore._initialize_kv_caches (core.py:246) → determine_available_memory (gpu_worker.py:638; profile_run, CUDA-graph estimate; :735-739) → get_kv_cache_configs, num_blocks = available // bytes_per_block (core/kv_cache_planning.py:681) → cache_config.num_gpu_blocks (core.py:324).
- BlockPool (block_pool.py:136): KVCacheBlocks, FreeKVCacheBlockQueue (doubly linked, LRU), cached_block_hash_to_block; block 0 = null_block (:172-189).
- Hashing: hash_block_tokens(parent_hash, tokens, extra_keys) (kv_cache_utils.py:707); extra keys for MM, LoRA, prompt embeds (:522-637); per-request hasher core.py:216-225.
- Hybrid: specs FullAttention/SlidingWindow/MLA/Mamba/ChunkedLocal (kv_cache_interface.py:604-1212); KVCacheGroupSpec/KVCacheConfig (:1485, :1506); get_kv_cache_groups (kv_cache_planning.py:1113); HybridKVCacheCoordinator (kv_cache_coordinator.py:609); SlidingWindowManager, MambaManager (single_type_kv_cache_manager.py:942, :1424).

## Async scheduling + CUDA graphs
- Async scheduling default on unless incompatible (config/vllm.py:1664-1698) → AsyncScheduler (config/scheduler.py:233). max_concurrent_batches 2 (or pp+1 with V2) (config/vllm.py:587-597) → step_with_batch_queue (core.py:211, :227, :698). Sampled tokens stay on GPU in last_sampled_tokens, Triton kernel writes them to next inputs (gpu/input_batch.py:410-450); copy stream + event (async_utils.py:115-172). docs/design/model_runner_v2.md:49-102 races.
- CUDAGraphMode NONE/PIECEWISE/FULL/FULL_DECODE_ONLY/FULL_AND_PIECEWISE (config/compilation.py:53). V2: CudaGraphManager._init_candidates (gpu/cudagraph_utils.py:256), dispatch (:523-553), run_fullgraph / run_pw_graph / eager (model_runner.py:2032-2071). V1: CudagraphDispatcher (v1/cudagraph_dispatcher.py:233; gpu_model_runner.py:868).

## Concepts (G = taught on DeepInfer site, N = new)
G: paged KV blocks; continuous batching over a token budget; chunked prefill; preemption by recompute; CUDA graph modes; async scheduling.
N: unified computed-vs-total scheduling (no phases); front end / EngineCore split + ZMQ/msgpack; data contracts (EngineCoreRequest, Request, SchedulerOutput, ModelRunnerOutput, EngineCoreOutputs); hash-chained prefix caching + LRU free queue; KV cache groups/specs + hybrid coordinator; memory profiling + gpu_memory_utilization; executor backends + shm broadcast RPC; Model Runner V1 vs V2 + persistent batch; incremental detokenization + stop-string abort; execute_model/sample_tokens split (grammar bitmask overlap).

## Central files
scheduler.py 3,539 · engine/core.py 2,847 · core_client.py 2,162 · async_llm.py 1,445 · output_processor.py 1,047 · llm_engine.py 524 · kv_cache_manager.py 922 · block_pool.py 905 · kv_cache_utils.py 1,217 · kv_cache_planning.py 1,704 · gpu_worker.py 1,720 · worker/gpu/model_runner.py 2,483 (V2, default) · gpu_model_runner.py 7,184 (V1 fallback) · multiproc_executor.py 1,147.

## Docs and images
docs/design: arch_overview.md, prefix_caching.md, hybrid_kv_cache_manager.md, model_runner_v2.md, multiprocessing.md, cuda_graphs.md, metrics.md (paged_attention.md = kernels, out of scope).
Images docs/assets/design/: arch_overview/{entrypoints.excalidraw.png, v1_process_architecture_tp4.png, v1_process_architecture_tp2_dp4.png, llm_engine.excalidraw.png}; hierarchy.png; prefix_caching/*; hybrid_kv_cache_manager/*; model_runner_v2/*; cuda_graphs/*.

## Surprises
- Model Runner V2 is the default when Triton is present and features allow (config/vllm.py:713-748; gpu_worker.py:567-583), though its README says "[Experimental]" and the design doc "not yet feature-complete". Teach V2 as main, V1 as fallback.
- V0 is gone: vllm/engine/*.py are aliases. LLM docstring still mentions AsyncLLMEngine (llm.py:173).
- LLM is thin: logic in mixins under vllm/entrypoints/common/offline.py; rendering in vllm/renderers.
- Cascade attention removed (stubs: kv_cache_manager.py:692; sched/output.py:311).
- Single GPU: LLM still has a separate EngineCore process, but no separate worker process.
