# Survey: serving, config, start-up, distributed — vLLM @ 3e44c0db

Source: survey subagent, 2026-10-22. A guide, NOT a source: writers must open and cite the code.
Spot-checked by supervisor (all correct): envs.py:618 dict of callables; rust/README.md experimental Rust
frontend over ZMQ; parallel_state.py:2157-2163 rank layout (-1, DP, PP, PCP, TP).

## Size (git ls-files + wc by extension)
8,357 tracked files. Python 5,471 / ~2.00M lines (vllm/ ~1.13M, tests/ ~0.77M). Rust 375 / 152k. CUDA .cu 101 / 60k (+ .cuh, .cpp, .hpp, .h ~83k). YAML 546 / 71k. JSON 719 / 172k (kernel configs). Markdown 328 / 54k. Jinja 89 / 6.4k. Shell 147 / 24k.
Top level: vllm/ 3564, tests/ 2519, rust/ 537 (experimental Rust OpenAI frontend), docs/ 360, csrc/ 320 (out of scope), examples/ 272, .buildkite/ 224, recipes/ 188, benchmarks/ 141, tools/ 80, .github/ 43, requirements/ 28, .agents/ 23, docker/ 19, cmake/ 13, .claude/ 9.
vllm/: model_executor 1454; models 562 (new per-model packages); v1 429; entrypoints 245; distributed 198 (kv_transfer 88); transformers_utils 92; tool_parsers 60; reasoning 42; parser 45; multimodal 36; lora 45; compilation 44; kernels 34; utils 42; config 37; renderers 26; third_party 23; tokenizers 15; platforms 8; plugins 8; ir 7; snapshot 6; device_allocator 5; engine 5; others small.

## vllm serve start-up
1. Console script vllm = vllm.entrypoints.cli.main:main (pyproject.toml:43).
2. cli/main.py:main (:39): cli_env_setup (:68; spawn, common/startup.py:131-149); subparsers (:100-108); parse (:110); validate (:114); dispatch (:117).
3. make_arg_parser (cli/serve.py:172; common/cli_args.py:422) → AsyncEngineArgs.add_cli_args (arg_utils.py:3237): load_general_plugins (:3243), current_platform.pre_register_and_update(parser) (:3257); flags from config dataclasses via get_kwargs (:443).
4. ServeSubcommand.cmd (serve.py:54): gRPC (:59), multi-port DP supervisor (:148), headless (:150), multi-API-server / Rust frontend (:152), single: uvloop.run(run_server(args)) (:156).
5. setup_server binds socket before engine start (launchers/api_server/entry.py:164-176; launcher.py:365).
6. build_async_engine_client → AsyncEngineArgs.from_cli_args (entry.py:52; arg_utils.py:1962); __post_init__ loads plugins (:922-924).
7. create_engine_config (entry.py:82 → arg_utils.py:2234): current_platform resolved lazily (platforms/__init__.py:303-319, :240, :231); pre_register_and_update + validate_environ (:2246-2250); sub-configs; VllmConfig(...) (:2870).
8. AsyncLLM.from_vllm_config (entry.py:94; async_llm.py:250) → make_async_mp_client (async_llm.py:193; core_client.py:140).
9. MPClient binds ROUTER + PULL, launch_core_engines (core_client.py:671-695; v1/engine/utils.py:1201): DPCoordinator if needed (:1261); EngineCoreProc.run_engine_core per local DP rank (CoreEngineProcManager utils.py:237, :282-283).
10. EngineCore.__init__: plugins (core.py:130-132); executor (:147) via Executor.get_class (executor/abstract.py:64).
11. MultiprocExecutor._init_executor (multiproc_executor.py:119): rpc_broadcast_mq (:165); WorkerProc.worker_main per GPU (:713-746, :862).
12. init_device (multiproc_executor.py:680 → gpu_worker.py:446) → init_worker_distributed_environment (gpu_worker.py:1678): torch.distributed (parallel_state.py:1836), initialize_model_parallel (:2388, :2078).
13. load_model (multiproc_executor.py:688 → gpu_worker.py:591 → gpu_model_runner.py:4959).
14. _initialize_kv_caches (core.py:246): specs (:253), determine_available_memory (:300 → gpu_worker.py:638), profile_run (:657/:697).
15. initialize_from_config (core.py:362 → gpu_worker.py:857 → initialize_kv_cache :898).
16. compile_or_warm_up_model (core.py:379 → gpu_worker.py:924) → capture_model (:985).
17. startup_handshake (core.py:1389; client wait_for_engine_startup utils.py:1398) → build_and_serve (entry.py:115) → build_app (launchers/app.py:19) → init_app_state (api_server/app_state.py:24) → serve_http → uvicorn server.serve(sockets=[sock]) (launcher.py:177, :238).

## One streaming /v1/chat/completions
1. Route create_chat_completion (openai/chat_completion/api_router.py:43-59; @with_cancellation, @load_aware_call); handler app.state.openai_serving_chat (generate/api_router.py:149).
2. OpenAIServingChat.create_chat_completion (chat_completion/serving.py:287) → _create_chat_completion (:302).
3. render_chat_request → online_renderer.render_chat (serving.py:264-285; renderers/online_renderer.py:225); chat template via renderers/hf.py:safe_apply_chat_template (:805); tokenize.
4. engine_client.generate (serving.py:396) → AsyncLLM.generate (async_llm.py:726) → add_request (:397): process_inputs → EngineCoreRequest (:483/:500); RequestOutputCollector (:529).
5. AsyncMPClient.add_request_async → _send_input (ADD) over ROUTER (core_client.py:1333, :1275).
6. Engine: process_input_sockets DEALER (core.py:1921/1939); run_busy_loop (:1609) → step (:657): schedule → execute_model (:668) → update_from_output (:681); process_output_sockets PUSH (:2030/:2049).
7. API: process_outputs_socket task (core_client.py:1208); output_handler (async_llm.py:876) → process_outputs (output_processor.py:728) → per-request queues; generate yields (async_llm.py:~790).
8. chat_completion_stream_generator (serving.py:473) SSE chunks, `data: [DONE]` (:508); StreamingResponse text/event-stream + keep-alive (api_router.py:82).

## Processes and IPC
- API servers: N via --api-server-count (default DP size internal LB, 1 external, local DP hybrid; serve.py:107-130). run_multi_api_server (serve.py:269): bind_engine_zmq_listeners (utils.py:1159), engines, APIServerProcessManager (v1/utils.py:167) → run_api_server_worker_proc (:509); shared listen socket reuse_port (serve.py:298). Many-to-many API servers ↔ engines (docs/design/arch_overview.md:75).
- Front end ROUTER (out) / PULL (in) (core_client.py:655-681); engine DEALER (in) / PUSH (out) (core.py:1939, :2049); msgpack.
- Engine core one per DP rank (arch_overview.md:83); DPEngineCoreProc (core.py:2269).
- DPCoordinator (coordinator.py:23; proc :148): queue stats for LB, request wave, START_DP_WAVE lockstep for MoE (docstring :24-56); ZMQ XPUB/PULL. DPLBAsyncMPClient.get_core_engine_for_request (core_client.py:1743).
- Engine → workers: shm ring MessageQueue (shm_broadcast.py:265 ShmRingBuffer, :483 MessageQueue; multiproc_executor.py:165, :428); per-worker worker_response_mq (:625); cross-node ZMQ (create_mq_broadcaster :629). Tensors over NCCL / custom all-reduce.
- Other modes: --headless (serve.py:181); dp_supervisor; Ray (CoreEngineActorManager utils.py:493; executor/ray_executor_v2.py).

## Configuration
@config = pydantic dataclass extra="forbid" (config/utils.py:52-82). VllmConfig (vllm.py:348; compute_hash :459; __post_init__ :1492; set_current_vllm_config :3517). ModelConfig (model.py:152), ModelArchitectureConfig (model_arch.py:16), CacheConfig (cache.py:179), ParallelConfig (parallel.py:137), EPLBConfig (:64), SchedulerConfig (scheduler.py:25), CompilationConfig (compilation.py:383) + PassConfig (:114) + DynamicShapesConfig (:343), DeviceConfig, LoadConfig, LoRAConfig, SpeculativeConfig (speculative.py:387), StructuredOutputsConfig, ObservabilityConfig, KVTransferConfig (kv_transfer.py:57), KVEventsConfig, ECTransferConfig, EncoderCacheManagerConfig, MultiModalConfig, PoolerConfig, SpeechToTextConfig, AttentionConfig (attention.py:37), HiSparseConfig, KernelConfig (kernel.py:271), IrOpPriorityConfig, Offload configs, MambaConfig, EngramConfig, ProfilerConfig, LoggingConfig, ReasoningConfig, WatermarkConfig, DiffusionConfig, AuxOutputConfig, FaultToleranceConfig, WatchdogConfig, WeightTransferConfig.
EngineArgs (arg_utils.py:483) flat; defaults from config classes (:542); groups per config class with docstring help (:952-958); create_engine_config (:2234-2870).
envs.py: environment_variables dict of lambdas (:618), env_with_choices (:400), lazy module __getattr__ (:2291), enable_envs_cache; TYPE_CHECKING block types/defaults (:14-399); ~317 entries.
Generators: env vars via ast on envs.py (markers env-vars-definition :578/:2288); config fields via dataclasses.fields + config/utils.py get_attr_docs (:201); CLI flags via make_arg_parser and parser._action_groups — reuse idea of docs/mkdocs/gen_files/generate_argparse.py (torch mocked).

## Distributed
initialize_model_parallel (parallel_state.py:2078); ranks arange(world).reshape(-1, DP, PP, PCP, TP) (:2157-2163). Groups _TP :1590, _PP :1640, _DP :1648, _EP :1656, _EPLB :1668, _DCP :1624, _PCP :1680, _ETP/_ENGRAM_DP :1598/:1607, _WORLD/_INNER_DP_WORLD :1492. GroupCoordinator (:427) + device communicators (pynccl, custom all-reduce, symm_mem, shm).
Implemented: TP, PP, DP, EP (enable_expert_parallel, reuses DP×TP ranks), EPLB, PCP, DCP, elastic EP, DBO (parallel.py:140-389). No ZeRO/FSDP.
TP layers: ColumnParallelLinear (linear.py:436; gather_output :627), MergedColumnParallelLinear :686, QKVParallelLinear :1011, RowParallelLinear :1639 (all-reduce :1804), VocabParallelEmbedding (vocab_parallel_embedding.py:206).
PP: make_layers (models/utils.py:886) + get_pp_indices (distributed/utils.py:190) + PPMissingLayer (:852); IntermediateTensors (llama.py:378-394).
kv_transfer: kv_connector/v1/ NIXL, LMCache, Mooncake, hf3fs, FlexKV, example; disagg prefill (docs/assets/features/disagg_prefill/high_level_design.png).

## Central files
cli/main.py 123 · cli/serve.py 407 · launchers/api_server/entry.py 230 · launchers/launcher.py 392 · openai/chat_completion/serving.py 1332 · engine/arg_utils.py 3266 · config/vllm.py 3617 · config/parallel.py 1201 · envs.py 2493 · v1/engine/utils.py · v1/engine/coordinator.py 462 · multiproc_executor.py 1147 · parallel_state.py 2767 · shm_broadcast.py 1148.

## Docs / images
docs/design: arch_overview, multiprocessing, plugin_system, cuda_graphs, torch_compile, metrics, optimization_levels, shm_tensor_arena. docs/serving: data_parallel_deployment, parallelism_scaling, expert_parallel_deployment, context_parallel_deployment, distributed_troubleshooting, online_serving/*. docs/configuration: engine_args, serve_args, env_vars, conserving_memory, optimization.
Images: docs/assets/design/arch_overview/*.png, hierarchy.png, docs/assets/deployment/dp_{internal,external}_lb.png, docs/assets/design/cuda_graphs/*.png, vllm/distributed/kv_transfer/disagg_prefill_workflow.jpg.

## Git / surprises
23,594 commits, full clone, first 2023-02-09. No submodules.
rust/: Cargo workspace vllm-frontend-rs, experimental Rust OpenAI frontend (axum + gRPC) over the same ZMQ/msgpack boundary; VLLM_USE_RUST_FRONTEND=1; RustFrontendProcessManager (v1/utils.py:280; serve.py:346-365); setup.py setuptools-rust (:21, :49).
vllm/models/ new per-model tree; .agents/, .claude/, AGENTS.md, recipes/; `vllm --omni` hands off to vllm-omni (cli/main.py:17-36).
