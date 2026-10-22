# Survey: plugins and extensibility — vLLM @ 3e44c0db

Source: survey subagent, 2026-10-22. A guide, NOT a source: writers must open and cite the code.
Spot-checked by supervisor (all correct): plugins/__init__.py:16-34 groups + guard; :60-90 load/skip/guard
(import errors swallowed, func() errors not caught); platforms/__init__.py:270-292 OOT wins over builtin;
custom_op.py:335-356 register_oot (string positional → TypeError); docs/design/custom_op.md:11 shows the
broken `@CustomOp.register_oot("op_name")` form; core.py:130-132 load_general_plugins in EngineCore.

## Extension points
| # | Point | Register | Loaded at | When / process | Changes |
|---|---|---|---|---|---|
| 1 | General plugins | entry point `vllm.general_plugins` (plugins/__init__.py:18); vLLM ships 2 (pyproject.toml:45-47, LoRA resolvers) | load_general_plugins() :77; called arg_utils.py:924 (EngineArgs.__post_init__), :3243 (add_cli_args), core.py:132 (EngineCore.__init__), worker_base.py:284 (init_worker), registry.py:1612 (inspection subprocess) | every process | anything; the "host" that calls the registries below |
| 2 | Platform plugins | `vllm.platform_plugins` (:23); func returns qualname or None | platforms/__init__.py:250 | lazily on first `current_platform` access (:303-322), every process | whole hardware layer |
| 3 | IO processor | `vllm.io_processor_plugins`; selected by `io_processor_plugin` in hf_config/ModelConfig (config/model.py:410) | io_processors/__init__.py:59 | pooling serving, front end only | pre/post-processing of pooling requests (interface.py:18-59) |
| 4 | Stat logger | `vllm.stat_logger_plugins`; StatLoggerBase subclass (loggers.py:84-97) | async_llm.py:156 | AsyncLLM construction, front end | metrics sinks |
| 5 | Endpoint plugins | `vllm.endpoint_plugins` → factory returning EndpointPlugin (endpoint_plugins/interface.py:47) | plugins/__init__.py:93-159 via attach_endpoint_plugins (interface.py:108; launchers/app.py:12) | API server only; opt-in, needs VLLM_PLUGINS | add or REPLACE routes (attach_router), app state (init_state) |
| 6 | Logits processors | `vllm.logits_processors` (v1/sample/logits_processor/__init__.py:48); also classes / "mod:Cls" via logits_processors= | __init__.py:57-84; built gpu_model_runner.py:720; V2 loader v1/worker/gpu/sample/logits_processor/loader.py:46 | worker | batch-level logits |
| 7 | Models | ModelRegistry.register_model(arch, cls or "mod:Cls") (registry.py:1160) | from a general plugin | every process | new architectures |
| 8 | CustomOp / PluggableLayer | @X.register_oot (custom_op.py:335; PluggableLayer :83) | __new__ swaps class (:107-126) | model-building processes (workers) | kernel layer or whole layer |
| 9 | Attention backend | register_backend(enum or "NAME", class_path) (v1/attention/backends/registry.py:348) + Platform.get_attn_backend_cls | selector.py:218 | worker | attention impl |
| 10 | Quantization | @register_quantization_config("name") (quantization/__init__.py:61); appends to current_platform.supported_quantization (:96-97) | — | every process | new --quantization |
| 11 | Model loader | @register_model_loader(fmt) (model_loader/__init__.py:69) | not checked | worker | new --load-format |
| 12 | KV connector | KVConnectorFactory.register_connector (kv_connector/factory.py:31) or kv_connector_module_path (wins, :103-108) | create_connector(config, role) (:43) | engine core (scheduler role) + workers | KV transfer/offload |
| 13 | EC connector | ECConnectorFactory.register_connector (ec_connector/factory.py:24) | not checked | — | encoder-cache transfer |
| 14 | Tool parser | ToolParserManager.register_module / register_lazy_module (abstract_tool_parser.py:314, 328); --tool-parser-plugin via import_plugin (import_utils.py:81) | cli_args.py:576-583 | front end | tool-call parsing |
| 15 | Reasoning parser | ReasoningParserManager.register_module (abs_reasoning_parsers.py:317, 331); --reasoning-parser-plugin | cli_args.py:582-583 AND v1/structured_output/__init__.py:103-104 | front end + engine core | reasoning extraction |
| 16 | LoRA resolver | LoRAResolverRegistry.register_resolver (lora/resolver.py:52) | read in openai/models/serving.py:114 | registered by general plugins; used in front end | unknown adapter names |
| 17 | Scheduler | scheduler_cls config (config/scheduler.py:231) | core.py:164 | engine core | scheduling policy |
| 18 | Worker class / extension | parallel_config.worker_cls, worker_extension_cls (config/parallel.py:306, 309) | worker_base.py:293-320 | worker | worker class, extra RPC methods |
| 19 | Sleep-mode backend | SleepModeBackendFactory.register_backend (sleep_mode_backend.py:171) | — | worker | sleep/wake memory |
| 20 | KV-cache dtype | @register_kv_cache_dtype (config/cache.py:111) | — | at platform activation (docstring) | new --kv-cache-dtype |

## Loading
1. load_plugins_by_group (plugins/__init__.py:36) → importlib.metadata.entry_points(group=) (:42).
2. VLLM_PLUGINS filter: unset → None (all), else comma list (envs.py:1163-1167); matches entry-point NAME (:64).
3. ep.load(), returns {name: callable} without calling (:62-74); load() errors logged and skipped (:71-72).
4. load_general_plugins (:77): module global plugins_loaded guard (:33, :82-85), flag set BEFORE running; calls each func (:89-90); exceptions not caught.
5. Every process: front end (arg_utils.py:924/3243), engine core (core.py:132), each worker (worker_base.py:284), registry inspection subprocess (registry.py:1612). Registries are module-level dicts → per process. Docstring (:78-80): plugins must be re-entrant.

## Platform plugins
- current_platform lazy via module __getattr__ (platforms/__init__.py:303-322); comment :306-315 why (OOT can subclass Platform before detection).
- resolve_current_platform_cls_qualname (:240): VLLM_TARGET_DEVICE=cpu short-circuit (:244-248); load OOT platform plugins (:250); builtin detectors tpu, cuda, rocm, xpu, cpu (:231-237) + OOT detectors (:254-265); one OOT wins over builtins (:277-279); 2 OOT → error (:272); 2 builtins → error (:280); none → UnspecifiedPlatform (:291). _init_trace (:321).
- Platform (platforms/interface.py): identity _enum (OOT → is_out_of_tree :218), device_name/device_type (:138-140), dispatch_key (:145), ray_device_key (:150), device_control_env_var (:160), dist_backend (:175), supported_quantization (:177). Config hooks: pre_register_and_update (:639; called arg_utils.py:2246, 3257; for OOT quant configs), apply_config_platform_defaults (:654; vllm.py:1804), check_and_update_config (:667; vllm.py:2046; sets worker_cls, CUDA cuda.py:330-331). Pickers: get_attn_backend_cls (:389; selector.py:218), get_device_communicator_cls (:1219; parallel_state.py:551), get_punica_wrapper (:1204; punica_selector.py:14), get_static_graph_wrapper_cls (:1329; compilation/backends.py:691), get_pass_manager_cls / get_compile_backend (:247, :254), get_kv_cache_config_builder_cls (:1471). Kernels: import_kernels (:366), register_triton_kernel_overrides (:376; worker_base.py:290). Platform.__getattr__ (:1300) forwards to torch.<device_type>.
- CustomOp.dispatch_forward checks is_out_of_tree() → forward_oot (custom_op.py:202-203).

## Out-of-tree models
- register_model (registry.py:1160-1203): nn.Module class → _RegisteredModel, inspected at once (:975-978); "module:Class" → _LazyRegisteredModel (:1189-1194). Docstring: use strings to avoid "Cannot re-initialize CUDA in forked subprocess" (:1170-1174). Same arch overwrites silently (:1180-1186). In-tree models also lazy (:1565-1573).
- inspect_model_cls (:1072-1118): hashes model source, caches _ModelInfo JSON in VLLM_CACHE_ROOT/modelinfos (:996-1001); miss → _run_in_subprocess (:1578) cloudpickles to `python -m vllm.model_executor.models.registry` (:823); _run loads general plugins first (:1608-1612).
- _ModelInfo (:891-915) from Protocols in models/interfaces.py: SupportsMultiModal (:178), SupportsLoRA (:720), SupportsPP (:808), HasInnerState (:937), IsAttentionFree (:963); ClassVar flags (:723); supports_lora warns on missing attrs (:760-780).

## CustomOp
- @CustomOp.register("name") → op_registry (custom_op.py:311-323). __init__ binds forward once (:128-134). dispatch_forward (:172-205): disabled → compiled forward_native; enabled → rocm forward_hip, cpu forward_cpu, tpu forward_tpu, xpu forward_xpu, OOT forward_oot, else forward_cuda. Defaults: hip→cuda; xpu/cpu/tpu/oot→native (:147-170).
- enabled() (:269-290): +name/-name in compilation_config.custom_ops; default_on() needs exactly one of all/none (:293-307); default none with Inductor + compile, else all (config/vllm.py:1819-1826).
- register_oot (:335-356) → op_registry_oot keyed by in-tree CLASS NAME; __new__ instantiates OOT class (:107-126). PluggableLayer (:32-100) whole-layer, no per-platform dispatch.

## Minimal examples
- Model plugin: tests/plugins/vllm_add_dummy_model/setup.py:6-13 + __init__.py:7-27.
- Platform plugin: tests/plugins/vllm_add_dummy_platform/setup.py:10-17, __init__.py:5-13, dummy_platform.py:15-32, dummy_custom_ops.py:9-19.
- Stat logger: tests/plugins/vllm_add_dummy_stat_logger/setup.py:6-15 + dummy_stat_logger.py:7-30.
- Endpoint: tests/plugins/vllm_add_dummy_endpoint_plugin/…/__init__.py:19-37 + setup.py:10-14.
- IO processor: tests/plugins/prithvi_io_processor_plugin/setup.py:10-14 + __init__.py:5-6; examples/pooling/plugin/prithvi_geospatial_mae_io_processor.py.
- Scheduler swap: tests/plugins_tests/test_scheduler_plugins.py:12-36. Platform ordering: tests/plugins_tests/test_platform_plugins.py:37-48.
- Logits processor: examples/features/logits_processor/custom.py:50, :123-125.

## Docs accuracy
- docs/design/plugin_system.md mostly right; gaps: line 7 implies all groups load everywhere; no vllm.logits_processors; no pre_register_and_update / apply_config_platform_defaults; doesn't say OOT beats builtin; line 144 possibly stale (register_triton_kernel_overrides exists).
- docs/design/custom_op.md lines 11, 285: `@CustomOp.register_oot("MMEncoderAttention")` is WRONG (TypeError). Lines 13-47 correct.
- docs/contributing/model/registration.md matches. docs/design/lora_resolver_plugins.md:28 says VLLM_PLUGINS "must include" (unset = all; its own line 40 agrees with code). endpoint_plugins.md ok. custom_logitsprocs.md:158-172 ok.

## Concepts / proposed pages
Concepts: entry points; plugin groups; VLLM_PLUGINS; once-per-process guard; why reload per process; lazy current_platform; detection order; Platform interface; qualname strings / resolve_obj_by_qualname; ModelRegistry eager vs lazy; capability Protocols / _ModelInfo; subprocess inspection + cache; CustomOp dispatch + custom_ops; register_oot / PluggableLayer; name-keyed registries.
Pages: what a plugin is; loading and processes; platform plugins + current_platform; Platform interface; OOT models + registry; CustomOp + OOT ops; feature registries; front-end plugins; engine-side hooks; tour "Life of a plugin".

## Surprises
- VLLM_PLUGINS is one list for all groups: allowlisting only an endpoint plugin disables general plugins (incl. built-in LoRA resolvers) and OOT platforms (silent fallback to builtin).
- vllm.logits_processors ignores VLLM_PLUGINS; load failure raises.
- OOT platform detector called twice (platforms/__init__.py:257, :278).
- Import errors swallowed; plugin function errors crash start-up.
- register_oot renames OOT class .name to in-tree class name → +/-op toggles may not apply (untested).
- vllm/model_executor/hw_agnostic/custom_op.py: second CustomOp with its own op_registry_oot.
- Test plugin's attention backend imports nonexistent vllm.attention.backends.placeholder_attn (stale).
- Endpoint plugins can override built-in routes (why opt-in).
- Parser plugin paths imported only if len(path) > 3 (cli_args.py:579, 582).
- TPU detector can return external tpu_inference TpuPlatform with Pathways (platforms/__init__.py:37-39).
- validate_logits_processors_parameters: no caller found (unconfirmed).
