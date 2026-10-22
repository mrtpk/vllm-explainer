# Survey: model execution and features — vLLM @ 3e44c0db

Source: survey subagent, 2026-10-22. A guide, NOT a source: writers must open and cite the code.
Spot-checked by supervisor (all correct): vllm/attention/ does not exist; vllm/models/ has per-family packages;
FusedMoEFactory is a function (fused_moe/layer.py:92); attention.py:570-574 unified custom ops.

## Key facts
- Model Runner V2 default when Triton present + features allow (config/vllm.py:713-748); V1 fallback gpu_model_runner.py (7,184).
- Second model tree vllm/models/ (557 .py, 13 per-family packages, nvidia/ + amd/ subfolders); registry.py:83-208 points there.
- Attention layer in vllm/model_executor/layers/attention/.

## Size (.py files / lines)
model_executor 837/383,603 (models 309/203,892; layers 366/136,378; fused_moe 115/52,907; quantization 130/40,993; rotary 21/3,920; model_loader 29/8,115) · vllm/models 557/203,840 · v1/attention 114/68,489 (backends 66/39,466) · compilation 44/15,998 · v1/sample 15/5,932 · v1/spec_decode 19/6,373 · gpu/spec_decode ~9,404 · v1/structured_output 8/2,814 · lora 44/14,775 · multimodal 36/13,444 · ir 7/876 · kernels 20/3,428 · v1/pool 7/849 · forward_context.py 392. ~414 registry architecture entries (_TEXT_GENERATION_MODELS :73, _EMBEDDING_MODELS :217, _MULTIMODAL_MODELS :348, _SPECULATIVE_DECODING_MODELS :647).

## One forward pass (MRV2, Llama)
1. load_model (gpu/model_runner.py:385) → get_model_loader().load_model (model_loader/__init__.py:123, base_loader.py:63); each Attention registers in compilation_config.static_forward_context[prefix] (attention.py:445).
2. initialize_kv_cache (model_runner.py:622) → init_kv_cache (gpu/attn_utils.py:397) → bind_kv_cache_to_layers (v1/worker/utils.py:868): layer.kv_cache.
3. execute_model (:1742): prepare_inputs (:1835), prepare_attn (:1838, :1577), build_slot_mappings_by_layer (:1926).
4. model_state.prepare_attn (:1940 → model_states/default.py:167) → build_attn_metadata (attn_utils.py:458): CommonAttentionMetadata per KV group → backend builder.build → {layer_name: metadata} (:574-580).
5. set_forward_context(...) (model_runner.py:2041; forward_context.py:278); no_compile_layers = static_forward_context (forward_context.py:141-143, 248).
6. run_fullgraph (:2032) / run_pw_graph (:2065) / eager self.model(...) (:2070).
7. LlamaForCausalLM.forward (llama.py:534) → LlamaModel.forward (:406; @support_torch_compile :337) → layers loop (:428) → LlamaDecoderLayer.forward (:313).
8. LlamaAttention.forward (:224): qkv_proj → split → rotary_emb → self.attn → o_proj.
9. Attention.forward (attention.py:490) → torch.ops.vllm.unified_kv_cache_update / unified_attention_with_output (:570-573).
10. get_attention_context(layer_name) (attention.py:682) reads forward-context metadata + kv_cache; impl.do_kv_cache_update (:741), impl.forward (:784). String layer name → opaque op for torch.compile.
11. sample_tokens (:2114) → sample (:1615); compute_logits (llama.py:546) → LogitsProcessor (layers/logits_processor.py:98; TP gather :122); grammar bitmask (model_runner.py:1643).
12. Sampler.__call__ (gpu/sample/sampler.py:134): bias/penalties (:71), temperature, min_p, top-k/p, Gumbel (:23); RejectionSampler with drafts (model_runner.py:1670); speculator.propose (:2271).

## Model file pattern (llama.py, 585)
LlamaMLP :82 / LlamaAttention :125 / LlamaDecoderLayer :251 → LlamaModel :347 → LlamaForCausalLM :464. Interfaces SupportsLoRA/SupportsPP/SupportsEagle (interfaces.py:720, :808, :1684). TP layers: QKVParallelLinear (:165), MergedColumnParallelLinear (:95), RowParallelLinear (:103/:175), VocabParallelEmbedding/ParallelLMHead (:382/:503). linear.py: column all-gather optional (:623-627), row all-reduce (:1801-1804), quant_method in LinearBase.__init__ (:290-296). PP: make_layers (models/utils.py:886), PPMissingLayer (:852), IntermediateTensors (llama.py:444). Loading: DefaultModelLoader "auto" (model_loader/__init__.py:51), _get_weights_iterator (default_loader.py:265, :338) → model.load_weights (:503); AutoWeightsLoader (models/utils.py:216) + WeightsMapper(orig_to_new_stacked) (llama.py:348-357); shard id (utils.py:163); default_weight_loader (weight_utils.py:1529); ~38 files use stacked_params_mapping, 99 orig_to_new_stacked; process_weights_after_loading (model_loader/utils.py:139).

## Attention backends
backend.py (1,150): AttentionBackend :59 (get_impl_cls, get_builder_cls, supports_*, validate_configuration :279); CommonAttentionMetadata :390; AttentionMetadataBuilder.build :592/:686, build_for_cudagraph_capture :720, AttentionCGSupport :557; AttentionImpl.forward :890/:917; MLAAttentionImpl :1034. Selection: Attention.__init__ → get_attn_backend (attention.py:350; selector.py:109; overrides :186-193) → current_platform.get_attn_backend_cls (selector.py:218; cuda.py:420); _get_backend_priorities (cuda.py:84); SM90 non-MLA FLASH_ATTN > FLASHINFER > TRITON_ATTN > FLEX > TURBOQUANT (:169-180); SM100 FLASHINFER first (:160). AttentionBackendEnum (backends/registry.py:34) 49 members incl. MLA family, DSV4 sparse, ROCm AITER, CPU_ATTN, Mamba/GDN/short-conv.

## Quantization
QuantizationConfig (base_config.py:109: get_name, from_config :159, override_quantization_method :164, get_quant_method :203); QuantizeMethodBase (:23: create_weights :44, apply :53, process_weights_after_loading :80). Choice: ModelConfig._verify_quantization (config/model.py:1383-1452) → VllmConfig._get_quantization_config (config/vllm.py:898) → get_quant_config (weight_utils.py:197) → get_quantization_config (quantization/__init__.py:112, map :156) → resolve_quant_method (base_config.py:295). 30 names (__init__.py:15-49).

## torch.compile + CUDA graphs
@support_torch_compile (decorators.py:121); TorchCompileWithNoGuardsWrapper (wrapper.py); CompilationMode NONE/STOCK/DYNAMO_TRACE_ONCE/VLLM_COMPILE (config/compilation.py:37); VllmBackend.__call__ (backends.py:1048) → split_graph at splitting_ops (backends.py:576, :1212; _attention_ops compilation.py:768; set_splitting_ops_for_v1 :1162); PiecewiseCompileInterpreter (:710) → PiecewiseBackend (piecewise_backend.py:86) in CUDAGraphWrapper (cuda_graph.py:145). CUDAGraphMode (compilation.py:53-63, docs :620-640); MRV2 CudaGraphManager (gpu/cudagraph_utils.py:150; run_fullgraph :555; run_pw_graph :578). PostGradPassManager (passes/pass_manager.py:88) + PassConfig (compilation.py:114): noop elim, RMSNorm+quant, act+quant, allreduce+RMS, QK-norm+RoPE+KV, RoPE+KV, attn+quant (:154-214).

## Features
- Sampler: MRV2 gpu/sample/sampler.py (360); custom via gpu/sample/logits_processor/loader.py. V1 v1/sample/sampler.py:78; LogitsProcessor ABC v1/sample/logits_processor/interface.py:60. docs/design/logits_processors.md.
- Spec decode: config/speculative.py:74 methods (ngram, ngram_gpu, draft_model, suffix, custom_class, eagle, eagle3, extract_hidden_states, dflash, dspark, 28 MTP types :38). MRV2 init_speculator (gpu/spec_decode/__init__.py:13). Rejection: gpu/spec_decode/rejection_sampler.py:147; V1 v1/sample/rejection_sampler.py:50; variants (speculative.py:83).
- Structured output: StructuredOutputManager (v1/structured_output/__init__.py:49; grammar_init :133); backends xgrammar/guidance/outlines/lm-format-enforcer (:159-179); ABCs backend_types.py:31, :97; bitmask on worker (gpu/structured_outputs.py:70).
- LoRA: LoRAModelManager / LRUCacheLoRAModelManager (lora/model_manager.py:76, :1425); BaseLayerWithLoRA (lora/layers/base.py:17); PunicaWrapperBase (punica_wrapper/punica_base.py:120); WorkerLoRAManager (worker_manager.py:27).
- Multimodal: MULTIMODAL_REGISTRY (multimodal/__init__.py:7; registry.py:79); BaseMultiModalProcessor (processing/processor.py:1032); embed_multimodal (interfaces.py:178, :247); MRV2 gpu/mm/. docs/design/mm_processing.md.
- Pooling: as_embedding_model / as_seq_cls_model (models/adapters.py:250, :307); Pooler ABC (layers/pooler/abstract.py:16); gpu/pool/pooling_runner.py.
- MoE: FusedMoEFactory (fused_moe/layer.py:92) → MoERunner (runner/moe_runner.py:228), router (router/router_factory.py), RoutedExperts (routed_experts.py:73); EP (layer.py:272-278), ExpertMapManager (:291); modular kernel FusedMoEPrepareAndFinalize (modular_kernel.py:186) + FusedMoEExperts (:487); oracle/. docs/design/fused_moe_modular_kernel.md.

## vLLM-specific concepts
TP linear layers; stacked params + WeightsMapper; model registry; ForwardContext; static_forward_context + bind_kv_cache; backend/builder/impl triad; backend priority selection; QuantizationConfig → QuantizeMethodBase; CustomOp/PluggableLayer; vLLM IR + kernel providers; @support_torch_compile + piecewise; CUDA-graph modes + capture sizes; Inductor fusion passes; MRV2 persistent batch; modular MoE kernel; Punica LoRA; MM processor + placeholders; pooling adapters.

## Central files
gpu/model_runner.py 2,483 · gpu/attn_utils.py · models/llama.py 585 · models/registry.py 1,623 · models/utils.py 1,194 · layers/linear.py 1,819 · layers/attention/attention.py 803 · v1/attention/backend.py 1,150 · v1/attention/selector.py 250 · forward_context.py 392 · quantization/base_config.py 356 · compilation/backends.py 1,371 · config/compilation.py 1,572 · model_loader/default_loader.py 572 · gpu/sample/sampler.py 360 · layers/logits_processor.py 324.

## Docs / images / surprises
docs/design: model_runner_v2, torch_compile, cuda_graphs, attention_backends, custom_op, fusions, vllm_ir, fused_moe_modular_kernel, logits_processors, mm_processing, huggingface_integration. docs/contributing/model/{basic,registration,multimodal}.md. Images: docs/assets/design/{cuda_graphs, fused_moe_modular_kernel, model_runner_v2, debug_vllm_compile}/*, speculators-user-flow svgs.
- vllm/ir: functional IR; ops (rms_norm, fused_add_rms_norm...) declared with @register_op (ir/ops/layernorm.py:9, :43); kernel choice delayed for fusion (docs/design/vllm_ir.md:1-30).
- vllm/kernels: providers registered onto IR ops (vllm_c, aiter_ops, oink_ops, triton, Helion + tuned JSON).
- FusedMoE is a factory function; 49 attention backends, many model-specific.
