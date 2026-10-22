# Basics already covered in deepinfer-explainer

Source: `../deepinfer-explainer/src/learn/` ("Learn the basics", 12 pages, all Beginner level).
Purpose: do not teach these again in the vLLM explainer.

**Rule (user decision 2026-10-22):** when a vLLM page needs one of these basics, give a one-line recap AND a link to the DeepInfer page. Do not re-explain it.

Backlinks: the DeepInfer site is published at https://explaindeepinfer.projects.thomaspaul.dev.
A page id maps to `<domain>/<id>.html`, for example `learn/kv-cache` → https://explaindeepinfer.projects.thomaspaul.dev/learn/kv-cache.html (checked: HTTP 200).

Each page in that site has this layout: idea, analogy, diagram, the DeepInfer code, a deep dive, what can go wrong, and check-yourself questions.
The general ML idea on each page applies to any engine. Only the "In DeepInfer" part is specific to DeepInfer.

| # | Page (id) | General ideas covered |
|---|---|---|
| 1 | What an LLM does (`learn/what-an-llm-does`) | Tokens in, a probability for every next token out. Generation as a loop (pick, append, run again). Where the knowledge lives (the weights). What an inference engine adds. |
| 2 | Tokens and tokenizers (`learn/tokens`) | Why tokens instead of letters or words. How a tokenizer finds the pieces (subword/BPE). Special tokens (BOS/EOS/PAD). Ids are only names. |
| 3 | Inside a transformer (`learn/transformer`) | Embedding, then a stack of layers, then the output head. Inside one layer: attention + FFN/MLP, residuals and norms. How the model knows token order (positions/RoPE). |
| 4 | Attention, step by step (`learn/attention`) | Query, key and value. Score, scale (1/√d), softmax, weighted sum. A worked example, multi-head attention. MHA vs GQA vs MQA vs MLA and what each does to KV size. Causal masking. Interactive attention toy. |
| 5 | Prefill and decode (`learn/prefill-decode`) | The two phases of generation. Why prefill runs in parallel and decode does not. Prefill is compute-bound and decode is memory-bound. |
| 6 | The KV cache (`learn/kv-cache`) | Why keys and values are kept. How to calculate KV memory size. Blocks and block tables (the paging idea). Preemption when blocks run out. Cache sizing from a memory profile. Interactive block allocator. |
| 7 | Sampling (`learn/sampling`) | Logits to probabilities. Temperature, top-k, top-p and the order they are applied in. Seeded random draw. |
| 8 | Batching (`learn/batching`) | Why batch (read the weights once). Static vs continuous batching. Padded vs packed (varlen) batches. Chunked prefill. Why decode is memory-bound. Mixed vs separate prefill/decode steps. |
| 9 | GPUs, kernels, and speed (`learn/gpus-kernels`) | What is inside a GPU (SMs, HBM). Kernels and launches. Memory-bound vs compute-bound (roofline idea). Streams and events. CUDA graphs and launch overhead. NPUs in brief. |
| 10 | Parallelism basics (`learn/parallelism`) | The five splits: DP, TP, PP, CP, EP. Collectives explained with small numbers (all-reduce, all-gather, all-to-all, ...). The cost of communication. |
| 11 | Mixture of Experts (`learn/moe`) | From one FFN to many experts. Router/gating with top-k. Shared experts. Why MoE is cheap per token. Load balance and capacity. Why MoE decode is memory-bound. |
| 12 | Number formats (`learn/number-formats`) | Sign, exponent and mantissa. FP32, BF16, FP16, FP8 (E4M3/E5M2), FP4/MX formats. Worked conversion of 0.3. Block scales. |

## General concepts also explained there (in the DeepInfer-specific part)

These are in `concepts/` and `tours/`. They are written around DeepInfer's code, but the general idea is already explained.
For vLLM: give a short recap and link, then focus on how vLLM does it differently.

- Engine step loop, request lifecycle, the scheduler (preemption, chunked prefill)
- Paged KV cache and block manager (DeepInfer: 128-token blocks)
- Varlen input metadata
- CUDA graphs (capture and replay)
- Async / zero-overhead scheduling (CPU and GPU overlap)
- Speculative decoding (draft, verify, accept or reject)
- Structured outputs (token masks)
- Determinism and batch invariance
- Expert parallelism (dispatch/combine), EPLB
- Weight loading from Hugging Face checkpoints, the Python ↔ C++ (pybind11) boundary

## Not covered there (fair game for the vLLM explainer)

- Tensor parallelism and pipeline parallelism in practice (DeepInfer has no TP or PP group. They are explained only as ideas.)
- How prefix caching works (hash-based block sharing). Only a definition is there, in a deep dive on `concepts/paged-kv-cache` and the glossary. DeepInfer has no prefix cache.
- Serving: OpenAI-compatible API server, AsyncLLM, multiprocess engine core
- Multi-model support, the model registry, LoRA adapters ("LoRA" appears there only as MLA's low-rank projections, not as adapters)
- Quantization methods as plugins (AWQ, GPTQ, and others)
- Attention backends as plugins (FlashAttention, FlashInfer, and others)
- Disaggregated prefill / KV transfer (mentioned there only as "not in this release")
- torch.compile integration

Note: DeepInfer is derived from vLLM, so many DeepInfer concept pages show the same design as vLLM in a simpler form.
