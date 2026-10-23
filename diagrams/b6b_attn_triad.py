"""B6b: the three classes of an attention backend and when each one runs.

Used by: src/concepts/attention-backends.html, src/code/v1-attention-backend.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/attention/backend.py:81-94                         get_name, get_impl_cls, get_builder_cls
  vllm/model_executor/layers/attention/attention.py:419-432  one impl per Attention layer
  vllm/v1/worker/gpu/attn_utils.py:248-265                   attention groups by (backend, spec, heads)
  vllm/v1/worker/gpu/attn_utils.py:291-300                   builders per group
  vllm/v1/worker/gpu/attn_utils.py:314-341                   min AttentionCGSupport over groups
  vllm/v1/worker/gpu/model_runner.py:754-763                 resolve_cudagraph_mode_and_sizes
  vllm/v1/worker/gpu/attn_utils.py:527-580                   CommonAttentionMetadata per group, build, dict by name
  vllm/v1/attention/backends/flash_attn.py:812-829           FlashAttentionMetadataBuilder.build
  vllm/model_executor/layers/attention/attention.py:740-746, 786-796   impl.do_kv_cache_update, impl.forward
  vllm/v1/attention/backends/flash_attn.py:1199-1200         kv_cache view [num_blocks, num_kv_heads, block_size, 2*head_size]
Example: Llama-3-8B, TP = 1, BF16, block size 16, FlashAttention (8 KV heads, head size 128).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("An attention backend: description, metadata builder, implementation",
         "Llama-3-8B, TP = 1, BF16, block size 16, FlashAttention  ·  one KV cache group, one attention group",
         params="backend class: static  ·  builder: one per attention group  ·  impl: one per layer (32)")

BW = 300
XS = [46, 400, 754]

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 244
d.panel(y0, P1_H, 1, "What is made at start-up?",
        "The layer makes its impl; the runner groups layers, makes one builder per group, and checks CUDA graph support.")
BY = y0 + 70
d.box(XS[0], BY, BW, 124, "FlashAttentionBackend", [
    "get_name() -> FLASH_ATTN",
    "get_impl_cls()",
    "get_builder_cls()",
    "supports_*, validate_*",
    "forward_includes_kv_cache_",
    "  update = False",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 124, "Attention.__init__ (x 32)", [
    "impl = impl_cls(num_heads,",
    "  head_size, scale,",
    "  num_kv_heads, ...)",
    "FlashAttentionImpl",
    "  one per layer",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 124, "init_attn_backend", [
    "group: (backend, spec, heads)",
    "-> 1 AttentionGroup",
    "builder = builder_cls(spec,",
    "  layer_names, config, dev)",
    "min CG support: ALWAYS (FA3)",
    "-> mode stays as set",
], col=BLUE, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 40, XS[1] - 2, BY + 40, VIOLET)
d.arrow_label((XS[0] + BW + XS[1]) / 2, BY + 40, "impl", VIOLET)
d.path_arrow(f"M {XS[0] + BW / 2:.1f},{BY + 124:.1f} L {XS[0] + BW / 2:.1f},{BY + 150:.1f} "
             f"L {XS[2] + BW / 2:.1f},{BY + 150:.1f} L {XS[2] + BW / 2:.1f},{BY + 126:.1f}", VIOLET)
d.text(XS[1] + BW / 2, BY + 145, "builder class", 11.5, VIOLET[0], "bold", "middle")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 220
d.panel(y1, P2_H, 2, "What runs once per step and KV cache group?",
        "build_attn_metadata makes the common description; the builder turns it into backend metadata.")
BY = y1 + 70
d.box(XS[0], BY, BW, 124, "CommonAttentionMetadata", [
    "query_start_loc [R+1]",
    "seq_lens [R]",
    "block_table_tensor [R, nb]",
    "slot_mapping [T]",
    "num_reqs, num_actual_tokens",
    "max_query_len, max_seq_len",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 124, "builder.build(0, common)", [
    "FlashAttentionMetadata",
    "  copies the common fields",
    "  + scheduler_metadata",
    "  + causal, DCP fields",
    "one object per group",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 124, "attn_metadata dict", [
    "model.layers.0.self_attn.attn -> m",
    "model.layers.1.self_attn.attn -> m",
    "...",
    "model.layers.31.self_attn.attn -> m",
    "(the same object m)",
], col=BLUE, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 59, XS[1] - 2, BY + 59, BLUE)
d.arrow(XS[1] + BW, BY + 59, XS[2] - 2, BY + 59, BLUE)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 196
d.panel(y2, P3_H, 3, "What runs once per layer, inside the forward pass?",
        "The two custom ops look up the metadata by name and call the layer's impl.")
BY = y2 + 70
d.box(XS[0], BY, BW, 100, "KV cache view (this layer)", [
    "[num_blocks, 8, 16, 256] bf16",
    "  blocks, KV heads, tokens,",
    "  key (128) + value (128)",
    "bound once as layer.kv_cache",
], col=AMBER, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 100, "impl.do_kv_cache_update", [
    "(layer, k, v, kv_cache,",
    "  slot_mapping)",
    "scatter new K/V by slot",
    "unified_kv_cache_update",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 100, "impl.forward", [
    "(layer, q, k, v, kv_cache,",
    "  m, output=out)",
    "kernel writes into out",
    "unified_attention_with_output",
], col=EMERALD, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 50, XS[1] - 2, BY + 50, AMBER)
d.arrow(XS[1] + BW, BY + 50, XS[2] - 2, BY + 50, EMERALD)

y = y2 + P3_H + 24
d.legend(30, y, [(VIOLET, "backend class (static)"), (BLUE, "builder and metadata"), (EMERALD, "impl, per layer"),
                 (AMBER, "KV cache memory")])
y += 40
y = d.footnotes(30, y, [
    "R = requests, T = tokens, nb = blocks per request. Models with several KV cache groups repeat panel 2 per group.",
    "FA3/FA4 builders report ALWAYS; FA2 reports UNIFORM_BATCH. Kernel internals are not shown.",
])
d.save(height=y + 14)
