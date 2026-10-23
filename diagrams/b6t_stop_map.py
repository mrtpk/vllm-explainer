"""B6t: the 14 stops of one forward pass, and the layer of code where each stop runs.

Used by: src/tours/forward-pass.html
Example: meta-llama/Meta-Llama-3-8B-Instruct, BF16, TP = 1, one H100, default flags (O2,
FULL_AND_PIECEWISE, FLASH_ATTN), Model Runner V2. Batch: R1 and R2 decode 1 token each, R3 runs its
whole 298-token prompt -> 300 tokens -> PIECEWISE graph of 304 tokens.
Sources (at the pinned commit in explainer.json):
  vllm/v1/worker/gpu_worker.py:1350-1353                 Worker.execute_model -> model runner
  vllm/v1/worker/gpu/model_runner.py:1753-1760           apply the plan (finish, add, update, block tables)
  vllm/v1/worker/gpu/model_runner.py:1777-1821           gather_batch_req_state, dispatch_cg_and_sync_dp
  vllm/v1/worker/gpu/cudagraph_utils.py:111-138, 523-553 _is_compatible, dispatch
  vllm/v1/worker/gpu/model_runner.py:1375-1504           prepare_inputs
  vllm/v1/worker/gpu/model_runner.py:1577-1597           prepare_attn (block tables, slot mappings)
  vllm/v1/worker/gpu/attn_utils.py:527-580               CommonAttentionMetadata, builder.build, one object per name
  vllm/v1/worker/gpu/model_runner.py:2033-2067           set_forward_context, run_pw_graph
  vllm/model_executor/models/llama.py:313-330, 406-457   decoder layer, LlamaModel.forward
  vllm/model_executor/layers/attention/attention.py:561-581, 705-723, 786-796   ops, lookup, impl.forward
  vllm/v1/worker/gpu/model_runner.py:1637-1638, 1665, 2210-2218, 2249-2255, 2311   logits, sampler, AsyncOutput
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, NEUTRAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One forward pass: 14 stops in four layers of code",
         "Llama-3-8B  ·  BF16  ·  TP = 1  ·  one H100  ·  O2, FULL_AND_PIECEWISE  ·  FLASH_ATTN  ·  Model Runner V2",
         params="batch: R1 1 token + R2 1 token + R3 298 prompt tokens = 300 tokens -> PIECEWISE graph of 304")

BW = 300
XS = [46, 400, 754]
CH = 96
RGAP = 34


def card(x, y, num, title, col, tag, rows):
    d.step_card(x, y, BW, CH, num, title, col, tag=tag, rows=rows, key_w=40)


def row_arrows(y, col_list):
    for i in range(2):
        d.arrow(XS[i] + BW, y + CH / 2, XS[i + 1] - 2, y + CH / 2, col_list[i])


def wrap_arrow(y_top_row, col):
    """From the bottom of the right card of a row to the top of the left card of the next row."""
    x1 = XS[2] + BW / 2
    x2 = XS[0] + BW / 2
    yb = y_top_row + CH
    ym = yb + RGAP / 2
    d.path_arrow(f"M{x1:.1f},{yb:.1f} L{x1:.1f},{ym:.1f} L{x2:.1f},{ym:.1f} L{x2:.1f},{yb + RGAP - 2:.1f}", col)


# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 70 + CH + RGAP + CH + 20
d.panel(y0, P1_H, 1, "How does the plan become GPU tensors? (stops 1-6)",
        "execute_model in Model Runner V2 queues small kernels and builds the metadata. It never waits for the GPU.")
r1 = y0 + 70
card(XS[0], r1, 1, "Apply the plan", BLUE, "runner", [
    ("in", "SchedulerOutput, 3 requests"),
    ("do", "add R3, update R1 and R2"),
    ("at", "model_runner.py:1753"),
])
card(XS[1], r1, 2, "Sort, pick graph", BLUE, "runner", [
    ("order", "R1, R2, R3 (decodes first)"),
    ("mode", "PIECEWISE, 300 -> 304"),
    ("at", "model_runner.py:1806"),
])
card(XS[2], r1, 3, "prepare_inputs", BLUE, "runner", [
    ("qsl", "[0, 1, 2, 300]"),
    ("ids", "int32 [304], pos int64 [304]"),
    ("logit", "rows [0, 1, 299]"),
])
row_arrows(r1, [BLUE, BLUE])
wrap_arrow(r1, BLUE)
r2 = r1 + CH + RGAP
card(XS[0], r2, 4, "prepare_attn", BLUE, "runner", [
    ("bt", "int32 [3, max_num_blocks]"),
    ("slot", "int64 [1, 304], pad = -1"),
    ("at", "model_runner.py:1838"),
])
card(XS[1], r2, 5, "Attention metadata", VIOLET, "backend", [
    ("in", "CommonAttentionMetadata"),
    ("do", "FlashAttention builder.build"),
    ("out", "1 object, 32 layer names"),
])
card(XS[2], r2, 6, "Forward context", BLUE, "runner", [
    ("mode", "PIECEWISE, 304 tokens"),
    ("do", "with set_forward_context:"),
    ("", "  run_pw_graph -> model()"),
])
row_arrows(r2, [VIOLET, BLUE])

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 70 + CH + RGAP + CH + 20
d.panel(y1, P2_H, 2, "What runs inside the model? (stops 7-12)",
        "33 compiled pieces replay CUDA graphs of 304 tokens. The 32 attention calls run eagerly between them.")
r3 = y1 + 70
card(XS[0], r3, 7, "LlamaModel.forward", EMERALD, "model", [
    ("emb", "[304] -> [304, 4096] bf16"),
    ("loop", "32 decoder layers"),
    ("at", "llama.py:406-457"),
])
card(XS[1], r3, 8, "Add + RMSNorm", EMERALD, "model", [
    ("op", "fused_add_rms_norm (IR op)"),
    ("impl", "native, compiled by Inductor"),
    ("out", "x, residual [304, 4096]"),
])
card(XS[2], r3, 9, "QKV and RoPE", EMERALD, "model", [
    ("qkv", "[304, 4096] -> [304, 6144]"),
    ("q", "[304, 4096]  k, v [304, 1024]"),
    ("rope", "forward_native, in the piece"),
])
row_arrows(r3, [EMERALD, EMERALD])
wrap_arrow(r3, EMERALD)
r4 = r3 + CH + RGAP
card(XS[0], r4, 10, "Attention ops", AMBER, "eager", [
    ("op", "unified_kv_cache_update"),
    ("op", "unified_attention_with_output"),
    ("->", "FlashAttentionImpl (FA3)"),
])
card(XS[1], r4, 11, "o_proj and MLP", EMERALD, "model", [
    ("o", "[304, 4096], no all-reduce"),
    ("mlp", "[304, 28672] -> [304, 14336]"),
    ("down", "-> [304, 4096]"),
])
card(XS[2], r4, 12, "Final norm, keep", EMERALD, "model", [
    ("out", "hidden_states [304, 4096]"),
    ("keep", "execute_model_state"),
    ("ret", "execute_model -> None"),
])
row_arrows(r4, [EMERALD, EMERALD])
d.text(XS[0] + BW / 2, r4 + CH + 22, "stops 8-11 repeat for each of the 32 layers", 11.5, MUTE,
       anchor="middle", italic=True)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 70 + CH + 22
d.panel(y2, P3_H, 3, "How do hidden states become tokens? (stops 13-14)",
        "sample_tokens computes logits for 3 rows only, samples on the GPU, and starts a copy to the CPU.")
r5 = y2 + 70
card(XS[0], r5, 13, "compute_logits", BLUE, "runner", [
    ("rows", "hidden[[0, 1, 299]] [3, 4096]"),
    ("head", "lm_head [128256, 4096]"),
    ("out", "logits bf16 [3, 128256]"),
])
card(XS[1], r5, 14, "Sampler, AsyncOutput", BLUE, "runner", [
    ("smp", "FlashInfer top-k/top-p or"),
    ("", "  Gumbel -> ids [3, 1]"),
    ("copy", "side stream -> pinned CPU"),
])
d.box(XS[2], r5, BW, CH, "Back in the engine core", [
    "AsyncOutputFuture.result()",
    "-> get_output():",
    "copy_event.synchronize()",
    "(engine-step tour, stop 10)",
], col=NEUTRAL, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, r5 + CH / 2, XS[1] - 2, r5 + CH / 2, BLUE)
d.arrow(XS[1] + BW, r5 + CH / 2, XS[2] - 2, r5 + CH / 2, INK, dash="5 4")

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "Model Runner V2 (CPU queues GPU work)"), (VIOLET, "attention backend"),
                 (EMERALD, "model code in compiled pieces"), (AMBER, "attention layer ops, eager")])
y += 40
y = d.footnotes(30, y, [
    "Shapes are for the padded batch of 304 tokens; the 4 padding rows have slot -1. Logits rows come from logits_indices.",
    "Simplified: LoRA, speculative decoding, structured output and KV connectors are off, so their steps are not shown.",
])
d.save(height=y + 14)
