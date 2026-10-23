"""B6t: one Llama decoder layer during the example step, with the tensor shapes.

Used by: src/tours/forward-pass.html
Example: Llama-3-8B (HF config: hidden 4096, intermediate 14336, 32 query heads, 8 KV heads,
head size 128), BF16, TP = 1, one H100, PIECEWISE step padded to 304 tokens.
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/models/llama.py:313-330      LlamaDecoderLayer.forward (norm, attention, norm, MLP)
  vllm/model_executor/models/llama.py:224-234      LlamaAttention.forward (qkv, split, rope, attn, o_proj)
  vllm/model_executor/models/llama.py:118-122      LlamaMLP.forward (gate_up, act, down)
  vllm/model_executor/layers/layernorm.py:79-99    RMSNorm.forward_native -> IR ops rms_norm / fused_add_rms_norm
  vllm/platforms/cuda.py:716-720                   IR priority ["native"] with Inductor, else ["vllm_c", "native"]
  vllm/model_executor/layers/attention/attention.py:529-581   output buffer, views, the two custom ops
  vllm/v1/attention/backends/flash_attn.py:1199-1200          KV cache view per layer
  vllm/model_executor/layers/linear.py:1803-1806   all-reduce only when TP > 1
  vllm/model_executor/layers/utils.py:83-89, 689-691   unquantized GEMM = F.linear by default on CUDA
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, EMERALD, PINK, Diagram, text_width  # noqa: E402

d = Diagram(width=1100)
d.header("One decoder layer of the step, with shapes",
         "Llama-3-8B layer i (1 to 31)  ·  BF16  ·  TP = 1  ·  T = 304 rows (300 tokens + 4 padding rows)",
         params="hidden 4096  ·  32 query heads, 8 KV heads, head size 128  ·  intermediate 14336  (HF config)")

BW = 300
XS = [46, 400, 754]
BH = 104
RGAP = 34


def box(x, y, title, lines, col, tag=None):
    d.box(x, y, BW, BH, title, lines, col=col, align="start", mono_size=11.5)
    if tag:
        d.pill_auto(x + BW - 8 - (text_width(tag, 11, mono=True) + 20), y + 8, tag, col)


def row_arrows(y, cols):
    for i in range(2):
        d.arrow(XS[i] + BW, y + BH / 2, XS[i + 1] - 2, y + BH / 2, cols[i])


def wrap_arrow(y_row, col):
    x1 = XS[2] + BW / 2
    x2 = XS[0] + BW / 2
    yb = y_row + BH
    ym = yb + RGAP / 2
    d.path_arrow(f"M{x1:.1f},{yb:.1f} L{x1:.1f},{ym:.1f} L{x2:.1f},{ym:.1f} L{x2:.1f},{yb + RGAP - 2:.1f}", col)


# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 70 + BH + RGAP + BH + 20
d.panel(y0, P1_H, 1, "What happens in the attention half?",
        "The norm, QKV and RoPE run in compiled piece i; the two attention ops run eagerly; o_proj starts piece i + 1.")
r1 = y0 + 70
box(XS[0], r1, "input_layernorm", [
    "in   hidden, residual [304, 4096]",
    "op   fused add + RMSNorm",
    "out  x [304, 4096]",
    "     residual = hidden + residual",
], EMERALD, "piece i")
box(XS[1], r1, "qkv_proj", [
    "QKVParallelLinear",
    "W    [6144, 4096] bf16",
    "out  [304, 6144]",
    "     = 4096 q + 1024 k + 1024 v",
], EMERALD, "piece i")
box(XS[2], r1, "split + rotary_emb", [
    "q    [304, 4096]",
    "k    [304, 1024]   v [304, 1024]",
    "RoPE on q and k, by positions",
    "     positions int64 [304]",
], EMERALD, "piece i")
row_arrows(r1, [EMERALD, EMERALD])
wrap_arrow(r1, EMERALD)
r2 = r1 + BH + RGAP
box(XS[0], r2, "Attention.forward", [
    "output = empty [304, 4096]",
    "q view    [304, 32, 128]",
    "k, v view [304, 8, 128]",
    "-> two torch.ops.vllm ops",
], EMERALD, "piece i")
box(XS[1], r2, "Two attention ops", [
    "update: k, v -> KV cache slots",
    "  (padding rows: slot -1)",
    "attention: 300 query rows",
    "kv view [num_blocks, 8, 16, 256]",
], AMBER, "eager")
box(XS[2], r2, "o_proj", [
    "RowParallelLinear",
    "W    [4096, 4096] bf16",
    "out  [304, 4096]",
    "TP = 1: no all-reduce",
], EMERALD, "piece i+1")
row_arrows(r2, [AMBER, AMBER])

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 70 + BH + 20
d.panel(y1, P2_H, 2, "What happens in the MLP half?",
        "All of it runs in compiled piece i + 1, together with the norm, QKV and RoPE of layer i + 1.")
r3 = y1 + 70
box(XS[0], r3, "post_attention_layernorm", [
    "in   attn out, residual",
    "op   fused add + RMSNorm",
    "out  x [304, 4096]",
    "     residual [304, 4096]",
], EMERALD, None)
box(XS[1], r3, "gate_up_proj + act_fn", [
    "MergedColumnParallelLinear",
    "W    [28672, 4096] -> [304, 28672]",
    "SiluAndMul: silu(gate) * up",
    "out  [304, 14336]",
], EMERALD, None)
box(XS[2], r3, "down_proj", [
    "RowParallelLinear",
    "W    [4096, 14336] -> [304, 4096]",
    "TP = 1: no all-reduce",
    "-> layer i + 1 (hidden, residual)",
], EMERALD, None)
row_arrows(r3, [EMERALD, EMERALD])

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 70 + 120 + 20
d.panel(y2, P3_H, 3, "Which code runs for each op?",
        "By default the small ops are plain PyTorch that Inductor compiles; --enforce-eager uses the CUDA kernels.")
r4 = y2 + 70
d.box(XS[0], r4, BW, 120, "Default (O2, compiled piece)", [
    "RMSNorm: CustomOp off ->",
    "  forward_native -> IR op,",
    "  lowered to its native body",
    "RoPE, SiluAndMul: forward_native",
    "linear: F.linear (bf16 GEMM)",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[1], r4, BW, 120, "Default (eager attention ops)", [
    "update: impl.do_kv_cache_update",
    "  -> reshape_and_cache_flash",
    "attention: impl.forward",
    "  -> flash_attn_varlen_func",
    "  fa_version 3 on H100",
], col=AMBER, align="start", mono_size=11.5)
d.box(XS[2], r4, BW, 120, "With --enforce-eager", [
    "CustomOps on (base mode all)",
    "RMSNorm -> IR op, vllm_c first",
    "SiluAndMul -> _C.silu_and_mul",
    "RoPE -> ops.rotary_embedding",
    "no compile, no CUDA graphs",
], col=PINK, align="start", mono_size=11.5)

y = y2 + P3_H + 24
d.legend(30, y, [(EMERALD, "inside a compiled piece (graph replay)"), (AMBER, "attention ops, eager"),
                 (PINK, "--enforce-eager alternative")])
y += 40
y = d.footnotes(30, y, [
    "Layer 0 has no residual yet: its input_layernorm is a plain RMSNorm, and piece 0 also holds the embedding.",
    "Weights are [out, in]. Padding rows are computed too, but their KV slot is -1 and no logits row reads them.",
])
d.save(height=y + 14)
