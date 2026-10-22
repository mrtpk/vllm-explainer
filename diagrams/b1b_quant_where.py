"""Quantized checkpoints: what is quantized where (weights, activations, KV cache), and what it frees.

Used by: src/learn/quantized-checkpoints.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/layers/quantization/auto_awq.py:241-244       AWQ: w_bit/bits, q_group_size/group_size, zero_point
  vllm/model_executor/layers/quantization/auto_gptq.py:194-197      GPTQ: bits, group_size, desc_act, sym
  vllm/model_executor/layers/quantization/fp8.py:95-100             FP8: activation_scheme (dynamic), weight_block_size
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py:32   NVFP4 group 16
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_mxfp4.py:27-37 MXFP4 group 32, W4A4 or W4A16
  vllm/config/cache.py:232                                          --kv-cache-dtype default "auto"
Memory numbers (panel 2) are derived for a Llama-3.1-8B-shaped model: 6.98 B linear weights, 1.05 B embedding + LM head
weights kept in BF16, 128 KiB KV cache per token in BF16, a 24 GB GPU with gpu_memory_utilization 0.92, activations ignored.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, INK, MUTE, PINK, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Quantized checkpoints: what is small, and where",
         "linear-layer weights  ·  activations that enter the GEMM  ·  the KV cache (a separate setting)",
         params="16 bit = BF16/FP16     8 bit = FP8 E4M3 or INT8     4 bit = INT4, FP4 E2M1")

LX, LW = 46, 236
CX = (296, 548, 800)
CW = 240
RH = 54


def cell(x, y, col, l1, l2):
    d.rect(x, y, CW, RH - 8, fill=col[1], stroke=col[0], rx=6, sw=1.3)
    d.text(x + 10, y + 19, l1, 12, col[0], "bold", mono=True)
    d.text(x + 10, y + 36, l2, 11.5, INK)


# ---------------------------------------------------------------- panel 1
y0 = 116
rows = [
    ("No quantization", "BF16 checkpoint",
     (BLUE, "BF16", "plain weights"), (BLUE, "BF16", "no scales"), (BLUE, "BF16", "auto = model dtype")),
    ("W4A16 weight-only", "AWQ, GPTQ (auto_awq, auto_gptq)",
     (PINK, "INT4 + group scales", "group_size 128 is common"), (BLUE, "BF16 / FP16", "kernel expands the weights"),
     (BLUE, "BF16", "unless --kv-cache-dtype")),
    ("W8A8-FP8", "fp8, compressed-tensors, modelopt",
     (AMBER, "FP8 + scales", "per tensor, channel or block"), (AMBER, "FP8", "dynamic or static scales"),
     (BLUE, "BF16", "or FP8 if the checkpoint says")),
    ("W8A8-INT8", "compressed-tensors",
     (AMBER, "INT8 + scales", "per channel or per tensor"), (AMBER, "INT8", "dynamic per token or static"),
     (BLUE, "BF16", "unless --kv-cache-dtype")),
    ("FP4", "NVFP4 (modelopt_fp4), MXFP4",
     (PINK, "FP4 + block scales", "16 (NVFP4) or 32 (MXFP4)"), (PINK, "FP4 or BF16", "depends on GPU and kernel"),
     (BLUE, "BF16", "unless --kv-cache-dtype")),
]
P1_H = 92 + len(rows) * RH + 64
d.panel(y0, P1_H, 1, "Each scheme makes different tensors small",
        "Weight-only formats save memory. W8A8 and FP4 formats also run the matrix multiply in the small format.")
hy = y0 + 72
d.text(LX, hy + 12, "scheme / methods", 12, MUTE, "bold")
for x, h in zip(CX, ("weights", "activations into the GEMM", "KV cache (--kv-cache-dtype)")):
    d.text(x + 10, hy + 12, h, 12, MUTE, "bold")
ry = hy + 24
for name, methods, w, a, k in rows:
    d.text(LX, ry + 19, name, 13, INK, "bold")
    d.text(LX, ry + 36, methods, 11.5, MUTE)
    cell(CX[0], ry, *w)
    cell(CX[1], ry, *a)
    cell(CX[2], ry, *k)
    ry += RH
d.rect(CX[2], ry + 4, CW, 40, fill="#ffffff", stroke=AMBER[0], rx=6, sw=1.2, dash="4 3")
d.text(CX[2] + 10, ry + 20, "--kv-cache-dtype fp8", 12, AMBER[0], "bold", mono=True)
d.text(CX[2] + 10, ry + 37, "works with any row: half the KV bytes", 11.5, INK)
d.text(LX, ry + 26, "KV cache quantization is independent of the weight format.", 12, MUTE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 170
d.panel(y1, P2_H, 2, "An 8B model on a 24 GB GPU: smaller weights leave more room for the KV cache",
        "Budget 24 GB x 0.92 = 22.1 GB. KV cache 128 KiB per token in BF16 (64 KiB with FP8 KV). Activations ignored.")
by = y1 + 70
BW = 324
xs = (46, 388, 730)
d.box(xs[0], by, BW, 84, "BF16 weights: 16.1 GB", ["free for KV: 6.0 GB", "45,900 tokens  (BF16 KV)", "91,900 tokens  (FP8 KV)"],
      col=BLUE)
d.box(xs[1], by, BW, 84, "FP8 weights: 9.1 GB", ["free for KV: 13.0 GB", "99,200 tokens  (BF16 KV)", "198,300 tokens  (FP8 KV)"],
      col=AMBER)
d.box(xs[2], by, BW, 84, "INT4 weights (g128): 5.7 GB", ["free for KV: 16.4 GB", "124,800 tokens  (BF16 KV)",
                                                           "249,500 tokens  (FP8 KV)"], col=PINK)

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "16 bit"), (AMBER, "8 bit"), (PINK, "4 bit")])
y += 40
y = d.footnotes(30, y, [
    "Panel 2 assumes that the embedding and the LM head (1.05 B weights) stay in BF16, as in many quantized checkpoints.",
    "Panel 2 counts 1 GB = 10^9 bytes and a GPU of 24 x 10^9 bytes. A card sold as 24 GB usually has 24 GiB: about 1.6 GB more budget.",
    "MXFP4 runs as true W4A4 only on some GPUs; otherwise vLLM runs it weight-only (W4A16). MoE expert layers can use other schemes.",
])
d.save(height=y + 14)
