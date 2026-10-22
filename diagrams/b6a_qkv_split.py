"""QKVParallelLinear: how 32 query heads and 8 KV heads go to the TP ranks, with TP = 2 and TP = 16.

Used by: src/concepts/tp-layers.html, src/code/model-executor-layers-linear.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/layers/linear.py:1064-1077   num_heads = total / tp; KV: tp >= total_kv -> 1 head, replicas = tp / total_kv
  vllm/model_executor/layers/linear.py:1101-1117   offsets and sizes of q, k, v inside the stacked parameter
  vllm/model_executor/parameter.py:175-198         load_qkv_weight: q slice = tp_rank, k/v slice = tp_rank // replicas
  vllm/model_executor/models/llama.py:143-161      LlamaAttention: the same head counts per rank, q_size and kv_size
Shapes: Llama-3.1-8B / Llama-3-8B (Hugging Face config.json): 32 query heads, 8 KV heads, head size 128, hidden 4096.
GQA: query head h reads KV head h // 4 (32 / 8 = 4 query heads per KV head).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import INK, MUTE, SERIES, WHITE, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("QKVParallelLinear: which heads each TP rank owns",
         "32 query heads, 8 KV heads, head size 128  ·  colour = KV head (GQA group of 4 query heads)",
         params="per rank: q_size = num_heads x 128, kv_size = num_kv_heads x 128, output = q_size + 2 x kv_size")

TINT = ["#eff6ff", "#fdf2f8", "#f0fdf4", "#fff7ed", "#f5f3ff", "#ecfeff", "#fefce8", "#fef2f2"]


def head_cell(x, y, w, h, label, kv, bold=False):
    d.rect(x, y, w, h, fill=TINT[kv], stroke=SERIES[kv], rx=3, sw=1.4 if bold else 1)
    d.text(x + w / 2, y + h / 2 + 4, label, 11, SERIES[kv], "bold", "middle", mono=True)


# ---------------------------------------------------------------- panel 1: TP = 2
y0 = 116
P1_H = 300
d.panel(y0, P1_H, 1, "TP = 2: every rank gets 16 query heads and 4 KV heads; nothing is copied",
        "num_heads = 32 / 2 = 16, num_kv_heads = 8 / 2 = 4, num_kv_head_replicas = 1. qkv_proj.weight per rank: [3072, 4096].")
CW = 28
for r in range(2):
    ry = y0 + 74 + r * 106
    d.text(46, ry + 18, f"rank {r}", 13, INK, "bold")
    d.text(46, ry + 36, "q: 16 heads", 11.5, MUTE)
    d.text(46, ry + 70, "k, v: 4 heads", 11.5, MUTE)
    for i in range(16):
        h = 16 * r + i
        head_cell(150 + i * (CW + 4), ry + 4, CW, 26, str(h), h // 4)
    for j in range(4):
        kv = 4 * r + j
        head_cell(150 + j * 2 * (CW + 4), ry + 56, 2 * CW + 4, 26, f"k{kv}", kv)
        head_cell(150 + (4 + j) * 2 * (CW + 4), ry + 56, 2 * CW + 4, 26, f"v{kv}", kv)
    d.text(680, ry + 22, f"q rows {2048 * r}-{2048 * r + 2047} of q_proj", 12, INK, mono=True)
    d.text(680, ry + 74, f"k, v rows {512 * r}-{512 * r + 511} of k_proj, v_proj", 12, INK, mono=True)

# ---------------------------------------------------------------- panel 2: TP = 16
y1 = y0 + P1_H + 20
P2_H = 330
d.panel(y1, P2_H, 2, "TP = 16: each rank gets 2 query heads; each KV head is copied to 2 ranks",
        "tp_size >= 8 KV heads: num_kv_heads = 1, num_kv_head_replicas = 16 / 8 = 2. qkv_proj.weight per rank: [512, 4096].")
COLW = 60
X0 = 46
for r in range(16):
    x = X0 + r * (COLW + 3)
    cy = y1 + 72
    d.text(x + COLW / 2, cy + 12, f"r{r}", 12, INK, "bold", "middle", mono=True)
    head_cell(x, cy + 22, COLW, 24, f"{2 * r}", (2 * r) // 4)
    head_cell(x, cy + 50, COLW, 24, f"{2 * r + 1}", (2 * r + 1) // 4)
    kv = r // 2
    head_cell(x, cy + 86, COLW, 24, f"k{kv}", kv, bold=True)
    head_cell(x, cy + 114, COLW, 24, f"v{kv}", kv, bold=True)
d.text(X0, y1 + 236, "q heads: rank r takes heads 2r and 2r+1 (slice index = tp_rank).", 12.5, INK)
d.text(X0, y1 + 256, "k, v: rank r takes KV head r // 2 (slice index = tp_rank // num_kv_head_replicas).", 12.5, INK)
d.text(X0, y1 + 276, "The query heads of a rank always use the KV head that the same rank holds: heads 2r, 2r+1 -> KV head r // 2.", 12.5, INK)
d.text(X0, y1 + 300, "KV cache: each rank stores 1 KV head per layer, so the 16 ranks store 16 copies of the 8 real KV heads (2 of each).", 12.5, MUTE)

y = y1 + P2_H + 24
d.text(30, y + 11, "Legend", 13, INK, "bold")
cx = 96
for kv in range(8):
    d.rect(cx, y, 14, 14, fill=TINT[kv], stroke=SERIES[kv], rx=3, sw=1.4)
    d.text(cx + 20, y + 11.5, f"KV head {kv}", 12, INK)
    cx += 106
y += 40
y = d.footnotes(30, y, [
    "Numbers in cells are query head ids; k3 and v3 are the key and value of KV head 3. Query head h uses KV head h // 4 (grouped-query attention).",
    "TP = 8 (not shown): 4 query heads and exactly 1 KV head per rank, replicas = 1. TP = 3 fails: 32 heads do not divide by 3.",
])
d.save(height=y + 14)
