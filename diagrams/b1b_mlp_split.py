"""Tensor parallelism in one MLP block: column-parallel gate_up_proj, then row-parallel down_proj.

Used by: src/learn/tp-pp-in-practice.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/models/llama.py:95-122            LlamaMLP: MergedColumnParallelLinear gate_up_proj, SiluAndMul, RowParallelLinear down_proj
  vllm/model_executor/layers/linear.py:496-497           column-parallel: output_size_per_partition = output_size / tp_size
  vllm/model_executor/layers/linear.py:1694-1695         row-parallel: input_size_per_partition = input_size / tp_size
  vllm/model_executor/layers/linear.py:1803-1804         row-parallel forward: all-reduce when reduce_results and tp_size > 1
Sizes: Llama-3.1-8B-style MLP (hidden 4096, intermediate 14336), TP = 2, BF16 (2 bytes per value).
Weights are stored [out, in] like torch.nn.Linear.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Tensor parallelism in one MLP block: split the columns, then the rows",
         "hidden 4096  ·  intermediate 14336  ·  TP = 2  ·  T = tokens in the step  ·  BF16",
         params="Y = down_proj( SiLU(gate(X)) * up(X) )     gate, up: [14336, 4096]     down: [4096, 14336]")

LW = 500
LX = (46, 554)
CW = 460

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 556
d.panel(y0, P1_H, 1, "Each GPU computes half of the MLP, then one all-reduce adds the halves",
        "No communication between the two linear layers: the column split of layer 1 is the row split of layer 2.")
ly = y0 + 66
LH = 402
for g, x in enumerate(LX):
    d.lane(x, ly, LW, LH, BLUE if g == 0 else VIOLET, f"GPU {g}  (TP rank {g})",
           [f"holds rows {7168 * g}-{7168 * g + 7167} of gate and up, columns {7168 * g}-{7168 * g + 7167} of down"],
           header_h=50)
    cx = x + 20
    cy = ly + 66
    col = BLUE if g == 0 else VIOLET
    d.step_card(cx, cy, CW, 58, 1, "input X (same copy on both GPUs)", INK, tag="replicated",
                rows=[("X", "[T, 4096]")])
    d.arrow(cx + CW / 2, cy + 58, cx + CW / 2, cy + 74, INK)
    cy += 76
    d.step_card(cx, cy, CW, 76, 2, "gate_up_proj: column-parallel", col, tag="MergedColumnParallelLinear",
                rows=[("W", "[2 x 7168, 4096]   (full: [2 x 14336, 4096])"), ("out", "[T, 2 x 7168]")])
    d.arrow(cx + CW / 2, cy + 76, cx + CW / 2, cy + 92, col)
    cy += 94
    d.step_card(cx, cy, CW, 58, 3, "SiluAndMul: SiLU(gate) * up", col, tag="local",
                rows=[("out", "[T, 7168]   (this GPU's slice of the 14336)")])
    d.arrow(cx + CW / 2, cy + 58, cx + CW / 2, cy + 74, col)
    cy += 76
    d.step_card(cx, cy, CW, 76, 4, "down_proj: row-parallel", col, tag="RowParallelLinear",
                rows=[("W", "[4096, 7168]   (full: [4096, 14336])"), ("out", f"Y{g} = partial sum  [T, 4096]")])
# all-reduce bar
ay = ly + LH + 22
d.rect(46, ay, 1008, 50, fill=AMBER[1], stroke=AMBER[0], rx=8, sw=1.5)
d.text(550, ay + 21, "all-reduce (sum) over the TP group", 13.5, AMBER[0], "bold", "middle")
d.text(550, ay + 39, "Y = Y0 + Y1   [T, 4096]   -> the same full result on both GPUs", 12, INK, anchor="middle", mono=True)
for x in LX:
    d.arrow(x + LW / 2, ly + LH, x + LW / 2, ay - 2, AMBER, sw=2)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 150
d.panel(y1, P2_H, 2, "What TP = 2 saves and what it costs, per MLP block",
        "Half the weights and half the FLOPs on each GPU. In exchange: one all-reduce of the full output.")
by = y1 + 66
BW3 = 324
xs = (46, 388, 730)
d.box(xs[0], by, BW3, 70, "Weights per GPU", ["88.1 M params = 176 MB (BF16)", "full MLP: 176.2 M = 352 MB"], col=EMERALD)
d.box(xs[1], by, BW3, 70, "Data per all-reduce", ["T x 4096 x 2 B", "T = 2048 -> 16 MiB"], col=AMBER)
d.box(xs[2], by, BW3, 70, "All-reduces per decoder layer", ["2: after o_proj, after down_proj", "32 layers -> 64 per step"], col=AMBER)

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "GPU 0 slice"), (VIOLET, "GPU 1 slice"), (AMBER, "communication"),
                 (EMERALD, "memory")])
y += 40
y = d.footnotes(30, y, [
    "gate_up_proj is one merged weight: each GPU holds its 7168 rows of gate and its 7168 rows of up, so SiluAndMul needs no data from the other GPU.",
    "Not shown: the attention block (also column-parallel qkv_proj, then row-parallel o_proj), custom all-reduce kernels, quantized weights.",
])
d.save(height=y + 14)
