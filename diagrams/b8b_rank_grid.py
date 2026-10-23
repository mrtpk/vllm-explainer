"""Distributed groups: the rank layout of TP=2 x PP=2 x DP=2 (MoE model) on 8 GPUs.

Used by: src/concepts/distributed-groups.html, src/code/distributed-parallel-state.html
Sources (at the pinned commit in explainer.json):
  vllm/distributed/parallel_state.py:2148-2163   all_ranks = arange(world).reshape(-1, DP, PP, PCP, TP)
  vllm/distributed/parallel_state.py:2168        TP groups: view(-1, TP)
  vllm/distributed/parallel_state.py:2278-2280   PP groups: transpose(2, 4)
  vllm/distributed/parallel_state.py:2295        DP groups: transpose(1, 4)
  vllm/distributed/parallel_state.py:2312-2323   EP groups (MoE only): transpose(1, 2), size DP*PCP*TP
  vllm/distributed/parallel_state.py:1856-1872   rank = dp_rank * world_size + rank when DP > 1
  vllm/distributed/parallel_state.py:549-560     device communicator only for groups with > 1 rank
  vllm/distributed/device_communicators/cuda_communicator.py:55-89  custom AR / symm mem / FlashInfer only for "tp"
  vllm/distributed/device_communicators/cuda_communicator.py:202-209  default all2all manager (allgather_reducescatter)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, WHITE, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One rank grid, sliced into process groups",
         "TP = 2  ·  PP = 2  ·  DP = 2  ·  MoE model (DPEngineCoreProc)  ·  8 GPUs on one node",
         params="all_ranks = arange(8).reshape(1, DP=2, PP=2, PCP=1, TP=2)   →   rank = 4·dp + 2·pp + tp")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 318
d.panel(y0, P1_H, 1, "Where does each rank sit?",
        "TP is the last (fastest) dimension, so the two ranks of a TP group are neighbours.")

LY = y0 + 68
LW = 486
LH = 214
for i, lx in enumerate((46, 568)):
    d.lane(lx, LY, LW, LH, AMBER, f"DP rank {i}  (engine core {i})",
           [f"workers of EngineCore_DP{i}  ·  GPUs {4 * i}–{4 * i + 3}"], header_h=50)
    for pp in range(2):
        cy = LY + 64 + pp * 72
        d.text(lx + 16, cy + 34, f"PP stage {pp}", 12.5, VIOLET[0], "bold")
        for tp in range(2):
            r = 4 * i + 2 * pp + tp
            cx = lx + 116 + tp * 182
            d.rect(cx, cy, 166, 56, fill=BLUE[1], stroke=BLUE[0], rx=8, sw=1.5)
            d.text(cx + 83, cy + 22, f"rank {r}", 13.5, BLUE[0], "bold", "middle")
            d.text(cx + 83, cy + 41, f"GPU {r}  tp={tp} pp={pp}", 11.5, INK, anchor="middle",
                   mono=True)
d.text(46, y0 + P1_H - 12,
       "Each DP rank is one engine core with world_size = TP·PP = 4 workers; their global rank adds dp_rank·4.",
       11.5, MUTE, italic=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 196
d.panel(y1, P2_H, 2, "Which ranks form each group?",
        "Move the group's dimension to the end, reshape to rows of its size: each row is one group.")
rows = [
    (BLUE, "TP  (size 2)", "view(-1, TP)", "[0,1] [2,3] [4,5] [6,7]"),
    (VIOLET, "PP  (size 2)", "transpose(2, 4)", "[0,2] [1,3] [4,6] [5,7]"),
    (AMBER, "DP  (size 2)", "transpose(1, 4)", "[0,4] [2,6] [1,5] [3,7]"),
    (EMERALD, "EP  (size 4)", "transpose(1, 2)", "[0,1,4,5] [2,3,6,7]      (MoE models only)"),
]
for k, (col, name, op, groups) in enumerate(rows):
    ry = y1 + 82 + k * 26
    d.rect(46, ry - 14, 14, 14, fill=col[1], stroke=col[0], rx=3, sw=1.4)
    d.text(70, ry - 2, name, 13, col[0], "bold")
    d.text(200, ry - 2, op, 12, INK, mono=True)
    d.text(380, ry - 2, groups, 12.5, INK, mono=True)
d.text(46, y1 + P2_H - 12,
       "PCP, DCP: groups of 1 rank by default.  EPLB: same ranks as EP, only with --enable-eplb.  World: all 8.",
       11.5, MUTE, italic=True)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 192
d.panel(y2, P3_H, 3, "Which communicator does each group get?",
        "A GroupCoordinator gets a device communicator only when the group has more than one rank.")
BY = y2 + 68
BW = 330
d.box(46, BY, BW, 104, "TP group (\"tp\")", [
    "CudaCommunicator with pynccl",
    "+ custom all-reduce, symm mem,",
    "  FlashInfer (TP groups only)",
    "+ MessageQueue broadcaster (4 MiB × 6)",
], col=BLUE, align="start")
d.box(385, BY, BW, 104, "PP and DP groups", [
    "CudaCommunicator with pynccl only",
    "PP: send / recv of IntermediateTensors",
    "DP (MoE): gloo cpu_group all-reduce",
    "  of the per-step padding",
], col=VIOLET, align="start")
d.box(724, BY, BW, 104, "EP group (\"ep\")", [
    "CudaCommunicator with pynccl",
    "+ all2all manager (DP > 1):",
    "  default allgather_reducescatter",
    "world group: no device communicator",
], col=EMERALD, align="start")

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "tensor parallel"), (VIOLET, "pipeline parallel"), (AMBER, "data parallel"),
                 (EMERALD, "expert parallel")])
y += 36
y = d.footnotes(30, y, [
    "Dense model: each DP rank is reconfigured to DP = 1, so each engine core has its own 4-rank world and no EP group.",
    "Not shown: the gloo cpu_group that every GroupCoordinator also creates, and the ETP / engram groups.",
])
d.save(height=y + 14)
