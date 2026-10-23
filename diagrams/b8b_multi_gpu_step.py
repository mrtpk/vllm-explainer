"""Tour: one step across GPUs with vllm serve -tp 2 -dp 2 (dense model), with stop numbers.

Used by: src/tours/multi-gpu-step.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/cli/serve.py:113-130, 151-152      2 API servers by default, CLI supervisor
  vllm/v1/engine/core_client.py:1599-1612, 1743-1804  DPLBAsyncMPClient picks an engine per request
  vllm/v1/engine/core.py:1534-1543                    dense: EngineCoreProc, DP reconfigured to 1
  vllm/v1/executor/multiproc_executor.py:165-172, 428 one shm MessageQueue write per call
  vllm/v1/executor/multiproc_executor.py:1041-1071    workers dequeue, only the output rank replies
  vllm/model_executor/layers/linear.py:1803-1806      row-parallel all-reduce
  vllm/model_executor/layers/vocab_parallel_embedding.py:558-563  embedding all-reduce
  vllm/model_executor/layers/logits_processor.py:122-134         logits all-gather
  vllm/v1/engine/core.py:2087-2091                    outputs to the PUSH socket of the client_index
  vllm/v1/engine/core.py:1622-1633; coordinator.py:249-276  counts to the coordinator, published to API servers
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, NEUTRAL, PINK, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One step across GPUs: the tour stops",
         "vllm serve meta-llama/Llama-3.1-8B-Instruct -tp 2 -dp 2  ·  dense model  ·  4 GPUs  ·  request A and request B",
         params="10 processes: CLI + ApiServer_0/1 + DP coordinator + EngineCore_DP0/1 + 4 workers (Worker_TP0/TP1 ×2)")


def stop(x, y, n, col=TEAL):
    d.badge(x, y, n, col, r=11, size=11)


# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 520
d.panel(y0, P1_H, 1, "Where does each stop run?",
        "Badges are tour stops. Each engine core and its two workers form a separate world of TP = 2.")
RY = y0 + 70
d.box(46, RY, 300, 84, "ApiServer_0", ["request A arrives", "DPLB score → engine 0", "receives A's tokens"],
      col=BLUE, align="start")
d.box(392, RY, 300, 84, "ApiServer_1", ["request B arrives", "DPLB score → engine 1", "receives B's tokens"],
      col=BLUE, align="start")
d.box(738, RY, 316, 84, "DP coordinator", ["counts in from engines", "publishes to API servers",
                                           "no waves (dense model)"], col=PINK, align="start")
stop(330, RY + 14, 2)
stop(676, RY + 14, 2)
stop(330, RY + 70, 11)
stop(676, RY + 70, 11)
stop(1038, RY + 14, 13)

LY = RY + 84 + 52
LH = 300
for i, lx in enumerate((46, 568)):
    d.lane(lx, LY, 486, LH, VIOLET, f"DP rank {i}", [f"GPUs {2 * i}, {2 * i + 1}  ·  its own world of 2 ranks"],
           header_h=50)
    ex = lx + 16
    d.box(ex, LY + 62, 454, 62, f"EngineCore_DP{i}  (EngineCoreProc)", [
        "schedule → execute_model / sample_tokens",
    ], col=VIOLET, align="start", tsize=13)
    stop(ex + 438, LY + 76, 4)
    stop(ex + 412, LY + 76, 3)
    # shm arrow
    d.arrow(ex + 120, LY + 124, ex + 120, LY + 158, EMERALD)
    d.text(ex + 132, LY + 146, "shm MessageQueue: 1 write, 2 readers", 11.5, EMERALD[0], "bold")
    stop(ex + 438, LY + 142, 5)
    # workers
    for t in range(2):
        wx = ex + t * 232
        d.rect(wx, LY + 162, 222, 84, fill=EMERALD[1], stroke=EMERALD[0], rx=8, sw=1.5)
        d.text(wx + 12, LY + 182, f"Worker_TP{t}  (GPU {2 * i + t})", 13, EMERALD[0], "bold")
        d.text(wx + 12, LY + 202, "execute_model: forward", 11.5, INK, mono=True)
        d.text(wx + 12, LY + 218, "sample_tokens: sample", 11.5, INK, mono=True)
        d.text(wx + 12, LY + 234, "reply" if t == 0 else "no reply", 11.5, INK, mono=True)
        stop(wx + 206, LY + 178, 6)
    stop(ex + 206, LY + 230, 10)
    # NCCL bar
    d.rect(ex, LY + 258, 454, 28, fill=AMBER[1], stroke=AMBER[0], rx=6, sw=1.4)
    d.text(ex + 227, LY + 277, "TP group [0, 1]: 65 all-reduces + 1 all-gather per step", 11.5, AMBER[0], "bold",
           "middle")
    stop(ex + 16, LY + 272, 7, AMBER)
    stop(ex + 40, LY + 272, 8, AMBER)
    stop(ex + 438, LY + 272, 9, AMBER)

# API server -> engines (requests) and engines -> coordinator
d.arrow(196, RY + 84, 196, LY - 2, BLUE, sw=2)
d.text(206, RY + 84 + 30, "A over ZMQ", 11.5, BLUE[0], "bold")
d.path_arrow(f"M 542,{RY + 84} L 542,{RY + 84 + 22} L 700,{RY + 84 + 22} L 700,{LY - 2}", BLUE, sw=2)
d.text(552, RY + 84 + 16, "B over ZMQ", 11.5, BLUE[0], "bold")
d.path_arrow(f"M 1000,{LY} L 1000,{RY + 86}", PINK, dash="5 4")
d.text(870, RY + 84 + 30, "counts (stop 12)", 11.5, PINK[0], "bold")
stop(1020, LY - 18, 12, PINK)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 196
d.panel(y1, P2_H, 2, "Which collectives run in one step, and on which group?",
        "Only the TP group has more than one rank, so it carries every collective of the step (Llama-3.1-8B, 32 layers).")
rows = [
    ("embedding", "VocabParallelEmbedding", "all-reduce", "× 1"),
    ("attention", "o_proj (RowParallelLinear)", "all-reduce", "× 32"),
    ("MLP", "down_proj (RowParallelLinear)", "all-reduce", "× 32"),
    ("logits", "LogitsProcessor._gather_logits", "all-gather", "× 1"),
]
for k, (part, layer, op, n) in enumerate(rows):
    ry = y1 + 86 + k * 24
    d.text(46, ry, part, 12.5, INK, "bold")
    d.text(160, ry, layer, 12, INK, mono=True)
    d.text(520, ry, op, 12, AMBER[0], "bold", mono=True)
    d.text(660, ry, n, 12, INK, mono=True)
d.text(46, y1 + P2_H - 12, "DP group, PP group: 1 rank each (dense DP is reconfigured to DP = 1) → no collectives.",
       11.5, MUTE, italic=True)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "API server, requests"), (VIOLET, "engine core"), (EMERALD, "workers, shm"),
                 (AMBER, "NCCL collectives"), (PINK, "load counts")])
y += 36
y = d.footnotes(30, y, [
    "Stop 1 (start-up) and the CLI supervisor are not drawn. Each request may land on either API server; the kernel",
    "  spreads connections over the shared listen socket. Engine choice is the DPLB score (stops 2-3).",
])
d.save(height=y + 14)
