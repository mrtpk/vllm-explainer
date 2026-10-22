"""B3b concept page "Hybrid KV cache": how a sliding-window group frees blocks over time.

Used by: src/concepts/hybrid-kv-cache.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/core/kv_cache_manager.py:546-559               remove_skipped_blocks runs first in allocate_slots,
                                                         on processed tokens (total computed - in-flight)
  vllm/v1/core/single_type_kv_cache_manager.py:681-719   skipped blocks = skipped tokens // block_size
  vllm/v1/core/single_type_kv_cache_manager.py:1144-1179 skipped tokens = computed - sliding_window + 1 (+ extra)
  vllm/v1/core/single_type_kv_cache_manager.py:654-679   free backwards, replace with the null block
  vllm/v1/core/single_type_kv_cache_manager.py:214-236, 400-410   new blocks = cdiv(tokens, block_size) - held
  vllm/v1/core/block_pool.py:776-807                     unhashed freed blocks go to the front of the free queue
  vllm/v1/kv_cache_interface.py:903-929                  admission cap per request for a sliding-window group
  vllm/config/vllm.py:600-608                            max_in_flight_tokens = max_concurrent_batches x budget
Illustrative numbers: sliding_window 32, block_size 16, a 96-token prompt in chunks of 32, synchronous steps.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, INK, LINE, MUTE, PINK, TEAL, WHITE, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("A sliding-window group frees blocks as the request moves on",
         "Illustrative: sliding_window 32  ·  block_size 16  ·  96-token prompt in chunks of 32  ·  synchronous steps",
         params="skipped tokens = max(0, computed - 32 + 1)   ·   skipped blocks = skipped tokens // 16   ·   null = block 0")

# ---------------------------------------------------------------- panel 1
y0 = 118
rows = [
    ("step 1", "computed 0, new 32", "skip 0 -> free none", ["1", "2"], 2, []),
    ("step 2", "computed 32, new 32", "skip 1 -> 0 blocks", ["1", "2", "3", "4"], 4, []),
    ("step 3", "computed 64, new 32", "skip 33 -> 2 blocks", ["N", "N", "3", "4", "5", "6"], 6, ["2", "1"]),
    ("step 4", "computed 96, new 1", "skip 65 -> 4 blocks", ["N", "N", "N", "N", "5", "6", "7"], 7, ["4", "3"]),
    ("step 20", "computed 112, new 1", "skip 81 -> 5 blocks", ["N", "N", "N", "N", "N", "6", "7", "8"], 8, ["5"]),
]
RH = 40
P1_H = 88 + len(rows) * RH + 30
d.panel(y0, P1_H, 1, "What does the sliding-window block table hold after each allocate_slots()?",
        "First free the blocks behind the window, then add blocks for the new tokens. Full attention would hold every block.")
ty = y0 + 84
CX = 452
CW, CG = 46, 5
d.text(CX, ty - 6, "sliding-window table (block ids)", 11.5, TEAL[0], "bold")
d.text(912, ty - 6, "freed", 11.5, PINK[0], "bold")
d.text(982, ty - 6, "full", 11.5, BLUE[0], "bold")
for r, (step, comp, skip, table, full_n, freed) in enumerate(rows):
    yy = ty + r * RH
    d.text(46, yy + 20, step, 12, INK, "bold", mono=True)
    d.text(118, yy + 20, comp, 11.5, INK, mono=True)
    d.text(296, yy + 20, skip, 11.5, MUTE, mono=True)
    for i, b in enumerate(table):
        x = CX + i * (CW + CG)
        nul = b == "N"
        d.rect(x, yy + 4, CW, 26, fill=WHITE if nul else TEAL[1], stroke=LINE if nul else TEAL[0], rx=3, sw=1.2,
               dash="3 3" if nul else None)
        d.text(x + CW / 2, yy + 22, "null" if nul else b, 11, MUTE if nul else TEAL[0], "bold", "middle", mono=True)
    d.text(912, yy + 22, ", ".join(freed) if freed else "-", 11.5, PINK[0], "bold", mono=True)
    d.text(982, yy + 22, f"{full_n} held", 11.5, BLUE[0], mono=True)
d.text(46, ty + len(rows) * RH + 14,
       "Real blocks in the sliding table: 2, 4, 4, 3, 3. The table keeps its length: freed places hold the null block.",
       12, MUTE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 216
d.panel(y1, P2_H, 2, "Where do the freed blocks go, and what limits the peak?",
        "Freed sliding-window blocks usually have no hash, so the pool reuses them first, for any group.")
XS = [46, 400, 754]
BW = 300
by = y1 + 68
BH = 120
d.box(XS[0], by, BW, BH, "Freed blocks", [
    "_remove_blocks_in_range():",
    "  walk backwards, put null",
    "free_blocks([2, 1]): ref_cnt 0",
    "no hash -> FRONT of the queue",
    "(a hash only at kept boundaries)",
], col=PINK, align="start")
d.box(XS[1], by, BW, BH, "Admission cap", [
    "max_admission_blocks_per_request",
    "  = cdiv(min(window - 1 +",
    "    in-flight, max_model_len),",
    "    block_size) + 1",
    "used when a new request must fit",
], col=AMBER, align="start")
d.box(XS[2], by, BW, BH, "Async scheduling", [
    "frees on PROCESSED tokens:",
    "  total computed - in-flight",
    "the step still in flight may",
    "  read the old blocks, so the",
    "  free comes one step later",
], col=BLUE, align="start")

y = y1 + P2_H + 22
d.legend(30, y, [(TEAL, "sliding-window block"), (PINK, "freed"), (BLUE, "full-attention group"),
                 (AMBER, "admission")])
y += 36
y = d.footnotes(30, y, [
    "Illustrative numbers. With retention interval 0 (the default), a sliding-window block gets a hash only at a kept boundary",
    "(the end of the prompt, a shared-prefix junction); the other freed blocks have none.",
])
d.save(height=y + 12)
