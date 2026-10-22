"""B3b concept page "Hybrid KV cache": groups and block tables of a 2-group model (sliding window + full attention).

Used by: src/concepts/hybrid-kv-cache.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/models/gpt_oss.py:174-175          sliding window only on even layers
  vllm/model_executor/layers/attention/attention.py:621-679   SlidingWindowSpec or FullAttentionSpec per layer
  vllm/v1/core/kv_cache_planning.py:416-556              _get_kv_cache_groups_uniform_page_size (group size 12)
  vllm/v1/core/kv_cache_coordinator.py:119-125, 157-172  one BlockPool, one manager per group
  vllm/v1/core/single_type_kv_cache_manager.py:1144-1179 skipped tokens = computed - window + 1
  vllm/v1/core/single_type_kv_cache_manager.py:654-719   skipped blocks replaced by the null block
  vllm/v1/kv_cache_interface.py:1455-1467                KVCacheTensor: groups alias the same bytes
  vllm/v1/core/kv_cache_planning.py:574-593, 676-733     bytes per block = the largest group; groups alias from byte 0
  vllm/v1/kv_cache_interface.py:570-586                  page = heads x states x (head_size + head_size_v) x dtype
Model numbers (24 layers, sliding_window 128, 8 KV heads, head size 64, BF16) are from the Hugging Face config of
gpt-oss-20b, not from the vLLM repo. Block ids are illustrative.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, INK, LINE, MUTE, TEAL, WHITE, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Two KV cache groups, two block tables, one block pool",
         "A sliding-window + full-attention model like gpt-oss-20b  ·  block_size 16  ·  sliding_window 128",
         params="page per layer = 8 KV heads x 16 tokens x (64 + 64) x 2 bytes = 32 KiB   ·   12 layers per group")

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 214
d.panel(y0, P1_H, 1, "Which layers share a block table?",
        "Layers with the same KV cache spec form a group. Each group gets one single-type manager and one block table.")
cy = y0 + 66
CW, CG = 38, 4
for i in range(24):
    col = TEAL if i % 2 == 0 else BLUE
    x = 46 + i * (CW + CG)
    d.rect(x, cy, CW, 28, fill=col[1], stroke=col[0], rx=3, sw=1.2)
    d.text(x + CW / 2, cy + 18, f"L{i}", 11, col[0], "bold", "middle", mono=True)
by = cy + 48
BW = 318
d.box(46, by, BW, 84, "sliding-window group", [
    "L0, L2, ..., L22 (12 layers)",
    "SlidingWindowSpec(window 128)",
    "-> SlidingWindowManager",
], col=TEAL)
d.box(392, by, BW, 84, "full-attention group", [
    "L1, L3, ..., L23 (12 layers)",
    "FullAttentionSpec",
    "-> FullAttentionManager",
], col=BLUE)
d.box(738, by, BW, 84, "one BlockPool", [
    "both managers take block ids",
    "from the same free queue",
    "-> HybridKVCacheCoordinator",
], col=AMBER)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 222
d.panel(y1, P2_H, 2, "One request with 200 computed tokens: two block tables",
        "Full attention keeps every block. The sliding window frees blocks it no longer reads: 200 - 128 + 1 = 73 tokens -> 4 blocks.")
ty = y1 + 74
TW, TG = 58, 5
full_ids = ["7", "3", "12", "40", "41", "9", "15", "22", "23", "30", "31", "33", "34"]
sw_ids = ["null", "null", "null", "null", "50", "51", "52", "53", "54", "55", "56", "57", "58"]
TX = 190
d.text(46, ty + 20, "full group", 12, BLUE[0], "bold")
d.text(46, ty + 62, "sliding group", 12, TEAL[0], "bold")
for i in range(13):
    x = TX + i * (TW + TG)
    d.rect(x, ty, TW, 30, fill=BLUE[1], stroke=BLUE[0], rx=4, sw=1.2)
    d.text(x + TW / 2, ty + 20, full_ids[i], 11.5, BLUE[0], "bold", "middle", mono=True)
    nul = sw_ids[i] == "null"
    d.rect(x, ty + 42, TW, 30, fill=WHITE if nul else TEAL[1], stroke=LINE if nul else TEAL[0], rx=4, sw=1.2,
           dash="3 3" if nul else None)
    d.text(x + TW / 2, ty + 62, "null" if nul else sw_ids[i], 11.5, MUTE if nul else TEAL[0], "bold", "middle",
           mono=True)
d.text(TX, ty + 96, "block index 0 (tokens 0-15) ... block index 12 (tokens 192-199)", 11.5, MUTE, mono=True)
d.text(46, ty + 120, "Held: 13 + 9 = 22 blocks x 384 KiB = 8.25 MiB. With --disable-hybrid-kv-cache-manager: one group of 24 layers,",
       12, MUTE)
d.text(46, ty + 138, "13 blocks x 768 KiB = 9.75 MiB, and the sliding layers never free a block.", 12, MUTE)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 20
P3_H = 226
d.panel(y2, P3_H, 3, "Why must all groups have the same page size?",
        "Both groups use the same bytes. A block id means the same byte range in every layer region of the allocation.")
ry = y2 + 70
RW = 300
for k in range(3):
    x = 46 + k * (RW + 46)
    d.rect(x, ry, RW, 96, fill=WHITE, stroke=INK, rx=6, sw=1.2)
    d.text(x + 10, ry + 18, f"region {k}: L{2 * k + 1} (full) / L{2 * k} (sliding)", 11.5, INK, "bold", mono=True)
    for j, (lab, col) in enumerate([("blk 40", BLUE), ("blk 41", BLUE), ("blk 50", TEAL), ("blk 51", TEAL)]):
        bx = x + 10 + j * 70
        d.rect(bx, ry + 32, 64, 26, fill=col[1], stroke=col[0], rx=3, sw=1.2)
        d.text(bx + 32, ry + 50, lab, 11, col[0], "bold", "middle", mono=True)
    d.text(x + 10, ry + 80, "32 KiB per block id", 11.5, MUTE, mono=True)
d.text(46 + 3 * (RW + 46) - 30, ry + 52, "...", 14, MUTE, "bold")
d.text(46, ry + 122, "12 regions, one per layer pair. Block 41 belongs to the full group, block 50 to the sliding group: one",
       12, INK)
d.text(46, ry + 140, "owner per block id, so the bytes never clash. bytes per block = 12 x 32 KiB = 384 KiB in both groups.",
       12, INK)

y = y2 + P3_H + 22
d.legend(30, y, [(BLUE, "full-attention group"), (TEAL, "sliding-window group"), (AMBER, "shared pool")])
y += 36
y = d.footnotes(30, y, [
    "Model numbers are from the Hugging Face config of gpt-oss-20b (not in the vLLM repo). Block ids are illustrative.",
    "Layout shown: layer-outermost. A block-outermost layout packs the groups per block instead (kv_cache_planning.py).",
])
d.save(height=y + 12)
