"""B3b concept page "Prefix caching in the engine core": the lookup walk over a hash chain, with ref counts.

Used by: src/concepts/prefix-caching.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/core/kv_cache_utils.py:966-1012          request block hasher: only full blocks, parent -> child chain
  vllm/v1/core/kv_cache_utils.py:707-737           hash_block_tokens(parent or NONE_HASH, tokens, extra keys)
  vllm/v1/core/kv_cache_manager.py:289-300         max_cache_hit_length = num_tokens - 1
  vllm/v1/core/single_type_kv_cache_manager.py:791-799   walk from block 0, stop at the first miss
  vllm/v1/core/block_pool.py:198-224               get_cached_block: hash + group id -> first block
  vllm/v1/core/single_type_kv_cache_manager.py:256-266   free hit blocks (ref_cnt 0) count against free capacity
  vllm/v1/core/block_pool.py:752-770               touch: out of the free queue if ref_cnt == 0, ref_cnt += 1
  vllm/v1/core/block_pool.py:666-700, 729-750      get_new_blocks: pop the front, evict an old hash, ref_cnt = 1
  vllm/v1/core/kv_cache_manager.py:607-616         cache the blocks that are full after this step
Block ids and the queue are illustrative; the rules are from the code.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, LINE, MUTE, PINK, VIOLET, WHITE, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("get_computed_blocks(): walk the hash chain, stop at the first miss",
         "One KV cache group (full attention)  ·  block_size 16  ·  a new request with 90 prompt tokens",
         params="max_cache_hit_length = num_tokens - 1 = 89   ·   hashes checked: at most 89 // 16 = 5")

COLS = 6
CW, GAP = 152, 18
X0 = 46
xs = [X0 + i * (CW + GAP) for i in range(COLS)]

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 176
d.panel(y0, P1_H, 1, "Which hashes does the request have?",
        "The engine core hashed every full block when the request arrived. The last 10 tokens fill no block: no hash.")
cy = y0 + 70
CH = 74
labels = [("block 0", "tokens 0-15", "h0 = H(NONE_HASH,", "  tokens, keys)"),
          ("block 1", "tokens 16-31", "h1 = H(h0, ...)", ""),
          ("block 2", "tokens 32-47", "h2 = H(h1, ...)", ""),
          ("block 3", "tokens 48-63", "h3 = H(h2, ...)", ""),
          ("block 4", "tokens 64-79", "h4 = H(h3, ...)", ""),
          ("block 5", "tokens 80-89", "not full", "no hash")]
for i, (a, b, c, e) in enumerate(labels):
    col = BLUE if i < 5 else (MUTE, "#f3f4f6", WHITE)
    d.rect(xs[i], cy, CW, CH, fill=col[1], stroke=col[0], rx=6, sw=1.4, dash=None if i < 5 else "4 3")
    d.text(xs[i] + 10, cy + 18, a, 12, col[0], "bold", mono=True)
    d.text(xs[i] + 10, cy + 34, b, 11.5, INK, mono=True)
    d.text(xs[i] + 10, cy + 51, c, 11.5, INK, mono=True)
    if e:
        d.text(xs[i] + 10, cy + 66, e, 11.5, INK if i < 5 else MUTE, mono=True)
    if i < 4:
        d.arrow(xs[i] + CW, cy + CH / 2, xs[i + 1] - 2, cy + CH / 2, BLUE)
d.text(X0, cy + CH + 26, "Each arrow carries the parent hash into the next hash, so h4 names all 80 tokens before it, "
       "plus the extra keys (LoRA, images, salt).", 12, MUTE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 214
d.panel(y1, P2_H, 2, "What does the hash map return?",
        "get_cached_block(h, [group 0]) for h0, h1, ... A miss ends the walk: every later hash includes the missed one.")
ry = y1 + 70
RH = 84
res = [
    (EMERALD, "HIT", "block 7", "ref_cnt 1", "(R1 uses it)"),
    (EMERALD, "HIT", "block 3", "ref_cnt 0", "(in free queue)"),
    (EMERALD, "HIT", "block 12", "ref_cnt 0", "(in free queue)"),
    (AMBER, "MISS", "no block", "stop here", ""),
    ((MUTE, "#f3f4f6", WHITE), "-", "not checked", "", ""),
    ((MUTE, "#f3f4f6", WHITE), "-", "no hash", "", ""),
]
for i, (col, tag, a, b, c) in enumerate(res):
    d.rect(xs[i], ry, CW, RH, fill=col[1], stroke=col[0], rx=6, sw=1.4)
    d.text(xs[i] + 10, ry + 19, f"map[h{i}]" if i < 5 else "block 5", 11.5, MUTE, mono=True)
    d.text(xs[i] + CW - 10, ry + 19, tag, 12, col[0], "bold", "end", mono=True)
    d.text(xs[i] + 10, ry + 40, a, 12, col[0], "bold", mono=True)
    if b:
        d.text(xs[i] + 10, ry + 57, b, 11.5, INK, mono=True)
    if c:
        d.text(xs[i] + 10, ry + 73, c, 11.5, MUTE, mono=True)
d.chip(X0, ry + RH + 18, 520, "hit = 3 blocks x 16 = 48 tokens; compute 90 - 48 = 42", EMERALD)
d.text(X0 + 540, ry + RH + 38, "Admission count: 3 new blocks + 2 free hit blocks = 5 free blocks needed.", 12, MUTE)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 20
P3_H = 292
d.panel(y2, P3_H, 3, "What does allocate_slots() change?",
        "touch() first takes the hits out of the free queue. Then get_new_blocks() pops new blocks from the front.")
qy = y2 + 76
QW, QG = 58, 6
d.text(X0, qy - 8, "free queue (front -> back)", 12, INK, "bold")


def strip(y, items, label):
    d.text(X0, y + 21, label, 11.5, MUTE, mono=True)
    x = X0 + 112
    for lab, kind in items:
        col = {"plain": (MUTE, "#f3f4f6", WHITE), "hash": EMERALD, "gone": (LINE, WHITE, WHITE)}[kind]
        d.rect(x, y, QW, 30, fill=col[1], stroke=col[0], rx=4, sw=1.2, dash="3 3" if kind == "gone" else None)
        d.text(x + QW / 2, y + 20, lab, 11.5, col[0] if kind != "gone" else MUTE, "bold", "middle", mono=True)
        x += QW + QG


strip(qy, [("9", "plain"), ("40", "plain"), ("3 #", "hash"), ("12 #", "hash"), ("21 #", "hash"), ("5 #", "hash")],
      "before")
strip(qy + 40, [("9", "plain"), ("40", "plain"), ("21 #", "hash"), ("5 #", "hash")], "after touch")
strip(qy + 80, [("5 #", "hash")], "after pop 3")
d.text(X0, qy + 134, "touch(7, 3, 12): ref_cnt 1->2, 0->1, 0->1;", 11.5, VIOLET, "bold", mono=True)
d.text(X0, qy + 150, "  blocks 3 and 12 leave the queue (O(1) remove).", 11.5, VIOLET, mono=True)
d.text(X0, qy + 168, "get_new_blocks(3) pops 9, 40, 21; block 21 loses", 11.5, PINK, "bold", mono=True)
d.text(X0, qy + 184, "  its old hash: that is the eviction.", 11.5, PINK, mono=True)

tx = 596
d.text(tx, qy - 8, "block table of the new request", 12, INK, "bold")
tbl = [("7", "h0", "2", EMERALD), ("3", "h1", "1", EMERALD), ("12", "h2", "1", EMERALD),
       ("9", "h3", "1", BLUE), ("40", "h4", "1", BLUE), ("21", "-", "1", BLUE)]
TW = 68
for i, (bid, h, rc, col) in enumerate(tbl):
    x = tx + i * (TW + 6)
    d.rect(x, qy, TW, 76, fill=col[1], stroke=col[0], rx=4, sw=1.3)
    d.text(x + TW / 2, qy + 20, f"blk {bid}", 11.5, col[0], "bold", "middle", mono=True)
    d.text(x + TW / 2, qy + 42, h, 11.5, INK, "normal", "middle", mono=True)
    d.text(x + TW / 2, qy + 63, f"ref {rc}", 11.5, MUTE, "normal", "middle", mono=True)
d.text(tx, qy + 98, "cache_full_blocks: 90 // 16 = 5 full blocks.", 11.5, INK, mono=True)
d.text(tx, qy + 114, "Blocks 9 and 40 get h3 and h4 now, before", 11.5, INK, mono=True)
d.text(tx, qy + 130, "the GPU runs the step. Block 21 has 10", 11.5, INK, mono=True)
d.text(tx, qy + 146, "tokens: no hash until it is full.", 11.5, INK, mono=True)

y = y2 + P3_H + 22
d.legend(30, y, [(EMERALD, "prefix-cache hit / cached block"), (BLUE, "new block"), (AMBER, "miss"),
                 (VIOLET, "touch"), (PINK, "eviction")])
y += 36
y = d.footnotes(30, y, [
    "# = block with a hash. Block ids and the queue are illustrative. With several KV cache groups, the coordinator runs this",
    "walk per group and keeps the shortest common hit (see the hybrid KV cache page).",
])
d.save(height=y + 12)
