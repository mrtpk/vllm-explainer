"""B3a module page "v1/core/block_pool.py": the free block queue and the life of one block.

Used by: src/code/v1-core-block-pool.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/core/kv_cache_utils.py:258-310     FreeKVCacheBlockQueue: doubly linked list with a fake head and tail
  vllm/v1/core/kv_cache_utils.py:347-389     popleft_n (front)
  vllm/v1/core/kv_cache_utils.py:391-412     remove (O(1), from the middle)
  vllm/v1/core/kv_cache_utils.py:438-484     prepend_n, append_n
  vllm/v1/core/block_pool.py:184-188         block 0 is popped once and becomes the null block
  vllm/v1/core/block_pool.py:666-700         get_new_blocks: popleft_n, evict old hash, ref_cnt = 1
  vllm/v1/core/block_pool.py:226-299         cache_full_blocks sets the hash of a full block
  vllm/v1/core/block_pool.py:752-770         touch: remove from the queue if ref_cnt == 0, ref_cnt += 1
  vllm/v1/core/block_pool.py:776-807         free_blocks: unhashed -> prepend_n, hashed -> append_n
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, LINE, MUTE, PINK, VIOLET, WHITE, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("BlockPool: the free block queue and the life of one block",
         "One BlockPool per scheduler, shared by all KV cache groups (a HiSparse host group has its own)  ·  Python objects only",
         params="KVCacheBlock(block_id, ref_cnt, _block_hash, prev_free_block, next_free_block, is_null)")

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 218
d.panel(y0, P1_H, 1, "How is the free queue ordered?",
        "A doubly linked list through the blocks themselves. Take from the front; freed blocks go to the front or the end.")
ly = y0 + 104
cells = [("head", None), ("9", PINK), ("40", PINK), ("15", None), ("21", None), ("12", EMERALD), ("3", EMERALD),
         ("7", EMERALD), ("tail", None)]
cw, gap = 78, 40
x0 = 66
for i, (lab, col) in enumerate(cells):
    x = x0 + i * (cw + gap) * 0.95
    if lab in ("head", "tail"):
        d.rect(x, ly, cw, 40, fill=WHITE, stroke=LINE, rx=6, sw=1.2, dash="4 3")
        d.text(x + cw / 2, ly + 25, f"fake {lab}", 11.5, MUTE, "normal", "middle", mono=True)
    else:
        c = col or (INK, "#f3f4f6", WHITE)
        d.rect(x, ly, cw, 40, fill=c[1], stroke=c[0], rx=6, sw=1.4)
        d.text(x + cw / 2, ly + 18, f"block {lab}", 11.5, c[0], "bold", "middle", mono=True)
        d.text(x + cw / 2, ly + 33, "hash" if col is EMERALD else "no hash", 11, MUTE, "normal", "middle", mono=True)
    if i < len(cells) - 1:
        xn = x0 + (i + 1) * (cw + gap) * 0.95
        d.line(x + cw, ly + 14, xn, ly + 14, INK, 1.2)
        d.line(x + cw, ly + 26, xn, ly + 26, INK, 1.2)
d.text(x0 + (cw + gap) * 0.95 + cw / 2, ly - 22, "get_new_blocks(): popleft_n", 11.5, BLUE, "bold", "middle")
d.arrow(x0 + (cw + gap) * 0.95 + cw / 2, ly - 16, x0 + (cw + gap) * 0.95 + cw / 2, ly - 2, BLUE)
xf = x0 + 2 * (cw + gap) * 0.95 + cw / 2
d.text(xf + 70, ly - 22, "freed, no hash: prepend_n", 11.5, PINK, "bold", "middle")
d.arrow(xf, ly - 16, xf, ly - 2, PINK)
xm = x0 + 6 * (cw + gap) * 0.95 + cw / 2
d.text(xm - 60, ly + 66, "touch() on a cached hit: remove() from the middle, O(1)", 11.5, VIOLET, "bold", "middle")
d.arrow(xm, ly + 54, xm, ly + 42, VIOLET)
xe = x0 + 7 * (cw + gap) * 0.95 + cw / 2
d.text(xe, ly - 22, "freed, hash: append_n", 11.5, EMERALD, "bold", "middle")
d.arrow(xe, ly - 16, xe, ly - 2, EMERALD)
d.text(46, ly + 96, "Front = evicted first. Unhashed blocks never match a prefix, so the pool reuses them first; cached blocks wait at the end (LRU).",
       12, MUTE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 236
d.panel(y1, P2_H, 2, "What happens to one block over time?",
        "ref_cnt counts the requests that use it. The hash stays after the free, until the block is popped again.")
XS = [46, 400, 754]
BW = 300
by = y1 + 70
BH = 120
d.box(XS[0], by, BW, BH, "1. Allocated", [
    "get_new_blocks(): pop front,",
    "old hash removed (eviction),",
    "ref_cnt 0 -> 1",
    "full: cache_full_blocks()",
    "  sets block_hash",
], col=BLUE, align="start")
d.box(XS[1], by, BW, BH, "2. Shared / freed", [
    "prefix hit: touch() ref_cnt += 1",
    "request ends: free_blocks()",
    "  ref_cnt -= 1",
    "ref_cnt 0: back in the queue",
    "  (hash kept, still a hit)",
], col=EMERALD, align="start")
d.box(XS[2], by, BW, BH, "3. Hit again or reused", [
    "hit while free: touch()",
    "  remove(), ref_cnt 0 -> 1",
    "reached the front: popped by",
    "  get_new_blocks(), hash evicted",
    "  -> back to 1. Allocated",
], col=AMBER, align="start")
d.arrow(XS[0] + BW, by + 40, XS[1] - 2, by + 40, BLUE)
d.arrow(XS[1] + BW, by + 40, XS[2] - 2, by + 40, EMERALD)
d.text(46, by + BH + 26, "Block 0 never enters this cycle: the pool pops it at start-up and marks it as the null block.", 12, MUTE)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "allocation"), (PINK, "freed without hash"), (EMERALD, "cached (has hash)"),
                 (VIOLET, "hit on a free block"), (AMBER, "reuse")])
y += 38
y = d.footnotes(30, y, [
    "Simplified: one KV cache group, prefix caching on. Block ids are illustrative. With prefix caching off, every freed block goes to the front.",
])
d.save(height=y + 14)
