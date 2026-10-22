"""B3a concept "The KV cache manager": the object hierarchy, one allocate_slots() call, and free().

Used by: src/concepts/kv-cache-manager.html, src/code/v1-core-kv-cache-manager.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/core/sched/scheduler.py:329-347               the Scheduler makes the KVCacheManager
  vllm/v1/core/kv_cache_manager.py:167-180, 197         coordinator and its block pool
  vllm/v1/core/kv_cache_coordinator.py:119-125, 157-172 one BlockPool, one SingleTypeKVCacheManager per KV cache group
  vllm/v1/core/kv_cache_coordinator.py:995-1038         NoPrefixCache / Unitary / Hybrid coordinator
  vllm/v1/core/block_pool.py:171-188                    blocks, free queue, hash map, null block
  vllm/v1/core/kv_cache_manager.py:499-618              allocate_slots
  vllm/v1/core/single_type_kv_cache_manager.py:214-266  get_num_blocks_to_allocate (evictable hits count)
  vllm/v1/core/single_type_kv_cache_manager.py:303-318  touch the hit blocks
  vllm/v1/core/single_type_kv_cache_manager.py:400-410  get new blocks
  vllm/v1/core/single_type_kv_cache_manager.py:590-598  free in reverse order
  vllm/v1/core/block_pool.py:666-700, 752-770, 776-807  get_new_blocks, touch, free_blocks
Example: the request of Tour 1 (61 prompt tokens, 48 in the prefix cache, block_size 16, block ids illustrative).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The KV cache manager: who owns the blocks, and one allocate_slots() call",
         "Engine core process, scheduler side  ·  bookkeeping only: the KV tensors live on the GPU in the workers",
         params="block_size 16  ·  request: 61 prompt tokens, 48 hit in the prefix cache (blocks 7, 3, 12)  ·  13 new tokens")

XS = [46, 400, 754]
BW = 300

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 330
d.panel(y0, P1_H, 1, "Who owns what?",
        "Four layers. The BlockPool owns every KVCacheBlock; the Scheduler gets KVCacheBlocks and block ids.")
r1 = y0 + 70
BH = 96
d.box(XS[0], r1, BW, BH, "Scheduler", [
    "self.kv_cache_manager",
    "allocate_slots(), free(),",
    "get_computed_blocks(), usage",
], col=INK, align="start")
d.box(XS[1], r1, BW, BH, "KVCacheManager", [
    "kv_cache_manager.py",
    "the API of the scheduler",
    "returns KVCacheBlocks (per group)",
], col=BLUE, align="start")
d.box(XS[2], r1, BW, BH, "KVCacheCoordinator", [
    "NoPrefixCache | Unitary | Hybrid",
    "loops over the KV cache groups",
    "owns the one BlockPool",
], col=VIOLET, align="start")
d.arrow(XS[0] + BW, r1 + 40, XS[1] - 2, r1 + 40, INK)
d.arrow(XS[1] + BW, r1 + 40, XS[2] - 2, r1 + 40, BLUE)
r2 = r1 + BH + 40
BH2 = 108
d.box(XS[1], r2, BW, BH2, "SingleTypeKVCacheManager", [
    "one per KV cache group",
    "FullAttention, SlidingWindow, Mamba ...",
    "req_to_blocks[req_id]  (block table)",
    "num_cached_block[req_id]",
], col=TEAL, align="start")
d.box(XS[2], r2, BW, BH2, "BlockPool", [
    "blocks[0 .. num_gpu_blocks-1]",
    "free_block_queue (LRU order)",
    "cached_block_hash_to_block",
    "null_block = blocks[0]",
], col=EMERALD, align="start")
d.path_arrow(f"M {XS[2] + BW / 2},{r1 + BH} L {XS[2] + BW / 2},{r1 + BH + 20} "
             f"L {XS[1] + BW / 2},{r1 + BH + 20} L {XS[1] + BW / 2},{r2 - 2}", VIOLET)
d.arrow(XS[2] + BW / 2 + 40, r1 + BH, XS[2] + BW / 2 + 40, r2 - 2, VIOLET)
d.arrow(XS[1] + BW, r2 + 50, XS[2] - 2, r2 + 50, TEAL)
d.aside(XS[0], r2 + 20, [
    "The Scheduler makes the manager in its",
    "__init__. The manager makes the",
    "coordinator; the coordinator makes",
    "the BlockPool and one single-type",
    "manager per KV cache group.",
])

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 268
d.panel(y1, P2_H, 2, "allocate_slots(request, 13, num_new_computed_tokens=48, new_computed_blocks=[7, 3, 12])",
        "Count first, fail early with None, then touch the hits, take new blocks, and cache the full blocks.")
by = y1 + 70
BH = 166
d.step_card(XS[0], by, BW, BH, 1, "Count and check", BLUE, "no change yet", rows=[
    ("tok", "48 computed + 13 new = 61"),
    ("need", "cdiv(61, 16) = 4 blocks"),
    ("have", "3 hit blocks -> 1 new"),
    ("+", "3 hits with ref_cnt 0 count"),
    ("", "  (they leave the free queue)"),
    ("if", "4 > free blocks: return None"),
], key_w=40)
d.step_card(XS[1], by, BW, BH, 2, "Touch hits, take new", TEAL, "blocks move", rows=[
    ("hit", "touch([7, 3, 12])"),
    ("", "ref_cnt += 1; leave free queue"),
    ("new", "get_new_blocks(1) -> [40]"),
    ("", "from the queue front;"),
    ("", "old hash evicted; ref_cnt 1"),
    ("tab", "[7, 3, 12, 40]"),
], key_w=36)
d.step_card(XS[2], by, BW, BH, 3, "Cache full blocks", EMERALD, "prefix cache", rows=[
    ("n", "min(48 + 13, 61) = 61"),
    ("full", "61 // 16 = 3 blocks"),
    ("", "all 3 cached: nothing new"),
    ("ret", "KVCacheBlocks(([40],))"),
    ("", "a new request gets all ids:"),
    ("", "block_ids=([7, 3, 12, 40],)"),
], key_w=40)
d.arrow(XS[0] + BW, by + 40, XS[1] - 2, by + 40, BLUE)
d.arrow(XS[1] + BW, by + 40, XS[2] - 2, by + 40, TEAL)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 196
d.panel(y2, P3_H, 3, "free(request): what happens when the request ends?",
        "Blocks go back in reverse order. A block returns to the free queue only when its ref_cnt reaches 0.")
by = y2 + 70
BH = 100
d.box(XS[0], by, BW, BH, "single-type manager", [
    "pop req_to_blocks[req_id]",
    "free_blocks(reversed(blocks))",
    "-> [40, 12, 3, 7]",
], col=TEAL, align="start")
d.box(XS[1], by, BW, BH, "BlockPool.free_blocks", [
    "ref_cnt -= 1 for each block",
    "ref_cnt 0 and no hash:",
    "  front of the free queue",
    "ref_cnt 0 with hash: the end",
], col=EMERALD, align="start")
d.box(XS[2], by, BW, BH, "free queue (front -> end)", [
    "40 | ... other free ... | 12, 3, 7",
    "40: reused first (no hash)",
    "7, 3, 12: kept for prefix hits",
    "  until popped from the front",
], col=AMBER, align="start")
d.arrow(XS[0] + BW, by + 40, XS[1] - 2, by + 40, TEAL)
d.arrow(XS[1] + BW, by + 40, XS[2] - 2, by + 40, EMERALD)

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "KVCacheManager"), (VIOLET, "coordinator"), (TEAL, "single-type manager"),
                 (EMERALD, "BlockPool"), (AMBER, "free queue order")])
y += 38
y = d.footnotes(30, y, [
    "Simplified: one KV cache group (full attention), no lookahead or spec tokens, no KV connector. Block ids are illustrative.",
    "Panel 3 assumes that block 40 never filled up (the request ended before 64 tokens) and that no other request shares 7, 3, 12.",
])
d.save(height=y + 14)
