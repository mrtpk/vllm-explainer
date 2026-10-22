"""B3b concept page "Prefix caching in the engine core": the life of one cached block.

Used by: src/concepts/prefix-caching.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/core/block_pool.py:666-700        get_new_blocks: popleft_n, _maybe_evict_cached_block, ref_cnt 0 -> 1
  vllm/v1/core/kv_cache_manager.py:602-616  allocate_slots caches blocks that are full after this step (not with
                                            delay_cache_blocks, i.e. an async KV connector load)
  vllm/v1/core/sched/async_scheduler.py:75-79   AsyncScheduler caches again when the sampled tokens arrive
  vllm/v1/core/block_pool.py:226-344        cache_full_blocks: set_block_hash, map insert, BlockStored event
  vllm/v1/core/single_type_kv_cache_manager.py:590-598   free in reverse order
  vllm/v1/core/block_pool.py:776-807        free_blocks: ref_cnt -= 1; at 0 a hashed block goes to the back
  vllm/v1/core/block_pool.py:752-770        touch: hit on a free block -> remove from the queue, ref_cnt += 1
  vllm/v1/core/block_pool.py:729-750, 611-624   eviction removes the hash and emits BlockRemoved
  vllm/v1/core/block_pool.py:831-870        reset_prefix_cache: only when no block is in use; AllBlocksCleared
  vllm/v1/core/sched/scheduler.py:3515-3516  a failed KV connector load can evict blocks (evict_blocks)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The life of one cached block",
         "One KV cache group, prefix caching on  ·  ref_cnt = requests that hold the block  ·  # = the block has a hash",
         params="get_new_blocks -> cache_full_blocks -> free_blocks -> touch (hit) or get_new_blocks (eviction)")

XS = [46, 400, 754]
BW = 300

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 248
d.panel(y0, P1_H, 1, "From a free block to a cached free block",
        "The hash is set when the block becomes full. It stays when the last request lets the block go.")
by = y0 + 70
BH = 136
d.box(XS[0], by, BW, BH, "1. Allocated", [
    "get_new_blocks(): pop the",
    "  front of the free queue",
    "old hash? remove it (eviction)",
    "ref_cnt 0 -> 1",
    "the request writes tokens into it",
], col=BLUE, align="start")
d.box(XS[1], by, BW, BH, "2. Full: hashed #", [
    "allocate_slots() caches every",
    "  block that is full after this",
    "  step, before the step runs",
    "set_block_hash(hash + group id)",
    "map insert; BlockStored event",
    "other requests can hit it now",
], col=EMERALD, align="start")
d.box(XS[2], by, BW, BH, "3. Freed, still cached #", [
    "request ends: free in reverse",
    "  order, ref_cnt -= 1",
    "ref_cnt > 0: another request",
    "  holds it, stays out of queue",
    "ref_cnt 0: append to the BACK",
    "  of the free queue, hash kept",
], col=EMERALD, align="start")
d.arrow(XS[0] + BW, by + 44, XS[1] - 2, by + 44, BLUE)
d.arrow(XS[1] + BW, by + 44, XS[2] - 2, by + 44, EMERALD)
d.text(46, by + BH + 26, "A block that is not full when its request ends has no hash: it goes to the FRONT and is reused first.",
       12, MUTE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 248
d.panel(y1, P2_H, 2, "Two ways out of the free queue",
        "A hit takes the block back into use. Otherwise it drifts to the front and the pool reuses it for other tokens.")
by2 = y1 + 70
d.box(XS[0], by2, BW, BH, "4a. Hit again", [
    "a new request's lookup finds",
    "  the hash in the map",
    "touch(): remove from the free",
    "  queue (O(1)), ref_cnt 0 -> 1",
    "-> back to state 2 (in use,",
    "  hash kept, no copy)",
], col=VIOLET, align="start")
d.box(XS[1], by2, BW, BH, "4b. Evicted", [
    "it reaches the front and",
    "  get_new_blocks() pops it",
    "hash removed from the map,",
    "  BlockRemoved event",
    "-> state 1 for another request,",
    "  with other tokens",
], col=PINK, align="start")
d.box(XS[2], by2, BW, BH, "4c. Reset", [
    "reset_prefix_cache(): drops",
    "  every hash at once",
    "only when no block is in use",
    "  (else WARNING, False)",
    "AllBlocksCleared event",
    "the dev route can call it",
], col=AMBER, align="start")
d.text(46, by2 + BH + 26, "No timer and no size limit: a cached block lives until 4b or 4c (or a failed KV connector load evicts it).", 12, MUTE)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "in use, not full"), (EMERALD, "has a hash"), (VIOLET, "hit"), (PINK, "eviction"),
                 (AMBER, "reset")])
y += 36
y = d.footnotes(30, y, [
    "Not shown: with an async KV connector load (delay_cache_blocks), caching waits until the transfer is done. The",
    "AsyncScheduler also caches when the sampled tokens arrive. Events are queued only with KV events on.",
])
d.save(height=y + 12)
