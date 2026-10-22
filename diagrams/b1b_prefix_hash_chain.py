"""Prefix caching: the block hash chain and reuse across two chat turns.

Used by: src/learn/prefix-caching.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/core/kv_cache_utils.py:707-737   hash_block_tokens(): hash((parent_hash, block_tokens, extra_keys)); NONE_HASH for block 0
  vllm/v1/core/kv_cache_utils.py:966-1010  request_block_hasher(): only full blocks get a hash, each chained to the previous one
  vllm/v1/core/kv_cache_utils.py:840-858   one KV cache group: hash block size = block size (16 by default)
  vllm/v1/core/single_type_kv_cache_manager.py:791-799  longest run of cached blocks from the start; stop at the first miss
  vllm/v1/core/kv_cache_manager.py:289-300 max hit length = num_tokens - 1
  vllm/config/cache.py:182                 DEFAULT_BLOCK_SIZE = 16
Example numbers are made up: turn 1 = 75 prompt tokens + 45 output tokens; turn 2 = 150 prompt tokens,
of which the first 118 equal the tokens of turn 1.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, NEUTRAL, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Prefix caching: a chain of block hashes, reused by the next chat turn",
         "block size 16  ·  one KV cache group  ·  default hash sha256  ·  made-up token counts",
         params="hash(block i) = sha256((hash(block i-1), tokens[16i : 16i+16], extra_keys))   hash(block -1) = NONE_HASH")

BW, GAP = 84, 14
X0 = 60


def bx(i):
    return X0 + i * (BW + GAP)


def block(i, y, col, top, bottom, h=58, dash=None):
    x = bx(i)
    d.rect(x, y, BW, h, fill=col[1], stroke=col[0], rx=4, sw=1.4, dash=dash)
    d.text(x + BW / 2, y + 22, top, 12, col[0], "bold", "middle", mono=True)
    d.text(x + BW / 2, y + 42, bottom, 11.5, INK, anchor="middle", mono=True)


# ---------------------------------------------------------------- panel 1
y0 = 120
P1_H = 236
d.panel(y0, P1_H, 1, "Turn 1: every full block gets a hash that includes its parent hash",
        "Request A: 75 prompt tokens + 45 output tokens. 7 full blocks get hashes h0 ... h6. The last block is not full.")
by = y0 + 96
d.text(X0, by - 12, "request A, 120 tokens", 12, MUTE, "bold", mono=True)
for i in range(7):
    block(i, by, BLUE, f"h{i}", f"{16 * i}-{16 * i + 15}")
block(7, by, NEUTRAL, "no hash", "112-119", dash="4 3")
# chain arrows between blocks
for i in range(6):
    d.arrow(bx(i) + BW + 1, by + 29, bx(i + 1) - 2, by + 29, BLUE, sw=1.6)
d.text(bx(8) + 4, by + 22, "8 of 16 tokens:", 11.5, MUTE)
d.text(bx(8) + 4, by + 39, "only full blocks", 11.5, MUTE)
d.text(bx(8) + 4, by + 56, "get a hash", 11.5, MUTE)
d.aside(X0, by + 86, [
    "h3 depends on h2, h2 on h1, h1 on h0. So h3 names the whole prefix of 64 tokens, not only the 16 tokens of block 3.",
    "After the request ends, the 7 hashed blocks stay in the cache as free blocks until the block pool needs them again (LRU).",
])

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 262
d.panel(y1, P2_H, 2, "Turn 2: look up the hashes from block 0, stop at the first miss",
        "Request B: 150 prompt tokens. Tokens 0-117 equal request A. Blocks 0-6 hit, block 7 has new tokens and misses.")
by = y1 + 96
d.text(X0, by - 12, "request B, 150 prompt tokens", 12, MUTE, "bold", mono=True)
for i in range(7):
    block(i, by, EMERALD, f"h{i} hit", f"{16 * i}-{16 * i + 15}")
block(7, by, AMBER, "h7' miss", "112-127")
block(8, by, AMBER, "h8' miss", "128-143")
block(9, by, NEUTRAL, "no hash", "144-149", dash="4 3")
for i in range(8):
    d.arrow(bx(i) + BW + 1, by + 29, bx(i + 1) - 2, by + 29, EMERALD if i < 6 else AMBER, sw=1.6)
d.text(bx(7) + BW / 2, by + 76, "tokens 118+ are new", 11.5, AMBER, "bold", "middle")
# brackets under the blocks
yb = by + 98
d.line(bx(0), yb, bx(6) + BW, yb, EMERALD[0], sw=2)
d.text((bx(0) + bx(6) + BW) / 2, yb + 18, "7 blocks x 16 = 112 tokens from the cache: no prefill for them",
       12, EMERALD[0], "bold", "middle")
d.line(bx(7), yb, bx(9) + BW, yb, AMBER[0], sw=2)
d.text((bx(7) + bx(9) + BW) / 2, yb + 18, "150 - 112 = 38 tokens to compute", 12, AMBER[0], "bold", "middle")
d.aside(X0, yb + 46, [
    "Block 8 misses too: its hash h8' depends on h7', which is new. A hit can only be a run of blocks from the start.",
])

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "hashed block in the cache"), (EMERALD, "cache hit"), (AMBER, "miss: computed"),
                 (NEUTRAL, "partial block: no hash")])
y += 40
y = d.footnotes(30, y, [
    "extra_keys: LoRA name and path, multimodal item hashes, the cache_salt (block 0 only) and prompt-embedding hashes.",
    "Not shown: hybrid models with several KV cache groups, finer prefix_match_unit hashing, and the hit limit of num_tokens - 1.",
])
d.save(height=y + 14)
