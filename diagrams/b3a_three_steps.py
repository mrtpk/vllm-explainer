"""B3a concept "The scheduler": a worked example of three schedule() calls with three requests,
a token budget of 16 and 8 KV blocks of 4 tokens; step 3 preempts request C.

Used by: src/concepts/scheduler.html
The numbers are illustrative (small on purpose). The rules are from the code:
  vllm/v1/core/sched/scheduler.py:713-904     running pass first; allocate_slots; FCFS victim = running.pop()
  vllm/v1/core/sched/scheduler.py:956-981     waiting pass only if nothing was preempted; max_num_seqs cap
  vllm/v1/core/sched/scheduler.py:1191-1231   chunk = min(num_tokens - computed, budget)
  vllm/v1/core/kv_cache_manager.py:522-539    a new request is admitted only if all its tokens fit
  vllm/v1/core/sched/scheduler.py:1658-1701   preemption: free blocks, num_computed_tokens = 0, front of waiting
  vllm/v1/core/block_pool.py:776-807          free order: unhashed block to the front, cached block to the end
  vllm/v1/core/sched/utils.py:114-119         a request stops at max_tokens
Simplified: synchronous (each step's token is known before the next schedule), no prefix-cache hit
for A and B, block 0 (the null block) not drawn.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BORDER, INK, LINE, MUTE, SERIES, WHITE, Diagram  # noqa: E402

COL = {"A": SERIES[0], "B": SERIES[1], "C": SERIES[2]}
TINT = {"A": "#dbeafe", "B": "#fce7f3", "C": "#dcfce7"}

d = Diagram(width=1100)
d.header("Three schedule() calls: chunking, decode and one preemption",
         "Token budget 16  ·  max_num_seqs 4  ·  block_size 4  ·  8 free KV blocks  ·  chunked prefill on  ·  FCFS",
         params="A: prompt 10, max_tokens 4   B: prompt 12, max_tokens 4   C: prompt 6, max_tokens 4   (all wait at step 1)")

STEPS = [
    {
        "title": "Step 1: A fits whole, B gets the rest of the budget",
        "sub": "Running pass: empty. Waiting pass: A takes 10 tokens, B takes a 6-token chunk. C waits: budget 0.",
        "budget": [("A", 10, "A 10 (prefill)"), ("B", 6, "B 6 (chunk)")],
        "blocks": ["A", "A", "A", "B", "B", None, None, None],
        "rows": [
            ("A", "RUNNING  computed 10/10 -> samples token 1   3 blocks"),
            ("B", "RUNNING  computed  6/12 -> chunk, no token   2 blocks"),
            ("C", "WAITING  budget used up: the pass stops"),
        ],
    },
    {
        "title": "Step 2: A decodes, B ends its prompt, C starts",
        "sub": "Running pass: A 1 token, B its last 6 prompt tokens. Waiting pass: C fits (2 blocks), 13 of 16 used.",
        "budget": [("A", 1, "A"), ("B", 6, "B 6"), ("C", 6, "C 6")],
        "blocks": ["A", "A", "A", "B", "B", "B", "C", "C"],
        "rows": [
            ("A", "RUNNING  1 token  -> token 2              3 blocks"),
            ("B", "RUNNING  6 tokens -> prompt done, token 1  3 blocks"),
            ("C", "RUNNING  6 tokens -> token 1               2 blocks"),
        ],
    },
    {
        "title": "Step 3: B needs a 4th block, none is free: preempt C",
        "sub": "B has 13 tokens = 4 blocks. allocate_slots returns None, so the last running request (C) is preempted.",
        "budget": [("A", 1, "A"), ("B", 1, "B")],
        "blocks": ["A", "A", "A", "B", "B", "B", "free", "B"],
        "rows": [
            ("A", "RUNNING  1 token -> token 3                3 blocks"),
            ("B", "RUNNING  1 token -> token 2                4 blocks"),
            ("C", "PREEMPTED  blocks freed, computed 0, front of waiting"),
        ],
        "note": "No waiting pass in this step (a request was preempted). C later recomputes 7 tokens (6 prompt + 1 output).",
    },
]

X0 = 46
CELL = 26          # one token of the budget
BCELL = 52         # one KV block
P_H = 214
y = 128
for i, st in enumerate(STEPS):
    d.panel(y, P_H, i + 1, st["title"], st["sub"])
    # token budget bar
    by = y + 86
    d.text(X0, by - 8, "token budget (16)", 11.5, MUTE, "bold")
    for k in range(16):
        d.rect(X0 + k * CELL, by, CELL - 2, 26, fill=WHITE, stroke=LINE, rx=2, sw=0.8)
    x = X0
    for rid, n, label in st["budget"]:
        w = n * CELL - 2
        d.rect(x, by, w, 26, fill=TINT[rid], stroke=COL[rid], rx=3, sw=1.4)
        d.text(x + w / 2, by + 17.5, label, 11.5, COL[rid], "bold", "middle", mono=True)
        x += n * CELL
    used = sum(n for _, n, _ in st["budget"])
    d.text(X0 + 16 * CELL + 6, by + 17.5, f"{used}/16", 12, INK, "bold", mono=True)
    # KV blocks
    ky = by + 58
    d.text(X0, ky - 8, "KV blocks 1-8 (4 tokens each)", 11.5, MUTE, "bold")
    for k, owner in enumerate(st["blocks"]):
        bx = X0 + k * BCELL
        if owner in COL:
            d.rect(bx, ky, BCELL - 4, 30, fill=TINT[owner], stroke=COL[owner], rx=3, sw=1.4)
            d.text(bx + (BCELL - 4) / 2, ky + 19.5, f"{owner} {k + 1}", 11.5, COL[owner], "bold", "middle", mono=True)
        else:
            dash = "4 3" if owner == "free" else None
            d.rect(bx, ky, BCELL - 4, 30, fill=WHITE, stroke=AMBER[0] if owner == "free" else BORDER,
                   rx=3, sw=1.2, dash=dash)
            d.text(bx + (BCELL - 4) / 2, ky + 19.5, f"{k + 1}", 11.5, AMBER[0] if owner == "free" else MUTE,
                   "normal", "middle", mono=True)
    # request states
    rx = 548
    for j, (rid, txt) in enumerate(st["rows"]):
        ry = by + 4 + j * 22
        d.text(rx, ry + 12, rid, 12.5, COL[rid], "bold", mono=True)
        d.text(rx + 20, ry + 12, txt, 11.5, AMBER[0] if "PREEMPTED" in txt else INK, mono=True)
    if "note" in st:
        d.text(X0, ky + 56, st["note"], 12, AMBER[0], "bold")
    else:
        free = sum(1 for o in st["blocks"] if o is None)
        d.text(X0, ky + 56, f"Free blocks after the step: {free}.", 12, MUTE)
    y += P_H + 20

d.legend(30, y + 4, [((COL["A"], TINT["A"], WHITE), "request A"), ((COL["B"], TINT["B"], WHITE), "request B"),
                     ((COL["C"], TINT["C"], WHITE), "request C"), (AMBER, "freed by preemption")])
y += 42
y = d.footnotes(30, y, [
    "Simplified: synchronous steps (the default AsyncScheduler plans step k+1 before the token of step k arrives), no EOS,",
    "  block 0 (the null block) not drawn. In step 3 the pool frees C's blocks in reverse order: block 8 (not full) goes to the",
    "  front of the free queue and B gets it; block 7 (full, cached) goes to the end. C can hit it later, unless a request pops it first.",
])
d.save(height=y + 14)
