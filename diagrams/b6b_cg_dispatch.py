"""B6b: how Model Runner V2 picks FULL, PIECEWISE or NONE for one batch.

Used by: src/concepts/compile-and-cuda-graphs.html, src/code/v1-worker-gpu-model-runner.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/worker/gpu/model_runner.py:1777-1819     gather state, uniform count, need_eager, dispatch_cg_and_sync_dp
  vllm/v1/worker/utils.py:958-976                  uniform decode token count
  vllm/v1/worker/gpu/dp_utils.py:256-276           need_eager -> NONE, else manager.dispatch
  vllm/v1/worker/gpu/cudagraph_utils.py:256-394    candidates: FULL descriptors first, then PIECEWISE
  vllm/v1/worker/gpu/cudagraph_utils.py:111-138    _is_compatible
  vllm/v1/worker/gpu/cudagraph_utils.py:523-553    dispatch: first compatible candidate, else NONE
  vllm/config/vllm.py:2440-2449, 2557-2570         capture sizes; max 512 here
Example: one H100, max_num_seqs 1024, max_num_batched_tokens 8192, no speculative decoding (decode_query_len 1).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Which CUDA graph runs this batch?",
         "Model Runner V2, mode FULL_AND_PIECEWISE (O2)  ·  checks run in order, the first match wins",
         params="capture sizes 1, 2, 4, 8 .. 248 (step 8), 256 .. 512 (step 16)  ·  decode_query_len = 1")

# ---------------------------------------------------------------- panel 1
y0 = 118
CH = 66
GAP = 22
P1_H = 66 + 4 * CH + 3 * GAP + 18
d.panel(y0, P1_H, 1, "What are the checks?",
        "dispatch looks up the candidates for the token count: FULL descriptors first, then PIECEWISE.")
CX, CW = 46, 700
y = y0 + 66
cards = [
    (1, "Forced eager?", VIOLET, "need_eager",
     [("if", "profile run, or encoder-decoder step with encoder inputs"), ("->", "NONE")]),
    (2, "Any graph for this token count?", PINK, "no candidates",
     [("if", "tokens > largest capture size, or nothing captured"), ("->", "NONE (eager)")]),
    (3, "Uniform decode with room?", EMERALD, "FULL",
     [("if", "each request has decode_query_len tokens, no prefill"), ("->", "FULL graph with enough request slots")]),
    (4, "Anything else", BLUE, "PIECEWISE",
     [("->", "PIECEWISE graph of the next capture size"), ("", "attention runs eagerly between the pieces")]),
]
for num, title, col, tag, rows in cards:
    d.step_card(CX, y, CW, CH, num, title, col, tag=tag, rows=rows, key_w=30)
    if num < 4:
        d.arrow(CX + 60, y + CH, CX + 60, y + CH + GAP - 2, INK)
        d.text(CX + 72, y + CH + 15, "no", 11, MUTE, "bold")
    y += CH + GAP
RX = 770
d.box(RX, y0 + 66, 284, 4 * CH + 3 * GAP, "Inputs of the check", [
    "num_reqs, num_tokens",
    "uniform_token_count:",
    "  max_query_len if all equal",
    "  and no real prefill, else None",
    "max_query_len: None if prefill",
    "num_active_loras",
    "DP > 1: ranks agree on",
    "  one padded size",
], col=VIOLET, align="start", mono_size=11.5)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 200
d.panel(y1, P2_H, 2, "Four example batches on one H100",
        "The padded size is the smallest captured size that is not smaller than the batch.")
BY = y1 + 66
rows = [
    ("3 decodes (1 token each)", "uniform 1", "FULL, 4 tokens, 4 requests", EMERALD),
    ("2 decodes + 6-token prompt", "None", "PIECEWISE, 8 tokens", BLUE),
    ("300-token prompt chunk", "None", "PIECEWISE, 304 tokens", BLUE),
    ("8192-token step", "None", "NONE (eager), above 512", PINK),
]
d.text(46, BY + 12, "batch", 12, MUTE, "bold")
d.text(380, BY + 12, "uniform count", 12, MUTE, "bold")
d.text(560, BY + 12, "result", 12, MUTE, "bold")
for i, (b, u, r, col) in enumerate(rows):
    ry = BY + 26 + i * 26
    d.text(46, ry + 14, b, 12, INK, mono=True)
    d.text(380, ry + 14, u, 12, INK, mono=True)
    d.chip(560, ry - 2, 300, r, col, h=24)

y = y1 + P2_H + 24
d.legend(30, y, [(EMERALD, "FULL graph"), (BLUE, "PIECEWISE graphs"), (PINK, "eager"), (VIOLET, "inputs")])
y += 40
y = d.footnotes(30, y, [
    "With k draft tokens, decode_query_len = 1 + k and FULL needs exactly 1 + k tokens per request.",
    "Not shown: LoRA variants, dual-batch overlap, and the varlen decode FULL graphs of adaptive verification.",
])
d.save(height=y + 14)
