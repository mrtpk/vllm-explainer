"""B6b: one model graph split into pieces with CUDA graph wrappers, and the two capture kinds.

Used by: src/concepts/compile-and-cuda-graphs.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/models/llama.py:337-346               @support_torch_compile on LlamaModel
  vllm/compilation/wrapper.py:168-175                       torch.compile(fullgraph=True, dynamic=False, backend)
  vllm/compilation/backends.py:1059-1107                    hash factors, cache dir
  vllm/compilation/backends.py:1206-1212                    split_graph at splitting ops
  vllm/config/compilation.py:1184-1213                      default splitting ops
  vllm/compilation/backends.py:765-795                      PiecewiseBackend + wrap_with_cudagraph_if_needed
  vllm/compilation/cuda_graph.py:246-256                    wrapper runs the piece when the mode does not match
  vllm/v1/worker/gpu/cudagraph_utils.py:447-512             capture order PIECEWISE then FULL; FULL uses mode NONE inside
The count of pieces (N + 1 for N layers) is derived, not measured.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, NEUTRAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Compiled pieces between attention ops, each with a CUDA graph wrapper",
         "Llama with 32 layers, default VLLM_COMPILE + FULL_AND_PIECEWISE, Inductor graph partition off",
         params="P = compiled piece  ·  A = attention subgraph (splitting ops, eager)  ·  sizes from Step 7")

BW = 300
XS = [46, 400, 754]

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 250
d.panel(y0, P1_H, 1, "What does the one-time compile produce?",
        "Dynamo traces LlamaModel.forward once; VllmBackend hashes, caches and splits the graph at the attention ops.")
BY = y0 + 70
d.box(XS[0], BY, BW, 84, "@support_torch_compile", [
    "LlamaModel.forward",
    "dim 0 of inputs dynamic",
    "no guards, fullgraph=True",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 84, "Dynamo -> one FX graph", [
    "first call: profile run",
    "symbolic token count",
    "custom ops stay opaque",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 84, "VllmBackend", [
    "hash -> cache dir",
    "split_graph(splitting_ops)",
    "Inductor per piece",
], col=VIOLET, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 42, XS[1] - 2, BY + 42, VIOLET)
d.arrow(XS[1] + BW, BY + 42, XS[2] - 2, BY + 42, VIOLET)

# strip of pieces
SY = BY + 110
labels = ["P0", "A0", "P1", "A1", "P2", "...", "P31", "A31", "P32"]
cw, gap = 84, 10
x = 46 + (1008 - (len(labels) * cw + (len(labels) - 1) * gap)) / 2
for lab in labels:
    if lab == "...":
        d.text(x + cw / 2, SY + 22, "...", 14, MUTE, "bold", "middle")
    elif lab.startswith("P"):
        d.rect(x, SY, cw, 34, fill=BLUE[1], stroke=BLUE[0], rx=4, sw=1.3)
        d.text(x + cw / 2, SY + 22, lab, 12, BLUE[0], "bold", "middle", mono=True)
    else:
        d.rect(x, SY, cw, 34, fill=AMBER[1], stroke=AMBER[0], rx=4, sw=1.3)
        d.text(x + cw / 2, SY + 22, lab, 12, AMBER[0], "bold", "middle", mono=True)
    x += cw + gap

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 214
d.panel(y1, P2_H, 2, "What runs in a PIECEWISE step?",
        "Blue pieces replay their graphs; amber attention subgraphs run Python and the backend kernel.")
BY = y1 + 70
d.box(XS[0], BY, BW, 118, "compiled piece P1", [
    "CUDAGraphWrapper(PIECEWISE)",
    "  -> PiecewiseBackend",
    "  -> Inductor code, range",
    "     [1, 8192]",
    "o_proj, MLP, norms, QKV, RoPE",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 118, "attention subgraph A1", [
    "not compiled, no wrapper",
    "unified_kv_cache_update",
    "unified_attention_with_",
    "  output",
    "any batch shape",
], col=AMBER, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 118, "wrapper's graph table", [
    "key: BatchDescriptor",
    "  (padded tokens, LoRA)",
    "one CUDA graph per",
    "  capture size",
    "mode != PIECEWISE: pass",
], col=BLUE, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 59, XS[1] - 2, BY + 59, BLUE)
d.arrow(XS[1] + BW, BY + 59, XS[2] - 2, BY + 59, AMBER)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 178
d.panel(y2, P3_H, 3, "How are both graph kinds captured at start-up?",
        "PIECEWISE first, then FULL; the largest size first in each mode.")
BY = y2 + 70
d.box(46, BY, 490, 84, "PIECEWISE capture (first)", [
    "forward in context mode PIECEWISE",
    "each piece wrapper records its graph",
    "attention runs between, eagerly",
], col=EMERALD, align="start", mono_size=11.5)
d.box(564, BY, 490, 84, "FULL capture (second)", [
    "torch.cuda.graph(...) around the forward",
    "context mode NONE: wrappers pass through",
    "one graph per uniform decode size",
], col=EMERALD, align="start", mono_size=11.5)

y = y2 + P3_H + 24
d.legend(30, y, [(VIOLET, "one-time compile"), (BLUE, "compiled piece and its graphs"), (AMBER, "attention, eager"),
                 (EMERALD, "capture at start-up")])
y += 40
y = d.footnotes(30, y, [
    "Derived: N layers give N + 1 compiled pieces when the cache write and the attention op are next to each other.",
    "With Inductor graph partition on, Inductor splits instead and no FX-level wrappers are added.",
])
d.save(height=y + 14)
