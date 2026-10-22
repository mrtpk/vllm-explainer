"""Multi-LoRA serving: one batch with three adapters on one base model.

Used by: src/learn/lora-adapters.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/worker/gpu/lora_utils.py:94-104        token_lora_mapping = lora id of each request, repeated per scheduled token
  vllm/lora/layers/base_linear.py:133-143        lora_a_stacked: [max_loras, 1, max_lora_rank, input_size] per slice
  vllm/lora/model_manager.py:343-349             GPU slots = max_loras, CPU cache capacity = max_cpu_loras
  vllm/v1/core/sched/scheduler.py:996-1008       scheduler admits at most max_loras distinct adapters per batch
  vllm/lora/peft_helper.py:117-120               scaling = lora_alpha / r (or / sqrt(r) for rsLoRA)
Example: 5 requests, 7 scheduled tokens, max_loras = 3, rank 16, q_proj of a Llama-3.1-8B-shaped model (4096 -> 4096).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, NEUTRAL, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Multi-LoRA serving: one batch, three adapters, one base model",
         "max_loras = 3 GPU slots  ·  rank 16  ·  q_proj 4096 -> 4096  ·  5 requests, 7 tokens in this step",
         params="y = x W  +  scaling * (x A[slot]) B[slot]      slot = the adapter of the token's request")

AD = {"sql": BLUE, "chat-de": PINK, "legal": EMERALD, "base": NEUTRAL}

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 250
d.panel(y0, P1_H, 1, "Each token carries the LoRA id of its request",
        "The model runner repeats each request's LoRA id once per scheduled token: that is the token-to-adapter map.")
ry = y0 + 70
reqs = [("R1", "sql", 3, "prefill chunk"), ("R2", "chat-de", 1, "decode"), ("R3", "base", 1, "decode"),
        ("R4", "sql", 1, "decode"), ("R5", "legal", 1, "decode")]
IDS = {"sql": 1, "chat-de": 2, "legal": 3, "base": 0}
CW, CG = 112, 10
x = 60
tok = 0
for name, ad, n, kind in reqs:
    col = AD[ad]
    w = n * CW + (n - 1) * CG
    lc = col[0] if ad != "base" else MUTE
    d.text(x + w / 2, ry + 4, name, 12, lc, "bold", "middle")
    d.text(x + w / 2, ry + 20, f'model="{ad}"' if ad != "base" else "base model name", 11.5, lc, anchor="middle", mono=True)
    for i in range(n):
        cx = x + i * (CW + CG)
        d.rect(cx, ry + 30, CW, 46, fill=col[1], stroke=col[0], rx=4, sw=1.4)
        d.text(cx + CW / 2, ry + 50, f"token {tok}", 12, INK, anchor="middle", mono=True)
        d.text(cx + CW / 2, ry + 67, f"lora id {IDS[ad]}", 11.5, lc, "bold", "middle", mono=True)
        tok += 1
    d.text(x + w / 2, ry + 94, kind, 11.5, MUTE, anchor="middle", italic=True)
    x += w + CG + 20
d.text(60, ry + 130, "token_lora_mapping = (1, 1, 1, 2, 0, 1, 3)    0 = no LoRA    3 distinct adapters = max_loras",
       12, INK, mono=True)
d.text(60, ry + 152, "A 4th adapter would wait: the scheduler admits a new adapter only while fewer than max_loras are in the batch.",
       11.5, MUTE, italic=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 214
d.panel(y1, P2_H, 2, "One base GEMM for all tokens, then a small LoRA pass per token",
        "The base weight is read once for the whole batch. The LoRA kernels map each lora id to a GPU slot and pick its A and B.")
by = y1 + 70
BW = 324
xs = (46, 388, 730)
d.box(xs[0], by, BW, 92, "Base: Y = X W", ["X [7, 4096]  W [4096, 4096]", "one GEMM for all 7 tokens", "same for every adapter"],
      col=INK)
d.box(xs[1], by, BW, 92, "Shrink: S = X A[slot]", ["A stacked [3, 1, 16, 4096]", "S [7, 16]", "token 4 (base) skips it"],
      col=VIOLET)
d.box(xs[2], by, BW, 92, "Expand and add: Y += S B[slot]", ["B stacked [3, 1, 4096, 16]", "times scaling = alpha / r",
                                                               "Y [7, 4096]"], col=VIOLET)
d.arrow(xs[0] + BW, by + 46, xs[1] - 2, by + 46, INK)
d.arrow(xs[1] + BW, by + 46, xs[2] - 2, by + 46, VIOLET)
d.text(46, by + 118, "GPU slots (max_loras = 3): sql, chat-de, legal.   CPU cache (max_cpu_loras, default = max_loras): adapters waiting to move to the GPU.",
       11.5, MUTE)

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "adapter sql"), (PINK, "adapter chat-de"), (EMERALD, "adapter legal"), (NEUTRAL, "no adapter"),
                 (VIOLET, "LoRA kernels")])
y += 40
y = d.footnotes(30, y, [
    "Shapes are for one q_proj on one GPU with TP = 1. vLLM runs the same pattern for every LoRA-capable linear layer.",
    "Simplified: vLLM runs shrink and expand as Punica-style batched kernels (vllm/lora/punica_wrapper/); their insides are not shown.",
])
d.save(height=y + 14)
