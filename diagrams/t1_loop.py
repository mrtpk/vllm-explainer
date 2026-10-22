"""Tour 1 (Life of a chat request): the per-token loop and the async-scheduling overlap.

Used by: src/tours/life-of-a-chat-request.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/engine/core.py:723-759        schedule + execute_model + sample_tokens, then return if the queue is not full
  vllm/v1/engine/core.py:767-783        pop the oldest step, wait for its result, update_from_output
  vllm/config/vllm.py:587-597           max_concurrent_batches = pp_size + 1 = 2 with async scheduling and MRV2
  vllm/v1/core/sched/scheduler.py:759-763   num_new_tokens = num_tokens_with_spec + placeholders - computed
  vllm/v1/core/sched/scheduler.py:1191, 1439, 1703-1716   prefill count, computed tokens after a prefix hit
  vllm/v1/core/sched/async_scheduler.py:25-55, 57-80      placeholders +1 at schedule, -1 at update
  vllm/v1/core/kv_cache_manager.py:289-300                cache hit is at most num_tokens - 1, full blocks only
  vllm/v1/core/sched/utils.py:98-119                      check_stop: EOS, stop ids, max_model_len, max_tokens
Values (61 prompt tokens, 48 cached tokens, block size 16) are illustrative.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, LINE, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Life of a chat request: one engine step per output token",
         "Illustrative request: 61 prompt tokens, 48 of them in the prefix cache (block size 16)  ·  no spec decoding",
         params="async scheduling on (default)  ·  batch queue size 2  ·  stream_interval 1")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 250
d.panel(y0, P1_H, 1, "How many engine steps does the request need?",
        "One step for the prompt, then one step per new token. Each step sends one SSE chunk.")
BY = y0 + 72
BH = 150
XS = [46, 388, 730]
W = 310
d.box(XS[0], BY, W, BH, "Step 1: the prompt (prefill)", [
    "prompt: 61 tokens",
    "prefix cache hit: 48 (3 blocks)",
    "scheduled: 13 tokens",
    "samples: token 1",
    "SSE: role chunk + first text",
], col=BLUE, align="start", mono_size=12)
d.box(XS[1], BY, W, BH, "Steps 2 ... N-1: one token each", [
    "scheduled: 1 token",
    "input: the last sampled token",
    "samples: the next token",
    "SSE: one chunk per step",
    "KV cache: one more slot",
], col=VIOLET, align="start", mono_size=12)
d.box(XS[2], BY, W, BH, "Step N: the last token", [
    "samples: EOS, or max_tokens",
    "check_stop -> FINISHED_*",
    "blocks go back to the pool",
    "SSE: chunk with finish_reason",
    "then: data: [DONE]",
], col=EMERALD, align="start", mono_size=12)
d.arrow(XS[0] + W, BY + BH / 2, XS[1] - 2, BY + BH / 2, INK)
d.arrow(XS[1] + W, BY + BH / 2, XS[2] - 2, BY + BH / 2, INK)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 276
d.panel(y1, P2_H, 2, "What overlaps when async scheduling is on?",
        "The engine core schedules and starts step k+1 before it waits for the result of step k.")
LX = 46
T0 = 250          # x of time 0
cpu_y = y1 + 102
gpu_y = cpu_y + 62
RH = 34
d.text(LX, cpu_y + 22, "engine core", 13, VIOLET, "bold")
d.text(LX, cpu_y + 38, "main thread (CPU)", 11.5, MUTE)
d.text(LX, gpu_y + 22, "GPU", 13, EMERALD, "bold")
d.text(LX, gpu_y + 38, "main stream", 11.5, MUTE)
d.line(T0 - 10, cpu_y + RH + 14, 1060, cpu_y + RH + 14, LINE, 1, dash="2 4")


def blk(x0, x1, y, label, col, dashed=False):
    if dashed:
        d.rect(x0, y + 8, x1 - x0, RH - 16, fill="#ffffff", stroke=MUTE, rx=3, sw=1, dash="3 3")
        return
    d.rect(x0, y, x1 - x0, RH, fill=col[1], stroke=col[0], rx=4, sw=1.3)
    d.text((x0 + x1) / 2, y + RH / 2 + 4.5, label, 12, col[0], "bold", "middle", mono=True)


# CPU: S = schedule + launch, U = wait for result + update_from_output
cpu = [(250, 330, "S1"), (338, 418, "S2"), (418, 470, None), (470, 530, "U1"),
       (538, 618, "S3"), (618, 630, None), (630, 690, "U2"),
       (698, 778, "S4"), (778, 790, None), (790, 850, "U3"),
       (858, 938, "S5"), (938, 950, None), (950, 1010, "U4")]
for x0, x1, lab in cpu:
    blk(x0, x1, cpu_y, lab, VIOLET, dashed=lab is None)
gpu = [(330, 470, "F1"), (470, 630, "F2"), (630, 790, "F3"), (790, 950, "F4")]
for x0, x1, lab in gpu:
    blk(x0 + 1, x1 - 1, gpu_y, lab, EMERALD)
# token outputs leave after each U
for x0, x1, lab in cpu:
    if lab and lab.startswith("U"):
        cx = (x0 + x1) / 2
        d.arrow(cx, cpu_y, cx, cpu_y - 18, AMBER)
d.text(470, cpu_y - 24, "token k goes to the output thread", 11.5, AMBER, "bold")
d.aside(LX, gpu_y + RH + 30, [
    "S k = schedule step k and start it on the GPU.  F k = forward pass + sampling of step k.",
    "U k = wait for the token of step k, then update_from_output.  Dashed = the CPU waits for the GPU.",
])

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 236
d.panel(y2, P3_H, 3, "How can the scheduler plan step k+1 without token k?",
        "It counts a placeholder for each token in flight. The model runner reads the real token on the GPU.")
ty = y2 + 84
COLS = "event                        computed     placeholders  num_tokens  new tokens"
rows = [
    "schedule step 1 (prefill)    48 -> 61      0 -> 1        61          61 - 48 = 13",
    "schedule step 2              61 -> 62      1 -> 2        61          61 + 1 - 61 = 1",
    "update step 1 (token 1)      62            2 -> 1        62          -",
    "schedule step 3              62 -> 63      1 -> 2        62          62 + 1 - 62 = 1",
    "update step 2 (token 2)      63            2 -> 1        63          -",
]
d.text(LX, ty, COLS, 12, MUTE, "bold", mono=True)
d.line(LX, ty + 8, 1050, ty + 8, LINE, 1)
for i, r in enumerate(rows):
    d.text(LX, ty + 28 + i * 21, r, 12.5, INK, mono=True)
d.aside(LX, ty + 28 + len(rows) * 21 + 6, [
    "new tokens = num_tokens + placeholders - computed (scheduler.py:759-763). Placeholders: async_scheduler.py:45-47 and 71-73.",
])

y = y2 + P3_H + 24
d.legend(30, y, [(VIOLET, "engine core CPU work"), (EMERALD, "GPU work"), (AMBER, "output to the front end")])
y += 40
y = d.footnotes(30, y, [
    "Simplified: one request alone. In a real batch many requests share each step. Durations are not to scale.",
    "With --no-async-scheduling the engine uses EngineCore.step(): schedule, run, wait, update, with no overlap (core.py:657-687).",
])
d.save(height=y + 14)
