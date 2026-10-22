"""B2b tour "One engine step": the example batch, the map of the 12 stops, and the time line of one call.

Used by: src/tours/engine-step.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/core/sched/scheduler.py:713-768        running pass, num_new_tokens = with_spec + placeholders - computed
  vllm/v1/core/sched/async_scheduler.py:25-50    placeholders only for requests that are not prefill chunks
  vllm/v1/worker/gpu/model_runner.py:2471-2483   sort_batch_req_ids: decodes first
  vllm/v1/worker/gpu/model_runner.py:1446-1456   query_start_loc = cumsum(num_scheduled_tokens)
  vllm/v1/worker/gpu/model_runner.py:1401-1408   one logits row per request (no spec decoding)
  vllm/v1/worker/gpu/sample/sampler.py:190-199   chunked-prefill rows sample no token
  vllm/v1/engine/core.py:724-785                 step_with_batch_queue
Example values (two requests, 8192-token budget) are illustrative.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, LINE, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One engine step: two requests, one call of step_with_batch_queue",
         "vllm serve, one GPU (uni executor), async scheduling, batch queue size 2  ·  example values are illustrative",
         params="token budget 8192  ·  B: decoding, 1 token  ·  A: prefill chunk, 8191 tokens of a 20,000-token prompt")

X0 = 46
XS = [46, 410, 774]
BW = 280

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 258
d.panel(y0, P1_H, 1, "What is in the batch of step k?",
        "8192 tokens in one flat batch: request B first (a decode), then request A (a prefill chunk).")
by = y0 + 86
BH = 40
bw_b = 40  # not to scale: B is 1 token
total_w = 1008
d.rect(X0, by, bw_b, BH, fill=BLUE[1], stroke=BLUE[0], rx=3, sw=1.4)
d.text(X0 + bw_b / 2, by + BH / 2 + 4.5, "B", 12, BLUE, "bold", "middle", mono=True)
d.rect(X0 + bw_b, by, total_w - bw_b, BH, fill=VIOLET[1], stroke=VIOLET[0], rx=3, sw=1.4)
d.text(X0 + bw_b + (total_w - bw_b) / 2, by + BH / 2 + 4.5,
       "A: prompt positions 8191 ... 16381  (8191 tokens)", 12, VIOLET, "bold", "middle", mono=True)
d.text(X0, by - 8, "token 0", 11, MUTE, mono=True)
d.text(X0 + total_w, by - 8, "token 8191", 11, MUTE, "normal", "end", mono=True)
rows = [
    "num_scheduled_tokens  {B: 1, A: 8191}         total 8192 = the whole token budget",
    "query_start_loc       [0, 1, 8192]            positions: B [540], A [8191 ... 16381]",
    "seq_lens              [541, 16382]            input ids: B from last_sampled_tokens (GPU)",
    "logits_indices        [0, 8191]               one logits row per request",
    "num_sampled           [1, 0]                  A is a chunk: no token this step",
]
for i, r in enumerate(rows):
    d.text(X0, by + BH + 28 + i * 19, r, 12, INK, mono=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 222
d.panel(y1, P2_H, 2, "Where does each stop run?",
        "All stops run in the engine core process. The GPU only receives queued work (stops 7 and 9).")
QY = y1 + 72
QH = 126
d.box(XS[0], QY, BW, QH, "EngineCore + scheduler", [
    "1  step_with_batch_queue",
    "2  Scheduler.schedule()",
    "3  SchedulerOutput, placeholders",
    "8  get_grammar_bitmask()",
    "10 batch queue: pop step k-1",
    "11 update_from_output(k-1)",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], QY, BW, QH, "Executor, worker, MRV2", [
    "4  execute_model(non_block)",
    "5  update states, CUDA graph?",
    "6  prepare_inputs / prepare_attn",
    "7  forward pass queued on GPU",
    "9  sample_tokens -> AsyncOutput",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], QY, BW, QH, "Output thread", [
    "12 output_queue.get()",
    "   encode EngineCoreOutputs",
    "   PUSH -> front end PULL",
    "",
    "(the front end then detokenizes:",
    " Tour 1, stop 15)",
], col=AMBER, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, QY + 30, XS[1] - 2, QY + 30, VIOLET)
d.arrow(XS[1], QY + 96, XS[0] + BW + 2, QY + 96, EMERALD, dash="5 4")

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 210
d.panel(y2, P3_H, 3, "What does this one call do over time?",
        "It queues step k behind step k-1 on the GPU, then waits only for step k-1.")
T0 = 220
RH = 30
cpu_y = y2 + 92
gpu_y = cpu_y + 50
d.text(X0, cpu_y + 14, "engine core", 13, VIOLET, "bold")
d.text(X0, cpu_y + 28, "main thread", 11.5, MUTE)
d.text(X0, gpu_y + 14, "GPU", 13, EMERALD, "bold")
d.text(X0, gpu_y + 28, "main stream", 11.5, MUTE)
d.line(T0 - 8, cpu_y + RH + 10, 1060, cpu_y + RH + 10, LINE, 1, dash="2 4")


def blk(x0, x1, y, label, col, dashed=False):
    if dashed:
        d.rect(x0 + 1, y + 7, x1 - x0 - 2, RH - 14, fill="#ffffff", stroke=MUTE, rx=3, sw=1, dash="3 3")
        return
    d.rect(x0 + 1, y, x1 - x0 - 2, RH, fill=col[1], stroke=col[0], rx=4, sw=1.3)
    d.text((x0 + x1) / 2, y + RH / 2 + 4.5, label, 11.5, col[0], "bold", "middle", mono=True)


cpu = [(220, 340, "2-3 schedule k", VIOLET), (340, 450, "4-7 launch k", EMERALD),
       (450, 570, "8-9 sample k", EMERALD), (570, 650, None, None),
       (650, 790, "10-11 update k-1", VIOLET), (790, 900, "12 queue out", AMBER)]
for a, b, lab, col in cpu:
    blk(a, b, cpu_y, lab, col, dashed=lab is None)
blk(T0 - 0, 650, gpu_y, "step k-1 (forward + sample)", EMERALD)
blk(650, 1050, gpu_y, "step k (forward 8192 tokens + sample)", EMERALD)
d.text(610, cpu_y - 6, "wait: get_output(k-1)", 11, MUTE, "normal", "middle")

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "request B (decode)"), (VIOLET, "engine core / request A"), (EMERALD, "worker and GPU"),
                 (AMBER, "output")])
y += 40
y = d.footnotes(30, y, [
    "Not to scale: B is 1 token and A is 8191 tokens. Times are illustrative.",
    "The token of B for step k-1 reaches the front end in this call. The token of B for step k comes in the next call.",
])
d.save(height=y + 14)
