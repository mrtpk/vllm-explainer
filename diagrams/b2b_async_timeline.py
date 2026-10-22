"""B2b: CPU lane and GPU lane with and without async scheduling.

Used by: src/concepts/async-scheduling.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/engine/core.py:657-687      EngineCore.step: schedule, execute, future.result(), sample, update (no overlap)
  vllm/v1/engine/core.py:724-759      step_with_batch_queue: schedule + start, return while the queue is not full
  vllm/v1/engine/core.py:767-785      pop the oldest step, wait, update_from_output
  vllm/v1/core/sched/async_scheduler.py:45-47, 71-73   placeholders +1 at schedule, -1 when the token arrives
  vllm/v1/worker/gpu/model_runner.py:1491-1504         combine_sampled_and_draft_tokens reads last_sampled_tokens
  vllm/v1/worker/gpu/model_runner.py:1702-1726         postprocess_sampled -> post_update writes last_sampled_tokens
  vllm/v1/worker/gpu/model_runner.py:2209-2218         AsyncOutput starts the copy to the CPU
  vllm/v1/worker/gpu/async_utils.py:137-140, 171-174   copy stream waits for the main stream; get_output synchronizes
Durations are illustrative (not measured): S = 60, F = 150, U = 50 time units.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, EMERALD, INK, LINE, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Async scheduling: the CPU plans step N+1 while the GPU runs step N",
         "One engine core, TP = 1, one decoding request  ·  durations are illustrative, not measured",
         params="S = schedule + start  ·  F = forward + sample on the GPU  ·  U = wait for the token + update")

LX = 46
T0 = 220
SC = 1.0  # px per time unit
RH = 32


def blk(x0, x1, y, label, col, dashed=False):
    x0 = T0 + x0 * SC
    x1 = T0 + x1 * SC
    if dashed:
        d.rect(x0 + 1, y + 7, x1 - x0 - 2, RH - 14, fill="#ffffff", stroke=MUTE, rx=3, sw=1, dash="3 3")
        if label:
            d.text((x0 + x1) / 2, y + RH / 2 + 4, label, 11, MUTE, "normal", "middle", mono=True)
        return
    d.rect(x0 + 1, y, x1 - x0 - 2, RH, fill=col[1], stroke=col[0], rx=4, sw=1.3)
    d.text((x0 + x1) / 2, y + RH / 2 + 4.5, label, 12, col[0], "bold", "middle", mono=True)


def lanes(y, cpu_label, gpu_label):
    cpu_y = y
    gpu_y = y + 52
    d.text(LX, cpu_y + 15, "engine core", 13, VIOLET, "bold")
    d.text(LX, cpu_y + 30, cpu_label, 11.5, MUTE)
    d.text(LX, gpu_y + 15, "GPU", 13, EMERALD, "bold")
    d.text(LX, gpu_y + 30, gpu_label, 11.5, MUTE)
    d.line(T0 - 8, cpu_y + RH + 10, 1060, cpu_y + RH + 10, LINE, 1, dash="2 4")
    return cpu_y, gpu_y


def token_marks(cpu_y, xs, labels):
    for x, lab in zip(xs, labels):
        cx = T0 + x * SC
        d.arrow(cx, cpu_y, cx, cpu_y - 14, AMBER)
        d.text(cx, cpu_y - 18, lab, 11, AMBER, "bold", "middle", mono=True)


# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 214
d.panel(y0, P1_H, 1, "Without async scheduling: EngineCore.step()",
        "Each step waits for its own result. The GPU is idle while the CPU updates and schedules.")
cpu_y, gpu_y = lanes(y0 + 96, "main thread", "main stream")
seq = [(0, 60, "S1"), (60, 210, None), (210, 260, "U1"),
       (260, 320, "S2"), (320, 470, None), (470, 520, "U2"),
       (520, 580, "S3"), (580, 730, None), (730, 780, "U3")]
for a, b, lab in seq:
    blk(a, b, cpu_y, lab, VIOLET, dashed=lab is None)
for a, b, lab in [(60, 210, "F1"), (320, 470, "F2"), (580, 730, "F3")]:
    blk(a, b, gpu_y, lab, EMERALD)
for a, b in [(0, 60), (210, 320), (470, 580)]:
    blk(a, b, gpu_y, "idle", EMERALD, dashed=True)
token_marks(cpu_y, [260, 520, 780], ["tok 1", "tok 2", "tok 3"])

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 214
d.panel(y1, P2_H, 2, "With async scheduling: step_with_batch_queue()",
        "Step N+1 is scheduled and queued on the GPU before the CPU waits for step N. The GPU does not stop.")
cpu_y, gpu_y = lanes(y1 + 96, "main thread", "main stream")
seq = [(0, 60, "S1"), (60, 120, "S2"), (120, 210, None), (210, 260, "U1"),
       (260, 320, "S3"), (320, 360, None), (360, 410, "U2"),
       (410, 470, "S4"), (470, 510, None), (510, 560, "U3"),
       (560, 620, "S5"), (620, 660, None), (660, 710, "U4")]
for a, b, lab in seq:
    blk(a, b, cpu_y, lab, VIOLET, dashed=lab is None)
for a, b, lab in [(60, 210, "F1"), (210, 360, "F2"), (360, 510, "F3"), (510, 660, "F4"), (660, 810, "F5")]:
    blk(a, b, gpu_y, lab, EMERALD)
token_marks(cpu_y, [260, 410, 560, 710], ["tok 1", "tok 2", "tok 3", "tok 4"])

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 214
d.panel(y2, P3_H, 3, "Where does step N+1 get its input token?",
        "From a GPU tensor. The CPU copy of the token arrives later and only updates the scheduler.")
BY = y2 + 72
BH = 112
XS = [46, 410, 774]
BW = 280
d.box(XS[0], BY, BW, BH, "sample_tokens (step N)", [
    "sampler -> sampled_token_ids",
    "postprocess_sampled()",
    "  post_update kernel writes",
    "  last_sampled_tokens[req]",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, BH, "execute_model (step N+1)", [
    "prepare_inputs()",
    "combine_sampled_and_draft_",
    "  tokens() reads",
    "  last_sampled_tokens[req]",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, BH, "CPU, later (U N)", [
    "AsyncOutput: copy stream",
    "get_output(): event.synchronize()",
    "update_from_output():",
    "  append token, placeholders -1",
], col=VIOLET, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 40, XS[1] - 2, BY + 40, EMERALD, sw=2)
d.text((XS[0] + BW + XS[1]) / 2, BY + 30, "GPU", 11.5, EMERALD, "bold", "middle")
d.arrow(XS[1] + BW, BY + 80, XS[2] - 2, BY + 80, AMBER, dash="5 4")
d.text((XS[1] + BW + XS[2]) / 2, BY + 70, "D2H", 11.5, AMBER, "bold", "middle")

y = y2 + P3_H + 24
d.legend(30, y, [(VIOLET, "engine core CPU work"), (EMERALD, "GPU work"), (AMBER, "token reaches the CPU")])
y += 40
y = d.footnotes(30, y, [
    "Dashed CPU blocks: the main thread waits for the GPU (future.result()). Dashed GPU blocks: the GPU has no work.",
    "Simplified: one request, equal step times. Real step times change with the batch. The D2H copy runs on a separate copy stream.",
])
d.save(height=y + 14)
