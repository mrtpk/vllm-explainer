"""B6b: one Model Runner V2 step over time: execute_model -> sample_tokens -> get_output.

Used by: src/concepts/model-runner.html, src/code/v1-worker-gpu-model-runner.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/worker/gpu/model_runner.py:1753-1760   execute_model: update the request states
  vllm/v1/worker/gpu/model_runner.py:1806-1819   dispatch_cg_and_sync_dp
  vllm/v1/worker/gpu/model_runner.py:2024-2070   forward: FULL replay, PIECEWISE or eager
  vllm/v1/worker/gpu/model_runner.py:2090-2110   store ExecuteModelState, return None
  vllm/v1/engine/core.py:737-745                  engine core: grammar bitmask, then sample_tokens
  vllm/v1/worker/gpu/model_runner.py:1637-1677   logits, bitmask, sampler / rejection sampler
  vllm/v1/worker/gpu/model_runner.py:2210-2218   AsyncOutput starts the copy on output_copy_stream
  vllm/v1/worker/gpu/async_utils.py:137-140, 171 copy stream waits for the main stream; copy_event.record
  vllm/v1/worker/gpu/model_runner.py:2244-2289   postprocess_sampled (post_update), speculator.propose
  vllm/v1/worker/gpu/async_utils.py:173-184      get_output: copy_event.synchronize(), tolist, trim
  vllm/v1/executor/uniproc_executor.py:38-48, 108-111   uni: AsyncOutputFuture.result() -> get_output()
  vllm/v1/engine/core.py:767-773                  engine core pops the step and calls future.result()
  vllm/v1/executor/multiproc_executor.py:690-699, 1000-1039   mp: WorkerAsyncOutputCopy thread
Durations are illustrative, not measured.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, EMERALD, INK, LINE, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One MRV2 step: the CPU queues work, and waits only in get_output()",
         "execute_model and sample_tokens return at once  ·  durations are illustrative, not measured",
         params="time runs left to right  ·  the main stream runs kernels in queue order  ·  the copy stream overlaps")

LX = 46
T0 = 236
SC = 1.0
RH = 34


def blk(x0, x1, y, label, col, dashed=False):
    a = T0 + x0 * SC
    b = T0 + x1 * SC
    if dashed:
        d.rect(a + 1, y + 6, b - a - 2, RH - 12, fill="#ffffff", stroke=MUTE, rx=3, sw=1, dash="3 3")
        if label:
            d.text((a + b) / 2, y + RH / 2 + 4, label, 11, MUTE, "normal", "middle", mono=True)
        return
    d.rect(a + 1, y, b - a - 2, RH, fill=col[1], stroke=col[0], rx=4, sw=1.3)
    d.text((a + b) / 2, y + RH / 2 + 4.5, label, 11.5, col[0], "bold", "middle", mono=True)


# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 300
d.panel(y0, P1_H, 1, "What runs where, in one step?",
        "Three lanes: the thread that calls the runner, the GPU main stream, and the runner's copy stream.")
lanes = [("caller thread", "engine core or worker", VIOLET),
         ("GPU main stream", "kernels in order", EMERALD),
         ("GPU copy stream", "output_copy_stream", AMBER)]
ys = []
for i, (name, sub, col) in enumerate(lanes):
    ly = y0 + 78 + i * 62
    ys.append(ly)
    d.text(LX, ly + 15, name, 13, col[0], "bold")
    d.text(LX, ly + 31, sub, 11.5, MUTE)
    d.line(T0 - 8, ly + RH + 13, 1060, ly + RH + 13, LINE, 1, dash="2 4")

cy, gy, sy = ys
blk(0, 170, cy, "execute_model", VIOLET)
blk(170, 230, cy, "bitmask", VIOLET, dashed=True)
blk(230, 400, cy, "sample_tokens", VIOLET)
blk(400, 700, cy, "free: next step", VIOLET, dashed=True)
blk(700, 815, cy, "get_output()", VIOLET)

blk(150, 450, gy, "forward pass", EMERALD)
blk(450, 520, gy, "sampler", EMERALD)
blk(520, 610, gy, "post_update", EMERALD)
blk(610, 690, gy, "drafter", EMERALD)

blk(150, 520, sy, "waits for main stream", AMBER, dashed=True)
blk(520, 600, sy, "D2H copy", AMBER)

# copy_event marker and the wait arrow (routed right of the GPU blocks)
ex = T0 + 600 * SC
ax = T0 + 708 * SC
d.circle(ex, sy + RH / 2, 5, AMBER[0])
d.text(ex - 70, sy + RH + 30, "copy_event.record", 11, AMBER[0], "bold", mono=True)
d.path_arrow(f"M {ex + 5:.1f},{sy + RH / 2:.1f} L {ax:.1f},{sy + RH / 2:.1f} L {ax:.1f},{cy + RH + 2:.1f}",
             AMBER, dash="5 4")
d.text(ax + 8, gy + RH / 2 + 4, "synchronize()", 11, AMBER[0], "bold", mono=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 196
d.panel(y1, P2_H, 2, "Who calls get_output()?",
        "It depends on the executor; in both cases the engine core gets plain Python lists.")
BY = y1 + 68
BW = 490
d.box(46, BY, BW, 104, "uni executor (TP = PP = DP = 1)", [
    "sample_tokens -> AsyncOutputFuture (no wait)",
    "engine core main thread, later:",
    "  batch_queue.pop() -> future.result()",
    "  -> AsyncOutput.get_output()",
], col=VIOLET, align="start", mono_size=11.5)
d.box(564, BY, BW, 104, "multiproc + async scheduling (output rank)", [
    "worker main thread: handle_output()",
    "  -> async_output_queue (no wait)",
    "thread WorkerAsyncOutputCopy:",
    "  get_output() -> reply MessageQueue",
], col=VIOLET, align="start", mono_size=11.5)

y = y1 + P2_H + 24
d.legend(30, y, [(VIOLET, "CPU work"), (EMERALD, "GPU kernels"), (AMBER, "copy to the CPU and its event")])
y += 40
y = d.footnotes(30, y, [
    "Dashed blocks: no work of this step (the engine core computes the bitmask; the caller is free for the next step).",
    "The drafter runs only with speculative decoding. With async scheduling, step N+1 is queued before get_output() of step N.",
])
d.save(height=y + 14)
