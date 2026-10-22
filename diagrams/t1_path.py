"""Tour 1 (Life of a chat request): the map of the 16 stops.

Used by: src/tours/life-of-a-chat-request.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/openai/chat_completion/api_router.py:43-85     stop 1
  vllm/entrypoints/openai/chat_completion/serving.py:302-437      stop 2
  vllm/renderers/base.py:1371-1411, vllm/renderers/hf.py:1213-1220 stop 3
  vllm/v1/engine/async_llm.py:726-802                              stop 4
  vllm/v1/engine/input_processor.py:354-555                        stop 5
  vllm/v1/engine/async_llm.py:567-587, core_client.py:1275-1336    stop 6
  vllm/v1/engine/core.py:1977-2028, 1114-1136                      stop 7
  vllm/v1/engine/core.py:1609-1676                                 stop 8
  vllm/v1/engine/core.py:698-813                                   stop 9
  vllm/v1/core/sched/scheduler.py:622-1643                         stop 10
  vllm/v1/worker/gpu/model_runner.py:1742-1950                     stop 11
  vllm/v1/worker/gpu/model_runner.py:2023-2070                     stop 12
  vllm/v1/worker/gpu/model_runner.py:2114-2311                     stop 13
  vllm/v1/core/sched/scheduler.py:2117-2523, core.py:2030-2096     stop 14
  vllm/v1/engine/async_llm.py:876-946, output_processor.py:728-895 stop 15
  vllm/entrypoints/openai/chat_completion/serving.py:473-508       stop 16
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Life of a chat request: the 16 stops",
         "One streaming POST /v1/chat/completions  ·  read left to right, then down  ·  colour = where the code runs",
         params="stops 1-6 once  ·  stops 9-14 once per engine step  ·  stops 15-16 once per output")

CW, CH = 310, 74
XS = [46, 388, 730]
ROW_GAP = 34


def card(x, y, n, title, col, where, what):
    d.step_card(x, y, CW, CH, n, title, col, rows=[("in", where), ("do", what)], key_w=30)


def row_arrows(y, count, col=INK):
    for i in range(count - 1):
        d.arrow(XS[i] + CW, y + CH / 2, XS[i + 1] - 2, y + CH / 2, col)


def wrap_arrow(y_from, y_to, col=INK):
    # from the bottom of the last card in a row to the top of the first card of the next row
    x_end = XS[2] + CW / 2
    x_start = XS[0] + CW / 2
    mid = y_from + CH + ROW_GAP / 2
    d.path_arrow(f"M {x_end:.1f},{y_from + CH:.1f} L {x_end:.1f},{mid:.1f} "
                 f"L {x_start:.1f},{mid:.1f} L {x_start:.1f},{y_to - 2:.1f}", col)


# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 76 + 2 * CH + ROW_GAP + 18
d.panel(y0, P1_H, 1, "Front end: from HTTP to an EngineCoreRequest",
        "Stops 1-6 run once, in the asyncio event loop of the API server process.")
r1 = y0 + 72
r2 = r1 + CH + ROW_GAP
card(XS[0], r1, 1, "Route handler", BLUE, "api_router.py", "create_chat_completion")
card(XS[1], r1, 2, "Serving class", BLUE, "chat_completion/serving.py", "SamplingParams, request id")
card(XS[2], r1, 3, "Chat template + tokenize", BLUE, "renderers/hf.py, base.py", "messages -> token ids")
row_arrows(r1, 3)
wrap_arrow(r1, r2)
card(XS[0], r2, 4, "The stream starts", BLUE, "async_llm.py: generate()", "first pull runs add_request")
card(XS[1], r2, 5, "EngineCoreRequest", BLUE, "input_processor.py", "process_inputs, unique id")
card(XS[2], r2, 6, "Register and send", BLUE, "async_llm.py, core_client.py", "collector, ZMQ ADD")
row_arrows(r2, 3)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 76 + 3 * CH + 2 * ROW_GAP + 18
d.panel(y1, P2_H, 2, "Engine core process and GPU: one engine step",
        "Stops 7-8 run once. Stops 9-14 repeat in every engine step until the request finishes.")
q1 = y1 + 72
q2 = q1 + CH + ROW_GAP
q3 = q2 + CH + ROW_GAP
card(XS[0], q1, 7, "Input thread", VIOLET, "core.py", "decode -> Request")
card(XS[1], q1, 8, "Busy loop", VIOLET, "core.py: run_busy_loop", "scheduler.add_request")
card(XS[2], q1, 9, "Engine step", VIOLET, "core.py", "step_with_batch_queue")
row_arrows(q1, 3)
wrap_arrow(q1, q2)
card(XS[0], q2, 10, "Schedule", VIOLET, "sched/scheduler.py", "token budget, prefix hit")
card(XS[1], q2, 11, "Prepare inputs", EMERALD, "gpu/model_runner.py", "input ids, attn metadata")
card(XS[2], q2, 12, "Forward pass", EMERALD, "gpu/model_runner.py", "CUDA graph or eager")
row_arrows(q2, 3)
wrap_arrow(q2, q3)
card(XS[0], q3, 13, "Sample", EMERALD, "model_runner.sample_tokens", "token id, async copy")
card(XS[1], q3, 14, "Update and send", VIOLET, "scheduler.update_from_output", "stop check, output thread")
row_arrows(q3, 2)
# loop back: from stop 14 to stop 9 (next engine step)
lx = XS[2] + CW + 22
d.path_arrow(f"M {XS[1] + CW:.1f},{q3 + CH / 2:.1f} L {lx:.1f},{q3 + CH / 2:.1f} "
             f"L {lx:.1f},{q1 + CH / 2:.1f} L {XS[2] + CW + 2:.1f},{q1 + CH / 2:.1f}", VIOLET, dash="5 4")
d.text(XS[2] + 10, q3 + CH / 2 - 12, "next engine step", 11.5, VIOLET, "bold")
d.text(XS[2] + 10, q3 + CH / 2 + 22, "(one per output token)", 11.5, MUTE)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 76 + CH + 18
d.panel(y2, P3_H, 3, "Front end again: from token ids to SSE chunks",
        "Stops 15-16 run once for every EngineCoreOutputs message that has our request.")
s1 = y2 + 72
card(XS[0], s1, 15, "Detokenize", BLUE, "output_processor.py", "ids -> text, collector")
card(XS[1], s1, 16, "SSE chunk", BLUE, "chat_completion/serving.py", "data: {...} / data: [DONE]")
row_arrows(s1, 2)
d.chip(XS[2] + 40, s1 + 22, 230, "HTTP client", INK)
d.arrow(XS[1] + CW, s1 + CH / 2, XS[2] + 38, s1 + CH / 2, INK)

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "front end (API server process)"), (VIOLET, "engine core process (CPU)"),
                 (EMERALD, "GPU work, started by the model runner")])
y += 40
y = d.footnotes(30, y, [
    "Between stop 6 and stop 7 the request crosses ZMQ as msgpack; between stop 14 and stop 15 the outputs cross back.",
    "Simplified: no spec decoding, no structured output, n = 1. Paths are relative to vllm/ (vllm/entrypoints/openai/..., vllm/v1/...).",
])
d.save(height=y + 14)
