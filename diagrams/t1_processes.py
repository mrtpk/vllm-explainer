"""Tour 1 (Life of a chat request): the process picture.

Used by: src/tours/life-of-a-chat-request.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/cli/serve.py:154-156                    one API server runs in the CLI process
  vllm/config/parallel.py:1099-1100                         world_size 1 -> "uni" executor (worker in engine core)
  vllm/v1/executor/uniproc_executor.py:51-69                UniProcExecutor makes the worker in its own process
  vllm/v1/engine/core_client.py:670-682                     front end binds ZMQ ROUTER (in) and PULL (out)
  vllm/v1/engine/utils.py:1148-1151                         ipc:// addresses when all engines are local
  vllm/v1/engine/core_client.py:731-732                     MsgpackEncoder / MsgpackDecoder(EngineCoreOutputs)
  vllm/v1/engine/core.py:1936-1941, 2047-2050               engine DEALER (in) and PUSH (out) sockets
  vllm/v1/engine/core.py:1243-1263                          input and output IO threads
  vllm/v1/engine/core.py:1609-1620                          busy loop (main thread)
  vllm/v1/engine/async_llm.py:876-946                       output_handler task
  vllm/v1/engine/core_client.py:1208-1262                   EngineCoreOutputQueueTask
  vllm/v1/worker/gpu/async_utils.py:137-171                 copy stream for the device-to-host copy
  vllm/entrypoints/serve/utils/api_utils.py:93-109          with_cancellation runs the handler in its own task
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Life of a chat request: the processes",
         "vllm serve on one GPU  ·  one API server  ·  TP = 1, DP = 1  ·  default settings",
         params="front end process  ⇄ ZMQ + msgpack ⇄  engine core process  ⇄ CUDA calls ⇄  GPU")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 300
d.panel(y0, P1_H, 1, "Which processes does the request cross?",
        "Two processes. With one GPU the worker is a plain object inside the engine core process.")

BX_Y = y0 + 74
BX_H = 190
AX, AW = 46, 260
BXX, BW = 420, 260
CX, CW = 794, 260
d.box(AX, BX_Y, AW, BX_H, "Front end (API server process)", [
    "uvicorn + FastAPI, one asyncio loop",
    "OpenAIServingChat",
    "renderer: chat template + tokenizer",
    "AsyncLLM",
    "InputProcessor / OutputProcessor",
    "AsyncMPClient: ROUTER + PULL",
    "stops 1-6, 15-16",
], col=BLUE, align="start", mono_size=11.5)
d.box(BXX, BX_Y, BW, BX_H, "Engine core process", [
    "EngineCoreProc",
    "input thread: DEALER socket",
    "main thread: busy loop",
    "  Scheduler (AsyncScheduler)",
    "  UniProcExecutor -> Worker",
    "  Model Runner V2 (CPU side)",
    "output thread: PUSH socket",
], col=VIOLET, align="start", mono_size=11.5)
d.box(CX, BX_Y, CW, BX_H, "GPU (one device)", [
    "model weights",
    "KV cache blocks",
    "input buffers, CUDA graphs",
    "forward pass + sampler",
    "last_sampled_tokens",
    "copy stream: results to CPU",
    "stops 11-13",
], col=EMERALD, align="start", mono_size=11.5)

# arrows between front end and engine core
ay1 = BX_Y + 60
ay2 = BX_Y + 130
d.arrow(AX + AW, ay1, BXX - 2, ay1, AMBER)
d.arrow_label((AX + AW + BXX) / 2, ay1, "requests", AMBER)
d.text((AX + AW + BXX) / 2, ay1 + 18, "msgpack", 11, AMBER, "bold", "middle", mono=True)
d.arrow(BXX, ay2, AX + AW + 2, ay2, AMBER)
d.arrow_label((AX + AW + BXX) / 2, ay2, "outputs", AMBER)
d.text((AX + AW + BXX) / 2, ay2 + 18, "msgpack", 11, AMBER, "bold", "middle", mono=True)

# arrows between engine core and GPU
d.arrow(BXX + BW, ay1, CX - 2, ay1, INK)
d.arrow_label((BXX + BW + CX) / 2, ay1, "inputs, launches", INK)
d.arrow(CX, ay2, BXX + BW + 2, ay2, EMERALD, dash="5 4")
d.arrow_label((BXX + BW + CX) / 2, ay2, "token ids", EMERALD)
d.text((BXX + BW + CX) / 2, ay2 + 18, "async copy", 11, EMERALD, "bold", "middle")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 236
d.panel(y1, P2_H, 2, "What crosses each boundary, and in which format?",
        "HTTP carries JSON. ZMQ carries msgspec structs as msgpack. The GPU boundary is a Python call.")

CARD_Y = y1 + 70
CARD_H = 146
d.step_card(46, CARD_Y, 330, CARD_H, 1, "Client ⇄ API server", BLUE, tag="HTTP", rows=[
    ("in", "POST /v1/chat/completions"),
    ("", "JSON body, stream: true"),
    ("out", "200, text/event-stream"),
    ("", "data: {chunk JSON}"),
    ("end", "data: [DONE]"),
], key_w=40)
d.step_card(385, CARD_Y, 330, CARD_H, 2, "API server ⇄ engine core", AMBER, tag="ZMQ", rows=[
    ("in", "ADD + EngineCoreRequest"),
    ("out", "EngineCoreOutputs"),
    ("fmt", "msgpack (msgspec structs)"),
    ("sock", "ROUTER→DEALER, PUSH→PULL"),
    ("addr", "ipc:// (one node, DP = 1)"),
], key_w=40)
d.step_card(724, CARD_Y, 330, CARD_H, 3, "Engine core ⇄ GPU", EMERALD, tag="call", rows=[
    ("in", "SchedulerOutput"),
    ("", "-> input buffers (H2D)"),
    ("out", "AsyncOutput"),
    ("", "-> ModelRunnerOutput"),
    ("fmt", "Python objects, tensors"),
], key_w=40)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 214
d.panel(y2, P3_H, 3, "Which threads and tasks do the work?",
        "Socket I/O runs beside the main work, so the busy loop and the event loop do not wait for it.")

T_Y = y2 + 70
T_H = 120
d.box(46, T_Y, 330, T_H, "Front end: asyncio tasks", [
    "handler task: route, render",
    "response task: SSE stream, generate()",
    "output_handler task",
    "EngineCoreOutputQueueTask",
    "thread pool: template, tokenizer",
], col=BLUE, align="start", mono_size=11.5)
d.box(385, T_Y, 330, T_H, "Engine core: three threads", [
    "input thread: recv + decode",
    "main thread: busy loop, step",
    "output thread: encode + send",
    "queues: input_queue, output_queue",
], col=VIOLET, align="start", mono_size=11.5)
d.box(724, T_Y, 330, T_H, "GPU: two CUDA streams", [
    "main stream: forward, sample",
    "copy stream: token ids to CPU",
    "event: copy_event",
], col=EMERALD, align="start", mono_size=11.5)

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "front end"), (VIOLET, "engine core process"), (EMERALD, "GPU"),
                 (AMBER, "ZMQ message")])
y += 40
y = d.footnotes(30, y, [
    "Simplified: one request, one API server, TP = 1. With TP > 1 each GPU has its own worker process (see tours/multi-gpu-step).",
    "Stop numbers refer to the stops of the tour page. Socket types: core_client.py:670-682, core.py:1936-1941 and 2047-2050.",
])
d.save(height=y + 14)
