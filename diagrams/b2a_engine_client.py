"""Engine client: what AsyncLLM owns, and how one output reaches generate().

Used by: src/concepts/engine-client.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/engine/async_llm.py:167-213        renderer, InputProcessor, OutputProcessor, engine core client, loggers
  vllm/v1/engine/async_llm.py:521-536        assign_request_id, start output_handler, make RequestOutputCollector
  vllm/v1/engine/async_llm.py:575-584        register in OutputProcessor, then add_request_async
  vllm/v1/engine/core_client.py:1275-1294    ADD frames on the ROUTER socket
  vllm/v1/engine/core_client.py:1208-1262    EngineCoreOutputQueueTask: PULL -> decode -> outputs_queue
  vllm/v1/engine/async_llm.py:876-946        output_handler: chunks, process_outputs, aborts, stats
  vllm/v1/engine/output_processor.py:53-103  RequestOutputCollector put / get / get_nowait
  vllm/v1/engine/async_llm.py:771-802        generate(): add_request, then loop on the collector
  vllm/envs.py:1474-1476                     VLLM_V1_OUTPUT_PROC_CHUNK_SIZE = 128
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, INK, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("AsyncLLM: the engine client of the API server",
         "one API server  ·  DP = 1  ·  streaming request  ·  default settings",
         params="generate() → add_request() → ZMQ → … → output_handler → RequestOutputCollector → generate()")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 262
d.panel(y0, P1_H, 1, "What does AsyncLLM own?",
        "A renderer and two processors in this process, and a client for the engine core process.")
BY = y0 + 72
BH = 166
d.box(46, BY, 330, BH, "Input side", [
    "renderer  (shared with serving)",
    "  chat template, tokenizer",
    "InputProcessor",
    "  check params, fill defaults",
    "  → EngineCoreRequest",
    "  assign_request_id(): +8 chars",
], col=BLUE, align="start")
d.box(385, BY, 330, BH, "Engine core client", [
    "AsyncMPClient  (DP = 1)",
    "  ROUTER socket: requests out",
    "  PULL socket: outputs in",
    "  outputs_queue (asyncio.Queue)",
    "  utility_results: call_id → future",
    "  resources.engine_dead flag",
], col=AMBER, align="start")
d.box(724, BY, 330, BH, "Output side", [
    "OutputProcessor",
    "  request_states: id → RequestState",
    "    detokenizer, logprobs,",
    "    collector (queue), stats",
    "  external id → internal ids",
    "StatLoggerManager (log_stats)",
], col=VIOLET, align="start")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 352
d.panel(y1, P2_H, 2, "How does one output reach the generate() task of its request?",
        "Two background tasks move outputs; one collector per request hands them over.")

R1 = y1 + 74
RH = 96
d.box(46, R1, 300, RH, "PULL socket", [
    "frames from the engine core",
    "msgpack bytes",
    "one EngineCoreOutputs per step",
], col=AMBER, align="start")
d.box(400, R1, 300, RH, "EngineCoreOutputQueueTask", [
    "decode → EngineCoreOutputs",
    "utility reply → set the future",
    "else → outputs_queue",
], col=AMBER, align="start")
d.box(754, R1, 300, RH, "output_handler task", [
    "await get_output_async()",
    "slices of 128 outputs",
    "process_outputs(), stats",
], col=VIOLET, align="start")
d.arrow(346, R1 + 48, 398, R1 + 48, AMBER)
d.arrow(700, R1 + 48, 752, R1 + 48, AMBER)

R2 = R1 + RH + 50
d.box(754, R2, 300, RH, "OutputProcessor.process_outputs", [
    "detokenize, stop strings",
    "make RequestOutput",
    "queue.put(output)",
], col=VIOLET, align="start")
d.box(400, R2, 300, RH, "RequestOutputCollector", [
    "one per request, one slot",
    "DELTA: merge if slow reader",
    "asyncio.Event wakes the reader",
], col=TEAL, align="start")
d.box(46, R2, 300, RH, "generate() task", [
    "q.get_nowait() or await q.get()",
    "yield RequestOutput",
    "stop when finished = True",
], col=BLUE, align="start")
d.arrow(904, R1 + RH, 904, R2 - 2, VIOLET)
d.arrow(754, R2 + 48, 702, R2 + 48, VIOLET)
d.arrow(400, R2 + 48, 348, R2 + 48, TEAL)
d.arrow_label(727, R2 + 48, "put", VIOLET)
d.arrow_label(373, R2 + 48, "get", TEAL)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "request task"), (AMBER, "engine core client"), (VIOLET, "output side"),
                 (TEAL, "per-request collector")])
y += 40
y = d.footnotes(30, y, [
    "All boxes in panel 2 run on one asyncio event loop in the API server process; there is no extra thread.",
    "With DP > 1 the client is DPLBAsyncMPClient or DPAsyncMPClient (concepts/data-parallel-serving).",
])
d.save(height=y + 14)
