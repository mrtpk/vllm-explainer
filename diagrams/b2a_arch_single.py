"""Architecture overview (a): vllm serve on one GPU, processes, owners and threads.

Used by: src/concepts/architecture-overview.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/cli/serve.py:153-156             one API server runs in the CLI process (uvloop)
  vllm/v1/engine/async_llm.py:167-213               AsyncLLM owns renderer, InputProcessor, OutputProcessor, client, loggers
  vllm/v1/engine/core_client.py:158-173             DP size 1 -> AsyncMPClient
  vllm/v1/engine/core_client.py:670-682             front end binds ROUTER (in) and PULL (out)
  vllm/v1/engine/utils.py:1137-1151                 ipc:// when all engines are local
  vllm/v1/engine/core.py:147, 161, 177-185          EngineCore owns executor, StructuredOutputManager, scheduler
  vllm/v1/core/sched/scheduler.py:329               scheduler owns the KV cache manager
  vllm/config/parallel.py:1055, 1099-1100           world_size_across_dp == 1 -> "uni"
  vllm/v1/executor/uniproc_executor.py:51-75        UniProcExecutor builds the worker in its own process
  vllm/v1/engine/core.py:1243-1265                  input and output threads
  vllm/v1/engine/core.py:1609-1620                  busy loop in the main thread
  vllm/v1/engine/async_llm.py:946                   output_handler task
  vllm/v1/engine/core_client.py:1260-1262           EngineCoreOutputQueueTask
  vllm/v1/engine/core_client.py:843-845             MPClientEngineMonitor thread
  vllm/renderers/base.py:143-149                    renderer thread pools
  vllm/entrypoints/launchers/launcher.py:233-237    watchdog task
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("vllm serve on one GPU: processes, owners and threads",
         "TP = 1, PP = 1, DP = 1  ·  one API server  ·  default settings",
         params="API server process  ⇄ ZMQ + msgpack ⇄  engine core process  ⇄ CUDA calls ⇄  GPU")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 336
d.panel(y0, P1_H, 1, "Which processes run, and which objects does each own?",
        "Two processes. The worker is a plain object in the engine core process (uni executor).")

BY = y0 + 72
BH = 236
AX, AW = 46, 286
BX, BW = 466, 290
CX, CW = 862, 192
d.box(AX, BY, AW, BH, "API server process (front end)", [
    "CLI process: uvloop.run(run_server)",
    "uvicorn + FastAPI app",
    "OpenAIServingChat (and others)",
    "AsyncLLM  (the EngineClient)",
    "  renderer: template, tokenizer",
    "  InputProcessor",
    "  OutputProcessor",
    "    RequestState per request",
    "  AsyncMPClient",
    "    ROUTER socket (requests out)",
    "    PULL socket (outputs in)",
    "  StatLoggerManager",
], col=BLUE, align="start")
d.box(BX, BY, BW, BH, "Engine core process", [
    "EngineCoreProc  (title EngineCore)",
    "  DEALER socket  (input thread)",
    "  PUSH socket    (output thread)",
    "  busy loop      (main thread)",
    "Scheduler  (AsyncScheduler)",
    "  KVCacheManager",
    "StructuredOutputManager",
    "UniProcExecutor",
    "  Worker  (rank 0, same process)",
    "    Model Runner V2",
], col=VIOLET, align="start")
d.box(CX, BY, CW, BH, "GPU 0", [
    "model weights",
    "KV cache blocks",
    "CUDA graphs",
    "forward pass, sampler",
    "copy stream:",
    "  tokens to CPU",
], col=EMERALD, align="start")

ay1 = BY + 70
ay2 = BY + 150
mid1 = (AX + AW + BX) / 2
d.arrow(AX + AW, ay1, BX - 2, ay1, AMBER)
d.arrow_label(mid1, ay1, "EngineCoreRequest", AMBER)
d.text(mid1, ay1 + 18, "ipc:// msgpack", 11, AMBER, "bold", "middle", mono=True)
d.arrow(BX, ay2, AX + AW + 2, ay2, AMBER)
d.arrow_label(mid1, ay2, "EngineCoreOutputs", AMBER)
d.text(mid1, ay2 + 18, "ipc:// msgpack", 11, AMBER, "bold", "middle", mono=True)
mid2 = (BX + BW + CX) / 2
d.arrow(BX + BW, ay1, CX - 2, ay1, INK)
d.arrow_label(mid2, ay1, "launches", INK)
d.text(mid2, ay1 + 18, "plain call", 11, INK, "bold", "middle")
d.arrow(CX, ay2, BX + BW + 2, ay2, EMERALD, dash="5 4")
d.arrow_label(mid2, ay2, "token ids", EMERALD)
d.text(mid2, ay2 + 18, "async copy", 11, EMERALD, "bold", "middle")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 250
d.panel(y1, P2_H, 2, "Which threads and asyncio tasks do the work?",
        "One event loop in the front end. Three threads in the engine core: I/O beside the busy loop.")

TY = y1 + 70
TH = 156
d.box(AX, TY, AW, TH, "Front end: event loop and helpers", [
    "task per HTTP request (handler)",
    "task: StreamingResponse body",
    "task: output_handler",
    "task: EngineCoreOutputQueueTask",
    "task: watchdog_loop (every 5 s)",
    "thread pool: template, tokenizer",
    "thread: MPClientEngineMonitor",
], col=BLUE, align="start")
d.box(BX, TY, BW, TH, "Engine core: three threads", [
    "input thread",
    "  process_input_sockets()",
    "main thread",
    "  run_busy_loop()",
    "output thread",
    "  process_output_sockets()",
    "queues: input_queue, output_queue",
], col=VIOLET, align="start")
d.box(CX, TY, CW, TH, "GPU: two CUDA streams", [
    "main stream:",
    "  forward pass, sampler",
    "copy stream:",
    "  token ids to CPU",
    "event: copy_event",
], col=EMERALD, align="start")

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "API server process"), (VIOLET, "engine core process"), (EMERALD, "GPU"),
                 (AMBER, "ZMQ message")])
y += 40
y = d.footnotes(30, y, [
    "Not shown: libzmq I/O threads, the multimodal thread pool, the structured-output thread pools of the engine core.",
    "With TP > 1, PP > 1 or DP > 1 the executor is MultiprocExecutor and each GPU gets a worker process (second diagram).",
])
d.save(height=y + 14)
