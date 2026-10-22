"""What vLLM is: two entry points into one engine core.

Used by: src/learn/what-is-vllm.html
Sources (at the pinned commit in explainer.json):
  pyproject.toml:42-43                               the `vllm` console script -> vllm.entrypoints.cli.main:main
  vllm/entrypoints/cli/serve.py:39-40                 `vllm serve` launches an OpenAI-compatible API server
  vllm/entrypoints/cli/serve.py:154-156               one API server runs in the CLI process
  vllm/entrypoints/launchers/api_server/entry.py:94   the API server builds AsyncLLM
  vllm/v1/engine/async_llm.py:193                     AsyncLLM always uses make_async_mp_client
  vllm/entrypoints/llm.py:341-343                     LLM builds a V1 LLMEngine
  vllm/v1/engine/llm_engine.py:186-188                VLLM_ENABLE_V1_MULTIPROCESSING -> multiprocess mode
  vllm/v1/engine/core_client.py:98-136                make_client: SyncMPClient or InprocClient
  vllm/engine/async_llm_engine.py:6, llm_engine.py:6  old names are aliases only
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Two entry points into one engine core",
         "vllm serve (online, the main path of this site)  ·  the LLM class (offline, a side note)",
         params="same EngineCore class (vllm/v1/engine/core.py) behind both entry points")

BW, BH = 270, 106
X1, X2, X3 = 46, 413, 780

# ---------------------------------------------------------------- panel 1
y0 = 112
P1_H = 206
d.panel(y0, P1_H, 1, "Online: vllm serve",
        "An HTTP client talks to the API server. The API server talks to the engine core process over ZMQ.")
by = y0 + 74
d.box(X1, by, BW, BH, "HTTP client", [
    "curl, the openai SDK, a router",
    "POST /v1/chat/completions",
    "JSON in, JSON or SSE out",
    "port 8000 by default",
], col=INK, align="start")
d.box(X2, by, BW, BH, "API server process (front end)", [
    "uvicorn + FastAPI routes",
    "OpenAIServingChat, renderer",
    "AsyncLLM",
    "AsyncMPClient (ZMQ, msgpack)",
], col=BLUE, align="start")
d.box(X3, by, BW, BH, "Engine core process", [
    "EngineCoreProc: busy loop",
    "Scheduler + KV cache manager",
    "executor -> worker(s)",
    "worker -> GPU(s)",
], col=VIOLET, align="start")
d.arrow(X1 + BW, by + 50, X2 - 2, by + 50, INK)
d.arrow_label((X1 + BW + X2) / 2, by + 50, "HTTP", INK)
d.arrow(X2 + BW, by + 50, X3 - 2, by + 50, BLUE)
d.arrow_label((X2 + BW + X3) / 2, by + 50, "ZMQ", BLUE)
d.arrow(X3, by + 90, X2 + BW + 2, by + 90, VIOLET)
d.arrow_label((X2 + BW + X3) / 2, by + 90, "outputs", VIOLET)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 206
d.panel(y1, P2_H, 2, "Offline: the LLM class",
        "Your script calls LLM.generate(). The LLMEngine reaches the same engine core, by default in its own process.")
by = y1 + 74
d.box(X1, by, BW, BH, "Your Python script", [
    "llm = LLM(model=...)",
    "llm.generate(prompts)",
    "llm.chat(messages)",
    "no HTTP, no FastAPI",
], col=AMBER, align="start")
d.box(X2, by, BW, BH, "LLMEngine (in your process)", [
    "vllm.v1.engine.llm_engine",
    "multiprocess mode (default):",
    "  SyncMPClient -> ZMQ",
    "else InprocClient (same process)",
], col=AMBER, align="start")
d.box(X3, by, BW, BH, "Engine core", [
    "same EngineCore class",
    "own process with SyncMPClient",
    "in your process with InprocClient",
    "executor -> worker(s) -> GPU(s)",
], col=VIOLET, align="start")
d.arrow(X1 + BW, by + 50, X2 - 2, by + 50, AMBER)
d.arrow_label((X1 + BW + X2) / 2, by + 50, "Python call", AMBER)
d.arrow(X2 + BW, by + 50, X3 - 2, by + 50, AMBER)
d.arrow_label((X2 + BW + X3) / 2, by + 50, "ZMQ or call", AMBER)

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "online front end (vllm serve)"), (AMBER, "offline path (LLM class)"),
                 (VIOLET, "engine core (shared)")])
y += 40
y = d.footnotes(30, y, [
    "Simplified: one API server and one engine core (no data parallelism). With TP = 1 the worker is an object inside",
    "  the engine core process; with TP > 1 each GPU gets its own worker process.",
    "AsyncLLMEngine and vllm.engine.LLMEngine are only aliases of AsyncLLM and the V1 LLMEngine. V0 is gone.",
])
d.save(height=y + 14)
