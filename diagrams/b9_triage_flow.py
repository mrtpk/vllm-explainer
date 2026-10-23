"""Triage flow for a vllm serve problem: start-up, serving, output quality.

Used by: src/concepts/troubleshooting.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/v1/engine/core.py:1570-1572              "EngineCore failed to start." / "EngineCore encountered a fatal error."
  vllm/v1/engine/utils.py:1468-1477             "Engine core initialization failed. See root cause above."
  vllm/v1/core/kv_cache_planning.py:56-87       no KV memory / max_model_len does not fit
  vllm/compilation/decorators.py:326-330        "Directly load AOT compilation from path ..."
  vllm/entrypoints/serve/exception_handling/error_response.py:43-96   400 / 500 / 503 mapping
  vllm/entrypoints/serve/engine/serving.py:75-84   404 for an unknown model name
  vllm/entrypoints/serve/instrumentator/health.py:22-33   /health 503 after EngineDeadError
  vllm/parser/parser_manager.py:34-35           no tool parser without --enable-auto-tool-choice
  vllm/v1/structured_output/__init__.py:143-150 the engine keeps its first structured-output backend
  vllm/config/compilation.py:1343-1361          custom op switch that matches no op: WARNING only
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, PINK, Diagram  # noqa: E402

START, SERVE, OUT = AMBER, BLUE, PINK

d = Diagram(width=1100)
d.header("Triage: in which phase does the problem show?",
         "vllm serve  ·  read the log of every process: (APIServer pid=...), (EngineCore pid=...), (Worker_TP0 pid=...)",
         params="first question: did the log reach \"Application startup complete.\"?")

y0 = 118
LW, LH = 336, 548
LX = [30, 382, 734]
d.lane(LX[0], y0, LW, LH, START, "1  Start-up fails or is slow", ["before \"Application startup complete.\""])
d.lane(LX[1], y0, LW, LH, SERVE, "2  Requests fail or are slow", ["the server runs; HTTP errors or latency"])
d.lane(LX[2], y0, LW, LH, OUT, "3  HTTP 200, wrong output", ["tools, JSON, quality, plugins"])

CH, PITCH = 100, 112
items = [
    [  # start-up
        ("Engine core initialization failed", [("see", "the first traceback above it"),
                                               ("log", "EngineCore failed to start."),
                                               ("often", "plugin error, bad env value")]),
        ("ValueError about KV cache memory", [("see", "No available memory ..."),
                                             ("or", "max seq len ... larger than"),
                                             ("fix", "--max-model-len, util")]),
        ("CUDA out of memory at start-up", [("in", "profile run or graph capture"),
                                           ("fix", "max_num_batched_tokens,"),
                                           ("", "--enforce-eager, util")]),
        ("Start-up takes minutes", [("log", "torch.compile took X s"),
                                    ("warm", "Directly load AOT compilation"),
                                    ("keep", "VLLM_CACHE_ROOT on disk")]),
    ],
    [  # serving
        ("HTTP 400", [("read", "error.param in the body"),
                      ("often", "context length, stream_options"),
                      ("", "max_tokens, tool_choice")]),
        ("HTTP 404", [("why", "model is not a served name"),
                      ("fix", "--served-model-name, LoRA"),
                      ("", "name, or GET /v1/models")]),
        ("HTTP 500 and /health 503", [("why", "the engine core died"),
                                      ("log", "EngineDeadError + traceback"),
                                      ("then", "the server exits (5 s)")]),
        ("Slow: TTFT or ITL high", [("look", "num_requests_waiting > 0"),
                                    ("", "kv_cache_usage_perc near 1"),
                                    ("", "num_preemptions rate > 0")]),
    ],
    [  # output
        ("Tool calls arrive as text", [("need", "--enable-auto-tool-choice"),
                                       ("and", "--tool-call-parser NAME"),
                                       ("", "that matches the model")]),
        ("Structured output rejected", [("why", "engine keeps 1st backend"),
                                        ("fix", "set one backend in"),
                                        ("", "--structured-outputs-config")]),
        ("Quality changed", [("why", "defaults from the model's"),
                             ("", "generation_config.json"),
                             ("fix", "--generation-config vllm")]),
        ("Plugin or op has no effect", [("look", "VLLM_PLUGINS filter"),
                                        ("", "VLLM_LOGGING_LEVEL=DEBUG"),
                                        ("", "op name: +rms_norm")]),
    ],
]
cols = [START, SERVE, OUT]
for li, lane_items in enumerate(items):
    for i, (title, rows) in enumerate(lane_items):
        y = y0 + 84 + i * PITCH
        d.step_card(LX[li] + 12, y, LW - 24, CH, i + 1, title, cols[li], rows=rows, key_w=50)

y = y0 + LH + 24
d.legend(30, y, [(START, "start-up"), (SERVE, "serving"), (OUT, "output quality")])
y += 40
y = d.footnotes(30, y, [
    "Each card names the first thing to check. The page text gives the cause, the fix and the code line for each one.",
    "Simplified: multi-node hangs (NCCL) are not shown; see docs/serving/distributed_troubleshooting.md.",
])
d.save(height=y + 14)
