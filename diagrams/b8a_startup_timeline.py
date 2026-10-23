"""Start-up of `vllm serve <model>` on one GPU: the 18 tour stops on a time line in two processes.

Used by: src/tours/serve-startup.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/entrypoints/cli/main.py:39-117                       stops 1-2: env setup, parsers, parse
  vllm/engine/arg_utils.py:3237-3258                        stop 2: plugins + pre_register_and_update per parser
  vllm/entrypoints/cli/serve.py:54-156                      stop 3: single API server -> run_server
  vllm/entrypoints/launchers/launcher.py:365-392            stop 4: banner, non-default args, bind the socket
  vllm/engine/arg_utils.py:1961-1970, 2234-2905             stops 5-7: EngineArgs, create_engine_config
  vllm/config/vllm.py:1492-2318                             stop 8: VllmConfig.__post_init__
  vllm/v1/engine/async_llm.py:167-201                       stop 9: AsyncLLM
  vllm/v1/engine/core_client.py:672-697, utils.py:1357-1395 stop 10: sockets, spawn EngineCore, wait
  vllm/v1/engine/core.py:1449-1472, 1190-1226, 129-147      stop 11: engine core process, handshake, plugins
  vllm/v1/executor/uniproc_executor.py:51-75                stops 12-13: worker in the same process
  vllm/v1/worker/gpu_worker.py:446-549, 591-598             stops 12-13: init_device, load_model
  vllm/v1/engine/core.py:288-342                            stop 14: profile run, KV cache size
  vllm/v1/engine/core.py:362, 379                           stops 15-16: allocate, compile + capture
  vllm/v1/engine/core.py:1364-1386, 1243-1273               stop 17: READY, I/O threads
  vllm/v1/engine/core_client.py:753-777                     stop 17: ready response, monitor
  vllm/entrypoints/launchers/api_server/entry.py:115-161    stop 18: build_app, init_app_state, uvicorn
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, TEAL, VIOLET, Diagram  # noqa: E402

GRAY = ("#4b5563", "#f3f4f6", "#fafafa")
CLI, CONF, IPC, GPU = BLUE, VIOLET, AMBER, TEAL

d = Diagram(width=1100)
d.header("Start-up of vllm serve on one GPU",
         "vllm serve <model>  ·  TP = PP = DP = 1  ·  uni executor: the worker runs inside the engine core process",
         params="badge = tour stop  ·  time runs downward  ·  log lines in quotes")

y0 = 118
ROW, IH = 46, 38
stops_a1 = [
    (1, "cli/main.py: main", "imports, cli_env_setup -> spawn", CLI),
    (2, "build parsers, parse argv", "plugins, pre_register x 3", CLI),
    (3, "ServeSubcommand.cmd", "one API server -> run_server", CLI),
    (4, "setup_server: bind the port", "\"non-default args: {...}\"", CLI),
    (5, "AsyncEngineArgs.from_cli_args", "Namespace -> EngineArgs", CONF),
    (6, "create_engine_config: model", "\"Resolved architecture: ...\"", CONF),
    (7, "create_engine_config: the rest", "Cache, Parallel, Scheduler", CONF),
    (8, "VllmConfig.__post_init__", "async sched, -O2, graphs", CONF),
    (9, "AsyncLLM.__init__", "tokenizer, input/output proc", CLI),
    (10, "MPClient + launch_core_engines", "ZMQ sockets, spawn EngineCore", IPC),
]
stops_b = [
    (11, "run_engine_core -> EngineCore", "HELLO; \"Initializing a V1 ...\"", IPC),
    (12, "uni executor: init_device", "torch.distributed, world 1", GPU),
    (13, "load_model", "\"Model loading took ... GiB\"", GPU),
    (14, "_initialize_kv_caches", "compile in profile run; KV size", GPU),
    (15, "initialize_from_config", "allocate the KV cache tensors", GPU),
    (16, "compile_or_warm_up_model", "\"Graph capturing finished ...\"", GPU),
]
stops_a2 = [
    (17, "READY -> wait_for_engine_startup", "ready response on input socket", IPC),
    (18, "build_and_serve: uvicorn", "\"Application startup complete.\"", CLI),
]

n_rows = len(stops_a1) + len(stops_b) + len(stops_a2)
LANE_TOP = y0 + 70
HEAD = 52
lane_h = HEAD + 16 + n_rows * ROW + 8
P_H = 70 + lane_h + 18
d.panel(y0, P_H, 1, "Which process does what, and when?",
        "Stops 1-10 run in the API server process. Stops 11-16 run in the engine core. Stop 17 joins them.")

LA, LB, LW = 40, 570, 490
d.lane(LA, LANE_TOP, LW, lane_h, BLUE, "API server process (the vllm CLI)", ["log prefix: (APIServer pid=...)"],
       header_h=HEAD)
d.lane(LB, LANE_TOP, LW, lane_h, EMERALD, "Engine core process (worker inside)", ["log prefix: (EngineCore pid=...)"],
       header_h=HEAD)


def item(x, row, n, title, sub, col):
    y = LANE_TOP + HEAD + 16 + row * ROW
    d.rect(x + 12, y, LW - 24, IH, fill=col[1], stroke=col[0], rx=8, sw=1.3)
    d.badge(x + 32, y + IH / 2, n, col[0], r=11, size=11)
    d.text(x + 52, y + 16, title, 12.5, INK, "bold", mono=True)
    d.text(x + 52, y + 31, sub, 11.5, MUTE, mono=True)
    return y


row = 0
for s in stops_a1:
    item(LA, row, *s)
    row += 1
first_b = row
for s in stops_b:
    item(LB, row, *s)
    row += 1
last_b = row - 1
for s in stops_a2:
    item(LA, row, *s)
    row += 1

# engine core does not exist yet (rows 0..9)
ny = LANE_TOP + HEAD + 16
d.rect(LB + 12, ny, LW - 24, first_b * ROW - 8, fill="#ffffff", stroke=GRAY[0], rx=8, sw=1, dash="4 3")
d.text(LB + LW / 2, ny + first_b * ROW / 2 - 8, "not started yet", 12.5, MUTE, "bold", "middle")
d.text(LB + LW / 2, ny + first_b * ROW / 2 + 10, "the process starts at stop 10", 11.5, MUTE, anchor="middle")

# API server waits (rows first_b..last_b)
wy = LANE_TOP + HEAD + 16 + first_b * ROW
wh = (last_b - first_b + 1) * ROW - 8
d.rect(LA + 12, wy, LW - 24, wh, fill="#ffffff", stroke=IPC[0], rx=8, sw=1.2, dash="4 3")
d.text(LA + LW / 2, wy + wh / 2 - 18, "waits in wait_for_engine_startup()", 12.5, IPC[0], "bold", "middle",
       mono=True)
d.text(LA + LW / 2, wy + wh / 2 + 2, "polls every 10 s; no time limit with one API server", 11.5, INK,
       anchor="middle")
d.text(LA + LW / 2, wy + wh / 2 + 20, "(v1/engine/utils.py:1434-1447)", 11.5, MUTE, anchor="middle", mono=True)

# arrows: stop 10 -> stop 11, stop 16 -> stop 17
y10 = LANE_TOP + HEAD + 16 + (first_b - 1) * ROW + IH / 2
y11 = LANE_TOP + HEAD + 16 + first_b * ROW + IH / 2
d.path_arrow(f"M {LA + LW - 12:.0f},{y10:.0f} L {LA + LW + 20:.0f},{y10:.0f} L {LA + LW + 20:.0f},{y11:.0f} "
             f"L {LB + 10:.0f},{y11:.0f}", IPC)
d.text((LA + LW + LB) / 2, y10 - 8, "spawn", 11.5, IPC[0], "bold", "middle")
y16 = LANE_TOP + HEAD + 16 + last_b * ROW + IH / 2
y17 = LANE_TOP + HEAD + 16 + (last_b + 1) * ROW + IH / 2
d.path_arrow(f"M {LB + 12:.0f},{y16:.0f} L {LB - 20:.0f},{y16:.0f} L {LB - 20:.0f},{y17:.0f} "
             f"L {LA + LW - 10:.0f},{y17:.0f}", IPC)
d.text((LA + LW + LB) / 2, y16 - 8, "ready", 11.5, IPC[0], "bold", "middle")

y = y0 + P_H + 18
d.legend(30, y, [(CLI, "CLI and HTTP server"), (CONF, "configuration"), (IPC, "processes and handshake"),
                 (GPU, "GPU work in the worker")])
y += 34
y = d.footnotes(30, y, [
    "Stops 12-16 run in the worker object; with TP = 1 it lives in the engine core process (uniproc_executor.py:68).",
    "Times are not to scale: on a cold start, stops 13, 14 and 16 (weights, compile, capture) take most of the time.",
])
d.save(height=y + 8)
