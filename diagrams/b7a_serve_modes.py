"""The launch modes of `vllm serve` (ServeSubcommand.cmd) and the processes each one makes.

Used by: src/concepts/openai-server.html, src/code/entrypoints-cli-serve.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/cli/serve.py:54-156     ServeSubcommand.cmd: grpc, headless, LB modes, api_server_count default, dispatch
  vllm/entrypoints/cli/serve.py:181-266    run_headless: CoreEngineProcManager, no API server
  vllm/entrypoints/cli/serve.py:269-407    run_multi_api_server: setup_server(reuse_port), launch_core_engines, APIServerProcessManager
  vllm/entrypoints/launchers/api_server/entry.py:164-201  run_server / run_server_worker (one API server)
  vllm/entrypoints/launchers/grpc_server.py:56-82         serve_grpc: AsyncLLM + gRPC servicer
  vllm/v1/utils.py:167-261                 APIServerProcessManager: spawn ApiServer_i, client_index / client_count
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("How vllm serve picks a launch mode",
         "ServeSubcommand.cmd() in vllm/entrypoints/cli/serve.py  ·  checks run top to bottom",
         params="api_server_count default: DP size (internal LB) · 1 (external LB, multi-port, Rust) · local DP (hybrid) · 0 (--headless)")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 372
d.panel(y0, P1_H, 1, "Which branch runs?",
        "The first matching check wins; most deployments reach the last box.")
BW = 300
X1, X2, X3 = 46, 400, 754
R1 = y0 + 70
BH = 112
d.step_card(X1, R1, BW, BH, 1, "--grpc", PINK, tag="serve.py:59-63", rows=[
    ("run", "serve_grpc(args)"),
    ("has", "AsyncLLM + gRPC servicer"),
    ("no", "HTTP routes, no FastAPI"),
])
d.step_card(X2, R1, BW, BH, 2, "multi-port LB", PINK, tag="serve.py:147-148", rows=[
    ("run", "run_dp_supervisor(args)"),
    ("has", "one server per local DP rank"),
    ("via", "--data-parallel-multi-port-..."),
])
d.step_card(X3, R1, BW, BH, 3, "count < 1", AMBER, tag="serve.py:149-150", rows=[
    ("run", "run_headless(args)"),
    ("has", "engine cores only, no HTTP"),
    ("via", "--headless sets count 0"),
])
R2 = R1 + BH + 30
d.step_card(X1, R2, BW + 160, BH, 4, "count > 1, or Rust frontend", VIOLET, tag="serve.py:151-152",
            rows=[
                ("run", "run_multi_api_server(args)"),
                ("has", "CLI process = supervisor, N API server processes"),
                ("via", "-dp N (default count = N) or --api-server-count N"),
            ])
d.step_card(X1 + BW + 160 + 18, R2, 1054 - (X1 + BW + 160 + 18), BH, 5, "else: one API server", BLUE,
            tag="serve.py:153-156", rows=[
                ("run", "uvloop.run(run_server(args))"),
                ("has", "HTTP server in the CLI process"),
                ("via", "the default: TP any, DP = 1"),
            ])
d.text(X1, R2 + BH + 28,
       "Before the branches: --headless with --api-server-count > 0 raises ValueError; more than one DP load-balancing mode raises ValueError.",
       11.5, MUTE, italic=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 230
d.panel(y1, P2_H, 2, "Which processes does each common mode make?",
        "Example values; each engine core process may start more worker processes for TP > 1.")
R3 = y1 + 70
BH3 = 136
d.box(X1, R3, BW, BH3, "vllm serve M  (TP = 1)", [
    "CLI process = API server",
    "  uvicorn + FastAPI + AsyncLLM",
    "1 engine core process",
    "  worker inside it (uni executor)",
    "total: 2 processes",
], col=BLUE, align="start")
d.box(X2, R3, BW, BH3, "vllm serve M -dp 2  (dense)", [
    "CLI process = supervisor only",
    "ApiServer_0, ApiServer_1 (spawn)",
    "  one shared listen socket",
    "2 engine core processes",
    "  + 1 worker process each",
], col=VIOLET, align="start")
d.box(X3, R3, BW, BH3, "vllm serve M --headless", [
    "CLI process = engine manager",
    "local DP engine cores only",
    "API servers run on another node",
    "  and connect over ZMQ (TCP)",
    "used for multi-node DP",
], col=AMBER, align="start")

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "single API server"), (VIOLET, "several API servers"), (AMBER, "no API server"),
                 (PINK, "other front ends")])
y += 40
y = d.footnotes(30, y, [
    "-dp 2 dense, total 8 processes: CLI supervisor, 2 API servers, DP coordinator (internal LB), 2 engine cores, 2 workers.",
    "Not shown in panel 2: the DP coordinator box and the Rust frontend binary (VLLM_USE_RUST_FRONTEND=1).",
])
d.save(height=y + 14)
