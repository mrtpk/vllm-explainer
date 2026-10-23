"""Deployment shapes of vllm serve and the processes that each shape starts.

Used by: src/concepts/deployment-recipes.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/entrypoints/cli/serve.py:54-156          mode selection: one API server, multi API server, headless, supervisor
  vllm/entrypoints/cli/serve.py:113-130         --api-server-count default = DP size (internal LB), 1 (external LB)
  vllm/entrypoints/cli/serve.py:217-231         headless node with node_rank_within_dp > 0: MultiprocExecutor only
  vllm/config/parallel.py:1055, 1099-1100       uni executor only when world size across DP is 1
  vllm/config/vllm.py:853-874                   DP coordinator for MoE, or for dense models with internal LB
  vllm/v1/engine/coordinator.py:101-115         the coordinator is its own process
  vllm/engine/arg_utils.py:2445-2454            dense models: no external DP mode, run independent instances
  vllm/entrypoints/serve/instrumentator/health.py:22-33   /health 200, or 503 after EngineDeadError
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

API, CORE, WORKER, LB = BLUE, VIOLET, EMERALD, AMBER

d = Diagram(width=1100)
d.header("Four deployment shapes of vllm serve, as processes",
         "One box = one OS process (or a group of equal processes)  ·  arrows = who starts or feeds whom",
         params="dense model unless marked MoE  ·  GPU numbers are local to the node")

CX = [40, 570]
CW = 490


def cmd(x, y, s):
    d.text(x, y, s, 12, INK, mono=True)


def col_box(x, y, h, title, lines, col):
    d.box(x, y, CW, h, title, lines, col=col, align="left")


def down(x, y1, y2, col, label=None):
    d.arrow(x, y1, x, y2 - 2, col)
    if label:
        d.text(x + 10, (y1 + y2) / 2 + 4, label, 11.5, MUTE, mono=True)


# ---------------------------------------------------------------- panel 1
y0 = 118
P1 = 384
d.panel(y0, P1, 1, "One model copy on one node: how many processes?",
        "TP = 1 keeps the worker inside the engine core. TP > 1 adds one worker process per GPU.")
top = y0 + 78
cmd(CX[0], top, "A  vllm serve M")
cmd(CX[1], top, "B  vllm serve M -tp 2")
bh = 66
y = top + 16
col_box(CX[0], y, bh, "API server process (the vllm CLI)", ["uvicorn + FastAPI + AsyncLLM", "port 8000"], API)
col_box(CX[1], y, bh, "API server process (the vllm CLI)", ["uvicorn + FastAPI + AsyncLLM", "port 8000"], API)
y2 = y + bh + 34
down(CX[0] + 60, y + bh, y2, API[0], "ZMQ")
down(CX[1] + 60, y + bh, y2, API[0], "ZMQ")
col_box(CX[0], y2, bh + 14, "EngineCore process", ["scheduler + KV cache manager",
                                                     "uni executor: the worker runs here", "GPU 0"], CORE)
col_box(CX[1], y2, bh, "EngineCore process", ["scheduler + KV cache manager", "mp executor"], CORE)
y3 = y2 + bh + 34
down(CX[1] + 60, y2 + bh, y3, CORE[0], "shared-memory MessageQueue")
col_box(CX[1], y3, bh - 8, "2 worker processes", ["Worker_TP0 on GPU 0, Worker_TP1 on GPU 1"], WORKER)
d.text(CX[0] + 4, y3 + bh + 12, "= 2 processes", 13, INK, "bold")
d.text(CX[1] + CW - 4, y3 + bh + 12, "= 4 processes", 13, INK, "bold", "end")

# ---------------------------------------------------------------- panel 2
y0b = y0 + P1 + 18
P2 = 404
d.panel(y0b, P2, 2, "Scale out: data parallel on one node, or one model over two nodes?",
        "-dp N copies the whole engine N times. --nnodes splits one engine's ranks over several machines.")
top = y0b + 78
cmd(CX[0], top, "C  vllm serve M -dp 2")
cmd(CX[1], top, "D  -tp 8 -pp 2 --nnodes 2 --node-rank 0|1")
y = top + 16
col_box(CX[0], y, bh - 8, "CLI process: supervisor only", ["starts and watches the others"], API)
col_box(CX[1], y, bh, "Node 0: API server + EngineCore", ["2 processes, --master-addr = node 0",
                                                            "mp executor of 16 ranks"], CORE)
y2 = y + bh + 26
down(CX[0] + 60, y + bh - 8, y2, API[0])
down(CX[1] + 60, y + bh, y2, CORE[0], "ranks 0-7")
col_box(CX[0], y2, bh, "ApiServer_0, ApiServer_1", ["one port (SO_REUSEPORT)", "each with its own AsyncLLM"], API)
col_box(CX[1], y2, bh - 8, "Node 0: 8 worker processes", ["GPUs 0-7, global ranks 0-7"], WORKER)
y3 = y2 + bh + 26
down(CX[0] + 60, y2 + bh, y3, API[0], "ZMQ")
col_box(CX[0], y3, bh + 14, "2 EngineCore + 2 worker processes", ["EngineCore_DP0 -> worker on GPU 0",
                                                                   "EngineCore_DP1 -> worker on GPU 1",
                                                                   "+ DP coordinator process"], CORE)
col_box(CX[1], y3 - 8, bh + 22, "Node 1: --headless", ["CLI process = executor monitor",
                                                       "8 worker processes, ranks 8-15",
                                                       "no API server, no EngineCore"], WORKER)
d.text(CX[0] + 4, y3 + bh + 38, "= 8 processes", 13, INK, "bold")
d.text(CX[1] + CW - 4, y3 + bh + 38, "= 10 + 9 processes", 13, INK, "bold", "end")

# ---------------------------------------------------------------- panel 3
y0c = y0b + P2 + 18
P3 = 250
d.panel(y0c, P3, 3, "Behind a load balancer: one engine per endpoint?",
        "For a dense model, start independent servers. External DP flags are only for MoE models.")
top = y0c + 74
d.box(380, top, 340, 56, "Load balancer / router", ["probes GET /health: 200 or 503"], col=LB)
yb = top + 56 + 40
d.path_arrow(f"M 470,{top + 56} L 470,{top + 76} L 285,{top + 76} L 285,{yb - 2}", LB[0])
d.path_arrow(f"M 630,{top + 56} L 630,{top + 76} L 815,{top + 76} L 815,{yb - 2}", LB[0])
d.box(CX[0], yb, CW, 66, "vllm serve M --port 8000", ["CUDA_VISIBLE_DEVICES=0, no --data-parallel-*",
                                                       "a full shape A: 2 processes"], col=API, align="left")
d.box(CX[1], yb, CW, 66, "vllm serve M --port 8001", ["CUDA_VISIBLE_DEVICES=1, no --data-parallel-*",
                                                       "a full shape A: 2 processes"], col=API, align="left")

y = y0c + P3 + 24
d.legend(30, y, [(API, "API server / CLI"), (CORE, "engine core"), (WORKER, "GPU workers"),
                 (LB, "outside vLLM")])
y += 40
y = d.footnotes(30, y, [
    "Counts are derived from the launch code (serve.py, parallel.py, coordinator.py). D: node 0 = API server + EngineCore + 8 workers;",
    "node 1 = the headless CLI process + 8 workers. MoE external LB: --data-parallel-size N --data-parallel-rank i per server.",
])
d.save(height=y + 14)
