"""Architecture overview (b): vllm serve with TP = 2, DP = 2 and two API servers.

Used by: src/concepts/architecture-overview.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/cli/serve.py:113-130, 151-152   api_server_count defaults to DP size -> run_multi_api_server
  vllm/entrypoints/cli/serve.py:298, 333-338, 368-377  shared listen socket, ZMQ listeners, engines, API servers
  vllm/v1/utils.py:230-254                         one ApiServer_i process per server, client_index i of client_count
  vllm/v1/engine/utils.py:1230-1265                DP coordinator only on DP rank 0 and when needed
  vllm/config/vllm.py:855-874                      needs_dp_coordinator: MoE, or internal/hybrid LB
  vllm/v1/engine/utils.py:282-284                  EngineCore_DP<i> processes run run_engine_core
  vllm/v1/engine/core.py:1534-1543                 dense model: plain EngineCoreProc per rank, DP reconfigured to 1
  vllm/v1/engine/core.py:1936-1943, 2047-2052      one DEALER and one PUSH socket per API server
  vllm/v1/engine/core_client.py:158-169, 1743-1804 DPLBAsyncMPClient picks an engine per request
  vllm/v1/executor/multiproc_executor.py:165-171, 428, 745-750  shm MessageQueue broadcast, VllmWorker processes
  vllm/v1/executor/multiproc_executor.py:549-563   only the output rank replies
  vllm/v1/engine/coordinator.py:23-54, 264-267     coordinator publishes load stats (100 ms when changed)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, NEUTRAL, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("vllm serve -tp 2 -dp 2: ten processes",
         "dense (non-MoE) model  ·  one node, 4 GPUs  ·  internal load balancing  ·  api_server_count = DP size = 2",
         params="1 CLI supervisor + 2 API servers + 1 DP coordinator + 2 engine cores + 4 workers")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 594
d.panel(y0, P1_H, 1, "Which processes run, and which ones talk to each other?",
        "Every API server talks to every engine core. Each engine core drives its own two workers.")

# CLI supervisor bar
SY = y0 + 68
d.rect(46, SY, 1008, 40, fill=NEUTRAL[2], stroke=NEUTRAL[0], rx=8, sw=1.4)
d.text(550, SY + 25, "CLI process: run_multi_api_server()  ·  starts and watches all processes  ·  serves no HTTP",
       13, INK, "bold", "middle")

# API servers + coordinator
RY = SY + 62
RH = 112
A0X, A1X, COX = 46, 360, 740
AW = 284
COW = 314
d.box(A0X, RY, AW, RH, "API server 0  (ApiServer_0)", [
    "client_index = 0, client_count = 2",
    "AsyncLLM + DPLBAsyncMPClient",
    "ROUTER + PULL pair #0",
    "shares the one listen socket",
], col=BLUE, align="start")
d.box(A1X, RY, AW, RH, "API server 1  (ApiServer_1)", [
    "client_index = 1, client_count = 2",
    "AsyncLLM + DPLBAsyncMPClient",
    "ROUTER + PULL pair #1",
    "shares the one listen socket",
], col=BLUE, align="start")
d.box(COX, RY, COW, RH, "DP coordinator", [
    "VLLM_DP_Coordinator process",
    "collects waiting, running, KV usage",
    "publishes them to the API servers",
    "dense model: no request waves",
], col=PINK, align="start")

# ZMQ bus under the API servers
BUSY = RY + RH + 34
BUSW = A1X + AW - A0X
d.rect(A0X, BUSY, BUSW, 36, fill=AMBER[1], stroke=AMBER[0], rx=8, sw=1.4)
d.text(A0X + BUSW / 2, BUSY + 23, "ZMQ, msgpack: each engine core connects a DEALER + PUSH to every pair",
       12, AMBER[0], "bold", "middle")
for x in (A0X + AW / 2, A1X + AW / 2):
    d.arrow(x, RY + RH, x, BUSY - 2, AMBER)

# engine lanes
LY = BUSY + 36 + 40
LH = 214
L0X, L1X = 46, 574
LW = 480
for i, lx in enumerate((L0X, L1X)):
    d.lane(lx, LY, LW, LH, VIOLET, f"DP rank {i}", [f"engine index {i}  ·  GPUs {2 * i} and {2 * i + 1}"],
           header_h=50)
    ex = lx + 16
    d.box(ex, LY + 62, LW - 32, 62, f"EngineCore_DP{i}  (EngineCoreProc)", [
        "scheduler, KV cache manager, MultiprocExecutor",
    ], col=VIOLET, align="start", tsize=13)
    d.box(ex, LY + 150, LW - 32, 52, "2 worker processes (VllmWorker, TP group)", [
        f"Worker_TP0 on GPU {2 * i}   Worker_TP1 on GPU {2 * i + 1}",
    ], col=EMERALD, align="start", tsize=13)
    d.arrow(ex + 60, LY + 124, ex + 60, LY + 148, EMERALD)
    d.text(ex + 70, LY + 140, "shm MessageQueue (pickle)", 11, EMERALD[0], "bold", mono=True)

# bus -> lanes
d.arrow(L0X + LW / 2, BUSY + 36, L0X + LW / 2, LY - 2, AMBER)
d.path_arrow(f"M {A1X + AW / 2 + 60},{BUSY + 36} L {A1X + AW / 2 + 60},{BUSY + 56} "
             f"L {L1X + 120},{BUSY + 56} L {L1X + 120},{LY - 2}", AMBER)
# coordinator <-> engines (dashed)
cx = COX + COW / 2
d.arrow(cx, LY, cx, RY + RH + 2, PINK, dash="5 4")
d.text(cx + 10, RY + RH + 40, "load stats in,", 11.5, PINK[0], "bold")
d.text(cx + 10, RY + RH + 56, "published to API servers", 11.5, PINK[0], "bold")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 232
d.panel(y1, P2_H, 2, "Which channel carries what?",
        "Three channels with three formats: msgpack over ZMQ, pickle over shared memory, tensors over NCCL.")
CY = y1 + 70
CH = 136
d.step_card(46, CY, 330, CH, 1, "API server ⇄ engine core", AMBER, tag="ZMQ", rows=[
    ("in", "ADD / ABORT / UTILITY"),
    ("", "EngineCoreRequest"),
    ("out", "EngineCoreOutputs"),
    ("fmt", "msgpack, ipc:// on one node"),
], key_w=40)
d.step_card(385, CY, 330, CH, 2, "Engine core → its workers", EMERALD, tag="shm", rows=[
    ("in", "execute_model(SchedulerOutput)"),
    ("", "broadcast to every worker"),
    ("out", "ModelRunnerOutput"),
    ("fmt", "pickle; reply from rank 0 only"),
], key_w=40)
d.step_card(724, CY, 330, CH, 3, "Worker ⇄ worker (TP group)", INK, tag="NCCL", rows=[
    ("what", "activations, all-reduce"),
    ("when", "inside the forward pass"),
    ("where", "GPU memory, no Python"),
    ("PP", "send/recv between stages"),
], key_w=46)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "API server"), (PINK, "DP coordinator"), (VIOLET, "engine core"),
                 (EMERALD, "worker / GPU"), (AMBER, "ZMQ")])
y += 40
y = d.footnotes(30, y, [
    "Process names: ApiServer_i (v1/utils.py:250), EngineCore_DPi (v1/engine/utils.py:284), VLLM_DP_Coordinator (coordinator.py:103).",
    "Workers start as VllmWorker-<rank> (multiproc_executor.py:748) and later rename themselves, for example Worker_TP0.",
    "MoE model: DPEngineCoreProc instead, and the ranks run in lock-step waves with dummy batches (concepts/data-parallel-serving).",
])
d.save(height=y + 14)
