"""Data parallel serving: API servers, engine cores and the DP coordinator for -dp 4 (dense model).

Used by: src/concepts/data-parallel-serving.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/cli/serve.py:113-130            api_server_count defaults (internal LB: DP size)
  vllm/v1/engine/utils.py:1227-1262                coordinator only on DP rank 0 when needs_dp_coordinator
  vllm/config/vllm.py:856-874                      needs_dp_coordinator
  vllm/v1/engine/core.py:1534-1543                 dense: independent EngineCoreProc per rank
  vllm/v1/engine/core.py:1622-1633                 engines send [waiting, running] + kv usage when counts change
  vllm/v1/engine/coordinator.py:249-276            publish every 100 ms when changed, else every 5 s
  vllm/v1/engine/core_client.py:1572-1586          API server keeps the latest counts (lb_engines)
  vllm/v1/engine/core_client.py:1743-1804          DPLBAsyncMPClient score and pick
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, NEUTRAL, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Data parallel serving: who talks to whom",
         "vllm serve M -dp 4  ·  dense model  ·  one node, 4 GPUs  ·  internal load balancing (the default)",
         params="1 CLI supervisor + 4 API servers + 1 DP coordinator + 4 engine cores + 4 workers = 14 processes")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 400
d.panel(y0, P1_H, 1, "Which processes run?",
        "Every API server can send to every engine core. The coordinator only moves load counts.")
SY = y0 + 66
d.rect(46, SY, 1008, 36, fill=NEUTRAL[2], stroke=NEUTRAL[0], rx=8, sw=1.4)
d.text(550, SY + 23, "CLI process: starts and watches the other processes  ·  serves no HTTP", 12.5, INK,
       "bold", "middle")

RY = SY + 56
d.box(46, RY, 500, 96, "4 API servers  (ApiServer_0 … ApiServer_3)", [
    "one shared listen socket (port 8000)",
    "each: AsyncLLM + DPLBAsyncMPClient",
    "client_index i of client_count 4",
], col=BLUE, align="start")
d.box(600, RY, 454, 96, "DP coordinator  (VLLM_DP_Coordinator)", [
    "counts in: [waiting, running, kv_usage]",
    "counts out: XPUB to all API servers",
    "dense model: no request waves",
], col=PINK, align="start")

LY = RY + 96 + 46
LH = 108
d.lane(46, LY, 1008, LH, VIOLET, "4 engine cores: independent EngineCoreProc, one per DP rank", [], header_h=34)
for k in range(2):
    for j in range(2):
        i = 2 * k + j
        cx = 62 + j * 496 + (0 if True else 0)
        cy = LY + 42 + k * 32
        d.rect(cx, cy, 480, 26, fill=VIOLET[1], stroke=VIOLET[0], rx=6, sw=1.2)
        d.text(cx + 12, cy + 18, f"EngineCore_DP{i}   +1 worker process on GPU {i}   (identity {i:02x} 00)",
               11.5, INK, mono=True)

# arrows
d.arrow(296, RY + 96, 296, LY - 2, AMBER, sw=2)
d.text(308, RY + 96 + 26, "requests (ZMQ ROUTER → DEALER)", 11.5, AMBER[0], "bold")
d.arrow(250, LY, 250, RY + 98, EMERALD)
d.text(150, RY + 96 + 40, "outputs (PUSH)", 11.5, EMERALD[0], "bold")
d.arrow(827, LY, 827, RY + 98, PINK, dash="5 4")
d.text(837, RY + 96 + 30, "counts (client index −1)", 11.5, PINK[0], "bold")
d.path_arrow(f"M 600,{RY + 70} L 548,{RY + 70}", PINK, dash="5 4")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 214
d.panel(y1, P2_H, 2, "How does one API server pick an engine?",
        "It scores each engine with its newest counts and its own in-flight requests; the lowest score wins.")
d.step_card(46, y1 + 70, 330, 118, 1, "Engines report", PINK, tag="core.py", rows=[
    ("when", "counts change"),
    ("send", "waiting, running,"),
    ("", "kv_cache_usage (0-1)"),
    ("to", "coordinator"),
], key_w=44)
d.step_card(385, y1 + 70, 330, 118, 2, "Coordinator publishes", PINK, tag="100 ms", rows=[
    ("every", "100 ms if changed"),
    ("else", "every 5 s"),
    ("msg", "(counts, wave, running)"),
    ("to", "every API server"),
], key_w=44)
d.step_card(724, y1 + 70, 330, 118, 3, "API server scores", BLUE, tag="per request", rows=[
    ("s =", "max(C·inflight, w + r)"),
    ("  +", "6·w·max(0, kv − 0.5)"),
    ("pick", "lowest s, scan from"),
    ("", "a rotating start"),
], key_w=44)
d.arrow(376, y1 + 129, 383, y1 + 129, PINK)
d.arrow(715, y1 + 129, 722, y1 + 129, PINK)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "API server"), (VIOLET, "engine core"), (PINK, "load counts"),
                 (AMBER, "requests"), (EMERALD, "outputs")])
y += 36
y = d.footnotes(30, y, [
    "C = client_count (number of API servers); inflight = this API server's own unfinished requests on that engine;",
    "  w, r = waiting and running counts from the coordinator. A request with X-data-parallel-rank skips the score.",
    "MoE model: engines are DPEngineCoreProc and the coordinator also tracks request waves (see the deep dive).",
])
d.save(height=y + 14)
