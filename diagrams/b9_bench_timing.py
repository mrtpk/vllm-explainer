"""Client timing points of vllm bench serve vs server timing points of vllm serve, for one streamed request.

Used by: src/concepts/benchmarking.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/benchmarks/lib/endpoint_request_func.py:248-307   st = perf_counter() before POST; ttft at first chunk with
                                                          choices; itl per later chunk; latency = last chunk - st
  vllm/benchmarks/serve.py:654-660                         TPOT = (latency - ttft) / (n - 1), only n > 1
  vllm/renderers/base.py:1380                              arrival_time = time.time() at render start (chat)
  vllm/v1/core/sched/scheduler.py:2746-2747, 1420-1422     QUEUED, SCHEDULED events (time.monotonic)
  vllm/v1/metrics/stats.py:477-478, 525-527, 553-558       TTFT = front-end time.time() - arrival; ITL from engine
  vllm/v1/metrics/stats.py:585-616                         e2e, queue, prefill, decode; TPOT = decode / (n - 1)
  vllm/entrypoints/generate/base/serving.py:73-74          per-request TTFT = first_token_ts - scheduled_ts
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

CLIENT, FRONT, ENGINE = BLUE, VIOLET, EMERALD

d = Diagram(width=1100)
d.header("Client timing vs server timing of one streamed request",
         "vllm bench serve -> POST /v1/completions (stream) -> vllm serve  ·  schematic, not to scale",
         params="client: time.perf_counter()  ·  API server: time.time()  ·  engine core: time.monotonic()")

y0 = 118
P1 = 350
d.panel(y0, P1, 1, "Where does each tool start and stop its clock?",
        "The client clock starts first and stops last. The server clocks start only when rendering starts.")

# schematic axis (not to scale): x positions chosen so that no labels overlap
ay = y0 + 176
d.line(70, ay, 1060, ay, INK, 1.4)
d.text(1060, ay + 16, "time (not to scale)", 11.5, MUTE, anchor="end")

XC = {"st": 100, "c1": 610, "c2": 700, "c3": 790, "cl": 1000}
XS = {"arr": 190, "q": 290, "s": 390, "t1": 530, "tl": 920}

d.text(70, y0 + 82, "Client (vllm bench serve)", 13, BLUE[0], "bold")
for key, label in [("st", "POST sent: st"), ("c1", "chunk 1"), ("c2", "chunk 2"), ("c3", "chunk 3"),
                   ("cl", "last chunk")]:
    x = XC[key]
    d.line(x, ay - 60, x, ay, BLUE[0], 1.2, dash="3 3")
    d.circle(x, ay - 60, 4, BLUE[0])
    d.text(x, ay - 70, label, 11.5, BLUE[0], "bold", "middle")
bk = ay - 34
d.line(XC["st"], bk, XC["c1"], bk, BLUE[0], 2)
d.text((XC["st"] + XC["c1"]) / 2, bk - 6, "client ttft", 11.5, BLUE[0], "bold", "middle", mono=True)
d.line(XC["c1"], bk, XC["c2"], bk, BLUE[0], 2, dash="5 3")
d.text(XC["c3"] + 12, bk + 4, "<- itl: one gap per chunk", 11.5, BLUE[0], "bold", mono=True)

for key, label, col in [("arr", "arrival_time", FRONT), ("q", "QUEUED", ENGINE), ("s", "SCHEDULED", ENGINE),
                        ("t1", "1st token", ENGINE), ("tl", "last token", ENGINE)]:
    x = XS[key]
    d.line(x, ay, x, ay + 60, col[0], 1.2, dash="3 3")
    d.circle(x, ay + 60, 4, col[0])
    d.text(x, ay + 78, label, 11.5, col[0], "bold", "middle")
sk = ay + 30
d.line(XS["arr"], sk, XS["t1"] + 14, sk, VIOLET[0], 2)
d.text(XS["t1"] + 24, sk + 4, "Prometheus TTFT (front-end clock)", 11.5, VIOLET[0], "bold", mono=True)
pk = ay + 104
d.line(XS["s"], pk, XS["t1"], pk, EMERALD[0], 2)
d.text(XS["t1"] + 10, pk + 4, "per-request time_to_first_token_ms (engine clock)", 11.5, EMERALD[0], "bold",
       mono=True)
d.text(70, ay + 136, "Server (vllm serve)", 13, VIOLET[0], "bold")

y1 = y0 + P1 + 18
P2 = 196
d.panel(y1, P2, 2, "Which numbers does each tool report?",
        "Same names, different start points, clocks and units of work.")
bx = [40, 384, 728]
bw, bh = 332, 112
byy = y1 + 70
d.box(bx[0], byy, bw, bh, "vllm bench serve (client)", [
    "TTFT: POST -> 1st chunk",
    "TPOT: (E2EL - TTFT)/(n - 1), n > 1",
    "ITL: gap between SSE chunks",
    "E2EL: POST -> last chunk"], col=CLIENT, align="left")
d.box(bx[1], byy, bw, bh, "Prometheus (API server)", [
    "TTFT: arrival -> output handled",
    "TPOT: decode / (n - 1), 0 if n <= 1",
    "ITL: gap of engine timestamps",
    "e2e: arrival -> finished"], col=FRONT, align="left")
d.box(bx[2], byy, bw, bh, "Per-request \"metrics\" field", [
    "time_to_first_token_ms:",
    "SCHEDULED -> 1st token",
    "(prefill only, no queue)",
    "queue_time_ms: QUEUED -> SCHED."], col=ENGINE, align="left")

y = y1 + P2 + 24
d.legend(30, y, [(CLIENT, "client clock"), (FRONT, "API server clock"), (ENGINE, "engine core clock")])
y += 40
y = d.footnotes(30, y, [
    "The axis is schematic. With --stream-interval > 1 one SSE chunk carries several tokens: the client ITL grows, Prometheus ITL does not.",
    "Simplified: the network, HTTP parsing and SSE serialization are inside the client numbers only.",
])
d.save(height=y + 14)
