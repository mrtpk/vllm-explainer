"""One request's timeline with the serving metrics, the three clocks, and goodput over five requests.

Used by: src/learn/serving-metrics.html
All times are a made-up example (marked on the drawing). Sources (pinned commit 3e44c0d):
  vllm/renderers/base.py:1380                     arrival_time = time.time() when chat rendering starts
  vllm/v1/core/sched/scheduler.py:2746-2747       QUEUED event in Scheduler.add_request
  vllm/v1/core/sched/scheduler.py:664, 1420-1422  SCHEDULED event with one timestamp per schedule() call
  vllm/v1/engine/__init__.py:296-297               EngineCoreOutputs.timestamp = time.monotonic()
  vllm/v1/metrics/stats.py:524-527                 TTFT = frontend time - arrival_time
  vllm/v1/metrics/stats.py:551-558                 ITL = gap between engine-core output timestamps
  vllm/v1/metrics/stats.py:594-616                 e2e, queued, prefill, decode, inference, mean TPOT
  vllm/benchmarks/lib/endpoint_request_func.py:443-501  client TTFT / ITL / latency with perf_counter
  vllm/benchmarks/serve.py:669-692                 goodput: a request is good if it meets every SLO
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import (  # noqa: E402
    AMBER, BLUE, INK, MUTE, NEUTRAL, PINK, TEAL, VIOLET, Diagram, _c, _t,
)

d = Diagram(width=1100)
d.header("Serving metrics on one request, and on many",
         "Example numbers (made up)  ·  one streamed chat request  ·  then 5 requests in one benchmark run",
         params="TTFT  ·  TPOT  ·  ITL  ·  E2E latency  ·  throughput  ·  goodput")

# ---------------------------------------------------------------- panel 1
y0 = 124
P1_H = 300
d.panel(y0, P1_H, 1, "Where do the metrics start and stop?",
        "Each metric is the time between two events of the same request. Not to scale.")

ax = y0 + 190            # axis centre line
X0, XQ, XS, XF, XL = 80, 175, 330, 540, 1020
band_h = 24

# phase bands
d.rect(X0, ax - band_h / 2, XQ - X0, band_h, fill=_t(NEUTRAL), stroke=_c(NEUTRAL), rx=4, sw=1.2)
d.text((X0 + XQ) / 2, ax + 4, "render", 11.5, INK, "bold", "middle")
d.rect(XQ, ax - band_h / 2, XS - XQ, band_h, fill=_t(AMBER), stroke=_c(AMBER), rx=4, sw=1.2)
d.text((XQ + XS) / 2, ax + 4, "waiting", 11.5, _c(AMBER), "bold", "middle")
d.rect(XS, ax - band_h / 2, XF - XS, band_h, fill=_t(VIOLET), stroke=_c(VIOLET), rx=4, sw=1.2)
d.text((XS + XF) / 2, ax + 4, "prefill (all chunks)", 11.5, _c(VIOLET), "bold", "middle")
d.rect(XF, ax - band_h / 2, XL - XF, band_h, fill=_t(TEAL), stroke=_c(TEAL), rx=4, sw=1.2)

# output ticks in the decode band: regular gaps, one long gap (a stall)
ticks = [XF]
x = XF
gaps = [40, 40, 40, 40, 110, 40, 40, 40, 40, 40, 50]
for g in gaps:
    x += g
    ticks.append(x)
ticks[-1] = XL
for t in ticks:
    d.line(t, ax - band_h / 2, t, ax + band_h / 2, _c(TEAL), sw=2)
d.text(XF + 40 * 4 + 55, ax + 4, "stall", 11, _c(TEAL), "bold", "middle")

# brackets above the axis
def bracket(xa, xb, y, col, label, up=True):
    h = 7 if up else -7
    d.line(xa, y, xb, y, col, sw=1.6)
    d.line(xa, y, xa, y + h, col, sw=1.6)
    d.line(xb, y, xb, y + h, col, sw=1.6)
    ty = y - 6 if up else y + 16
    d.text((xa + xb) / 2, ty, label, 12, col, "bold", "middle")

bracket(X0, XL, ax - 92, INK, "E2E latency = 2,200 ms  (arrival_time to last output)")
bracket(X0, XF, ax - 50, _c(BLUE), "TTFT = 200 ms")
bracket(XF, XL, ax - 50, _c(BLUE), "decode time 2,000 ms  ·  TPOT = 2,000 / (101 - 1) = 20 ms")

# brackets below the axis
bracket(XQ, XS, ax + 26, _c(AMBER), "queue 50 ms", up=False)
bracket(XS, XF, ax + 26, _c(VIOLET), "prefill 145 ms", up=False)
bracket(ticks[1], ticks[2], ax + 26, _c(BLUE), "ITL", up=False)
bracket(ticks[4], ticks[5], ax + 26, _c(BLUE), "long ITL", up=False)

# event labels
ev_y = ax + 74
for xv, name, t in [(X0, "arrival_time", "0"), (XQ, "QUEUED", "5"), (XS, "SCHEDULED", "55"),
                    (XF, "first token", "200"), (XL, "last token", "2,200")]:
    d.line(xv, ax + band_h / 2, xv, ev_y - 14, MUTE, sw=1, dash="3 3")
    anchor = "end" if xv == XL else "middle"
    xt = xv + 8 if xv == XL else xv
    d.text(xt, ev_y, name, 11.5, INK, "bold", anchor, mono=True)
    d.text(xt, ev_y + 16, f"{t} ms", 11.5, MUTE, anchor=anchor, mono=True)

# ---------------------------------------------------------------- panel 2
y2 = y0 + P1_H + 20
P2_H = 210
d.panel(y2, P2_H, 2, "Who measures, with which clock?",
        "An interval uses two timestamps from one process. So the client and the server report different numbers.")
bw, bh, by = 300, 118, y2 + 72
bx = [46, 400, 754]
d.box(bx[0], by, bw, bh, "Benchmark client", [
    "vllm bench serve",
    "time.perf_counter()",
    "TTFT: send -> 1st SSE chunk",
    "ITL: gap between SSE chunks",
    "E2EL: send -> last chunk",
], col=BLUE, mono_size=11.5)
d.box(bx[1], by, bw, bh, "API server (AsyncLLM)", [
    "time.time()  (wall clock)",
    "arrival_time: render starts",
    "TTFT, E2E: on this clock",
    "computes all server metrics",
    "when outputs arrive",
], col=BLUE, mono_size=11.5)
d.box(bx[2], by, bw, bh, "EngineCore process", [
    "time.monotonic()",
    "QUEUED, SCHEDULED events",
    "one timestamp per step output",
    "timestamps for queue, prefill,",
    "decode, ITL, TPOT",
], col=BLUE, mono_size=11.5)
d.arrow(bx[0] + bw, by + bh / 2, bx[1] - 2, by + bh / 2, INK)
d.arrow_label((bx[0] + bw + bx[1]) / 2, by + bh / 2, "HTTP", INK)
d.arrow(bx[1] + bw, by + bh / 2, bx[2] - 2, by + bh / 2, INK)
d.arrow_label((bx[1] + bw + bx[2]) / 2, by + bh / 2, "ZMQ", INK)

# ---------------------------------------------------------------- panel 3
y3 = y2 + P2_H + 20
rows = [
    # name, start s, ttft ms, e2e ms, verdict, colour
    ("A", 0.0, 200, 2200, "good", INK),
    ("B", 0.5, 150, 1400, "good", INK),
    ("C", 1.0, 650, 5050, "misses TTFT", PINK),
    ("D", 2.2, 300, 5800, "misses TPOT", PINK),
    ("E", 3.0, 120, 120, "good", INK),
]
row_h = 28
P3_H = 92 + len(rows) * row_h + 34
d.panel(y3, P3_H, 3, "How many requests per second meet the SLO?",
        "SLO: TTFT <= 500 ms and TPOT <= 50 ms. Throughput = 5 / 8 s = 0.625 req/s. Goodput = 3 / 8 s = 0.375 req/s.")
TX0, TX1 = 110, 890          # 0 s .. 8 s
sx = (TX1 - TX0) / 8.0
ry = y3 + 80
for i, (name, st, ttft, e2e, verdict, col) in enumerate(rows):
    yy = ry + i * row_h
    d.text(TX0 - 22, yy + 15, name, 12.5, INK, "bold", "middle", mono=True)
    xa = TX0 + st * sx
    xf = TX0 + (st + ttft / 1000) * sx
    xb = TX0 + (st + e2e / 1000) * sx
    d.rect(xa, yy + 3, max(xf - xa, 3), 18, fill=_t(BLUE), stroke=_c(BLUE), rx=3, sw=1.1)
    if xb - xf > 2:
        d.rect(xf, yy + 3, xb - xf, 18, fill=_t(TEAL), stroke=_c(TEAL), rx=3, sw=1.1)
    d.chip(910, yy + 1, 150, verdict, col, h=22)
axis_y = ry + len(rows) * row_h + 6
d.line(TX0, axis_y, TX1, axis_y, MUTE, sw=1)
for s in range(0, 9):
    xs = TX0 + s * sx
    d.line(xs, axis_y, xs, axis_y + 5, MUTE, sw=1)
    d.text(xs, axis_y + 19, f"{s} s", 11.5, MUTE, anchor="middle", mono=True)

# ---------------------------------------------------------------- bottom matter
y = y3 + P3_H + 24
d.legend(30, y, [(BLUE, "metric; where it is measured; time to first token"), (NEUTRAL, "render"),
                 (AMBER, "waiting"), (VIOLET, "prefill"), (TEAL, "decode outputs"), (PINK, "misses an SLO")])
y += 36
y = d.footnotes(30, y, [
    "All numbers are a made-up example. Panel 3 bars: blue = until the first token, teal = decode.",
    "vLLM sets arrival_time when the API server starts to render the chat (renderers/base.py:1380); QUEUED and SCHEDULED are engine-core events.",
    "A preemption does not move SCHEDULED: vLLM keeps the first SCHEDULED time (v1/metrics/stats.py:577).",
])
d.save(height=y + 10)
