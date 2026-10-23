"""Where each observability signal of vllm serve comes from.

Used by: src/concepts/metrics-and-observability.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/v1/engine/__init__.py:176-200            EngineCoreEvent types, monotonic timestamps
  vllm/v1/core/sched/scheduler.py:2746-2747, 1420-1422, 1696-1697   QUEUED / SCHEDULED / PREEMPTED events
  vllm/v1/core/sched/scheduler.py:3023-3050     SchedulerStats (running, waiting, kv usage, prefix cache)
  vllm/v1/engine/async_llm.py:880-885, 929-935  IterationStats per output batch, record()
  vllm/v1/metrics/stats.py:477-478, 525-527, 553-558   front-end clock for TTFT, engine timestamps for ITL
  vllm/v1/metrics/loggers.py:1385-1419          StatLoggerManager: log-line logger + Prometheus logger
  vllm/entrypoints/serve/instrumentator/metrics.py:56-81   /metrics
  vllm/entrypoints/launchers/utils/server_utils.py:75-86   log every VLLM_LOG_STATS_INTERVAL
  vllm/v1/engine/output_processor.py:888-889, 941-1001     llm_request span
  vllm/entrypoints/generate/base/serving.py:45-97          per-request metrics in the response
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import BLUE, EMERALD, VIOLET, Diagram  # noqa: E402

ENG, FE, OUT = EMERALD, BLUE, VIOLET

d = Diagram(width=1100)
d.header("Where the signals of vllm serve come from",
         "One API server, one engine core  ·  stats on (no --disable-log-stats)",
         params="engine core: time.monotonic()  ·  API server: time.time()  ·  bench client: time.perf_counter()")

y0 = 118
P1 = 190
d.panel(y0, P1, 1, "What does the engine core send?",
        "Timestamps and counters only. It computes no latency metric itself.")
bx = [40, 384, 728]
bw, bh = 332, 100
by = y0 + 72
d.box(bx[0], by, bw, bh, "Request events", [
    "QUEUED   in add_request",
    "SCHEDULED one per schedule()",
    "PREEMPTED when recomputed"], col=ENG, align="left")
d.box(bx[1], by, bw, bh, "EngineCoreOutputs", [
    "new token ids per request",
    "timestamp = time.monotonic()",
    "+ events of each request"], col=ENG, align="left")
d.box(bx[2], by, bw, bh, "SchedulerStats", [
    "running, waiting, kv_cache_usage",
    "prefix cache queries / hits",
    "spec decode drafts / accepted"], col=ENG, align="left")
d.arrow(bx[0] + bw, by + bh / 2, bx[1] - 2, by + bh / 2, ENG)
d.arrow(bx[2], by + bh / 2, bx[1] + bw + 2, by + bh / 2, ENG)

y1 = y0 + P1 + 18
P2 = 190
d.panel(y1, P2, 2, "What does the API server compute?",
        "AsyncLLM's output handler turns each output batch into stats and gives them to the stat loggers.")
by = y1 + 72
d.box(bx[0], by, bw, bh, "IterationStats", [
    "TTFT = time.time() - arrival",
    "ITL = gap of engine timestamps",
    "tokens, finished requests"], col=FE, align="left")
d.box(bx[1], by, bw, bh, "StatLoggerManager.record()", [
    "LoggingStatLogger (1 API server)",
    "PrometheusStatLogger (always)",
    "+ stat logger plugins"], col=FE, align="left")
d.box(bx[2], by, bw, bh, "Per request (finished)", [
    "e2e, queue, prefill, decode,",
    "inference time, TPOT",
    "RequestStateStats"], col=FE, align="left")
d.arrow(bx[0] + bw, by + bh / 2, bx[1] - 2, by + bh / 2, FE)
d.arrow(bx[2], by + bh / 2, bx[1] + bw + 2, by + bh / 2, FE)

y2 = y1 + P2 + 18
P3 = 190
d.panel(y2, P3, 3, "Where can you read them?",
        "Three places. Each one has its own switch.")
bx3 = [40, 384, 728]
by = y2 + 72
d.box(bx3[0], by, bw, bh, "GET /metrics (Prometheus)", [
    "vllm:* histograms, gauges, counters",
    "off: --disable-log-stats",
    "N API servers: multiprocess dir"], col=OUT, align="left")
d.box(bx3[1], by, bw, bh, "Log line every 10 s", [
    "\"Avg prompt throughput: ...\"",
    "VLLM_LOG_STATS_INTERVAL",
    "only with 1 API server"], col=OUT, align="left")
d.box(bx3[2], by, bw, bh, "Traces and response fields", [
    "--otlp-traces-endpoint: llm_request",
    "--enable-per-request-metrics:",
    "\"metrics\" in the response body"], col=OUT, align="left")

y = y2 + P3 + 24
d.legend(30, y, [(ENG, "engine core process"), (FE, "API server process"), (OUT, "what you read")])
y += 40
y = d.footnotes(30, y, [
    "Simplified: with DP > 1, each API server records one series per engine (label engine=\"0\", \"1\", ...).",
    "Not shown: http_* metrics of prometheus_fastapi_instrumentator, the torch profiler routes /start_profile and /stop_profile.",
])
d.save(height=y + 14)
