"""Disaggregated prefill with NixlConnector: one request across a router, a prefill and a decode instance.

Used by: src/concepts/disaggregated-prefill.html
Sources (at the pinned commit in explainer.json):
  tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:153-188, 211-245   the toy router
  vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:34-106     get_num_new_matched_tokens
  vllm/distributed/kv_transfer/kv_connector/v1/nixl/pull_scheduler.py:191-293    request_finished: delay free + params
  vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_scheduler.py:76-88      side channel host/port, lease 30 s
  vllm/v1/core/sched/scheduler.py:1384-1411   async load → WAITING_FOR_REMOTE_KVS
  vllm/v1/core/sched/scheduler.py:3274-3290   finished recving → WAITING
  vllm/v1/core/sched/scheduler.py:3459-3539   failed loads: fail (FINISHED_ERROR) or recompute
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Prefill on one instance, decode on another",
         "NixlConnector (pull)  ·  toy router from tests/  ·  one chat request with a 2,000-token prompt",
         params='P: kv_role "kv_producer", port 8100  ·  D: kv_role "kv_consumer", port 8200  ·  router: port 8192')

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 380
d.panel(y0, P1_H, 1, "Who does what for one request?",
        "The router runs the request twice: a 1-token prefill on P, then the real request on D with P's KV.")
BY = y0 + 112
BW, BH = 316, 186
d.box(46, BY, BW, BH, "Router (not part of vLLM)", [
    "1  receives POST /v1/chat/completions",
    "2  sends a copy to P: max_tokens 1,",
    "   stream false, kv_transfer_params",
    "   {do_remote_decode: true}",
    "4  sends the original body to D with",
    "   P's returned kv_transfer_params",
    "7  streams D's answer to the client",
], col=BLUE, align="start")
d.box(392, BY, BW, BH, "Prefill instance P", [
    "3  computes the 2,000-token prompt,",
    "   samples 1 token, finishes",
    "   request_finished(): keep blocks",
    "   (lease 30 s), return params:",
    "   do_remote_prefill: true,",
    "   remote_block_ids, remote_host,",
    "   remote_port, remote_engine_id …",
], col=AMBER, align="start")
d.box(738, BY, BW, BH, "Decode instance D", [
    "5  connector: (2000 tokens, async)",
    "   → WAITING_FOR_REMOTE_KVS",
    "   worker pulls the blocks (NIXL READ)",
    "   and notifies P (P frees them)",
    "6  finished_recving → WAITING;",
    "   recompute the last prompt token,",
    "   sample, then decode as usual",
], col=EMERALD, align="start")
d.arrow(362, BY + 50, 390, BY + 50, BLUE)
d.text(376, BY + 42, "2", 12, BLUE[0], "bold", "middle")
d.path_arrow(f"M 550,{BY + BH} L 550,{BY + BH + 24} L 204,{BY + BH + 24} L 204,{BY + BH + 2}", AMBER)
d.text(222, BY + BH + 40, "3  response JSON carries kv_transfer_params", 11.5, AMBER[0], "bold")
d.path_arrow(f"M 300,{BY} L 300,{BY - 22} L 896,{BY - 22} L 896,{BY - 2}", BLUE)
d.text(560, BY - 28, "4  request + P's kv_transfer_params", 11.5, BLUE[0], "bold")
d.arrow(708, BY + 110, 736, BY + 110, PINK, sw=2)
d.arrow(738, BY + 150, 710, BY + 150, EMERALD, dash="5 4")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 200
d.panel(y1, P2_H, 2, "What states does the request pass on D?",
        "The scheduler parks the request until the worker reports finished_recving, then schedules the rest.")
CY = y1 + 76
d.chip(46, CY, 140, "WAITING", VIOLET)
d.arrow(186, CY + 15, 216, CY + 15, VIOLET)
d.chip(218, CY, 270, "WAITING_FOR_REMOTE_KVS", AMBER)
d.arrow(488, CY + 15, 518, CY + 15, AMBER)
d.chip(520, CY, 140, "WAITING", VIOLET)
d.arrow(660, CY + 15, 690, CY + 15, VIOLET)
d.chip(692, CY, 140, "RUNNING", EMERALD)
d.text(46, CY + 54, "load failed + kv_load_failure_policy \"fail\" (default)  →  FINISHED_ERROR, the router sees an error",
       12, PINK[0], "bold")
d.text(46, CY + 76, "load failed + \"recompute\"  →  num_computed_tokens cut at the first bad block; D prefills the rest itself",
       12, INK)
d.text(46, CY + 98, "P side: blocks stay until D's read notification, or until the lease runs out without a heartbeat",
       12, INK)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "router / HTTP"), (AMBER, "prefill instance"), (EMERALD, "decode instance"),
                 (PINK, "KV bytes (RDMA / NIXL)")])
y += 36
y = d.footnotes(30, y, [
    "Numbers are an example. The router in vLLM's repo is a toy (tests/.../toy_proxy_server.py); production routers are",
    "  external projects. The P instance's 1 sampled token is discarded by the router.",
])
d.save(height=y + 14)
