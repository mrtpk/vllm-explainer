"""Scheduler-side and worker-side KV connector calls in one engine step.

Used by: src/concepts/kv-connectors.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/v1/core/sched/scheduler.py:1027-1039   get_num_new_matched_tokens(request, block-aligned local hit)
  vllm/v1/core/sched/scheduler.py:1320-1337   allocate_slots(..., delay_cache_blocks=load_kv_async)
  vllm/v1/core/sched/scheduler.py:1352-1357   update_state_after_alloc(request, blocks, num_external_tokens)
  vllm/v1/core/sched/scheduler.py:1384-1411   async load -> WAITING_FOR_REMOTE_KVS
  vllm/v1/core/sched/scheduler.py:1615-1617   build_connector_meta -> scheduler_output.kv_connector_metadata
  vllm/v1/worker/gpu/kv_connector.py:75-96    pre_forward: handle_preemptions, bind_connector_metadata, sync start_load_kv
  vllm/model_executor/layers/attention/kv_transfer_utils.py:50-57  wait_for_layer_load / save_kv_layer per layer
  vllm/v1/worker/gpu/kv_connector.py:136-160  post_forward: async start_load_kv, wait_for_save, get_transfer_results
  vllm/v1/core/sched/scheduler.py:2427-2429, 3312-3338   update_connector_output, finished_recving / finished_sending
  vllm/v1/core/sched/scheduler.py:3105-3158, 2840-2842   request_finished -> delay the block free
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One step with a KV connector",
         "Engine core = scheduler role  ·  each worker = worker role  ·  the metadata object links them",
         params="KVConnectorBase_V1  ·  vllm/v1/core/sched/scheduler.py  ·  vllm/v1/worker/gpu/kv_connector.py")

LX = [40, 390, 740]
LW = 320
SCH, WRK, BACK = BLUE, VIOLET, EMERALD
CH = 104

# ---- Panel 1: scheduler side ----------------------------------------------------------------------
y1 = 120
P1_H = 70 + CH + 22
d.panel(y1, P1_H, 1, "Scheduler role: what to load, what to save? (engine core, schedule())",
        "For a new request the connector reports external hits; the scheduler allocates blocks and asks for a plan.")
r = y1 + 70
d.step_card(LX[0], r, LW, CH, 1, "get_num_new_matched_tokens", SCH, rows=[
    ("in", "request, local hit tokens"),
    ("out", "(ext_tokens, load_async)"),
    ("None", "ask again next step  :1043"),
], key_w=40)
d.step_card(LX[1], r, LW, CH, 2, "allocate + update_state_after_alloc", SCH, rows=[
    ("alloc", "delay_cache_blocks=load_async"),
    ("tell", "blocks, num_external_tokens"),
    ("async", "-> WAITING_FOR_REMOTE_KVS"),
], key_w=46)
d.step_card(LX[2], r, LW, CH, 3, "build_connector_meta", SCH, rows=[
    ("in", "the SchedulerOutput of the step"),
    ("out", "KVConnectorMetadata"),
    ("", "-> scheduler_output  :1617"),
], key_w=30)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r + 52, LX[i + 1] - 2, r + 52, SCH)

# ---- Panel 2: worker side -------------------------------------------------------------------------
y2 = y1 + P1_H + 18
P2_H = 70 + CH + 22
d.panel(y2, P2_H, 2, "Worker role: move the bytes (each worker, execute_model)",
        "The worker binds the metadata, loads before or after the forward, and saves layer by layer.")
r = y2 + 70
d.step_card(LX[0], r, LW, CH, 4, "pre_forward", WRK, tag="kv_connector.py:75", rows=[
    ("1", "handle_preemptions(meta)"),
    ("2", "bind_connector_metadata(meta)"),
    ("3", "sync loads: start_load_kv()"),
], key_w=18)
d.step_card(LX[1], r, LW, CH, 5, "each attention layer", WRK, tag="utils.py:50", rows=[
    ("in", "wait_for_layer_load(name)"),
    ("run", "attention of the layer"),
    ("out", "save_kv_layer(name, kv, meta)"),
], key_w=30)
d.step_card(LX[2], r, LW, CH, 6, "post_forward", WRK, tag="kv_connector.py:136", rows=[
    ("1", "async loads: start_load_kv()"),
    ("2", "wait_for_save()"),
    ("3", "get_transfer_results -> output"),
], key_w=18)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r + 52, LX[i + 1] - 2, r + 52, WRK)

# ---- Panel 3: back in the engine core ---------------------------------------------------------------
y3 = y2 + P2_H + 18
P3_H = 70 + CH + 22
d.panel(y3, P3_H, 3, "Scheduler role again: what finished? (engine core, update_from_output())",
        "KVConnectorOutput reports finished sends and receives; the scheduler frees or wakes requests.")
r = y3 + 70
d.step_card(LX[0], r, LW, CH, 7, "update_connector_output", BACK, tag=":3322", rows=[
    ("recv", "request may run next step"),
    ("send", "free the blocks now  :3338"),
    ("fail", "kv_load_failure_policy"),
], key_w=38)
d.step_card(LX[1], r, LW, CH, 8, "request_finished", BACK, tag=":3151", rows=[
    ("in", "request, block_ids"),
    ("out", "(delay_free, kv_transfer_params)"),
    ("True", "keep blocks until sent"),
], key_w=38)
d.step_card(LX[2], r, LW, CH, 9, "next schedule()", BACK, tag=":3285", rows=[
    ("ready", "cache the loaded blocks,"),
    ("", "WAITING -> schedule the rest"),
    ("", "of the prompt"),
], key_w=46)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r + 52, LX[i + 1] - 2, r + 52, BACK)

y = y3 + P3_H + 16
d.callout(30, y, 1040, "One connector class, two objects:",
          "KVConnectorFactory.create_connector(role=SCHEDULER) in the engine core, role=WORKER in each worker.",
          AMBER, h=34)
y += 50
d.legend(30, y, [(SCH, "scheduler role, schedule()"), (WRK, "worker role"), (BACK, "scheduler role, update")])
y += 34
y = d.footnotes(30, y, [
    "Line numbers without a file are in vllm/v1/core/sched/scheduler.py; utils.py:50 is",
    "  model_executor/layers/attention/kv_transfer_utils.py. Not shown: take_events, KV events, stats, preemption,",
    "  HMA (request_finished_all_groups) and the EC connector, which mirrors these calls.",
])
d.save(height=y + 8)
