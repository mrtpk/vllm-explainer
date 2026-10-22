"""B3a concept "The scheduler": one call of Scheduler.schedule() with its two passes.

Used by: src/concepts/scheduler.html, src/code/v1-core-sched-scheduler.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/core/sched/scheduler.py:713-715     running pass: from the front of self.running, while budget > 0
  vllm/v1/core/sched/scheduler.py:759-777     num_new_tokens and its caps
  vllm/v1/core/sched/scheduler.py:811-829     num_new_tokens == 0 -> continue (not break)
  vllm/v1/core/sched/scheduler.py:831-904     allocate_slots, preemption loop, break if still None
  vllm/v1/core/sched/scheduler.py:956-981     waiting pass only if nothing was preempted; max_num_seqs cap
  vllm/v1/core/sched/scheduler.py:986-1009    blocked requests and LoRA are skipped
  vllm/v1/core/sched/scheduler.py:1016-1024   prefix-cache lookup for a new or preempted request
  vllm/v1/core/sched/scheduler.py:1191-1231   num_new_tokens, chunked prefill off -> break
  vllm/v1/core/sched/scheduler.py:1320-1346   allocate_slots (full prompt must fit), None -> break
  vllm/v1/core/sched/scheduler.py:1413-1439   the request becomes RUNNING
  vllm/v1/core/sched/scheduler.py:1588-1643   SchedulerOutput, then _update_after_schedule
  vllm/v1/core/sched/async_scheduler.py:25-50 placeholders for requests that sample a token
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One call of Scheduler.schedule(): two passes over one token budget",
         "No prefill phase and no decode phase  ·  RUNNING requests first, then WAITING requests  ·  then the plan",
         params="token_budget = max_num_scheduled_tokens (= max_num_batched_tokens)  ·  blocks from KVCacheManager.allocate_slots")

XS = [46, 400, 754]
BW = 300


def row_arrows(y, col):
    d.arrow(XS[0] + BW, y, XS[1] - 2, y, col)
    d.arrow(XS[1] + BW, y, XS[2] - 2, y, col)


# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 270
d.panel(y0, P1_H, 1, "Pass 1: what does each RUNNING request get?",
        "Front to back through self.running. Each request asks for the tokens it is missing; the budget caps it.")
by = y0 + 70
BH = 150
d.box(XS[0], by, BW, BH, "a. How many tokens?", [
    "num_new = num_tokens_with_spec",
    "        + num_output_placeholders",
    "        - num_computed_tokens",
    "cap: long_prefill_token_threshold",
    "cap: token_budget",
    "cap: max_model_len - computed - 1",
    "num_new == 0  ->  continue",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[1], by, BW, BH, "b. KV blocks for them?", [
    "allocate_slots(request, num_new)",
    "",
    "blocks OK:",
    "  scheduled[req] = num_new",
    "  token_budget -= num_new",
    "  next request",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[2], by, BW, BH, "c. Blocks short: preempt", [
    "fcfs:     victim = running.pop()",
    "priority: max(priority, arrival)",
    "free victim's blocks,",
    "  num_computed_tokens = 0,",
    "  status PREEMPTED, to waiting",
    "retry allocate_slots",
    "victim is this request -> stop pass",
], col=AMBER, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, by + 40, XS[1] - 2, by + 40, BLUE)
d.arrow(XS[1] + BW, by + 40, XS[2] - 2, by + 40, AMBER)
d.arrow_label((XS[1] + BW + XS[2]) / 2, by + 40, "None", AMBER)
d.path_arrow(f"M {XS[2] + BW / 2},{by + BH} L {XS[2] + BW / 2},{by + BH + 22} "
             f"L {XS[1] + BW / 2},{by + BH + 22} L {XS[1] + BW / 2},{by + BH + 2}", AMBER, dash="5 4")
d.text(XS[1] + BW / 2 + 10, by + BH + 38, "retry after a victim is freed", 11.5, AMBER, "bold")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 262
d.panel(y1, P2_H, 2, "Pass 2: which WAITING requests can start?",
        "Runs only if pass 1 preempted nothing. Stops at the first request that does not fit.")
by = y1 + 70
BH = 150
d.box(XS[0], by, BW, BH, "a. May it start now?", [
    "no preemption in this step",
    "len(running) < max_num_seqs",
    "grammar compiled / KV loaded",
    "  (else skip, try the next)",
    "LoRA: <= max_loras adapters",
    "  (else skip, try the next)",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], by, BW, BH, "b. Prefix hit and tokens", [
    "computed = get_computed_blocks()",
    "  (only if num_computed_tokens 0)",
    "num_new = num_tokens - computed",
    "chunked prefill off and",
    "  num_new > budget -> stop pass",
    "num_new = min(num_new, budget)",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[2], by, BW, BH, "c. allocate_slots()", [
    "all num_tokens must fit (default)",
    "None -> stop pass (break)",
    "",
    "OK: pop from waiting,",
    "  append to running,",
    "  status RUNNING",
], col=VIOLET, align="start", mono_size=11.5)
row_arrows(by + 40, VIOLET)
d.text(XS[0], by + BH + 30, "Loop: while token_budget > 0 and the waiting queue is not empty. A preempted request is at the front of the FCFS queue.",
       12, MUTE)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 210
d.panel(y2, P3_H, 3, "What leaves schedule()?",
        "One SchedulerOutput for the executor. Only after it is built do the counters move forward.")
by = y2 + 70
BH = 112
d.box(XS[0], by, BW, BH, "SchedulerOutput", [
    "scheduled_new_reqs (+ resumed, MRV2)",
    "scheduled_cached_reqs (new blocks)",
    "num_scheduled_tokens {req: n}",
    "finished_req_ids, preempted_req_ids",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[1], by, BW, BH, "_update_after_schedule", [
    "num_computed_tokens += n",
    "is_prefill_chunk =",
    "  computed < num_tokens + placeh.",
    "clear finished / preempted ids",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], by, BW, BH, "AsyncScheduler (default)", [
    "not a prefill chunk:",
    "  num_output_placeholders += 1",
    "  (+ spec tokens)",
    "token arrives later: -1",
], col=EMERALD, align="start", mono_size=11.5)
row_arrows(by + 40, EMERALD)

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "running pass"), (VIOLET, "waiting pass"), (AMBER, "preemption"),
                 (EMERALD, "the plan of the step")])
y += 38
y = d.footnotes(30, y, [
    "Left out: encoder inputs, Mamba alignment, KV connector loads, pause states, DP prefill throttling, spec-decode padding.",
    "Source: vllm/v1/core/sched/scheduler.py:622-1643 and async_scheduler.py:25-50 at the pinned commit.",
])
d.save(height=y + 14)
