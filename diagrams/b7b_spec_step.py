"""One speculative decoding step in Model Runner V2: verify, accept/reject, draft again, bookkeeping.

Used by: src/concepts/speculative-decoding.html
Example: EAGLE, num_speculative_tokens k = 3, one request with 100 computed tokens; drafts d1 d2 accepted, d3 rejected.
Sources (at the pinned commit in explainer.json):
  vllm/config/vllm.py:645-650                              num_lookahead_tokens = k for EAGLE
  vllm/config/vllm.py:677-695                              uniform_decode_query_len = 1 + k
  vllm/v1/worker/gpu/model_runner.py:1663-1676             Sampler without drafts, RejectionSampler with drafts
  vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:416-479   RejectionSampler.__call__
  vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py:600-616, 695-697   accept rules
  vllm/v1/worker/gpu/input_batch.py:508-535, 613-617      num_rejected = num_logits - num_sampled; num_computed += query_len - num_rejected
  vllm/v1/worker/gpu/model_runner.py:2249-2289             postprocess_sampled, speculator.propose, draft_tokens on GPU
  vllm/v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py:165-370   EAGLE propose: draft prefill + k-1 decode steps
  vllm/v1/core/sched/async_scheduler.py:25-50              -1 placeholders, num_output_placeholders += 1 + k
  vllm/v1/core/sched/scheduler.py:2203-2221                num_accepted, rollback of num_computed_tokens
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One speculative decoding step (EAGLE, k = 3)",
         "vllm serve  ·  Model Runner V2  ·  async scheduling on  ·  one request: 100 tokens in the KV cache, last token x0 not yet computed",
         params="query length = 1 + k = 4  ·  logits rows = 4  ·  accepted 2 + 1 recovered = 3 tokens out  ·  rejected = 1")

BW = 300
X1, X2, X3 = 46, 400, 754
BH = 136

# ---------------------------------------------------------------- panel 1
y0 = 124
P_H = 236
d.panel(y0, P_H, 1, "What does the GPU verify in this step?",
        "The target model runs x0 and the 3 drafts in one forward pass; the rejection sampler checks each draft.")
R1 = y0 + 70
d.box(X1, R1, BW, BH, "Input of the target model", [
    "input_ids  [x0, d1, d2, d3]",
    "positions  [100, 101, 102, 103]",
    "drafts from req_states.draft_tokens",
    "  (written on the GPU last step)",
    "KV written at positions 100..103",
    "model_runner.py:1493-1504",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "Target logits: 4 rows", [
    "row 0 → next token after x0",
    "row 1 → after d1, row 2 → after d2",
    "row 3 → after d3 (bonus row)",
    "logits [4, vocab]",
    "sampling params applied first",
    "rejection_sampler.py:264-293",
], col=BLUE, align="start")
d.box(X3, R1, BW, BH, "RejectionSampler", [
    "d1: accept   d2: accept",
    "d3: reject → resample r from row 2",
    "  residual max(p - q, 0)",
    "sampled = [d1, d2, r]",
    "num_sampled 3, num_rejected 1",
    "rejection_sampler_utils.py:600-697",
], col=TEAL, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, BLUE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P_H + 22
d.panel(y1, P_H, 2, "What happens on the GPU right after sampling?",
        "The worker updates its own counters and the drafter proposes the next 3 drafts, with no wait for the CPU.")
R2 = y1 + 70
d.box(X1, R2, BW, BH, "post_update (worker)", [
    "num_computed += 4 - 1 = 3",
    "  100 → 103",
    "last_sampled_tokens = r",
    "all_token_ids += [d1, d2, r]",
    "position 103 holds stale d3 KV",
    "input_batch.py:579-617",
], col=AMBER, align="start")
d.box(X2, R2, BW, BH, "speculator.propose (EAGLE)", [
    "draft prefill: same batch shape",
    "  as the target, reuses its",
    "  attention metadata",
    "then k - 1 = 2 draft decode steps,",
    "  1 token each (FULL graph if eligible)",
    "target_dependent_ar/speculator.py",
], col=VIOLET, align="start")
d.box(X3, R2, BW, BH, "Next drafts stay on the GPU", [
    "draft_tokens[slot] = [e1, e2, e3]",
    "next input: [r, e1, e2, e3]",
    "  at positions 103..106",
    "r overwrites the d3 KV at 103",
    "AsyncOutput copy runs meanwhile",
    "model_runner.py:2209-2289",
], col=VIOLET, align="start")
d.arrow(X1 + BW, R2 + BH / 2, X2 - 2, R2 + BH / 2, AMBER)
d.arrow(X2 + BW, R2 + BH / 2, X3 - 2, R2 + BH / 2, VIOLET)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P_H + 22
d.panel(y2, P_H, 3, "What does the engine core do with it?",
        "It schedules the next step with -1 placeholders, then rolls back the one rejected draft when the output arrives.")
R3 = y2 + 70
d.box(X1, R3, BW, BH, "AsyncScheduler (before output)", [
    "num_computed 100 → 104",
    "placeholders += 1 + 3",
    "spec_token_ids = [-1, -1, -1]",
    "allocate_slots: 4 tokens",
    "  + 3 lookahead slots",
    "async_scheduler.py:25-50",
], col=EMERALD, align="start")
d.box(X2, R3, BW, BH, "update_from_output", [
    "generated = [d1, d2, r]",
    "num_accepted = 3 - 1 = 2",
    "num_rejected = 3 - 2 = 1",
    "num_computed 104 → 103",
    "placeholders -= 1 (then -= 3)",
    "scheduler.py:2203-2221",
], col=EMERALD, align="start")
d.box(X3, R3, BW, BH, "Output and metrics", [
    "one EngineCoreOutput, 3 tokens",
    "  → one ITL sample, one SSE chunk",
    "num_drafts +1",
    "num_draft_tokens +3",
    "num_accepted_tokens +2",
    "spec_decode/metrics.py:41-49",
], col=EMERALD, align="start")
d.arrow(X1 + BW, R3 + BH / 2, X2 - 2, R3 + BH / 2, EMERALD)
d.arrow(X2 + BW, R3 + BH / 2, X3 - 2, R3 + BH / 2, EMERALD)

y = y2 + P_H + 24
d.legend(30, y, [(BLUE, "target model"), (TEAL, "verification"), (AMBER, "worker state"),
                 (VIOLET, "drafter"), (EMERALD, "engine core")])
y += 40
y = d.footnotes(30, y, [
    "Panel 3 runs one step behind panels 1-2: with async scheduling, step N+1 is scheduled before the output of step N arrives.",
    "The order of the boxes in panel 3 is the order of the engine core calls, not of the GPU work.",
    "Token ids, positions and the accept/reject outcome are an example; the formulas and fields come from the cited lines.",
])
d.save(height=y + 14)
