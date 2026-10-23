"""B6b: Model Runner V2 - persistent GPU state and the per-step inputs.

Used by: src/concepts/model-runner.html, src/code/v1-worker-gpu-model-runner.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/worker/gpu/states.py:28-85            RequestState: rows indexed by request slot
  vllm/v1/worker/gpu/states.py:99-102           add_request pops a free slot (last first)
  vllm/v1/worker/gpu/block_table.py:66-72       one block table per KV cache group [max_num_reqs, max_blocks]
  vllm/v1/worker/gpu/model_runner.py:2471-2483  sort_batch_req_ids: drafts, decodes, then by size
  vllm/v1/worker/gpu/model_runner.py:1392-1395  idx_mapping copied to the GPU
  vllm/v1/worker/gpu/model_runner.py:1446-1504  query_start_loc, prefill tokens, pos/seq_lens, combine kernel
  vllm/v1/worker/gpu/model_runner.py:1577-1597  prepare_attn: gather block tables, slot mappings
  vllm/v1/worker/gpu/block_table.py:409         slot = block_number * block_size + offset
  vllm/v1/worker/gpu/attn_utils.py:527-580      CommonAttentionMetadata per group, metadata per layer name
Example values are derived (block ids illustrative), block size 16.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Model Runner V2: fixed GPU rows, per-step inputs built on the GPU",
         "One step with three requests  ·  max_num_seqs = 1024  ·  block size 16  ·  no speculative decoding",
         params="R1: slot 1023, decode, 40 computed  ·  R2: slot 1022, decode, 17 computed  ·  R3: slot 1021, new 6-token prompt")

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 214
d.panel(y0, P1_H, 1, "What stays on the GPU between steps?",
        "One row per request slot in RequestState, and one block table per KV cache group, also by slot.")
BY = y0 + 66
ROWS = [("slot", "all_token_ids (UVA)", "computed", "last_sampled", "prefill_len"),
        ("1023", "R1: 16 prompt + 25 outputs", "40", "t(R1)", "16"),
        ("1022", "R2: 8 prompt + 10 outputs", "17", "t(R2)", "8"),
        ("1021", "R3: 6 prompt tokens", "0", "-", "6")]
FMT = "{:<6} {:<27} {:<9} {:<13} {}"
d.box(46, BY, 640, 132, "RequestState  (rows = request slots 0 .. 1023)", [
    *[FMT.format(*r) for r in ROWS],
    "0..1020   free_indices (pop() takes the last free slot)",
], col=BLUE, align="start", mono_size=11.5)
d.box(706, BY, 348, 132, "BlockTables, group 0", [
    "int32 [1024, max_num_blocks]",
    "1023   [12, 7, 31]",
    "1022   [4, 22]",
    "1021   [9]",
    "rows written by staged writes",
], col=BLUE, align="start", mono_size=11.5)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 236
d.panel(y1, P2_H, 2, "What does one step build?",
        "The CPU sorts and sends two small arrays; GPU kernels gather everything else from the slot rows.")
BY = y1 + 70
BW = 300
XS = [46, 400, 754]
d.box(XS[0], BY, BW, 140, "CPU: plan and sort", [
    "num_scheduled_tokens",
    "  {R3: 6, R1: 1, R2: 1}",
    "sort: decodes, then by size",
    "  -> [R1, R2, R3]",
    "idx_mapping  [1023, 1022, 1021]",
    "query_start_loc [0, 1, 2, 8]",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 140, "GPU kernels", [
    "prepare_prefill_inputs",
    "  R3 tokens from all_token_ids",
    "prepare_pos_seq_lens",
    "combine_sampled_and_draft_",
    "  tokens: R1, R2 input from",
    "  last_sampled_tokens",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 140, "InputBatch (step tensors)", [
    "input_ids [t(R1), t(R2), p0..p5]",
    "positions [40, 17, 0, 1, .., 5]",
    "seq_lens  [41, 18, 6]",
    "logits_indices [0, 1, 7]",
    "num_tokens 8 (PIECEWISE, 8)",
], col=EMERALD, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 50, XS[1] - 2, BY + 50, AMBER, sw=2)
d.arrow_label((XS[0] + BW + XS[1]) / 2, BY + 50, "H2D", AMBER)
d.arrow(XS[1] + BW, BY + 50, XS[2] - 2, BY + 50, EMERALD)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 196
d.panel(y2, P3_H, 3, "What goes to the attention layers?",
        "prepare_attn gathers block tables and slot mappings per group; the model state builds the metadata.")
BY = y2 + 70
d.box(XS[0], BY, BW, 100, "gather_block_tables", [
    "batch order, group 0:",
    "row 0 [12, 7, 31]   (R1)",
    "row 1 [4, 22]       (R2)",
    "row 2 [9]           (R3)",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 100, "compute_slot_mappings", [
    "slot = block * 16 + offset",
    "R1 pos 40 -> 31*16+8 = 504",
    "R2 pos 17 -> 22*16+1 = 353",
    "R3 pos 0..5 -> 144 .. 149",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 100, "model_state.prepare_attn", [
    "CommonAttentionMetadata",
    "  one per KV cache group",
    "builder.build() -> metadata",
    "attn_metadata[layer name]",
], col=EMERALD, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 50, XS[1] - 2, BY + 50, EMERALD)
d.arrow(XS[1] + BW, BY + 50, XS[2] - 2, BY + 50, EMERALD)

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "persistent GPU state"), (VIOLET, "CPU work"), (EMERALD, "per-step GPU tensors and kernels"),
                 (AMBER, "CPU-to-GPU copy each step")])
y += 40
y = d.footnotes(30, y, [
    "Derived example. Block ids are illustrative. t(R1) = the token that R1 sampled in the previous step, still only on the GPU.",
    "Not shown: prompt_len, total_len, draft_tokens, the UVA buffers, multimodal inputs, LoRA and pipeline parallelism.",
])
d.save(height=y + 14)
