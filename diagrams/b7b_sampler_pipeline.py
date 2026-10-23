"""The Model Runner V2 sampler pipeline, and where custom logits processors plug in.

Used by: src/concepts/sampling-and-logits-processors.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/worker/gpu/model_runner.py:1615-1676              compute_logits, grammar bitmask, Sampler / RejectionSampler
  vllm/v1/worker/gpu/sample/sampler.py:64-76                 built-in processors in pipeline order, then custom ones
  vllm/v1/worker/gpu/sample/sampler.py:226-276               apply_sampling_params: FP32 copy, processors, thinking budget, temperature, min_p
  vllm/v1/worker/gpu/sample/sampler.py:278-360               fused FlashInfer sampler vs top-k/top-p + Gumbel
  vllm/v1/worker/gpu/sample/states.py:41-72                  needs_logits_processing rule
  vllm/v1/worker/gpu/sample/gumbel.py:171-212                Gumbel-max with (seed, pos) noise; argmax at temperature 0
  vllm/v1/worker/gpu/sample/logits_processor/interface.py:92-158   custom processor interface
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The sampler in Model Runner V2",
         "One call of Sampler.__call__ per step  ·  rows = one logits row per sampled position  ·  no speculative decoding",
         params="logits: model dtype [rows, vocab] → FP32 copy only if a request needs processing  ·  sampled: int64 [rows, 1]")

BW = 300
X1, X2, X3 = 46, 400, 754
BH = 136

# ---------------------------------------------------------------- panel 1
y0 = 124
P_H = 236
d.panel(y0, P_H, 1, "What happens before the sampler?",
        "The model runner computes logits for the sampled rows only, then applies the grammar mask.")
R1 = y0 + 70
d.box(X1, R1, BW, BH, "compute_logits", [
    "hidden_states[logits_indices]",
    "  → logits [rows, vocab]",
    "rows = last token of each request",
    "  (1 + k rows with drafts)",
    "model dtype, e.g. BF16",
    "model_runner.py:1637-1638",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "Grammar bitmask (optional)", [
    "structured-output rows only",
    "-inf where the bit is 0",
    "in place, before any processor",
    "custom processors must not",
    "  raise a masked logit",
    "model_runner.py:1643-1651",
], col=BLUE, align="start")
d.box(X3, R1, BW, BH, "Needs processing?", [
    "per request, set in add_request:",
    "T != 0 and (T != 1 or min_p,",
    "  top_k or top_p set),",
    "  or a stage's add_request says True",
    "none in batch → raw logits",
    "states.py:67-72, sampler.py:238",
], col=AMBER, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, BLUE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P_H + 22
d.panel(y1, P_H, 2, "In which order do the logits change?",
        "Built-in processors first, then your processors, then forcing, temperature and min_p, all on the FP32 copy.")
R2 = y1 + 70
d.box(X1, R2, BW, BH, "Built-in processors", [
    "1 LogitBiasState: allowed ids,",
    "  logit_bias, min_tokens",
    "2 PenaltiesState: repetition,",
    "  frequency, presence",
    "3 BadWordsState",
    "sampler.py:69-76",
], col=TEAL, align="start")
d.box(X2, R2, BW, BH, "Custom processors (yours)", [
    "4 vllm.logits_processors group",
    "  then --logits-processors",
    "apply(logits, ctx) on the batch",
    "add_request(slot, params)",
    "apply_staged_writes() per step",
    "logits_processor/interface.py",
], col=VIOLET, align="start")
d.box(X3, R2, BW, BH, "Forcing and scaling", [
    "5 thinking budget forcing",
    "  (wins over custom edits)",
    "6 temperature: logits / T",
    "7 min_p",
    "top-k / top-p not yet",
    "sampler.py:258-268",
], col=TEAL, align="start")
d.arrow(X1 + BW, R2 + BH / 2, X2 - 2, R2 + BH / 2, TEAL)
d.arrow(X2 + BW, R2 + BH / 2, X3 - 2, R2 + BH / 2, VIOLET)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P_H + 22
d.panel(y2, P_H, 3, "How is the token drawn?",
        "A fused FlashInfer sampler when the whole batch allows it; else top-k/top-p masking and the Gumbel sampler.")
R3 = y2 + 70
d.box(X1, R3, BW, BH, "Fused path (FlashInfer)", [
    "batch uses top-k or top-p",
    "no greedy request",
    "no request with a seed",
    "no processed logprobs",
    "and the GPU supports it",
    "sampler.py:304-343",
], col=EMERALD, align="start")
d.box(X2, R3, BW, BH, "Fallback: mask + Gumbel", [
    "apply_top_k_top_p (native)",
    "gumbel_sample: argmax of",
    "  logits / T + noise(seed, pos)",
    "temperature 0 → plain argmax",
    "unseeded: random seed per request",
    "sampler.py:346-359, states.py:54-58",
], col=EMERALD, align="start")
d.box(X3, R3, BW, BH, "SamplerOutput", [
    "sampled_token_ids [rows, 1]",
    "logprobs: from raw logits (default)",
    "  or processed logits",
    "num_sampled 0 for a prefill chunk",
    "stays on the GPU; copied async",
    "sampler.py:171-224",
], col=AMBER, align="start")
d.arrow(X1 + BW, R3 + BH / 2, X2 - 2, R3 + BH / 2, EMERALD, dash="5 4")
d.arrow(X2 + BW, R3 + BH / 2, X3 - 2, R3 + BH / 2, EMERALD)

y = y2 + P_H + 24
d.legend(30, y, [(BLUE, "model runner"), (AMBER, "decision / output"), (TEAL, "built-in stage"),
                 (VIOLET, "your plugin"), (EMERALD, "token draw")])
y += 40
y = d.footnotes(30, y, [
    "The dashed arrow means \"or\": a batch takes exactly one of the two paths in panel 3.",
    "With drafts, the RejectionSampler calls the same apply_sampling_params (with top-k/top-p) and then its own test.",
    "Watermarking replaces the Sampler with a subclass (GPUWatermarkSampler); it is not shown.",
])
d.save(height=y + 14)
