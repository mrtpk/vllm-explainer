"""Structured outputs end to end: request fields → backend choice → grammar compile → per-step bitmask → masked sampling.

Used by: src/concepts/structured-outputs.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/generate/base/protocol.py:181-211       response_format → StructuredOutputsParams
  vllm/sampling_params.py:1221-1445                        _validate_structured_outputs: backend auto → xgrammar, else guidance / outlines
  vllm/v1/engine/core.py:1114-1136                         input thread: Request(...), grammar_init
  vllm/v1/request.py:90-117                                WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR
  vllm/v1/structured_output/__init__.py:79-96, 133-228     thread pools, backend init, compile in executor
  vllm/v1/core/sched/scheduler.py:3291-3303                grammar ready → WAITING; error → FINISHED_ERROR later
  vllm/v1/core/sched/scheduler.py:2086-2115                get_grammar_bitmask → GrammarOutput
  vllm/v1/structured_output/__init__.py:382-531            bitmask rows, constraint bounds (reasoning)
  vllm/v1/engine/core.py:733-810                           step_with_batch_queue: sample_tokens(grammar_output), deferral
  vllm/v1/worker/gpu/model_runner.py:1615-1665             logits → apply_grammar_bitmask → sampler
  vllm/v1/worker/gpu/structured_outputs.py:52-171          int32 [rows, ceil(V/32)] bitmask; -inf where bit = 0
  vllm/v1/structured_output/__init__.py:533-553            accept_tokens after the step
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, MUTE, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Structured outputs: from a JSON schema to masked logits",
         "vllm serve  ·  backend auto → xgrammar  ·  one GPU  ·  no speculative decoding  ·  async scheduling on",
         params="bitmask: int32 [num_rows, ceil(vocab / 32)]  ·  vocab 151,936 → 4,748 words = 18,992 bytes per row")

BW = 300
X1, X2, X3 = 46, 400, 754

# ---------------------------------------------------------------- panel 1
y0 = 124
P1_H = 236
d.panel(y0, P1_H, 1, "What does the API server do with the constraint?",
        "It turns the request fields into StructuredOutputsParams and picks a backend for this request.")
R1 = y0 + 70
BH = 136
d.box(X1, R1, BW, BH, "Request fields", [
    "response_format: json_schema",
    "  or json_object",
    "structured_outputs: json, regex,",
    "  choice, grammar, structural_tag",
    "tool_choice required / named",
    "  (set by the tool parser)",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "StructuredOutputsParams", [
    "exactly one constraint",
    "inside SamplingParams",
    "validated in InputProcessor",
    "  (AsyncLLM.add_request)",
    "bad schema → 400",
    "sampling_params.py:102-142",
], col=BLUE, align="start")
d.box(X3, R1, BW, BH, "Backend choice (auto)", [
    "xgrammar validates? → xgrammar",
    "else guidance (default fallback)",
    "else outlines (non-Tekken Mistral",
    "  or guidance-unsupported schema)",
    "stored in params._backend",
    "sampling_params.py:1390-1441",
], col=BLUE, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, BLUE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 236
d.panel(y1, P2_H, 2, "What happens before the first step of the request?",
        "The engine core compiles the grammar in a thread pool; the request waits until it is ready.")
R2 = y1 + 70
d.box(X1, R2, BW, BH, "Input thread: grammar_init", [
    "Request(...) with",
    "  structured_output_request",
    "status WAITING_FOR_",
    "  STRUCTURED_OUTPUT_GRAMMAR",
    "first request builds the backend",
    "  (one backend per engine)",
], col=VIOLET, align="start")
d.box(X2, R2, BW, BH, "Compile (ThreadPoolExecutor)", [
    "workers = (CPUs + 1) // 2",
    "xgrammar GrammarCompiler",
    "  cache: VLLM_XGRAMMAR_CACHE_MB 512",
    "→ GrammarMatcher (an FSM)",
    "Future stored on the request",
    "__init__.py:95-96, 195-204",
], col=VIOLET, align="start")
d.box(X3, R2, BW, BH, "Scheduler waiting pass", [
    "Future not done → skip request",
    "done → status WAITING",
    "  → normal admission",
    "exception → FINISHED_ERROR",
    "  (400 validation_error)",
    "scheduler.py:3291-3303",
], col=VIOLET, align="start")
d.arrow(X1 + BW, R2 + BH / 2, X2 - 2, R2 + BH / 2, VIOLET)
d.arrow(X2 + BW, R2 + BH / 2, X3 - 2, R2 + BH / 2, VIOLET)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 236
d.panel(y2, P3_H, 3, "What happens in every decode step?",
        "The engine core fills one bitmask row per request; the worker masks the logits before sampling.")
R3 = y2 + 70
d.box(X1, R3, BW, BH, "get_grammar_bitmask (engine core)", [
    "one row per structured request",
    "  (no prefill chunks)",
    "reasoning not ended → row = all 1",
    "else matcher.fill_next_token_bitmask",
    "→ GrammarOutput (numpy)",
    "scheduler.py:2086-2115",
], col=TEAL, align="start")
d.box(X2, R3, BW, BH, "sample_tokens (worker, GPU)", [
    "logits = compute_logits(hidden)",
    "copy bitmask to GPU (side stream)",
    "kernel: logit = -inf where bit 0",
    "sampler → token id",
    "model_runner.py:1638-1665",
    "structured_outputs.py:70-171",
], col=AMBER, align="start")
d.box(X3, R3, BW, BH, "update_from_output (engine core)", [
    "accept_tokens(new ids)",
    "  advances the FSM",
    "</think> seen → reasoning_ended",
    "  latched, grammar starts",
    "FSM terminated → full rows",
    "__init__.py:533-553",
], col=EMERALD, align="start")
d.arrow(X1 + BW, R3 + BH / 2, X2 - 2, R3 + BH / 2, TEAL)
d.arrow(X2 + BW, R3 + BH / 2, X3 - 2, R3 + BH / 2, AMBER)

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "API server"), (VIOLET, "engine core, once"), (TEAL, "engine core, each step"),
                 (AMBER, "worker / GPU"), (EMERALD, "after the step")])
y += 40
y = d.footnotes(30, y, [
    "Vocab 151,936 is the Qwen2.5 value (inferred from its config, not in the repo); the bitmask width follows cdiv(vocab, 32).",
    "Async scheduling: when a structured request in decode has its last token still in flight, sample_tokens of the whole step waits",
    "  until the previous step is processed (pending_structured_output_tokens, core.py:737-749, 789-810).",
])
d.save(height=y + 14)
