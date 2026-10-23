"""One batch with three LoRA adapters: requests -> GPU slots -> token mapping -> one wrapped layer.

Used by: src/concepts/lora-serving.html
Example: --enable-lora --max-loras 3; adapters sql (id 1), chat-de (id 2), legal (id 3); one base-model request.
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/serve/engine/serving.py:194-200          model field -> LoRARequest
  vllm/v1/core/sched/scheduler.py:945-954, 998-1008         running adapters, max_loras admission
  vllm/lora/worker_manager.py:273-324                       _apply_adapters, add_adapter (disk -> CPU LRU), activate
  vllm/lora/model_manager.py:355-412, 1440-1452             activate_adapter: first free slot, LRU eviction
  vllm/v1/worker/gpu/lora_utils.py:94-104                   token_lora_mapping = lora_ids.repeat(num_scheduled_tokens)
  vllm/lora/punica_wrapper/utils.py:87-111                  LoRA id -> slot index, 0 -> -1
  vllm/lora/layers/base_linear.py:210-241                   base GEMM, then punica add_lora_linear
  vllm/lora/punica_wrapper/punica_gpu.py:206-260            shrink into an FP32 buffer, then expand
  vllm/lora/lora_weights.py:37-42                           alpha / r merged into lora_b
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Multi-LoRA serving: from the model field to the wrapped layer",
         "vllm serve --enable-lora --max-loras 3  ·  Model Runner V2  ·  one step with 4 requests and 7 tokens",
         params="requests: sql (id 1), chat-de (id 2), base (none), legal (id 3)  ·  GPU slots: 3  ·  rank r = 16")

BW = 300
X1, X2, X3 = 46, 400, 754
BH = 136

# ---------------------------------------------------------------- panel 1
y0 = 124
P_H = 236
d.panel(y0, P_H, 1, "Which adapter does each request use, and may it run?",
        "The API server maps the model name to a LoRARequest; the scheduler allows at most max_loras adapters.")
R1 = y0 + 70
d.box(X1, R1, BW, BH, "API server", [
    "model: \"sql\" → LoRARequest(",
    "  lora_name \"sql\", lora_int_id 1,",
    "  lora_path \"/adapters/sql\")",
    "base model name → None",
    "unknown name → 404 (or resolver)",
    "serve/engine/serving.py:194-200",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "Scheduler (engine core)", [
    "running adapters {1, 2, 3}",
    "  = max_loras (3)",
    "waiting request for id 4:",
    "  skipped this step",
    "id already running: admitted",
    "scheduler.py:945-954, 998-1008",
], col=VIOLET, align="start")
d.box(X3, R1, BW, BH, "Worker: _apply_adapters", [
    "for each adapter in the batch:",
    "not in CPU cache → load from",
    "  disk or HF Hub (max_cpu_loras)",
    "activate_adapter → GPU slot",
    "  (LRU evicts if all slots used)",
    "worker_manager.py:273-324",
], col=AMBER, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, VIOLET)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P_H + 22
d.panel(y1, P_H, 2, "How does each token find its adapter on the GPU?",
        "Each request gives its LoRA id to all its tokens; Punica turns ids into slot indices.")
R2 = y1 + 70
d.box(X1, R2, BW, BH, "GPU slots", [
    "lora_index_to_id = [1, 3, 2]",
    "  slot 0: sql      (id 1)",
    "  slot 1: legal    (id 3)",
    "  slot 2: chat-de  (id 2)",
    "stacked A/B buffers [3, 1, ...]",
    "model_manager.py:355-412",
], col=AMBER, align="start")
d.box(X2, R2, BW, BH, "token_lora_mapping (MRV2)", [
    "lora_ids per request: [1, 2, 0, 3]",
    "tokens per request:   [3, 1, 1, 2]",
    "repeat → (1,1,1, 2, 0, 3,3)",
    "prompt mapping: (1, 2, 0, 3)",
    "0 = no LoRA",
    "gpu/lora_utils.py:94-104",
], col=TEAL, align="start")
d.box(X3, R2, BW, BH, "Punica update_metadata", [
    "LoRA id → slot index:",
    "  (0,0,0, 2, -1, 1,1)",
    "-1 = no adapter for the token",
    "recomputed only when the mapping",
    "  or the slot layout changes",
    "punica_wrapper/utils.py:87-111",
], col=TEAL, align="start")
d.arrow(X1 + BW, R2 + BH / 2, X2 - 2, R2 + BH / 2, AMBER)
d.arrow(X2 + BW, R2 + BH / 2, X3 - 2, R2 + BH / 2, TEAL)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P_H + 22
d.panel(y2, P_H, 3, "What does one wrapped linear layer compute?",
        "One base GEMM for all 7 tokens, then a batched shrink and expand that pick A and B per token by slot.")
R3 = y2 + 70
d.box(X1, R3, BW, BH, "Base layer (unchanged)", [
    "y = quant_method.apply(x)",
    "x [7, 4096] → y [7, 4096]",
    "one GEMM, all tokens",
    "wrapper: e.g. QKV/Column/Row",
    "  ParallelLinearWithLoRA",
    "base_linear.py:210-213",
], col=EMERALD, align="start")
d.box(X2, R3, BW, BH, "add_shrink", [
    "buf = x @ A[slot of token]",
    "buf FP32 [slices, 7, r=16]",
    "slot -1: no LoRA update",
    "A stacked [3, 1, 16, 4096]",
    "  (max_loras, 1, r, in)",
    "punica_gpu.py:247-262",
], col=EMERALD, align="start")
d.box(X3, R3, BW, BH, "add_expand", [
    "y += buf @ B[slot of token]",
    "B stacked [3, 1, 4096, 16]",
    "  (max_loras, 1, out, r)",
    "scale 1.0: alpha / r is",
    "  merged into B at load",
    "lora_weights.py:37-42",
], col=EMERALD, align="start")
d.arrow(X1 + BW, R3 + BH / 2, X2 - 2, R3 + BH / 2, EMERALD)
d.arrow(X2 + BW, R3 + BH / 2, X3 - 2, R3 + BH / 2, EMERALD)

y = y2 + P_H + 24
d.legend(30, y, [(BLUE, "API server"), (VIOLET, "scheduler"), (AMBER, "adapter memory"),
                 (TEAL, "per-step mapping"), (EMERALD, "forward pass")])
y += 40
y = d.footnotes(30, y, [
    "Slot order depends on the order of activation; [1, 3, 2] is one possible layout. Shapes are for a 4096 x 4096 projection at TP = 1.",
    "Shrink and expand are Triton kernels (out of scope); the page explains only how vLLM calls them.",
])
d.save(height=y + 14)
