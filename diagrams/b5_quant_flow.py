"""From a checkpoint to per-layer quantize methods: where each QuantizationConfig hook runs.

Used by: src/concepts/quantization-plugins.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/config/model.py:1383-1470          _verify_quantization: checkpoint quant_method, override loop, checks
  vllm/config/model.py:1396-1426          overrides list; custom methods are tried before the built-in overrides
  vllm/config/vllm.py:1586-1589, 898-932  VllmConfig.__post_init__ -> _get_quantization_config (capability, act dtypes)
  vllm/model_executor/model_loader/weight_utils.py:197-400  get_quant_config -> quant_cls.from_config(...)
  vllm/v1/engine/utils.py:256-287         the VllmConfig (with quant_config) goes to the child processes pickled
  vllm/model_executor/models/llama.py:370 the model reads vllm_config.quant_config
  vllm/model_executor/layers/linear.py:291-296, 379-387, 420   resolve_quant_method, create_weights, apply
  vllm/model_executor/layers/quantization/base_config.py:295-356  resolve_quant_method -> get_quant_method
  vllm/model_executor/model_loader/utils.py:157-186   process_weights_after_loading for every quant_method
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("From a checkpoint to quantized layers",
         "vllm serve  ·  one QuantizationConfig object per model  ·  one quantize method per layer",
         params="example: an AWQ checkpoint, quant_method \"awq\"  ->  auto_awq")

LX = [40, 390, 740]
LW = 320
FE, WK, STEP = BLUE, VIOLET, EMERALD

# ---- Panel 1: front end ------------------------------------------------------------------------
y1 = 120
BH = 112
P1_H = 70 + BH + 22
d.panel(y1, P1_H, 1, "Which method, and which settings? (front end)",
        "ModelConfig picks the method name; VllmConfig makes one config object from the checkpoint.")
r = y1 + 70
d.step_card(LX[0], r, LW, BH, 1, "Read the checkpoint", FE, tag="model.py:1389", rows=[
    ("cfg", "config.json quantization_config"),
    ("", "{\"quant_method\": \"awq\", ...}"),
    ("flag", "--quantization (optional)"),
], key_w=40)
d.step_card(LX[1], r, LW, BH, 2, "_verify_quantization", FE, tag="model.py:1383", rows=[
    ("loop", "override_quantization_method"),
    ("", "custom first, then built-in"),
    ("name", "\"auto_awq\"; platform check"),
], key_w=40)
d.step_card(LX[2], r, LW, BH, 3, "_get_quantization_config", FE, tag="vllm.py:898", rows=[
    ("make", "cls.from_config(hf cfg)"),
    ("check", "get_min_capability()"),
    ("", "get_supported_act_dtypes()"),
], key_w=46)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r + 56, LX[i + 1] - 2, r + 56, FE)

# ---- Panel 2: worker build ---------------------------------------------------------------------
y2 = y1 + P1_H + 18
P2_H = 70 + BH + 22
d.panel(y2, P2_H, 2, "How does each layer get its method? (worker, model build)",
        "The config object arrives inside the pickled VllmConfig. Each layer asks it once.")
r = y2 + 70
d.step_card(LX[0], r, LW, BH, 4, "Model __init__", WK, tag="llama.py:370", rows=[
    ("get", "vllm_config.quant_config"),
    ("pass", "quant_config, prefix to"),
    ("", "every linear layer"),
], key_w=40)
d.step_card(LX[1], r, LW, BH, 5, "LinearBase.__init__", WK, tag="linear.py:293", rows=[
    ("ask", "get_quant_method(layer, prefix)"),
    ("", "e.g. \"...0.self_attn.qkv_proj\""),
    ("None", "ValueError for a linear"),
], key_w=40)
d.step_card(LX[2], r, LW, BH, 6, "quant_method", WK, tag="linear.py:379", rows=[
    ("make", "create_weights(layer, ...)"),
    ("load", "checkpoint tensors"),
    ("post", "process_weights_after_loading"),
], key_w=40)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r + 56, LX[i + 1] - 2, r + 56, WK)

# ---- Panel 3: each step ------------------------------------------------------------------------
y3 = y2 + P2_H + 18
SH = 80
P3_H = 70 + SH + 22
d.panel(y3, P3_H, 3, "What runs in each step?",
        "The layer's forward() calls apply(); the method chooses the kernel.")
r = y3 + 70
d.box(LX[0], r, LW, SH, "layer.forward(x)", ["linear.py:414-420"], col=STEP)
d.box(LX[1], r, LW, SH, "quant_method.apply(layer, x, bias)", ["base_config.py:52-57"], col=STEP)
d.box(LX[2], r, LW, SH, "kernel of the method", ["the GEMM kernel that the method", "selects (not described here)"], col=STEP)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r + 40, LX[i + 1] - 2, r + 40, STEP)

y = y3 + P3_H + 16
d.callout(30, y, 1040, "Plugin:", "@register_quantization_config(\"my_quant\") must run in the front end; "
          "a worker only needs the module importable.", AMBER, h=34)
y += 50
d.legend(30, y, [(FE, "front end (config)"), (WK, "worker (model build)"), (STEP, "worker (every step)")])
y += 34
y = d.footnotes(30, y, [
    "Not shown: online quantization (fp8_per_tensor ...), the KV cache scale method of attention layers,",
    "  and MoE layers, which ask get_quant_method the same way. Kernel internals are out of scope.",
])
d.save(height=y + 8)
