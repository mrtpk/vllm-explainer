"""The dispatch decision of one CustomOp object: class swap, enabled check, platform method.

Used by: src/concepts/custom-ops.html, src/code/model-executor-custom-op.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/model_executor/custom_op.py:107-126   __new__: op_registry_oot keyed by class __name__ -> swap class
  vllm/model_executor/custom_op.py:128-134   __init__ binds _forward_method once; forward() calls it
  vllm/model_executor/custom_op.py:172-205   dispatch_forward
  vllm/model_executor/custom_op.py:268-307   enabled() and default_on()
  vllm/model_executor/custom_op.py:147-170   default bodies: hip -> cuda; xpu/cpu/tpu/oot -> native
  vllm/config/vllm.py:1819-1826              base mode: "none" with Inductor + compile, else "all"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("How one CustomOp picks its forward method",
         "Runs once, when the model builds the layer  ·  forward() then calls the bound method",
         params="vllm/model_executor/custom_op.py  ·  compilation_config.custom_ops")

LX = [40, 390, 740]
LW = 320

# ---- Panel 1: construction ------------------------------------------------------------------
y1 = 120
P1_H = 190
d.panel(y1, P1_H, 1, "Which class is built?",
        "__new__ looks up the class name in op_registry_oot. A registered out-of-tree class replaces it.")
by = y1 + 70
BH = 100
d.box(LX[0], by, LW, BH, "Model code", [
    "RotaryEmbedding(16, 16, ...)",
    "inside set_current_vllm_config",
    "(model __init__ in the worker)",
], col=BLUE, align="start")
d.box(LX[1], by, LW, BH, "CustomOp.__new__  :107-126", [
    "key = cls.__name__",
    "\"RotaryEmbedding\" in",
    "  op_registry_oot ?",
], col=PINK, align="start")
d.box(LX[2], by, LW, BH, "__init__  :128-131", [
    "yes: DummyRotaryEmbedding",
    "no : RotaryEmbedding",
    "-> dispatch_forward()",
], col=BLUE, align="start")
for i in range(2):
    d.arrow(LX[i] + LW + 2, by + 50, LX[i + 1] - 2, by + 50, INK)

# ---- Panel 2: enabled? ----------------------------------------------------------------------
y2 = y1 + P1_H + 18
P2_H = 190
d.panel(y2, P2_H, 2, "Is the op enabled?",
        "custom_ops holds one base mode plus +name / -name switches. The name is cls.name.")
cy = y2 + 70
CH = 100
d.step_card(LX[0], cy, LW, CH, 1, "Base mode", VIOLET, tag=":1819-1826", rows=[
    ("none", "Inductor + compile mode"),
    ("", "(default for vllm serve, O2)"),
    ("all", "otherwise, e.g. --enforce-eager"),
], key_w=40)
d.step_card(LX[1], cy, LW, CH, 2, "Switches", VIOLET, tag=":282-290", rows=[
    ("+x", "turns op x on"),
    ("-x", "turns op x off (wins)"),
    ("both", "ValueError"),
], key_w=40)
d.step_card(LX[2], cy, LW, CH, 3, "Result", VIOLET, tag="enabled()", rows=[
    ("rule", "(base on or +x) and not -x"),
    ("or", "enforce_enable=True"),
    ("log", "enabled/disabled_custom_ops"),
], key_w=40)

# ---- Panel 3: method -------------------------------------------------------------------------
y3 = y2 + P2_H + 18
P3_H = 196
d.panel(y3, P3_H, 3, "Which method does forward() call?",
        "Disabled: forward_native, maybe compiled. Enabled: the method of current_platform.")
ey = y3 + 70
EH = 106
d.box(LX[0], ey, LW, EH, "Disabled", [
    "maybe_compile(forward_native,",
    "  enable=compile_native)",
    "plain forward_native unless",
    "compile_native=True  :189-192",
], col=AMBER, align="start")
d.box(LX[1], ey, LW, EH, "Enabled: platform order", [
    "rocm -> forward_hip",
    "cpu -> forward_cpu   tpu -> _tpu",
    "xpu -> forward_xpu",
    "OOT -> forward_oot  else _cuda",
], col=EMERALD, align="start")
d.box(LX[2], ey, LW, EH, "Default bodies  :147-170", [
    "forward_hip -> forward_cuda",
    "forward_xpu/cpu/tpu/oot",
    "  -> forward_native",
    "forward_cuda/native: subclass",
], col=EMERALD, align="start")
d.arrow(LX[1] + LW + 2, ey + 50, LX[2] - 2, ey + 50, INK, dash="5 4")
d.arrow_label(LX[1] + LW + 15, ey + 44, "if not", INK)

y = y3 + P3_H + 22
d.legend(30, y, [(BLUE, "model build"), (PINK, "out-of-tree swap"), (VIOLET, "enable switch"),
                 (AMBER, "disabled path"), (EMERALD, "enabled path")])
y += 34
y = d.footnotes(30, y, [
    "The choice is made once per object, in __init__ (:131). Changing custom_ops later has no effect on built layers.",
    "\"if not\": a subclass that does not override the platform method gets the default body on the right.",
])
d.save(height=y + 8)
