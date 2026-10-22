"""How one process picks its platform: the detection flow of resolve_current_platform_cls_qualname().

Used by: src/concepts/platform-plugins.html, src/code/platforms-init.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/platforms/__init__.py:303-322   module __getattr__: lazy, once per process (:317-321)
  vllm/platforms/__init__.py:244-248   VLLM_TARGET_DEVICE == "cpu" short-circuit (no plugin group)
  vllm/platforms/__init__.py:250       load_plugins_by_group(vllm.platform_plugins), filtered by VLLM_PLUGINS
  vllm/platforms/__init__.py:231-237   built-in detectors: tpu, cuda, rocm, xpu, cpu
  vllm/platforms/__init__.py:254-265   every detector runs; an exception -> DEBUG log, not active
  vllm/platforms/__init__.py:272-279   >= 2 out-of-tree -> RuntimeError; exactly 1 -> it wins (called again)
  vllm/platforms/__init__.py:280-289   >= 2 built-in -> RuntimeError; exactly 1 -> it wins (called again)
  vllm/platforms/__init__.py:290-292   none -> vllm.platforms.interface.UnspecifiedPlatform
  vllm/platforms/__init__.py:319       resolve_obj_by_qualname(name)() -> the one Platform object
  vllm/envs.py:622                     VLLM_TARGET_DEVICE default "cuda"
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("How a process picks its platform",
         "resolve_current_platform_cls_qualname()  ·  runs once per process, on the first read of current_platform",
         params="vllm/platforms/__init__.py:240-293  ·  VLLM_TARGET_DEVICE default \"cuda\" (envs.py:622)")

LX = [40, 390, 740]
LW = 320

# ---- Panel 1: the three stages --------------------------------------------------------------
y1 = 120
P1_H = 220
d.panel(y1, P1_H, 1, "What runs on the first read of current_platform?",
        "A CPU target skips all detection. Otherwise every detector runs, built-in ones first.")
cy = y1 + 72
CH = 122
d.step_card(LX[0], cy, LW, CH, 1, "Lazy trigger", BLUE, tag="__getattr__", rows=[
    ("code", "from vllm.platforms import"),
    ("", "    current_platform"),
    ("once", "_current_platform is None?"),
    ("line", ":303-319"),
], key_w=44)
d.step_card(LX[1], cy, LW, CH, 2, "CPU target?", AMBER, tag="short-circuit", rows=[
    ("if", "VLLM_TARGET_DEVICE == \"cpu\""),
    ("then", "cpu_platform_plugin()"),
    ("", "no plugin group is loaded"),
    ("line", ":244-248"),
], key_w=44)
d.step_card(LX[2], cy, LW, CH, 3, "Run every detector", EMERALD, tag="load + call", rows=[
    ("load", "vllm.platform_plugins"),
    ("", "(filtered by VLLM_PLUGINS)"),
    ("run", "tpu, cuda, rocm, xpu, cpu, OOT"),
    ("error", "DEBUG log, not active"),
    ("line", ":250, :254-265"),
], key_w=44)
for i in range(2):
    d.arrow(LX[i] + LW + 2, cy + 40, LX[i + 1] - 2, cy + 40, INK)
d.arrow_label(LX[1] + LW + 15, cy + 40, "no", INK)

# ---- Panel 2: the decision ------------------------------------------------------------------
y2 = y1 + P1_H + 18
P2_H = 190
d.panel(y2, P2_H, 2, "Which detector wins?",
        "The code checks the out-of-tree count first. One out-of-tree platform beats any number of built-in ones.")
by = y2 + 72
BH = 100
d.box(LX[0], by, LW, BH, "Out-of-tree detectors", [
    ">= 2 active  ->  RuntimeError",
    "exactly 1    ->  call it again,",
    "                 it wins (INFO log)",
    ":272-279",
], col=PINK, align="start")
d.box(LX[1], by, LW, BH, "Then: built-in detectors", [
    ">= 2 active  ->  RuntimeError",
    "exactly 1    ->  call it again,",
    "                 it wins (DEBUG log)",
    ":280-289",
], col=EMERALD, align="start")
d.box(LX[2], by, LW, BH, "None active", [
    "UnspecifiedPlatform",
    "device_type = \"\"",
    "(DEBUG log only)",
    ":290-292",
], col=VIOLET, align="start")
for i in range(2):
    d.arrow(LX[i] + LW + 2, by + BH / 2, LX[i + 1] - 2, by + BH / 2, INK)
    d.arrow_label(LX[i] + LW + 15, by + BH / 2, "0", INK)

# ---- Panel 3: three worked examples ---------------------------------------------------------
y3 = y2 + P2_H + 18
P3_H = 156
d.panel(y3, P3_H, 3, "Three machines",
        "acme_hw is a made-up out-of-tree platform plugin. The result is a dotted class name; line 319 makes the object.")
ey = y3 + 72
EH = 66
d.box(LX[0], ey, LW, EH, "NVIDIA GPU host", [
    "active: cuda",
    "-> vllm.platforms.cuda.CudaPlatform",
], col=EMERALD, align="start")
d.box(LX[1], ey, LW, EH, "GPU host + acme_hw installed", [
    "active: cuda, acme_hw",
    "-> the acme_hw class (OOT wins)",
], col=PINK, align="start")
d.box(LX[2], ey, LW, EH, "Same host, VLLM_TARGET_DEVICE=cpu", [
    "no detector loop at all",
    "-> vllm.platforms.cpu.CpuPlatform",
], col=AMBER, align="start")

y = y3 + P3_H + 22
d.legend(30, y, [(BLUE, "lazy trigger"), (AMBER, "CPU short-circuit"), (EMERALD, "built-in"),
                 (PINK, "out-of-tree"), (VIOLET, "fallback")])
y += 34
y = d.footnotes(30, y, [
    "On an AMD Zen CPU with zentorch installed, cpu_platform_plugin() returns ZenCpuPlatform instead (:205-212).",
    "Every process (front end, engine core, each worker) runs this flow itself. It gets the same answer when its environment is the same.",
])
d.save(height=y + 8)
