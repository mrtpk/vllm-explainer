"""Where the config-driven hooks plug into the engine core and the worker.

Used by: src/concepts/engine-hooks.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/v1/engine/core.py:132, 147, 160, 164      general plugins, executor, KV caches, get_scheduler_cls()
  vllm/config/scheduler.py:231-256               scheduler_cls resolution
  vllm/v1/worker/worker_base.py:284, 292-331     init_worker: general plugins, worker_cls, worker_extension_cls
  vllm/v1/worker/worker_base.py:356-363          init_device: init_worker_extension hook
  vllm/v1/worker/gpu/model_runner.py:398-406     get_model_loader(load_config) -> load_model
  vllm/v1/worker/gpu/model_runner.py:463-466     build_custom_logits_processors (vllm.logits_processors + logits_processors=)
  vllm/v1/worker/gpu_worker.py:250-258           SleepModeBackendFactory.create_backend on first use
  vllm/v1/serial_utils.py:485-508                run_method: collective_rpc by method name
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Engine hooks: where each one plugs in",
         "Config fields, registries and one plugin group, read in the engine core and the workers",
         params="violet = config field (class name)  ·  blue = registry  ·  green = entry-point group")

LX = [40, 390, 740]
LW = 320
CFG, REG, GRP = VIOLET, BLUE, EMERALD
GRAY = ("#4b5563", "#f3f4f6", "#fafafa")
CH = 104

# ---- Panel 1: engine core --------------------------------------------------------------------------
y1 = 120
P1_H = 70 + CH + 22
d.panel(y1, P1_H, 1, "Engine core process: EngineCore.__init__",
        "The scheduler class is picked last, after the workers exist and the KV cache is planned.")
r = y1 + 70
d.step_card(LX[0], r, LW, CH, 1, "general plugins", GRP, tag="core.py:132", rows=[
    ("run", "load_general_plugins()"),
    ("so", "registries are filled here"),
    ("", "before anything reads them"),
], key_w=30)
d.step_card(LX[1], r, LW, CH, 2, "executor + KV caches", GRAY, tag="core.py:147,160", rows=[
    ("start", "workers (panel 2)"),
    ("plan", "KV cache config"),
    ("", ""),
], key_w=42)
d.step_card(LX[2], r, LW, CH, 3, "scheduler_cls", CFG, tag="core.py:164", rows=[
    ("unset", "AsyncScheduler / Scheduler"),
    ("str", "resolve_obj_by_qualname"),
    ("warn", "not under vllm.: WARNING"),
], key_w=42)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r + 52, LX[i + 1] - 2, r + 52, INK)

# ---- Panel 2: worker start-up ------------------------------------------------------------------------
y2 = y1 + P1_H + 18
P2_H = 70 + CH + 18 + CH + 22
d.panel(y2, P2_H, 2, "Worker: init_worker, init_device, load_model",
        "The worker class and its extension are fixed first; the loader and logits processors come with the model.")
r1 = y2 + 70
r2 = r1 + CH + 18
d.step_card(LX[0], r1, LW, CH, 4, "general plugins", GRP, tag="worker_base.py:284", rows=[
    ("run", "load_general_plugins()"),
    ("then", "register_triton_kernel_"),
    ("", "  overrides()"),
], key_w=34)
d.step_card(LX[1], r1, LW, CH, 5, "worker_cls", CFG, tag="worker_base.py:292", rows=[
    ("str", "dotted name, must be a str"),
    ("auto", "platform sets it earlier"),
    ("", "(check_and_update_config)"),
], key_w=34)
d.step_card(LX[2], r1, LW, CH, 6, "worker_extension_cls", CFG, tag=":304", rows=[
    ("add", "class to worker __bases__"),
    ("clash", "assert: no shared names"),
    ("hook", "init_worker_extension(w)"),
], key_w=42)
d.step_card(LX[0], r2, LW, CH, 7, "model loader", REG, tag="model_runner.py:398", rows=[
    ("key", "load_config.load_format"),
    ("reg", "register_model_loader(...)"),
    ("call", "loader.load_model(...)"),
], key_w=30)
d.step_card(LX[1], r2, LW, CH, 8, "logits processors", GRP, tag="model_runner.py:463", rows=[
    ("grp", "vllm.logits_processors"),
    ("cfg", "+ logits_processors= list"),
    ("V2", "LogitsProcessor subclasses"),
], key_w=30)
d.step_card(LX[2], r2, LW, CH, 9, "KV cache dtype", REG, tag="cache.py:111", rows=[
    ("reg", "register_kv_cache_dtype"),
    ("note", "no caller in vllm/"),
    ("", "(tests only)"),
], key_w=34)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r1 + 52, LX[i + 1] - 2, r1 + 52, INK)
d.arrow(LX[0] + LW + 2, r2 + 52, LX[1] - 2, r2 + 52, INK)
d.path_arrow(f"M {LX[2] + LW / 2:.0f},{r1 + CH + 2} L {LX[2] + LW / 2:.0f},{r1 + CH + 9} "
             f"L {LX[0] + LW / 2:.0f},{r1 + CH + 9} L {LX[0] + LW / 2:.0f},{r2 - 2}", INK)

# ---- Panel 3: at run time ------------------------------------------------------------------------------
y3 = y2 + P2_H + 18
P3_H = 70 + CH + 22
d.panel(y3, P3_H, 3, "At run time",
        "Extension methods are reached by name; the sleep-mode backend is made on first use.")
r = y3 + 70
d.step_card(LX[0], r, LW, CH, 10, "collective_rpc(\"my_method\")", AMBER, tag="", rows=[
    ("from", "LLM, AsyncLLM, or POST"),
    ("", "/collective_rpc (dev mode)"),
], key_w=34)
d.step_card(LX[1], r, LW, CH, 11, "run_method (each worker)", AMBER, tag=":485", rows=[
    ("get", "getattr(worker, name)"),
    ("miss", "NotImplementedError"),
    ("", "via the wrapper __getattr__"),
], key_w=34)
d.step_card(LX[2], r, LW, CH, 12, "sleep mode backend", REG, tag="gpu_worker.py:250", rows=[
    ("key", "model_config.sleep_mode_"),
    ("", "  backend (\"cumem\")"),
    ("reg", "SleepModeBackendFactory"),
], key_w=30)
d.arrow(LX[0] + LW + 2, r + 52, LX[1] - 2, r + 52, AMBER)

y = y3 + P3_H + 16
d.legend(30, y, [(CFG, "config field"), (REG, "registry"), (GRP, "entry-point group"),
                 (AMBER, "run-time call")])
y += 34
y = d.footnotes(30, y, [
    "With TP = 1 (uni executor) panel 2 runs inside the engine core process.",
    "The front end also loads vllm.logits_processors, to check request parameters (not shown).",
    "serial_utils.py:485 = vllm/v1/serial_utils.py. Card 9 is a registry with no caller at this commit, not a start-up step.",
])
d.save(height=y + 8)
