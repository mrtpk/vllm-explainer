"""Which plugin groups load in which vLLM process (vllm serve, one GPU per worker).

Used by: src/concepts/plugin-loading.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/plugins/__init__.py:16-30      the five group names and the comments on where they load
  vllm/plugins/__init__.py:36-74      load_plugins_by_group: discover (:42), filter (:64), load (:69), errors (:71-72)
  vllm/plugins/__init__.py:77-90      load_general_plugins: guard (:83-85), call each function (:89-90)
  vllm/platforms/__init__.py:244-248  VLLM_TARGET_DEVICE=cpu skips the platform plugin group
  vllm/engine/arg_utils.py:3243, :924 front end: add_cli_args, EngineArgs.__post_init__
  vllm/platforms/__init__.py:250      platform plugins, inside resolve_current_platform_cls_qualname (first current_platform access)
  vllm/v1/engine/async_llm.py:156     stat logger plugins (AsyncLLM.__init__)
  vllm/entrypoints/launchers/app.py:51            endpoint plugins (build_app -> attach_endpoint_plugins, plugins/endpoint_plugins/interface.py:108)
  vllm/entrypoints/pooling/pooling/io_processor.py:51  IO processor plugins (get_io_processor)
  vllm/v1/engine/input_processor.py:107           logits processors (front-end validator, MRV2)
  vllm/v1/engine/core.py:132          engine core: EngineCore.__init__
  vllm/v1/worker/worker_base.py:284   worker: WorkerWrapperBase.init_worker
  vllm/v1/worker/gpu/model_runner.py:464          logits processors (worker, model runner load_model)
  vllm/v1/executor/uniproc_executor.py:68         one GPU: init_worker in the engine core process
  vllm/model_executor/models/registry.py:823, :1608-1612  model registry subprocess
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

GRAY = ("#4b5563", "#f3f4f6", "#fafafa")

d = Diagram(width=1100)
d.header("Which plugin groups load in which process",
         "vllm serve with multiprocessing: one front end, one engine core process, one worker process per GPU",
         params="vllm/plugins/__init__.py  ·  load_plugins_by_group()  ·  load_general_plugins()")

# ---- Panel 1: processes and groups ---------------------------------------------------------
y0 = 122
ITEM_H, ITEM_GAP, HEAD = 64, 10, 64
n_items = 6
lane_h = HEAD + 14 + n_items * (ITEM_H + ITEM_GAP) + 4
P1_H = 74 + lane_h + 20
d.panel(y0, P1_H, 1, "Which process loads which group?",
        "General and platform plugins load in every process. Three groups load only in the front end.")

LX = [40, 390, 740]
LW = 320
ly = y0 + 74
lanes = [
    ("API server (front end)", ["vllm serve: FastAPI, AsyncLLM"], [
        ("vllm.general_plugins", ["add_cli_args, EngineArgs.__post_init__", "engine/arg_utils.py:3243, :924"], EMERALD),
        ("vllm.platform_plugins", ["first current_platform access", "platforms/__init__.py:250"], EMERALD),
        ("vllm.stat_logger_plugins", ["AsyncLLM.__init__", "v1/engine/async_llm.py:156"], BLUE),
        ("vllm.endpoint_plugins", ["build_app; only if named in VLLM_PLUGINS", "entrypoints/launchers/app.py:51"], BLUE),
        ("vllm.io_processor_plugins", ["pooling model asks for a plugin", "pooling/pooling/io_processor.py:51"], BLUE),
        ("vllm.logits_processors", ["InputProcessor.__init__ (validator)", "v1/engine/input_processor.py:107"], VIOLET),
    ]),
    ("Engine core process", ["one per DP rank: scheduler, KV cache"], [
        ("vllm.general_plugins", ["EngineCore.__init__", "v1/engine/core.py:132"], EMERALD),
        ("vllm.platform_plugins", ["first current_platform access", "platforms/__init__.py:250"], EMERALD),
    ]),
    ("Worker process (one per GPU)", ["owns one GPU and the model runner"], [
        ("vllm.general_plugins", ["WorkerWrapperBase.init_worker", "v1/worker/worker_base.py:284"], EMERALD),
        ("vllm.platform_plugins", ["first current_platform access", "platforms/__init__.py:250"], EMERALD),
        ("vllm.logits_processors", ["model runner load_model", "v1/worker/gpu/model_runner.py:464"], VIOLET),
    ]),
]
for x, (name, details, items) in zip(LX, lanes):
    d.lane(x, ly, LW, lane_h, GRAY, name, details)
    iy = ly + HEAD + 14
    for title, lines, col in items:
        d.box(x + 12, iy, LW - 24, ITEM_H, title, lines, col=col, align="start", tsize=13)
        iy += ITEM_H + ITEM_GAP

# "starts" arrows between the lane headers (in the 30 px gaps)
for i in range(2):
    xa = LX[i] + LW
    d.arrow(xa + 2, ly + 32, LX[i + 1] - 2, ly + 32, INK)
d.text(LX[1] + 14, ly + lane_h - 14, "started by the front end", 11.5, MUTE, italic=True)
d.text(LX[2] + 14, ly + lane_h - 14, "started by the engine core (executor)", 11.5, MUTE, italic=True)

# ---- Panel 2: what one load does -------------------------------------------------------------
y2 = y0 + P1_H + 18
P2_H = 236
d.panel(y2, P2_H, 2, "What does load_plugins_by_group() do?",
        "It finds, filters and imports. It does not call the objects: the caller decides.")
cy = y2 + 72
CH = 98
cards = [
    (1, "Discover", [("call", "entry_points(group=g)"), ("line", "plugins/__init__.py:42")]),
    (2, "Filter by name", [("env", "VLLM_PLUGINS (one list)"), ("unset", "load every plugin"), ("line", ":40, :64")]),
    (3, "Import", [("call", "plugin.load()"), ("error", "logged and skipped"), ("line", ":69, :71-72")]),
]
for x, (n, title, rows) in zip(LX, cards):
    d.step_card(x, cy, LW, CH, n, title, BLUE, rows=rows, key_w=52)
for i in range(2):
    d.arrow(LX[i] + LW + 2, cy + CH / 2, LX[i + 1] - 2, cy + CH / 2, BLUE)
d.callout(40, cy + CH + 16, 1020, "load_general_plugins():",
          "returns at once if plugins_loaded is True (:83-84); else sets it (:85), then calls every function (:89-90). Errors are not caught.",
          AMBER, h=36)

# ---- Panel 3: two special cases --------------------------------------------------------------
y3 = y2 + P2_H + 18
P3_H = 178
d.panel(y3, P3_H, 3, "Two more cases",
        "One GPU: the worker shares the engine core process. Lazy model info: a new Python process loads the plugins again.")
by = y3 + 72
d.box(40, by, 495, 86, "One GPU: UniProcExecutor", [
    "init_worker runs inside the engine core process",
    "plugins_loaded is already True: no second load",
    "v1/executor/uniproc_executor.py:68",
], col=EMERALD, align="start")
d.box(565, by, 495, 86, "Model registry subprocess", [
    "python -m vllm.model_executor.models.registry",
    "_run() calls load_general_plugins() first",
    "models/registry.py:823, :1608-1612",
], col=EMERALD, align="start")

# ---- legend and footnotes --------------------------------------------------------------------
y = y3 + P3_H + 24
d.legend(30, y, [(EMERALD, "every process"), (BLUE, "front end only"), (VIOLET, "front end + workers"),
                 (AMBER, "errors stop start-up")])
y += 36
y = d.footnotes(30, y, [
    "Boxes in a lane are not in time order. Platform plugins load once per process, when code first reads current_platform (not with VLLM_TARGET_DEVICE=cpu).",
    "vllm.logits_processors does not use load_plugins_by_group() or VLLM_PLUGINS. Not shown: Ray, the DP coordinator, extra API servers.",
])
d.save(height=y + 8)
