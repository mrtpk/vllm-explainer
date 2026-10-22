"""Where vllm/plugins/__init__.py sits: who calls its three functions, and what they call.

Used by: src/code/plugins-init.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/plugins/__init__.py:16-33      group constants, plugins_loaded
  vllm/plugins/__init__.py:36-74      load_plugins_by_group: VLLM_PLUGINS (:40), entry_points (:42), filter (:64), load (:69), errors (:71-72)
  vllm/plugins/__init__.py:77-90      load_general_plugins: guard (:83-85), group (:87), call each function (:89-90)
  vllm/plugins/__init__.py:93-159     load_endpoint_plugins: unset -> warning + [] (:122-131), loader (:133), factory (:138), tasks gate (:143-154)
  vllm/engine/arg_utils.py:3243, :924             front-end callers of load_general_plugins
  vllm/v1/engine/core.py:132                      engine core caller
  vllm/v1/worker/worker_base.py:284               worker caller
  vllm/model_executor/models/registry.py:1612     model registry subprocess caller
  vllm/platforms/__init__.py:250                  platform plugins
  vllm/v1/metrics/loggers.py:87                   stat logger plugins
  vllm/plugins/io_processors/__init__.py:59-61    IO processor plugins
  vllm/plugins/endpoint_plugins/interface.py:108  attach_endpoint_plugins (called by launchers/app.py:51)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import BLUE, EMERALD, INK, MUTE, NEUTRAL, PINK, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("vllm/plugins/__init__.py: callers, functions and calls",
         "159 lines  ·  three functions, five group names, one flag  ·  vLLM at 3e44c0d",
         params="load_general_plugins()  ·  load_plugins_by_group(group)  ·  load_endpoint_plugins(supported_tasks)")

# ---- Panel 1 -------------------------------------------------------------------------------
y0 = 118
LX, LW = 40, 400       # callers
CX, CW = 480, 250      # functions in this file
RX, RW = 770, 290      # what they call

rows = [
    # (function title, col, caller lines, call lines, height)
    ("load_general_plugins()", EMERALD, [
        f"{'engine/arg_utils.py:3243':<29}add_cli_args",
        f"{'engine/arg_utils.py:924':<29}__post_init__",
        f"{'v1/engine/core.py:132':<29}EngineCore.__init__",
        f"{'v1/worker/worker_base.py:284':<29}init_worker",
        f"{'models/registry.py:1612':<29}_run (subprocess)",
    ], [
        "guard: plugins_loaded  :83-85",
        "func() for each plugin  :89-90",
        "no try block around func()",
    ], ":77"),
    ("load_plugins_by_group()", BLUE, [
        f"{'platforms/__init__.py:250':<29}platform detection",
        f"{'v1/metrics/loggers.py:87':<29}stat loggers",
        f"{'io_processors/__init__.py:59':<29}get_io_processor",
    ], [
        "envs.VLLM_PLUGINS  :40",
        "entry_points(group=group)  :42",
        "plugin.load() in try  :68-72",
    ], ":36"),
    ("load_endpoint_plugins()", PINK, [
        "endpoint_plugins/interface.py:108",
        "  attach_endpoint_plugins",
        "  <- build_app, launchers/app.py:51",
    ], [
        "VLLM_PLUGINS unset: [] (+ warning)  :122-131",
        "factory() in try  :137-141",
        "required_tasks gate  :143-154",
    ], ":93"),
]

def row_h(n):
    return 38 + n * 15 + 6

heights = [max(row_h(len(r[2])), row_h(len(r[3]))) for r in rows]
GAP = 34
P1_H = 76 + sum(heights) + GAP * (len(rows) - 1) + 40
d.panel(y0, P1_H, 1, "Who calls which function, and what does each one call?",
        "Nine call sites in vLLM use the three functions. Two of the functions call the shared loader.")
d.text(LX, y0 + 74, "CALLERS (file:line)", 12, MUTE, "bold", mono=True)
d.text(CX, y0 + 74, "IN THIS FILE", 12, MUTE, "bold", mono=True)
d.text(RX, y0 + 74, "WHAT IT CALLS", 12, MUTE, "bold", mono=True)

y = y0 + 88
centers = []
for (title, col, callers, calls, line), h in zip(rows, heights):
    d.box(LX, y, LW, h, "callers of " + title, callers, col=col, align="start", tsize=13)
    fy = y + h / 2 - 30
    d.box(CX, fy, CW, 60, title, ["plugins/__init__.py" + line], col=col, tsize=13.5)
    d.box(RX, y, RW, h, "it calls", calls, col=NEUTRAL, align="start", tsize=13)
    d.arrow(LX + LW + 2, fy + 30, CX - 2, fy + 30, col)
    d.arrow(CX + CW + 2, fy + 30, RX - 2, fy + 30, INK)
    centers.append((fy, fy + 60))
    y += h + GAP

# load_general_plugins and load_endpoint_plugins both use load_plugins_by_group
mid = CX + CW / 2
d.arrow(mid, centers[0][1] + 2, mid, centers[1][0] - 2, EMERALD)
d.arrow(mid, centers[2][0] - 2, mid, centers[1][1] + 2, PINK)
d.text(mid + 8, (centers[0][1] + centers[1][0]) / 2 + 4, "group :87", 11.5, EMERALD, "bold", mono=True)
d.text(mid + 8, (centers[1][1] + centers[2][0]) / 2 + 4, "group :133", 11.5, PINK, "bold", mono=True)

# ---- Panel 2 -------------------------------------------------------------------------------
y2 = y0 + P1_H + 18
P2_H = 196
d.panel(y2, P2_H, 2, "How do the three functions treat VLLM_PLUGINS and errors?",
        "Each one adds its own rule on top of the shared loader.")
by = y2 + 70
BW = 320
cards = [
    ("load_plugins_by_group()", BLUE, ["VLLM_PLUGINS unset: load all", "set: exact name match  :64",
                                        "import error: log, skip  :71-72", "returns dict name -> object"]),
    ("load_general_plugins()", EMERALD, ["same filter (calls the loader)", "once per process  :83-85",
                                         "error in func(): not caught", "  -> the process stops"]),
    ("load_endpoint_plugins()", PINK, ["VLLM_PLUGINS unset: none load", "factory error: log, skip",
                                       "tasks do not match: INFO, skip", "returns list[EndpointPlugin]"]),
]
for i, (title, col, lines) in enumerate(cards):
    d.box(40 + i * (BW + 30), by, BW, 104, title, lines, col=col, align="start", tsize=13.5)

# ---- legend and footnotes --------------------------------------------------------------------
y = y2 + P2_H + 24
d.legend(30, y, [(EMERALD, "general plugins"), (BLUE, "shared loader"), (PINK, "endpoint plugins"),
                 (NEUTRAL, "code outside the plugins")])
y += 36
y = d.footnotes(30, y, [
    "Line numbers are in vllm/plugins/__init__.py unless a path is given. Callers in tests are not shown.",
    "vllm.logits_processors does not use this file: its loaders call entry_points() themselves.",
])
d.save(height=y + 8)
