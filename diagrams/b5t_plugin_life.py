"""Life of a plugin: the 16 tour stops in the three process lanes of vllm serve -tp 2.

Used by: src/tours/life-of-a-plugin.html
Sources (vLLM at the pinned commit 3e44c0d):
  tests/plugins/vllm_add_dummy_platform/setup.py:10-17       stop 1: two entry points (platform + general)
  tests/plugins/vllm_add_dummy_model/setup.py:10-12          stop 1: one general entry point
  vllm/entrypoints/cli/main.py:67-68                         stop 2: cli_env_setup (spawn)
  vllm/platforms/__init__.py:250-279                         stop 3: detectors; one OOT winner
  vllm/engine/arg_utils.py:3243                              stop 4: general plugins, first run
  vllm/engine/arg_utils.py:3257 (x4), :2246                  stop 5: pre_register_and_update five times
  vllm/entrypoints/launchers/launcher.py:368, common/startup.py:70-71  stop 5: setup_server builds the parser again
  vllm/entrypoints/cli/launch.py:39, cli/run_batch.py:62     stop 5: two more parsers call add_cli_args
  vllm/config/model.py:741                                   stop 6: registry inspection
  vllm/config/vllm.py:1804, :2046                            stop 7: platform config hooks
  vllm/v1/engine/core.py:1336-1337                           stop 7 again: engine core re-runs __post_init__ after the handshake
  vllm/v1/engine/async_llm.py:156, :193                      stop 8: stat logger group, engine core start
  vllm/v1/engine/core.py:132, :147                           stops 9-10: general plugins, executor
  vllm/v1/executor/multiproc_executor.py:191-207, :745-750   stop 10: one spawned process per rank
  vllm/v1/worker/worker_base.py:284-296                      stops 11-12
  vllm/v1/attention/selector.py:218                          stop 13
  vllm/model_executor/custom_op.py:117-126, :183-205         stops 14-15
  vllm/v1/executor/multiproc_executor.py:702, :710           stop 16
  vllm/v1/executor/uniproc_executor.py:68-75                 TP = PP = DP = 1 callout (config/parallel.py:1055, 1099-1100)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

GRAY = ("#4b5563", "#f3f4f6", "#fafafa")
DET, GEN, CONF, MODEL = AMBER, EMERALD, VIOLET, BLUE

d = Diagram(width=1100)
d.header("Life of a plugin: where each stop runs",
         "vllm serve -tp 2 with the dummy vendor package (two vLLM test plugins)  ·  badge = tour stop",
         params="every process detects the platform and runs the general plugins itself")

y0 = 120
lanes = [
    ("API server (front end)", ["CLI, EngineArgs, VllmConfig"], [
        (2, "vllm serve: spawn mode", "cli/main.py:67-68", GRAY),
        (3, "dummy detector wins", "platforms/__init__.py:277", DET),
        (4, "general plugins: run 1", "arg_utils.py:3243", GEN),
        (5, "pre_register_and_update x 5", "arg_utils.py:3257 (x 4), 2246", CONF),
        (6, "registry: MyOPTForCausalLM", "config/model.py:741", GEN),
        (7, "platform config hooks", "config/vllm.py:1804, 2046", CONF),
        (8, "start the engine core", "async_llm.py:156, 193", GRAY),
    ]),
    ("Engine core process", ["scheduler, executor"], [
        (9, "detect + general plugins", "v1/engine/core.py:132", GEN),
        (10, "spawn 2 worker processes", "multiproc_executor.py:197", GRAY),
        (7, "config hooks again", "core.py:1336-1337 (after 16)", CONF),
    ]),
    ("Worker process (x 2)", ["one per rank"], [
        (11, "detect + plugins + Triton", "worker_base.py:284-290", GEN),
        (12, "resolve worker_cls", "worker_base.py:293-296", CONF),
        (13, "attention backend", "v1/attention/selector.py:218", MODEL),
        (14, "CustomOp.__new__ swap", "custom_op.py:117-126", MODEL),
        (15, "forward_oot if enabled", "custom_op.py:183, 202-203", MODEL),
        (16, "block size, then ready", "multiproc_executor.py:702", GRAY),
    ]),
]

ITEM_H, GAP, HEAD = 46, 8, 64
n_max = max(len(items) for _, _, items in lanes)
lane_h = HEAD + 14 + n_max * (ITEM_H + GAP) + 6
INSTALL_H = 50
P1_H = 72 + INSTALL_H + 22 + lane_h + 16
d.panel(y0, P1_H, 1, "Which process runs which stop?",
        "Stops 2-8 run in the front end. Stops 9-10 (and stop 7 again) run in the engine core. Stops 11-16 run in each worker.")

# Stop 1: before any process exists.
iy = y0 + 70
d.rect(40, iy, 1020, INSTALL_H, fill=GEN[1], stroke=GEN[0], rx=8, sw=1.3)
d.badge(62, iy + INSTALL_H / 2, 1, GEN[0], r=11, size=11)
d.text(84, iy + 21, "pip install: entry_points.txt gets 3 entry points", 13, INK, "bold")
d.text(84, iy + 39, "vllm.platform_plugins: dummy_platform_plugin   ·   vllm.general_plugins: dummy_custom_ops, "
       "register_dummy_model", 11.5, MUTE, mono=True)

LX = [40, 390, 740]
LW = 320
ly = iy + INSTALL_H + 22
for x, (name, details, items) in zip(LX, lanes):
    d.lane(x, ly, LW, lane_h, GRAY, name, details)
    jy = ly + HEAD + 14
    for n, title, where, col in items:
        d.rect(x + 12, jy, LW - 24, ITEM_H, fill=col[1], stroke=col[0], rx=8, sw=1.3)
        d.badge(x + 32, jy + ITEM_H / 2, n, col[0], r=11, size=11)
        d.text(x + 52, jy + 19, title, 12.5, INK, "bold", mono=True)
        d.text(x + 52, jy + 36, where, 11.5, MUTE, mono=True)
        jy += ITEM_H + GAP

for i in range(2):
    xa = LX[i] + LW
    d.arrow(xa + 2, ly + 32, LX[i + 1] - 2, ly + 32)
d.arrow(550, iy + INSTALL_H + 2, 550, ly - 2, GEN, dash="5 4")

y = y0 + P1_H + 16
d.callout(30, y, 1040, "TP = PP = DP = 1:", "the uni executor runs stops 11-16 inside the engine core process; "
          "the second load_general_plugins() call there returns at once.", AMBER, h=36)
y += 52
d.legend(30, y, [(DET, "platform detection"), (GEN, "install / general plugin / registry"),
                 (CONF, "config hook"), (MODEL, "model build"), (GRAY, "start-up step")])
y += 34
y = d.footnotes(30, y, [
    "Dummy vendor package = tests/plugins/vllm_add_dummy_platform + vllm_add_dummy_model, installed together.",
    "  vLLM's CI installs them one at a time; the test platform sets no worker_cls (stop 12).",
    "Not shown: the registry subprocess (MyOPTForCausalLM is an eager entry) and the groups the package does not use.",
])
d.save(height=y + 8)
