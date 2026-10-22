"""Build a toy plugin: the files of the toy package, its entry points, and what each one registers.

Used by: src/tours/build-a-toy-plugin.html
The toy package is illustrative (not in the repo). The vLLM APIs it uses:
  vllm/model_executor/models/registry.py:1187-1193       register_model with "module:Class" (lazy)
  vllm/model_executor/custom_op.py:334-356, :107-126     register_oot key = class name; __new__ swap
  vllm/config/vllm.py:1819-1826                          base mode "none" with Inductor + compile mode
  vllm/model_executor/custom_op.py:282-290               "+<name>" / "-<name>" switches
  vllm/v1/metrics/loggers.py:84-98                       stat logger group: StatLoggerBase subclass
  vllm/v1/engine/async_llm.py:155-156                    stat loggers load in AsyncLLM.__init__ (front end)
  vllm/plugins/__init__.py:122-131                       endpoint plugins load only if VLLM_PLUGINS names them
  vllm/plugins/__init__.py:63-64                         VLLM_PLUGINS filters the other groups too
Templates: tests/plugins/vllm_add_dummy_model, vllm_add_dummy_platform, vllm_add_dummy_stat_logger,
  vllm_add_dummy_endpoint_plugin.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

GEN, FRONT, SWITCH = EMERALD, BLUE, VIOLET

d = Diagram(width=1100)
d.header("The toy plugin package: files, entry points, effects",
         "vllm_toy_plugin (illustrative)  ·  three entry-point groups  ·  every API cited on the page",
         params="general -> every process   ·   stat logger + endpoint -> front end only")

# ---------------- Panel 1 ----------------
y0 = 120
ROW_H, ROW_GAP = 84, 14
P1_H = 96 + 3 * ROW_H + 2 * ROW_GAP + 12
d.panel(y0, P1_H, 1, "What does each file publish, and what does it register?",
        "One general plugin registers a model and an op. Two front-end plugins add a stat logger and a route.")

CX = [40, 390, 760]
CW = [300, 320, 300]
hy = y0 + 76
for x, w, t in zip(CX, CW, ["Files in vllm_toy_plugin/", "Entry point (pyproject.toml)", "Registers, and where"]):
    d.text(x + w / 2, hy, t, 13, MUTE, "bold", "middle")

rows = [
    (GEN,
     ("__init__.py: register()", ["imports ops.py", "model.py: ToyLlamaForCausalLM"]),
     ("vllm.general_plugins", ["toy_register =", "  vllm_toy_plugin:register"]),
     ("every process", ["ModelRegistry: lazy entry", "op_registry_oot: RMSNorm"])),
    (FRONT,
     ("stat_logger.py", ["class ToyStatLogger", "  (StatLoggerBase)"]),
     ("vllm.stat_logger_plugins", ["toy_stat_logger =", "  ...stat_logger:ToyStatLogger"]),
     ("front end only", ["AsyncLLM stat loggers", "record() each step"])),
    (FRONT,
     ("endpoint.py", ["class ToyEndpointPlugin", "  attach_router, init_state"]),
     ("vllm.endpoint_plugins", ["toy_endpoint =", "  ...endpoint:ToyEndpointPlugin"]),
     ("front end, if allowlisted", ["GET /v1/toy/hello", "needs VLLM_PLUGINS"])),
]
ry = hy + 14
for col, f, e, r in rows:
    for x, w, (title, lines) in zip(CX, CW, (f, e, r)):
        d.box(x, ry, w, ROW_H - 14, title, lines, col=col, tsize=13, mono_size=11.5)
    for i in range(2):
        xa = CX[i] + CW[i]
        d.arrow(xa + 2, ry + (ROW_H - 14) / 2, CX[i + 1] - 2, ry + (ROW_H - 14) / 2, col)
    ry += ROW_H + ROW_GAP

# ---------------- Panel 2 ----------------
y2 = y0 + P1_H + 20
CARD_H = 86
P2_H = 76 + 2 * CARD_H + 16 + 14
d.panel(y2, P2_H, 2, "What makes each part act at run time?",
        "Installing is not enough for three of the four parts.")
cards = [
    (1, "the model", GEN, [("use", "--hf-overrides '{\"architectures\": [...]}'"),
                           ("or", "a config.json that names ToyLlamaForCausalLM")]),
    (2, "the op's forward_cuda", SWITCH, [("on", "custom_ops \"+RMSNorm\" (class name!)"),
                                          ("or", "base mode \"all\", e.g. --enforce-eager")]),
    (3, "the stat logger", FRONT, [("on", "always, once installed"),
                                   ("off", "VLLM_PLUGINS set without its name")]),
    (4, "the route", AMBER, [("on", "VLLM_PLUGINS names toy_endpoint"),
                             ("then", "list every plugin name you need")]),
]
cy = y2 + 72
for i, (n, title, col, rws) in enumerate(cards):
    x = 40 if i % 2 == 0 else 560
    yy = cy + (i // 2) * (CARD_H + 16)
    d.step_card(x, yy, 500, CARD_H, n, title, col, rows=rws, key_w=44)

y = y2 + P2_H + 18
d.legend(30, y, [(GEN, "every process"), (FRONT, "front end only"), (SWITCH, "config switch"),
                 (AMBER, "allowlist")])
y += 34
y = d.footnotes(30, y, [
    "The package is a teaching example. Its vLLM calls are the ones that the test plugins in tests/plugins/ use.",
    "\"...\" stands for vllm_toy_plugin. Without +RMSNorm or \"all\", the swapped op runs the inherited forward_native.",
])
d.save(height=y + 8)
