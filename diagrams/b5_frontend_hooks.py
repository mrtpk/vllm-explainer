"""The API server process with the hook point of each front-end plugin.

Used by: src/concepts/frontend-plugins.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/engine/arg_utils.py:3243                general plugins while the CLI builds the parser (LoRA resolvers register)
  vllm/entrypoints/launchers/launcher.py:369-370  init_parser_plugin + validate_api_server_args
  vllm/entrypoints/common/cli_args.py:576-583  --tool-parser-plugin / --reasoning-parser-plugin, only if len > 3
  vllm/v1/engine/async_llm.py:155-156          stat logger plugins
  vllm/entrypoints/launchers/app.py:51         attach_endpoint_plugins (phase A)
  vllm/entrypoints/launchers/api_server/app_state.py:173  init_endpoint_plugins_state (phase B)
  vllm/entrypoints/openai/models/serving.py:113-117, 325-326  LoRA resolvers
  vllm/entrypoints/serve/engine/serving.py:62-66  resolve_lora only with VLLM_ALLOW_RUNTIME_LORA_UPDATING
  vllm/entrypoints/pooling/factories.py:73-80  IO processor plugin for pooling models
  vllm/v1/structured_output/__init__.py:100-112  reasoning parser plugin also in the engine core
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Front-end plugins: where each one plugs in",
         "The API server process of vllm serve  ·  start-up order (panel 1)  ·  one request (panel 2)",
         params="green = entry-point group  ·  blue = registry  ·  violet = CLI flag")

LX = [40, 390, 740]
LW = 320
GRP, REG, FLAG = EMERALD, BLUE, VIOLET
CH = 104

# ---- Panel 1: start-up -------------------------------------------------------------------------
y1 = 120
P1_H = 70 + CH + 18 + CH + 22
d.panel(y1, P1_H, 1, "When does each plugin load? (API server start-up)",
        "Numbers give the order. Each box names the line that loads the plugin.")
r1 = y1 + 70
r2 = r1 + CH + 18
d.step_card(LX[0], r1, LW, CH, 1, "vllm.general_plugins", GRP, tag="arg_utils.py:3243", rows=[
    ("run", "while the CLI builds the parser"),
    ("e.g.", "LoRA resolvers register"),
    ("", "in LoRAResolverRegistry"),
], key_w=34)
d.step_card(LX[1], r1, LW, CH, 2, "parser plugin flags", FLAG, tag="launcher.py:369", rows=[
    ("tool", "--tool-parser-plugin"),
    ("rsn", "--reasoning-parser-plugin"),
    ("rule", "imported only if len > 3"),
], key_w=34)
d.step_card(LX[2], r1, LW, CH, 3, "stat logger plugins", GRP, tag="async_llm.py:156", rows=[
    ("in", "AsyncLLM.__init__"),
    ("type", "StatLoggerBase subclass"),
    ("else", "TypeError"),
], key_w=34)
d.step_card(LX[0], r2, LW, CH, 4, "endpoint plugins (A)", GRP, tag="app.py:51", rows=[
    ("gate", "named in VLLM_PLUGINS"),
    ("", "+ required_tasks match"),
    ("call", "attach_router(app)"),
], key_w=34)
d.step_card(LX[1], r2, LW, CH, 5, "endpoint plugins (B)", GRP, tag="app_state.py:173", rows=[
    ("call", "init_state(engine_client,"),
    ("", "  state, args)"),
    ("render", "engine_client is None"),
], key_w=46)
d.step_card(LX[2], r2, LW, CH, 6, "IO processor plugins", GRP, tag="factories.py:73", rows=[
    ("only", "pooling models"),
    ("pick", "--io-processor-plugin or"),
    ("", "hf_config io_processor_plugin"),
], key_w=34)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r1 + 52, LX[i + 1] - 2, r1 + 52, INK)
    d.arrow(LX[i] + LW + 2, r2 + 52, LX[i + 1] - 2, r2 + 52, INK)
d.path_arrow(f"M {LX[2] + LW / 2:.0f},{r1 + CH + 2} L {LX[2] + LW / 2:.0f},{r1 + CH + 9} "
             f"L {LX[0] + LW / 2:.0f},{r1 + CH + 9} L {LX[0] + LW / 2:.0f},{r2 - 2}", INK)

# ---- Panel 2: one request ----------------------------------------------------------------------
y2 = y1 + P1_H + 18
P2_H = 70 + CH + 18 + CH + 22
d.panel(y2, P2_H, 2, "What does each plugin touch in one request?",
        "From the HTTP route to the engine and back. The engine core also runs the reasoning parser.")
r1 = y2 + 70
r2 = r1 + CH + 18
d.step_card(LX[0], r1, LW, CH, 1, "HTTP route", GRP, tag="interface.py:69", rows=[
    ("plugin", "endpoint routes; same path"),
    ("", "and method replace the core"),
    ("", "route"),
], key_w=50)
d.step_card(LX[1], r1, LW, CH, 2, "unknown model name", REG, tag="serving.py:325", rows=[
    ("if", "VLLM_ALLOW_RUNTIME_LORA_"),
    ("", "  UPDATING=1"),
    ("ask", "each LoRA resolver in order"),
], key_w=30)
d.step_card(LX[2], r1, LW, CH, 3, "pooling input", GRP, tag="pooling", rows=[
    ("io", "IOProcessor.pre_process"),
    ("", "(only with an IO processor"),
    ("", " plugin)"),
], key_w=24)
d.step_card(LX[0], r2, LW, CH, 4, "engine core", AMBER, tag="so/__init__.py:100", rows=[
    ("rsn", "other process: imports the"),
    ("", "reasoning parser plugin again,"),
    ("", "for structured outputs"),
], key_w=30)
d.step_card(LX[1], r2, LW, CH, 5, "output text", REG, tag="ParserManagers", rows=[
    ("rsn", "--reasoning-parser name"),
    ("tool", "--tool-call-parser name"),
    ("", "-> class from the manager"),
], key_w=34)
d.step_card(LX[2], r2, LW, CH, 6, "every engine step", GRP, tag="loggers.py:1398", rows=[
    ("all", "stat loggers record()"),
    ("plus", "PrometheusStatLogger unless"),
    ("", "a plugin is one"),
], key_w=34)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r1 + 52, LX[i + 1] - 2, r1 + 52, INK)
    d.arrow(LX[i] + LW + 2, r2 + 52, LX[i + 1] - 2, r2 + 52, INK)
d.path_arrow(f"M {LX[2] + LW / 2:.0f},{r1 + CH + 2} L {LX[2] + LW / 2:.0f},{r1 + CH + 9} "
             f"L {LX[0] + LW / 2:.0f},{r1 + CH + 9} L {LX[0] + LW / 2:.0f},{r2 - 2}", INK)

y = y2 + P2_H + 16
d.legend(30, y, [(GRP, "entry-point group"), (REG, "registry"), (FLAG, "CLI flag"),
                 (AMBER, "outside the API server")])
y += 34
y = d.footnotes(30, y, [
    "so/__init__.py = vllm/v1/structured_output/__init__.py; serving.py:325 = entrypoints/openai/models/serving.py.",
    "Not shown: logits processors (front-end request check) and model inspection; see the plugin map.",
])
d.save(height=y + 8)
