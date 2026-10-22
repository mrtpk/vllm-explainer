"""KVConnectorFactory: how a name becomes a class, and which process asks.

Used by: src/code/distributed-kv-transfer-kv-connector-factory.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/distributed/kv_transfer/kv_connector/factory.py:30-40     register_connector (lazy loader, duplicate -> ValueError)
  vllm/distributed/kv_transfer/kv_connector/factory.py:96-129    get_connector_class: module path, registry, error
  vllm/distributed/kv_transfer/kv_connector/factory.py:42-75     create_connector: HMA check, log, cls(config, role, kv_cache_config)
  vllm/config/vllm.py:2243, 1242, 1966; vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:57   front end
  vllm/v1/core/sched/scheduler.py:177-181         engine core, SCHEDULER role
  vllm/distributed/kv_transfer/kv_transfer_state.py:88-92   worker, WORKER role
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("KVConnectorFactory: from a name to a connector object",
         "vllm/distributed/kv_transfer/kv_connector/factory.py  ·  one class-level registry per process",
         params="kv_transfer_config.kv_connector  ·  kv_connector_module_path")

LX = [40, 390, 740]
LW = 320
LOOK, ERR, PROC = BLUE, AMBER, VIOLET
CH = 104

y1 = 120
P1_H = 70 + CH + 22
d.panel(y1, P1_H, 1, "get_connector_class: which class? (lines 96-129)",
        "The module path is checked first; the registry second; anything else is an error.")
r = y1 + 70
d.step_card(LX[0], r, LW, CH, 1, "module path set?", LOOK, tag=":106-124", rows=[
    ("yes", "import_module(path)"),
    ("", "getattr(module, kv_connector)"),
    ("need", "kv_cache_config argument"),
], key_w=42)
d.step_card(LX[1], r, LW, CH, 2, "name in _registry?", LOOK, tag=":125-126", rows=[
    ("yes", "call the stored loader:"),
    ("", "import module, getattr class"),
    ("when", "only now, on first use"),
], key_w=38)
d.step_card(LX[2], r, LW, CH, 3, "neither", ERR, tag=":127-128", rows=[
    ("raise", "ValueError"),
    ("", "\"Unsupported connector"),
    ("", "  type: <name>\""),
], key_w=42)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r + 52, LX[i + 1] - 2, r + 52, LOOK)
    d.arrow_label((LX[i] + LW + LX[i + 1]) / 2, r + 52, "no", LOOK)

y2 = y1 + P1_H + 18
P2_H = 70 + CH + 22
d.panel(y2, P2_H, 2, "Which process asks, and for what?",
        "Only the engine core and the workers make objects; the front end needs the class for checks.")
r = y2 + 70
d.step_card(LX[0], r, LW, CH, 4, "front end", PROC, tag="class only", rows=[
    ("hma", "HMA check  vllm.py:2243"),
    ("sleep", "sleep mode check :1242"),
    ("cg", "CUDA graph mode  :1966"),
], key_w=42)
d.step_card(LX[1], r, LW, CH, 5, "engine core", PROC, tag="scheduler.py:177", rows=[
    ("call", "create_connector(config,"),
    ("", "  role=SCHEDULER, kv_cache_cfg)"),
    ("hma", "ValueError if unsupported"),
], key_w=34)
d.step_card(LX[2], r, LW, CH, 6, "each worker", PROC, tag="kv_transfer_state.py:88", rows=[
    ("call", "create_connector(config,"),
    ("", "  role=WORKER, kv_cache_cfg)"),
    ("once", "global _KV_CONNECTOR_AGENT"),
], key_w=34)

y = y2 + P2_H + 16
d.callout(30, y, 1040, "Registration:", "register_connector(name, module_path, class_name) stores a loader; "
          "a taken name raises ValueError (lines 30-40).", AMBER, h=34)
y += 50
d.legend(30, y, [(LOOK, "lookup"), (ERR, "error"), (PROC, "process")])
y += 34
y = d.footnotes(30, y, [
    "The 17 built-in names are registered when the module is imported (lines 149-249).",
    "The metrics logger also reads the class in the front end (v1/metrics.py:57).",
])
d.save(height=y + 8)
