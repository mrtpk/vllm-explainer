"""The plugin map: where each extension point acts, grouped by the process of vllm serve.

Used by: src/concepts/plugin-map.html
Sources (vLLM at the pinned commit 3e44c0d):
  API server (front end)
    vllm/engine/arg_utils.py:3243                      vllm.general_plugins (add_cli_args)
    vllm/platforms/__init__.py:250                     vllm.platform_plugins (every process)
    vllm/v1/engine/async_llm.py:156                    vllm.stat_logger_plugins
    vllm/entrypoints/launchers/app.py:51               vllm.endpoint_plugins (attach_endpoint_plugins)
    vllm/entrypoints/pooling/pooling/io_processor.py:51  vllm.io_processor_plugins (pooling only)
    vllm/v1/engine/input_processor.py:103-107          vllm.logits_processors (front-end validator, MRV2)
    vllm/config/model.py:741                           ModelRegistry.inspect_model_cls
    vllm/config/model.py:1430                          get_quantization_config (custom methods too)
    vllm/entrypoints/openai/models/serving.py:114      LoRAResolverRegistry
    vllm/config/vllm.py:2243 (-> kv_connector/factory.py:138)  KV connector class (HMA check; also metrics loggers.py:132)
    vllm/entrypoints/common/cli_args.py:579-583        --tool-parser-plugin / --reasoning-parser-plugin
  Engine core process
    vllm/v1/engine/core.py:132                         vllm.general_plugins
    vllm/v1/engine/core.py:164                         scheduler_cls (get_scheduler_cls)
    vllm/v1/core/sched/scheduler.py:177, :203          KV / EC connector, scheduler role
    vllm/v1/structured_output/__init__.py:103-104      --reasoning-parser-plugin again
  Worker process
    vllm/v1/worker/worker_base.py:284, :293-331        vllm.general_plugins, worker_cls, worker_extension_cls
    vllm/v1/worker/gpu/model_runner.py:463-466         vllm.logits_processors
    vllm/model_executor/model_loader/utils.py:310      ModelRegistry.resolve_model_cls
    vllm/model_executor/custom_op.py:56-59, :117-120   op_registry_oot swap in PluggableLayer / CustomOp.__new__
    vllm/v1/attention/backends/registry.py:210         _ATTN_OVERRIDES read in get_path
    vllm/model_executor/model_loader/__init__.py:138   get_model_loader
    vllm/distributed/kv_transfer/kv_transfer_state.py:88  KV connector, worker role
    vllm/v1/worker/gpu_worker.py:256                   SleepModeBackendFactory.create_backend
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, MUTE, VIOLET, Diagram, text_width  # noqa: E402

GRAY = ("#4b5563", "#f3f4f6", "#fafafa")
EP, REG, CFG = EMERALD, BLUE, VIOLET

d = Diagram(width=1100)
d.header("Where each extension point acts",
         "vllm serve with worker processes  ·  one column per process  ·  each item: name and the line that reads it",
         params="green = entry-point group  ·  blue = registry API (a general plugin fills it)  ·  violet = config field / flag")

lanes = [
    ("API server (front end)", ["FastAPI, AsyncLLM, CLI parsing"], [
        ("vllm.general_plugins", "engine/arg_utils.py:3243", EP),
        ("vllm.platform_plugins", "platforms/__init__.py:250", EP),
        ("vllm.stat_logger_plugins", "v1/engine/async_llm.py:156", EP),
        ("vllm.endpoint_plugins", "launchers/app.py:51", EP),
        ("vllm.io_processor_plugins", "pooling/io_processor.py:51", EP),
        ("vllm.logits_processors (check)", "input_processor.py:103-107", EP),
        ("ModelRegistry (inspect)", "config/model.py:741", REG),
        ("custom quantization config", "config/model.py:1430", REG),
        ("LoRAResolverRegistry", "openai/models/serving.py:114", REG),
        ("KVConnectorFactory (class check)", "config/vllm.py:2243", REG),
        ("--tool/--reasoning-parser-plugin", "common/cli_args.py:579-583", CFG),
    ]),
    ("Engine core process", ["scheduler, KV cache manager"], [
        ("vllm.general_plugins", "v1/engine/core.py:132", EP),
        ("vllm.platform_plugins", "platforms/__init__.py:250", EP),
        ("scheduler_cls", "v1/engine/core.py:164", CFG),
        ("KVConnectorFactory (scheduler)", "sched/scheduler.py:177", REG),
        ("ECConnectorFactory (scheduler)", "sched/scheduler.py:203", REG),
        ("--reasoning-parser-plugin", "structured_output/__init__.py:104", CFG),
    ]),
    ("Worker process (one per GPU)", ["model runner, KV cache tensors"], [
        ("vllm.general_plugins", "v1/worker/worker_base.py:284", EP),
        ("vllm.platform_plugins", "platforms/__init__.py:250", EP),
        ("worker_cls, worker_extension_cls", "worker_base.py:293-331", CFG),
        ("vllm.logits_processors", "gpu/model_runner.py:463-466", EP),
        ("ModelRegistry (resolve)", "model_loader/utils.py:310", REG),
        ("CustomOp / PluggableLayer OOT", "custom_op.py:56-59, 117-120", REG),
        ("register_backend (attention)", "backends/registry.py:210", REG),
        ("register_model_loader", "model_loader/__init__.py:138", REG),
        ("KVConnectorFactory (worker)", "kv_transfer_state.py:88", REG),
        ("SleepModeBackendFactory", "v1/worker/gpu_worker.py:256", REG),
    ]),
]

y0 = 120
ITEM_H, GAP, HEAD = 44, 8, 64
n_max = max(len(items) for _, _, items in lanes)
lane_h = HEAD + 14 + n_max * (ITEM_H + GAP) + 30
P1_H = 72 + lane_h + 16
d.panel(y0, P1_H, 1, "Which extension points does each process read?",
        "Each process loads its own plugins and reads its own registries. The front end reads the most.")

LX = [40, 390, 740]
LW = 320
ly = y0 + 70
for x, (name, details, items) in zip(LX, lanes):
    d.lane(x, ly, LW, lane_h, GRAY, name, details)
    iy = ly + HEAD + 14
    for title, where, col in items:
        d.rect(x + 12, iy, LW - 24, ITEM_H, fill=col[1], stroke=col[0], rx=8, sw=1.3)
        d.rect(x + 12, iy, 6, ITEM_H, fill=col[0], stroke=None, rx=3)
        d.text(x + 26, iy + 18, title, 12.5, col[0], "bold", mono=True)
        d.text(x + 26, iy + 35, where, 11.5, MUTE, mono=True)
        iy += ITEM_H + GAP

for i in range(2):
    xa = LX[i] + LW
    d.arrow(xa + 2, ly + 32, LX[i + 1] - 2, ly + 32)
d.text(LX[1] + 14, ly + lane_h - 12, "started by the front end", 11.5, MUTE, italic=True)
d.text(LX[2] + 14, ly + lane_h - 12, "started by the executor (TP > 1)", 11.5, MUTE, italic=True)

# ---- callout and legend --------------------------------------------------------------------
y = y0 + P1_H + 16
d.callout(30, y, 1040, "One general plugin, many registries:",
          "its function runs once in each process and fills the registries of that process.", AMBER, h=36)
y += 52
d.legend(30, y, [(EP, "entry-point group"), (REG, "registry API"), (CFG, "config field or flag"),
                 (AMBER, "rule for all registries")])
y += 34
y = d.footnotes(30, y, [
    "Not in time order. With TP = 1 the worker column runs inside the engine core process (uni executor).",
    "Config fields are set in the front end and reach the other processes inside the pickled VllmConfig.",
    "Not shown: Ray workers, the DP coordinator, the model registry subprocess, register_kv_cache_dtype (no caller in vllm/).",
])
d.save(height=y + 8)
