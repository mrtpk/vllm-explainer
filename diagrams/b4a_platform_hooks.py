"""Start-up timeline of vllm serve with the Platform hook calls, in order, per process.

Used by: src/concepts/platform-interface.html, src/code/platforms-interface.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/engine/arg_utils.py:122            module import of current_platform -> detection (front end)
  vllm/engine/arg_utils.py:3243, :3257    add_cli_args: general plugins, then pre_register_and_update(parser)
  vllm/engine/arg_utils.py:2246, :2250    create_engine_config: pre_register_and_update(), validate_environ
  vllm/config/vllm.py:1804                VllmConfig.__post_init__: apply_config_platform_defaults
  vllm/config/vllm.py:2046                VllmConfig.__post_init__: check_and_update_config (sets worker_cls)
  vllm/v1/engine/async_llm.py:193         AsyncLLM starts the engine core process
  vllm/v1/engine/__init__.py:14           engine core child: import chain resolves the platform
  vllm/v1/engine/core.py:132, :147        general plugins, then the executor
  vllm/v1/executor/__init__.py:3          worker child: import chain resolves the platform
  vllm/v1/worker/worker_base.py:284-296   general plugins, register_triton_kernel_overrides, worker_cls
  vllm/distributed/parallel_state.py:549-552  get_device_communicator_cls (only when the group has > 1 rank)
  vllm/v1/attention/selector.py:218       get_attn_backend_cls (layers built in load_model)
  vllm/model_executor/layers/attention/attention.py:440  opaque_attention_op
  vllm/v1/executor/multiproc_executor.py:702   update_block_size_for_backend (each worker, mp)
  vllm/v1/executor/uniproc_executor.py:75      update_block_size_for_backend (engine core, uni)
  vllm/v1/engine/core.py:311 -> vllm/v1/core/kv_cache_planning.py:1497 -> kv_cache_config_builder.py:101
                                          get_kv_cache_config_builder_cls
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

GRAY = ("#4b5563", "#f3f4f6", "#fafafa")
DET, CONF, PICK, KER = AMBER, VIOLET, BLUE, PINK

d = Diagram(width=1100)
d.header("When vLLM calls the Platform hooks",
         "vllm serve, TP = 2 (worker processes)  ·  numbers give the order  ·  each process has its own current_platform",
         params="config hooks run in the front end  ·  component pickers run where the component is built")

lanes = [
    ("API server (front end)", ["CLI, EngineArgs, VllmConfig"], [
        (1, "detect the platform", "import of arg_utils.py:122", DET),
        (2, "pre_register_and_update(parser)", "engine/arg_utils.py:3257", CONF),
        (3, "pre_register_and_update()", "engine/arg_utils.py:2246", CONF),
        (4, "validate_environ", "engine/arg_utils.py:2250", CONF),
        (5, "apply_config_platform_defaults", "config/vllm.py:1804", CONF),
        (6, "check_and_update_config", "config/vllm.py:2046", CONF),
        (7, "start the engine core", "v1/engine/async_llm.py:193", GRAY),
    ]),
    ("Engine core process", ["scheduler, KV cache planning"], [
        (8, "detect the platform", "import of v1/engine/__init__.py:14", DET),
        (9, "start the workers", "v1/engine/core.py:147", GRAY),
        (16, "get_kv_cache_config_builder_cls", "kv_cache_config_builder.py:101", PICK),
    ]),
    ("Worker process (x 2)", ["one per GPU"], [
        (10, "detect the platform", "import of v1/executor/__init__.py:3", DET),
        (11, "register_triton_kernel_overrides", "v1/worker/worker_base.py:290", KER),
        (12, "get_device_communicator_cls", "parallel_state.py:551", PICK),
        (13, "get_attn_backend_cls", "v1/attention/selector.py:218", PICK),
        (14, "opaque_attention_op", "attention/attention.py:440", PICK),
        (15, "update_block_size_for_backend", "multiproc_executor.py:702", CONF),
    ]),
]

y0 = 120
ITEM_H, GAP, HEAD = 46, 8, 64
n_max = max(len(items) for _, _, items in lanes)
lane_h = HEAD + 14 + n_max * (ITEM_H + GAP) + 6
P1_H = 72 + lane_h + 16
d.panel(y0, P1_H, 1, "Which hook runs where, and in which order?",
        "Steps 1-7 shape the VllmConfig once. Steps 8-16 repeat in each process that needs the part.")

LX = [40, 390, 740]
LW = 320
ly = y0 + 70
for x, (name, details, items) in zip(LX, lanes):
    d.lane(x, ly, LW, lane_h, GRAY, name, details)
    iy = ly + HEAD + 14
    for n, title, where, col in items:
        d.rect(x + 12, iy, LW - 24, ITEM_H, fill=col[1], stroke=col[0], rx=8, sw=1.3)
        d.badge(x + 32, iy + ITEM_H / 2, n, col[0], r=11, size=11)
        d.text(x + 52, iy + 19, title, 12.5, INK, "bold", mono=True)
        d.text(x + 52, iy + 36, where, 11.5, MUTE, mono=True)
        iy += ITEM_H + GAP

for i in range(2):
    xa = LX[i] + LW
    d.arrow(xa + 2, ly + 32, LX[i + 1] - 2, ly + 32)

y = y0 + P1_H + 16
d.callout(30, y, 1040, "TP = 1:", "the uni executor runs steps 11, 13-15 inside the engine core process "
          "(uniproc_executor.py:68-75); step 12 does not run (one rank).", AMBER, h=36)
y += 52
d.legend(30, y, [(DET, "platform detection"), (CONF, "config hook"), (PICK, "component picker"),
                 (KER, "kernel hook"), (GRAY, "start-up step")])
y += 34
y = d.footnotes(30, y, [
    "Not shown: import_kernels (when vllm._custom_ops is imported), get_punica_wrapper (LoRA only),",
    "  compile pickers (get_pass_manager_cls, get_static_graph_wrapper_cls, get_compile_backend) and log_warnings.",
    "Order inside the worker: init_worker (11), init_device (12), load_model (13-14), then step 15 (multiproc_executor.py:672-702).",
])
d.save(height=y + 8)
