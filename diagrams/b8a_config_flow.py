"""How vllm serve turns flags and environment variables into one VllmConfig, and where the config goes.

Used by: src/concepts/config-system.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/utils/argparse_utils.py:313-502        parse_args: --config YAML, _ -> -, -O<n>, dotted keys -> JSON
  vllm/utils/argparse_utils.py:516-604        --config file is inserted after the subcommand (cli > file > defaults)
  vllm/engine/arg_utils.py:324-453            get_kwargs: type, default and help from each config field
  vllm/engine/arg_utils.py:1961-1970          EngineArgs.from_cli_args copies the parsed values
  vllm/engine/arg_utils.py:2243-2250          logging, pre_register_and_update, DeviceConfig, validate_environ
  vllm/engine/arg_utils.py:2269-2275          ModelConfig first, then model-based defaults
  vllm/engine/arg_utils.py:2296, 2558, 2631-2672   CacheConfig, ParallelConfig, batch defaults, SchedulerConfig
  vllm/engine/arg_utils.py:2870-2903          VllmConfig(...)
  vllm/config/vllm.py:1492-2318               VllmConfig.__post_init__ phases
  vllm/v1/engine/utils.py:256-288             engine core process gets vllm_config as a spawn argument
  vllm/v1/engine/core.py:1336-1337            engine core runs __post_init__ again after the handshake
  vllm/v1/executor/multiproc_executor.py:731-750   worker process gets vllm_config as a spawn argument
  vllm/v1/engine/core.py:241-243, multiproc_executor.py:708-710   enable_envs_cache
  vllm/v1/engine/core_client.py:847-879       ready response: max_model_len, num_gpu_blocks, block_size back
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

SRC, CONF, PROC, ENV = BLUE, VIOLET, EMERALD, AMBER

d = Diagram(width=1100)
d.header("From flags to VllmConfig to every process",
         "vllm serve with one API server  ·  the front end builds the config; the other processes get a pickled copy",
         params="CLI + YAML -> Namespace -> EngineArgs -> create_engine_config() -> VllmConfig -> pickled to children")

# ---------------- Panel 1 ----------------
y0 = 118
P1_H = 214
d.panel(y0, P1_H, 1, "Where do the values come from?",
        "The parser rewrites the command line, then argparse fills a Namespace with the flag defaults.")
by = y0 + 72
BH = 118
d.box(46, by, 310, BH, "Command line (+ YAML)", [
    "vllm serve <model> -tp 2",
    "-O1  -cc.mode=3",
    "--max_num_seqs 64",
    "--config serve.yaml",
], col=SRC, align="start")
d.arrow(356, by + BH / 2, 392, by + BH / 2, SRC)
d.box(394, by, 330, BH, "FlexibleArgumentParser", [
    "YAML keys -> flags after 'serve'",
    "_ -> - in flag names",
    "-O1 -> --optimization-level 1",
    "-cc.mode=3 -> -cc '{\"mode\": 3}'",
], col=SRC, align="start")
d.arrow(724, by + BH / 2, 760, by + BH / 2, CONF)
d.box(762, by, 292, BH, "EngineArgs (flat)", [
    "from_cli_args(Namespace)",
    "unset tuning flags = None",
    "flag defaults = field defaults",
    "arg_utils.py:1961-1970",
], col=CONF, align="start")

# ---------------- Panel 2 ----------------
y1 = y0 + P1_H + 18
P2_H = 384
d.panel(y1, P2_H, 2, "How does EngineArgs become one VllmConfig?",
        "create_engine_config() builds the sub-configs in this order; __post_init__ then fixes the rest.")
CW, CH = 322, 86
cx = [46, 389, 732]
row1 = y1 + 70
cards1 = [
    (1, "platform + device", ["pre_register_and_update()", "DeviceConfig, validate_environ", "arg_utils.py:2243-2250"]),
    (2, "ModelConfig first", ["HF config, max_model_len", "registry: architecture", "arg_utils.py:2269"]),
    (3, "defaults from the model", ["chunked prefill, prefix cache", "None -> model support", "arg_utils.py:2275"]),
]
cards2 = [
    (4, "Cache + Parallel", ["CacheConfig(block_size=None)", "ParallelConfig -> uni / mp", "arg_utils.py:2296, 2558"]),
    (5, "batch defaults", ["usage context + GPU memory", "max_num_seqs, token budget", "arg_utils.py:2631-2635"]),
    (6, "VllmConfig(...)", ["Scheduler, LoRA, Load, ...", "34 attributes in one object", "arg_utils.py:2870"]),
]
for row_y, cards in ((row1, cards1), (row1 + CH + 22, cards2)):
    for x, (n, title, rows) in zip(cx, cards):
        d.rect(x, row_y, CW, CH, fill=CONF[1], stroke=CONF[0], rx=8, sw=1.4)
        d.badge(x + 22, row_y + 20, n, CONF[0], r=11, size=11)
        d.text(x + 42, row_y + 25, title, 13.5, INK, "bold")
        for i, s in enumerate(rows):
            d.text(x + 16, row_y + 46 + i * 15, s, 11.5, INK if i < 2 else MUTE, mono=True)
    for i in range(2):
        d.arrow(cx[i] + CW, row_y + CH / 2, cx[i + 1] - 2, row_y + CH / 2, CONF)
# link row 1 -> row 2
ry2 = row1 + CH + 22
d.path_arrow(f"M {cx[2] + CW / 2:.0f},{row1 + CH:.0f} L {cx[2] + CW / 2:.0f},{row1 + CH + 10:.0f} "
             f"L {cx[0] + CW / 2:.0f},{row1 + CH + 10:.0f} L {cx[0] + CW / 2:.0f},{ry2 - 2:.0f}", CONF)

# __post_init__ wide box
py = ry2 + CH + 22
PH = 104
d.rect(46, py, 1008, PH, fill=CONF[1], stroke=CONF[0], rx=8, sw=1.4)
d.badge(68, py + 20, 7, CONF[0], r=11, size=11)
d.text(88, py + 25, "VllmConfig.__post_init__  (config/vllm.py:1492-2318)", 13.5, INK, "bold")
phases = [
    "model hooks (1521)  ->  async scheduling decision (1664-1698)  ->  --enforce-eager (1745-1763)",
    "apply_config_platform_defaults (1804)  ->  compile mode from -O (1806-1810)  ->  level defaults (1833-1834)",
    "CUDA graph sizes (1992)  ->  check_and_update_config (2046)  ->  KV transfer, hybrid KV manager (2199-2273)",
]
for i, s in enumerate(phases):
    d.text(68, py + 50 + i * 17, s, 11.5, INK, mono=True)

# ---------------- Panel 3 ----------------
y2 = y1 + P2_H + 18
P3_H = 262
d.panel(y2, P3_H, 3, "How does the config reach the other processes?",
        "As a pickled argument of each spawned process. Environment variables are inherited and read separately.")
bx = [46, 410, 774]
BW, BH3 = 280, 136
b3y = y2 + 72
d.box(bx[0], b3y, BW, BH3, "API server (front end)", [
    "builds VllmConfig once",
    "env vars: read live,",
    "  no enable_envs_cache()",
    "gets max_model_len,",
    "  num_gpu_blocks back",
], col=PROC, align="start")
d.box(bx[1], b3y, BW, BH3, "Engine core process", [
    "VllmConfig = pickled copy",
    "__post_init__ runs again",
    "  after the handshake",
    "env cache: on after init",
    "  (core.py:243)",
], col=PROC, align="start")
d.box(bx[2], b3y, BW, BH3, "Worker process (TP > 1)", [
    "VllmConfig = pickled copy",
    "TP = 1: no process, the",
    "  worker is in the engine core",
    "env cache: on after init",
    "  (multiproc_executor.py:710)",
], col=PROC, align="start")
for i in range(2):
    ax0 = bx[i] + BW
    ax1 = bx[i + 1] - 2
    d.arrow(ax0, b3y + 46, ax1, b3y + 46, PROC)
d.arrow_label((bx[0] + BW + bx[1]) / 2, b3y + 46, "spawn", PROC)
d.arrow_label((bx[1] + BW + bx[2]) / 2, b3y + 46, "spawn", PROC)
d.arrow(bx[1], b3y + 100, bx[0] + BW + 2, b3y + 100, PROC, dash="5 4")
d.arrow_label((bx[0] + BW + bx[1]) / 2, b3y + 100, "ready", PROC)
ey = b3y + BH3 + 14
d.rect(46, ey, 1008, 30, fill=ENV[1], stroke=ENV[0], rx=6, sw=1.2)
d.text(60, ey + 20, "Environment: each child inherits os.environ at spawn  ·  envs.X calls a lambda on every read until enable_envs_cache()",
       11.5, INK, mono=True)

y = y2 + P3_H + 20
d.legend(30, y, [(SRC, "command line and parser"), (CONF, "config objects and steps"),
                 (PROC, "processes"), (ENV, "environment variables")])
y += 34
y = d.footnotes(30, y, [
    "Line numbers are in vllm/engine/arg_utils.py and vllm/config/vllm.py unless named. Not shown: Ray, several API servers,",
    "  data parallelism, and the per-field checks of each config class. The -tp 2 example only names the worker box.",
])
d.save(height=y + 8)
