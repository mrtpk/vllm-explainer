"""B2b: how Worker.determine_available_memory() splits the GPU memory (worked example).

Used by: src/code/v1-worker-gpu-worker.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/worker/gpu_worker.py:528-541      init snapshot after NCCL + cuBLAS init; requested_memory
  vllm/v1/worker/utils.py:724-726           requested_memory = ceil(total_memory * gpu_memory_utilization)
  vllm/utils/mem_utils.py:312-321           total_consumed, transient_peak_headroom, non_kv_cache_memory
  vllm/v1/worker/gpu_worker.py:682-697      memory_profiling + profile_run
  vllm/v1/worker/gpu_worker.py:704-715      CUDA graph estimate (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1)
  vllm/v1/worker/gpu_worker.py:744-748      available = requested - non_kv_cache - cudagraph estimate
  vllm/v1/core/kv_cache_planning.py:678-681 num_blocks = available_memory // bytes_per_block
  vllm/config/cache.py:212                  gpu_memory_utilization default 0.92
All GiB values are illustrative (an 8B BF16 model on an 80 GiB GPU); the formula is from the code.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, LINE, MUTE, NEUTRAL, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Memory profiling: what is left for the KV cache",
         "Worked example (illustrative numbers): 8B model in BF16, one 80 GiB GPU, default --gpu-memory-utilization",
         params="gpu_memory_utilization = 0.92  ·  block_size = 16  ·  32 layers x 8 KV heads x 128 dims")

X0 = 46
BARW = 1008
SCALE = BARW / 80.0  # px per GiB
BH = 40


def seg(x, gib, y, col, label):
    w = gib * SCALE
    d.rect(x, y, w, BH, fill=col[1] if isinstance(col, tuple) else "#ffffff",
           stroke=col[0] if isinstance(col, tuple) else col, rx=3, sw=1.3)
    if label and w > 60:
        d.text(x + w / 2, y + BH / 2 + 4.5, label, 12, col[0] if isinstance(col, tuple) else INK,
               "bold", "middle", mono=True)
    return x + w


# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 324
d.panel(y0, P1_H, 1, "How is the memory split?",
        "vLLM may use 0.92 of the total. Weights, buffers, activations and CUDA graphs come first; the rest is KV cache.")
y = y0 + 84
d.text(X0, y - 8, "total_memory = 80 GiB", 12, MUTE, "bold", mono=True)
x = seg(X0, 73.6, y, BLUE, "requested_memory = ceil(80 x 0.92) = 73.6 GiB")
seg(x, 6.4, y, NEUTRAL, "not used")

y2 = y + BH + 46
d.text(X0, y2 - 8, "requested_memory = 73.6 GiB, split by the profile run", 12, MUTE, "bold", mono=True)
x = X0
xa = x
x = seg(x, 16.0, y2, VIOLET, "16.0")
xb = x
x = seg(x, 2.5, y2, AMBER, "")
xc = x
x = seg(x, 0.5, y2, PINK, "")
xd = x
x = seg(x, 54.6, y2, EMERALD, "KV cache: 54.6 GiB")
# labels under the small segments
ly = y2 + BH + 18
d.line((xa + xb) / 2, y2 + BH, (xa + xb) / 2, ly - 12, VIOLET[0], 1)
d.text(xa, ly + 2, "total_consumed: weights 15.0", 11.5, VIOLET, "bold", mono=True)
d.text(xa, ly + 18, "+ buffers, non-torch 1.0", 11.5, VIOLET, "bold", mono=True)
d.line((xc + xd) / 2, y2 + BH, (xc + xd) / 2, ly + 34, PINK[0], 1)
d.text((xc + xd) / 2 + 8, ly + 46, "CUDA graph estimate 0.5", 11.5, PINK, "bold", mono=True)
d.line((xb + xc) / 2, y2 + BH, (xb + xc) / 2, ly + 62, AMBER[0], 1)
d.text((xb + xc) / 2 + 8, ly + 74, "transient peak headroom 2.5 (activations of the profile run)", 11.5, AMBER, "bold", mono=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 236
d.panel(y1, P2_H, 2, "How do the numbers come out of the code?",
        "Three measurements, one subtraction, one division.")
ty = y1 + 84
rows = [
    "requested_memory      = ceil(total_memory x gpu_memory_utilization)      = 73.6 GiB",
    "total_consumed        = free(init snapshot) - free(after profile run)    = 16.0 GiB",
    "peak headroom         = torch_peak - torch_allocated (after profile run) =  2.5 GiB",
    "non_kv_cache_memory   = total_consumed + peak headroom                   = 18.5 GiB",
    "available_kv_cache    = 73.6 - 18.5 - 0.5 (CUDA graph estimate)          = 54.6 GiB",
    "bytes_per_block       = 32 layers x 2 (K, V) x 16 x 8 x 128 x 2 bytes    =  2 MiB",
    "num_blocks            = 54.6 GiB // 2 MiB                                = 27,955 blocks",
    "KV capacity           = 27,955 x 16                                      = 447,280 tokens",
]
for i, r in enumerate(rows):
    d.text(X0, ty + i * 19, r, 12, INK, mono=True)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "budget"), (VIOLET, "consumed"), (AMBER, "peak headroom"), (PINK, "CUDA graphs"),
                 (EMERALD, "KV cache")])
y += 40
y = d.footnotes(30, y, [
    "Illustrative values. Real values come from the log line \"Available KV cache memory: ... GiB\" (gpu_worker.py:763-766).",
    "The budget is a share of the total memory: other processes must fit in the other 6.4 GiB.",
    "If free memory at start-up is below 73.6 GiB, start-up fails (v1/worker/utils.py:761-770).",
])
d.save(height=y + 14)
