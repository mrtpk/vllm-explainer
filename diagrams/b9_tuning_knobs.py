"""Where each tuning knob acts: memory (start-up), scheduler (each step), execution and front end.

Used by: src/concepts/tuning.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/v1/worker/utils.py:724-726               requested = ceil(total x gpu_memory_utilization)
  vllm/v1/worker/gpu_worker.py:653-679, 744-748 --kv-cache-memory-bytes skips the profile; available KV memory
  vllm/v1/core/kv_cache_planning.py:1610-1611   num_blocks = available // bytes_per_block
  vllm/v1/core/kv_cache_utils.py:1015-1108      maximum concurrency log line
  vllm/engine/arg_utils.py:2958-2993, 3171-3175 max_num_seqs / max_num_batched_tokens defaults, throughput mode
  vllm/v1/core/sched/scheduler.py:674-689       long_prefill_token_threshold cap
  vllm/config/vllm.py:2439-2449, 2552-2556      capture sizes; interactivity mode sizes 1..32
  vllm/config/scheduler.py:216                  stream_interval (front-end output processor)
  vllm/entrypoints/cli/serve.py:113-130         --api-server-count default
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, VIOLET, Diagram  # noqa: E402

MEM, SCHED, EXEC, FRONT = EMERALD, VIOLET, BLUE, AMBER

d = Diagram(width=1100)
d.header("Where each vLLM tuning knob acts",
         "vllm serve  ·  one engine core  ·  the knob, the place in the code, and what it changes",
         params="defaults for vllm serve on one 80 GiB H100: util 0.92 · 8192 tokens · 1024 seqs · block 16 · -O2")

y0 = 118
P1 = 196
d.panel(y0, P1, 1, "Start-up: how many KV blocks exist?",
        "Memory knobs are read once. They fix the block pool and so the maximum concurrency.")
bx = [40, 384, 728]
bw, bh = 332, 112
by = y0 + 70
d.box(bx[0], by, bw, bh, "GPU memory budget", [
    "--gpu-memory-utilization 0.92",
    "requested = ceil(total x util)",
    "--kv-cache-memory-bytes N:",
    "no memory profiling, util ignored"], col=MEM, align="left")
d.box(bx[1], by, bw, bh, "Bytes per block", [
    "--kv-cache-dtype fp8: 1 byte, not 2",
    "--block-size 16 (backend may change)",
    "--quantization: smaller weights",
    "-tp N: weights / N, KV heads / N"], col=MEM, align="left")
d.box(bx[2], by, bw, bh, "Blocks and concurrency", [
    "blocks = available // bytes_per_block",
    "--max-model-len L must fit",
    "log: Maximum concurrency for",
    "L tokens per request: X.XXx"], col=MEM, align="left")
d.arrow(bx[0] + bw, by + bh / 2, bx[1] - 2, by + bh / 2, MEM)
d.arrow(bx[1] + bw, by + bh / 2, bx[2] - 2, by + bh / 2, MEM)

y1 = y0 + P1 + 18
P2 = 196
d.panel(y1, P2, 2, "Each step: which requests run, and how many tokens?",
        "Scheduler knobs shape every batch: they trade TTFT against ITL and throughput.")
by = y1 + 70
d.box(bx[0], by, bw, bh, "Batch width", [
    "--max-num-seqs 1024",
    "running requests per step",
    "also sizes runner buffers",
    "and the CUDA graph sizes"], col=SCHED, align="left")
d.box(bx[1], by, bw, bh, "Token budget per step", [
    "--max-num-batched-tokens 8192",
    "decode tokens first, then prefill",
    "--long-prefill-token-threshold 0",
    "--performance-mode throughput: x2"], col=SCHED, align="left")
d.box(bx[2], by, bw, bh, "Order", [
    "--scheduling-policy fcfs",
    "priority: lower value first",
    "preemption = recompute",
    "--enable-prefix-caching: on"], col=SCHED, align="left")

y2 = y1 + P2 + 18
P3 = 196
d.panel(y2, P3, 3, "Execution and front end: how fast is one step, and who feeds it?",
        "These knobs change start-up time, step time or CPU work, not the block count.")
by = y2 + 70
d.box(bx[0], by, bw, bh, "Compile and CUDA graphs", [
    "-O2: FULL_AND_PIECEWISE graphs",
    "--enforce-eager / -O0: fast start,",
    "slower steps, no graph memory",
    "interactivity: sizes 1..32"], col=EXEC, align="left")
d.box(bx[1], by, bw, bh, "Fewer steps per token", [
    "--speculative-config",
    "drafts k tokens, verifies in 1 step",
    "async scheduling: on by default",
    "(CPU work overlaps the GPU)"], col=EXEC, align="left")
d.box(bx[2], by, bw, bh, "Front-end CPU", [
    "--api-server-count N processes",
    "--renderer-num-workers 1 thread",
    "--stream-interval 1 token",
    "-dp N: N engine cores"], col=FRONT, align="left")

y = y2 + P3 + 24
d.legend(30, y, [(MEM, "memory: fixed at start-up"), (SCHED, "scheduler: per step"),
                 (EXEC, "model execution"), (FRONT, "API server process")])
y += 40
y = d.footnotes(30, y, [
    "Defaults: 0.92 (config/cache.py:212), 8192 / 1024 for >= 70 GiB non-A100 GPUs (engine/arg_utils.py:2958-2993), -O2 (config/vllm.py:434).",
    "Simplified: hybrid models have several KV cache groups; MoE and multimodal models add more knobs. See the page text.",
])
d.save(height=y + 14)
