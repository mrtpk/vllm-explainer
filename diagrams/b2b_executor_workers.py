"""B2b: engine core -> executor -> workers, with the shared-memory message queues.

Used by: src/concepts/executors-and-workers.html, src/code/v1-executor-multiproc-executor.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/executor/abstract.py:63-103            Executor.get_class: ray / mp / uni / external_launcher / class
  vllm/config/parallel.py:1055-1100              mp when world_size_across_dp > 1, uni when world_size == 1
  vllm/v1/executor/uniproc_executor.py:51-75, 90-117   UniProcExecutor: worker in this process, direct call
  vllm/v1/executor/multiproc_executor.py:165-171  rpc_broadcast_mq = MessageQueue(world_size, local_world_size)
  vllm/v1/executor/multiproc_executor.py:191-216  one WorkerProc per local rank, wait_for_ready
  vllm/v1/executor/multiproc_executor.py:383-456  collective_rpc: enqueue once, read the reply of output_rank
  vllm/v1/executor/multiproc_executor.py:549-563  output_rank = world_size - TP * PCP
  vllm/v1/executor/multiproc_executor.py:300-331  MultiprocWorkerMonitor thread
  vllm/v1/executor/multiproc_executor.py:615-625  per-worker response MessageQueue(1, 1)
  vllm/v1/executor/multiproc_executor.py:690-699, 1024-1048  WorkerAsyncOutputCopy thread, worker_busy_loop
  vllm/v1/executor/multiproc_executor.py:835-859  DeathPipeMonitor thread
  vllm/v1/executor/ray_executor_v2.py:239-247     RayExecutorV2 subclasses MultiprocExecutor
  vllm/envs.py:1856-1858                          VLLM_MQ_MAX_CHUNK_BYTES_MB = 16
  vllm/distributed/device_communicators/shm_broadcast.py:491-492   max_chunks = 10
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Executors and workers: how the engine core reaches the GPUs",
         "vllm serve on one node  ·  Model Runner V2  ·  async scheduling on (default)",
         params="rpc_broadcast_mq: 10 chunks x 16 MiB shared memory  ·  reply queue per worker: MessageQueue(1, 1)")

XS = [46, 410, 774]
BW = 280

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 214
d.panel(y0, P1_H, 1, "Which executor runs the workers?",
        "ParallelConfig picks the backend; Executor.get_class() maps the name to a class.")
BY = y0 + 72
BH = 118
d.box(XS[0], BY, BW, BH, "uni: UniProcExecutor", [
    "default only if TP=PP=DP=1",
    "worker object in the engine",
    "  core process",
    "collective_rpc = direct call",
    "no extra process, no queue",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, BH, "mp: MultiprocExecutor", [
    "default when world_size_across_dp",
    "  > 1 on one node",
    "one VllmWorker-<rank> process",
    "  per local rank (spawn)",
    "shared-memory MessageQueue",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, BH, "ray: RayExecutorV2", [
    "subclass of MultiprocExecutor",
    "workers are Ray actors",
    "same MessageQueue control path",
    "picked: DP backend ray, or a",
    "  Ray placement group",
], col=PINK, align="start", mono_size=11.5)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 300
d.panel(y1, P2_H, 2, "How does one execute_model call travel with TP = 2 (mp)?",
        "The executor writes the call once into shared memory. Every worker reads it; only output_rank replies.")
EY = y1 + 76
d.box(XS[0], EY, BW, 196, "Engine core process", [
    "busy loop (main thread)",
    "  executor.execute_model(",
    "    scheduler_output,",
    "    non_block=True)",
    "MultiprocExecutor.collective_rpc",
    "  rpc_broadcast_mq.enqueue(",
    "   (method, args, kwargs, 0))",
    "  FutureWrapper -> futures_queue",
    "thread MultiprocWorkerMonitor",
    "  waits on process sentinels",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], EY, BW, 92, "rpc_broadcast_mq (1 writer)", [
    "ShmRingBuffer, 2 local readers",
    "pickle; >= 16 MiB -> ZMQ XPUB",
    "readers: rank 0 and rank 1",
], col=AMBER, align="start", mono_size=11.5)
d.box(XS[1], EY + 112, BW, 84, "response MQ of rank 0", [
    "MessageQueue(1, 1)",
    "(SUCCESS, ModelRunnerOutput)",
    "read by FutureWrapper.result()",
], col=AMBER, align="start", mono_size=11.5)
d.box(XS[2], EY, BW, 92, "VllmWorker-1  (GPU 1)", [
    "TP rank 1",
    "same call, same inputs",
    "no reply (rank != output_rank)",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], EY + 112, BW, 84, "VllmWorker-0  (GPU 0)", [
    "TP rank 0 = output_rank",
    "worker_busy_loop: dequeue",
    "  -> Worker.execute_model",
], col=EMERALD, align="start", mono_size=11.5)
# engine core -> broadcast mq
d.arrow(XS[0] + BW, EY + 46, XS[1] - 2, EY + 46, VIOLET, sw=2)
d.text((XS[0] + BW + XS[1]) / 2, EY + 36, "enqueue", 11.5, VIOLET, "bold", "middle")
# broadcast mq -> both workers
d.arrow(XS[1] + BW, EY + 46, XS[2] - 2, EY + 46, AMBER, sw=2)
d.path_arrow(f"M {XS[1] + BW + 30},{EY + 46} L {XS[1] + BW + 30},{EY + 132} L {XS[2] - 2},{EY + 132}", AMBER)
d.text((XS[1] + BW + XS[2]) / 2, EY + 36, "dequeue", 11.5, AMBER, "bold", "middle")
# worker 0 -> its response mq
d.arrow(XS[2], EY + 176, XS[1] + BW + 2, EY + 176, EMERALD, dash="5 4")
d.text((XS[1] + BW + XS[2]) / 2, EY + 168, "reply", 11.5, EMERALD, "bold", "middle")
# response mq -> engine core
d.arrow(XS[1], EY + 160, XS[0] + BW + 2, EY + 160, EMERALD)
d.text((XS[0] + BW + XS[1]) / 2, EY + 150, "result", 11.5, EMERALD, "bold", "middle")
# NCCL between workers: label right of the worker column
d.text(XS[2] + BW - 6, EY + 106, "NCCL all-reduce in forward", 11, MUTE, "normal", "end", italic=True)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 214
d.panel(y2, P3_H, 3, "Which threads run inside one worker process?",
        "The main thread runs the calls. Two helper threads move results and watch the parent.")
QY = y2 + 72
QH = 118
d.box(XS[0], QY, BW, QH, "Main thread", [
    "worker_busy_loop()",
    "rpc_broadcast_mq.dequeue()",
    "getattr(worker, method)(...)",
    "if output_rank is None or",
    "  rank == output_rank: reply",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[1], QY, BW, QH, "WorkerAsyncOutputCopy", [
    "only with async scheduling",
    "async_output_queue.get()",
    "AsyncOutput.get_output()",
    "  (waits for the D2H copy)",
    "worker_response_mq.enqueue()",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], QY, BW, QH, "DeathPipeMonitor", [
    "death_pipe.recv() blocks",
    "EOFError = parent exited",
    "shut down both queues",
    "Parent side: the executor's",
    "  MultiprocWorkerMonitor",
], col=EMERALD, align="start", mono_size=11.5)

y = y2 + P3_H + 24
d.legend(30, y, [(VIOLET, "engine core"), (AMBER, "shared-memory queue"), (EMERALD, "worker"),
                 (BLUE, "mp backend"), (PINK, "Ray backend")])
y += 40
y = d.footnotes(30, y, [
    "With a KV or EC connector, every worker replies and an aggregator merges the outputs (multiproc_executor.py:405-419).",
    "Not shown: multi-node (remote readers over ZMQ), PP stages, the ready and death pipes at start-up.",
])
d.save(height=y + 14)
