"""Processes and IPC: the three channels of vllm serve.

Used by: src/learn/processes-and-ipc.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/engine/core_client.py:672-682          front end binds ROUTER (inputs) and PULL (outputs)
  vllm/v1/engine/utils.py:1148-1151               ipc:// when all engines are local, else tcp://host:0
  vllm/v1/engine/core_client.py:749-751           engine identity = DP rank as 2 little-endian bytes
  vllm/v1/engine/core_client.py:1284, 1294        frames: (identity, type, *msgpack buffers)
  vllm/v1/engine/__init__.py:300-312              request type bytes (ADD = 0x00, ABORT = 0x01, ...)
  vllm/v1/engine/core.py:1936-1941                engine input thread: DEALER, connect, identity
  vllm/v1/engine/core.py:2047-2050                engine output thread: PUSH, connect
  vllm/utils/network_utils.py:366-367             default bind: all types except PUSH, SUB, XSUB
  vllm/v1/serial_utils.py:256-270                 tensors >= 256 B go to extra frames
  vllm/v1/executor/multiproc_executor.py:152,165-171  rpc_broadcast_mq = MessageQueue(world_size, local_world_size, 16 MiB)
  vllm/distributed/device_communicators/shm_broadcast.py:265-330  ring buffer layout
  vllm/distributed/device_communicators/shm_broadcast.py:483-493  max_chunks = 10
  vllm/distributed/device_communicators/shm_broadcast.py:900-966  enqueue: pickle; overflow -> XPUB socket
  vllm/v1/executor/multiproc_executor.py:625      one response MessageQueue(1, 1) per worker
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, INK, MUTE, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The three channels between the processes of vllm serve",
         "one node  ·  one API server  ·  one engine core (DP = 1)  ·  TP = 2, so two worker processes",
         params="channel 1: ZMQ ROUTER -> DEALER   channel 2: ZMQ PUSH -> PULL   channel 3: shared-memory MessageQueue")

BW = 300
X1, X2, X3 = 46, 400, 754

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 276
d.panel(y0, P1_H, 1, "Front end <-> engine core: two ZMQ sockets",
        "Requests go in through ROUTER -> DEALER. Outputs come back through PUSH -> PULL. Payload: msgpack.")
by = y0 + 72
BH = 150
d.box(X1, by, BW, BH, "API server process", [
    "AsyncMPClient",
    "ROUTER  bind  ipc://<tmp>/<uuid>",
    "PULL    bind  ipc://<tmp>/<uuid>",
    "MsgpackEncoder (requests)",
    "MsgpackDecoder(EngineCoreOutputs)",
], col=BLUE, align="start")
d.box(X2, by, BW, BH, "One ADD message = frames", [
    "[0] identity  b'\\x00\\x00' (DP rank 0)",
    "[1] type      b'\\x00' (ADD)",
    "[2] msgpack   EngineCoreRequest",
    "[3..] raw buffers of tensors",
    "      >= 256 B (zero copy)",
    "ROUTER removes [0] on the way",
], col=INK, align="start")
d.box(X3, by, BW, BH, "Engine core process", [
    "input thread:",
    "  DEALER connect, identity = rank",
    "busy loop (main thread)",
    "output thread:",
    "  PUSH connect",
], col=VIOLET, align="start")
d.arrow(X1 + BW, by + 50, X2 - 2, by + 50, BLUE, sw=2)
d.arrow(X2 + BW, by + 50, X3 - 2, by + 50, BLUE, sw=2)
d.badge(X1 + BW + 27, by + 30, 1, BLUE, r=10, size=11)
# output path routed under the boxes
oy = by + BH + 30
d.path_arrow(f"M {X3 + 150},{by + BH} L {X3 + 150},{oy} L {X1 + 150},{oy} L {X1 + 150},{by + BH + 2}",
             VIOLET, sw=2)
d.badge(X2 + 150, oy, 2, VIOLET, r=10, size=11)
d.text(X2 + 172, oy - 8, "EngineCoreOutputs (msgpack), PUSH -> PULL", 11.5, VIOLET, "bold")

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 276
d.panel(y1, P2_H, 2, "Engine core -> workers: a shared-memory ring buffer",
        "The executor writes one pickled RPC into a ring slot. Every worker reads the same slot. Replies use one queue per worker.")
by = y1 + 72
d.box(X1, by, BW, BH, "Engine core process", [
    "MultiprocExecutor",
    "rpc_broadcast_mq.enqueue(",
    "  (method, args, kwargs,",
    "   output_rank))",
    "pickle + out-of-band buffers",
], col=VIOLET, align="start")
d.box(X2, by, BW, BH, "ShmRingBuffer (/dev/shm)", [
    "10 chunks x 16 MiB data",
    "10 x (1 + n_reader) flag bytes",
    "flags: written | read0 | read1",
    "writer: 1 process",
    "readers: 2 workers",
    "too big -> ZMQ XPUB socket",
], col=TEAL, align="start")
d.box(X3, by, BW, BH, "Worker processes", [
    "VllmWorker-0  (rank 0, GPU 0)",
    "VllmWorker-1  (rank 1, GPU 1)",
    "worker_busy_loop():",
    "  rpc_broadcast_mq.dequeue()",
    "  getattr(worker, method)(...)",
], col=AMBER, align="start")
d.arrow(X1 + BW, by + 50, X2 - 2, by + 50, TEAL, sw=2)
d.arrow(X2 + BW, by + 50, X3 - 2, by + 50, TEAL, sw=2)
d.badge(X1 + BW + 27, by + 30, 3, TEAL, r=10, size=11)
oy = by + BH + 30
d.path_arrow(f"M {X3 + 150},{by + BH} L {X3 + 150},{oy} L {X1 + 150},{oy} L {X1 + 150},{by + BH + 2}",
             AMBER, sw=1.6, dash="5 4")
d.text(X2 + 10, oy - 8, "replies: one MessageQueue(1, 1) per worker", 11.5, AMBER, "bold")

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "front end"), (VIOLET, "engine core"), (TEAL, "shared memory"),
                 (AMBER, "workers")])
y += 40
y = d.footnotes(30, y, [
    "With TP = 1 there is no channel 3: the worker is an object inside the engine core process (uni executor).",
    "ipc:// paths are used when all engines run on this node; otherwise the sockets use tcp://. <tmp> = VLLM_RPC_BASE_PATH.",
    "Not shown: the DP coordinator (XPUB / PULL sockets, only with DP > 1) and the tensor data between GPUs (NCCL).",
])
d.save(height=y + 14)
