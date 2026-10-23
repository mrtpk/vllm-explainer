"""shm_broadcast.py: the ShmRingBuffer layout, one chunk's bytes, and the flag states.

Used by: src/code/distributed-device-communicators-shm-broadcast.html
Sources (at the pinned commit in explainer.json):
  vllm/distributed/device_communicators/shm_broadcast.py:265-325   ShmRingBuffer layout and flag states (docstring)
  vllm/distributed/device_communicators/shm_broadcast.py:717-786   acquire_write: wait, clear written, write, reset reads, set written
  vllm/distributed/device_communicators/shm_broadcast.py:870-898   acquire_read: set own read flag
  vllm/distributed/device_communicators/shm_broadcast.py:900-966   enqueue: pickle, OOB buffers >= 1 MiB, overflow to XPUB
  vllm/v1/executor/multiproc_executor.py:152, 165-172             executor queue: 16 MiB chunks (VLLM_MQ_MAX_CHUNK_BYTES_MB)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, LINE, MUTE, PINK, VIOLET, WHITE, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("MessageQueue: a lock-free ring of chunks in /dev/shm",
         "the executor's broadcast queue with TP = 2 on one node  ·  1 writer (engine core), 2 readers (workers)",
         params="max_chunks = 10  ·  max_chunk_bytes = 16 MiB  ·  metadata = 10 × (1 + 2) bytes")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 196
d.panel(y0, P1_H, 1, "What is in the shared memory segment?",
        "Ten data chunks, then one flag row per chunk: a written flag and one read flag per reader.")
CY = y0 + 78
cw, gap = 92, 6
for i in range(10):
    x = 46 + i * (cw + gap)
    cur = i == 3
    d.rect(x, CY, cw, 44, fill=BLUE[1] if i < 3 else WHITE, stroke=BLUE[0] if i < 4 else LINE, rx=3,
           sw=2.6 if cur else 1.2)
    d.text(x + cw / 2, CY + 18, f"chunk {i}", 11.5, INK, "bold", "middle")
    d.text(x + cw / 2, CY + 34, "16 MiB", 11, MUTE, anchor="middle", mono=True)
    flags = ["111", "111", "110", "000"][i] if i < 4 else "000"
    d.text(x + cw / 2, CY + 66, flags, 12, VIOLET[0], "bold", "middle", mono=True)
d.text(46 + 3 * (cw + gap) + cw / 2, CY - 8, "▼ writer current_idx", 11.5, BLUE[0], "bold", "middle")
d.text(46, CY + 92, "flag bytes per chunk: [written, read by worker 0, read by worker 1]   ·   data part = 160 MiB,"
       " metadata = 30 bytes", 11.5, MUTE, italic=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 196
d.panel(y1, P2_H, 2, "What does one enqueue() write?",
        "Pickle the object; big CPU buffers go out of band; everything fits in one chunk or goes over ZMQ.")
BY = y1 + 70
BW = 330
d.box(46, BY, BW, 100, "1  pickle (protocol 5)", [
    "CPU tensors → PickleBuffer",
    "buffers < 1 MiB: inline",
    "buffers ≥ 1 MiB: out of band",
], col=VIOLET, align="start")
d.box(385, BY, BW, 100, "2a  fits in the chunk", [
    "[0] = 0 (no overflow)",
    "[1:3] = buffer count (big endian)",
    "then [4-byte len][bytes] each",
], col=BLUE, align="start")
d.box(724, BY, BW, 100, "2b  too large", [
    "[0] = 1 (overflow)",
    "all buffers → XPUB socket",
    "readers recv_multipart()",
], col=AMBER, align="start")
d.arrow(376, BY + 50, 383, BY + 50, VIOLET)
d.arrow(715, BY + 50, 722, BY + 50, VIOLET, dash="5 4")

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 150
d.panel(y2, P3_H, 3, "How do the flags of one chunk change?",
        "The writer resets the read flags before it sets the written flag, so a reader never sees a half state.")
SY = y2 + 74
states = [("0??", "writable", BLUE), ("100", "written", EMERALD), ("110", "worker 0 read", EMERALD),
          ("111", "all read: writable", BLUE)]
sx = 46
for k, (bits, label, col) in enumerate(states):
    d.chip(sx, SY, 120, bits, col)
    d.text(sx + 60, SY + 50, label, 11.5, MUTE, anchor="middle")
    if k < 3:
        d.arrow(sx + 128, SY + 15, sx + 266, SY + 15, INK)
        lbl = ["writer writes", "worker 0 reads", "worker 1 reads"][k]
        d.text(sx + 197, SY + 5, lbl, 11.5, INK, anchor="middle")
    sx += 276

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "writer side"), (EMERALD, "reader side"), (VIOLET, "flags / pickling"),
                 (AMBER, "ZMQ fallback")])
y += 36
y = d.footnotes(30, y, [
    "Flag states as in the ShmRingBuffer docstring (case 1-4). Readers on other nodes never touch the ring: they read",
    "  every message from a TCP XPUB socket. Wake-ups use a separate ZMQ PUB/SUB notify socket (SpinCondition).",
])
d.save(height=y + 14)
