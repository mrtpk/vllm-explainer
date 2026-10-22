"""B2b: the engine core process: three threads, two queues and the batch queue.

Used by: src/concepts/engine-core-loop.html, src/code/v1-engine-core.html
Sources (at the pinned commit in explainer.json):
  vllm/v1/engine/core.py:1172-1176     input_queue, output_queue, executor_fail_callback -> EXECUTOR_FAILED
  vllm/v1/engine/core.py:1238-1265     input thread and output thread (daemon threads)
  vllm/v1/engine/core.py:1921-2028     process_input_sockets: DEALER sockets, decode, preprocess_add_request, queues
  vllm/v1/engine/core.py:2030-2096     process_output_sockets: output_queue.get, encode_into, PUSH to client_index
  vllm/v1/engine/core.py:1609-1620     run_busy_loop
  vllm/v1/engine/core.py:1635-1664     _process_input_queue: blocking get while there is no work
  vllm/v1/engine/core.py:1666-1684     _process_engine_step: step_fn, outputs into output_queue
  vllm/v1/engine/core.py:205-211, 227-229   batch queue deque(maxlen=max_concurrent_batches), step_fn choice
  vllm/v1/engine/core.py:698-812       step_with_batch_queue
  vllm/config/vllm.py:587-597          max_concurrent_batches = pp_size + 1 with async scheduling and MRV2
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The engine core process: three threads and two queues",
         "One engine core process (EngineCoreProc) with TP = 1  ·  async scheduling on (default)",
         params="input_queue, output_queue: queue.Queue  ·  batch_queue: deque(maxlen=2)")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 268
d.panel(y0, P1_H, 1, "Which thread does what?",
        "The I/O threads move bytes and decode them. Only the main thread touches the scheduler and the GPU.")
BY = y0 + 76
BH = 170
XS = [46, 410, 774]
BW = 280
d.box(XS[0], BY, BW, BH, "Input thread", [
    "process_input_sockets()",
    "ZMQ DEALER <- ROUTER (front end)",
    "poll, recv_multipart",
    "ADD: decode EngineCoreRequest",
    "   -> preprocess_add_request()",
    "   -> Request + block hashes",
    "ABORT: also into aborts_queue",
    "UTILITY: decode (id, method, args)",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, BH, "Main thread: the busy loop", [
    "run_busy_loop()",
    "1 _process_input_queue()",
    "    blocks while there is no work",
    "    add / abort / utility calls",
    "2 _process_engine_step()",
    "    step_fn = step_with_batch_queue",
    "    outputs -> output_queue",
    "3 post_step()",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, BH, "Output thread", [
    "process_output_sockets()",
    "output_queue.get()  (blocks)",
    "(client_index, EngineCoreOutputs)",
    "MsgpackEncoder.encode_into()",
    "ZMQ PUSH -> PULL (front end)",
    "sockets[client_index]",
    "-1: to the DP coordinator",
    "ENGINE_CORE_DEAD: send, stop",
], col=EMERALD, align="start", mono_size=11.5)
ay = BY + 60
d.arrow(XS[0] + BW, ay, XS[1] - 2, ay, BLUE, sw=2)
d.text((XS[0] + BW + XS[1]) / 2, ay - 10, "input_queue", 11, BLUE, "bold", "middle", mono=True)
d.arrow(XS[1] + BW, ay, XS[2] - 2, ay, EMERALD, sw=2)
d.text((XS[1] + BW + XS[2]) / 2, ay - 10, "output_queue", 11, EMERALD, "bold", "middle", mono=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 262
d.panel(y1, P2_H, 2, "What does step_with_batch_queue do on each call?",
        "It schedules and starts step N+1 first. Only then does it wait for step N and update the scheduler.")
CY = y1 + 76
CH = 160
d.box(XS[0], CY, BW, CH, "Call 1 (queue empty)", [
    "schedule() -> step 1",
    "execute_model(non_block=True)",
    "sample_tokens(non_block=True)",
    "appendleft(step 1)",
    "queue: [1]   (1 < 2)",
    "return None: no outputs yet",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], CY, BW, CH, "Call 2 (queue now full)", [
    "schedule() -> step 2, start it",
    "appendleft(step 2)",
    "queue: [2, 1]   (full)",
    "pop() -> step 1",
    "future.result()  (wait for GPU)",
    "update_from_output(step 1)",
    "return outputs of step 1",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[2], CY, BW, CH, "Call 3 and later", [
    "queue at start: [2]",
    "schedule() -> step 3, start it",
    "queue: [3, 2]  -> pop() step 2",
    "wait, update_from_output(step 2)",
    "return outputs of step 2",
    "So the GPU always has the next",
    "step in its stream.",
], col=VIOLET, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, CY + CH / 2, XS[1] - 2, CY + CH / 2, INK)
d.arrow(XS[1] + BW, CY + CH / 2, XS[2] - 2, CY + CH / 2, INK)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 214
d.panel(y2, P3_H, 3, "Who else writes into the two queues?",
        "Control messages use the same queues, so they keep their order with the requests.")
QY = y2 + 72
QH = 118
d.box(XS[0], QY, BW, QH, "Into input_queue", [
    "input thread: ADD, ABORT, UTILITY",
    "executor failure callback:",
    "   EXECUTOR_FAILED",
    "signal -> wakeup callback: WAKEUP",
    "DP only: START_DP_WAVE",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[1], QY, BW, QH, "Into output_queue", [
    "busy loop: step outputs",
    "utility results (UtilityOutput)",
    "abort / error outputs",
    "DP counts: client_index -1",
    "fatal error: ENGINE_CORE_DEAD",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], QY, BW, QH, "Blocking rules", [
    "input thread: blocks in poll()",
    "busy loop: blocks in",
    "   input_queue.get() when idle",
    "   future.result() when queue full",
    "output thread: output_queue.get()",
], col=AMBER, align="start", mono_size=11.5)

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "input path"), (VIOLET, "main thread work"), (EMERALD, "output path"),
                 (AMBER, "where a thread waits")])
y += 40
y = d.footnotes(30, y, [
    "Queue size 2 = max_concurrent_batches = pp_size + 1 with async scheduling and Model Runner V2 (vllm.py:587-597).",
    "Not shown: the DP coordinator sockets, the stack-dump watchdog and fault tolerance. Uni executor: the worker runs in the main thread.",
])
d.save(height=y + 14)
