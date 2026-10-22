"""Data contracts: the chain of types across the processes of vllm serve.

Used by: src/concepts/data-contracts.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/openai/chat_completion/protocol.py:225   ChatCompletionRequest (pydantic)
  vllm/renderers/base.py:1380-1411                           EngineInput dict with arrival_time
  vllm/v1/engine/__init__.py:113-165                        EngineCoreRequest (msgspec, array_like)
  vllm/v1/request.py:60-241                                 Request (engine core only)
  vllm/v1/core/sched/output.py:41-56, 138-152, 232-313      NewRequestData, CachedRequestData, SchedulerOutput
  vllm/v1/outputs.py:264-311                                ModelRunnerOutput
  vllm/v1/engine/__init__.py:203-297                        EngineCoreOutput, EngineCoreOutputs
  vllm/outputs.py:58-98, 137-209                            CompletionOutput, RequestOutput
  vllm/entrypoints/openai/chat_completion/protocol.py:162-193  stream choice and chunk
Example values: Tour 1 (tours/life-of-a-chat-request), first engine step.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The chain of types: one chat request, first engine step",
         "vllm serve on one GPU  ·  Tour 1 example: 61 prompt tokens, 48 cached, first token 785",
         params="pydantic → dict → msgspec ═ZMQ═ Python objects → msgspec ═ZMQ═ Python objects → pydantic")

BW = 290
X = (46, 405, 764)
BH = 124


def row(y, boxes, col, arrows=("", "")):
    for (title, lines), x in zip(boxes, X):
        d.box(x, y, BW, BH, title, lines, col=col, align="start", tsize=13)
    for i, lab in enumerate(arrows or ()):
        x1 = X[i] + BW
        x2 = X[i + 1]
        d.arrow(x1, y + 62, x2 - 2, y + 62, col)
        if lab:
            d.arrow_label((x1 + x2) / 2, y + 62, lab, col)


# ---------------------------------------------------------------- panel 1
y0 = 116
P_H = 212
d.panel(y0, P_H, 1, "Front end, on the way in",
        "HTTP JSON becomes a pydantic object, then a rendered dict, then one msgspec struct.")
row(y0 + 66, [
    ("ChatCompletionRequest", ["pydantic, from the JSON body", "messages: 4", "stream: True",
                               "max_tokens: None", "request_id: 3f9c...e1"]),
    ("EngineInput (dict)", ['type: "token"', "prompt_token_ids: 61 ids", 'prompt: "<|im_start|>..."',
                            "arrival_time: 1761140000.12"]),
    ("EngineCoreRequest", ["request_id: chatcmpl-3f9c...-a1b2c3d4", "external_req_id: chatcmpl-3f9c...e1",
                           "prompt_token_ids: 61 ids", "sampling_params: max_tokens 32707",
                           "client_index: 0"]),
], BLUE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P_H + 46
d.arrow(X[2] + BW / 2, y0 + 66 + BH, X[2] + BW / 2, y1 - 2, AMBER, sw=2)
d.text(X[2] + BW / 2 - 12, y0 + P_H + 28, "ADD frame, msgpack over ZMQ", 11.5, AMBER[0], "bold", "end")
d.panel(y1, P_H, 2, "Engine core process (and the worker)",
        "The struct becomes the engine's own Request; each step makes a plan and gets a result.")
row(y1 + 66, [
    ("ModelRunnerOutput", ["dataclass, from get_output()", "req_ids: [chatcmpl-...-a1b2c3d4]",
                           "sampled_token_ids: [[785]]", "logprobs: None"]),
    ("SchedulerOutput", ["dataclass, plain call (one GPU)", "scheduled_new_reqs: 1",
                         "  num_computed_tokens: 48", "  block_ids: ([7, 3, 12, 40],)",
                         "num_scheduled_tokens: {id: 13}"]),
    ("Request", ["Python object, never sent", "status: WAITING → RUNNING", "num_prompt_tokens: 61",
                 "num_computed_tokens: 0 → 48 → 61", "block_hashes: 3 full blocks"]),
], VIOLET, arrows=None)
# panel 2 flows right to left
d.arrow(X[2], y1 + 66 + 62, X[1] + BW + 2, y1 + 66 + 62, EMERALD)
d.arrow_label((X[1] + BW + X[2]) / 2, y1 + 66 + 62, "schedule", EMERALD)
d.arrow(X[1], y1 + 66 + 62, X[0] + BW + 2, y1 + 66 + 62, EMERALD)
d.arrow_label((X[0] + BW + X[1]) / 2, y1 + 66 + 62, "run", EMERALD)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P_H + 46
d.arrow(X[0] + BW / 2, y1 + 66 + BH, X[0] + BW / 2, y2 - 2, AMBER, sw=2)
d.text(X[0] + BW / 2 + 12, y1 + P_H + 28, "update_from_output → EngineCoreOutputs, msgpack over ZMQ",
       11.5, AMBER[0], "bold")
d.panel(y2, P_H, 3, "Front end, on the way out",
        "Token ids come back as a msgspec struct; text appears only here, in the detokenizer.")
row(y2 + 66, [
    ("EngineCoreOutputs", ["engine_index: 0, timestamp: monotonic", "outputs: [EngineCoreOutput(",
                           "  new_token_ids: [785],", "  finish_reason: None,", "  events: [QUEUED, SCHEDULED])]"]),
    ("RequestOutput", ["request_id: chatcmpl-3f9c...e1", "outputs: [CompletionOutput(",
                       '  index 0, text "The",', "  token_ids [785])]", "finished: False"]),
    ("ChatCompletionStreamResponse", ['object: "chat.completion.chunk"', 'choices[0].delta.content: "The"',
                                      "finish_reason: None", "→ data: {...}\\n\\n (SSE)"]),
], BLUE)

y = y2 + P_H + 24
d.legend(30, y, [(BLUE, "API server process"), (VIOLET, "engine core process"), (EMERALD, "flow inside the engine core"),
                 (AMBER, "crosses a process boundary")])
y += 40
y = d.footnotes(30, y, [
    "Panel 2 reads right to left. With worker processes (TP, PP or DP > 1), both dataclasses travel pickled through shared memory.",
    "Values are the illustrative values of Tour 1 (ChatML template, block size 16, an earlier turn left 3 blocks cached).",
])
d.save(height=y + 14)
