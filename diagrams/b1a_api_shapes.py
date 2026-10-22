"""The OpenAI-compatible API: request and response shapes of /v1/chat/completions.

Used by: src/learn/openai-api.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/openai/chat_completion/protocol.py:225-249   OpenAI request fields
  vllm/entrypoints/openai/chat_completion/protocol.py:278-288   vLLM extra sampling fields (top_k, min_p, ...)
  vllm/entrypoints/openai/chat_completion/protocol.py:692-700   OpenAI defaults (temperature 1.0, top_p 1.0, top_k 0)
  vllm/entrypoints/openai/chat_completion/protocol.py:735-747   priority: request -> server default -> OpenAI default
  vllm/entrypoints/serve/utils/api_utils.py:234-270             get_max_tokens
  vllm/entrypoints/openai/chat_completion/protocol.py:797-799   stream -> RequestOutputKind.DELTA
  vllm/entrypoints/openai/chat_completion/serving.py:485-508    "data: ...\n\n" and "data: [DONE]"
  vllm/entrypoints/openai/chat_completion/serving.py:590-598    first chunk: role + empty content
  vllm/entrypoints/openai/chat_completion/serving.py:854-903    usage chunk with choices = []
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One chat request: what goes in, what comes out",
         "POST /v1/chat/completions  ·  stream = true  ·  model Qwen/Qwen3-0.6B  ·  illustrative values",
         params="JSON body -> ChatCompletionRequest -> SamplingParams -> engine -> SSE chunks")

# ---------------------------------------------------------------- panel 1
y0 = 112
P1_H = 290
d.panel(y0, P1_H, 1, "What does the server make from the request body?",
        "Pydantic checks the JSON (HTTP 400 on error). Missing sampling fields: model generation config, then OpenAI defaults.")
by = y0 + 72
BH = 200
d.box(46, by, 400, BH, "Request body (JSON)", [
    '{"model": "Qwen/Qwen3-0.6B",',
    ' "messages": [',
    '   {"role": "user", "content": "Hi"}],',
    ' "max_completion_tokens": 32,',
    ' "temperature": 0.7,',
    ' "stream": true,',
    ' "stream_options": {"include_usage": true},',
    ' "top_k": 20,                 <- vLLM extra',
    ' "chat_template_kwargs":      <- vLLM extra',
    '    {"enable_thinking": false}}',
], col=BLUE, align="start")
d.box(476, by, 270, BH, "OpenAIServingChat", [
    "1 check the model name",
    "2 chat template + tokenize",
    "3 max_tokens = min(32,",
    "    max_model_len - prompt len)",
    "4 to_sampling_params():",
    "    request -> server default",
    "    -> OpenAI default",
], col=INK, align="start")
d.box(776, by, 278, BH, "SamplingParams", [
    "temperature   0.7",
    "top_k         20",
    "top_p         1.0 or model value",
    "max_tokens    32",
    "output_kind   DELTA",
    "               (stream = true)",
], col=VIOLET, align="start")
d.arrow(446, by + 60, 474, by + 60, BLUE)
d.arrow(746, by + 60, 774, by + 60, VIOLET)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 250
d.panel(y1, P2_H, 2, "What does the client get back?",
        "Without stream: one JSON object. With stream: one SSE event per output, a usage chunk, then data: [DONE].")
by = y1 + 72
d.box(46, by, 330, 160, "stream = false (one JSON body)", [
    '{"id": "chatcmpl-...",',
    ' "object": "chat.completion",',
    ' "choices": [{"index": 0,',
    '   "message": {"role": "assistant",',
    '     "content": "Hello! How can ..."},',
    '   "finish_reason": "stop"}],',
    ' "usage": {"prompt_tokens": 13,',
    '   "total_tokens": 23, ...}}',
], col=EMERALD, align="start")
d.box(408, by, 646, 160, "stream = true (text/event-stream)", [
    'data: {..."choices":[{"index":0,"delta":{"role":"assistant","content":""},...}]}',
    'data: {..."choices":[{"index":0,"delta":{"content":"Hello"},...}]}',
    'data: {..."choices":[{"index":0,"delta":{"content":"!"},...}]}',
    '...',
    'data: {..."choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop",...}]}',
    'data: {..."choices":[],"usage":{"prompt_tokens":13,...,"completion_tokens":10},...}',
    'data: [DONE]',
], col=EMERALD, align="start", mono_size=11)

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "client JSON"), (INK, "API server code"), (VIOLET, "engine input"),
                 (EMERALD, "HTTP response")])
y += 40
y = d.footnotes(30, y, [
    "Each SSE event ends with a blank line (not shown). The id, object, created and model fields are not shown.",
    "The usage chunk comes only with stream_options.include_usage = true or --enable-force-include-usage.",
    "Illustrative numbers: 13 prompt tokens (Qwen3 chat template, thinking off), 10 output tokens including the end token.",
])
d.save(height=y + 14)
