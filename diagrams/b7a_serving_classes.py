"""The class hierarchy of the serving classes, and what each layer adds.

Used by: src/concepts/openai-server.html, src/code/entrypoints-openai-chat-completion-serving.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/serve/engine/serving.py:31-213               BaseServing: _check_model, create_error_response, _base_request_id
  vllm/entrypoints/generate/base/serving.py:129-265             GenerateBaseServing(BaseServing, BeamSearchOnlineMixin): _preflight, headers
  vllm/entrypoints/pooling/base/serving.py:32, 257              PoolingBaseServing, PoolingServing
  vllm/entrypoints/serve/tokenize/serving.py:32                 ServingTokenization(BaseServing)
  vllm/entrypoints/openai/chat_completion/serving.py:124        OpenAIServingChat(GenerateBaseServing)
  vllm/entrypoints/openai/completion/serving.py:57              OpenAIServingCompletion(GenerateBaseServing)
  vllm/entrypoints/openai/responses/serving.py:104              OpenAIServingResponses(GenerateBaseServing)
  vllm/entrypoints/anthropic/serving.py:208                     AnthropicServingMessages(OpenAIServingChat)
  vllm/entrypoints/cohere/serving.py:221                        CohereServingChatV2(OpenAIServingChat)
  vllm/entrypoints/openai/chat_completion/batch_serving.py:34   OpenAIServingChatBatch(OpenAIServingChat)
  vllm/entrypoints/openai/models/serving.py:83                  OpenAIServingModels (no base class)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import BLUE, INK, MUTE, NEUTRAL, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The serving classes: one base, many routes",
         "vllm/entrypoints/  ·  classes that app.state holds  ·  arrows point from parent to child",
         params="BaseServing → GenerateBaseServing → OpenAIServingChat → Anthropic / Cohere / Batch")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 484
d.panel(y0, P1_H, 1, "Which class serves which route?",
        "Every serving class derives from BaseServing; the generate routes share GenerateBaseServing.")

BW = 300
X1, X2, X3 = 46, 400, 754
R0 = y0 + 70
d.box(X2, R0, BW, 64, "BaseServing", ["serve/engine/serving.py:31"], col=INK)

R1 = R0 + 100
BH1 = 104
d.box(X1, R1, BW, BH1, "GenerateBaseServing", [
    "generate/base/serving.py:129",
    "+ BeamSearchOnlineMixin",
    "routes that call generate()",
], col=VIOLET, align="start")
d.box(X2, R1, BW, BH1, "PoolingBaseServing", [
    "pooling/base/serving.py:32",
    "embeddings, pooling, classify,",
    "score, rerank (encode())",
], col=NEUTRAL, align="start")
d.box(X3, R1, BW, BH1, "Other direct subclasses", [
    "ServingTokenization (/tokenize)",
    "ServingRender, ServingDerender",
    "decisions, generative scoring",
], col=NEUTRAL, align="start")
# parent → children: down from BaseServing, then out
bus_y = R0 + 64 + 18
d.line(X2 + BW / 2, R0 + 64, X2 + BW / 2, bus_y, INK, 1.6)
d.line(X1 + BW / 2, bus_y, X3 + BW / 2, bus_y, INK, 1.6)
for x in (X1, X2, X3):
    d.arrow(x + BW / 2, bus_y, x + BW / 2, R1 - 2, INK)

R2 = R1 + BH1 + 48
BH2 = 120
d.box(X1, R2, BW, BH2, "OpenAIServingChat", [
    "POST /v1/chat/completions",
    "chat_completion/serving.py:124",
    "children: Anthropic /v1/messages,",
    "  Cohere v2 chat, chat batch",
], col=TEAL, align="start")
d.box(X2, R2, BW, BH2, "OpenAIServingCompletion", [
    "POST /v1/completions",
    "completion/serving.py:57",
    "raw prompt or token ids,",
    "  no chat template",
], col=VIOLET, align="start")
d.box(X3, R2, BW, BH2, "More GenerateBaseServing children", [
    "OpenAIServingResponses",
    "  (/v1/responses)",
    "ServingTokens (/inference/v1/generate)",
    "speech-to-text, realtime",
], col=VIOLET, align="start")
bus2 = R1 + BH1 + 22
d.line(X1 + BW / 2, R1 + BH1, X1 + BW / 2, bus2, VIOLET, 1.6)
d.line(X1 + BW / 2, bus2, X3 + BW / 2, bus2, VIOLET, 1.6)
for x in (X1, X2, X3):
    d.arrow(x + BW / 2, bus2, x + BW / 2, R2 - 2, VIOLET)
d.text(X1, R2 + BH2 + 26,
       "OpenAIServingModels (/v1/models, LoRA names) is a helper object, not a BaseServing subclass (models/serving.py:83).",
       11.5, MUTE, italic=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 228
d.panel(y1, P2_H, 2, "What does each layer add?",
        "Model checks at the bottom, engine checks in the middle, the API format at the top.")
R3 = y1 + 70
BH3 = 136
d.box(X1, R3, BW, BH3, "BaseServing", [
    "_check_model(): name, LoRA, resolver",
    "  wrong name → 404 NotFoundError",
    "create_error_response()",
    "_base_request_id(): X-Request-Id",
    "_log_inputs() (--enable-log-requests)",
    "_maybe_get_adapters(): LoRARequest",
], col=INK, align="start")
d.box(X2, R3, BW, BH3, "GenerateBaseServing", [
    "holds engine_client and renderer",
    "_preflight(n): errored? check_admission",
    "_get_priority(): X-Vllm-Priority",
    "_get_data_parallel_rank(): header",
    "_raise_if_error(): finish 'error'",
    "system_fingerprint",
], col=VIOLET, align="start")
d.box(X3, R3, BW, BH3, "OpenAIServingChat", [
    "parser_cls (tool + reasoning)",
    "render_chat_request()",
    "_create_chat_completion()",
    "chat_completion_stream_generator()",
    "chat_completion_full_generator()",
    "_create_chat_logprobs()",
], col=TEAL, align="start")

y = y1 + P2_H + 24
d.legend(30, y, [(INK, "base class"), (VIOLET, "generate family"), (TEAL, "chat (the main path)"),
                 (NEUTRAL, "other families")])
y += 40
y = d.footnotes(30, y, [
    "init_generate_state() makes one object per class at start-up and stores it in app.state (generate/api_router.py:57-228).",
    "Not shown: the pooling subclasses (embed, classify, score, token embed) and the speech-to-text subclasses.",
])
d.save(height=y + 14)
