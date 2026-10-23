"""From chat messages to token ids: what the renderer does for one chat request.

Used by: src/concepts/renderers-and-chat-templates.html
Sources (at the pinned commit in explainer.json):
  vllm/renderers/online_renderer.py:774-815   default kwargs (tools, tokenize), chat params, render_chat_async
  vllm/entrypoints/openai/chat_completion/protocol.py:635-689  build_chat_params, build_tok_params
  vllm/renderers/base.py:1371-1411            render_chat_async: arrival_time, render, tokenize, process_for_engine
  vllm/renderers/hf.py:264-319                resolve_chat_template: given → processor (no tools) → tokenizer → fallback
  vllm/renderers/hf.py:761-789                resolve_chat_template_kwargs: keep only kwargs the template uses
  vllm/renderers/hf.py:1169-1220              render_messages_async, timeout wrapper, thread pool
  vllm/renderers/base.py:528-540              chat tokenize params: add_special_tokens=False, max_total_tokens=max_model_len
  vllm/renderers/params.py:518-538            length check → VLLMValidationError (HTTP 400)
  examples/template_chatml.jinja:1-2          the example template
Token ids: Qwen2.5 tokenizer (tokenizer.json of Qwen/Qwen2.5-0.5B), measured outside the repo.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("From chat messages to token ids",
         "HfRenderer  ·  ChatML template (examples/template_chatml.jinja)  ·  Qwen2.5 tokenizer  ·  3 messages",
         params="messages → chat template → prompt string → tokenizer → 31 token ids + arrival_time")

BW = 300
X1, X2, X3 = 46, 400, 754

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 300
d.panel(y0, P1_H, 1, "How do the messages become one prompt string?",
        "The renderer picks a template, merges the template kwargs, and renders in a thread pool.")
R1 = y0 + 70
BH = 196
d.box(X1, R1, BW, BH, "Request: 3 messages", [
    "system: You are a helpful",
    "        assistant.",
    "user:   What is the capital",
    "        of France?",
    "assistant: The capital of",
    "        France is",
    "continue_final_message: true",
    "add_generation_prompt: false",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "Chat template (Jinja)", [
    "1 request / --chat-template",
    "2 processor template (no tools)",
    "3 tokenizer template",
    "4 built-in fallback by model type",
    "kwargs: request > server default",
    "  unknown names dropped",
    "timeout 30 s (HF renderer)",
    "hf.py:264-319, 761-789",
], col=VIOLET, align="start")
d.box(X3, R1, BW, BH, "Prompt string", [
    "<|im_start|>system\\n",
    "You are a helpful assistant.",
    "<|im_end|>\\n",
    "<|im_start|>user\\n",
    "What is the capital of France?",
    "<|im_end|>\\n",
    "<|im_start|>assistant\\n",
    "The capital of France is",
], col=VIOLET, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, VIOLET)
d.text(X3, R1 + BH + 22, "No <|im_end|> after the last message: the turn stays open.", 11.5, MUTE, italic=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 260
d.panel(y1, P2_H, 2, "How does the string become an engine input?",
        "The tokenizer runs in the same thread pool; then a length check and the arrival time.")
R2 = y1 + 70
BH2 = 156
d.box(X1, R2, BW, BH2, "Tokenizer (thread pool)", [
    "tokenizer(prompt, **kwargs)",
    "add_special_tokens = False",
    "  (the template has them)",
    "→ 31 token ids:",
    "[151644, 8948, 198, 2610,",
    "  ..., 6722, 315, 9625, 374]",
    "base.py:663-678",
], col=VIOLET, align="start")
d.box(X2, R2, BW, BH2, "Length check", [
    "max_total_tokens = max_model_len",
    "31 ≤ max_model_len → pass",
    "longer → VLLMValidationError",
    "  → HTTP 400 (no engine call)",
    "params.py:518-538",
], col=AMBER, align="start")
d.box(X3, R2, BW, BH2, "EngineInput (dict)", [
    "type: 'token'",
    "prompt_token_ids: [31 ids]",
    "prompt: the string of panel 1",
    "arrival_time: time.time()",
    "  taken BEFORE the template",
    "base.py:1380, 1274",
], col=EMERALD, align="start")
d.arrow(X1 + BW, R2 + BH2 / 2, X2 - 2, R2 + BH2 / 2, VIOLET)
d.arrow(X2 + BW, R2 + BH2 / 2, X3 - 2, R2 + BH2 / 2, AMBER)

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "request"), (VIOLET, "renderer work (thread pool)"), (AMBER, "check"),
                 (EMERALD, "result for AsyncLLM")])
y += 40
y = d.footnotes(30, y, [
    "Token ids measured with the Qwen2.5 tokenizer outside the repo; another model gives other ids and counts.",
    "Not shown: multimodal content parts (placeholders, MM processor) and the other renderers (Mistral, DeepSeek, Kimi K3, ...).",
])
d.save(height=y + 14)
