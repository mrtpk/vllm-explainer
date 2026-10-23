"""Model tokens → reasoning parser → tool parser → API deltas, with one streamed example.

Used by: src/concepts/tool-and-reasoning-parsers.html
Sources (at the pinned commit in explainer.json):
  vllm/parser/parser_manager.py:23-54, 151-274     get_tool_parser (needs --enable-auto-tool-choice), _compose_parser → DelegatingParser
  vllm/entrypoints/openai/chat_completion/stream_processor.py:39-83  ChatStreamProcessor.step → parse_delta, finish_reason
  vllm/parser/abstract_parser.py:857-1007          DelegatingParser.parse_delta: prompt check, reasoning phase, tool phase
  vllm/reasoning/basic_parsers.py:82-151           is_reasoning_end_streaming, extract_reasoning_streaming (<think> ... </think>)
  vllm/reasoning/deepseek_r1_reasoning_parser.py:27-66
  vllm/tool_parsers/hermes_tool_parser.py:122-274  _extract_content, _extract_tool_call_jsons, extract_tool_calls_streaming
  vllm/entrypoints/serve/utils/tool_calls_utils.py:47-65  resolve_finish_reason → "tool_calls"
  vllm/entrypoints/openai/chat_completion/serving.py:731-741  delta None → no chunk
Example: --reasoning-parser deepseek_r1 --enable-auto-tool-choice --tool-call-parser hermes; one token per output; derived.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, INK, MUTE, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("From model tokens to API deltas",
         "streaming chat  ·  DelegatingParser  ·  deepseek_r1 reasoning parser + hermes tool parser  ·  tool_choice auto",
         params="token delta → reasoning phase → (</think>) → tool phase → DeltaMessage(reasoning | content | tool_calls)")

BW = 300
X1, X2, X3 = 46, 400, 754

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 244
d.panel(y0, P1_H, 1, "Which object looks at each new piece of output?",
        "One Parser per choice; it runs the reasoning parser first and the tool parser after the end marker.")
R1 = y0 + 70
BH = 150
d.box(X1, R1, BW, BH, "Output of one engine step", [
    "delta_text (detokenized)",
    "delta_token_ids",
    "finish_reason or None",
    "→ ChatStreamProcessor.step()",
    "  → parser.parse_delta()",
    "stream_processor.py:39-83",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "Reasoning phase", [
    "until </think> id in delta",
    "<think> or </think> alone → None",
    "text → DeltaMessage(reasoning)",
    "first check: prompt already",
    "  ended reasoning? skip phase",
    "basic_parsers.py:82-151",
], col=VIOLET, align="start")
d.box(X3, R1, BW, BH, "Tool phase", [
    "Hermes: re-parse full text",
    "text before <tool_call> → content",
    "name with closing quote → id, name",
    "new argument text → arguments",
    "nothing new → None (no chunk)",
    "hermes_tool_parser.py:204-274",
], col=TEAL, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, VIOLET)
d.arrow_label(X2 + BW + 27, R1 + BH / 2, "</think>", VIOLET)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
rows = [
    ("1", "<think>", "special", "None: a start marker alone", MUTE),
    ("2", "Weather", "text", '{"reasoning": "Weather"}', VIOLET),
    ("3", " is needed.", "text", '{"reasoning": " is needed."}', VIOLET),
    ("4", "</think>", "special", "None: phase switch, nothing to send", MUTE),
    ("5", "\\n\\n", "text", '{"content": "\\n\\n"}', BLUE),
    ("6", "<tool_call>", "special", "None: no complete name yet", MUTE),
    ("7", '{"name": "get_weather",', "text", '{"tool_calls":[{"index":0,"id":"chatcmpl-tool-..",', TEAL),
    ("", "", "", '  "type":"function","function":{"name":"get_weather"}}]}', TEAL),
    ("8", ' "arguments": {"city": "Paris"}}', "text", '{"tool_calls":[{"index":0,"function":', TEAL),
    ("", "", "", '  {"arguments":"{\\"city\\": \\"Paris\\"}"}}]}', TEAL),
    ("9", "</tool_call>", "special", "None: the arguments did not change", MUTE),
    ("10", "(EOS, finished)", "special", 'delta {} + "finish_reason": "tool_calls"', AMBER),
]
P2_H = 96 + len(rows) * 20 + 14
d.panel(y1, P2_H, 2, "What does the client get for each output?",
        "Ten rows; rows 1, 4, 6 and 9 send no chunk. The role chunk comes before row 1.")
ty = y1 + 74
d.text(46, ty, "#", 11, MUTE, "bold", mono=True)
d.text(76, ty, "delta_text", 11, MUTE, "bold", mono=True)
d.text(400, ty, "kind", 11, MUTE, "bold", mono=True)
d.text(470, ty, "choices[0].delta in the SSE chunk", 11, MUTE, "bold", mono=True)
d.line(46, ty + 7, 1054, ty + 7, "#e5e7eb")
for i, (n, txt, tid, out, col) in enumerate(rows):
    ry = ty + 24 + i * 20
    d.text(46, ry, n, 12, INK, mono=True)
    d.text(76, ry, txt, 12, INK, mono=True)
    d.text(400, ry, tid, 12, MUTE, mono=True)
    d.text(470, ry, out, 12, col, "bold" if col != MUTE else "normal", mono=True)

y = y1 + P2_H + 24
d.legend(30, y, [(VIOLET, "reasoning delta"), (BLUE, "content delta"), (TEAL, "tool-call delta"),
                 (AMBER, "final chunk")])
y += 40
y = d.footnotes(30, y, [
    "Derived from the code. Rows 2, 3, 7 and 8 each stand for several one-token outputs; the row shows their sum.",
    "Row 7: only the output with the closing quote of the name sends the chunk (the name regex needs it, hermes_tool_parser.py:169-172).",
    "Markers (special) arrive as single tokens. Rows 1, 4, 6 and 9 send no chunk because the delta is None (serving.py:731-740).",
    "Hermes sets skip_special_tokens = False when tools are used, so the marker text reaches the parsers (hermes_tool_parser.py:59-68).",
])
d.save(height=y + 14)
