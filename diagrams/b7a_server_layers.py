"""The layers of the API server that one HTTP request crosses, and the three answer types.

Used by: src/concepts/openai-server.html
Sources (at the pinned commit in explainer.json):
  vllm/entrypoints/launchers/launcher.py:177-238       serve_http: uvicorn.Config, NoSignalServer, server.serve(sockets)
  vllm/entrypoints/launchers/api_server/entry.py:141-161 timeout_keep_alive=VLLM_HTTP_TIMEOUT_KEEP_ALIVE (5 s)
  vllm/entrypoints/serve/middleware/register.py:19-75  CORS, API key, X-Request-Id, scaling, --middleware
  vllm/entrypoints/openai/chat_completion/api_router.py:43-85  route, @with_cancellation, @load_aware_call, 3 answer types
  vllm/entrypoints/openai/chat_completion/serving.py:264-438   _check_model, _preflight, render, sampling params, generate()
  vllm/renderers/online_renderer.py:225-318, 753-861  OnlineRenderer.render_chat / preprocess_chat
  vllm/renderers/base.py:1371-1411                    render_chat_async (arrival_time, template, tokenize)
  vllm/v1/engine/async_llm.py:726-802                 AsyncLLM.generate
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The layers of the API server for one request",
         "vllm serve  ·  one API server  ·  POST /v1/chat/completions  ·  default flags",
         params="uvicorn → middleware → router → serving class → renderer → engine client (AsyncLLM)")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 404
d.panel(y0, P1_H, 1, "Which layers does a request cross on its way in?",
        "Three HTTP layers from uvicorn and FastAPI, then three layers of vLLM code in the same process.")

BW, BH = 300, 120
X1, X2, X3 = 46, 400, 754
R1 = y0 + 72
d.box(X1, R1, BW, BH, "uvicorn (NoSignalServer)", [
    "socket bound before engine start",
    "h11: max 256 headers, 4 MiB event",
    "HTTP keep-alive: 5 s idle",
    "  VLLM_HTTP_TIMEOUT_KEEP_ALIVE",
    "launcher.py:177-238",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "Middleware stack", [
    "CORSMiddleware (always)",
    "AuthenticationMiddleware (--api-key)",
    "XRequestIdMiddleware (flag)",
    "ScalingMiddleware (generate task)",
    "--middleware (user classes)",
], col=BLUE, align="start")
d.box(X3, R1, BW, BH, "FastAPI router: route function", [
    "pydantic: JSON → ChatCompletionRequest",
    "@with_cancellation: disconnect watch",
    "@load_aware_call: /load counter",
    "handler: app.state.openai_serving_chat",
    "api_router.py:43-85",
], col=BLUE, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, BLUE)

R2 = R1 + BH + 56
d.box(X1, R2, BW, BH, "Serving class: OpenAIServingChat", [
    "_check_model: name or LoRA, else 404",
    "_preflight(n): dead engine? admission",
    "max_tokens, to_sampling_params()",
    "request id: chatcmpl-<X-Request-Id>",
    "serving.py:302-438",
], col=VIOLET, align="start")
d.box(X2, R2, BW, BH, "Renderer: OnlineRenderer → HfRenderer", [
    "tool_choice check, template trust",
    "arrival_time = time.time()",
    "chat template → prompt string",
    "tokenize (thread pool), length check",
    "→ EngineInput (token ids)",
], col=VIOLET, align="start")
d.box(X3, R2, BW, BH, "Engine client: AsyncLLM.generate()", [
    "an async generator: no work yet",
    "first pull: add_request()",
    "InputProcessor → EngineCoreRequest",
    "ZMQ → engine core process",
    "async_llm.py:726-802",
], col=AMBER, align="start")
# route (box 3) calls the serving class (box 4): down, left, down
mid_y = R1 + BH + 28
d.path_arrow(f"M {X3 + BW / 2},{R1 + BH} L {X3 + BW / 2},{mid_y} L {X1 + BW / 2},{mid_y} "
             f"L {X1 + BW / 2},{R2 - 2}", VIOLET)
d.arrow_label((X1 + X3 + BW) / 2, mid_y, "create_chat_completion(request, raw_request)", VIOLET)
d.arrow(X1 + BW, R2 + BH / 2, X2 - 2, R2 + BH / 2, VIOLET)
d.arrow(X2 + BW, R2 + BH / 2, X3 - 2, R2 + BH / 2, AMBER)
d.text(X1, R2 + BH + 26, "The renderer runs before the sampling parameters are made; generate() only builds a generator object.",
       11.5, INK, italic=True)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 196
d.panel(y1, P2_H, 2, "What can the serving class give back to the route?",
        "Three types; the route turns each one into a different HTTP response.")
R3 = y1 + 72
BH2 = 100
d.box(X1, R3, BW, BH2, "ErrorResponse", [
    "JSONResponse(status = error.code)",
    "for example 404 for a wrong model",
    "api_router.py:69-72",
], col=EMERALD, align="start")
d.box(X2, R3, BW, BH2, "ChatCompletionResponse", [
    "stream = false (FINAL_ONLY)",
    "JSONResponse, HTTP 200",
    "api_router.py:74-78",
], col=EMERALD, align="start")
d.box(X3, R3, BW, BH2, "AsyncGenerator[str]", [
    "stream = true (DELTA)",
    "StreamingResponse text/event-stream",
    "+ with_sse_keep_alive (off at 0 s)",
], col=EMERALD, align="start")

y = y1 + P2_H + 24
d.legend(30, y, [(BLUE, "HTTP layers (uvicorn, Starlette, FastAPI)"), (VIOLET, "vLLM serving code"),
                 (AMBER, "engine boundary"), (EMERALD, "response type")])
y += 40
y = d.footnotes(30, y, [
    "All six layers run in one process and on one asyncio event loop; only template and tokenizer work moves to a thread pool.",
    "Exceptions that escape the route go to the exception handlers of build_app (400, 404, 422, 500, 501, 503).",
])
d.save(height=y + 14)
