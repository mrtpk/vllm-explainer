/* request_stepper.js - step through the 16 stops of Tour 1 "Life of a chat request" (vLLM Explained).
 *
 * Mounts into <x-viz id="request-stepper">. Vanilla JS, no network, no randomness (nothing to seed).
 * A small process map (API server process / engine core process / GPU) shows where the code of each
 * stop runs. The values are the illustrative values of src/tours/life-of-a-chat-request.html
 * (61 prompt tokens, 48 of them in the prefix cache, block size 16).
 * "Read stop N" links to #stop-N, the ids that build.py makes for <x-stop>.
 *
 * Rules copied from the code (pinned commit, see the page for the citations):
 *   stops 1-6 run in the API server process (api_router.py, serving.py, renderers/, async_llm.py)
 *   stop 6 sends ADD + msgpack over ZMQ (core_client.py:1275-1294, 1333-1336)
 *   stops 7-14 run in the engine core process; with TP = 1 the worker is in the same process
 *     (config/parallel.py:1099-1100, uniproc_executor.py:51-69)
 *   stops 11-13 start GPU work (gpu/model_runner.py)
 *   stops 15-16 run in the API server process again (async_llm.py:876-946, serving.py:473-508)
 * Left out: other requests in the batch, spec decoding, structured output, errors.
 */
(function () {
  "use strict";

  var VIZ_ID = "request-stepper";
  var PARTS = [
    { key: "api", name: "API server process", sub: "FastAPI + AsyncLLM" },
    { key: "core", name: "Engine core process", sub: "scheduler + worker" },
    { key: "gpu", name: "GPU", sub: "forward + sampler" }
  ];

  var STOPS = [
    { title: "The route receives the request", on: ["api"],
      where: "entrypoints/openai/chat_completion/api_router.py · create_chat_completion",
      says: "FastAPI parses the JSON body into a ChatCompletionRequest and calls the serving class.",
      data: "POST /v1/chat/completions\n{\"model\": \"...\", \"stream\": true,\n \"messages\": [4 messages]}" },
    { title: "OpenAIServingChat prepares the request", on: ["api"],
      where: "chat_completion/serving.py · _create_chat_completion",
      says: "It renders the prompt (stop 3), makes the request id and the SamplingParams, and makes a generator.",
      data: "request_id = \"chatcmpl-<uuid>\"\nsampling_params.output_kind = DELTA   # stream: true\nmax_tokens = max_model_len - 61" },
    { title: "Chat template and tokenizer", on: ["api"],
      where: "renderers/base.py · render_chat_async, renderers/hf.py",
      says: "The chat template turns the messages into one text. The tokenizer turns the text into ids, in a thread pool.",
      data: "\"<|im_start|>system\\nYou are a helpful ...\"\nprompt_token_ids = [...]   # 61 ids\narrival_time = time.time()" },
    { title: "The stream starts: AsyncLLM.generate()", on: ["api"],
      where: "v1/engine/async_llm.py · generate",
      says: "The route returned a StreamingResponse. Its first pull runs generate(), which calls add_request().",
      data: "HTTP 200, text/event-stream   # already sent\nq = await self.add_request(...)" },
    { title: "add_request: the EngineCoreRequest", on: ["api"],
      where: "v1/engine/input_processor.py · process_inputs, assign_request_id",
      says: "The InputProcessor checks the inputs and makes a msgspec struct. The request id gets 8 random characters.",
      data: "EngineCoreRequest(\n  request_id=\"chatcmpl-<uuid>-a1b2c3d4\",\n  prompt_token_ids=[61 ids], sampling_params=...)" },
    { title: "Register, then send over ZMQ", on: ["api", "core"], msg: "ADD + msgpack ->",
      where: "async_llm.py · _add_request, core_client.py · add_request_async",
      says: "The OutputProcessor registers the request and its collector. Then the client sends ADD on the ROUTER socket.",
      data: "frames = [identity, b\"\\x00\" (ADD), msgpack(EngineCoreRequest)]" },
    { title: "The engine core input thread", on: ["core"],
      where: "v1/engine/core.py · process_input_sockets, preprocess_add_request",
      says: "A thread decodes the message and makes a Request. The Request computes its block hashes.",
      data: "Request: status=WAITING, num_computed_tokens=0\nblock_hashes: 3 (61 // 16 full blocks)" },
    { title: "The busy loop adds the request", on: ["core"],
      where: "v1/engine/core.py · run_busy_loop, Scheduler.add_request",
      says: "The main thread takes the request from input_queue and puts it in the waiting queue of the scheduler.",
      data: "scheduler.waiting = [chatcmpl-...-a1b2c3d4]" },
    { title: "One engine step (batch queue)", on: ["core"],
      where: "v1/engine/core.py · step_with_batch_queue",
      says: "The step schedules, starts the model and the sampler, and waits only for the oldest step in the queue.",
      data: "batch_queue size = 2   # async scheduling + MRV2" },
    { title: "Scheduler.schedule()", on: ["core"],
      where: "v1/core/sched/scheduler.py · schedule",
      says: "The scheduler finds 48 cached tokens, allocates blocks, and gives 13 tokens of the token budget.",
      data: "num_scheduled_tokens = {req: 13}\nnum_computed_tokens: 48 -> 61\nstatus: WAITING -> RUNNING" },
    { title: "Model Runner V2 prepares the inputs", on: ["core", "gpu"],
      where: "v1/worker/gpu/model_runner.py · execute_model, prepare_inputs, prepare_attn",
      says: "The runner updates its request states and fills the input buffers, block tables, slot mappings and attention metadata.",
      data: "input_ids [13], positions 48..60\nquery_start_loc, seq_lens, slot_mappings" },
    { title: "The forward pass", on: ["gpu"],
      where: "v1/worker/gpu/model_runner.py · execute_model",
      says: "The runner replays a CUDA graph or calls the model eagerly. Default: FULL graph for a decode-only batch, PIECEWISE for our prefill.",
      data: "step 1 (13-token prefill): PIECEWISE\nlater steps (1-token decode): FULL\nhidden_states [13 + padding, hidden_size]" },
    { title: "sample_tokens: logits and sampler", on: ["gpu", "core"],
      where: "v1/worker/gpu/model_runner.py · sample_tokens",
      says: "The runner computes logits for the last position, samples one token (FlashInfer or Gumbel sampler) and starts a copy to the CPU.",
      data: "logits [1, vocab_size] -> sampled_token_ids [1, 1]\nAsyncOutput (copy stream + event)" },
    { title: "update_from_output, then send back", on: ["core", "api"], msg: "<- msgpack",
      where: "sched/scheduler.py · update_from_output, core.py · process_output_sockets",
      says: "The scheduler appends the token and runs check_stop. The output thread sends EngineCoreOutputs on PUSH.",
      data: "EngineCoreOutput(request_id=..., new_token_ids=[token 1],\n                 finish_reason=None)" },
    { title: "output_handler and detokenizer", on: ["api"],
      where: "async_llm.py · output_handler, v1/engine/output_processor.py · process_outputs",
      says: "A background task decodes the message. The OutputProcessor turns the ids into text and puts a RequestOutput in the collector.",
      data: "RequestOutput(outputs=[CompletionOutput(\n  text=\"The\", token_ids=[token 1])], finished=False)" },
    { title: "SSE chunks and data: [DONE]", on: ["api"],
      where: "chat_completion/serving.py · chat_completion_stream_generator",
      says: "Each RequestOutput becomes one JSON chunk. After the last one the generator writes data: [DONE].",
      data: "data: {\"id\":\"chatcmpl-...\", ...,\n       \"choices\":[{\"index\":0,\"delta\":{\"content\":\"The\"}, ...}]}\n...\ndata: [DONE]" }
  ];

  var CSS = [
    ".rqs-root{font-family:var(--sans);color:var(--ink)}",
    ".rqs-controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}",
    ".rqs-count{font:600 14px var(--mono);color:var(--mute);margin-left:4px}",
    ".rqs-list{display:grid;grid-template-columns:repeat(16,minmax(0,1fr));gap:4px;margin:0 0 14px;padding:0;list-style:none}",
    ".rqs-num{width:100%;min-height:40px;border:1px solid var(--line);border-radius:6px;background:var(--surface);font:600 13px var(--mono);color:var(--ink);cursor:pointer;padding:0}",
    ".rqs-num[aria-current=step]{border-color:var(--blue);background:var(--blue-tint,var(--panel-bg));box-shadow:inset 0 -3px 0 var(--blue)}",
    ".rqs-map{display:flex;flex-wrap:wrap;gap:8px;align-items:stretch;margin:0 0 12px;padding:0;list-style:none}",
    ".rqs-part{flex:1 1 150px;border:1px dashed var(--line);border-radius:8px;padding:8px 10px;color:var(--mute);background:var(--surface)}",
    ".rqs-part b{display:block;font-size:14px}",
    ".rqs-part span{font-size:12.5px}",
    ".rqs-part.rqs-on{border-style:solid;border-width:2px;color:var(--ink)}",
    ".rqs-on.rqs-api{border-color:var(--viz-c1);background:var(--blue-tint,var(--panel-bg))}",
    ".rqs-on.rqs-core{border-color:var(--viz-c5);background:var(--violet-tint,var(--panel-bg))}",
    ".rqs-on.rqs-gpu{border-color:var(--viz-c3);background:var(--emerald-tint,var(--panel-bg))}",
    ".rqs-msg{margin:-4px 0 10px;font:600 13px var(--mono);color:var(--viz-c4)}",
    ".rqs-title{margin:0 0 6px;font-size:17px;font-weight:700}",
    ".rqs-title span{display:inline-block;min-width:1.8em;color:var(--blue);font-family:var(--mono)}",
    ".rqs-where{margin:0 0 6px;font-size:14px;color:var(--mute)}",
    ".rqs-where code{font:13px var(--mono);color:var(--ink);overflow-wrap:anywhere}",
    ".rqs-says{margin:0 0 10px;font-size:15px}",
    ".rqs-pre{margin:0;padding:10px 12px;border-radius:6px;background:var(--code-bg,var(--panel-bg));border:1px solid var(--border);font:12.5px/1.55 var(--mono);color:var(--ink);overflow-x:auto;white-space:pre;max-width:100%}",
    ".rqs-foot{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;margin-top:10px;font-size:14px}",
    ".rqs-foot a{font-weight:600}",
    ".rqs-note{color:var(--mute);font-size:13px}",
    ".rqs-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}",
    "@media (max-width:560px){.rqs-list{grid-template-columns:repeat(8,minmax(0,1fr))}.rqs-part{flex-basis:100%;padding:6px 10px}.rqs-title{font-size:16px}}"
  ].join("\n");

  function injectStyle() {
    if (document.getElementById("rqs-style")) return;
    var st = document.createElement("style");
    st.id = "rqs-style";
    var P = ".viz-mount[data-viz='" + VIZ_ID + "'] ";
    st.textContent = CSS.split("\n").map(function (r) {
      if (r.charAt(0) === "@") {
        return r.replace(/\{\.rqs-/g, "{" + P + ".rqs-").replace(/\}\.rqs-/g, "}" + P + ".rqs-");
      }
      return r.split("{")[0].split(",").map(function (s) { return P + s.trim(); }).join(",") +
        "{" + r.split("{").slice(1).join("{");
    }).join("\n");
    document.head.appendChild(st);
  }

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function Stepper(mount) {
    this.mount = mount;
    this.cur = 0;
    this.build();
    this.show(0, false);
  }

  Stepper.prototype.build = function () {
    var self = this;
    this.mount.innerHTML = "";                       // removes the .viz-fallback text
    var root = el("div", "rqs-root");

    var ctl = el("div", "rqs-controls");
    this.prev = el("button", "viz-btn", "\u2190 Prev");
    this.next = el("button", "viz-btn viz-btn-primary", "Next \u2192");
    this.reset = el("button", "viz-btn", "Reset");
    [this.prev, this.next, this.reset].forEach(function (b) { b.type = "button"; ctl.appendChild(b); });
    this.count = el("span", "rqs-count");
    ctl.appendChild(this.count);
    root.appendChild(ctl);
    this.prev.addEventListener("click", function () { self.go(self.cur - 1); });
    this.next.addEventListener("click", function () { self.go(self.cur + 1); });
    this.reset.addEventListener("click", function () { self.go(0); });

    var list = el("ol", "rqs-list");
    list.setAttribute("aria-label", "Stops of the tour");
    this.nums = STOPS.map(function (s, i) {
      var li = el("li");
      var b = el("button", "rqs-num", String(i + 1));
      b.type = "button";
      b.setAttribute("aria-label", "Stop " + (i + 1) + ": " + s.title);
      b.title = s.title;
      b.addEventListener("click", function () { self.go(i); });
      li.appendChild(b);
      list.appendChild(li);
      return b;
    });
    root.appendChild(list);

    var map = el("ul", "rqs-map");
    map.setAttribute("aria-label", "Where the code of this stop runs");
    this.parts = PARTS.map(function (p) {
      var li = el("li", "rqs-part rqs-" + p.key);
      li.appendChild(el("b", null, p.name));
      li.appendChild(el("span", null, p.sub));
      map.appendChild(li);
      return li;
    });
    root.appendChild(map);
    this.msg = el("p", "rqs-msg");
    root.appendChild(this.msg);

    this.card = el("div", "rqs-card");
    root.appendChild(this.card);

    this.live = el("p", "rqs-live");
    this.live.setAttribute("aria-live", "polite");
    root.appendChild(this.live);

    root.addEventListener("keydown", function (ev) {
      var t = ev.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA")) return;
      if (ev.key === "ArrowRight") { self.go(self.cur + 1, true); ev.preventDefault(); }
      else if (ev.key === "ArrowLeft") { self.go(self.cur - 1, true); ev.preventDefault(); }
    });

    this.mount.appendChild(root);
  };

  Stepper.prototype.go = function (i, focusNum) {
    if (i < 0 || i >= STOPS.length) return;
    this.show(i, true);
    if (focusNum && document.activeElement && document.activeElement.classList.contains("rqs-num")) {
      this.nums[i].focus();
    }
  };

  Stepper.prototype.show = function (i, announce) {
    var s = STOPS[i];
    this.cur = i;
    this.prev.disabled = i === 0;
    this.next.disabled = i === STOPS.length - 1;
    this.count.textContent = "Stop " + (i + 1) + " of " + STOPS.length;
    this.nums.forEach(function (b, k) {
      if (k === i) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    });
    PARTS.forEach(function (p, k) {
      var on = s.on.indexOf(p.key) >= 0;
      this.parts[k].classList.toggle("rqs-on", on);
      this.parts[k].setAttribute("aria-label", p.name + (on ? ": active" : ": idle"));
    }, this);
    this.msg.textContent = s.msg ? "Message on ZMQ: " + s.msg : "";
    this.msg.style.display = s.msg ? "" : "none";

    var c = this.card;
    c.innerHTML = "";
    var h = el("h3", "rqs-title");
    h.appendChild(el("span", null, String(i + 1)));
    h.appendChild(document.createTextNode(s.title));
    c.appendChild(h);
    var w = el("p", "rqs-where");
    w.appendChild(document.createTextNode("Where: "));
    w.appendChild(el("code", null, s.where));
    c.appendChild(w);
    c.appendChild(el("p", "rqs-says", s.says));
    var pre = el("pre", "rqs-pre");
    pre.setAttribute("tabindex", "0");
    pre.setAttribute("aria-label", "Data at stop " + (i + 1) + " (illustrative)");
    pre.textContent = s.data;
    c.appendChild(pre);

    var foot = el("div", "rqs-foot");
    var a = el("a", null, "Read stop " + (i + 1) + " \u2193");
    a.href = "#stop-" + (i + 1);
    foot.appendChild(a);
    foot.appendChild(el("span", "rqs-note", "Simplified: one request; values are illustrative."));
    c.appendChild(foot);

    if (announce) this.live.textContent = "Stop " + (i + 1) + " of " + STOPS.length + ": " + s.title;
  };

  function init() {
    var mounts = document.querySelectorAll('[data-viz="' + VIZ_ID + '"]');
    if (!mounts.length) return;
    injectStyle();
    for (var i = 0; i < mounts.length; i++) new Stepper(mounts[i]);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
