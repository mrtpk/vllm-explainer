/* scheduler_sim.js - a small simulator of Scheduler.schedule() for concepts/scheduler (vLLM Explained).
 *
 * Mounts into <x-viz id="scheduler-sim">. Vanilla JS, no network, no randomness: the result depends only on
 * the inputs, so every reader sees the same run. "Back" and every settings change replay the run from step 0.
 *
 * Rules copied from the code (pinned commit; the page lists them with citations):
 *   - RUNNING pass first, front to back, while budget > 0 (sched/scheduler.py:713-715)
 *   - tokens = num_tokens - num_computed_tokens, capped by the budget (scheduler.py:759-768)
 *   - 0 tokens -> continue, not break (scheduler.py:811-829)
 *   - blocks short -> preempt the LAST running request (FCFS), free its blocks, num_computed_tokens = 0,
 *     put it at the front of the waiting queue; stop the pass if it preempted itself (scheduler.py:831-904, 1658-1701)
 *   - WAITING pass only if nothing was preempted (scheduler.py:957); len(running) < max_num_seqs (979-981)
 *   - chunked prefill off and the prompt does not fit the rest of the budget -> stop the pass (1220-1229)
 *   - a new or preempted request starts only if all its tokens fit in free blocks (kv_cache_manager.py:522-539,
 *     scheduler_reserve_full_isl default True); no fit -> stop the pass (scheduler.py:1339-1346)
 *   - blocks = ceil((computed + new) / block_size) (single_type_kv_cache_manager.py:214, 401-402)
 *   - freed blocks go to the front of the free queue in reverse order (block_pool.py:776-807, no prefix cache)
 *   - a request that reaches the end of its known tokens samples 1 token; it stops at max_tokens (sched/utils.py:114-119)
 * Left out: async placeholders (each Step = schedule + update), prefix caching, spec decode, EOS, max_model_len,
 * long_prefill_token_threshold, LoRA, structured output, priority policy, watermark.
 */
(function () {
  "use strict";

  var VIZ_ID = "scheduler-sim";
  var LETTERS = "ABCDEFGHIJKL";
  var MAX_REQS = 12;
  var DEFAULT_CFG = { budget: 16, seqs: 4, blocks: 8, bs: 4, chunked: true };
  var DEFAULT_SCRIPT = [
    { prompt: 10, maxOut: 4, arrive: 1 },
    { prompt: 12, maxOut: 4, arrive: 1 },
    { prompt: 6, maxOut: 4, arrive: 1 }
  ];
  var LIMITS = {
    budget: [1, 256], seqs: [1, 16], blocks: [1, 64], bs: [1, 32], prompt: [1, 200], maxOut: [1, 50]
  };

  function cdiv(a, b) { return Math.floor((a + b - 1) / b); }
  function clamp(v, k) {
    var n = parseInt(v, 10);
    if (isNaN(n)) n = LIMITS[k][0];
    return Math.max(LIMITS[k][0], Math.min(LIMITS[k][1], n));
  }

  /* ---------------------------------------------------------------- the model */

  function newState(cfg) {
    var free = [];
    for (var b = 1; b <= cfg.blocks; b++) free.push(b);   // block 0 is the null block: never handed out
    return { reqs: [], running: [], waiting: [], free: free, step: 0, log: [], used: 0, sched: {} };
  }

  function freeBlocks(st, r) {
    // free_blocks(reversed(blocks)); without a hash every block goes to the front (prepend_n keeps the order)
    st.free = r.blockIds.slice().reverse().concat(st.free);
    r.blockIds = [];
  }

  function allocate(st, r, n, cfg) {
    var need = cdiv(r.computed + n, cfg.bs) - r.blockIds.length;
    if (need <= 0) return 0;
    if (need > st.free.length) return -1;
    r.blockIds = r.blockIds.concat(st.free.splice(0, need));   // popleft_n
    return need;
  }

  function stepOnce(st, cfg, script) {
    st.step += 1;
    st.log = [];
    st.sched = {};
    var L = st.log;
    script.forEach(function (s, i) {
      if (s.arrive === st.step) {
        var r = { id: LETTERS[i], idx: i, prompt: s.prompt, maxOut: s.maxOut, numTokens: s.prompt, out: 0,
                  computed: 0, status: "WAITING", blockIds: [], preemptions: 0 };
        st.reqs.push(r);
        st.waiting.push(r);
        L.push(r.id + " arrives and joins the waiting queue.");
      }
    });

    var budget = cfg.budget;
    var preempted = [];

    // ---- RUNNING pass
    var i = 0;
    while (i < st.running.length && budget > 0) {
      var r = st.running[i];
      var n = Math.min(r.numTokens - r.computed, budget);
      if (n === 0) { i++; continue; }
      var got;
      for (;;) {
        got = allocate(st, r, n, cfg);
        if (got >= 0) break;
        var needMsg = cdiv(r.computed + n, cfg.bs) - r.blockIds.length;
        var victim = st.running.pop();
        L.push(r.id + " needs " + needMsg + " new block(s), only " + st.free.length + " free: preempt " +
               victim.id + " (last in running). " + victim.blockIds.length + " block(s) freed, " +
               "num_computed_tokens = 0, front of the waiting queue.");
        freeBlocksPreempt(st, victim);
        preempted.push(victim);
        if (victim === r) break;
      }
      if (got < 0) { L.push("The running pass stops: " + r.id + " preempted itself."); break; }
      st.sched[r.id] = n;
      budget -= n;
      L.push("Running: " + r.id + " gets " + n + " token" + (n > 1 ? "s" : "") +
             (got ? " and " + got + " new block" + (got > 1 ? "s" : "") : "") + ".");
      i++;
    }

    // ---- WAITING pass
    if (preempted.length) {
      L.push("No waiting pass: a request was preempted in this step.");
    } else {
      while (budget > 0 && st.waiting.length) {
        if (st.running.length >= cfg.seqs) {
          L.push("Waiting pass stops: " + st.running.length + " running = max_num_seqs.");
          break;
        }
        var w = st.waiting[0];
        var nw = w.numTokens - w.computed;
        if (!cfg.chunked && nw > budget) {
          L.push("Waiting pass stops: " + w.id + " needs " + nw + " tokens, only " + budget +
                 " left, and chunked prefill is off.");
          if (nw > cfg.budget) L.push("Note: " + w.id + " can never fit. vLLM refuses this setup at start-up: " +
                                       "without chunked prefill, max_num_batched_tokens must be >= max_model_len.");
          break;
        }
        nw = Math.min(nw, budget);
        var full = cdiv(w.numTokens, cfg.bs);
        if (full > st.free.length) {
          L.push("Waiting pass stops: " + w.id + " needs " + full + " blocks for all its tokens, " +
                 st.free.length + " free.");
          if (full > cfg.blocks) L.push("Note: " + w.id + " can never fit in " + cfg.blocks +
                                         " blocks. vLLM checks at start-up that max_model_len fits the KV cache.");
          break;
        }
        var gw = allocate(st, w, nw, cfg);
        st.waiting.shift();
        st.running.push(w);
        var was = w.status;
        w.status = "RUNNING";
        st.sched[w.id] = nw;
        budget -= nw;
        L.push((was === "PREEMPTED" ? "Waiting (resumed): " : "Waiting: ") + w.id + " starts with " + nw +
               " token" + (nw > 1 ? "s" : "") + (nw < w.numTokens ? " (a chunk)" : "") + " and " + gw +
               " block" + (gw === 1 ? "" : "s") + ".");
      }
    }
    st.used = cfg.budget - budget;
    if (!Object.keys(st.sched).length) L.push("Nothing is scheduled in this step.");

    // ---- update_from_output (synchronous: the token of this step is known now)
    var stopped = [];
    st.running.forEach(function (r) {
      var n = st.sched[r.id];
      if (!n) return;
      r.computed += n;
      if (r.computed >= r.numTokens) {          // not a prefill chunk -> one sampled token
        r.numTokens += 1;
        r.out += 1;
        if (r.out >= r.maxOut) stopped.push(r);
      }
    });
    stopped.forEach(function (r) {
      r.status = "FINISHED";
      freeBlocks(st, r);
      st.running.splice(st.running.indexOf(r), 1);
      L.push(r.id + " reaches max_tokens (" + r.maxOut + "): finished, blocks freed.");
    });
  }

  function freeBlocksPreempt(st, r) {
    freeBlocks(st, r);
    r.status = "PREEMPTED";
    r.computed = 0;
    r.preemptions += 1;
    st.waiting.unshift(r);                       // prepend_request: front of the FCFS queue
  }

  function run(cfg, script, steps) {
    var st = newState(cfg);
    for (var k = 0; k < steps; k++) stepOnce(st, cfg, script);
    return st;
  }

  /* ---------------------------------------------------------------- the view */

  var CSS = [
    ".scs-root{font-family:var(--sans);color:var(--ink)}",
    ".scs-form{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px 12px;margin:0 0 10px;align-items:end}",
    ".scs-form label{display:flex;flex-direction:column;gap:2px;font-size:13.5px;color:var(--mute)}",
    ".scs-form input[type=number]{min-height:36px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);font:14px var(--mono);width:100%;box-sizing:border-box}",
    ".scs-check{flex-direction:row!important;align-items:center;gap:8px!important;min-height:40px}",
    ".scs-check input{width:20px;height:20px}",
    ".scs-row{display:flex;flex-wrap:wrap;gap:8px;align-items:flex-end;margin:0 0 12px}",
    ".scs-row label{display:flex;flex-direction:column;gap:2px;font-size:13.5px;color:var(--mute);width:120px}",
    ".scs-row input{min-height:36px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);font:14px var(--mono);width:100%;box-sizing:border-box}",
    ".scs-sep{border:0;border-top:1px solid var(--border);margin:6px 0 12px}",
    ".scs-head{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline;margin:0 0 6px;font-size:14px}",
    ".scs-head b{font:700 16px var(--sans)}",
    ".scs-lab{font-size:13px;color:var(--mute);margin:8px 0 4px}",
    ".scs-bar{display:flex;flex-wrap:wrap;gap:2px;margin:0 0 4px}",
    ".scs-cell{min-width:22px;height:26px;padding:0 3px;box-sizing:border-box;border:1px solid var(--line);border-radius:3px;background:var(--surface);font:600 11.5px/24px var(--mono);text-align:center;color:var(--mute)}",
    ".scs-blk{min-width:34px}",
    ".scs-tw{overflow-x:auto;margin:10px 0 0;max-width:100%}",
    ".scs-table{border-collapse:collapse;font:13px var(--mono);width:100%;min-width:300px}",
    ".scs-table th,.scs-table td{border-bottom:1px solid var(--border);padding:4px 6px;text-align:left;white-space:nowrap}",
    ".scs-table th{font:600 12.5px var(--sans);color:var(--mute)}",
    ".scs-st-PREEMPTED{color:var(--viz-c4);font-weight:700}",
    ".scs-st-FINISHED{color:var(--mute)}",
    ".scs-log{margin:10px 0 0;padding:8px 10px 8px 26px;border:1px solid var(--border);border-radius:6px;background:var(--panel-bg);font-size:14px}",
    ".scs-log li{margin:2px 0}",
    ".scs-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}",
    "@media (max-width:560px){.scs-form{grid-template-columns:1fr 1fr}.scs-row label{width:calc(50% - 4px)}.scs-cell{min-width:18px}.scs-blk{min-width:30px}}"
  ].join("\n");

  function injectStyle() {
    if (document.getElementById("scs-style")) return;
    var st = document.createElement("style");
    st.id = "scs-style";
    var P = ".viz-mount[data-viz='" + VIZ_ID + "'] ";
    st.textContent = CSS.split("\n").map(function (r) {
      if (r.charAt(0) === "@") {
        return r.replace(/\{\.scs-/g, "{" + P + ".scs-").replace(/\}\.scs-/g, "}" + P + ".scs-");
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
  function color(idx) { return "var(--viz-c" + ((idx % 8) + 1) + ")"; }

  function numInput(labelText, key, value) {
    var lab = el("label");
    lab.appendChild(document.createTextNode(labelText));
    var inp = el("input");
    inp.type = "number";
    inp.min = LIMITS[key][0];
    inp.max = LIMITS[key][1];
    inp.step = 1;
    inp.value = value;
    inp.setAttribute("inputmode", "numeric");
    lab.appendChild(inp);
    return { label: lab, input: inp };
  }

  function Sim(mount) {
    this.mount = mount;
    this.reset();
    this.build();
    this.render(false);
  }

  Sim.prototype.reset = function () {
    this.cfg = { budget: DEFAULT_CFG.budget, seqs: DEFAULT_CFG.seqs, blocks: DEFAULT_CFG.blocks,
                 bs: DEFAULT_CFG.bs, chunked: DEFAULT_CFG.chunked };
    this.script = DEFAULT_SCRIPT.map(function (s) { return { prompt: s.prompt, maxOut: s.maxOut, arrive: s.arrive }; });
    this.steps = 0;
  };

  Sim.prototype.build = function () {
    var self = this;
    this.mount.innerHTML = "";
    var root = el("div", "scs-root");

    var form = el("div", "scs-form");
    this.inBudget = numInput("Token budget (max_num_batched_tokens)", "budget", this.cfg.budget);
    this.inSeqs = numInput("max_num_seqs", "seqs", this.cfg.seqs);
    this.inBlocks = numInput("Free KV blocks", "blocks", this.cfg.blocks);
    this.inBs = numInput("block_size (tokens)", "bs", this.cfg.bs);
    [this.inBudget, this.inSeqs, this.inBlocks, this.inBs].forEach(function (f) { form.appendChild(f.label); });
    var ck = el("label", "scs-check");
    this.inChunk = el("input");
    this.inChunk.type = "checkbox";
    this.inChunk.checked = this.cfg.chunked;
    ck.appendChild(this.inChunk);
    ck.appendChild(document.createTextNode("Chunked prefill"));
    form.appendChild(ck);
    root.appendChild(form);
    [this.inBudget, this.inSeqs, this.inBlocks, this.inBs].forEach(function (f) {
      f.input.addEventListener("change", function () { self.readCfg(); });
    });
    this.inChunk.addEventListener("change", function () { self.readCfg(); });

    var add = el("div", "scs-row");
    this.inPrompt = numInput("Prompt tokens", "prompt", 8);
    this.inOut = numInput("max_tokens", "maxOut", 3);
    add.appendChild(this.inPrompt.label);
    add.appendChild(this.inOut.label);
    this.addBtn = el("button", "viz-btn", "Add request");
    this.addBtn.type = "button";
    add.appendChild(this.addBtn);
    root.appendChild(add);
    this.addBtn.addEventListener("click", function () { self.addRequest(); });

    root.appendChild(el("hr", "scs-sep"));

    var ctl = el("div", "scs-row");
    this.stepBtn = el("button", "viz-btn viz-btn-primary", "Step →");
    this.backBtn = el("button", "viz-btn", "← Back");
    this.resetBtn = el("button", "viz-btn", "Reset");
    [this.stepBtn, this.backBtn, this.resetBtn].forEach(function (b) { b.type = "button"; ctl.appendChild(b); });
    root.appendChild(ctl);
    this.stepBtn.addEventListener("click", function () { self.steps += 1; self.render(true); });
    this.backBtn.addEventListener("click", function () { if (self.steps > 0) { self.steps -= 1; self.render(true); } });
    this.resetBtn.addEventListener("click", function () {
      self.reset();
      self.inBudget.input.value = self.cfg.budget;
      self.inSeqs.input.value = self.cfg.seqs;
      self.inBlocks.input.value = self.cfg.blocks;
      self.inBs.input.value = self.cfg.bs;
      self.inChunk.checked = self.cfg.chunked;
      self.render(true);
    });

    this.out = el("div", "scs-out");
    root.appendChild(this.out);
    this.live = el("p", "scs-live");
    this.live.setAttribute("aria-live", "polite");
    root.appendChild(this.live);
    this.mount.appendChild(root);
  };

  Sim.prototype.readCfg = function () {
    this.cfg.budget = clamp(this.inBudget.input.value, "budget");
    this.cfg.seqs = clamp(this.inSeqs.input.value, "seqs");
    this.cfg.blocks = clamp(this.inBlocks.input.value, "blocks");
    this.cfg.bs = clamp(this.inBs.input.value, "bs");
    this.cfg.chunked = this.inChunk.checked;
    this.inBudget.input.value = this.cfg.budget;
    this.inSeqs.input.value = this.cfg.seqs;
    this.inBlocks.input.value = this.cfg.blocks;
    this.inBs.input.value = this.cfg.bs;
    this.render(true, "Settings changed: the run is replayed from step 0 to step " + this.steps + ".");
  };

  Sim.prototype.addRequest = function () {
    if (this.script.length >= MAX_REQS) {
      this.live.textContent = "At most " + MAX_REQS + " requests.";
      return;
    }
    var p = clamp(this.inPrompt.input.value, "prompt");
    var m = clamp(this.inOut.input.value, "maxOut");
    this.inPrompt.input.value = p;
    this.inOut.input.value = m;
    this.script.push({ prompt: p, maxOut: m, arrive: this.steps + 1 });
    this.render(true, "Request " + LETTERS[this.script.length - 1] + " added. It joins the waiting queue at step " +
                (this.steps + 1) + ".");
  };

  Sim.prototype.render = function (announce, extra) {
    var cfg = this.cfg;
    var st = run(cfg, this.script, this.steps);
    this.backBtn.disabled = this.steps === 0;
    this.addBtn.disabled = this.script.length >= MAX_REQS;
    var o = this.out;
    o.innerHTML = "";

    var head = el("div", "scs-head");
    head.appendChild(el("b", null, this.steps === 0 ? "Before step 1" : "After step " + this.steps));
    head.appendChild(el("span", null, "budget used: " + (this.steps ? st.used : 0) + " / " + cfg.budget));
    head.appendChild(el("span", null, "free blocks: " + st.free.length + " / " + cfg.blocks));
    head.appendChild(el("span", null, "running: " + st.running.length + " / " + cfg.seqs));
    o.appendChild(head);

    // token budget of the last step
    o.appendChild(el("p", "scs-lab", "Token budget of the last step (one cell = one token)"));
    var bar = el("div", "scs-bar");
    bar.setAttribute("role", "img");
    var parts = [];
    var k;
    if (this.steps) {
      st.reqs.forEach(function (r) {
        var n = st.sched[r.id];
        if (!n) return;
        parts.push(r.id + " " + n);
        for (k = 0; k < n; k++) {
          var c = el("span", "scs-cell", k === 0 ? r.id : "");
          c.style.borderColor = color(r.idx);
          c.style.color = color(r.idx);
          c.style.boxShadow = "inset 0 -4px 0 " + color(r.idx);
          bar.appendChild(c);
        }
      });
    }
    var usedNow = this.steps ? st.used : 0;
    for (k = usedNow; k < cfg.budget; k++) bar.appendChild(el("span", "scs-cell", ""));
    bar.setAttribute("aria-label", "Token budget: " + (parts.length ? parts.join(", ") : "nothing scheduled") +
                     "; " + (cfg.budget - usedNow) + " unused.");
    o.appendChild(bar);

    // KV blocks
    o.appendChild(el("p", "scs-lab", "KV blocks 1-" + cfg.blocks + " (block 0 is the null block, not shown)"));
    var owner = {};
    st.reqs.forEach(function (r) { r.blockIds.forEach(function (b) { owner[b] = r; }); });
    var blk = el("div", "scs-bar");
    blk.setAttribute("role", "img");
    var held = [];
    for (k = 1; k <= cfg.blocks; k++) {
      var r = owner[k];
      var c2 = el("span", "scs-cell scs-blk", r ? r.id + k : String(k));
      if (r) {
        c2.style.borderColor = color(r.idx);
        c2.style.color = color(r.idx);
        c2.style.boxShadow = "inset 0 -4px 0 " + color(r.idx);
      }
      blk.appendChild(c2);
    }
    st.reqs.forEach(function (r) { if (r.blockIds.length) held.push(r.id + " holds " + r.blockIds.join(", ")); });
    blk.setAttribute("aria-label", "KV blocks: " + (held.length ? held.join("; ") : "all free") + ".");
    o.appendChild(blk);

    // request table
    var tw = el("div", "scs-tw");
    tw.setAttribute("tabindex", "0");
    tw.setAttribute("role", "region");
    tw.setAttribute("aria-label", "Requests");
    var t = el("table", "scs-table");
    var thr = el("tr");
    ["req", "status", "prompt", "out / max", "computed / tokens", "this step", "blocks"].forEach(function (h) {
      thr.appendChild(el("th", null, h));
    });
    var th = el("thead");
    th.appendChild(thr);
    t.appendChild(th);
    var tb = el("tbody");
    var self = this;
    this.script.forEach(function (s, i) {
      var r = null;
      st.reqs.forEach(function (x) { if (x.idx === i) r = x; });
      var tr = el("tr");
      var idc = el("td", null, LETTERS[i]);
      idc.style.color = color(i);
      idc.style.fontWeight = "700";
      tr.appendChild(idc);
      if (!r) {
        tr.appendChild(el("td", null, "arrives at step " + s.arrive));
        tr.appendChild(el("td", null, String(s.prompt)));
        tr.appendChild(el("td", null, "0 / " + s.maxOut));
        tr.appendChild(el("td", null, "-"));
        tr.appendChild(el("td", null, "-"));
        tr.appendChild(el("td", null, "0"));
      } else {
        tr.appendChild(el("td", "scs-st-" + r.status, r.status + (r.preemptions ? " (p" + r.preemptions + ")" : "")));
        tr.appendChild(el("td", null, String(r.prompt)));
        tr.appendChild(el("td", null, r.out + " / " + r.maxOut));
        tr.appendChild(el("td", null, r.computed + " / " + r.numTokens));
        tr.appendChild(el("td", null, self.steps && st.sched[r.id] ? String(st.sched[r.id]) : "0"));
        tr.appendChild(el("td", null, String(r.blockIds.length)));
      }
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    tw.appendChild(t);
    o.appendChild(tw);

    // log of the last step
    var log = el("ol", "scs-log");
    log.setAttribute("aria-label", "What schedule() did in the last step");
    var lines = this.steps ? st.log : ["Press Step to run schedule() once. The default example is the one in the figure above."];
    lines.forEach(function (s) { log.appendChild(el("li", null, s)); });
    o.appendChild(log);

    if (announce) {
      this.live.textContent = (extra ? extra + " " : "") +
        (this.steps ? "Step " + this.steps + ": " + (parts.length ? parts.join(", ") + " tokens" : "nothing scheduled") +
         ". " + st.free.length + " free blocks." : "Back at the start.");
    }
  };

  function init() {
    var mounts = document.querySelectorAll('[data-viz="' + VIZ_ID + '"]');
    if (!mounts.length) return;
    injectStyle();
    for (var i = 0; i < mounts.length; i++) new Sim(mounts[i]);
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
  if (typeof module !== "undefined") module.exports = { run: run };   // for a node check, unused in the browser
})();
