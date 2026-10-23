/* tuning_playground.js - KV cache sizing playground for concepts/tuning (vLLM Explained).
 *
 * Mounts into <x-viz id="tuning-playground">. Vanilla JS, no network, no randomness: the result depends
 * only on the inputs, so every reader sees the same numbers. "Reset" returns to the default example
 * (the 80 GiB / Llama-3.1-8B example of concepts/kv-cache-manager).
 *
 * Formulas copied from the code (pinned commit; the page lists them with citations):
 *   requested   = ceil(total_memory * gpu_memory_utilization)                 (v1/worker/utils.py:724-726)
 *   available   = requested - (consumed + peak_headroom) - cudagraph_estimate  (v1/worker/gpu_worker.py:744-748,
 *                                                                               utils/mem_utils.py:312-321)
 *   with --kv-cache-memory-bytes: available = that value, no memory profiling (v1/worker/gpu_worker.py:653-679)
 *   bytes_per_block = kv_bytes_per_token * block_size  (one full-attention group: page size per layer
 *                     summed over the layers, v1/kv_cache_interface.py:571-579, v1/core/kv_cache_planning.py:587-593)
 *   num_blocks  = available // bytes_per_block                             (v1/core/kv_cache_planning.py:1610-1611)
 *   fit check   = cdiv(max_model_len, block_size) * bytes_per_block <= available - bytes_per_block
 *                 (one block held back for the null block)                 (v1/core/kv_cache_planning.py:1572-1598, 68-87)
 *   no memory   = available - bytes_per_block <= 0 ("No available memory") (v1/core/kv_cache_planning.py:56-64, 1590-1598)
 *   max_concurrency = num_blocks / cdiv(max_model_len, block_size)           (v1/core/kv_cache_utils.py:1015-1047)
 *   logged tokens   = int(max_concurrency * max_model_len)                   (v1/core/kv_cache_utils.py:1074-1082)
 *   vllm serve batch defaults by GPU memory, throughput mode x2, caps       (engine/arg_utils.py:2960-2992, 3171-3221)
 * Formatting as in vLLM: "%.2f" (round half to even on exact ties) for the concurrency, str(round(x, 2)) for GiB
 * in log lines (v1/core/kv_cache_utils.py:1101-1108; utils/mem_utils.py:30-31).
 * Preset KV sizes and weights are inferred from the models' Hugging Face configs (not in the vLLM repository).
 * Left out: hybrid models (several KV cache groups), TP (enter per-GPU values), prefix caching, the real
 * growth of blocks during decode, preemption, speculative decoding look-ahead slots.
 */
(function () {
  "use strict";

  var VIZ_ID = "tuning-playground";
  var GiB = 1073741824;
  var KiB = 1024;

  var PRESETS = [
    { id: "l8", label: "Llama-3.1-8B, BF16 KV (128 KiB/token, 15 GiB weights)", kv: 128, w: 15 },
    { id: "l8f", label: "Llama-3.1-8B, FP8 KV (64 KiB/token, 15 GiB weights)", kv: 64, w: 15 },
    { id: "l70", label: "Llama-3.1-70B, TP=4, BF16 KV, per GPU (80 KiB/token, 33 GiB weights)", kv: 80, w: 33 },
    { id: "custom", label: "Custom (type your own values)", kv: null, w: null }
  ];

  var DEFAULTS = {
    preset: "l8", gpu: 80, util: 0.92, weights: 15, other: 1.0, act: 2.5, cg: 0.5,
    kvKiB: 128, bs: 16, mml: 131072, seqs: 1024, tokens: 8192, avg: 2500, kvBytes: 0,
    a100: false, tput: false
  };

  var LIMITS = {
    gpu: [1, 1024], util: [0.01, 1], weights: [0, 1024], other: [0, 64], act: [0, 256], cg: [0, 64],
    kvKiB: [1, 4096], bs: [1, 2048], mml: [16, 10485760], seqs: [1, 65536], tokens: [1, 1048576],
    avg: [1, 10485760], kvBytes: [0, 1024]
  };

  function cdiv(a, b) { return Math.floor((a + b - 1) / b); }
  function clampNum(v, k, isInt) {
    var n = isInt ? parseInt(v, 10) : parseFloat(v);
    if (isNaN(n)) n = DEFAULTS[k];
    n = Math.max(LIMITS[k][0], Math.min(LIMITS[k][1], n));
    return isInt ? Math.round(n) : n;
  }
  function fmtInt(n) { return Math.floor(n).toLocaleString("en-US"); }
  function fmtGiB(b) { return (Math.round(b / GiB * 100) / 100).toFixed(2); }
  // vLLM's format_gib: str(round(b / GiB, 2)), for example "54.6" (utils/mem_utils.py:30-31)
  function logGiB(b) {
    var v = Math.round(b / GiB * 100) / 100;
    return v === Math.floor(v) ? v.toFixed(1) : String(v);
  }
  // Python "%.2f" rounds the exact binary value, half to even. An exact tie at the third decimal is only
  // possible for odd multiples of 1/8 (x.125, x.375, x.625, x.875); toFixed rounds those up, so handle them here.
  function fmt2(x) {
    var e = x * 8;   // exact in binary
    if (e === Math.floor(e) && Math.abs(e % 2) === 1) {
      var lo = Math.floor(x * 100);  // x * 100 is exact here (x.125 -> x12.5)
      return ((lo % 2 === 0 ? lo : lo + 1) / 100).toFixed(2);
    }
    return x.toFixed(2);
  }

  /* ---------------------------------------------------------------- the model */

  // engine/arg_utils.py get_batch_defaults(): usage context OPENAI_API_SERVER (vllm serve)
  function serveDefaults(gpuGiB, isA100, tput) {
    var tok, seqs;
    if (gpuGiB >= 160) { tok = 16384; seqs = 1024; }
    else if (gpuGiB >= 70 && !isA100) { tok = 8192; seqs = 1024; }
    else { tok = 2048; seqs = 256; }
    if (tput) { tok *= 2; seqs *= 2; }
    return { tokens: tok, seqs: seqs };
  }

  // the two caps that apply only to defaults (chunked prefill on)
  function defaultsWithCaps(c) {
    var d = serveDefaults(c.gpu, c.a100, c.tput);
    var tok = Math.min(d.seqs * c.mml, d.tokens);   // arg_utils.py:3208-3211
    var seqs = Math.min(d.seqs, tok);                // arg_utils.py:3219-3221
    return { tokens: tok, seqs: seqs };
  }

  function compute(c) {
    var r = {};
    r.total = c.gpu * GiB;
    r.requested = Math.ceil(r.total * c.util);
    r.consumed = (c.weights + c.other) * GiB;
    r.headroom = c.act * GiB;
    r.cgEst = c.cg * GiB;
    r.fixed = c.kvBytes > 0;
    if (r.fixed) {
      r.available = Math.floor(c.kvBytes * GiB);
    } else {
      r.available = r.requested - (r.consumed + r.headroom) - r.cgEst;
    }
    r.bpt = c.kvKiB * KiB;
    r.bpb = r.bpt * c.bs;
    r.numBlocks = r.available > 0 ? Math.floor(r.available / r.bpb) : 0;
    r.blocksPerMax = cdiv(c.mml, c.bs);
    r.neededForMax = r.blocksPerMax * r.bpb;
    r.checkMemory = r.available - r.bpb;
    r.noMemory = r.checkMemory <= 0;   // the check runs on available - one block (kv_cache_planning.py:1575-1598)
    r.fits = !r.noMemory && r.neededForMax <= r.checkMemory;
    // estimate_max_model_len: the largest length whose blocks fit in checkMemory
    r.estMaxLen = r.checkMemory > 0 ? Math.floor(r.checkMemory / r.bpb) * c.bs : 0;
    if (r.estMaxLen > c.mml) r.estMaxLen = c.mml;
    r.conc = r.numBlocks / r.blocksPerMax;
    r.loggedTokens = Math.floor(r.conc * c.mml);
    r.slots = r.numBlocks * c.bs;
    r.blocksPerAvg = cdiv(Math.min(c.avg, c.mml), c.bs);
    r.avgFit = r.numBlocks > 1 ? Math.floor((r.numBlocks - 1) / r.blocksPerAvg) : 0;
    r.running = Math.min(r.avgFit, c.seqs);
    r.limitBy = r.avgFit < c.seqs ? "KV cache" : "max_num_seqs";
    r.decodeTok = Math.min(r.running, c.tokens);
    r.prefillLeft = Math.max(0, c.tokens - r.running);
    r.utilEff = r.fixed ? null : r.available / r.total;
    // arg_utils caps (only for defaults; shown as a note when the user's values break them)
    r.capTokens = c.tokens > c.seqs * c.mml;
    r.capSeqs = c.seqs > c.tokens;
    return r;
  }

  /* ---------------------------------------------------------------- the view */

  var CSS = [
    ".tpg-root{font-family:var(--sans);color:var(--ink)}",
    ".tpg-sel{display:flex;flex-direction:column;gap:2px;font-size:13.5px;color:var(--mute);margin:0 0 10px}",
    ".tpg-sel select{min-height:40px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);font:14px var(--sans);max-width:100%}",
    ".tpg-form{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:8px 12px;margin:0 0 10px;align-items:end}",
    ".tpg-form label{display:flex;flex-direction:column;gap:2px;font-size:13.5px;color:var(--mute)}",
    ".tpg-form input[type=number]{min-height:36px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);font:14px var(--mono);width:100%;box-sizing:border-box}",
    ".tpg-check{flex-direction:row!important;align-items:center;gap:8px!important;min-height:40px}",
    ".tpg-check input{width:20px;height:20px}",
    ".tpg-grp{font:600 13px var(--sans);color:var(--ink);margin:10px 0 6px}",
    ".tpg-row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}",
    ".tpg-sep{border:0;border-top:1px solid var(--border);margin:6px 0 12px}",
    ".tpg-bar{display:flex;width:100%;height:30px;border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:4px 0 4px;background:var(--surface)}",
    ".tpg-seg{height:100%;min-width:0}",
    ".tpg-leg{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:13px;color:var(--mute);margin:0 0 10px}",
    ".tpg-leg span{display:inline-flex;align-items:center;gap:6px}",
    ".tpg-sw{display:inline-block;width:12px;height:12px;border-radius:3px}",
    ".tpg-out{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:8px;margin:8px 0}",
    ".tpg-card{border:1px solid var(--border);border-radius:8px;padding:8px 10px;background:var(--panel-bg)}",
    ".tpg-card b{display:block;font:700 18px var(--mono);color:var(--ink)}",
    ".tpg-card span{font-size:13px;color:var(--mute)}",
    ".tpg-log{font:13px var(--mono);background:var(--panel-bg);border:1px solid var(--border);border-radius:6px;padding:8px 10px;margin:8px 0;white-space:pre-wrap;word-break:break-word}",
    ".tpg-warn{border:1px solid var(--viz-c4);border-left-width:5px;border-radius:6px;padding:8px 10px;margin:8px 0;font-size:14px}",
    ".tpg-ok{border:1px solid var(--viz-c3);border-left-width:5px;border-radius:6px;padding:8px 10px;margin:8px 0;font-size:14px}",
    ".tpg-steps{font-size:14px;margin:8px 0 0;padding-left:22px}",
    ".tpg-steps li{margin:2px 0}",
    ".tpg-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}",
    "@media (max-width:560px){.tpg-form{grid-template-columns:1fr 1fr}.tpg-out{grid-template-columns:1fr}}"
  ].join("\n");

  function injectStyle() {
    if (document.getElementById("tpg-style")) return;
    var st = document.createElement("style");
    st.id = "tpg-style";
    var P = ".viz-mount[data-viz='" + VIZ_ID + "'] ";
    st.textContent = CSS.split("\n").map(function (r) {
      if (r.charAt(0) === "@") {
        return r.replace(/\{\.tpg-/g, "{" + P + ".tpg-").replace(/\}\.tpg-/g, "}" + P + ".tpg-");
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

  function numInput(labelText, key, value, step) {
    var lab = el("label");
    lab.appendChild(document.createTextNode(labelText));
    var inp = el("input");
    inp.type = "number";
    inp.min = LIMITS[key][0];
    inp.max = LIMITS[key][1];
    inp.step = step || 1;
    inp.value = value;
    inp.setAttribute("inputmode", step && step < 1 ? "decimal" : "numeric");
    lab.appendChild(inp);
    return { label: lab, input: inp, key: key, isInt: !(step && step < 1) };
  }

  function checkInput(labelText, value) {
    var lab = el("label", "tpg-check");
    var inp = el("input");
    inp.type = "checkbox";
    inp.checked = value;
    lab.appendChild(inp);
    lab.appendChild(document.createTextNode(labelText));
    return { label: lab, input: inp };
  }

  function Playground(mount) {
    this.mount = mount;
    this.reset();
    this.build();
    this.render(false);
  }

  Playground.prototype.reset = function () {
    this.c = {};
    for (var k in DEFAULTS) if (Object.prototype.hasOwnProperty.call(DEFAULTS, k)) this.c[k] = DEFAULTS[k];
  };

  Playground.prototype.build = function () {
    var self = this;
    this.mount.innerHTML = "";
    var root = el("div", "tpg-root");

    var sel = el("label", "tpg-sel");
    sel.appendChild(document.createTextNode("Model preset (values per GPU, inferred from each model's config)"));
    this.inPreset = el("select");
    PRESETS.forEach(function (p) {
      var o = el("option", null, p.label);
      o.value = p.id;
      self.inPreset.appendChild(o);
    });
    sel.appendChild(this.inPreset);
    root.appendChild(sel);
    this.inPreset.addEventListener("change", function () { self.applyPreset(); });

    this.fields = [];
    function group(title, defs) {
      root.appendChild(el("p", "tpg-grp", title));
      var form = el("div", "tpg-form");
      defs.forEach(function (d) {
        var f = numInput(d[0], d[1], self.c[d[1]], d[2]);
        self.fields.push(f);
        form.appendChild(f.label);
        f.input.addEventListener("change", function () { self.readCfg(f.key); });
      });
      root.appendChild(form);
      return form;
    }
    group("GPU memory (one GPU)", [
      ["GPU memory (GiB)", "gpu", 1],
      ["gpu_memory_utilization", "util", 0.01],
      ["--kv-cache-memory-bytes (GiB, 0 = off)", "kvBytes", 0.5]
    ]);
    group("What the profile measures (your inputs)", [
      ["Model weights (GiB)", "weights", 0.5],
      ["Other consumed: buffers, NCCL, cuBLAS (GiB)", "other", 0.1],
      ["Peak activations of the profile run (GiB)", "act", 0.1],
      ["CUDA graph estimate (GiB)", "cg", 0.1]
    ]);
    group("KV cache shape", [
      ["KV bytes per token (KiB)", "kvKiB", 1],
      ["block_size (tokens)", "bs", 1],
      ["max_model_len (tokens)", "mml", 1]
    ]);
    var sched = group("Scheduler and workload", [
      ["max_num_seqs", "seqs", 1],
      ["max_num_batched_tokens", "tokens", 1],
      ["Average request length, prompt + output (tokens)", "avg", 1]
    ]);

    var row = el("div", "tpg-row");
    this.ckA100 = checkInput("GPU name has A100, or GB10", this.c.a100);
    this.ckTput = checkInput("--performance-mode throughput", this.c.tput);
    row.appendChild(this.ckA100.label);
    row.appendChild(this.ckTput.label);
    this.defBtn = el("button", "viz-btn", "Use vllm serve defaults");
    this.defBtn.type = "button";
    row.appendChild(this.defBtn);
    this.resetBtn = el("button", "viz-btn", "Reset");
    this.resetBtn.type = "button";
    row.appendChild(this.resetBtn);
    root.appendChild(row);
    this.defBtn.addEventListener("click", function () { self.useDefaults(); });
    this.resetBtn.addEventListener("click", function () {
      self.reset();
      self.syncInputs();
      self.render(true, "Back to the default example.");
    });
    this.ckA100.input.addEventListener("change", function () { self.c.a100 = self.ckA100.input.checked; self.render(true); });
    this.ckTput.input.addEventListener("change", function () { self.c.tput = self.ckTput.input.checked; self.render(true); });
    void sched;

    root.appendChild(el("hr", "tpg-sep"));
    this.out = el("div");
    root.appendChild(this.out);
    this.live = el("p", "tpg-live");
    this.live.setAttribute("aria-live", "polite");
    root.appendChild(this.live);
    this.mount.appendChild(root);
    this.syncInputs();
  };

  Playground.prototype.syncInputs = function () {
    var c = this.c;
    this.fields.forEach(function (f) { f.input.value = c[f.key]; });
    this.inPreset.value = c.preset;
    this.ckA100.input.checked = c.a100;
    this.ckTput.input.checked = c.tput;
  };

  Playground.prototype.applyPreset = function () {
    var id = this.inPreset.value;
    this.c.preset = id;
    var p = null;
    PRESETS.forEach(function (x) { if (x.id === id) p = x; });
    if (p && p.kv != null) {
      this.c.kvKiB = p.kv;
      this.c.weights = p.w;
    }
    this.syncInputs();
    this.render(true, "Preset: " + (p ? p.label : id) + ".");
  };

  Playground.prototype.readCfg = function (changed) {
    var c = this.c;
    this.fields.forEach(function (f) { c[f.key] = clampNum(f.input.value, f.key, f.isInt); });
    if (changed === "kvKiB" || changed === "weights") c.preset = "custom";
    this.syncInputs();
    this.render(true);
  };

  Playground.prototype.useDefaults = function () {
    var d = defaultsWithCaps(this.c);
    this.c.tokens = d.tokens;
    this.c.seqs = d.seqs;
    this.syncInputs();
    this.render(true, "vllm serve defaults: max_num_batched_tokens " + d.tokens + ", max_num_seqs " + d.seqs + ".");
  };

  function card(parent, value, label) {
    var d = el("div", "tpg-card");
    d.appendChild(el("b", null, value));
    d.appendChild(el("span", null, label));
    parent.appendChild(d);
  }

  Playground.prototype.render = function (announce, extra) {
    var c = this.c;
    var r = compute(c);
    var o = this.out;
    o.innerHTML = "";

    // memory bar
    o.appendChild(el("p", "tpg-grp", r.fixed ? "GPU memory (KV size fixed by --kv-cache-memory-bytes)" :
                     "How the requested memory splits (one GPU)"));
    var bar = el("div", "tpg-bar");
    bar.setAttribute("role", "img");
    var parts = [];
    var total = r.total;
    function seg(bytes, col, name) {
      if (bytes <= 0) return;
      var s = el("div", "tpg-seg");
      s.style.width = Math.max(0, Math.min(100, bytes / total * 100)) + "%";
      s.style.background = col;
      bar.appendChild(s);
      parts.push(name + " " + fmtGiB(bytes) + " GiB");
    }
    if (r.fixed) {
      seg(r.consumed, "var(--viz-c1)", "weights and other");
      seg(r.available, "var(--viz-c3)", "KV cache (fixed)");
    } else {
      seg(r.consumed, "var(--viz-c1)", "weights and other");
      seg(r.headroom, "var(--viz-c5)", "peak activations");
      seg(r.cgEst, "var(--viz-c7)", "CUDA graph estimate");
      seg(Math.max(0, r.available), "var(--viz-c3)", "KV cache");
    }
    bar.setAttribute("aria-label", "Memory: " + parts.join(", ") + " of " + fmtGiB(total) + " GiB.");
    o.appendChild(bar);
    var leg = el("div", "tpg-leg");
    [["var(--viz-c1)", "weights + other consumed"], ["var(--viz-c5)", "peak activations"],
     ["var(--viz-c7)", "CUDA graph estimate"], ["var(--viz-c3)", "KV cache"],
     ["var(--surface)", "not requested / free"]].forEach(function (x) {
      var s = el("span");
      var sw = el("i", "tpg-sw");
      sw.style.background = x[0];
      sw.style.border = "1px solid var(--line)";
      s.appendChild(sw);
      s.appendChild(document.createTextNode(x[1]));
      leg.appendChild(s);
    });
    o.appendChild(leg);

    // result cards
    var grid = el("div", "tpg-out");
    card(grid, fmtGiB(Math.max(0, r.available)) + " GiB", r.fixed ? "KV cache (as given)" :
         "Available KV cache memory (requested " + fmtGiB(r.requested) + " GiB)");
    card(grid, fmtInt(r.numBlocks), "KV blocks (" + fmtInt(r.bpb / KiB) + " KiB each); block 0 is the null block");
    card(grid, fmtInt(r.slots), "token slots = blocks × block_size");
    card(grid, r.fits ? fmt2(r.conc) + "x" : "-", "Maximum concurrency for " + fmtInt(c.mml) +
         " tokens per request" + (r.fits ? "" : " (not logged: start-up fails)"));
    card(grid, fmtInt(r.avgFit), "requests of " + fmtInt(c.avg) + " tokens that fit in the KV cache at once");
    card(grid, fmtInt(r.running), "can run at once (limit: " + r.limitBy + ")");
    o.appendChild(grid);

    // the log lines vLLM prints
    var log = el("div", "tpg-log");
    var lines = [];
    if (r.fixed) {
      lines.push("Initial free memory ... GiB, reserved " + logGiB(r.available) +
                 " GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling. ...");
    } else {
      lines.push("Available KV cache memory: " + logGiB(r.available) + " GiB");
    }
    if (!r.noMemory && r.fits) {
      lines.push("GPU KV cache size: " + fmtInt(r.loggedTokens) + " tokens, Maximum concurrency for " +
                 fmtInt(c.mml) + " tokens per request: " + fmt2(r.conc) + "x");
    }
    log.textContent = lines.join("\n");
    log.setAttribute("aria-label", "Start-up log lines that vLLM would print");
    o.appendChild(log);

    // warnings
    var msg;
    if (r.noMemory) {
      msg = el("div", "tpg-warn");
      msg.textContent = "Start-up fails: ValueError \"No available memory for the cache blocks.\" " +
        "Raise gpu_memory_utilization, or lower the memory that the profile measures (for example a smaller " +
        "max_num_batched_tokens gives a smaller activation peak).";
      o.appendChild(msg);
    } else if (!r.fits) {
      msg = el("div", "tpg-warn");
      msg.textContent = "Start-up fails: one request of max_model_len " + fmtInt(c.mml) + " needs " +
        fmtGiB(r.neededForMax) + " GiB of KV cache, but only " + fmtGiB(r.checkMemory) +
        " GiB are left after the null block." + (r.estMaxLen > 0 ? " vLLM says: \"the estimated maximum model length is " +
        r.estMaxLen + "\"." : "") + " Lower --max-model-len, use --max-model-len auto, or add memory.";
      o.appendChild(msg);
    } else {
      msg = el("div", "tpg-ok");
      msg.textContent = "Start-up passes the KV check: one request of max_model_len needs " +
        fmtInt(r.blocksPerMax) + " of " + fmtInt(r.numBlocks - 1) + " usable blocks.";
      o.appendChild(msg);
    }
    if (r.capSeqs) {
      var n2 = el("div", "tpg-warn");
      n2.textContent = "Start-up fails: max_num_batched_tokens (" + c.tokens + ") must be greater than or " +
        "equal to max_num_seqs (" + c.seqs + "). vLLM caps only its own default max_num_seqs; " +
        "a value that you set raises ValueError.";
      o.appendChild(n2);
    }
    var d = defaultsWithCaps(c);
    if (c.tokens !== d.tokens || c.seqs !== d.seqs) {
      var n = el("p", "tpg-steps");
      n.textContent = "Note: vllm serve on this GPU defaults to max_num_batched_tokens " + d.tokens +
        " and max_num_seqs " + d.seqs + " (press \"Use vllm serve defaults\").";
      o.appendChild(n);
    }

    // step budget
    var ol = el("ol", "tpg-steps");
    ol.setAttribute("aria-label", "One decode-heavy step");
    [
      "One step at full load: " + fmtInt(r.running) + " running requests each take 1 decode token: " +
        fmtInt(r.decodeTok) + " of " + fmtInt(c.tokens) + " budget tokens.",
      "Tokens left in the step for prefill chunks: " + fmtInt(r.prefillLeft) + ".",
      "A new prompt of " + fmtInt(c.avg) + " tokens needs about " +
        (r.prefillLeft > 0 ? fmtInt(cdiv(c.avg, r.prefillLeft)) : "many") + " step(s) of prefill at this load."
    ].forEach(function (s) { ol.appendChild(el("li", null, s)); });
    o.appendChild(ol);

    if (announce) {
      this.live.textContent = (extra ? extra + " " : "") + fmtInt(r.numBlocks) + " blocks, " +
        (r.noMemory ? "no KV memory, start-up fails" :
         (r.fits ? "maximum concurrency " + fmt2(r.conc) + "x" : "max_model_len does not fit, start-up fails")) +
        ". " + fmtInt(r.running) + " requests of the average length fit at once.";
    }
  };

  function init() {
    var mounts = document.querySelectorAll('[data-viz="' + VIZ_ID + '"]');
    if (!mounts.length) return;
    injectStyle();
    for (var i = 0; i < mounts.length; i++) new Playground(mounts[i]);
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
  if (typeof module !== "undefined") module.exports = { compute: compute, serveDefaults: serveDefaults, fmt2: fmt2 };
})();
