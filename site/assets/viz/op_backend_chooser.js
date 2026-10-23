/* op_backend_chooser.js - which forward_* method an RMSNorm CustomOp runs, and which attention backend
 * a layer gets, for a chosen platform and set of flags (vLLM Explained).
 *
 * Mounts into <x-viz id="op-backend-chooser"> on concepts/attention-backend-selection. Vanilla JS, no
 * network, no randomness (nothing to seed). The reader sets six switches; the visual shows (a) the
 * effective custom_ops list and the bound RMSNorm method, (b) the attention backend and the reasons.
 *
 * Fixed assumptions (the page states them): the GPU wheels have vllm_flash_attn with FA3 (SM90) and
 * FA4 (SM100), FlashInfer and FlashMLA import; ROCm runs with VLLM_ROCM_USE_AITER unset; the CPU is x86
 * with AVX-512 and no AMX; "acme_hw" is a made-up out-of-tree platform that keeps vLLM's custom_ops list
 * and returns its own backend class for every layer (like the test platform). No VLLM_BATCH_INVARIANT,
 * no --block-size, no KV connector, no sliding window, no sinks.
 *
 * Rules, copied from the code at the pinned commit (the page lists them with citations):
 *   C1 mode: --enforce-eager -> NONE (config/vllm.py:1745-1746); else O2 -> VLLM_COMPILE (:1806-1810)
 *   C2 compile backend "" -> platform default "inductor" (config/compilation.py:1080-1081; platforms/interface.py:172)
 *   C3 base mode added only if the list has neither "all" nor "none": "none" when backend is inductor and
 *      mode is not NONE, else "all" (config/vllm.py:1819-1826)
 *   C4 enabled = (base "all" or +name) and not -name (model_executor/custom_op.py:282-290, :299-307)
 *   C5 disabled -> forward_native; enabled -> rocm forward_hip, cpu forward_cpu, oot forward_oot, else
 *      forward_cuda (custom_op.py:189-205); RMSNorm overrides forward_native/forward_cuda (layernorm.py:79, :101)
 *      and forward_cuda calls forward_native without batch invariance (:120); forward_hip -> forward_cuda,
 *      forward_cpu / forward_oot -> forward_native by default (custom_op.py:147-149, :156-159, :167-170)
 *   C6 RMSNorm.forward_native calls the IR op rms_norm or fused_add_rms_norm (layernorm.py:79-99); CUDA default provider list is
 *      ["native"] with Inductor and a compile mode, else ["vllm_c", "native"] (platforms/cuda.py:716-734)
 *   Ignored on purpose (Llama-like model, RMSNorm only): +name entries that vLLM adds itself (+quant_fp8 for
 *   block-quantized weights, config/vllm.py:1799-1802; ROCm and ARM CPU additions), enforce_enable=True layers,
 *   classes that override enabled(), TORCH_COMPILE_DISABLE=1 and breakable CUDA graphs (both set mode NONE,
 *   config/vllm.py:1765-1770, :849-851)
 *   A1 CUDA: an explicit backend is validated alone; invalid -> ValueError (platforms/cuda.py:430-449)
 *   A2 CUDA priority lists (cuda.py:84-181); first valid entry wins (:479-485); none -> ValueError (:471-475)
 *   A3 validate_configuration: head size first (v1/attention/backend.py:302-303); use_mla must match is_mla() (:314-318);
 *      kv dtype (:306-307). An MLA layer has head_size 576: FLASH_ATTN (flash_attn.py:423-433) and FLASHINFER
 *      (flashinfer.py:662-664) reject it, TRITON_ATTN (head_size >= 32, triton_attn.py:378-379) does not
 *   A4 FLASH_ATTN: capability >= 8.0; fp8 KV needs FA3 on SM90 or FA4 on SM100 (flash_attn.py:288-294, :456-457;
 *      fa_utils.py:306-312)
 *   A5 FLASHINFER: head sizes 64/128/256/512, capability 8.0-12.1, fp8 KV accepted (flashinfer.py:544-553, :662-675)
 *   A6 TRITON_ATTN: any capability, fp8 KV accepted (triton_attn.py:325-334, :404-405)
 *   A7 FLASH_ATTN_MLA: KV auto/float16/bfloat16 only, capability major 9 (mla/flashattn_mla.py:42-46, :69-70)
 *   A8 FLASHMLA: fp8 KV accepted, dense only on Hopper (mla/flashmla.py:48-54; ops/flashmla.py:56-57)
 *   A9 FLASHINFER_MLA: capability major 10, fp8 KV accepted (mla/flashinfer_mla.py:195-201, :220-221)
 *   A10 ROCm: MHA list starts with ROCM_ATTN; MLA list is [TRITON_MLA] without AITER MLA (platforms/rocm.py:499-526);
 *       explicit invalid -> ValueError (:781-785)
 *   A11 CPU: an explicit backend other than CPU_ATTN only logs "Cannot use ..." (platforms/cpu.py:183-184);
 *       MLA + fp8 -> ValueError (:152-157); no AMX -> CPU_MLA (:158-182); fp8 needs AMX or AVX-512 (cpu_attn.py:69-75)
 *   A12 OOT: the platform's get_attn_backend_cls decides; the test platform ignores the selected backend
 *       (tests/plugins/vllm_add_dummy_platform/.../dummy_platform.py:25-32)
 */
(function () {
  "use strict";

  var VIZ_ID = "op-backend-chooser";

  var SWITCHES = [
    { key: "plat", label: "Platform", opts: [["sm90", "CUDA SM90"], ["sm100", "CUDA SM100"], ["rocm", "ROCm"], ["cpu", "CPU"], ["oot", "OOT acme_hw"]] },
    { key: "comp", label: "Compile mode", opts: [["o2", "default (O2, Inductor)"], ["eager", "--enforce-eager"]] },
    { key: "ops", label: "custom_ops", opts: [["def", "default"], ["all", "\"all\""], ["plus", "\"+rms_norm\""], ["minus", "\"-rms_norm\""]] },
    { key: "model", label: "Model kind", opts: [["mha", "MHA/GQA (Llama-3-8B)"], ["mla", "MLA (DeepSeek-V3)"]] },
    { key: "kv", label: "kv_cache_dtype", opts: [["auto", "auto"], ["fp8", "fp8"]] },
    { key: "be", label: "--attention-backend", opts: [["none", "not set"], ["FLASH_ATTN", "FLASH_ATTN"], ["FLASHINFER", "FLASHINFER"], ["TRITON_ATTN", "TRITON_ATTN"]] }
  ];
  var START = { plat: "sm90", comp: "o2", ops: "def", model: "mha", kv: "auto", be: "none" };

  var CSS = [
    ".obc-root{font-family:var(--sans);color:var(--ink)}",
    ".obc-sw{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 16px;margin:0 0 12px}",
    ".obc-group{border:0;margin:0;padding:0;min-width:0}",
    ".obc-group legend{font:600 13.5px var(--sans);color:var(--mute);margin:0 0 4px;padding:0}",
    ".obc-opts{display:flex;flex-wrap:wrap;gap:6px}",
    ".obc-opt{min-height:40px;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:500 13.5px var(--mono);cursor:pointer;touch-action:manipulation}",
    ".obc-opt[aria-pressed=true]{border-color:var(--blue);box-shadow:inset 0 -3px 0 var(--blue);font-weight:700}",
    ".obc-bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}",
    ".obc-out{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}",
    ".obc-card{border:1px solid var(--border);border-radius:8px;padding:10px 12px;background:var(--panel-bg);min-width:0}",
    ".obc-card h4{margin:0 0 6px;font-size:15px}",
    ".obc-res{display:inline-block;margin:0 0 8px;padding:4px 10px;border-radius:6px;border:1.5px solid var(--viz-c3);font:700 14px var(--mono);overflow-wrap:anywhere}",
    ".obc-res.obc-err{border-color:var(--viz-c8)}",
    ".obc-list{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:2px 10px;margin:0 0 8px;font-size:13.5px}",
    ".obc-list dt{font-weight:600;color:var(--mute)}",
    ".obc-list dd{margin:0;font-family:var(--mono);font-size:13px;overflow-wrap:anywhere}",
    ".obc-why{margin:0;padding:0 0 0 18px;font-size:13.5px}",
    ".obc-why li{margin:0 0 4px;overflow-wrap:anywhere}",
    ".obc-why i{font-style:normal;font:600 11.5px var(--mono);color:var(--blue)}",
    ".obc-cand{list-style:none;margin:0 0 8px;padding:0;font:13px var(--mono)}",
    ".obc-cand li{padding:2px 0;overflow-wrap:anywhere}",
    ".obc-cand .obc-win{font-weight:700;color:var(--viz-c3)}",
    ".obc-cand .obc-no{color:var(--mute)}",
    ".obc-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}",
    "@media (max-width:760px){.obc-out{grid-template-columns:minmax(0,1fr)}.obc-sw{grid-template-columns:minmax(0,1fr)}}"
  ].join("\n");

  function injectStyle() {
    if (document.getElementById("obc-style")) return;
    var st = document.createElement("style");
    st.id = "obc-style";
    var P = ".viz-mount[data-viz='" + VIZ_ID + "'] ";
    st.textContent = CSS.split("\n").map(function (r) {
      if (r.charAt(0) === "@") {
        return r.replace(/\{\.obc-/g, "{" + P + ".obc-").replace(/\}\.obc-/g, "}" + P + ".obc-");
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

  /* ---- (a) the CustomOp model: rules C1-C5 ------------------------------------------------- */
  function customOp(s) {
    var why = [];
    var mode = s.comp === "eager" ? "NONE" : "VLLM_COMPILE";
    why.push([s.comp === "eager" ? "--enforce-eager sets the compilation mode to NONE" :
      "optimization level O2 sets the compilation mode to VLLM_COMPILE", "C1"]);
    why.push(["the compile backend is \"inductor\" (platform default)", "C2"]);
    var user = { def: [], all: ["all"], plus: ["+rms_norm"], minus: ["-rms_norm"] }[s.ops];
    var list = user.slice();
    if (list.indexOf("all") < 0 && list.indexOf("none") < 0) {
      var base = mode !== "NONE" ? "none" : "all";
      list.push(base);
      why.push(["no base mode in the list, so vLLM appends \"" + base + "\"", "C3"]);
    } else {
      why.push(["the list already has a base mode, so vLLM adds none", "C3"]);
    }
    if (s.plat === "cpu" && s.comp !== "eager") {
      why.push(["the CPU platform changes the mode to DYNAMO_TRACE_ONCE later, after the base mode is fixed", "C3"]);
    }
    var baseOn = list.indexOf("all") >= 0;
    var plus = list.indexOf("+rms_norm") >= 0, minus = list.indexOf("-rms_norm") >= 0;
    var enabled = (baseOn || plus) && !minus;
    why.push(["rms_norm is " + (enabled ? "enabled" : "disabled") + ": (" + (baseOn ? "base on" : "base off") +
      (plus ? ", +rms_norm" : "") + ")" + (minus ? " and -rms_norm" : ""), "C4"]);
    var method, runs;
    if (!enabled) {
      method = "forward_native";
      runs = "PyTorch code; under compilation Inductor generates the kernel";
    } else if (s.plat === "rocm") {
      method = "forward_hip";
      runs = "default forward_hip calls forward_cuda, which calls forward_native";
    } else if (s.plat === "cpu") {
      method = "forward_cpu";
      runs = "default forward_cpu calls forward_native";
    } else if (s.plat === "oot") {
      method = "forward_oot";
      runs = "default forward_oot calls forward_native, unless acme_hw replaces RMSNorm with register_oot";
    } else {
      method = "forward_cuda";
      runs = "forward_cuda calls forward_native (no batch invariance)";
    }
    why.push([enabled ? "an enabled op takes the method of current_platform" : "a disabled op takes forward_native", "C5"]);
    why.push([runs, "C5"]);
    if (s.plat === "sm90" || s.plat === "sm100") {
      why.push(["forward_native calls the IR op rms_norm (fused_add_rms_norm with a residual); CUDA's provider list for it is " +
        (mode === "NONE" ? "[\"vllm_c\", \"native\"] (no Inductor)" : "[\"native\"] (Inductor)"), "C6"]);
    }
    return { list: "[" + list.map(function (x) { return "\"" + x + "\""; }).join(", ") + "]", method: method, why: why };
  }

  /* ---- (b) the attention model: rules A1-A12 ------------------------------------------------ */
  // Returns a list of reasons why `name` is invalid for state s on a CUDA GPU; empty = valid.
  function cudaReasons(name, s) {
    var mla = s.model === "mla", fp8 = s.kv === "fp8", sm = s.plat === "sm90" ? 90 : 100;
    var isMla = /MLA/.test(name), r = [];
    // validate_configuration order: head_size, dtype, kv dtype, ..., MLA, sparse, ..., supports_combination.
    // An MLA layer has head_size 576: FLASH_ATTN (max 512 with FA4) and FLASHINFER (64/128/256/512) reject it.
    if (mla && (name === "FLASH_ATTN" || name === "FLASHINFER")) r.push("head_size not supported");
    if (mla !== isMla) r.push(mla ? "MLA not supported" : "non-MLA not supported");
    if (/SPARSE/.test(name)) r.push("non-sparse not supported");
    switch (name) {
      case "FLASH_ATTN":                                          // A4: capability >= 8.0; fp8 ok with FA3/FA4
        // head_size 576 on SM100: FA4 falls back to FA2, which cannot read an fp8 KV cache (fa_utils.py:194-209, :306-312)
        if (mla && fp8 && sm === 100) r.push("FP8 KV cache requires FA3 on SM90, FA4 with head_size=512 on SM90, or FA4 on SM100");
        break;
      case "FLASHINFER": break;                                   // A5: head_size 128, capability 9.0/10.0
      case "TRITON_ATTN": break;                                  // A6
      case "FLASH_ATTN_MLA":                                      // A7
        if (fp8) r.push("kv_cache_dtype not supported");
        if (sm !== 90) r.push("compute capability not supported");
        break;
      case "FLASHMLA":                                            // A8
        if (sm !== 90) r.push("FlashMLA Dense is only supported on Hopper devices.");
        break;
      case "FLASHINFER_MLA":                                      // A9
        if (sm !== 100) r.push("compute capability not supported");
        break;
      default: break;
    }
    return r;
  }

  function cudaList(s) {
    var mla = s.model === "mla";
    if (s.plat === "sm90") {
      return mla ? ["FLASH_ATTN_MLA", "FLASHMLA", "FLASHINFER_MLA", "TRITON_MLA", "FLASH_ATTN_MLA_SPARSE", "FLASHMLA_SPARSE", "FLASHINFER_MLA_SPARSE_SM90"]
        : ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN", "FLEX_ATTENTION", "TURBOQUANT"];
    }
    return mla ? ["FLASHINFER_MLA", "TOKENSPEED_MLA", "CUTLASS_MLA", "FLASH_ATTN_MLA", "FLASHMLA", "TRITON_MLA", "(sparse MLA entries)"]
      : ["FLASHINFER", "FLASH_ATTN", "TRITON_ATTN", "FLEX_ATTENTION", "TURBOQUANT"];
  }

  function attention(s) {
    var why = [], cand = [], mla = s.model === "mla", fp8 = s.kv === "fp8";
    var res = { name: "", err: false, why: why, cand: cand, listLabel: "" };

    if (s.plat === "oot") {
      res.name = "acme_hw's own backend class";
      why.push(["the out-of-tree platform's get_attn_backend_cls decides; acme_hw returns its own class" +
        (s.be !== "none" ? " and ignores --attention-backend " + s.be : ""), "A12"]);
      return res;
    }

    if (s.plat === "cpu") {
      if (mla) {
        if (fp8) {
          res.err = true;
          res.name = "ValueError";
          why.push(["\"--kv-cache-dtype fp8 is not supported by MLA attention on CPU.\"", "A11"]);
          return res;
        }
        if (s.be !== "none") why.push(["INFO log \"Cannot use AttentionBackendEnum." + s.be + " backend on CPU.\"; no error", "A11"]);
        res.name = "CPU_MLA";
        why.push(["no AMX on this CPU, so the reference MLA backend CPU_MLA is used", "A11"]);
        return res;
      }
      if (s.be !== "none" && s.be !== "CPU_ATTN") why.push(["INFO log \"Cannot use AttentionBackendEnum." + s.be + " backend on CPU.\"; no error", "A11"]);
      res.name = "CPU_ATTN";
      why.push([fp8 ? "fp8 KV is accepted because the CPU has AVX-512" : "CPU_ATTN accepts the auto KV dtype", "A11"]);
      return res;
    }

    if (s.plat === "rocm") {
      if (s.be !== "none") {
        if (s.be === "TRITON_ATTN") {
          if (mla) {
            res.err = true; res.name = "ValueError";
            why.push(["\"Selected backend AttentionBackendEnum.TRITON_ATTN is not valid for this configuration. Reason: ['MLA not supported']\"", "A10 A3"]);
          } else {
            res.name = "TRITON_ATTN";
            why.push(["the selected backend passes validate_configuration, so ROCm uses it", "A10"]);
          }
        } else {
          res.name = "not modelled";
          res.err = true;
          why.push(["the result of " + s.be + " on ROCm depends on the installed packages; this visual does not model it", "A10"]);
        }
        return res;
      }
      res.listLabel = mla ? "ROCm MLA list (AITER off)" : "ROCm MHA list";
      var rl = mla ? ["TRITON_MLA"] : ["ROCM_ATTN", "(ROCM_AITER_UNIFIED_ATTN, if aiter is installed and the GPU is CDNA 3 or newer)", "TRITON_ATTN", "TURBOQUANT", "ULTRAQUANT"];
      rl.forEach(function (n, i) { cand.push({ n: n, ok: i === 0, why: i === 0 ? "valid, first in the list" : "" }); });
      res.name = rl[0];
      why.push([mla ? "without AITER MLA the ROCm MLA list is only TRITON_MLA" : "ROCM_ATTN is first and accepts the head size and the " + s.kv + " KV dtype", "A10"]);
      return res;
    }

    // CUDA SM90 / SM100
    if (s.be !== "none") {
      var rs = cudaReasons(s.be, s);
      if (rs.length) {
        res.err = true; res.name = "ValueError";
        why.push(["\"Selected backend AttentionBackendEnum." + s.be + " is not valid for this configuration. Reason: ['" + rs.join("', '") + "']\"", "A1 A3"]);
        if (s.be === "FLASH_ATTN" && mla && fp8 && s.plat === "sm90") {
          why.push(["if the FA4 build also imports on SM90, the list gets a third, FP8 reason", "A4"]);
        }
        why.push(["vLLM does not fall back to the priority list", "A1"]);
      } else {
        res.name = s.be;
        why.push(["only " + s.be + " is validated, and it passes", "A1"]);
        if (fp8 && s.be === "FLASH_ATTN") why.push(["fp8 KV works because FA" + (s.plat === "sm90" ? "3" : "4") + " runs on this GPU", "A4"]);
      }
      return res;
    }
    var lst = cudaList(s);
    res.listLabel = (mla ? "MLA" : "non-MLA") + " list for " + (s.plat === "sm90" ? "SM90" : "SM100");
    var winner = null;
    lst.forEach(function (n) {
      if (winner) { cand.push({ n: n, ok: false, why: "also validated; a larger index cannot win" }); return; }
      var r = cudaReasons(n, s);
      if (r.length) cand.push({ n: n, ok: false, why: r.join("; ") });
      else { winner = n; cand.push({ n: n, ok: true, why: "valid: smallest priority index" }); }
    });
    res.name = winner;
    why.push(["no explicit backend, so CUDA validates every entry of the priority list", "A2"]);
    why.push(["the valid entry with the smallest index wins", "A2"]);
    if (mla && fp8 && s.plat === "sm90") why.push(["FLASH_ATTN_MLA accepts only auto/float16/bfloat16 KV", "A7"]);
    return res;
  }

  /* ---- the widget ---------------------------------------------------------------------------- */
  function Chooser(mount) {
    this.mount = mount;
    this.state = {};
    for (var k in START) this.state[k] = START[k];
    this.build();
    this.render(false);
  }

  Chooser.prototype.build = function () {
    var self = this;
    this.mount.innerHTML = "";                       // removes the .viz-fallback text
    var root = el("div", "obc-root");
    var sw = el("div", "obc-sw");
    this.btns = {};
    SWITCHES.forEach(function (g) {
      var fs = el("fieldset", "obc-group");
      fs.appendChild(el("legend", null, g.label));
      var row = el("div", "obc-opts");
      self.btns[g.key] = g.opts.map(function (o) {
        var b = el("button", "obc-opt", o[1]);
        b.type = "button";
        b.setAttribute("data-val", o[0]);
        b.addEventListener("click", function () { self.set(g.key, o[0]); });
        row.appendChild(b);
        return b;
      });
      fs.appendChild(row);
      sw.appendChild(fs);
    });
    root.appendChild(sw);

    var bar = el("div", "obc-bar");
    var reset = el("button", "viz-btn", "Reset");
    reset.type = "button";
    reset.addEventListener("click", function () {
      for (var k in START) self.state[k] = START[k];
      self.render(true);
    });
    bar.appendChild(reset);
    bar.appendChild(el("span", "obc-note", "Simplified model (rules C1-C6 and A1-A12 below)."));
    root.appendChild(bar);

    var out = el("div", "obc-out");
    this.opCard = el("section", "obc-card");
    this.opCard.setAttribute("aria-label", "RMSNorm forward method");
    this.attnCard = el("section", "obc-card");
    this.attnCard.setAttribute("aria-label", "Attention backend");
    out.appendChild(this.opCard);
    out.appendChild(this.attnCard);
    root.appendChild(out);

    this.live = el("p", "obc-live");
    this.live.setAttribute("aria-live", "polite");
    root.appendChild(this.live);
    this.mount.appendChild(root);
  };

  Chooser.prototype.set = function (key, val) {
    this.state[key] = val;
    this.render(true);
  };

  function whyList(items) {
    var ol = el("ol", "obc-why");
    items.forEach(function (w) {
      var li = el("li", null, w[0] + " ");
      li.appendChild(el("i", null, w[1]));
      ol.appendChild(li);
    });
    return ol;
  }

  Chooser.prototype.render = function (announce) {
    var s = this.state, self = this;
    SWITCHES.forEach(function (g) {
      self.btns[g.key].forEach(function (b) {
        b.setAttribute("aria-pressed", b.getAttribute("data-val") === s[g.key] ? "true" : "false");
      });
    });
    var op = customOp(s), at = attention(s);

    this.opCard.innerHTML = "";
    this.opCard.appendChild(el("h4", null, "(a) RMSNorm (CustomOp \"rms_norm\")"));
    this.opCard.appendChild(el("span", "obc-res", op.method));
    var dl = el("dl", "obc-list");
    dl.appendChild(el("dt", null, "custom_ops"));
    dl.appendChild(el("dd", null, op.list));
    this.opCard.appendChild(dl);
    this.opCard.appendChild(whyList(op.why));

    this.attnCard.innerHTML = "";
    this.attnCard.appendChild(el("h4", null, "(b) Attention backend"));
    this.attnCard.appendChild(el("span", "obc-res" + (at.err ? " obc-err" : ""), at.name));
    if (at.cand.length) {
      this.attnCard.appendChild(el("p", null, at.listLabel + ":"));
      var ul = el("ul", "obc-cand");
      at.cand.forEach(function (c, i) {
        var li = el("li", c.ok ? "obc-win" : "obc-no", (i + 1) + ". " + c.n + (c.why ? " - " + c.why : ""));
        ul.appendChild(li);
      });
      this.attnCard.appendChild(ul);
    }
    this.attnCard.appendChild(whyList(at.why));

    if (announce) {
      this.live.textContent = "RMSNorm runs " + op.method + ". Attention backend: " + at.name + ".";
    }
  };

  function init() {
    var mounts = document.querySelectorAll('[data-viz="' + VIZ_ID + '"]');
    if (!mounts.length) return;
    injectStyle();
    for (var i = 0; i < mounts.length; i++) new Chooser(mounts[i]);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
