/* plugin_timeline.js - start-up timeline of `vllm serve` with plugin groups and Platform hooks (vLLM Explained).
 *
 * Mounts into <x-viz id="plugin-timeline"> on concepts/plugin-map. Vanilla JS, no network, no randomness
 * (nothing to seed). The reader sets four switches; the visual lists, per process (API server, engine
 * core, worker), the plugin groups and Platform hooks that run, numbered in start-up order.
 *
 * Made-up installed packages (fixed): vLLM's own general plugins lora_filesystem_resolver and
 * lora_hf_hub_resolver (pyproject.toml:45-47), a stat logger plugin "acme_metrics", an endpoint plugin
 * "acme_admin", and (switch) an out-of-tree platform plugin "acme_hw". The machine has NVIDIA GPUs.
 *
 * Rules (R1-R16), copied from the code at the pinned commit; the page lists them with citations:
 *   R1  each process resolves current_platform on first use (platforms/__init__.py:303-319)
 *   R2  VLLM_TARGET_DEVICE=cpu -> CPU platform, platform plugin group not loaded (:244-248)
 *   R3  exactly one active out-of-tree platform wins over built-ins (:272-279)
 *   R4  VLLM_PLUGINS filters general, platform, stat logger, endpoint, IO processor groups (plugins/__init__.py:63-64)
 *   R5  general plugins: once per process, guard plugins_loaded (plugins/__init__.py:82-90)
 *   R6  front end: add_cli_args loads general plugins, then pre_register_and_update(parser) (arg_utils.py:3243, :3257)
 *   R7  create_engine_config: pre_register_and_update(), validate_environ, VllmConfig.__post_init__ with
 *       apply_config_platform_defaults and check_and_update_config (arg_utils.py:2246-2250; vllm.py:1804, :2046)
 *   R8  executor: uni when world size is 1, mp otherwise (parallel.py:1055-1100); CPU platform turns uni into mp
 *       (cpu.py:346-353); CPU turns async scheduling off (cpu.py:343)
 *   R9  worker_cls from check_and_update_config: CUDA gpu_worker.Worker (cuda.py:330-331), CPU CPUWorker (cpu.py:355-356)
 *   R10 stat logger plugins in AsyncLLM.__init__ (async_llm.py:155-156)
 *   R11 vllm.logits_processors ignores VLLM_PLUGINS (gpu/sample/logits_processor/loader.py:46; input_processor.py:103-107);
 *       the MRV2 CPU runner subclasses the GPU runner (v1/worker/cpu/model_runner.py:10), so load_model loads it too
 *   R12 engine core: general plugins (core.py:132), executor (:147), KV cache config (:311), scheduler class (:164)
 *   R13 worker: init_worker = general plugins, register_triton_kernel_overrides, worker_cls (worker_base.py:284-296);
 *       device communicator only with > 1 rank (parallel_state.py:549-552); attention backend (selector.py:218);
 *       update_block_size_for_backend (multiproc_executor.py:702, uniproc_executor.py:75)
 *   R14 endpoint plugins only when VLLM_PLUGINS names them (plugins/__init__.py:122-131), after the engine is ready
 *   R15 engine core and workers are spawned children: new interpreters (startup.py:147-149)
 *   R16 get_scheduler_cls: AsyncScheduler when async scheduling is on, else Scheduler (config/scheduler.py:231-239)
 * Simplified: no DP, one API server, no Ray, no model registry subprocess, no IO processor (pooling only).
 */
(function () {
  "use strict";

  var VIZ_ID = "plugin-timeline";

  var SWITCHES = [
    { key: "tp", label: "Tensor parallel size", opts: [["1", "TP = 1"], ["2", "TP = 2"]] },
    { key: "plugins", label: "VLLM_PLUGINS", opts: [["unset", "not set"], ["two", "acme_hw,acme_admin"], ["one", "acme_admin"]] },
    { key: "target", label: "VLLM_TARGET_DEVICE", opts: [["cuda", "not set (cuda)"], ["cpu", "cpu"]] },
    { key: "oot", label: "Out-of-tree platform acme_hw", opts: [["no", "not installed"], ["yes", "installed"]] }
  ];
  var START = { tp: "1", plugins: "unset", target: "cuda", oot: "no" };

  var LANES = [
    { key: "api", name: "API server (front end)" },
    { key: "core", name: "Engine core process" },
    { key: "wrk", name: "Worker process" }
  ];

  var CSS = [
    ".plt-root{font-family:var(--sans);color:var(--ink)}",
    ".plt-sw{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 16px;margin:0 0 12px}",
    ".plt-group{border:0;margin:0;padding:0;min-width:0}",
    ".plt-group legend{font:600 13.5px var(--sans);color:var(--mute);margin:0 0 4px;padding:0}",
    ".plt-opts{display:flex;flex-wrap:wrap;gap:6px}",
    ".plt-opt{min-height:40px;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);color:var(--ink);font:500 13.5px var(--mono);cursor:pointer;touch-action:manipulation}",
    ".plt-opt[aria-pressed=true]{border-color:var(--blue);box-shadow:inset 0 -3px 0 var(--blue);font-weight:700}",
    ".plt-bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}",
    ".plt-sum{margin:0 0 12px;padding:10px 12px;border:1px solid var(--border);border-radius:8px;background:var(--panel-bg);font-size:14.5px}",
    ".plt-sum dl{display:grid;grid-template-columns:max-content minmax(0,1fr);gap:2px 12px;margin:0}",
    ".plt-sum dt{font-weight:600;color:var(--mute)}",
    ".plt-sum dd{margin:0;font-family:var(--mono);font-size:13.5px;overflow-wrap:anywhere}",
    ".plt-lanes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}",
    ".plt-lane{border:1px dashed var(--line);border-radius:8px;padding:8px;min-width:0;background:var(--surface)}",
    ".plt-lane h4{margin:0 0 2px;font-size:15px}",
    ".plt-lane .plt-ln{margin:0 0 8px;font-size:12.5px;color:var(--mute)}",
    ".plt-ev{list-style:none;margin:0;padding:0}",
    ".plt-ev li{display:grid;grid-template-columns:2.1em minmax(0,1fr);gap:0 6px;margin:0 0 6px;padding:6px;border-radius:6px;border-left:4px solid var(--line);background:var(--panel-bg)}",
    ".plt-ev .plt-n{font:700 13px var(--mono);color:var(--mute);padding-top:1px}",
    ".plt-ev b{display:block;font:600 13px var(--mono);overflow-wrap:anywhere}",
    ".plt-ev span{display:block;font-size:13px;color:var(--mute);overflow-wrap:anywhere}",
    ".plt-ev i{font-style:normal;font:600 11.5px var(--mono);color:var(--blue)}",
    ".plt-ev li.plt-group-ev{border-left-color:var(--viz-c3)}",
    ".plt-ev li.plt-hook-ev{border-left-color:var(--viz-c5)}",
    ".plt-ev li.plt-step-ev{border-left-color:var(--gray-mid,var(--line))}",
    ".plt-ev li.plt-skip-ev{border-left-color:var(--line);background:transparent;opacity:.8}",
    ".plt-ev li.plt-skip-ev b{text-decoration:line-through}",
    ".plt-key{display:flex;flex-wrap:wrap;gap:4px 14px;margin:10px 0 0;font-size:13px;color:var(--mute)}",
    ".plt-key span:before{content:'';display:inline-block;width:10px;height:10px;margin-right:5px;border-radius:2px;vertical-align:-1px}",
    ".plt-key .k1:before{background:var(--viz-c3)}",
    ".plt-key .k2:before{background:var(--viz-c5)}",
    ".plt-key .k3:before{background:var(--gray-mid,var(--line))}",
    ".plt-key .k4:before{border:1px solid var(--line)}",
    ".plt-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}",
    "@media (max-width:760px){.plt-lanes{grid-template-columns:minmax(0,1fr)}.plt-sw{grid-template-columns:minmax(0,1fr)}}"
  ].join("\n");

  function injectStyle() {
    if (document.getElementById("plt-style")) return;
    var st = document.createElement("style");
    st.id = "plt-style";
    var P = ".viz-mount[data-viz='" + VIZ_ID + "'] ";
    st.textContent = CSS.split("\n").map(function (r) {
      if (r.charAt(0) === "@") {
        return r.replace(/\{\.plt-/g, "{" + P + ".plt-").replace(/\}\.plt-/g, "}" + P + ".plt-");
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

  /* ---- the model: rules R1-R16 turned into events ------------------------------------------ */
  function allowed(s, name) {
    if (s.plugins === "unset") return true;                       // R4: None = load all
    var list = s.plugins === "two" ? ["acme_hw", "acme_admin"] : ["acme_admin"];
    return list.indexOf(name) >= 0;                               // exact name match
  }

  function model(s) {
    var m = {};
    var cpu = s.target === "cpu";
    var ootActive = !cpu && s.oot === "yes" && allowed(s, "acme_hw");
    if (cpu) {
      m.platform = "CpuPlatform";
      m.platformWhy = "VLLM_TARGET_DEVICE=cpu: the platform plugin group is not loaded (R2)";
    } else if (ootActive) {
      m.platform = "acme_hw platform (out-of-tree)";
      m.platformWhy = "cuda and acme_hw detectors are active; one out-of-tree platform wins (R3)";
    } else {
      m.platform = "CudaPlatform";
      m.platformWhy = s.oot === "yes"
        ? "acme_hw is installed, but VLLM_PLUGINS does not name it: the cuda detector wins (R4)"
        : "only the built-in cuda detector is active";
    }
    var gen = allowed(s, "lora_filesystem_resolver");
    m.general = gen ? "lora_filesystem_resolver, lora_hf_hub_resolver run" : "no function runs: VLLM_PLUGINS names neither (R4)";
    m.uni = !cpu && s.tp === "1";
    m.workers = m.uni ? 0 : Number(s.tp);
    m.executor = m.uni ? "uni (worker inside the engine core process)" :
      "mp, " + s.tp + " worker process" + (s.tp === "1" ? "" : "es") + (cpu && s.tp === "1" ? " (CPU turns uni into mp)" : "");
    m.workerCls = cpu ? "vllm.v1.worker.cpu_worker.CPUWorker" :
      (ootActive ? "the class that acme_hw sets" : "vllm.v1.worker.gpu_worker.Worker");
    m.sched = cpu ? "Scheduler (CPU turns async scheduling off)" :
      (ootActive ? "AsyncScheduler, unless acme_hw turns async scheduling off" : "AsyncScheduler");
    m.genRuns = gen ? String(2 + m.workers) + " (one per process: front end, engine core" +
      (m.workers ? ", " + m.workers + " worker" + (m.workers > 1 ? "s" : "") : "") + ")" :
      "0: load_general_plugins() runs " + (2 + m.workers) + " times, but VLLM_PLUGINS filters out every general plugin";
    m.endpoint = s.plugins === "unset" ? "none: VLLM_PLUGINS is not set; a WARNING names acme_admin (R14)" :
      "acme_admin loads (named in VLLM_PLUGINS)";
    m.stat = allowed(s, "acme_metrics") ? "acme_metrics loads" : "acme_metrics skipped by VLLM_PLUGINS (R4)";
    m.cpu = cpu;
    m.ootActive = ootActive;
    return m;
  }

  function events(s, m) {
    var api = [], core = [], wrk = [];
    var n = 0;
    function add(list, kind, title, detail, rule) { n += 1; list.push({ n: n, kind: kind, title: title, detail: detail, rule: rule }); }
    function skip(list, title, detail, rule) { list.push({ n: null, kind: "skip", title: title, detail: detail, rule: rule }); }

    function detect(list, where) {
      if (m.cpu) {
        add(list, "hook", "detect platform -> CpuPlatform", where + "; no detector loop", "R1 R2");
        skip(list, "vllm.platform_plugins", "not loaded: VLLM_TARGET_DEVICE=cpu", "R2");
      } else {
        var oot = s.oot === "yes" ? (allowed(s, "acme_hw") ? "acme_hw loads" : "acme_hw filtered out") : "no plugin installed";
        add(list, "group", "vllm.platform_plugins", where + "; " + oot, "R1 R4");
        add(list, "hook", "detect platform -> " + (m.ootActive ? "acme_hw" : "CudaPlatform"), m.platformWhy, "R3");
      }
    }
    function general(list, where, guarded) {
      if (guarded) {
        skip(list, "load_general_plugins()", where + ": returns at once, flag already set", "R5");
      } else {
        add(list, "group", "vllm.general_plugins", where + ": " + m.general, "R4 R5");
      }
    }
    function workerInit(list, inCore) {
      general(list, "init_worker", inCore);
      add(list, "hook", "register_triton_kernel_overrides()", "before any kernel launch", "R13");
      add(list, "step", "worker_cls -> " + m.workerCls.split(".").pop(), m.workerCls, "R9 R13");
      if (s.tp === "2") add(list, "hook", "get_device_communicator_cls()", "init_device: the TP group has 2 ranks", "R13");
      else skip(list, "get_device_communicator_cls()", "not called: the group has 1 rank", "R13");
      add(list, "hook", "get_attn_backend_cls()", "load_model: the attention layers ask through the selector (cached)", "R13");
      add(list, "group", "vllm.logits_processors", (m.cpu ? "CPU model runner (a Model Runner V2 subclass)" : "Model Runner V2") + " load_model; ignores VLLM_PLUGINS", "R11");
      add(list, "hook", "update_block_size_for_backend()", inCore ? "uniproc_executor, after load_model" : "WorkerProc, after load_model", "R13");
    }

    // API server: CLI start, argument parsing, VllmConfig
    detect(api, "import of arg_utils.py");
    general(api, "add_cli_args", false);
    add(api, "hook", "pre_register_and_update(parser)", "after the parser has all flags", "R6");
    general(api, "EngineArgs.__post_init__", true);
    add(api, "hook", "pre_register_and_update(), validate_environ()", "create_engine_config", "R7");
    add(api, "hook", "apply_config_platform_defaults()", "VllmConfig.__post_init__", "R7");
    add(api, "hook", "check_and_update_config()", "sets worker_cls; executor = " + (m.uni ? "uni" : "mp"), "R7 R8 R9");
    add(api, "group", "vllm.stat_logger_plugins", "AsyncLLM.__init__: " + m.stat, "R10");
    add(api, "group", "vllm.logits_processors", "InputProcessor: request check; ignores VLLM_PLUGINS", "R11");
    add(api, "step", "start the engine core process", "spawn; then wait for READY", "R15");

    // Engine core
    detect(core, "import of vllm.v1.engine");
    general(core, "EngineCore.__init__", false);
    if (m.uni) {
      add(core, "step", "uni executor", "the worker runs in this process", "R8");
      workerInit(core, true);
    } else {
      add(core, "step", "mp executor: start " + m.workers + " worker" + (m.workers > 1 ? "s" : ""), "spawn; wait until each is ready", "R8 R15");
      detect(wrk, "import of vllm.v1.executor");
      workerInit(wrk, false);
    }
    add(core, "hook", "get_kv_cache_config_builder_cls()", "KV cache planning", "R12");
    add(core, "step", "scheduler class -> " + m.sched, "get_scheduler_cls()", "R12 R16");

    // API server again: the HTTP app
    if (s.plugins === "unset") skip(api, "vllm.endpoint_plugins", "build_app: " + m.endpoint, "R14");
    else add(api, "group", "vllm.endpoint_plugins", "build_app: " + m.endpoint, "R14");

    return { api: api, core: core, wrk: wrk };
  }

  /* ---- the widget ---------------------------------------------------------------------------- */
  function Timeline(mount) {
    this.mount = mount;
    this.state = {};
    for (var k in START) this.state[k] = START[k];
    this.build();
    this.render(false);
  }

  Timeline.prototype.build = function () {
    var self = this;
    this.mount.innerHTML = "";                       // removes the .viz-fallback text
    var root = el("div", "plt-root");

    var sw = el("div", "plt-sw");
    this.btns = {};
    SWITCHES.forEach(function (g) {
      var fs = el("fieldset", "plt-group");
      fs.appendChild(el("legend", null, g.label));
      var row = el("div", "plt-opts");
      self.btns[g.key] = g.opts.map(function (o) {
        var b = el("button", "plt-opt", o[1]);
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

    var bar = el("div", "plt-bar");
    var reset = el("button", "viz-btn", "Reset");
    reset.type = "button";
    reset.addEventListener("click", function () {
      for (var k in START) self.state[k] = START[k];
      self.render(true);
    });
    bar.appendChild(reset);
    bar.appendChild(el("span", "plt-note", "Simplified model of the start-up (rules R1-R16 below)."));
    root.appendChild(bar);

    this.sum = el("div", "plt-sum");
    root.appendChild(this.sum);

    var lanes = el("div", "plt-lanes");
    this.lanes = {};
    LANES.forEach(function (l) {
      var box = el("section", "plt-lane");
      box.setAttribute("aria-label", l.name);
      var h = el("h4", null, l.name);
      var ln = el("p", "plt-ln");
      var ol = el("ol", "plt-ev");
      box.appendChild(h);
      box.appendChild(ln);
      box.appendChild(ol);
      lanes.appendChild(box);
      self.lanes[l.key] = { box: box, ln: ln, ol: ol };
    });
    root.appendChild(lanes);

    var key = el("p", "plt-key");
    [["k1", "plugin group"], ["k2", "Platform hook"], ["k3", "start-up step"], ["k4", "skipped (crossed out)"]].forEach(function (k) {
      key.appendChild(el("span", k[0], k[1]));
    });
    root.appendChild(key);

    this.live = el("p", "plt-live");
    this.live.setAttribute("aria-live", "polite");
    root.appendChild(this.live);
    this.mount.appendChild(root);
  };

  Timeline.prototype.set = function (key, val) {
    this.state[key] = val;
    this.render(true);
  };

  Timeline.prototype.render = function (announce) {
    var s = this.state, m = model(s), ev = events(s, m), self = this;
    SWITCHES.forEach(function (g) {
      self.btns[g.key].forEach(function (b) {
        b.setAttribute("aria-pressed", b.getAttribute("data-val") === s[g.key] ? "true" : "false");
      });
    });

    this.sum.innerHTML = "";
    var dl = el("dl");
    [["Platform", m.platform], ["Executor", m.executor], ["worker_cls", m.workerCls],
     ["General plugin runs", m.genRuns],
     ["Endpoint plugins", m.endpoint]].forEach(function (r) {
      dl.appendChild(el("dt", null, r[0]));
      dl.appendChild(el("dd", null, r[1]));
    });
    this.sum.appendChild(dl);

    LANES.forEach(function (l) {
      var lane = self.lanes[l.key], list = ev[l.key];
      lane.ol.innerHTML = "";
      if (l.key === "wrk") {
        lane.ln.textContent = m.uni ? "No worker process: with TP = 1 the worker runs inside the engine core process."
          : "Repeats in each of the " + m.workers + " worker process" + (m.workers > 1 ? "es." : ".");
      } else if (l.key === "core") {
        lane.ln.textContent = m.uni ? "Also runs the worker (uni executor)." : "One engine core (no DP).";
      } else {
        lane.ln.textContent = "The vllm serve process.";
      }
      list.forEach(function (e) {
        var li = el("li", "plt-" + e.kind + "-ev");
        li.appendChild(el("span", "plt-n", e.n == null ? "-" : String(e.n)));
        var body = el("div");
        body.appendChild(el("b", null, e.title));
        var d = el("span", null, e.detail + " ");
        d.appendChild(el("i", null, e.rule));
        body.appendChild(d);
        li.appendChild(body);
        li.setAttribute("aria-label", (e.n == null ? "Skipped: " : "Step " + e.n + ": ") + e.title + ". " + e.detail + ". Rule " + e.rule);
        lane.ol.appendChild(li);
      });
    });

    if (announce) {
      this.live.textContent = "Platform " + m.platform + ". Executor " + m.executor + ". General plugin runs: " + m.genRuns + ". Endpoint plugins: " + m.endpoint + ".";
    }
  };

  function init() {
    var mounts = document.querySelectorAll('[data-viz="' + VIZ_ID + '"]');
    if (!mounts.length) return;
    injectStyle();
    for (var i = 0; i < mounts.length; i++) new Timeline(mounts[i]);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
