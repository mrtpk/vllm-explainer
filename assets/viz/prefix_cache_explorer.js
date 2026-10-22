/* prefix_cache_explorer.js - a small model of the vLLM prefix cache for concepts/prefix-caching (vLLM Explained).
 *
 * Mounts into <x-viz id="prefix-cache-explorer">. Vanilla JS, no network, no randomness: the result depends only
 * on the inputs. "Undo" and every settings change replay the list of actions from an empty pool.
 *
 * Rules copied from the code (pinned commit; the page lists them with citations):
 *   - only full blocks get a hash; hash = H(parent hash, block tokens); block 0 uses NONE_HASH as parent
 *     (kv_cache_utils.py:707-737, 966-1012)
 *   - lookup walks the hashes from block 0 and stops at the first miss (single_type_kv_cache_manager.py:791-799)
 *   - the hit is capped at num_tokens - 1 tokens: only (num_tokens - 1) // block_size blocks are checked
 *     (kv_cache_manager.py:289-300, single_type_kv_cache_manager.py:793)
 *   - admission: new blocks + hit blocks with ref_cnt 0 must fit in the free queue (single_type...:256-266,
 *     kv_cache_manager.py:561-579)
 *   - touch: ref_cnt += 1; a block with ref_cnt 0 leaves the free queue (block_pool.py:752-770)
 *   - new blocks: pop from the front; an old hash on a popped block is evicted (block_pool.py:666-700, 729-750)
 *   - cache: every block that is full now gets its hash (kv_cache_manager.py:612-616, block_pool.py:226-301)
 *   - free: reverse order; ref_cnt -= 1; at 0 unhashed -> front, hashed -> back (single_type...:597-598,
 *     block_pool.py:776-807)
 *   - block 0 is the null block and is never handed out (block_pool.py:184-188)
 * Stand-in hash: FNV-1a 32-bit over a string, NOT SHA-256. Left out: steps, chunked prefill, async scheduling,
 * KV groups, extra keys, preemption, the token budget.
 */
(function () {
  "use strict";

  var VIZ_ID = "prefix-cache-explorer";
  var LABELS = ["A", "B", "C"];
  var MAX_TOKENS = 64;
  var MAX_RUNNING = 8;
  var BS_OPTIONS = [4, 8, 16];
  var POOL_OPTIONS = [8, 12, 16, 24];
  var DEFAULT_CFG = { bs: 4, pool: 12 };
  var DEFAULT_PROMPTS = [
    "SYS you are a helpful bot . USER what is vLLM ?",
    "SYS you are a helpful bot . USER what is a KV cache ?",
    "SYS you are a helpful bot . USER what is vLLM ? BOT a serving engine . USER who made it ?"
  ];

  /* ---------------------------------------------------------------- the stand-in hash */

  function fnv(str) {
    var h = 0x811c9dc5;
    for (var i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return ("0000000" + h.toString(16)).slice(-8);
  }
  var NONE_HASH = fnv("vllm-none-hash");

  function blockHash(parent, tokens) {
    return fnv((parent || NONE_HASH) + "|" + tokens.join(","));
  }

  function parseTokens(text) {
    var items = String(text || "").trim().split(/\s+/).filter(function (s) { return s.length > 0; });
    if (items.length > MAX_TOKENS) items = items.slice(0, MAX_TOKENS);
    var numeric = items.length > 0 && items.every(function (s) { return /^\d+$/.test(s); });
    return {
      words: items,
      ids: items.map(function (s) { return numeric ? parseInt(s, 10) : parseInt(fnv("w:" + s), 16) % 100000; })
    };
  }

  /* ---------------------------------------------------------------- the model */

  function newState(cfg) {
    var blocks = [];
    for (var i = 0; i < cfg.pool; i++) blocks.push({ id: i, ref: 0, hash: null });
    var queue = [];
    for (var j = 1; j < cfg.pool; j++) queue.push(j);   // block 0 is popped once and becomes the null block
    return { cfg: cfg, blocks: blocks, queue: queue, map: {}, reqs: [], nextReq: 1,
             stats: { requests: 0, queries: 0, hits: 0 }, log: [], last: null };
  }

  function computeHashes(st, req) {
    var bs = st.cfg.bs;
    while ((req.hashes.length + 1) * bs <= req.tokens.length) {
      var start = req.hashes.length * bs;
      var parent = req.hashes.length ? req.hashes[req.hashes.length - 1] : null;
      req.hashes.push(blockHash(parent, req.tokens.slice(start, start + bs)));
    }
  }

  function mapAdd(st, hash, id) {
    if (!st.map[hash]) st.map[hash] = [];
    if (st.map[hash].indexOf(id) < 0) st.map[hash].push(id);
  }
  function mapRemove(st, hash, id) {
    var ids = st.map[hash];
    if (!ids) return;
    var k = ids.indexOf(id);
    if (k >= 0) ids.splice(k, 1);
    if (!ids.length) delete st.map[hash];
  }

  function popFront(st, n, L) {
    var out = st.queue.splice(0, n);
    if (out.length) L.push("get_new_blocks(" + n + ") pops block(s) " + out.join(", ") + " from the front of the free queue.");
    out.forEach(function (id) {
      var b = st.blocks[id];
      if (b.hash) {
        L.push("Block " + id + " still had hash #" + b.hash.slice(0, 4) +
               ": the pool removes it from the map (eviction).");
        mapRemove(st, b.hash, id);
        b.hash = null;
      }
      b.ref = 1;
    });
    return out;
  }

  function cacheFull(st, req) {
    var numFull = Math.floor(req.tokens.length / st.cfg.bs);
    for (var i = req.numCached; i < numFull; i++) {
      var b = st.blocks[req.blocks[i]];
      b.hash = req.hashes[i];
      mapAdd(st, b.hash, b.id);
    }
    if (numFull > req.numCached) req.numCached = numFull;
  }

  function send(st, label, tokens) {
    var L = st.log;
    var bs = st.cfg.bs;
    if (!tokens.length) { L.push("Prompt " + label + " is empty."); return; }
    if (st.reqs.length >= MAX_RUNNING) { L.push("At most " + MAX_RUNNING + " running requests in this model."); return; }
    var req = { id: "R" + st.nextReq, label: label, tokens: tokens.slice(), hashes: [], blocks: [], origin: [],
                numCached: 0, gen: 0, hitTokens: 0, walk: [] };
    computeHashes(st, req);

    // get_computed_blocks: cap at num_tokens - 1, walk from block 0, stop at the first miss
    var maxHit = tokens.length - 1;
    var nCheck = Math.min(Math.floor(maxHit / bs), req.hashes.length);
    var hits = [];
    var missed = false;
    for (var i = 0; i < req.hashes.length; i++) {
      if (i >= nCheck) { req.walk.push("over cap"); continue; }
      if (missed) { req.walk.push("not checked"); continue; }
      var ids = st.map[req.hashes[i]];
      if (ids && ids.length) { hits.push(ids[0]); req.walk.push("hit"); }
      else { req.walk.push("miss"); missed = true; }
    }
    var walkText = req.walk.map(function (w, k) { return "h" + k + " " + w; }).join(", ");
    L.push(label + ": " + tokens.length + " tokens, " + req.hashes.length + " full block(s). Lookup checks at most (" +
           tokens.length + " - 1) // " + bs + " = " + Math.floor(maxHit / bs) + " hash(es): " +
           (walkText || "none") + ".");

    // admission count: new blocks + hit blocks that are free now
    var need = Math.ceil(tokens.length / bs);
    var newN = need - hits.length;
    var evictable = hits.filter(function (id) { return st.blocks[id].ref === 0; }).length;
    if (newN + evictable > st.queue.length) {
      L.push("Not admitted: it needs " + newN + " new block(s) + " + evictable + " free hit block(s) = " +
             (newN + evictable) + ", but only " + st.queue.length + " block(s) are free. allocate_slots() returns " +
             "None. In vLLM the request would wait; this model drops it. Finish a request, then send again.");
      return;
    }
    st.nextReq += 1;

    // touch the hits, then pop new blocks
    hits.forEach(function (id) {
      var b = st.blocks[id];
      if (b.ref === 0) st.queue.splice(st.queue.indexOf(id), 1);
      b.ref += 1;
    });
    if (hits.length) L.push("touch(" + hits.join(", ") + "): ref_cnt + 1; free hit blocks leave the queue.");
    var fresh = popFront(st, newN, L);
    req.blocks = hits.concat(fresh);
    req.origin = hits.map(function () { return "hit"; }).concat(fresh.map(function () { return "new"; }));
    req.numCached = hits.length;
    req.hitTokens = hits.length * bs;
    cacheFull(st, req);
    st.reqs.push(req);
    st.stats.requests += 1;
    st.stats.queries += tokens.length;
    st.stats.hits += req.hitTokens;
    L.push(req.id + " runs. Hit " + req.hitTokens + " of " + tokens.length + " tokens; it computes " +
           (tokens.length - req.hitTokens) + ". Full blocks now have their hash.");
    st.last = req.id;
  }

  function findReq(st, rid) {
    for (var i = 0; i < st.reqs.length; i++) if (st.reqs[i].id === rid) return st.reqs[i];
    return null;
  }

  function addToken(st, rid) {
    var L = st.log;
    var req = findReq(st, rid);
    if (!req) return;
    var bs = st.cfg.bs;
    var need = Math.ceil((req.tokens.length + 1) / bs);
    if (need > req.blocks.length && !st.queue.length) {
      L.push(rid + " needs a new block, but none is free. The real scheduler would preempt a request here " +
             "(not in this model).");
      return;
    }
    req.gen += 1;
    var reqNum = parseInt(rid.slice(1), 10);
    req.tokens.push(90000 + reqNum * 100 + req.gen);
    computeHashes(st, req);
    if (need > req.blocks.length) {
      var fresh = popFront(st, 1, L);
      req.blocks = req.blocks.concat(fresh);
      req.origin.push("new");
      L.push(rid + " gets token " + req.tokens.length + " in the new block " + fresh[0] + ".");
    } else {
      L.push(rid + " gets token " + req.tokens.length + ".");
    }
    var before = req.numCached;
    cacheFull(st, req);
    if (req.numCached > before) L.push("Block " + req.blocks[req.numCached - 1] + " is now full and gets hash #" +
                                       req.hashes[req.numCached - 1].slice(0, 4) + ".");
    st.last = rid;
  }

  function finish(st, rid) {
    var L = st.log;
    var req = findReq(st, rid);
    if (!req) return;
    var first = [];
    var last = [];
    var kept = [];
    req.blocks.slice().reverse().forEach(function (id) {
      var b = st.blocks[id];
      b.ref -= 1;
      if (b.ref === 0) {
        if (b.hash) last.push(id); else first.push(id);
      } else kept.push(id);
    });
    st.queue = first.concat(st.queue).concat(last);
    st.reqs.splice(st.reqs.indexOf(req), 1);
    L.push(rid + " finishes. free_blocks(reversed) = [" + req.blocks.slice().reverse().join(", ") + "]." +
           (first.length ? " No hash -> front: " + first.join(", ") + "." : "") +
           (last.length ? " Hash -> back: " + last.join(", ") + "." : "") +
           (kept.length ? " Still used by another request: " + kept.join(", ") + "." : ""));
    st.last = null;
  }

  function run(cfg, actions) {
    var st = newState(cfg);
    actions.forEach(function (a, k) {
      st.log = [];
      if (a.t === "send") send(st, a.label, a.tokens);
      else if (a.t === "tok") addToken(st, a.rid);
      else if (a.t === "fin") finish(st, a.rid);
      st.lastAction = k;
    });
    return st;
  }

  /* ---------------------------------------------------------------- the view */

  var CSS = [
    ".pcx-root{font-family:var(--sans);color:var(--ink)}",
    ".pcx-row{display:flex;flex-wrap:wrap;gap:8px 12px;align-items:flex-end;margin:0 0 12px}",
    ".pcx-row label{display:flex;flex-direction:column;gap:2px;font-size:13.5px;color:var(--mute)}",
    ".pcx-row select{min-height:40px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);font:14px var(--mono)}",
    ".pcx-prompt{display:grid;grid-template-columns:auto 1fr auto;gap:6px 10px;align-items:center;margin:0 0 8px}",
    ".pcx-prompt input{min-height:40px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface);color:var(--ink);font:14px var(--mono);width:100%;box-sizing:border-box;min-width:0}",
    ".pcx-plab{font:700 15px var(--mono)}",
    ".pcx-pinfo{grid-column:2 / 4;font-size:12.5px;color:var(--mute);margin-top:-4px}",
    ".pcx-sep{border:0;border-top:1px solid var(--border);margin:6px 0 12px}",
    ".pcx-head{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:baseline;margin:0 0 6px;font-size:14px}",
    ".pcx-head b{font:700 16px var(--sans)}",
    ".pcx-lab{font-size:13px;color:var(--mute);margin:12px 0 4px}",
    ".pcx-reqs{display:flex;flex-direction:column;gap:8px}",
    ".pcx-req{border:1px solid var(--border);border-radius:8px;padding:8px 10px;background:var(--surface)}",
    ".pcx-req.pcx-new{border-color:var(--viz-c1)}",
    ".pcx-rhead{display:flex;flex-wrap:wrap;gap:4px 12px;align-items:center;font-size:14px;margin:0 0 6px}",
    ".pcx-rhead b{font:700 14px var(--mono)}",
    ".pcx-rbtns{display:flex;gap:6px;margin-left:auto}",
    ".pcx-chips{display:flex;flex-wrap:wrap;gap:4px;margin:0;padding:0;list-style:none}",
    ".pcx-chip{min-width:62px;padding:3px 6px;box-sizing:border-box;border:1px solid var(--line);border-radius:5px;background:var(--surface);font:600 11.5px/1.35 var(--mono);text-align:center;color:var(--mute)}",
    ".pcx-chip span{display:block;font-weight:400}",
    ".pcx-hit{border-color:var(--viz-c3);color:var(--viz-c3);box-shadow:inset 0 -3px 0 var(--viz-c3)}",
    ".pcx-newb{border-color:var(--viz-c1);color:var(--viz-c1);box-shadow:inset 0 -3px 0 var(--viz-c1)}",
    ".pcx-used{border-color:var(--viz-c1);color:var(--viz-c1)}",
    ".pcx-cached{border-color:var(--viz-c3);color:var(--viz-c3);border-style:dashed}",
    ".pcx-null{border-style:dotted}",
    ".pcx-walk{font:12.5px var(--mono);color:var(--mute);margin:6px 0 0;overflow-wrap:anywhere}",
    ".pcx-q{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:0;padding:0;list-style:none}",
    ".pcx-end{font-size:12.5px;color:var(--mute)}",
    ".pcx-log{margin:10px 0 0;padding:8px 10px 8px 26px;border:1px solid var(--border);border-radius:6px;background:var(--panel-bg);font-size:14px}",
    ".pcx-log li{margin:2px 0;overflow-wrap:anywhere}",
    ".pcx-live{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}",
    ".pcx-empty{font-size:14px;color:var(--mute);margin:0}",
    "@media (max-width:560px){.pcx-prompt{grid-template-columns:auto 1fr}.pcx-prompt .viz-btn{grid-column:1 / 3}.pcx-pinfo{grid-column:1 / 3;margin-top:0}.pcx-chip{min-width:54px}.pcx-rbtns{margin-left:0}}"
  ].join("\n");

  function injectStyle() {
    if (document.getElementById("pcx-style")) return;
    var st = document.createElement("style");
    st.id = "pcx-style";
    var P = ".viz-mount[data-viz='" + VIZ_ID + "'] ";
    st.textContent = CSS.split("\n").map(function (r) {
      if (r.charAt(0) === "@") {
        return r.replace(/\{\.pcx-/g, "{" + P + ".pcx-").replace(/\}\.pcx-/g, "}" + P + ".pcx-");
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
  function button(text, cls) {
    var b = el("button", "viz-btn" + (cls ? " " + cls : ""), text);
    b.type = "button";
    return b;
  }
  function select(labelText, options, value) {
    var lab = el("label");
    lab.appendChild(document.createTextNode(labelText));
    var s = el("select");
    options.forEach(function (o) {
      var op = el("option", null, String(o));
      op.value = String(o);
      if (o === value) op.selected = true;
      s.appendChild(op);
    });
    lab.appendChild(s);
    return { label: lab, input: s };
  }
  function short(h) { return h ? "#" + h.slice(0, 4) : "no hash"; }

  var nextUid = 1;
  function Explorer(mount) {
    this.mount = mount;
    this.uid = nextUid++;
    this.reset();
    this.build();
    this.render(false);
  }

  Explorer.prototype.reset = function () {
    this.cfg = { bs: DEFAULT_CFG.bs, pool: DEFAULT_CFG.pool };
    this.actions = [];
  };

  Explorer.prototype.build = function () {
    var self = this;
    this.mount.innerHTML = "";
    var root = el("div", "pcx-root");

    var row = el("div", "pcx-row");
    this.inBs = select("block_size (tokens)", BS_OPTIONS, this.cfg.bs);
    this.inPool = select("Blocks in the pool (with null block 0)", POOL_OPTIONS, this.cfg.pool);
    row.appendChild(this.inBs.label);
    row.appendChild(this.inPool.label);
    this.undoBtn = button("Undo");
    this.resetBtn = button("Reset");
    row.appendChild(this.undoBtn);
    row.appendChild(this.resetBtn);
    root.appendChild(row);
    [this.inBs, this.inPool].forEach(function (f) {
      f.input.addEventListener("change", function () { self.readCfg(); });
    });
    this.undoBtn.addEventListener("click", function () {
      if (!self.actions.length) return;
      self.actions.pop();
      self.render(true, "Last action undone.");
    });
    this.resetBtn.addEventListener("click", function () {
      self.reset();
      self.inBs.input.value = String(self.cfg.bs);
      self.inPool.input.value = String(self.cfg.pool);
      self.inputs.forEach(function (inp, i) { inp.value = DEFAULT_PROMPTS[i]; self.updateInfo(i); });
      self.render(true, "Reset: empty pool, default prompts.");
    });

    root.appendChild(el("p", "pcx-lab", "Prompts: one token per word, split on spaces. A list of numbers is used as token ids."));
    this.inputs = [];
    this.infos = [];
    LABELS.forEach(function (lab, i) {
      var pr = el("div", "pcx-prompt");
      var id = "pcx-" + self.uid + "-" + i;
      var l = el("label", "pcx-plab", lab);
      l.setAttribute("for", id);
      var inp = el("input");
      inp.type = "text";
      inp.id = id;
      inp.value = DEFAULT_PROMPTS[i];
      inp.setAttribute("spellcheck", "false");
      inp.setAttribute("autocomplete", "off");
      var b = button("Send " + lab, "viz-btn-primary");
      var info = el("div", "pcx-pinfo");
      pr.appendChild(l);
      pr.appendChild(inp);
      pr.appendChild(b);
      pr.appendChild(info);
      root.appendChild(pr);
      self.inputs.push(inp);
      self.infos.push(info);
      inp.addEventListener("input", function () { self.updateInfo(i); });
      b.addEventListener("click", function () { self.sendPrompt(i); });
      self.updateInfo(i);
    });

    root.appendChild(el("hr", "pcx-sep"));
    this.out = el("div", "pcx-out");
    root.appendChild(this.out);
    this.live = el("p", "pcx-live");
    this.live.setAttribute("aria-live", "polite");
    root.appendChild(this.live);
    this.mount.appendChild(root);
  };

  Explorer.prototype.updateInfo = function (i) {
    var p = parseTokens(this.inputs[i].value);
    var bs = this.cfg.bs;
    this.infos[i].textContent = p.ids.length + " tokens = " + Math.floor(p.ids.length / bs) + " full block(s) of " +
      bs + (p.ids.length % bs ? " + " + (p.ids.length % bs) + " token(s) in a block that is not full" : "") +
      (p.words.length >= MAX_TOKENS ? " (cut at " + MAX_TOKENS + ")" : "") + ".";
  };

  Explorer.prototype.readCfg = function () {
    this.cfg.bs = parseInt(this.inBs.input.value, 10);
    this.cfg.pool = parseInt(this.inPool.input.value, 10);
    var self = this;
    this.inputs.forEach(function (_, i) { self.updateInfo(i); });
    this.render(true, "Settings changed: the actions are replayed on an empty pool.");
  };

  Explorer.prototype.sendPrompt = function (i) {
    var p = parseTokens(this.inputs[i].value);
    this.actions.push({ t: "send", label: LABELS[i], tokens: p.ids });
    this.render(true);
  };

  Explorer.prototype.render = function (announce, extra) {
    var self = this;
    var st = run(this.cfg, this.actions);
    var cfg = this.cfg;
    this.undoBtn.disabled = this.actions.length === 0;
    var o = this.out;
    o.innerHTML = "";

    var usable = cfg.pool - 1;
    var cachedCount = Object.keys(st.map).length;
    var head = el("div", "pcx-head");
    head.appendChild(el("b", null, this.actions.length ? "After " + this.actions.length + " action(s)" : "Empty pool"));
    head.appendChild(el("span", null, "free: " + st.queue.length + " / " + usable));
    head.appendChild(el("span", null, "hashes in the map: " + cachedCount));
    head.appendChild(el("span", null, "prefix_cache_queries: " + st.stats.queries));
    head.appendChild(el("span", null, "prefix_cache_hits: " + st.stats.hits +
      (st.stats.queries ? " (" + Math.round(100 * st.stats.hits / st.stats.queries) + " %)" : "")));
    o.appendChild(head);

    // running requests
    o.appendChild(el("p", "pcx-lab", "Running requests and their block tables"));
    var reqs = el("div", "pcx-reqs");
    if (!st.reqs.length) reqs.appendChild(el("p", "pcx-empty", "No running request. Press Send A, B or C."));
    st.reqs.forEach(function (r) {
      var card = el("section", "pcx-req" + (r.id === st.last ? " pcx-new" : ""));
      card.setAttribute("aria-label", "Request " + r.id);
      var rh = el("div", "pcx-rhead");
      rh.appendChild(el("b", null, r.id + " (prompt " + r.label + ")"));
      rh.appendChild(el("span", null, r.tokens.length + " tokens, hit " + r.hitTokens));
      var btns = el("div", "pcx-rbtns");
      var tb = button("+1 token");
      var fb = button("Finish");
      tb.setAttribute("aria-label", "Add one generated token to " + r.id);
      fb.setAttribute("aria-label", "Finish " + r.id + " and free its blocks");
      tb.addEventListener("click", function () {
        self.actions.push({ t: "tok", rid: r.id });
        self.focusAfter = { rid: r.id };
        self.render(true);
      });
      fb.addEventListener("click", function () {
        self.actions.push({ t: "fin", rid: r.id });
        self.focusAfter = { undo: true };   // the card is gone: keep keyboard focus inside the visual
        self.render(true);
      });
      if (self.focusAfter && self.focusAfter.rid === r.id) { self.focusAfter = null; setTimeout(function () { tb.focus(); }, 0); }
      btns.appendChild(tb);
      btns.appendChild(fb);
      rh.appendChild(btns);
      card.appendChild(rh);
      var ul = el("ul", "pcx-chips");
      r.blocks.forEach(function (id, k) {
        var b = st.blocks[id];
        var li = el("li", "pcx-chip " + (r.origin[k] === "hit" ? "pcx-hit" : "pcx-newb"));
        li.appendChild(document.createTextNode("blk " + id));
        li.appendChild(el("span", null, k < r.numCached ? short(r.hashes[k]) : "not full"));
        li.appendChild(el("span", null, (r.origin[k] === "hit" ? "hit" : "new") + " · ref " + b.ref));
        li.setAttribute("aria-label", "position " + k + ": block " + id + ", " +
          (k < r.numCached ? "hash " + r.hashes[k].slice(0, 4) : "not full") + ", " + r.origin[k] + ", ref_cnt " + b.ref);
        ul.appendChild(li);
      });
      card.appendChild(ul);
      card.appendChild(el("p", "pcx-walk", "lookup at admission: " +
        (r.walk.length ? r.walk.map(function (w, k) { return "h" + k + "=" + w; }).join("  ") : "no full block")));
      reqs.appendChild(card);
    });
    o.appendChild(reqs);

    // the free queue
    o.appendChild(el("p", "pcx-lab", "Free block queue, front (evicted first) to back"));
    var q = el("ul", "pcx-q");
    q.setAttribute("aria-label", "Free block queue from front to back");
    q.appendChild(el("li", "pcx-end", "front →"));
    if (!st.queue.length) q.appendChild(el("li", "pcx-end", "(empty)"));
    st.queue.forEach(function (id) {
      var b = st.blocks[id];
      var li = el("li", "pcx-chip" + (b.hash ? " pcx-cached" : ""));
      li.appendChild(document.createTextNode("blk " + id));
      li.appendChild(el("span", null, short(b.hash)));
      q.appendChild(li);
    });
    q.appendChild(el("li", "pcx-end", "← back"));
    o.appendChild(q);

    // the pool
    o.appendChild(el("p", "pcx-lab", "All blocks of the pool"));
    var pool = el("ul", "pcx-chips");
    pool.setAttribute("aria-label", "All blocks");
    st.blocks.forEach(function (b) {
      var cls = "pcx-chip";
      var line2;
      if (b.id === 0) { cls += " pcx-null"; line2 = "null"; }
      else if (b.ref > 0) { cls += " pcx-used"; line2 = "ref " + b.ref; }
      else if (b.hash) { cls += " pcx-cached"; line2 = "free, cached"; }
      else line2 = "free";
      var li = el("li", cls);
      li.appendChild(document.createTextNode("blk " + b.id));
      li.appendChild(el("span", null, line2));
      li.appendChild(el("span", null, b.id === 0 ? "-" : short(b.hash)));
      pool.appendChild(li);
    });
    o.appendChild(pool);

    // log of the last action
    var log = el("ol", "pcx-log");
    log.setAttribute("aria-label", "What the last action did");
    var lines = this.actions.length ? st.log : [
      "Press Send A. Then Send B: B shares the first blocks of A. Send A again: the last block is computed again, " +
      "because the hit stops at num_tokens - 1."];
    lines.forEach(function (s) { log.appendChild(el("li", null, s)); });
    o.appendChild(log);

    if (this.focusAfter && this.focusAfter.undo) { this.focusAfter = null; this.undoBtn.focus(); }
    this.focusAfter = null;
    if (announce) {
      this.live.textContent = (extra ? extra + " " : "") + (this.actions.length ? st.log.join(" ") : "") +
        " " + st.queue.length + " free blocks.";
    }
  };

  function init() {
    var mounts = document.querySelectorAll('[data-viz="' + VIZ_ID + '"]');
    if (!mounts.length) return;
    injectStyle();
    for (var i = 0; i < mounts.length; i++) new Explorer(mounts[i]);
  }

  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
  if (typeof module !== "undefined") module.exports = { run: run, parseTokens: parseTokens };   // node check only
})();
