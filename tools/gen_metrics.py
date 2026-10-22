#!/usr/bin/env python3
"""Generate the Prometheus metric reference (src/reference/metrics.html) from the vllm source.

Owner: the reference-table writer. Python 3 standard library only; vllm is never imported.

Where vLLM declares metrics (found with `git grep` at the pinned commit):
  - vllm/v1/metrics/loggers.py: PrometheusStatLogger.__init__ creates the core engine metrics with
    `self._gauge_cls(name="vllm:...", documentation=..., labelnames=..., [buckets=...])`
    (_gauge_cls / _counter_cls / _histogram_cls are prometheus_client classes, or Ray wrappers in
    vllm/v1/metrics/ray_wrappers.py). It also builds helper objects that create more metrics the
    same way: SpecDecodingProm (vllm/v1/spec_decode/metrics.py), PerfMetricsProm
    (vllm/v1/metrics/perf.py), KVConnectorProm and ECConnectorProm (the connectors' own
    *Prom classes, for example nixl/stats.py, hf3fs_connector.py, offloading/metrics.py).
  - Offloading metrics are declared as data: dict entries `"vllm:...": OffloadingCounterMetadata(...)`
    (also Gauge/Histogram) or `metrics[KEY] = Offloading...Metadata(...)`; offloading/metrics.py
    turns each entry into a metric (_create_metric).
  - A few API-server modules call prometheus_client.Gauge/Counter/Histogram directly
    (vllm/entrypoints/rl/online/metrics.py, serve/dev/sleep/metrics.py,
    speech_to_text/realtime/metrics.py, vllm/parser/metrics.py).

What the script does:
  1. Parses each file that contains "vllm:" or an Offloading*Metadata( call with `ast`.
  2. Finds (a) calls of _gauge_cls/_counter_cls/_histogram_cls/Gauge/Counter/Histogram/Summary/Info
     and (b) Offloading{Counter,Gauge,Histogram}Metadata(...) calls (name = the dict key or the
     subscript key). It resolves the name to a string through literals, f-strings, local and
     module constants, class attributes (`_TransferMetricName.LOAD_BYTES`), imports, small
     helper functions that return an f-string (`_vllm_metric_name`), and loops over literal lists
     of tuples (the spec-decode counters). Only names that start with "vllm:" are kept.
  3. For each metric: type, documentation (first paragraph), label names, histogram buckets
     (literal, a module constant, or the bucket family of histogram_buckets() in
     vllm/v1/metrics/buckets.py), the `if` conditions around the call, the line of the name
     string and the line of the constructor call.
  4. Checks every cited line, and checks how show_hidden_metrics is used (see below).
  5. Writes notes/generated/metrics.json and, when src/reference/metrics.html has the markers,
     the HTML between <!-- BEGIN GENERATED metrics --> and <!-- END GENERATED metrics -->.

Label names: the stat loggers pass labelnames=["model_name", "engine"]
(vllm/v1/metrics/loggers.py:499) into every helper class. When a call uses that parameter
(`labelnames`, `self._labelnames`), the table shows "model_name, engine" and marks it.
PrometheusStatLogger itself uses a local list with the same two names; those rows are not marked.

Hidden / deprecated metrics: ObservabilityConfig.show_hidden_metrics (vllm/config/observability.py:32)
is True only when --show-hidden-metrics-for-version equals the previous minor version
(vllm/version.py:15; any value in a dev build). PrometheusStatLogger stores it
(loggers.py:494); the script lists every other read of `show_hidden_metrics` in vllm/. A metric is
marked "deprecated" when its name expression or documentation says so (for example
_DEPRECATED_TOTAL_BYTES in offloading/metrics.py).

Known limits:
  - Metrics whose name is built at run time from data (for example per-spec offloading metric
    definitions in modules this scan cannot resolve) are listed under "unresolved" in the report.
  - HTTP metrics come from prometheus_fastapi_instrumentator
    (vllm/entrypoints/serve/instrumentator/metrics.py:64) and process/python metrics from
    prometheus_client itself; they are not declared in vLLM code and are not in the table.
  - "Exposed as": prometheus_client adds `_total` to counter names (it strips a trailing `_total`
    first), and `_bucket`, `_sum`, `_count` to histograms. `_created` series are not listed.

Usage:
  python3 tools/gen_metrics.py            # write JSON (and the page region if the markers exist)
  python3 tools/gen_metrics.py --check    # exit 1 if the JSON or the page region is out of date
  python3 tools/gen_metrics.py --report   # also print a summary
"""
from __future__ import annotations

import ast
import itertools
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vllm_ast as va  # noqa: E402

NAME = "metrics"
PAGE_ID = "reference/metrics"
DOC_LIMIT = 300
BASE_LABELS = ["model_name", "engine"]
BUCKETS_FILE = "vllm/v1/metrics/buckets.py"

CTORS = {"_gauge_cls": "gauge", "_counter_cls": "counter", "_histogram_cls": "histogram",
         "Gauge": "gauge", "Counter": "counter", "Histogram": "histogram", "Summary": "summary", "Info": "info"}
META = {"OffloadingCounterMetadata": "counter", "OffloadingGaugeMetadata": "gauge",
        "OffloadingHistogramMetadata": "histogram"}
SKIP_FILES = {"vllm/v1/metrics/ray_wrappers.py"}

# Group by path prefix (first match wins).
GROUPS: list[tuple[str, str]] = [
    ("vllm/v1/metrics/loggers.py", "Engine core (PrometheusStatLogger)"),
    ("vllm/v1/metrics/perf.py", "Performance estimates (MFU)"),
    ("vllm/v1/spec_decode/", "Speculative decoding and diffusion"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/offloading/", "KV offloading connector"),
    ("vllm/v1/kv_offload/", "KV offloading connector"),
    ("vllm/v1/simple_kv_offload/", "Simple KV offload"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/nixl/", "KV connector: NIXL"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/mooncake/", "KV connector: Mooncake store"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/", "KV connector: 3FS"),
    ("vllm/distributed/kv_transfer/kv_connector/v1/hisparse/", "KV connector: HiSparse"),
    ("vllm/distributed/", "Other KV / EC connectors"),
    ("vllm/entrypoints/", "API server"),
    ("vllm/parser/", "API server"),
]


def group_of(path: str) -> str:
    for pre, g in GROUPS:
        if path.startswith(pre):
            return g
    return "Other"


def candidate_files() -> list[str]:
    files = va.git_grep_files(["-E", r"vllm:|Offloading(Counter|Gauge|Histogram)Metadata\("], ["vllm/"])
    return [f for f in files if f.endswith(".py") and f not in SKIP_FILES]


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------
_CTX: dict[str, "Ctx"] = {}


def ctx_of(mod: va.Module) -> "Ctx":
    if mod.rel not in _CTX:
        _CTX[mod.rel] = Ctx(mod)
    return _CTX[mod.rel]


class Ctx:
    def __init__(self, mod: va.Module):
        self.mod = mod
        self.parents: dict[ast.AST, ast.AST] = {}
        for p in ast.walk(mod.tree):
            for c in ast.iter_child_nodes(p):
                self.parents[c] = p

    def enclosing_func(self, node):
        p = self.parents.get(node)
        while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            p = self.parents.get(p)
        return p

    def local_assigns(self, func, name: str, before: int | None = None) -> list[ast.AST]:
        """Values assigned to `name` in `func`. With `before`: only the last assignment above that line
        (straight-line code: `buckets = [...]` is reassigned between metrics in nixl/stats.py)."""
        out = []
        if func is None:
            return out
        for n in ast.walk(func):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        out.append((n.lineno, n.value))
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name and n.value:
                out.append((n.lineno, n.value))
        out.sort(key=lambda x: x[0])
        if before is not None:
            prev = [v for ln, v in out if ln < before]
            return prev[-1:] if prev else []
        return [v for _, v in out]

    def conditions(self, node) -> list[str]:
        conds = []
        child, p = node, self.parents.get(node)
        while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            if isinstance(p, ast.If):
                t = " ".join(self.mod.src(p.test).split())
                if child in p.body:
                    conds.append(t)
                elif child in p.orelse:
                    conds.append(f"not ({t})")
            child, p = p, self.parents.get(p)
        return conds


def deref(ctx: Ctx, mod: va.Module, node: ast.AST, scope, binds: dict, depth: int = 0):
    """Follow names, dict lookups and `.attr` on calls to the expression that holds the value.
    Returns a list of (module, node)."""
    if depth > 10:
        return []
    if isinstance(node, ast.Name):
        if node.id in binds:
            return [(binds[node.id][0], binds[node.id][1])]
        loc = ctx.local_assigns(scope, node.id, getattr(node, "lineno", 10**9)) if mod is ctx.mod else []
        if loc:
            return [r for v in loc for r in deref(ctx, mod, v, scope, binds, depth + 1)]
        r = va.resolve(mod, node.id)
        if r and not isinstance(r[1], (ast.ClassDef, ast.FunctionDef, ast.Module)):
            m2 = r[0]
            c2 = ctx if m2 is ctx.mod else ctx_of(m2)
            return deref(c2, m2, r[1], None, {}, depth + 1)
        return [(mod, node)]
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id not in binds and node.value.id != "self":
            r = va.find_class(node.value.id, mod)
            if r:
                cm, cls = r
                for st in cls.body:
                    tgt = st.targets if isinstance(st, ast.Assign) else ([st.target] if isinstance(st, ast.AnnAssign) else [])
                    if any(isinstance(t, ast.Name) and t.id == node.attr for t in tgt) and st.value is not None:
                        c2 = ctx if cm is ctx.mod else ctx_of(cm)
                        return deref(c2, cm, st.value, None, {}, depth + 1)
        if isinstance(node.value, ast.Name) and node.value.id == "self" and mod is ctx.mod:
            cls = ctx.parents.get(scope) if scope is not None else None
            found = []
            if isinstance(cls, ast.ClassDef):
                for n in ast.walk(cls):
                    if isinstance(n, (ast.Assign, ast.AnnAssign)):
                        tg = n.targets if isinstance(n, ast.Assign) else [n.target]
                        if (any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self"
                                and t.attr == node.attr for t in tg) and n.value is not None
                                and isinstance(n.value, (ast.Constant, ast.List, ast.Tuple))):
                            found.append((mod, n.value))
            if found:
                return found
        # X.documentation where X derefs to a Call with that keyword
        out = []
        for m2, base in deref(ctx, mod, node.value, scope, binds, depth + 1):
            if isinstance(base, ast.Call):
                for k in base.keywords:
                    if k.arg == node.attr:
                        c2 = ctx if m2 is ctx.mod else ctx_of(m2)
                        out += deref(c2, m2, k.value, None, {}, depth + 1)
        return out or [(mod, node)]
    if isinstance(node, ast.Subscript):
        keys = resolve_str(ctx, mod, node.slice, scope, binds, depth + 1)
        out = []
        for m2, base in deref(ctx, mod, node.value, scope, binds, depth + 1):
            if isinstance(base, ast.Dict):
                c2 = ctx if m2 is ctx.mod else ctx_of(m2)
                for k, v in zip(base.keys, base.values):
                    if k is None:
                        continue
                    ks = resolve_str(c2, m2, k, None, {}, depth + 1)
                    if any(a == b for a, _, _ in ks for b, _, _ in keys):
                        out += deref(c2, m2, v, None, {}, depth + 1)
        return out or [(mod, node)]
    return [(mod, node)]


def resolve_str(ctx: Ctx, mod: va.Module, node: ast.AST, scope, binds: dict, depth: int = 0) -> list[tuple[str, str, int]]:
    """Possible string values of `node`: (value, file of the literal, line of the literal)."""
    if depth > 12 or node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [(node.value, mod.rel, node.lineno)]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for v in node.values:
            if isinstance(v, ast.Constant):
                parts.append([(str(v.value), None, None)])
            elif isinstance(v, ast.FormattedValue):
                r = resolve_str(ctx, mod, v.value, scope, binds, depth + 1)
                if not r:
                    return []
                parts.append(r)
        out = []
        for combo in itertools.product(*parts):
            lit_line = next(((f, ln) for _, f, ln in combo if ln), (mod.rel, node.lineno))
            out.append(("".join(s for s, _, _ in combo), lit_line[0], lit_line[1]))
        return out
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        r = va.resolve(mod, node.func.id)
        if r and isinstance(r[1], ast.FunctionDef) and node.args:
            fn = r[1]
            ret = fn.body[-1]
            if isinstance(ret, ast.Return) and isinstance(ret.value, ast.JoinedStr) and len(fn.args.args) == 1:
                vals = resolve_str(ctx, mod, node.args[0], scope, binds, depth + 1)
                p = fn.args.args[0].arg
                fm = r[0]
                fctx = ctx if fm is ctx.mod else ctx_of(fm)
                out = []
                for s, f, ln in vals:
                    sub = resolve_str(fctx, fm, ret.value, None, {p: (mod, ast.Constant(value=s))}, depth + 1)
                    out += [(x, f, ln) for x, _, _ in sub]
                return out
        return []
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        out = []
        for m2, v in deref(ctx, mod, node, scope, binds, depth + 1):
            if v is node:
                continue
            c2 = ctx if m2 is ctx.mod else ctx_of(m2)
            out += resolve_str(c2, m2, v, None, {}, depth + 1)
        return out
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = resolve_str(ctx, mod, node.left, scope, binds, depth + 1)
        b = resolve_str(ctx, mod, node.right, scope, binds, depth + 1)
        return [(x + y, f, ln) for x, f, ln in a for y, _, _ in b]
    return []


def loop_bindings(ctx: Ctx, call: ast.Call, name_node: ast.AST) -> list[dict]:
    """If the name is a loop / comprehension target over a literal list, one binding per element."""
    if not isinstance(name_node, ast.Name):
        return [{}]
    p = ctx.parents.get(call)
    while p is not None and not isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
        gens = []
        if isinstance(p, ast.For):
            gens = [(p.target, p.iter)]
        elif isinstance(p, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            gens = [(g.target, g.iter) for g in p.generators]
        for target, it in gens:
            names = [target] if isinstance(target, ast.Name) else (list(target.elts) if isinstance(target, ast.Tuple) else [])
            ids = [n.id if isinstance(n, ast.Name) else None for n in names]
            if name_node.id in ids:
                scope = ctx.enclosing_func(call)
                seqs = []
                if isinstance(it, (ast.List, ast.Tuple)):
                    seqs = [it]
                elif isinstance(it, ast.Name):
                    seqs = [v for v in ctx.local_assigns(scope, it.id) if isinstance(v, (ast.List, ast.Tuple))]
                out = []
                for seq in seqs:
                    for el in seq.elts:
                        if len(ids) == 1:
                            out.append({ids[0]: (ctx.mod, el)})
                        elif isinstance(el, ast.Tuple) and len(el.elts) == len(ids):
                            out.append({i: (ctx.mod, e) for i, e in zip(ids, el.elts) if i})
                return out or [{}]
        p = ctx.parents.get(p)
    return [{}]


def resolve_labels(ctx: Ctx, mod: va.Module, node, scope, binds, depth=0) -> tuple[list[str], bool, str]:
    """(labels, uses the passed-in base labels, unresolved source text)."""
    if node is None or depth > 8:
        return [], False, ""
    if isinstance(node, (ast.List, ast.Tuple)):
        labels, base, dyn = [], False, ""
        for e in node.elts:
            if isinstance(e, ast.Starred):
                l2, b2, d2 = resolve_labels(ctx, mod, e.value, scope, binds, depth + 1)
                labels += l2
                base |= b2
                dyn = dyn or d2
                continue
            s = resolve_str(ctx, mod, e, scope, binds, depth + 1)
            if s:
                labels.append(s[0][0])
            else:
                dyn = dyn or mod.src(e)
        return labels, base, dyn
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a = resolve_labels(ctx, mod, node.left, scope, binds, depth + 1)
        b = resolve_labels(ctx, mod, node.right, scope, binds, depth + 1)
        return a[0] + b[0], a[1] or b[1], a[2] or b[2]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("list", "tuple") and node.args:
        return resolve_labels(ctx, mod, node.args[0], scope, binds, depth + 1)
    if isinstance(node, ast.Name) and node.id == "labelnames" and node.id not in binds:
        loc = ctx.local_assigns(scope, "labelnames") if mod is ctx.mod else []
        if not loc:
            return list(BASE_LABELS), True, ""
    if isinstance(node, ast.Attribute) and mod.src(node) in ("self._labelnames", "self.labelnames"):
        return list(BASE_LABELS), True, ""
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        for m2, v in deref(ctx, mod, node, scope, binds, depth + 1):
            if v is not node:
                c2 = ctx if m2 is ctx.mod else ctx_of(m2)
                return resolve_labels(c2, m2, v, None, {}, depth + 1)
    return [], False, " ".join(mod.src(node).split())


def _pure_arith(node) -> bool:
    """Only numbers, arithmetic, range() and the comprehension's own loop names."""
    loop_names = {n.id for g in node.generators for n in ast.walk(g.target) if isinstance(n, ast.Name)}
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and n.id not in loop_names and n.id != "range":
            return False
        if isinstance(n, ast.Attribute):
            return False
        if isinstance(n, ast.Call) and not (isinstance(n.func, ast.Name) and n.func.id == "range"):
            return False
    return True


def literal_numbers(mod: va.Module, node) -> list | None:
    try:
        v = ast.literal_eval(node)
        if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            return list(v)
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def resolve_buckets(ctx: Ctx, mod: va.Module, node, scope, binds, depth=0) -> tuple[list | None, str]:
    """(bucket list, description). Description is set when the buckets are not a plain literal."""
    if node is None:
        return None, "prometheus_client default buckets"
    if depth > 8:
        return None, "?"
    nums = literal_numbers(mod, node)
    if nums is not None:
        return nums, ""
    if isinstance(node, ast.ListComp) and _pure_arith(node):
        try:
            vals = eval(compile(ast.Expression(node), "<buckets>", "eval"), {"__builtins__": {}}, {"range": range})
            return list(vals), "computed: " + " ".join(mod.src(node).split())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
        base, note = resolve_buckets(ctx, mod, node.value, scope, binds, depth + 1)
        sl = node.slice
        try:
            lo = ast.literal_eval(sl.lower) if sl.lower else None
            hi = ast.literal_eval(sl.upper) if sl.upper else None
        except ValueError:
            lo = hi = None
        if base is not None and sl.step is None:
            return base[lo:hi], (note + "; " if note else "") + " ".join(mod.src(node).split())
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "histogram_buckets":
        fam = va.str_const(node.args[0]) if node.args else None
        if fam == "request_tokens":
            return None, ('family "request_tokens": 1-2-5 series up to max_model_len '
                          "(build_1_2_5_buckets); --custom-histogram-buckets can replace it")
        bm = va.module(BUCKETS_FILE)
        d = bm.assigns.get("_STATIC_FAMILY_DEFAULTS")
        if isinstance(d, ast.Dict) and fam:
            for k, v in zip(d.keys, d.values):
                if va.str_const(k) == fam and isinstance(v, ast.Name):
                    vals = literal_numbers(bm, bm.assigns.get(v.id))
                    return vals, f'family "{fam}" ({v.id}); --custom-histogram-buckets can replace it'
        return None, f"histogram_buckets({fam!r})"
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        for m2, v in deref(ctx, mod, node, scope, binds, depth + 1):
            if v is not node:
                c2 = ctx if m2 is ctx.mod else ctx_of(m2)
                return resolve_buckets(c2, m2, v, None, {}, depth + 1)
    return None, "set at run time: " + " ".join(mod.src(node).split())


# ---------------------------------------------------------------------------
# Scan
# ---------------------------------------------------------------------------
def kwarg(call: ast.Call, name: str, pos: int | None = None):
    for k in call.keywords:
        if k.arg == name:
            return k.value
    if pos is not None and len(call.args) > pos:
        return call.args[pos]
    return None


def scan_file(rel: str, unresolved: list) -> list[dict]:
    mod = va.module(rel)
    ctx = ctx_of(mod)
    rows = []
    for call in (n for n in ast.walk(mod.tree) if isinstance(n, ast.Call)):
        f = call.func
        fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        scope = ctx.enclosing_func(call)
        if fname in CTORS:
            name_node = kwarg(call, "name", 0)
            if name_node is None:
                continue
            for binds in loop_bindings(ctx, call, name_node):
                names = resolve_str(ctx, mod, name_node, scope, binds)
                if not names:
                    if "vllm" in mod.src(call) or kwarg(call, "documentation", 1) is not None:
                        unresolved.append(f"{rel}:{call.lineno}: {' '.join(mod.src(name_node).split())[:80]}")
                    continue
                for nm, nf, nl in names:
                    if not nm.startswith("vllm:"):
                        continue
                    doc_node = kwarg(call, "documentation", 1)
                    docs = resolve_str(ctx, mod, doc_node, scope, binds) if doc_node is not None else []
                    labels = resolve_labels(ctx, mod, kwarg(call, "labelnames", 2), scope, binds)
                    kind = CTORS[fname]
                    buckets = resolve_buckets(ctx, mod, kwarg(call, "buckets"), scope, binds) if kind == "histogram" else (None, "")
                    mp = kwarg(call, "multiprocess_mode")
                    rows.append(make_row(nm, kind, rel, call.lineno, nf, nl, docs, labels, buckets,
                                         ctx.conditions(call), mod.src(name_node),
                                         va.str_const(mp) if mp is not None else None))
        elif fname in META:
            key_node = None
            p = ctx.parents.get(call)
            if isinstance(p, ast.Dict):
                for k, v in zip(p.keys, p.values):
                    if v is call:
                        key_node = k
            elif isinstance(p, ast.Assign) and isinstance(p.targets[0], ast.Subscript):
                key_node = p.targets[0].slice
            if key_node is None:
                unresolved.append(f"{rel}:{call.lineno}: {fname} without a dict key")
                continue
            names = resolve_str(ctx, mod, key_node, scope, {})
            if not names:
                unresolved.append(f"{rel}:{call.lineno}: {' '.join(mod.src(key_node).split())[:80]}")
                continue
            for nm, nf, nl in names:
                if not nm.startswith("vllm:"):
                    continue
                doc_node = kwarg(call, "documentation")
                docs = resolve_str(ctx, mod, doc_node, scope, {}) if doc_node is not None else []
                extra = resolve_labels(ctx, mod, kwarg(call, "labelnames"), scope, {})
                labels = (list(BASE_LABELS) + extra[0], True, extra[2])
                kind = META[fname]
                buckets = resolve_buckets(ctx, mod, kwarg(call, "buckets"), scope, {}) if kind == "histogram" else (None, "")
                rows.append(make_row(nm, kind, rel, call.lineno, nf, nl, docs, labels, buckets,
                                     ctx.conditions(call), mod.src(key_node), None, metadata=True))
    return rows


def make_row(name, kind, rel, call_line, name_file, name_line, docs, labels, buckets, conds, name_src, mp,
             metadata=False) -> dict:
    doc = docs[0][0] if docs else ""
    deprecated = bool(re.search(r"deprecat", name_src, re.I) or re.search(r"\bdeprecat", doc, re.I))
    return {
        "name": name,
        "type": kind,
        "documentation": va.first_paragraph(doc, DOC_LIMIT),
        "labels": labels[0],
        "base_labels": labels[1],
        "labels_unresolved": labels[2],
        "buckets": buckets[0],
        "buckets_note": buckets[1],
        "multiprocess_mode": mp,
        "file": rel,
        "line": call_line,
        "name_file": name_file,
        "name_line": name_line,
        "conditions": conds,
        "declared_as": "Offloading*Metadata entry (created in offloading/metrics.py _create_metric)" if metadata else "constructor call",
        "deprecated": deprecated,
        "group": group_of(rel),
    }


def exposed(name: str, kind: str) -> list[str]:
    if kind == "counter":
        base = name[:-6] if name.endswith("_total") else name
        return [base + "_total"]
    if kind == "histogram":
        return [name + "_bucket", name + "_sum", name + "_count"]
    if kind == "summary":
        return [name + "_sum", name + "_count"]
    if kind == "info":
        return [name + "_info"]
    return [name]


def hidden_mechanism() -> dict:
    uses = [(p, ln, t.strip()) for p, ln, t in va.git_grep(["-n", "show_hidden_metrics\\b"], ["vllm/"])]
    gated = [(p, ln, t) for p, ln, t in uses
             if not re.search(r"def show_hidden_metrics|self\.show_hidden_metrics = |show_hidden_metrics_for_version", t)]
    return {"uses": [f"{p}:{ln}" for p, ln, _ in uses], "gating_reads": [f"{p}:{ln}: {t}" for p, ln, t in gated]}


def build() -> tuple[dict, str, list[str]]:
    unresolved: list[str] = []
    rows: list[dict] = []
    for rel in candidate_files():
        rows += scan_file(rel, unresolved)
    # merge duplicates (same name declared in more than one place); a constructor call wins over a
    # metadata entry, because the call is what creates the metric (and has the real labels)
    rows.sort(key=lambda r: r["declared_as"] != "constructor call")
    merged: dict[str, dict] = {}
    for r in rows:
        if r["name"] in merged:
            m = merged[r["name"]]
            site = f"{r['file']}:{r['line']}"
            if site not in m["also_declared_at"] and site != f"{m['file']}:{m['line']}":
                m["also_declared_at"].append(site)
            if r["type"] != m["type"]:
                m.setdefault("type_conflict", []).append(r["type"])
            continue
        r["also_declared_at"] = []
        merged[r["name"]] = r
    out = list(merged.values())
    for r in out:
        r["exposed_as"] = exposed(r["name"], r["type"])
    order = []
    for _, g in GROUPS:
        if g not in order:
            order.append(g)
    order.append("Other")
    out.sort(key=lambda r: (order.index(r["group"]), r["file"], r["name_line"] if r["name_file"] == r["file"] else r["line"]))
    cites = []
    for r in out:
        cites.append((r["file"], r["line"], ""))
        cites.append((r["name_file"], r["name_line"], r["name"]))
    errors = va.verify_cites(cites)
    hm = hidden_mechanism()
    base = [(p, ln) for p, ln, t in va.git_grep(["-F", 'labelnames = ["model_name", "engine"]'], ["vllm/v1/metrics/loggers.py"])]
    if len(base) != 1:
        errors.append(f"expected one base labelnames line in loggers.py, found {base}")
    data = {
        "base_labels_site": f"{base[0][0]}:{base[0][1]}" if base else "",
        "count": len(out),
        "groups": {g: sum(1 for r in out if r["group"] == g) for g in order if any(r["group"] == g for r in out)},
        "by_type": {t: sum(1 for r in out if r["type"] == t) for t in ("counter", "gauge", "histogram", "summary", "info")},
        "deprecated": [r["name"] for r in out if r["deprecated"]],
        "show_hidden_metrics": hm,
        "unresolved": sorted(set(unresolved)),
        "metrics": out,
    }
    return data, render(data), errors


def render(data: dict) -> str:
    bt = data["by_type"]
    hm = data["show_hidden_metrics"]
    out = [f"<p>{data['count']} metric names that start with <code>vllm:</code>, declared in vLLM code: "
           f"{bt['counter']} counters, {bt['gauge']} gauges, {bt['histogram']} histograms. "
           f"The stat loggers use the labels <code>model_name</code> and <code>engine</code> "
           f"({va.cite(*_split(data['base_labels_site']))}). A † marks these two labels only where a "
           f"helper class (speculative decoding, performance, KV connectors) gets them through its "
           f"<code>labelnames</code> parameter. Most <code>PrometheusStatLogger</code> metrics have the "
           f"same two labels, without a †. "
           f"{len(data['deprecated'])} metrics are marked deprecated; "
           + ("no metric checks <code>show_hidden_metrics</code> at this commit." if not hm["gating_reads"]
              else f"{len(hm['gating_reads'])} places read <code>show_hidden_metrics</code>.")
           + "</p>"]
    if data["unresolved"]:
        sites = []
        for u in data["unresolved"]:
            loc, _, expr = u.partition(": ")
            p, _, ln = loc.rpartition(":")
            sites.append(f"{va.code(expr)} ({va.cite(p, int(ln))})")
        out.append(f"<p>Not in the table: {len(data['unresolved'])} names this scan cannot resolve "
                   f"(they come from code outside the repo): " + "; ".join(sites) + ".</p>")
    for g, n in data["groups"].items():
        out.append(f'<h3 id="metrics-{va.slug(g)}">{va.esc(g)} ({n})</h3>')
        out.append("<table>")
        out.append("  <thead><tr><th>Metric</th><th>Type and labels</th><th>Documentation string</th>"
                   "<th>Declared</th></tr></thead>")
        out.append("  <tbody>")
        for r in (r for r in data["metrics"] if r["group"] == g):
            name = va.code(r["name"])
            if r["deprecated"]:
                name += " <em>(deprecated)</em>"
            exp = r["exposed_as"]
            if exp != [r["name"]]:
                name += "<br>exposed as " + ", ".join(va.code(e) for e in exp)
            labels = []
            for lab in r["labels"]:
                mark = "†" if r["base_labels"] and lab in BASE_LABELS else ""
                labels.append(va.code(lab) + mark)
            if r["labels_unresolved"]:
                labels.append("set at run time: " + va.code(r["labels_unresolved"]))
            tl = r["type"] + "<br>labels: " + (", ".join(labels) if labels else "none")
            if r["type"] == "histogram":
                if r["buckets"] is not None:
                    tl += "<br>buckets: " + va.code(", ".join(_num(x) for x in r["buckets"]))
                if r["buckets_note"]:
                    tl += "<br>" + va.esc(r["buckets_note"])
            doc = va.esc(r["documentation"]) if r["documentation"] else "<em>(none)</em>"
            if r["conditions"]:
                doc += "<br><em>Created only if</em> " + " and ".join(va.code(c) for c in reversed(r["conditions"])) + "."
            cites = []
            if (r["name_file"], r["name_line"]) != (r["file"], r["line"]):
                cites.append(va.cite(r["name_file"], r["name_line"], label="<code>name</code>"))
            cites.append(va.cite(r["file"], r["line"]))
            for s in r["also_declared_at"]:
                p, _, ln = s.rpartition(":")
                cites.append(va.cite(p, int(ln)))
            out.append(f"    <tr><td>{name}</td><td>{tl}</td><td>{doc}</td><td>{'<br>'.join(cites)}</td></tr>")
        out.append("  </tbody>")
        out.append("</table>")
    return "\n".join(out)


def _split(site: str) -> tuple[str, int]:
    p, _, ln = site.rpartition(":")
    return p, int(ln)


def _num(x) -> str:
    if isinstance(x, float) and abs(x) >= 1e5:
        return f"{x:g}".replace("e+0", "e").replace("e+", "e")
    return repr(x)


def report(data: dict) -> None:
    print(f"{data['count']} metrics; by type {data['by_type']}")
    for g, n in data["groups"].items():
        print(f"  {n:4d}  {g}")
    print("\ndeprecated:", ", ".join(data["deprecated"]) or "-")
    print("show_hidden_metrics uses:", ", ".join(data["show_hidden_metrics"]["uses"]))
    print("show_hidden_metrics gating reads:", data["show_hidden_metrics"]["gating_reads"] or "none")
    print("\nunresolved names:")
    for u in data["unresolved"]:
        print("  " + u)
    print("\ndeclared more than once:")
    for r in data["metrics"]:
        if r["also_declared_at"]:
            print(f"  {r['name']}: {r['file']}:{r['line']} + {r['also_declared_at']}")
    print("\nlabels set at run time:")
    for r in data["metrics"]:
        if r["labels_unresolved"]:
            print(f"  {r['name']}: {r['labels_unresolved']}")
    print("\nconditional:")
    for r in data["metrics"]:
        if r["conditions"]:
            print(f"  {r['name']}: {' / '.join(r['conditions'])}")
    print("\nno documentation:", ", ".join(r["name"] for r in data["metrics"] if not r["documentation"]) or "-")


if __name__ == "__main__":
    sys.exit(va.run(NAME, PAGE_ID, build, report))
