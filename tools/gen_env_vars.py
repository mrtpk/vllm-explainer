#!/usr/bin/env python3
"""Generate the environment-variable reference (src/reference/env-vars.html) from vllm/envs.py.

Owner: the reference-table writer. Python 3 standard library only; vllm is never imported.

What the script does:
  1. Parses vllm/envs.py with `ast`:
       - every key of the dict `environment_variables` (one row each, cited at the key line);
       - the comment block right above the key (the description; first paragraph, whitespace
         collapsed, quoted as-is because it is code documentation). Section banners
         (`# ===== ... =====`) are not part of a description;
       - type and default from the `if TYPE_CHECKING:` block at the top of the file (the
         annotation and the value, as source text). Some entries have no TYPE_CHECKING line;
         the row then says so and shows only the parse default;
       - the "parse default": the second argument of the os.getenv / os.environ.get /
         env_with_choices call inside the dict value (what the code really falls back to),
         plus the choices of env_with_choices / env_list_with_choices / env_set_with_choices.
  2. Scans vllm/ (one `git grep` at the pinned commit) for uses: `envs.NAME` (also
     `vllm.envs.NAME`), `from vllm.envs import NAME` and `envs.is_set("NAME")`. Each row keeps
     the use count (lines) and the first 5 use sites. vllm/envs.py itself is not counted.
     A second `git grep -w` over the whole repo finds other mentions of the name (os.environ
     reads, setup.py, CMake, tests, docs). A variable with no use in vllm/ is flagged
     "no envs.* use"; the report says if anything else in the repo mentions it.
  3. Checks that every cited line exists and contains the variable name.
  4. Writes notes/generated/env-vars.json and, when src/reference/env-vars.html has the markers,
     the HTML between <!-- BEGIN GENERATED env-vars --> and <!-- END GENERATED env-vars -->.

Grouping rule:
  envs.py has only two section banners: "Installation Time Env Vars" (line 619) and
  "Runtime Env Vars" (line 710). Every entry under the first banner goes to the group
  "Installation time" (the code puts a few runtime switches there, for example
  VLLM_BATCH_INVARIANT; the table keeps the code's placement). Entries under the second banner
  are split by topic with the ordered name rules in TOPICS below: the first rule whose regular
  expression matches the name (without the VLLM_ prefix) wins; no match -> "Other runtime
  settings". The topic split is this tool's choice, not a structure in the code.

Usage:
  python3 tools/gen_env_vars.py            # write JSON (and the page region if the markers exist)
  python3 tools/gen_env_vars.py --check    # exit 1 if the JSON or the page region is out of date
  python3 tools/gen_env_vars.py --report   # also print a summary (groups, unused, default mismatches)
"""
from __future__ import annotations

import ast
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vllm_ast as va  # noqa: E402

ENVS = "vllm/envs.py"
NAME = "env-vars"
PAGE_ID = "reference/env-vars"
MAX_SITES = 5
DESC_LIMIT = 360

INSTALL = "Installation time"
OTHER = "Other runtime settings"
# Ordered topic rules for the runtime section: (group, regex on the name without "VLLM_").
TOPICS: list[tuple[str, str]] = [
    ("ROCm (AMD)", r"^ROCM_"),
    ("CPU backend", r"^CPU_|ZENTORCH"),
    ("Intel XPU", r"^XPU_"),
    ("TPU / XLA", r"^XLA_|^TPU_"),
    ("Logging and usage statistics",
     r"LOGGING|^LOG_|^NO_COLOR$|^FORCE_COLOR$|USAGE|DO_NOT_TRACK|TRACE_FUNCTION|LOG_LOGO|REQUEST_BODY_LOG|DEBUG_LOG_API"),
    ("Debugging and profiling",
     r"DEBUG|PROFILING|NVTX|MFU|PATTERN_MATCH|NANS|MODEL_INSPECTION|^TEST_|CI_ENV"),
    ("torch.compile and CUDA graphs", r"COMPILE|AOT|INDUCTOR|CUDAGRAPH|STANDALONE|PREGRAD|BYTECODE|BREAKABLE"),
    ("KV transfer, offload and disaggregation",
     r"NIXL|^P2P_|^EC_|MOONCAKE|KV_EVENTS|KV_OFFLOAD|SIMPLE_KV_OFFLOAD|NIC_|OBJECT_STORAGE"),
    ("MoE and expert parallelism", r"MOE|DEEPEP|DEEP_GEMM|EXPERT|GROUPED_TOPK|ELASTIC_EP"),
    ("CUDA and library paths",
     r"^CUDA_HOME$|LD_LIBRARY_PATH|CUDA_COMPAT|_SO_PATH|NCCL_INCLUDE_PATH|HAS_FLASHINFER_CUBIN"),
    ("Engine core and model runner",
     r"V2_MODEL_RUNNER|STARTUP_PLAN|SPINLOOP|V1_MULTIPROCESSING|OUTPUT_PROC|EXECUTE_MODEL|ENGINE_READY"
     r"|KEEP_ALIVE_ON_ENGINE|LONG_MAX_MODEL_LEN|SHUTDOWN_TIMEOUT"),
    ("Distributed and parallel execution",
     r"^DP_|RAY_|DISTRIBUTED|NCCL|ALLREDUCE|P2P_CHECK|^DBO_|DCP|PP_LAYER|WORKER_MULTIPROC|HOST_IP|^PORT$"
     r"|LOOPBACK|^MQ_|RANDOMIZE_DP|^LOCAL_RANK$|CUDA_VISIBLE|RPC_BASE|RINGBUFFER"),
    ("Multimodal and media", r"MEDIA|IMAGE|VIDEO|AUDIO|EMBED_DECODE|ASSETS"),
    ("LoRA", r"LORA"),
    ("Structured output and tool calling", r"XGRAMMAR|OUTLINES|REGEX|TOOL|BEAM_ALWAYS|PARSER|STRICT"),
    ("API server and frontend",
     r"API|SERVER|HTTP|CHAT_TEMPLATE|RESPONSES|COHERE|GPT_OSS|SYSTEM_START|MAX_N_SEQ|COMPLETION_PROMPTS|STOP_STRINGS"
     r"|BAD_WORDS|RUST|KEEP_ALIVE|ENGINE_READY|PROCESS_NAME|REQUEST_ID|MODEL_NAME_VALIDATION|OUTPUT_PROC|INSECURE_SERIAL"
     r"|MSGPACK"),
    ("Model loading, storage and caches",
     r"^S3_|MODELSCOPE|FASTOKENS|LOAD_FORMAT|FASTSAFETENSORS|MODEL_REDIRECT|CACHE_ROOT|CONFIG_ROOT|TUNED_CONFIG"
     r"|WEIGHT_OFFLOADING|PLUGINS|PIN_MEMORY"),
    ("Kernels, attention and quantization",
     r"FLASHINFER|MARLIN|HUMMING|TRITON|GEMM|KERNEL|FP4|FP8|MXFP|ATTN|MLA|OINK|HPC_OPS|FLA_|GDN|AWQ|SAMPLER"
     r"|KV_CACHE_LAYOUT|SSM_CONV|HW_AGNOSTIC|LAYERNAME|SPARSE_INDEXER|SPINLOOP|SHARED_EXPERT|STREAM|V2_MODEL_RUNNER"
     r"|ADAPTIVE_VERIFICATION|SLEEP_MEM|WORKSPACE|MEMORY_PROFILER|REPLICATE_EMBED"),
    ("Build and packaging", r"VERSION_SUFFIX"),
]

GETENV_FUNCS = {"getenv", "get"}
CHOICE_FUNCS = {"env_with_choices", "env_list_with_choices", "env_set_with_choices"}


def topic(name: str) -> str:
    short = name[5:] if name.startswith("VLLM_") else name
    for group, rx in TOPICS:
        if re.search(rx, short):
            return group
    return OTHER


def type_checking_block(mod: va.Module) -> tuple[int, dict[str, ast.AnnAssign]]:
    for node in mod.tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return node.lineno, {s.target.id: s for s in node.body
                                 if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)}
    raise SystemExit("envs.py: no `if TYPE_CHECKING:` block")


def env_dict(mod: va.Module) -> tuple[int, ast.Dict]:
    for node in mod.tree.body:
        if (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id == "environment_variables" and isinstance(node.value, ast.Dict)):
            return node.lineno, node.value
    raise SystemExit("envs.py: no `environment_variables` dict")


def comment_above(ls: list[str], line: int) -> tuple[str, int | None]:
    """Comment lines right above `line` (1-based). Returns (text, first comment line)."""
    out: list[str] = []
    first = None
    i = line - 1
    while i >= 1:
        s = ls[i - 1].strip()
        if not s.startswith("#") or re.match(r"#\s*=+.*=+\s*$", s) or s.startswith("# --8<--"):
            break
        out.append(re.sub(r"^#\s?", "", s))
        first = i
        i -= 1
    out.reverse()
    # A bare "#" line is a paragraph break.
    return "\n".join(out), first


def parse_default(mod: va.Module, name: str, value: ast.AST) -> tuple[str | None, list[str] | None, str]:
    """(parse default source, choices, how) from the dict value."""
    for node in ast.walk(value):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        fname = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else "")
        if fname in CHOICE_FUNCS and node.args and va.str_const(node.args[0]) == name:
            dflt = mod.src(node.args[1]) if len(node.args) > 1 else None
            choices = None
            if len(node.args) > 2:
                try:
                    choices = [str(c) for c in ast.literal_eval(node.args[2])]
                except ValueError:
                    choices = [mod.src(node.args[2])]
            return dflt, choices, fname
        if fname in GETENV_FUNCS and node.args and va.str_const(node.args[0]) == name:
            dflt = mod.src(node.args[1]) if len(node.args) > 1 else "None"
            return dflt, None, "getenv"
    if isinstance(value, ast.Name):
        return None, None, f"function {value.id}()"
    return None, None, "custom"


def default_mismatch(tc_default: str | None, parse: str | None, value: ast.AST) -> str:
    """Return a note when the TYPE_CHECKING default and the parse default clearly disagree.

    Only for a plain `os.getenv(NAME, default)` read: a value with `or`, `if ... else` or a dict
    lookup computes another fallback, so the raw getenv default says nothing there."""
    if parse is None or tc_default is None:
        return ""
    if any(isinstance(n, (ast.BoolOp, ast.IfExp, ast.Dict)) for n in ast.walk(value)):
        return ""
    try:
        tcv = ast.literal_eval(tc_default)
        pv = ast.literal_eval(parse)
    except (ValueError, SyntaxError):
        return ""
    if pv is None or tcv is None:
        return "" if (pv is None) == (tcv is None) else f"TYPE_CHECKING says {tc_default}, parse default is {parse}"
    try:
        if isinstance(tcv, bool):
            same = (str(pv).strip().lower() in ("1", "true")) == tcv
        elif isinstance(tcv, (int, float)):
            same = float(pv) == float(tcv)
        else:
            same = str(pv).lower() == str(tcv).lower()
    except ValueError:
        return ""
    return "" if same else f"TYPE_CHECKING says {tc_default}, parse default is {parse}"


def scan_uses(names: set[str]) -> dict[str, list[tuple[str, int]]]:
    uses: dict[str, list[tuple[str, int]]] = {n: [] for n in names}
    pat = r"envs\.[A-Z][A-Z0-9_]*|from vllm\.envs import|envs\.is_set\("
    for path, ln, text in va.git_grep(["-E", pat], ["vllm/"]):
        if path == ENVS or not path.endswith(".py"):
            continue
        found = set(re.findall(r"\benvs\.([A-Z][A-Z0-9_]*)\b", text))
        found |= set(re.findall(r"envs\.is_set\(\s*[\"']([A-Z][A-Z0-9_]*)[\"']", text))
        m = re.match(r"\s*from vllm\.envs import (.+)", text)
        if m:
            found |= {x.strip().split(" as ")[0] for x in m.group(1).strip("() ").split(",")}
        for n in sorted(found & names):
            uses[n].append((path, ln))
    for n in uses:
        uses[n] = sorted(set(uses[n]))
    return uses


def scan_mentions(names: set[str]) -> dict[str, list[tuple[str, int]]]:
    """Every whole-word mention of the name anywhere in the repo, except vllm/envs.py."""
    args = ["-w", "-F", "-I"]
    for n in sorted(names):
        args += ["-e", n]
    hits = va.git_grep(args)
    out: dict[str, list[tuple[str, int]]] = {n: [] for n in names}
    for path, ln, text in hits:
        if path == ENVS:
            continue
        for n in set(re.findall(r"[A-Z][A-Z0-9_]+", text)) & names:
            out[n].append((path, ln))
    return out


def build() -> tuple[dict, str, list[str]]:
    mod = va.module(ENVS)
    ls = va.lines(ENVS)
    tc_line, tc = type_checking_block(mod)
    dict_line, d = env_dict(mod)
    banners = [(i + 1, re.sub(r"[#=\s]+", " ", s).strip()) for i, s in enumerate(ls)
               if re.match(r"\s*#\s*=+.+=+\s*$", s)]
    rows = []
    errors: list[str] = []
    cites: list[tuple[str, int, str]] = [(ENVS, tc_line, "TYPE_CHECKING"), (ENVS, dict_line, "environment_variables")]
    for k, v in zip(d.keys, d.values):
        name = va.str_const(k)
        if name is None:
            errors.append(f"envs.py:{k.lineno}: dict key is not a string literal")
            continue
        section = ""
        for bl, text in banners:
            if bl < k.lineno:
                section = text
        group = INSTALL if section.startswith("Installation") else topic(name)
        text, cline = comment_above(ls, k.lineno)
        pdef, choices, how = parse_default(mod, name, v)
        t = tc.get(name)
        row = {
            "name": name,
            "group": group,
            "section": section,
            "line": k.lineno,
            "end_line": v.end_lineno,
            "type": mod.src(t.annotation) if t else None,
            "default": mod.src(t.value) if t and t.value is not None else None,
            "type_checking_line": t.lineno if t else None,
            "parse_default": pdef,
            "parse": how,
            "choices": choices,
            "description": va.first_paragraph(text, DESC_LIMIT),
            "comment_line": cline,
        }
        row["default_mismatch"] = default_mismatch(row["default"], pdef, v) if t and how == "getenv" else ""
        rows.append(row)
        cites.append((ENVS, k.lineno, f'"{name}"'))
        if t:
            cites.append((ENVS, t.lineno, name))
    names = {r["name"] for r in rows}
    dup = [n for n, c in Counter(r["name"] for r in rows).items() if c > 1]
    if dup:
        errors.append(f"duplicate keys: {dup}")
    uses = scan_uses(names)
    mentions = scan_mentions(names)
    for r in rows:
        u = uses[r["name"]]
        r["use_count"] = len(u)
        r["use_sites"] = [f"{p}:{ln}" for p, ln in u[:MAX_SITES]]
        other = [(p, ln) for p, ln in mentions[r["name"]] if (p, ln) not in set(u)]
        r["other_mentions"] = len(other)
        r["other_mention_files"] = sorted({p for p, _ in other}, key=_mention_rank)[:8]
        r["unused"] = not u
        for p, ln in u[:MAX_SITES]:
            cites.append((p, ln, r["name"]))
    errors += va.verify_cites(cites)
    groups = []
    for r in rows:
        if r["group"] not in groups:
            groups.append(r["group"])
    order = [INSTALL] + [g for g, _ in TOPICS if g in groups] + ([OTHER] if OTHER in groups else [])
    rows.sort(key=lambda r: (order.index(r["group"]), r["line"]))
    data = {
        "source": ENVS,
        "type_checking_line": tc_line,
        "dict_line": dict_line,
        "count": len(rows),
        "groups": {g: sum(1 for r in rows if r["group"] == g) for g in order},
        "unused": [r["name"] for r in rows if r["unused"]],
        "missing_type_checking": [r["name"] for r in rows if r["type"] is None],
        "rows": rows,
    }
    return data, render(data), errors


def render(data: dict) -> str:
    rows = data["rows"]
    out = [f"<p>{data['count']} variables in <code>environment_variables</code> "
           f"({va.cite(ENVS, data['dict_line'])}), in {len(data['groups'])} groups. "
           f"{len(data['unused'])} have no <code>envs.*</code> use in <code>vllm/</code>. "
           f"Type and default come from the <code>TYPE_CHECKING</code> block "
           f"({va.cite(ENVS, data['type_checking_line'])}).</p>"]
    for g, n in data["groups"].items():
        out.append(f'<h3 id="env-{va.slug(g)}">{va.esc(g)} ({n})</h3>')
        out.append("<table>")
        out.append("  <thead><tr><th>Variable</th><th>Type and default</th><th>Description (code comment)</th>"
                   "<th>Used in <code>vllm/</code></th></tr></thead>")
        out.append("  <tbody>")
        for r in (r for r in rows if r["group"] == g):
            name = r["name"]
            first = f'<code id="{name}">{name.replace("_", "_<wbr>")}</code><br>{va.cite(ENVS, r["line"])}'
            if r["type"] is not None:
                td = f"{va.code(r['type'])}, {va.code(r['default'])}"
            else:
                td = f"not in TYPE_CHECKING; parse default {va.code(r['parse_default'] or '?')}"
            if r["default_mismatch"]:
                td += f"<br><em>but the read falls back to</em> {va.code(r['parse_default'])}"
            if r["choices"]:
                td += "<br>choices: " + ", ".join(va.code(c) for c in r["choices"])
            desc = va.esc(r["description"]) if r["description"] else "<em>(no comment of its own)</em>"
            if r["unused"]:
                used = "<em>no <code>envs.*</code> use</em>"
                files = sorted(r["other_mention_files"], key=_mention_rank)[:3]
                if files:
                    used += "; named in " + ", ".join(
                        f"{{{{cite {f} | <code>{va.esc(f)}</code>}}}}" for f in files)
            else:
                sites = ", ".join(va.cite(*_split(s)) for s in r["use_sites"])
                more = f" (+{r['use_count'] - len(r['use_sites'])} more)" if r["use_count"] > len(r["use_sites"]) else ""
                used = f"{r['use_count']}: {sites}{more}"
            out.append(f"    <tr><td>{first}</td><td>{td}</td><td>{desc}</td><td>{used}</td></tr>")
        out.append("  </tbody>")
        out.append("</table>")
    return "\n".join(out)


def _mention_rank(path: str) -> tuple[int, str]:
    """Code that reads the variable first, then build files, then docs and tests."""
    for i, pre in enumerate(("vllm/", "setup.py", "csrc/", "rust/", "CMakeLists.txt", "cmake/", "tools/", "docker/")):
        if path.startswith(pre):
            return i, path
    return 99, path


def _split(site: str) -> tuple[str, int]:
    p, _, ln = site.rpartition(":")
    return p, int(ln)


def report(data: dict) -> None:
    print(f"{data['count']} variables in {len(data['groups'])} groups")
    for g, n in data["groups"].items():
        print(f"  {n:4d}  {g}")
    print(f"\nno envs.* use in vllm/ ({len(data['unused'])}):")
    for r in data["rows"]:
        if r["unused"]:
            files = ", ".join(r["other_mention_files"]) or "-"
            print(f"  {r['name']:50s} envs.py:{r['line']}  other mentions: {r['other_mentions']} ({files})")
    print(f"\nnot in TYPE_CHECKING ({len(data['missing_type_checking'])}): {', '.join(data['missing_type_checking'])}")
    print("\nTYPE_CHECKING default differs from the parse default:")
    for r in data["rows"]:
        if r["default_mismatch"]:
            print(f"  {r['name']:50s} envs.py:{r['line']}  {r['default_mismatch']}")
    print("\nno comment of their own:", ", ".join(r["name"] for r in data["rows"] if not r["description"]))
    top = sorted(data["rows"], key=lambda r: -r["use_count"])[:10]
    print("\nmost used:", ", ".join(f"{r['name']} ({r['use_count']})" for r in top))


if __name__ == "__main__":
    sys.exit(va.run(NAME, PAGE_ID, build, report))
