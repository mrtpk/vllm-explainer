#!/usr/bin/env python3
"""Shared helpers for the reference-table generators (gen_env_vars, gen_serve_flags,
gen_config_classes, gen_metrics). Python 3 standard library only.

The generators never import vllm (it is not importable here). They read the source of the
target repo with `ast` and `git grep`, at the commit that explainer.json pins.

What this module gives:
  - REPO, SHA, ROOT and a check that the checkout is at the pinned commit;
  - Module: a parsed source file with its imports, module-level assignments and classes,
    plus a resolver that follows `from x import y` to the module that defines a name;
  - git_grep(): `git grep -n` at the pinned commit (fast on the slow Windows mount);
  - text helpers: first_paragraph(), esc() (HTML-escape and break "{{" so that build.py
    does not read it as an inline tag), cite() in the site's {{cite path:line}} format;
  - verify_cites(): every cited line exists and contains an expected token;
  - run(): the shared main loop (write JSON + page region, --check, --report).
"""
from __future__ import annotations

import ast
import html
import inspect
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "explainer.json").read_text(encoding="utf-8"))
REPO = (ROOT / CFG["repo_path"]).resolve()
SHA = CFG["sha"]
GEN_DIR = ROOT / "notes" / "generated"


def check_head() -> None:
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != SHA:
        sys.exit(f"{REPO} is at {head[:12]}, but explainer.json pins {SHA[:12]}")


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def lines(rel: str) -> list[str]:
    return read(rel).splitlines()


def git_grep(args: list[str], paths: list[str] | None = None) -> list[tuple[str, int, str]]:
    """Run `git grep -n <args> SHA -- <paths>`; return (path, line, text) tuples."""
    cmd = ["git", "-C", str(REPO), "grep", "-n", *args, SHA]
    if paths:
        cmd += ["--", *paths]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode not in (0, 1):
        sys.exit(f"git grep failed: {out.stderr.strip()}")
    res = []
    prefix = SHA + ":"
    for ln in out.stdout.splitlines():
        if ln.startswith(prefix):
            ln = ln[len(prefix):]
        path, _, rest = ln.partition(":")
        num, _, text = rest.partition(":")
        if num.isdigit():
            res.append((path, int(num), text))
    return res


def git_grep_files(args: list[str], paths: list[str] | None = None) -> list[str]:
    """`git grep -l <args> SHA -- <paths>`: the matching file paths."""
    cmd = ["git", "-C", str(REPO), "grep", "-l", *args, SHA]
    if paths:
        cmd += ["--", *paths]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if out.returncode not in (0, 1):
        sys.exit(f"git grep failed: {out.stderr.strip()}")
    return sorted(ln[len(SHA) + 1:] if ln.startswith(SHA + ":") else ln for ln in out.stdout.splitlines())


def module_file(dotted: str) -> str | None:
    """'vllm.config.model' -> 'vllm/config/model.py' (or the package __init__.py)."""
    base = dotted.replace(".", "/")
    for cand in (base + ".py", base + "/__init__.py"):
        if (REPO / cand).is_file():
            return cand
    return None


def file_module(rel: str) -> str:
    p = rel[:-3] if rel.endswith(".py") else rel
    if p.endswith("/__init__"):
        p = p[: -len("/__init__")]
    return p.replace("/", ".")


# ---------------------------------------------------------------------------
# Parsed modules and a small name resolver
# ---------------------------------------------------------------------------
class Module:
    def __init__(self, rel: str):
        self.rel = rel
        self.text = read(rel)
        self.tree = ast.parse(self.text)
        self.dotted = file_module(rel)
        self.is_pkg = rel.endswith("__init__.py")
        self.imports: dict[str, tuple[str, str | None]] = {}   # local name -> (module, attr or None)
        self.assigns: dict[str, ast.AST] = {}                  # module-level name -> value node
        self.assign_nodes: dict[str, ast.stmt] = {}
        self.classes: dict[str, ast.ClassDef] = {}
        self.functions: dict[str, ast.FunctionDef] = {}
        self.star_imports: list[str] = []                      # `from x import *`
        self._collect(self.tree.body)

    def _collect(self, body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.If):              # e.g. `if TYPE_CHECKING:` blocks
                self._collect(node.body)
                self._collect(node.orelse)
            elif isinstance(node, ast.Try):
                self._collect(node.body)
            elif isinstance(node, ast.ImportFrom):
                mod = self._abs_module(node.module, node.level)
                for a in node.names:
                    if a.name == "*":
                        self.star_imports.append(mod)
                    else:
                        self.imports[a.asname or a.name] = (mod, a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        self.imports[a.asname] = (a.name, None)
                    else:
                        self.imports[a.name.split(".")[0]] = (a.name.split(".")[0], None)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        self.assigns[t.id] = node.value
                        self.assign_nodes[t.id] = node
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
                self.assigns[node.target.id] = node.value
                self.assign_nodes[node.target.id] = node
            elif hasattr(ast, "TypeAlias") and isinstance(node, getattr(ast, "TypeAlias")):
                self.assigns[node.name.id] = node.value
                self.assign_nodes[node.name.id] = node
            elif isinstance(node, ast.ClassDef):
                self.classes[node.name] = node
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.functions[node.name] = node

    def _abs_module(self, module: str | None, level: int) -> str:
        if not level:
            return module or ""
        parts = self.dotted.split(".")
        if not self.is_pkg:
            parts = parts[:-1]
        if level > 1:
            parts = parts[: len(parts) - (level - 1)]
        return ".".join(parts + ([module] if module else []))

    def src(self, node: ast.AST | None) -> str:
        if node is None:
            return ""
        return ast.get_source_segment(self.text, node) or ast.unparse(node)


@lru_cache(maxsize=None)
def module(rel: str) -> Module:
    return Module(rel)


def module_by_name(dotted: str) -> Module | None:
    rel = module_file(dotted)
    return module(rel) if rel else None


def resolve(mod: Module, name: str, depth: int = 0) -> tuple[Module, ast.AST] | None:
    """Find what `name` means in `mod`: (defining module, ClassDef / FunctionDef / value node)."""
    if depth > 8:
        return None
    if name in mod.classes:
        return mod, mod.classes[name]
    if name in mod.assigns:
        return mod, mod.assigns[name]
    if name in mod.functions:
        return mod, mod.functions[name]
    if name in mod.imports:
        target, attr = mod.imports[name]
        if attr is None:
            return None
        tm = module_by_name(target)
        r = resolve(tm, attr, depth + 1) if tm is not None else None
        if r is not None:
            return r
        sub = module_by_name(f"{target}.{attr}")     # `from pkg import submodule`
        if sub is not None:
            return sub, sub.tree
        return None
    for star in mod.star_imports:
        sm = module_by_name(star)
        r = resolve(sm, name, depth + 1) if sm is not None else None
        if r is not None:
            return r
    return None


def find_class(name: str, hint: Module | None = None) -> tuple[Module, ast.ClassDef] | None:
    """Resolve a class by name: through the imports of `hint`, else by a repo-wide `git grep`."""
    if hint is not None:
        r = resolve(hint, name)
        if r and isinstance(r[1], ast.ClassDef):
            return r  # type: ignore[return-value]
    hits = _class_index().get(name, [])
    if len(hits) >= 1:
        m = module(hits[0])
        return m, m.classes[name]
    return None


@lru_cache(maxsize=None)
def _class_index() -> dict[str, list[str]]:
    idx: dict[str, list[str]] = {}
    for path, _, text in git_grep(["-E", r"^class [A-Za-z_0-9]+\b"], ["vllm/"]):
        m = re.match(r"class ([A-Za-z_0-9]+)", text)
        if m and path.endswith(".py"):
            idx.setdefault(m.group(1), []).append(path)
    return idx


def decorator_names(cls: ast.ClassDef) -> list[str]:
    out = []
    for d in cls.decorator_list:
        f = d.func if isinstance(d, ast.Call) else d
        if isinstance(f, ast.Name):
            out.append(f.id)
        elif isinstance(f, ast.Attribute):
            out.append(f.attr)
    return out


def str_const(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def docstring_after(body: list[ast.stmt], i: int) -> ast.Constant | None:
    """The string literal right after body[i] (the attribute-docstring convention)."""
    if i + 1 < len(body):
        nxt = body[i + 1]
        if isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant) and isinstance(nxt.value.value, str):
            return nxt.value
    return None


# ---------------------------------------------------------------------------
# Text and HTML
# ---------------------------------------------------------------------------
def first_paragraph(text: str | None, limit: int = 0) -> str:
    """First paragraph of a docstring or comment block, whitespace collapsed."""
    if not text:
        return ""
    t = inspect.cleandoc(text)
    para = re.split(r"\n\s*\n", t.strip(), maxsplit=1)[0]
    para = " ".join(para.split())
    if limit and len(para) > limit:
        cut = para[:limit].rsplit(" ", 1)[0]
        para = cut + " …"
    return para


def esc(text: str) -> str:
    """HTML-escape, and break '{' '}' so build.py never reads code text as an inline tag."""
    return html.escape(text, quote=False).replace("{", "&#123;").replace("}", "&#125;")


def code(text: str) -> str:
    return f"<code>{esc(text)}</code>"


def short_path(path: str) -> str:
    return path[len("vllm/"):] if path.startswith("vllm/") else path


def cite(path: str, line: int, end: int | None = None, label: str | None = None) -> str:
    rng = f"{line}" if not end or end == line else f"{line}-{end}"
    lab = label if label is not None else f"<code>{esc(short_path(path))}:{rng}</code>"
    return f"{{{{cite {path}:{rng} | {lab}}}}}"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def verify_cites(cites: list[tuple[str, int, str]]) -> list[str]:
    """cites: (path, line, token). Error when the line does not exist or lacks the token."""
    errors = []
    for path, ln, token in cites:
        if not (REPO / path).is_file():
            errors.append(f"cite {path}:{ln}: no such file")
            continue
        ls = lines(path)
        if not 1 <= ln <= len(ls):
            errors.append(f"cite {path}:{ln}: out of range ({len(ls)} lines)")
            continue
        if token and token not in ls[ln - 1]:
            errors.append(f"cite {path}:{ln}: line does not contain {token!r}: {ls[ln - 1].strip()[:80]!r}")
    return errors


# ---------------------------------------------------------------------------
# Output: JSON + page region, --check, --report
# ---------------------------------------------------------------------------
def splice(page: str, marker: str, content: str) -> str | None:
    pat = re.compile(rf"(<!-- BEGIN GENERATED {re.escape(marker)} -->\n?).*?(<!-- END GENERATED {re.escape(marker)} -->)", re.S)
    if not pat.search(page):
        return None
    return pat.sub(lambda m: m.group(1).rstrip("\n") + "\n" + content + "\n" + m.group(2), page, count=1)


def dump_json(data: dict) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False, sort_keys=False) + "\n"


def run(name: str, page_id: str, build: Callable[[], tuple[dict, str, list[str]]],
        report: Callable[[dict], None]) -> int:
    """build() -> (json data, HTML for the page region, errors). Handles --check and --report."""
    check_head()
    data, body, errors = build()
    for e in errors:
        print("error:", e, file=sys.stderr)
    if errors:
        return 1
    data = {"generator": f"tools/gen_{name.replace('-', '_')}.py", "repo": CFG.get("repo_label", ""),
            "sha": SHA, **data}
    if "--report" in sys.argv:
        report(data)
    json_path = GEN_DIR / f"{name}.json"
    new_json = dump_json(data)
    page_path = ROOT / "src" / f"{page_id}.html"
    page_old = page_path.read_text(encoding="utf-8") if page_path.is_file() else None
    page_new = splice(page_old, name, body) if page_old is not None else None
    if page_old is None:
        note = f"note: {page_path.relative_to(ROOT)} does not exist yet; only the JSON is written"
    elif page_new is None:
        note = f"note: {page_path.relative_to(ROOT)} has no <!-- BEGIN GENERATED {name} --> markers; only the JSON is written"
    else:
        note = ""
    if "--check" in sys.argv:
        bad = []
        if not json_path.is_file() or json_path.read_text(encoding="utf-8") != new_json:
            bad.append(str(json_path.relative_to(ROOT)))
        if page_new is not None and page_new != page_old:
            bad.append(str(page_path.relative_to(ROOT)))
        if bad:
            print(f"out of date: {', '.join(bad)}; run python3 tools/gen_{name.replace('-', '_')}.py", file=sys.stderr)
            return 1
        print(f"ok: {name} is up to date" + (f" ({note})" if note else ""))
        return 0
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    if not json_path.is_file() or json_path.read_text(encoding="utf-8") != new_json:
        json_path.write_text(new_json, encoding="utf-8")
    if page_new is not None and page_new != page_old:
        page_path.write_text(page_new, encoding="utf-8")
    print(f"wrote {json_path.relative_to(ROOT)}" + (f" and {page_path.relative_to(ROOT)}" if page_new is not None else ""))
    if note:
        print(note)
    return 0
