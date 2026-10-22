#!/usr/bin/env python3
"""Generate the config-class reference (src/reference/config-classes.html) from vllm/config/*.py.

Owner: the reference-table writer. Python 3 standard library only; vllm is never imported.

What the script does:
  1. Parses every vllm/config/*.py file with `ast` and keeps each top-level class whose decorator
     is `config` (`@config` or `@config(...)`, the pydantic-dataclass decorator of
     vllm/config/utils.py). For each class: name, file and line of the `class` statement, the
     first paragraph of the class docstring, the base classes and the decorator source.
  2. For each class-body annotated assignment `name: T [= default]`: the field name and line,
     the annotation and the default as source text (whitespace collapsed; long text is cut with
     "…"), and the field docstring = the string literal right after the assignment. That is the
     convention that get_attr_docs (vllm/config/utils.py:201) reads, and the CLI help text of
     `vllm serve` comes from it. ClassVar annotations are listed with kind "ClassVar" (not a
     dataclass field) and InitVar annotations with kind "InitVar" (an __init__ argument only; it
     is not in dataclasses.fields(), so get_kwargs() never makes a flag from it).
  3. Maps each VllmConfig attribute to the config class it holds (a config-class name in the
     attribute's annotation).
  4. Checks that every cited line exists and contains the class or field name.
  5. Writes notes/generated/config-classes.json and, when src/reference/config-classes.html
     has the markers, the HTML between <!-- BEGIN GENERATED config-classes --> and
     <!-- END GENERATED config-classes -->.

Known limits:
  - Inherited fields are not copied into the subclass table; the "Bases" column names the base.
  - Defaults are source text, not evaluated values (for example `Field(default_factory=CacheConfig)`).
  - Only vllm/config/*.py is scanned. @config classes elsewhere (for example FrontendArgs in
    vllm/entrypoints/common/cli_args.py) are not in this table; gen_serve_flags.py covers those.

Usage:
  python3 tools/gen_config_classes.py            # write JSON (and the page region if the markers exist)
  python3 tools/gen_config_classes.py --check    # exit 1 if the JSON or the page region is out of date
  python3 tools/gen_config_classes.py --report   # also print a summary
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vllm_ast as va  # noqa: E402

NAME = "config-classes"
PAGE_ID = "reference/config-classes"
SRC_LIMIT = 140
DOC_LIMIT = 400


def collapse(text: str, limit: int = SRC_LIMIT) -> str:
    t = " ".join(text.split())
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\[\s+", "[", t)
    t = re.sub(r",\s*\]", "]", t)
    t = re.sub(r",\s*\)", ")", t)
    if len(t) > limit:
        t = t[:limit].rstrip() + " …"
    return t


def is_config_class(cls: ast.ClassDef) -> bool:
    return "config" in va.decorator_names(cls)


def config_files() -> list[str]:
    return sorted(p.relative_to(va.REPO).as_posix() for p in (va.REPO / "vllm" / "config").glob("*.py"))


def parse_class(mod: va.Module, cls: ast.ClassDef) -> dict:
    deco = next(d for d in cls.decorator_list
                if (isinstance(d, ast.Name) and d.id == "config")
                or (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == "config"))
    fields = []
    for i, node in enumerate(cls.body):
        if not (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)):
            continue
        ann = mod.src(node.annotation)
        kind = "field"
        if re.match(r"(typing\.)?ClassVar\b", ann):
            kind = "ClassVar"
        elif re.match(r"(dataclasses\.)?InitVar\b", ann):
            kind = "InitVar"
        doc = va.docstring_after(cls.body, i)
        fields.append({
            "name": node.target.id,
            "line": node.lineno,
            "kind": kind,
            "type": collapse(ann),
            "default": collapse(mod.src(node.value)) if node.value is not None else None,
            "doc": va.first_paragraph(doc.value, DOC_LIMIT) if doc else "",
        })
    return {
        "name": cls.name,
        "file": mod.rel,
        "line": cls.lineno,
        "decorator": "@" + collapse(mod.src(deco)),
        "decorator_line": deco.lineno,
        "bases": [mod.src(b) for b in cls.bases],
        "doc": va.first_paragraph(ast.get_docstring(cls, clean=False), DOC_LIMIT),
        "fields": fields,
    }


def build() -> tuple[dict, str, list[str]]:
    classes = []
    cites: list[tuple[str, int, str]] = []
    for rel in config_files():
        mod = va.module(rel)
        for node in mod.tree.body:
            if isinstance(node, ast.ClassDef) and is_config_class(node):
                c = parse_class(mod, node)
                classes.append(c)
                cites.append((rel, c["line"], f"class {c['name']}"))
                for f in c["fields"]:
                    cites.append((rel, f["line"], f["name"]))
    names = {c["name"] for c in classes}
    errors = va.verify_cites(cites)
    dup = sorted({c["name"] for c in classes if sum(1 for d in classes if d["name"] == c["name"]) > 1})
    if dup:
        errors.append(f"duplicate class names: {dup}")
    for c in classes:
        c["config_bases"] = [b for b in c["bases"] if b in names]
    # VllmConfig attribute -> config class
    vllm_cfg = next((c for c in classes if c["name"] == "VllmConfig"), None)
    holds = []
    if vllm_cfg is None:
        errors.append("VllmConfig not found")
    else:
        for f in vllm_cfg["fields"]:
            if f["kind"] != "field":
                continue
            toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", f["type"])
            cls = next((t for t in toks if t in names), None)
            other = next((t for t in toks if t.endswith("Config") and t not in names), None)
            holds.append({"attribute": f["name"], "line": f["line"], "class": cls,
                          "non_config_class": other if cls is None else None,
                          "type": f["type"], "default": f["default"]})
    um = va.module("vllm/config/utils.py")
    deco_line = next(n.lineno for n in um.tree.body
                     if isinstance(n, ast.FunctionDef) and n.name == "config"
                     and not any(isinstance(d, ast.Name) and d.id == "overload" for d in n.decorator_list))
    errors += va.verify_cites([("vllm/config/utils.py", deco_line, "def config(")])
    data = {
        "source": "vllm/config/*.py",
        "config_decorator_line": deco_line,
        "count": len(classes),
        "field_count": sum(1 for c in classes for f in c["fields"] if f["kind"] == "field"),
        "files": sorted({c["file"] for c in classes}),
        "vllm_config_attributes": holds,
        "classes": classes,
    }
    return data, render(data), errors


def _anchor(name: str) -> str:
    return f"cfg-{name}"


def render(data: dict) -> str:
    classes = data["classes"]
    out = [f"<p>{data['count']} classes with the <code>@config</code> decorator in "
           f"{len(data['files'])} files of {{{{cite vllm/config/ | <code>vllm/config/</code>}}}}, with "
           f"{data['field_count']} fields. The decorator makes a pydantic dataclass that forbids extra fields "
           f"({va.cite('vllm/config/utils.py', data['config_decorator_line'])}).</p>"]
    out.append('<h3 id="cfg-overview">All config classes</h3>')
    out.append("<table>")
    out.append("  <thead><tr><th>Class</th><th>Defined in</th><th>Fields</th><th>Docstring (first paragraph)</th></tr></thead>")
    out.append("  <tbody>")
    for c in classes:
        n = sum(1 for f in c["fields"] if f["kind"] == "field")
        out.append(f'    <tr><td><a href="#{_anchor(c["name"])}"><code>{c["name"]}</code></a></td>'
                   f'<td>{va.cite(c["file"], c["line"])}</td><td>{n}</td><td>{va.esc(c["doc"])}</td></tr>')
    out.append("  </tbody>")
    out.append("</table>")

    out.append('<h3 id="cfg-vllmconfig-attributes">What each <code>VllmConfig</code> attribute holds</h3>')
    out.append("<table>")
    out.append("  <thead><tr><th>Attribute</th><th>Holds</th><th>Type</th><th>Default</th></tr></thead>")
    out.append("  <tbody>")
    for h in data["vllm_config_attributes"]:
        if h["class"]:
            holds = f'<a href="#{_anchor(h["class"])}"><code>{h["class"]}</code></a>'
        elif h["non_config_class"]:
            holds = f"{va.code(h['non_config_class'])} (not a <code>@config</code> class)"
        else:
            holds = "—"
        vf = next(c for c in classes if c["name"] == "VllmConfig")["file"]
        out.append(f"    <tr><td>{va.code(h['attribute'])}<br>{va.cite(vf, h['line'])}</td><td>{holds}</td>"
                   f"<td>{va.code(h['type'])}</td><td>{va.code(h['default']) if h['default'] is not None else '<em>required</em>'}</td></tr>")
    out.append("  </tbody>")
    out.append("</table>")

    for c in classes:
        out.append(f'<h3 id="{_anchor(c["name"])}">{c["name"]}</h3>')
        bases = ""
        if c["bases"]:
            bases = " Bases: " + ", ".join(
                f'<a href="#{_anchor(b)}"><code>{va.esc(b)}</code></a>' if b in c["config_bases"] else va.code(b)
                for b in c["bases"]) + "."
        out.append(f"<p>{va.cite(c['file'], c['line'])}, decorator {va.code(c['decorator'])}.{bases}</p>")
        if not c["fields"]:
            out.append("<p><em>No annotated fields in the class body.</em></p>")
            continue
        out.append("<table>")
        out.append("  <thead><tr><th>Field</th><th>Type</th><th>Default</th><th>Docstring (first paragraph)</th></tr></thead>")
        out.append("  <tbody>")
        for f in c["fields"]:
            kind = "" if f["kind"] == "field" else f" <em>({f['kind']})</em>"
            dflt = va.code(f["default"]) if f["default"] is not None else "<em>required</em>"
            out.append(f"    <tr><td>{va.code(f['name'])}{kind}<br>{va.cite(c['file'], f['line'])}</td>"
                       f"<td>{va.code(f['type'])}</td><td>{dflt}</td><td>{va.esc(f['doc'])}</td></tr>")
        out.append("  </tbody>")
        out.append("</table>")
    return "\n".join(out)


def report(data: dict) -> None:
    print(f"{data['count']} @config classes in {len(data['files'])} files, {data['field_count']} fields")
    for c in data["classes"]:
        kinds = [f["kind"] for f in c["fields"]]
        nodoc = [f["name"] for f in c["fields"] if f["kind"] == "field" and not f["doc"]]
        extra = f" ClassVar={kinds.count('ClassVar')}" if "ClassVar" in kinds else ""
        extra += f" InitVar={kinds.count('InitVar')}" if "InitVar" in kinds else ""
        print(f"  {c['name']:32s} {c['file']}:{c['line']:<5d} fields={kinds.count('field'):3d}{extra}"
              f"  bases={','.join(c['bases']) or '-'}  undocumented={len(nodoc)}")
    print("\nVllmConfig attributes:")
    for h in data["vllm_config_attributes"]:
        print(f"  {h['attribute']:28s} -> {h['class'] or h['non_config_class'] or '-'}")
    nodoc = [(c["name"], f["name"]) for c in data["classes"] for f in c["fields"] if f["kind"] == "field" and not f["doc"]]
    print(f"\nfields without a docstring: {len(nodoc)}")
    print("  " + ", ".join(f"{a}.{b}" for a, b in nodoc))


if __name__ == "__main__":
    sys.exit(va.run(NAME, PAGE_ID, build, report))
