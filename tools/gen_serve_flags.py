#!/usr/bin/env python3
"""Generate the `vllm serve` flag reference (src/reference/serve-flags.html) without importing vllm.

Owner: the reference-table writer. Python 3 standard library only.

How `vllm serve` builds its parser (the code this script reproduces):
  - vllm/entrypoints/cli/serve.py:172 calls make_arg_parser()
    (vllm/entrypoints/common/cli_args.py:422). It adds `model_tag`, --headless, --api-server-count,
    --config and --grpc directly, then calls FrontendArgs.add_cli_args() and
    AsyncEngineArgs.add_cli_args().
  - FrontendArgs (cli_args.py:281, base BaseFrontendArgs :72) is a @config dataclass.
    BaseFrontendArgs.add_cli_args (:255) runs get_kwargs(cls), lets _customize_cli_kwargs()
    patch the kwargs, and adds one flag per field: `--` + name with "_" -> "-", in a group
    titled cls.__name__ without "Args" ("Frontend").
  - AsyncEngineArgs.add_cli_args (vllm/engine/arg_utils.py:3237) calls EngineArgs.add_cli_args
    (:952) and adds --enable-log-requests. EngineArgs.add_cli_args lists every flag by hand: for
    each config class it runs `X_kwargs = get_kwargs(XConfig)`, makes an argument group titled
    with the class name, and calls `group.add_argument("--flag", **X_kwargs["field"])`, sometimes
    with `**{**X_kwargs["field"], "default": None}` or explicit keywords. So the flag NAME is
    written in add_cli_args, and type/default/choices/help come from the config-class FIELD.
  - get_kwargs -> _compute_kwargs (arg_utils.py:325) turns a dataclass field into argparse
    kwargs: default (field default or default_factory()), help = the field docstring
    (get_attr_docs, vllm/config/utils.py:201), and type/action/nargs/choices from the type hint:
    dataclass -> JSON parsed into that class; {bool, str, None} -> optional value with const True;
    bool -> BooleanOptionalAction; Literal -> choices (metavar if str is also allowed);
    tuple/list/set -> nargs; int (human-readable for some names); float; dict -> JSON; str or any
    non-builtin class -> str; None in the hint -> optional_type and None appended to choices.
  - FlexibleArgumentParser (vllm/utils/argparse_utils.py:158) registers EnableDisableAction for
    BooleanOptionalAction: `--enable-x` pairs with `--disable-x` (and the reverse); `--no-enable-x`
    still parses but is hidden. Other bool flags get the normal `--x` / `--no-x` pair.

This script parses those functions with `ast` and interprets the statements above
(get_kwargs assignments, kwargs patches such as `X["f"]["type"] = ...`, `+=`, `.remove()`, `del`,
add_argument_group, add_argument). Field type hints are resolved through module imports and
type aliases (for example RunnerOption -> Literal[...]) and classified with the rules of
_compute_kwargs.

Where this reconstruction may differ from the real parser:
  1. Plugins and platforms: AsyncEngineArgs.add_cli_args calls load_general_plugins() and
     current_platform.pre_register_and_update(parser). Both can add or change arguments at
     run time; this script does not see them.
  2. Runtime values: choices built at run time are shown as source text (for example
     --mm-processor-device, whose choices depend on current_platform.device_type;
     --tool-call-parser, whose metavar lists the registered parsers).
  3. Defaults are source text or literal values, not the objects argparse holds. A
     default_factory of a config class is shown as `ClassName()`; `envs.X` defaults are shown as
     the expression (the real default depends on the environment).
  4. --model is skipped by the real parser only when argv contains both "serve" and "--help"
     (arg_utils.py:965); the table lists it with that note.
  5. Type hints the resolver cannot follow (classes defined in modules that need torch, dynamic
     aliases) count as "non-builtin class", which _compute_kwargs also maps to `str`; a dict
     subclass that cannot be resolved would be wrong (shown as str instead of JSON).
  6. Help text: only the first paragraph of the field docstring or the explicit help= string.
     Additions made by `["help"] += ...` and the JSON tips are kept in JSON as notes, not shown.
  7. Python version: on Python < 3.13 FlexibleArgumentParser handles `deprecated=` itself; no flag
     here uses it, so "deprecated" is taken from the help text.
  8. Order: groups are in registration order; inside a group, flags are sorted by option strings,
     as SortedHelpFormatter (argparse_utils.py:115) does in `--help`.

Cross-check: docs/cli/serve.md is generated at docs build time by
docs/mkdocs/gen_files/generate_argparse.py (it mocks torch and calls make_arg_parser), so the
repo has no static flag list to compare with; docs/configuration/serve_args.md only links to it.

Usage:
  python3 tools/gen_serve_flags.py            # write JSON (and the page region if the markers exist)
  python3 tools/gen_serve_flags.py --check    # exit 1 if the JSON or the page region is out of date
  python3 tools/gen_serve_flags.py --report   # also print a summary
"""
from __future__ import annotations

import ast
import copy
import inspect
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _vllm_ast as va  # noqa: E402

NAME = "serve-flags"
PAGE_ID = "reference/serve-flags"
CLI_ARGS = "vllm/entrypoints/common/cli_args.py"
ARG_UTILS = "vllm/engine/arg_utils.py"
HELP_LIMIT = 400

BUILTINS = {"int", "str", "bool", "float", "dict", "list", "tuple", "set", "bytes", "object"}
GENERIC_ALIASES = {"List": "list", "Dict": "dict", "Tuple": "tuple", "Set": "set", "FrozenSet": "set",
                   "Sequence": "other", "Mapping": "other"}
HUMAN_READABLE_INT_ARGS = {"max_num_batched_tokens", "max_num_scheduled_tokens", "kv_cache_memory_bytes",
                           "safetensors_prefetch_block_size", "max_num_queued_tokens"}


# ---------------------------------------------------------------------------
# Type hints -> a small model of what typing.get_args() would give
# ---------------------------------------------------------------------------
class Hint:
    """kind: none | builtin | generic | literal | dataclass | dictsub | other."""

    def __init__(self, kind: str, name: str = "", args: list | None = None, values: list | None = None):
        self.kind, self.name, self.args, self.values = kind, name, args or [], values or []

    def key(self):
        return (self.kind, self.name, tuple(a.key() if isinstance(a, Hint) else str(a) for a in self.args),
                tuple(map(repr, self.values)))

    def __eq__(self, o):
        return isinstance(o, Hint) and self.key() == o.key()

    def __hash__(self):
        return hash(self.key())

    def show(self) -> str:
        if self.kind == "none":
            return "None"
        if self.kind == "literal":
            return "Literal[" + ", ".join(map(repr, self.values)) + "]"
        if self.kind == "union":
            return " | ".join(a.show() for a in self.args)
        if self.kind == "generic":
            return f"{self.name}[" + ", ".join(a.show() if isinstance(a, Hint) else str(a) for a in self.args) + "]"
        return self.name


def _sub_args(node: ast.Subscript) -> list[ast.AST]:
    s = node.slice
    return list(s.elts) if isinstance(s, ast.Tuple) else [s]


def hint_of(mod: va.Module, node: ast.AST, depth: int = 0) -> list[Hint]:
    """Flatten a type annotation into the set of hints that get_type_hints() would give."""
    if depth > 12:
        return [Hint("other", "?")]
    if isinstance(node, ast.Constant):
        if node.value is None:
            return [Hint("none")]
        if isinstance(node.value, str):          # forward reference
            try:
                return hint_of(mod, ast.parse(node.value, mode="eval").body, depth + 1)
            except SyntaxError:
                return [Hint("other", node.value)]
        return [Hint("other", repr(node.value))]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return hint_of(mod, node.left, depth + 1) + hint_of(mod, node.right, depth + 1)
    if isinstance(node, ast.Subscript):
        base = node.value
        bname = base.id if isinstance(base, ast.Name) else (base.attr if isinstance(base, ast.Attribute) else "")
        args = _sub_args(node)
        if bname == "Optional":
            return hint_of(mod, args[0], depth + 1) + [Hint("none")]
        if bname == "Union":
            return [h for a in args for h in hint_of(mod, a, depth + 1)]
        if bname in ("Annotated", "SkipValidation", "InitVar", "ClassVar"):
            return hint_of(mod, args[0], depth + 1)
        if bname == "Literal":
            return [Hint("literal", values=literal_values(mod, args, depth))]
        if bname in BUILTINS or bname in GENERIC_ALIASES:
            name = GENERIC_ALIASES.get(bname, bname)
            if name == "other":
                return [Hint("other", bname)]
            inner = []
            for a in args:
                if isinstance(a, ast.Constant) and a.value is Ellipsis:
                    inner.append("...")
                else:
                    hs = hint_of(mod, a, depth + 1)
                    inner.append(hs[0] if len(hs) == 1 else Hint("union", args=hs))
            return [Hint("generic", name, inner)]
        if bname == "type":
            return [Hint("other", "type[...]")]
        return [Hint("other", mod.src(node))]
    if isinstance(node, ast.Name):
        n = node.id
        if n in BUILTINS:
            return [Hint("builtin", n)]
        if n in ("Any", "Callable"):
            return [Hint("other", n)]
        r = va.resolve(mod, n)
        if r is None:
            return [Hint("other", n)]
        m2, target = r
        if isinstance(target, ast.ClassDef):
            return [class_hint(m2, target)]
        if isinstance(target, (ast.Subscript, ast.BinOp, ast.Name, ast.Constant, ast.Attribute)):
            if isinstance(target, ast.Call):
                return [Hint("other", n)]
            return hint_of(m2, target, depth + 1)
        if isinstance(target, ast.Call):      # NewType / TypeVar / TypeAliasType
            return [Hint("other", n)]
        return [Hint("other", n)]
    if isinstance(node, ast.Attribute):
        return [Hint("other", mod.src(node))]
    return [Hint("other", mod.src(node))]


def literal_values(mod: va.Module, args: list[ast.AST], depth: int) -> list:
    vals: list = []
    for a in args:
        if isinstance(a, ast.Constant):
            vals.append(a.value)
            continue
        # nested Literal alias (Literal["auto", RunnerType]) or Enum member
        hs = hint_of(mod, a, depth + 1)
        for h in hs:
            if h.kind == "literal":
                vals.extend(h.values)
            else:
                vals.append(f"<{mod.src(a)}>")
    out = []
    for v in vals:                      # typing de-duplicates Literal args
        if v not in out:
            out.append(v)
    return out


def class_hint(mod: va.Module, cls: ast.ClassDef) -> Hint:
    decos = va.decorator_names(cls)
    if "config" in decos or "dataclass" in decos:
        return Hint("dataclass", cls.name)
    bases = [mod.src(b) for b in cls.bases]
    if any(b == "TypedDict" or b.endswith(".TypedDict") for b in bases):
        return Hint("other", cls.name)
    if any(b == "dict" or b.startswith("dict[") or b.startswith("Dict") for b in bases):
        return Hint("dictsub", cls.name)
    return Hint("other", cls.name)


# ---------------------------------------------------------------------------
# Config-class fields and _compute_kwargs()
# ---------------------------------------------------------------------------
class Field:
    def __init__(self, mod: va.Module, cls_name: str, node: ast.AnnAssign, doc: str):
        self.mod, self.cls_name, self.node, self.doc = mod, cls_name, node, doc
        self.name = node.target.id
        self.line = node.lineno
        self.file = mod.rel


def class_fields(mod: va.Module, cls: ast.ClassDef, depth: int = 0) -> dict[str, Field]:
    """Dataclass fields in MRO order (bases first); docs follow get_attr_docs (subclass wins)."""
    out: dict[str, Field] = {}
    for b in cls.bases:
        if isinstance(b, ast.Name) and depth < 6:
            r = va.find_class(b.id, mod)
            if r:
                for k, v in class_fields(r[0], r[1], depth + 1).items():
                    out[k] = v
    for i, node in enumerate(cls.body):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            ann = mod.src(node.annotation)
            if re.match(r"(typing\.)?ClassVar\b|(dataclasses\.)?InitVar\b", ann):
                continue
            d = va.docstring_after(cls.body, i)
            doc = inspect.cleandoc(d.value) if d else ""
            prev = out.get(node.target.id)
            if not doc and prev:
                doc = prev.doc
            out[node.target.id] = Field(mod, cls.name, node, doc)
    return out


def lit(v) -> tuple:
    return ("lit", v)


def src(s: str) -> tuple:
    return ("src", s)


def show_val(v: tuple | None) -> str:
    if v is None:
        return ""
    kind, x = v
    if kind == "lit":
        return repr(x) if not isinstance(x, str) else repr(x)
    return x


def const_value(mod: va.Module, node: ast.AST, depth: int = 0):
    """Evaluate a literal, following module-level names. Raises ValueError if not literal."""
    if depth > 6:
        raise ValueError
    if isinstance(node, ast.Name):
        r = va.resolve(mod, node.id)
        if r and not isinstance(r[1], (ast.ClassDef, ast.FunctionDef, ast.Module)):
            return const_value(r[0], r[1], depth + 1)
        raise ValueError
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        r = va.find_class(node.value.id, mod)
        if r:
            cm, cls = r
            for st in cls.body:
                if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == node.attr for t in st.targets):
                    return const_value(cm, st.value, depth + 1)
                if (isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.target.id == node.attr
                        and st.value is not None):
                    return const_value(cm, st.value, depth + 1)
        raise ValueError
    return ast.literal_eval(node)


def field_default(f: Field) -> tuple | None:
    """What _compute_kwargs puts in "default" (as ('lit', v) or ('src', text)); None = no default."""
    v = f.node.value
    mod = f.mod
    if v is None:
        return None
    if isinstance(v, ast.Call) and isinstance(v.func, (ast.Name, ast.Attribute)):
        fname = v.func.id if isinstance(v.func, ast.Name) else v.func.attr
        if fname in ("Field", "field"):
            kws = {k.arg: k.value for k in v.keywords if k.arg}
            if "default" in kws:
                return expr_default(mod, kws["default"])
            if "default_factory" in kws:
                fac = kws["default_factory"]
                if isinstance(fac, ast.Name) and fac.id in ("list", "dict", "set", "tuple"):
                    return lit({"list": [], "dict": {}, "set": set(), "tuple": ()}[fac.id])
                if isinstance(fac, ast.Lambda):
                    return expr_default(mod, fac.body)
                return src(f"{mod.src(fac)}()")
            if v.args:
                return expr_default(mod, v.args[0])
            return None
    return expr_default(mod, v)


def expr_default(mod: va.Module, node: ast.AST) -> tuple:
    try:
        v = const_value(mod, node)
        if isinstance(node, (ast.Name, ast.Attribute)):      # a named constant: show both
            return src(f"{mod.src(node)} (= {v!r})")
        return lit(v)
    except (ValueError, SyntaxError, TypeError):
        return src(" ".join(mod.src(node).split()))


def compute_kwargs(mod: va.Module, cls: ast.ClassDef) -> dict[str, dict]:
    """Reproduce _compute_kwargs (arg_utils.py:325) for each field of `cls`."""
    out = {}
    for name, f in class_fields(mod, cls).items():
        hints = hint_of(f.mod, f.node.annotation)
        hints = [Hint("builtin", "dict") if h.kind == "dictsub" else h for h in hints]
        uniq: list[Hint] = []
        for h in hints:
            if h not in uniq:
                uniq.append(h)
        hints = uniq
        kinds = {(h.kind, h.name) for h in hints}
        has = lambda n: any((h.kind == "builtin" and h.name == n) or (h.kind == "generic" and h.name == n) for h in hints)  # noqa: E731
        has_none = any(h.kind == "none" for h in hints)
        has_literal = any(h.kind == "literal" for h in hints)
        nonbuiltin = any(h.kind in ("other", "dataclass") for h in hints)
        dc = next((h for h in hints if h.kind == "dataclass"), None)
        kw: dict = {"default": field_default(f), "help": lit(f.doc.replace("%", "%%")),
                    "_field": f, "_hints": " | ".join(h.show() for h in hints), "_notes": []}
        if f.node.value is None:
            kw["_notes"].append("field has no default")
        if dc is not None:
            kw["type"] = src(f"JSON -> {dc.name}")
            kw["_dataclass"] = dc.name
        elif kinds == {("builtin", "bool"), ("builtin", "str"), ("none", "")}:
            kw["type"] = src("str")
            kw["nargs"] = lit("?")
            kw["const"] = lit(True)
        elif has("bool"):
            kw["action"] = src("BooleanOptionalAction")
        elif has_literal:
            kw.update(literal_kwargs(hints))
        elif has("tuple") or has("list") or has("set"):
            cname = next(n for n in ("tuple", "list", "set") if has(n))
            coll = next(h for h in hints if h.name == cname)
            kw.update(collection_kwargs(coll))
        elif has("int"):
            if name == "max_model_len":
                kw["type"] = src("human_readable_int_or_auto")
            elif name in HUMAN_READABLE_INT_ARGS:
                kw["type"] = src("human_readable_int")
            else:
                kw["type"] = src("int")
        elif has("float"):
            kw["type"] = src("float")
        elif has("dict") and (has("str") or nonbuiltin):
            kw["type"] = src("union_dict_and_str")
        elif has("dict"):
            kw["type"] = src("json.loads")
        elif has("str") or nonbuiltin:
            kw["type"] = src("str")
        else:
            kw["type"] = src(f"? unsupported {kw['_hints']}")
            kw["_notes"].append("unsupported type: _compute_kwargs would raise ValueError")
        if has_none and not has("bool"):
            kw["_optional"] = True
            if kw.get("choices"):
                kw["choices"] = lit(list(kw["choices"][1]) + [None])
        out[name] = kw
    return out


def literal_kwargs(hints: list[Hint]) -> dict:
    lh = next(h for h in hints if h.kind == "literal")
    vals = lh.values
    try:
        vals = sorted(vals)
    except TypeError:
        pass
    has_str = any(h.kind == "builtin" and h.name == "str" for h in hints)
    t = type(lh.values[0]).__name__ if lh.values else "str"
    return {"type": src(t), ("metavar" if has_str else "choices"): lit(vals)}


def collection_kwargs(h: Hint) -> dict:
    args = h.args
    elems = [a for a in args if a != "..."]
    elem = elems[0] if elems else Hint("builtin", "str")
    if isinstance(elem, Hint) and elem.kind == "union":
        elem = Hint("builtin", "str")
    nargs = "+" if h.name != "tuple" or "..." in args else len(args)
    out = {"nargs": lit(nargs)}
    if isinstance(elem, Hint) and elem.kind == "literal":
        out.update(literal_kwargs([elem]))
    else:
        out["type"] = src(elem.show() if isinstance(elem, Hint) else str(elem))
    return out


# ---------------------------------------------------------------------------
# Interpreter for add_cli_args() / make_arg_parser()
# ---------------------------------------------------------------------------
class KwargsDict(dict):
    """get_kwargs(Cls) result."""
    cls_name = ""


class Group:
    def __init__(self, title: str, line: int):
        self.title, self.line = title, line


class Interp:
    def __init__(self):
        self.flags: list[dict] = []
        self.groups: list[str] = []
        self.notes: list[str] = []

    # -- helpers --
    def get_kwargs(self, mod: va.Module, cls_name: str) -> KwargsDict:
        r = va.find_class(cls_name, mod)
        if r is None:
            raise SystemExit(f"cannot find class {cls_name}")
        kd = KwargsDict(copy.copy(compute_kwargs(r[0], r[1])))
        for k in kd:
            kd[k] = dict(kd[k])
        kd.cls_name = cls_name
        return kd

    @staticmethod
    def _is_get_kwargs(node) -> str | None:
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_kwargs"
                and node.args and isinstance(node.args[0], ast.Name)):
            return node.args[0].id
        return None

    def lookup(self, mod: va.Module, node, env: dict):
        """Evaluate X, X["f"], get_kwargs(C)["f"] to a kwargs dict (shared, like Python)."""
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Subscript):
            key = va.str_const(node.slice)
            base = self._is_get_kwargs(node.value)
            if base is not None:
                kd = self.get_kwargs(mod, base)
                return kd.get(key)
            obj = self.lookup(mod, node.value, env)
            if isinstance(obj, dict) and key is not None:
                return obj.get(key)
        return None

    def value(self, mod: va.Module, node) -> tuple:
        if isinstance(node, ast.Attribute) and mod.src(node) == "argparse.SUPPRESS":
            return src("argparse.SUPPRESS")
        if isinstance(node, ast.Attribute) and mod.src(node) == "argparse.BooleanOptionalAction":
            return src("BooleanOptionalAction")
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "get_args"
                and node.args):
            hs = hint_of(mod, node.args[0])
            if len(hs) == 1 and hs[0].kind == "literal":
                return lit(list(hs[0].values))
        try:
            return lit(const_value(mod, node))
        except (ValueError, SyntaxError, TypeError):
            pass
        if isinstance(node, (ast.Call, ast.BinOp, ast.JoinedStr)):
            s = self.string(mod, node)
            if s is not None:
                return lit(s)
        return src(" ".join(mod.src(node).split()))

    def string(self, mod: va.Module, node) -> str | None:
        """Concatenated string literals / module constants (help texts)."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = self.string(mod, node.left), self.string(mod, node.right)
            return a + b if a is not None and b is not None else None
        if isinstance(node, ast.Name):
            try:
                v = const_value(mod, node)
                return v if isinstance(v, str) else None
            except (ValueError, SyntaxError, TypeError):
                return None
        return None

    # -- statements --
    def run(self, mod: va.Module, body: list[ast.stmt], env: dict, cond: list[str]) -> None:
        for st in body:
            self.stmt(mod, st, env, cond)

    def stmt(self, mod: va.Module, st: ast.stmt, env: dict, cond: list[str]) -> None:
        if isinstance(st, ast.If):
            known = self.eval_cond(st.test, env)
            if known is not None:
                self.run(mod, st.body if known else st.orelse, env, cond)
                return
            c = " ".join(mod.src(st.test).split())
            self.run(mod, st.body, env, cond + [c])
            if st.orelse:
                self.run(mod, st.orelse, env, cond + [f"not ({c})"])
            return
        if isinstance(st, (ast.Assign, ast.AnnAssign)):
            targets = st.targets if isinstance(st, ast.Assign) else [st.target]
            val = st.value
            for t in targets:
                if isinstance(t, ast.Name):
                    self.assign_name(mod, t.id, val, env, cond)
                elif isinstance(t, ast.Subscript):
                    obj = self.lookup(mod, t.value, env)
                    key = va.str_const(t.slice)
                    if isinstance(obj, dict) and key is not None:
                        obj[key] = self.value(mod, val)
                        if key == "type":
                            obj.pop("_dataclass", None)
                            obj.pop("_optional", None)
            return
        if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Subscript):
            obj = self.lookup(mod, st.target.value, env)
            key = va.str_const(st.target.slice)
            if isinstance(obj, dict) and key == "help":
                obj.setdefault("_notes", []).append("help extended: " + " ".join(mod.src(st.value).split())[:120])
            return
        if isinstance(st, ast.Delete):
            for t in st.targets:
                if isinstance(t, ast.Subscript):
                    obj = self.lookup(mod, t.value, env)
                    key = va.str_const(t.slice)
                    if isinstance(obj, dict) and key in obj:
                        del obj[key]
            return
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
            self.call(mod, st.value, env, cond)
            return
        if isinstance(st, ast.Return):
            if isinstance(st.value, ast.Call):
                self.call(mod, st.value, env, cond)
            return
        if isinstance(st, (ast.ImportFrom, ast.Import, ast.Pass)):
            return
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):   # docstring
            return
        self.notes.append(f"{mod.rel}:{st.lineno}: statement not interpreted: {type(st).__name__}")

    @staticmethod
    def eval_cond(test, env):
        """True/False for `name` / `not name` when name is a parameter with a known default."""
        neg = False
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            neg, test = True, test.operand
        if isinstance(test, ast.Name) and isinstance(env.get(test.id), tuple) and env[test.id][0] == "const":
            return (not env[test.id][1]) if neg else bool(env[test.id][1])
        return None

    def assign_name(self, mod, name, val, env, cond):
        cls = self._is_get_kwargs(val)
        if cls is not None:
            env[name] = self.get_kwargs(mod, cls)
            return
        if isinstance(val, ast.Subscript):
            obj = self.lookup(mod, val, env)
            if obj is not None:
                env[name] = obj
                return
        if isinstance(val, ast.Call):
            res = self.call(mod, val, env, cond)
            if res is not None:
                env[name] = res
            return
        env.pop(name, None)

    def call(self, mod, call: ast.Call, env, cond):
        f = call.func
        if isinstance(f, ast.Attribute):
            meth = f.attr
            recv = f.value
            if meth == "add_argument_group":
                title = None
                for k in call.keywords:
                    if k.arg == "title":
                        title = va.str_const(k.value)
                if title is None and call.args:
                    title = va.str_const(call.args[0])
                g = Group(title or "?", call.lineno)
                if g.title not in self.groups:
                    self.groups.append(g.title)
                return g
            if meth == "add_argument":
                grp = env.get(recv.id) if isinstance(recv, ast.Name) else None
                self.add_argument(mod, call, grp if isinstance(grp, Group) else None, env, cond)
                return None
            if meth == "remove" and isinstance(recv, ast.Subscript):
                obj = self.lookup(mod, recv.value, env)
                key = va.str_const(recv.slice)
                if isinstance(obj, dict) and key in obj and obj[key][0] == "lit":
                    try:
                        rm = const_value(mod, call.args[0])
                        obj[key] = lit([x for x in obj[key][1] if x != rm])
                        obj.setdefault("_notes", []).append(f"choice {rm!r} removed")
                    except (ValueError, SyntaxError, TypeError):
                        obj.setdefault("_notes", []).append(f"choice {mod.src(call.args[0])} removed")
                return None
            if meth == "add_cli_args" and isinstance(recv, ast.Name):
                self.class_add_cli_args(mod, recv.id, call, env, cond)
                return env.get("parser")
            if meth in ("pre_register_and_update",):
                self.notes.append(f"{mod.rel}:{call.lineno}: {mod.src(call)} may change the parser at run time "
                                  "(not reproduced)")
                return None
            if meth == "pop":
                return None
        if isinstance(f, ast.Name) and f.id == "load_general_plugins":
            self.notes.append(f"{mod.rel}:{call.lineno}: load_general_plugins() may add arguments (not reproduced)")
            return None
        return None

    def class_add_cli_args(self, mod, cls_name, call, env, cond):
        r = va.find_class(cls_name, mod)
        if r is None:
            raise SystemExit(f"cannot find {cls_name}")
        cm, cls = r
        # find add_cli_args in the class or its bases
        chain = [(cm, cls)]
        while True:
            m, c = chain[-1]
            if not c.bases or not isinstance(c.bases[0], ast.Name):
                break
            rb = va.find_class(c.bases[0].id, m)
            if rb is None:
                break
            chain.append(rb)
        owner = next(((m, c, fn) for m, c in chain for fn in c.body
                      if isinstance(fn, ast.FunctionDef) and fn.name == "add_cli_args"), None)
        if owner is None:
            raise SystemExit(f"{cls_name}.add_cli_args not found")
        om, oc, fn = owner
        if any(isinstance(fn2, ast.FunctionDef) and fn2.name == "_customize_cli_kwargs" for fn2 in oc.body):
            self.dataclass_flags(cm, cls, chain, fn, om)
        else:
            sub_env: dict = {"parser": "parser"}
            params = fn.args.args
            for a, d in zip(params[len(params) - len(fn.args.defaults):], fn.args.defaults):
                try:
                    sub_env[a.arg] = ("const", ast.literal_eval(d))
                except ValueError:
                    pass
            self.run(om, fn.body, sub_env, cond)

    def dataclass_flags(self, cm, cls, chain, fn, om):
        """BaseFrontendArgs.add_cli_args: one flag per field, after _customize_cli_kwargs()."""
        kd = self.get_kwargs(cm, cls.name)
        # _customize_cli_kwargs chain: base first (each override calls super() first)
        for m, c in reversed(chain):
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and f.name == "_customize_cli_kwargs":
                    pname = f.args.args[1].arg
                    body = [s for s in f.body if not (isinstance(s, ast.Assign) and "super()" in m.src(s.value))]
                    self.run(m, body, {pname: kd}, [])
        title = cls.name.replace("Args", "")
        if title not in self.groups:
            self.groups.append(title)
        loop = next((n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and n.func.attr == "add_argument"), None)
        for key, kw in kd.items():
            flag = "--" + key.replace("_", "-")
            self.register([flag], kw, {}, Group(title, 0), om, loop.lineno if loop else fn.lineno, [], positional=False)

    def add_argument(self, mod, call: ast.Call, grp: Group | None, env, cond):
        names = [va.str_const(a) for a in call.args]
        if not names or any(n is None for n in names):
            self.notes.append(f"{mod.rel}:{call.lineno}: add_argument with non-literal names")
            return
        base: dict = {}
        explicit: dict = {}
        for k in call.keywords:
            if k.arg is None:
                v = k.value
                if isinstance(v, ast.Dict):
                    for dk, dv in zip(v.keys, v.values):
                        if dk is None:
                            b = self.lookup(mod, dv, env)
                            if isinstance(b, dict):
                                base = dict(b)
                        else:
                            explicit[va.str_const(dk)] = self.value(mod, dv)
                else:
                    b = self.lookup(mod, v, env)
                    if isinstance(b, dict):
                        base = dict(b)
                    else:
                        self.notes.append(f"{mod.rel}:{call.lineno}: cannot resolve **{mod.src(v)}")
            else:
                explicit[k.arg] = self.value(mod, k.value)
        positional = not names[0].startswith("-")
        title = grp.title if grp else ("positional arguments" if positional else "options")
        if grp is None and title not in self.groups:
            self.groups.append(title)
        name_line = call.args[0].lineno
        self.register(names, base, explicit, Group(title, 0), mod, name_line, cond, positional)

    def register(self, names, base, explicit, grp, mod, line, cond, positional):
        kw = {k: v for k, v in base.items()}
        notes = list(kw.pop("_notes", []))
        fld: Field | None = kw.pop("_field", None)
        hints = kw.pop("_hints", "")
        optional = kw.pop("_optional", False)
        dc = kw.pop("_dataclass", None)
        kw.update(explicit)
        if "type" in explicit or "action" in explicit:
            dc = None
            if "type" in explicit:
                optional = False
        action = show_val(kw.get("action")).strip("'\"") if kw.get("action") else ""
        long_names = [n for n in names if n.startswith("--")]
        visible = list(names)
        if action in ("BooleanOptionalAction", "argparse.BooleanOptionalAction"):
            inverses = []
            for n in names:
                for pre, inv in (("--enable-", "--disable-"), ("--disable-", "--enable-")):
                    if n.startswith(pre):
                        inverses.append(inv + n[len(pre):])
            visible = names + inverses if inverses else names + ["--no-" + n[2:] for n in long_names]
        dest = show_val(explicit["dest"]).strip("'\"") if "dest" in explicit else (
            (long_names[0][2:] if long_names else names[0].lstrip("-")).replace("-", "_"))
        default = kw.get("default")
        if default is None and "default" not in kw:
            default_text = "False" if action == "store_true" else "None"
        elif default is None:
            default_text = "(field has no default)"
        else:
            default_text = show_val(default)
        help_full = ""
        if "help" in kw:
            h = kw["help"]
            help_full = h[1] if h[0] == "lit" else h[1]
            if isinstance(help_full, str):
                help_full = help_full.replace("%%", "%")
        typ = self.type_text(kw, action, optional, dc)
        choices = kw.get("choices")
        metavar = kw.get("metavar")
        deprecated = bool(re.search(r"\b(is|are) deprecated\b|^deprecated\b|will be removed in",
                                    help_full or "", re.I)) or ("deprecated" in explicit)
        flag = {
            "flags": visible,
            "registered": names,
            "dest": dest,
            "group": grp.title,
            "type": typ,
            "default": default_text,
            "choices": [("None" if c is None else str(c)) for c in choices[1]] if choices and choices[0] == "lit"
                       else ([show_val(choices)] if choices else None),
            "metavar": ([str(c) for c in metavar[1]] if metavar and metavar[0] == "lit" and isinstance(metavar[1], list)
                        else (str(metavar[1]) if metavar else None)),
            "help": va.first_paragraph(help_full, HELP_LIMIT) if help_full else "",
            "deprecated": deprecated,
            "file": mod.rel,
            "line": line,
            "field": None,
            "condition": " and ".join(cond) if cond else "",
            "notes": notes,
        }
        if fld is not None:
            flag["field"] = {"class": fld.cls_name, "name": fld.name, "file": fld.file, "line": fld.line,
                             "hint": hints}
        self.flags.append(flag)

    @staticmethod
    def type_text(kw, action, optional, dc) -> str:
        nargs = kw.get("nargs")
        if action in ("BooleanOptionalAction", "argparse.BooleanOptionalAction"):
            return "bool"
        if action:
            a = action.strip("'\"")
            if a == "store_true":
                return "flag (store_true)"
            if a == "append":
                t = show_val(kw.get("type")) if kw.get("type") else "str"
                return f"{t}, repeatable (append)"
            if a == "LoRAParserAction":
                return "list of name=path or JSON (LoRAParserAction)"
            if a not in ("store",):
                return f"custom action {a}"
        t = kw.get("type")
        tt = show_val(t) if t else "str"
        names = {"json.loads": "JSON", "parse_type(json.loads)": "JSON", "optional_type(json.loads)": "JSON",
                 "union_dict_and_str": "str or JSON dict", "human_readable_int": "int (accepts 1k, 2M, 1G)",
                 "human_readable_int_or_auto": "int or auto (accepts 1k, 2M; -1 = auto)",
                 "parse_reasoning_parser": "str (none = unset)", "optional_type(str)": "str"}
        if dc:
            tt = f"JSON ({dc})"
        elif tt.startswith("lambda") or "lambda" in tt:
            tt = "custom (lambda)"
        else:
            tt = names.get(tt.strip("'\""), tt.strip("'\""))
        if nargs:
            n = show_val(nargs)
            if n == "'+'":
                tt = f"list of {tt} (one or more)"
            elif n == "'?'":
                tt = f"{tt}; bare flag = {show_val(kw.get('const'))}" if kw.get("const") else f"{tt} (optional)"
            else:
                tt = f"{n} × {tt}"
        if optional:
            tt += " or None"
        return tt


def build() -> tuple[dict, str, list[str]]:
    it = Interp()
    mod = va.module(CLI_ARGS)
    fn = mod.functions.get("make_arg_parser")
    if fn is None:
        raise SystemExit("make_arg_parser not found")
    it.run(mod, fn.body, {"parser": "parser"}, [])
    flags = it.flags
    errors: list[str] = []
    cites: list[tuple[str, int, str]] = []
    for f in flags:
        tok = f["registered"][0] if f["file"] != CLI_ARGS or f["field"] is None else "add_argument"
        cites.append((f["file"], f["line"], tok))
        if f["field"]:
            cites.append((f["field"]["file"], f["field"]["line"], f["field"]["name"]))
    errors += va.verify_cites(cites)
    seen: dict[str, str] = {}
    for f in flags:
        for n in f["registered"]:
            if n in seen:
                errors.append(f"flag {n} registered twice ({seen[n]} and {f['file']}:{f['line']})")
            seen[n] = f"{f['file']}:{f['line']}"
    order = it.groups
    flags.sort(key=lambda f: (order.index(f["group"]), f["registered"]))
    au = va.module(ARG_UTILS)
    ea = au.classes["EngineArgs"]
    dests = {f["dest"] for f in flags}
    ea_fields = [n.target.id for n in au.classes["AsyncEngineArgs"].body + ea.body
                 if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
    ea_line = next(f.lineno for f in ea.body if isinstance(f, ast.FunctionDef) and f.name == "add_cli_args")
    data = {
        "source": f"{CLI_ARGS} make_arg_parser",
        "make_arg_parser_line": fn.lineno,
        "engine_args_add_cli_args_line": ea_line,
        "count": len(flags),
        "groups": {g: sum(1 for f in flags if f["group"] == g) for g in order if any(f["group"] == g for f in flags)},
        "not_reproduced": it.notes,
        "engine_args_fields_without_flag": sorted(set(ea_fields) - dests),
        "flags": flags,
    }
    return data, render(data), errors


def render(data: dict) -> str:
    flags = data["flags"]
    out = [f"<p>{data['count']} arguments of <code>vllm serve</code> in {len(data['groups'])} groups, rebuilt "
           f"from {va.cite(CLI_ARGS, data['make_arg_parser_line'], label='<code>make_arg_parser</code>')} and "
           f"{va.cite(ARG_UTILS, data['engine_args_add_cli_args_line'], label='<code>EngineArgs.add_cli_args</code>')}. "
           f"Inside a group the flags are sorted as in <code>vllm serve --help</code>.</p>"]
    for g, n in data["groups"].items():
        out.append(f'<h3 id="flags-{va.slug(g)}">{va.esc(g)} ({n})</h3>')
        out.append("<table>")
        out.append("  <thead><tr><th>Flag</th><th>Type and default</th><th>Help (first paragraph)</th>"
                   "<th>Source</th></tr></thead>")
        out.append("  <tbody>")
        for f in (f for f in flags if f["group"] == g):
            names = " ".join(va.code(n) for n in f["flags"])
            if f["deprecated"]:
                names += " <em>(deprecated)</em>"
            td = va.esc(f["type"]) + "<br>default " + (va.code(f["default"]) if f["default"] else "—")
            if f["choices"]:
                td += "<br>choices: " + ", ".join(va.code(c) for c in f["choices"])
            elif f["metavar"]:
                mv = f["metavar"] if isinstance(f["metavar"], list) else [f["metavar"]]
                td += "<br>suggested: " + ", ".join(va.code(c) for c in mv)
            help_ = va.esc(f["help"]) if f["help"] else "<em>(no help text)</em>"
            if f["condition"]:
                help_ += f"<br><em>Registered only if</em> {va.code(f['condition'])}."
            srcs = [va.cite(f["file"], f["line"], label="<code>add_argument</code>")]
            if f["field"]:
                fl = f["field"]
                srcs.append(va.cite(fl["file"], fl["line"], label=f"<code>{fl['class']}.{fl['name']}</code>"))
            out.append(f"    <tr><td>{names}</td><td>{td}</td><td>{help_}</td><td>{'<br>'.join(srcs)}</td></tr>")
        out.append("  </tbody>")
        out.append("</table>")
    return "\n".join(out)


def report(data: dict) -> None:
    print(f"{data['count']} flags in {len(data['groups'])} groups")
    for g, n in data["groups"].items():
        print(f"  {n:4d}  {g}")
    print("\nnot reproduced:")
    for n in data["not_reproduced"]:
        print("  " + n)
    print("\nEngineArgs fields with no flag:", ", ".join(data["engine_args_fields_without_flag"]))
    print("\ndeprecated:", ", ".join(f["flags"][0] for f in data["flags"] if f["deprecated"]))
    print("conditional:", ", ".join(f"{f['flags'][0]} [{f['condition']}]" for f in data["flags"] if f["condition"]))
    print("no help:", ", ".join(f["flags"][0] for f in data["flags"] if not f["help"]))
    print("no default from field (explicit add_argument):",
          ", ".join(f["flags"][0] for f in data["flags"] if not f["field"]))
    odd = [f for f in data["flags"] if "?" in f["type"] or "unsupported" in f["type"]]
    print("unclassified types:", ", ".join(f"{f['flags'][0]}={f['type']}" for f in odd) or "-")
    print("enable/disable pairs:", ", ".join("/".join(f["flags"]) for f in data["flags"]
                                             if any(x.startswith("--disable-") for x in f["flags"])
                                             and any(x.startswith("--enable-") for x in f["flags"])))


if __name__ == "__main__":
    sys.exit(va.run(NAME, PAGE_ID, build, report))
