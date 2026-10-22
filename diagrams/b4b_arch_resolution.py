"""From an architecture name in config.json to a model class: registry entries, lookup order, inspection cache.

Used by: src/concepts/oot-models.html, src/code/model-executor-models-registry.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/model_executor/models/registry.py:1565-1573   built-in tables -> _LazyRegisteredModel entries
  vllm/model_executor/models/registry.py:1187-1195   register_model: "module:Class" -> lazy; nn.Module class -> eager
  vllm/model_executor/models/registry.py:1351-1402   inspect_model_cls: lookup order
  vllm/model_executor/models/registry.py:1325-1349   _normalize_arch
  vllm/model_executor/models/registry.py:1205-1236   _raise_for_unsupported
  vllm/model_executor/models/registry.py:996-1001    cache dir VLLM_CACHE_ROOT/modelinfos, file <module>-<Class>.json
  vllm/model_executor/models/registry.py:1004-1020   hash of the source file(s)
  vllm/model_executor/models/registry.py:1072-1118   cache hit / miss -> _run_in_subprocess
  vllm/model_executor/models/registry.py:1578-1619   subprocess: python -m ...registry, loads general plugins
  vllm/config/model.py:703-744                       front end inspects in ModelConfig.__post_init__
  vllm/model_executor/model_loader/utils.py:305-333  worker resolves (imports) the class
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("From an architecture name to a model class",
         "ModelRegistry  ·  the front end inspects the class, the worker imports it",
         params="vllm/model_executor/models/registry.py  ·  cache: $VLLM_CACHE_ROOT/modelinfos/")

LX = [40, 390, 740]
LW = 320

# ---- Panel 1: where entries come from ------------------------------------------------------
y1 = 120
P1_H = 196
d.panel(y1, P1_H, 1, "Where do registry entries come from?",
        "Built-in tables and string registrations make lazy entries. A class registration is eager.")
by = y1 + 70
BH = 106
d.box(LX[0], by, LW, BH, "Built-in tables (406 names)", [
    "_VLLM_MODELS  :806-817",
    "\"LlamaForCausalLM\":",
    "  (\"llama\", \"LlamaForCausalLM\")",
    "-> _LazyRegisteredModel  :1565",
], col=BLUE, align="start")
d.box(LX[1], by, LW, BH, "register_model(arch, \"m:C\")", [
    "split at \":\"  :1187-1191",
    "-> _LazyRegisteredModel",
    "module not imported now",
    "(the form the docstring advises)",
], col=BLUE, align="start")
d.box(LX[2], by, LW, BH, "register_model(arch, MyClass)", [
    "nn.Module subclass  :1194",
    "-> _RegisteredModel",
    "_ModelInfo computed at once,",
    "in this process  :975-979",
], col=PINK, align="start")

# ---- Panel 2: lookup order -----------------------------------------------------------------
y2 = y1 + P1_H + 18
P2_H = 196
d.panel(y2, P2_H, 2, "Which entry answers for config.json \"architectures\"?",
        "inspect_model_cls() and resolve_model_cls() use the same order (:1351-1456).")
cy = y2 + 70
CH = 108
d.step_card(LX[0], cy, LW, CH, 1, "model_impl first", VIOLET, tag="--model-impl", rows=[
    ("trans", "Transformers backend class"),
    ("terra", "\"Terratorch\""),
    ("auto", "no name registered and"),
    ("", "no --convert: Transformers"),
], key_w=46)
d.step_card(LX[1], cy, LW, CH, 2, "Each name in order", EMERALD, tag="registry", rows=[
    ("exact", "name in ModelRegistry.models"),
    ("else", "_normalize_arch: base name"),
    ("", "of a *ForSequenceClass.. etc."),
    ("first", "that inspects/loads wins"),
], key_w=46)
d.step_card(LX[2], cy, LW, CH, 3, "Nothing found", AMBER, tag="ValueError", rows=[
    ("auto", "Transformers fallback again"),
    ("old", "\"supported until vX\""),
    ("oot", "\"install the plugin at ...\""),
    ("else", "\"not supported for now\""),
], key_w=46)
for i in range(2):
    d.arrow(LX[i] + LW + 2, cy + 50, LX[i + 1] - 2, cy + 50, INK)

# ---- Panel 3: inspect a lazy entry ---------------------------------------------------------
y3 = y2 + P2_H + 18
P3_H = 196
d.panel(y3, P3_H, 3, "How does the front end inspect a lazy entry?",
        "From a JSON cache when the source hash matches. Otherwise in a new Python process.")
ey = y3 + 70
EH = 106
d.box(LX[0], ey, LW, EH, "Cache lookup", [
    "modelinfos/vllm-model_executor-",
    "  models-llama-LlamaForCausalLM.json",
    "hash = md5(\"llama.py\" + source)",
    "match -> _ModelInfo  :1086-1094",
], col=EMERALD, align="start")
d.box(LX[1], ey, LW, EH, "Miss: subprocess", [
    "python -m vllm...models.registry",
    "load_general_plugins()  :1612",
    "import module, from_model_cls",
    "result back by pickle file",
], col=AMBER, align="start")
d.box(LX[2], ey, LW, EH, "Later, in each worker", [
    "resolve_model_cls() imports",
    "the class  :1404-1456",
    "initialize_model(vllm_config=,",
    "  prefix=)  model_loader/utils.py",
], col=BLUE, align="start")
d.arrow(LX[0] + LW + 2, ey + 50, LX[1] - 2, ey + 50, AMBER)
d.arrow_label(LX[0] + LW + 15, ey + 44, "miss", AMBER)
d.arrow(LX[1] + LW + 2, ey + 50, LX[2] - 2, ey + 50, INK, dash="5 4")
d.arrow_label(LX[1] + LW + 15, ey + 44, "later", INK)

y = y3 + P3_H + 22
d.legend(30, y, [(BLUE, "lazy entry / import"), (PINK, "eager entry"), (VIOLET, "model_impl"),
                 (EMERALD, "registry or cache hit"), (AMBER, "slow or error path")])
y += 34
y = d.footnotes(30, y, [
    "Each step uses an lru_cache per process (:1125-1150), so one architecture is inspected at most once per process.",
    "For a package __init__.py, the hash covers every .py file of the package (:1005-1009). md5 falls back to sha256 under FIPS.",
])
d.save(height=y + 8)
