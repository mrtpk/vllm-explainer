"""Python entry points: package metadata -> EntryPoint -> load().

Used by: src/learn/python-entry-points.html
Sources (at the pinned commit in explainer.json):
  pyproject.toml:42-47                              [project.scripts] vllm, [project.entry-points."vllm.general_plugins"]
  tests/plugins/vllm_add_dummy_model/setup.py:10-12 setup(entry_points={...}) in a test plugin
  vllm/plugins/__init__.py:36-74                     load_plugins_by_group(): entry_points(group=...), .name, .value, .load()
  vllm/plugins/__init__.py:77-90                     load_general_plugins(): call each loaded function
The entry_points.txt file format and the console-script wrapper are standard Python packaging
(not vLLM code); the paths are an example for a pip install into site-packages.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("From package metadata to a loaded plugin",
         "the vLLM wheel itself  ·  pip install  ·  importlib.metadata  ·  vllm/plugins/__init__.py",
         params='group "vllm.general_plugins"  ·  name "lora_filesystem_resolver"  ·  value "module:attr"')

BW = 300
X1, X2, X3 = 46, 400, 754
BH = 150

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 250
d.panel(y0, P1_H, 1, "Install time: where does an entry point live?",
        "The build tool copies the table from pyproject.toml into a small text file of the installed package.")
by = y0 + 72
d.box(X1, by, BW, BH, "pyproject.toml (source)", [
    "[project.scripts]",
    'vllm = "vllm.entrypoints.cli.',
    '        main:main"',
    '[project.entry-points.',
    '  "vllm.general_plugins"]',
    'lora_filesystem_resolver = "..."',
], col=BLUE, align="start")
d.box(X2, by, BW, BH, "pip install / build backend", [
    "builds the wheel",
    "writes the .dist-info folder",
    "makes a bin/vllm wrapper",
    "  for each console script",
], col=INK, align="start")
d.box(X3, by, BW, BH, "vllm-<ver>.dist-info/", [
    "entry_points.txt:",
    "[console_scripts]",
    "vllm = vllm.entrypoints.cli.",
    "         main:main",
    "[vllm.general_plugins]",
    "lora_filesystem_resolver = ...",
], col=EMERALD, align="start")
d.arrow(X1 + BW, by + 50, X2 - 2, by + 50, BLUE)
d.arrow(X2 + BW, by + 50, X3 - 2, by + 50, EMERALD)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 250
d.panel(y1, P2_H, 2, "Run time: how does vLLM find and call it?",
        "importlib.metadata reads every entry_points.txt on sys.path. load() imports the module and returns the object.")
by = y1 + 72
d.box(X1, by, BW, BH, "entry_points(group=...)", [
    'eps = entry_points(',
    '  group="vllm.general_plugins")',
    "reads metadata only:",
    "  nothing is imported yet",
    "returns EntryPoints (0..n)",
], col=EMERALD, align="start")
d.box(X2, by, BW, BH, "EntryPoint", [
    'ep.name  = "lora_filesystem_',
    '            resolver"',
    'ep.value = "vllm.plugins.lora_',
    '  resolvers.filesystem_resolver',
    '  :register_filesystem_resolver"',
    'ep.group = "vllm.general_plugins"',
], col=VIOLET, align="start")
d.box(X3, by, BW, BH, "ep.load() and the call", [
    "import_module(module part)",
    "getattr(module, attr part)",
    "-> a function object",
    "vLLM calls func()",
    "  (general plugins)",
], col=AMBER, align="start")
d.arrow(X1 + BW, by + 50, X2 - 2, by + 50, EMERALD)
d.arrow_label((X1 + BW + X2) / 2, by + 50, "for ep", EMERALD)
d.arrow(X2 + BW, by + 50, X3 - 2, by + 50, VIOLET)
d.arrow_label((X2 + BW + X3) / 2, by + 50, ".load()", VIOLET)

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "source declaration"), (EMERALD, "installed metadata"),
                 (VIOLET, "EntryPoint object"), (AMBER, "imported code")])
y += 40
y = d.footnotes(30, y, [
    "[project.scripts] becomes the group console_scripts. A setup.py writes the same file with setup(entry_points={...}).",
    "VLLM_PLUGINS filters by ep.name before load() (vllm/plugins/__init__.py:64). A failing load() is logged and skipped.",
])
d.save(height=y + 14)
