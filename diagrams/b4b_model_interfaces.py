"""Capability protocols: a model class with mixins -> _ModelInfo -> checks at start-up.

Used by: src/concepts/model-interfaces.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/model_executor/models/llama.py:464-482          LlamaForCausalLM bases and LoRA attributes
  vllm/model_executor/models/interfaces.py:586-589     supports_multimodal: getattr of a ClassVar flag
  vllm/model_executor/models/interfaces.py:760-795     supports_lora: helper protocol + warnings
  vllm/model_executor/models/interfaces.py:864-912     supports_pp: MRO/attributes + forward keyword
  vllm/model_executor/models/interfaces.py:1707-1710   supports_eagle: isinstance
  vllm/model_executor/models/interfaces_base.py:108-147  is_vllm_model / is_text_generation_model
  vllm/model_executor/models/registry.py:891-954       _ModelInfo fields and from_model_cls
  vllm/config/model.py:703-744, 1569-1576              runner check, PP check
  vllm/config/lora.py:137-143                          LoRA check
  vllm/v1/worker/lora_model_runner_mixin.py:49-50      worker LoRA check
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("From mixins to start-up checks",
         "A model class declares capabilities  ·  the registry records them  ·  the config checks them",
         params="interfaces.py  ·  interfaces_base.py  ·  registry.py:891-954 (_ModelInfo, 24 fields)")

LX = [40, 390, 740]
LW = 320

# ---- Panel 1: the class ---------------------------------------------------------------------
y1 = 120
P1_H = 196
d.panel(y1, P1_H, 1, "What does the model class declare?",
        "Base classes bring ClassVar flags. Methods and signatures are checked by name.")
by = y1 + 70
BH = 106
d.box(LX[0], by, LW, BH, "class LlamaForCausalLM(", [
    "  LocalArgmaxMixin, nn.Module,",
    "  SupportsLoRA, SupportsPP,",
    "  SupportsEagle, SupportsEagle3,",
    "  SupportsQuant)  llama.py:464-472",
], col=BLUE, align="start")
d.box(LX[1], by, LW, BH, "Flags it inherits", [
    "supports_lora = True",
    "supports_pp = True",
    "supports_eagle = True",
    "supports_eagle3 = True",
], col=BLUE, align="start")
d.box(LX[2], by, LW, BH, "Structure it has", [
    "__init__(*, vllm_config, prefix)",
    "embed_input_ids, forward(input_ids,",
    "  positions, intermediate_tensors)",
    "compute_logits, load_weights",
], col=BLUE, align="start")

# ---- Panel 2: the detection -----------------------------------------------------------------
y2 = y1 + P1_H + 18
P2_H = 196
d.panel(y2, P2_H, 2, "How does vLLM detect each capability?",
        "Three kinds of test. The registry stores the results in one frozen _ModelInfo.")
cy = y2 + 70
CH = 108
d.step_card(LX[0], cy, LW, CH, 1, "Flag only", EMERALD, tag="getattr", rows=[
    ("e.g.", "supports_multimodal"),
    ("", "is_hybrid, has_inner_state"),
    ("rule", "getattr(cls, flag, False)"),
    ("line", "interfaces.py:589"),
], key_w=40)
d.step_card(LX[1], cy, LW, CH, 2, "Protocol members", VIOLET, tag="isinstance", rows=[
    ("e.g.", "supports_eagle, mrope"),
    ("", "kv_cache_grouping"),
    ("rule", "all members present"),
    ("line", "interfaces.py:1710"),
], key_w=40)
d.step_card(LX[2], cy, LW, CH, 3, "Flag + structure", AMBER, tag="+ warnings", rows=[
    ("lora", "flag + 2 mapping attrs"),
    ("pp", "flag + forward keyword"),
    ("base", "no flag: init/embed/fwd"),
    ("line", ":760-788, :864-897"),
], key_w=40)

# ---- Panel 3: the checks --------------------------------------------------------------------
y3 = y2 + P2_H + 18
P3_H = 196
d.panel(y3, P3_H, 3, "Who reads the result, and when?",
        "The front end checks at config time. The worker checks the built model again.")
ey = y3 + 70
EH = 106
d.box(LX[0], ey, LW, EH, "ModelConfig (front end)", [
    "not generative -> ValueError",
    "  --runner generate  :728",
    "PP > 1, no supports_pp",
    "  -> NotImplementedError :1569",
], col=PINK, align="start")
d.box(LX[1], ey, LW, EH, "LoRAConfig (front end)", [
    "--enable-lora and",
    "not is_lora_supported",
    "-> ValueError",
    "config/lora.py:137-143",
], col=PINK, align="start")
d.box(LX[2], ey, LW, EH, "Worker (model object)", [
    "load_lora_model: supports_lora",
    "  lora_model_runner_mixin.py:49",
    "EAGLE-3: supports_eagle3",
    "  eagle3_utils.py:29",
], col=PINK, align="start")

y = y3 + P3_H + 22
d.legend(30, y, [(BLUE, "model class"), (EMERALD, "flag test"), (VIOLET, "protocol test"),
                 (AMBER, "combined test"), (PINK, "start-up check")])
y += 34
y = d.footnotes(30, y, [
    "The registry computes _ModelInfo from the class, not from an object (registry.py:917-954).",
    "Lazy entries get it from the modelinfos cache or a subprocess, so the front end does not import the model.",
])
d.save(height=y + 8)
