"""Hugging Face checkpoints: which file of a model folder each part of vLLM reads.

Used by: src/learn/hf-checkpoints.html
Sources (at the pinned commit in explainer.json):
  vllm/transformers_utils/config.py:873-958          get_config(): config.json (or params.json), architectures, quantization_config
  vllm/config/model.py:680-690, 739-744              ModelConfig: hf_config, registry.inspect_model_cls(architectures)
  vllm/transformers_utils/config.py:1269-1295        try_get_generation_config(): generation_config.json
  vllm/config/model.py:1860-1916                     get_diff_sampling_param(): default sampling values
  vllm/renderers/hf.py:264-319                       resolve_chat_template(): given > processor > tokenizer > fallback
  vllm/model_executor/model_loader/default_loader.py:147-263   _prepare_weights(): *.safetensors, index json filter
  vllm/model_executor/model_loader/weight_utils.py:437-543     download_weights_from_hf(): snapshot_download into the HF cache
  vllm/model_executor/models/registry.py:199          "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM")
Example folder: Qwen/Qwen3-0.6B (the default model). File names and values of that repo are inferred, not from vLLM.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("What vLLM reads from a Hugging Face model folder",
         "example: Qwen/Qwen3-0.6B, the default model  ·  files on the left, the vLLM code that reads them in the middle",
         params="HF cache: $HF_HOME/hub/models--Qwen--Qwen3-0.6B/snapshots/<commit>/")

FX, FW = 46, 300
RX, RW = 388, 324
OX, OW = 754, 300
BH = 84

# ---------------------------------------------------------------- panel 1
y0 = 116
rows = [
    (BLUE, "config.json", ['"architectures": ["Qwen3ForCausalLM"]', '"max_position_embeddings": 40960',
                           '"torch_dtype": "bfloat16"'],
     "get_config()", ["transformers_utils/config.py", "-> ModelConfig.hf_config"],
     "Model shape and class", ["architectures -> ModelRegistry", "max_model_len, dtype, quantization"]),
    (VIOLET, "generation_config.json", ['"temperature": 0.6', '"top_k": 20,  "top_p": 0.95'],
     "try_get_generation_config()", ["--generation-config auto", "-> get_diff_sampling_param()"],
     "Default sampling values", ["used when the request", "does not set them"]),
    (PINK, "tokenizer files", ["tokenizer.json, tokenizer_config.json", "chat_template.jinja (newer repos)"],
     "tokenizer + resolve_chat_template()", ["--tokenizer (default: the model)", "renderers/hf.py"],
     "Tokens and the prompt text", ["chat messages -> one prompt", "-> token ids"]),
    (AMBER, "weight files", ["model.safetensors, or", "model-0000i-of-0000n.safetensors", "+ model.safetensors.index.json"],
     "DefaultModelLoader", ["_prepare_weights(): find, download,", "filter by the index json"],
     "Weights in the model", ["model.load_weights(name, tensor)", "on every worker"]),
]
P1_H = 70 + len(rows) * (BH + 16) + 10
d.panel(y0, P1_H, 1, "Four kinds of files, four readers",
        "The engine reads config and weights; the API server also reads the generation config and the tokenizer files.")
ry = y0 + 70
for col, fname, flines, rname, rlines, oname, olines in rows:
    d.box(FX, ry, FW, BH, fname, flines, col=col, align="start", mono_size=11.5)
    d.box(RX, ry, RW, BH, rname, rlines, col=INK, fill="#ffffff", align="start")
    d.box(OX, ry, OW, BH, oname, olines, col=EMERALD, align="start")
    d.arrow(FX + FW, ry + BH / 2, RX - 2, ry + BH / 2, col)
    d.arrow(RX + RW, ry + BH / 2, OX - 2, ry + BH / 2, INK)
    ry += BH + 16

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 176
d.panel(y1, P2_H, 2, "From a model name to a Python class",
        "vllm serve Qwen/Qwen3-0.6B: download or find the folder, read architectures, look the name up in the registry.")
by = y1 + 70
BW = 324
xs = (46, 388, 730)
d.box(xs[0], by, BW, 84, "1. Name or path", ["Qwen/Qwen3-0.6B  (or /models/qwen)", "--revision: branch, tag or commit",
                                              "HF_HUB_OFFLINE=1: cache only"], col=BLUE, align="start")
d.box(xs[1], by, BW, 84, "2. ModelConfig", ["hf_config = get_config(model, ...)", "architectures[0] = Qwen3ForCausalLM",
                                             "inspect_model_cls(architectures)"], col=INK, fill="#ffffff", align="start")
d.box(xs[2], by, BW, 84, "3. ModelRegistry", ['"Qwen3ForCausalLM":', '  ("qwen3", "Qwen3ForCausalLM")',
                                               "-> model_executor/models/qwen3.py"], col=EMERALD, align="start")
d.arrow(xs[0] + BW, by + 42, xs[1] - 2, by + 42, BLUE)
d.arrow(xs[1] + BW, by + 42, xs[2] - 2, by + 42, INK)

y = y1 + P2_H + 26
d.legend(30, y, [(BLUE, "model config"), (VIOLET, "generation config"), (PINK, "tokenizer files"), (AMBER, "weights"),
                 (EMERALD, "what vLLM gets")])
y += 40
y = d.footnotes(30, y, [
    "The values in the file boxes come from the Qwen3-0.6B repository on the Hugging Face Hub, not from the vLLM repository.",
    "Not shown: Mistral-format folders (params.json, consolidated*.safetensors), remote code (trust_remote_code), multimodal processor files.",
])
d.save(height=y + 14)
