"""Weight loading in vLLM: files -> weights iterator -> WeightsMapper -> per-parameter weight_loader -> slice on each rank.

Used by: src/concepts/weight-loading.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/model_loader/default_loader.py:168-254   _prepare_weights: auto -> hf, patterns, index filter
  vllm/model_executor/model_loader/default_loader.py:337-349   safetensors_weights_iterator for the default case
  vllm/model_executor/model_loader/weight_utils.py:1243-1249   lazy safetensors: safe_open + get_tensor per name
  vllm/model_executor/model_loader/default_loader.py:503       model.load_weights(iterator)
  vllm/model_executor/models/llama.py:559-561                  LlamaForCausalLM.load_weights: AutoWeightsLoader, no mapper
  vllm/model_executor/models/llama.py:348-357, 459-461         LlamaModel: hf_to_vllm_mapper with orig_to_new_stacked
  vllm/model_executor/models/utils.py:125-129, 147-164         stacked rename + tensor.shard_id
  vllm/model_executor/models/utils.py:387-400                  a child with load_weights gets the sub-stream
  vllm/model_executor/layers/linear.py:1339-1354               QKVParallelLinear.load_weights -> param.weight_loader(param, w, shard_id)
  vllm/model_executor/layers/linear.py:1167-1209               weight_loader_v2 -> param.load_qkv_weight
  vllm/model_executor/parameter.py:175-198                     load_qkv_weight: narrow param and checkpoint tensor
Shapes: Llama-3-8B (Hugging Face config.json: hidden 4096, 32 query heads, 8 KV heads, head size 128), BF16, TP = 2.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Weight loading: from checkpoint files to one slice per GPU",
         "Default --load-format auto  ·  BF16 Llama-3-8B checkpoint  ·  TP = 2  ·  the same steps run in every worker",
         params="q_proj.weight [4096, 4096]   k_proj.weight [1024, 4096]   v_proj.weight [1024, 4096]   (checkpoint, [out, in])")

BW = 324
XS = (46, 388, 730)

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 196
d.panel(y0, P1_H, 1, "The loader picks the files and streams (name, tensor) pairs",
        "Every worker opens the same files; a tensor is read only when the iterator reaches its name.")
by = y0 + 70
BH = 104
d.box(XS[0], by, BW, BH, "Checkpoint folder", [
    "model-0000N-of-0000M.safetensors",
    "model.safetensors.index.json",
    "(HF cache or local path)",
], col=INK)
d.box(XS[1], by, BW, BH, "DefaultModelLoader._prepare_weights", [
    "auto -> \"hf\" (no consolidated*)",
    "patterns *.safetensors, *.bin, *.pt",
    "keep files named in the index",
], col=BLUE, tsize=12.5)
d.box(XS[2], by, BW, BH, "safetensors_weights_iterator", [
    "files in natural sort order",
    "safe_open(f, framework=\"pt\")",
    "yield (name, CPU tensor)",
], col=BLUE, tsize=12.5)
d.arrow(XS[0] + BW, by + BH / 2, XS[1] - 2, by + BH / 2, INK)
d.arrow(XS[1] + BW, by + BH / 2, XS[2] - 2, by + BH / 2, BLUE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 20
P2_H = 216
d.panel(y1, P2_H, 2, "Names are mapped to vLLM parameters, then each parameter loads itself",
        "q_proj, k_proj and v_proj all go into one stacked parameter, qkv_proj.weight, with shard ids \"q\", \"k\", \"v\".")
by = y1 + 70
BH = 124
d.box(XS[0], by, BW, BH, "LlamaForCausalLM.load_weights", [
    "AutoWeightsLoader(self)",
    "first name part \"model\"",
    "-> LlamaModel.load_weights",
    "(names lose \"model.\")",
], col=VIOLET, tsize=12.5)
d.box(XS[1], by, BW, BH, "WeightsMapper (hf_to_vllm_mapper)", [
    "layers.0.self_attn.q_proj.weight",
    "-> layers.0.self_attn.qkv_proj.weight",
    "tensor.shard_id = \"q\"",
    "(k -> \"k\", v -> \"v\")",
], col=VIOLET, tsize=12.5)
d.box(XS[2], by, BW, BH, "QKVParallelLinear.load_weights", [
    "param = self.weight",
    "param.weight_loader(param, w, \"q\")",
    "= weight_loader_v2 (BF16)",
    "-> param.load_qkv_weight(...)",
], col=VIOLET, tsize=12.5)
d.arrow(XS[0] + BW, by + BH / 2, XS[1] - 2, by + BH / 2, VIOLET)
d.arrow(XS[1] + BW, by + BH / 2, XS[2] - 2, by + BH / 2, VIOLET)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 20
P3_H = 268
d.panel(y2, P3_H, 3, "Each rank copies only its rows into its own qkv_proj.weight [3072, 4096]",
        "Left: rows of the stacked parameter (same on both ranks). Right: rows of the checkpoint tensor (rank x slice size).")
LW = 500
LX = (46, 554)
ly = y2 + 66
LH = 188
for r, x in enumerate(LX):
    col = AMBER if r == 0 else EMERALD
    d.lane(x, ly, LW, LH, col, f"TP rank {r}", [f"qkv_proj.weight [3072, 4096]  =  24 MiB in BF16"], header_h=46)
    cy = ly + 58
    rows = [
        ("q", "rows    0-2047", f"q_proj rows {2048 * r}-{2048 * r + 2047}", "heads " + f"{16 * r}-{16 * r + 15}"),
        ("k", "rows 2048-2559", f"k_proj rows {512 * r}-{512 * r + 511}", "KV heads " + f"{4 * r}-{4 * r + 3}"),
        ("v", "rows 2560-3071", f"v_proj rows {512 * r}-{512 * r + 511}", "KV heads " + f"{4 * r}-{4 * r + 3}"),
    ]
    for i, (sid, prow, crow, heads) in enumerate(rows):
        ry = cy + i * 42
        d.rect(x + 14, ry, LW - 28, 36, fill="#ffffff", stroke=col, rx=6, sw=1.2)
        d.text(x + 28, ry + 23, sid, 13, col, "bold", mono=True)
        d.text(x + 52, ry + 23, prow, 12, INK, mono=True)
        d.text(x + 170, ry + 23, "<-", 12, MUTE, mono=True)
        d.text(x + 196, ry + 23, crow, 12, INK, mono=True)
        d.text(x + LW - 24, ry + 23, heads, 11.5, MUTE, anchor="end")

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "file selection and reading"), (VIOLET, "name mapping and dispatch"),
                 (AMBER, "rank 0 slice"), (EMERALD, "rank 1 slice")])
y += 40
y = d.footnotes(30, y, [
    "Shapes from the Llama-3-8B config.json on the Hugging Face Hub; head size 128. Each rank: 16 query heads, 4 KV heads.",
    "A quantized layer makes other parameters (create_weights of its quant method) and may use weight_loader (v1) instead of v2.",
    "Not shown: process_weights_after_loading, which runs once per layer after all tensors are copied.",
])
d.save(height=y + 14)
