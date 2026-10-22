"""The module tree of LlamaForCausalLM: attribute names, layer types, prefixes, TP role, shapes per rank.

Used by: src/concepts/model-definition.html, src/code/model-executor-models-llama.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/models/llama.py:464-521    LlamaForCausalLM: model (prefix "model"), lm_head, logits_processor
  vllm/model_executor/models/llama.py:377-401    LlamaModel: embed_tokens (no prefix passed), make_layers("model.layers"), norm
  vllm/model_executor/models/llama.py:251-311    LlamaDecoderLayer: self_attn, mlp, two RMSNorm
  vllm/model_executor/models/llama.py:125-222    LlamaAttention: qkv_proj, o_proj, rotary_emb, attn
  vllm/model_executor/models/llama.py:82-116     LlamaMLP: gate_up_proj, down_proj, act_fn
  vllm/model_executor/models/utils.py:886-919    make_layers: prefix f"{prefix}.{idx}", PPMissingLayer outside the stage
  vllm/model_executor/layers/rotary_embedding/__init__.py:84-85, 351   get_rope caches one object per key
Shapes: Llama-3-8B values from its Hugging Face config.json (hidden 4096, intermediate 14336,
32 query heads, 8 KV heads, head size 128, vocab 128256, 32 layers), TP = 2. Weights [out, in].
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, NEUTRAL, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The module tree of LlamaForCausalLM, with prefixes and layer types",
         "One decoder layer shown in full  ·  shapes are per rank for Llama-3-8B with TP = 2  ·  weights [out, in]",
         params="prefix = dotted name passed to each constructor  ·  \"\" for the root (initialize_model)")

ROW_H = 30
GAP = 4
X_NAME = 46
X_PREFIX = 520
X_NOTE = 806
X_END = 1054

rows = [
    # (level, name, type, prefix, note, role)
    (0, "", "LlamaForCausalLM", '""', "root; mixins: LoRA, PP, Eagle, Quant", NEUTRAL),
    (1, "model", "LlamaModel", "model", "@support_torch_compile", NEUTRAL),
    (2, "embed_tokens", "VocabParallelEmbedding", "(none passed)", "W [64128, 4096]; PP first (+last if tied)", EMERALD),
    (2, "layers", "ModuleList (32)", "model.layers", "make_layers; PPMissingLayer slots", NEUTRAL),
    (3, "layers.0", "LlamaDecoderLayer", "model.layers.0", "one of 32 (shown in full)", NEUTRAL),
    (4, "input_layernorm", "RMSNorm", "(no prefix)", "CustomOp; W [4096]", PINK),
    (4, "self_attn", "LlamaAttention", "...layers.0.self_attn", "16 q heads, 4 KV heads here", NEUTRAL),
    (5, "qkv_proj", "QKVParallelLinear", "...self_attn.qkv_proj", "column; W [3072, 4096]", BLUE),
    (5, "rotary_emb", "RotaryEmbedding", "(no prefix)", "CustomOp; one object, all layers", PINK),
    (5, "attn", "Attention", "...self_attn.attn", "key in static_forward_context", AMBER),
    (5, "o_proj", "RowParallelLinear", "...self_attn.o_proj", "row; W [4096, 2048]; all-reduce", VIOLET),
    (4, "post_attention_layernorm", "RMSNorm", "(no prefix)", "CustomOp; W [4096]", PINK),
    (4, "mlp", "LlamaMLP", "...layers.0.mlp", "", NEUTRAL),
    (5, "gate_up_proj", "MergedColumnParallelLinear", "...mlp.gate_up_proj", "column; W [2 x 7168, 4096]", BLUE),
    (5, "act_fn", "SiluAndMul", "(no prefix)", "CustomOp; [T, 14336] -> [T, 7168]", PINK),
    (5, "down_proj", "RowParallelLinear", "...mlp.down_proj", "row; W [4096, 7168]; all-reduce", VIOLET),
    (3, "layers.1 ... 31", "LlamaDecoderLayer", "model.layers.1 ...", "same; PPMissingLayer off-stage", NEUTRAL),
    (2, "norm", "RMSNorm", "(no prefix)", "CustomOp; PP last rank only", PINK),
    (1, "lm_head", "ParallelLMHead", "lm_head", "W [64128, 4096]; PP last rank", EMERALD),
    (1, "logits_processor", "LogitsProcessor", "(no prefix)", "all-gather of logits; last rank", EMERALD),
]

y0 = 116
P1_H = 66 + 26 + len(rows) * (ROW_H + GAP) + 10
d.panel(y0, P1_H, 1, "Every constructor gets a prefix and passes a longer one to its children",
        "The prefix names the layer for the quantization config and the forward context. The TP layer type decides the split.")

hy = y0 + 66
d.text(X_NAME, hy + 12, "attribute: class", 12, MUTE, "bold")
d.text(X_PREFIX, hy + 12, "prefix", 12, MUTE, "bold")
d.text(X_NOTE, hy + 12, "role  ·  per-rank shape (TP = 2)", 12, MUTE, "bold")

y = hy + 26
INDENT = 22
parent_bottom = {}
for level, name, typ, prefix, note, role in rows:
    x = X_NAME + level * INDENT
    col = role[0]
    if level > 0:
        # tree connector: from the bottom of the parent row down to the middle of this row
        px = X_NAME + (level - 1) * INDENT + 10
        d.line(px, parent_bottom[level - 1], px, y + ROW_H / 2, col=MUTE, sw=1)
        d.line(px, y + ROW_H / 2, x, y + ROW_H / 2, col=MUTE, sw=1)
    d.rect(x, y, X_END - x, ROW_H, fill=role[1], stroke=col, rx=6, sw=1.2)
    label = f"{name}: {typ}" if name else typ
    d.text(x + 10, y + 19.5, label, 12, INK, "bold" if level <= 1 else "normal", mono=True)
    d.text(X_PREFIX, y + 19.5, prefix, 12, INK, mono=True)
    d.text(X_NOTE, y + 19.5, note, 11.5, INK)
    parent_bottom[level] = y + ROW_H
    y += ROW_H + GAP

y = y0 + P1_H + 24
d.legend(30, y, [(BLUE, "column-parallel"), (VIOLET, "row-parallel (all-reduce)"),
                 (EMERALD, "vocab-parallel"), (AMBER, "attention layer"), (PINK, "CustomOp"),
                 (NEUTRAL, "container")])
y += 40
y = d.footnotes(30, y, [
    "\"...\" stands for \"model.layers.0\". Shapes: Llama-3-8B config.json (Hugging Face Hub), TP = 2; 128256 is a multiple of 64, so no vocab padding.",
    "Llama passes no prefix to VocabParallelEmbedding (llama.py:382-386); RMSNorm, get_rope and SiluAndMul take no prefix.",
    "The rotary class depends on rope_parameters (Llama-3.1 with rope_type \"llama3\": Llama3RotaryEmbedding); get_rope returns one cached object per key.",
])
d.save(height=y + 14)
