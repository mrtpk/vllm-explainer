"""B6b: how an attention layer finds its KV cache and metadata through the forward context.

Used by: src/concepts/forward-context.html, src/code/forward-context.html,
         src/code/model-executor-layers-attention-attention.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/layers/attention/attention.py:443-445   register static_forward_context[prefix]
  vllm/v1/worker/gpu/model_runner.py:794-797                  init_kv_cache gets static_forward_context
  vllm/v1/worker/utils.py:868-880                             bind_kv_cache_to_layers -> layer.kv_cache
  vllm/v1/worker/gpu/model_runner.py:2034-2053                set_forward_context(...) around the model call
  vllm/forward_context.py:140-158, 247-259                     ForwardContext fields; no_compile_layers
  vllm/model_executor/layers/attention/attention.py:561-581   torch.ops.vllm.unified_* with the layer name
  vllm/model_executor/layers/attention/attention.py:705-723   get_attention_context
  vllm/model_executor/layers/attention/attention.py:740-746, 786-796   impl.do_kv_cache_update, impl.forward
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("The forward context: per-step data found by layer name",
         "Llama-3-8B, layer 0  ·  name = model.layers.0.self_attn.attn  ·  eager or PIECEWISE step",
         params="static_forward_context: built once  ·  ForwardContext: one per forward pass")

BW = 300
XS = [46, 400, 754]

# ---------------------------------------------------------------- panel 1
y0 = 118
P1_H = 196
d.panel(y0, P1_H, 1, "What is set up once, at start-up?",
        "Each layer registers under its name; later the runner binds a KV cache view to the layer object.")
BY = y0 + 70
d.box(XS[0], BY, BW, 100, "Attention.__init__(prefix)", [
    "selects the backend, makes impl",
    "static_forward_context[",
    "  prefix] = self",
    "attention.py:443-445",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 100, "static_forward_context", [
    "dict in compilation_config",
    "name -> layer object",
    "32 attention entries",
    "+ MoE, Mamba, LoRA layers",
], col=VIOLET, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 100, "init_kv_cache (KV init)", [
    "allocate one buffer, views",
    "bind_kv_cache_to_layers:",
    "  layer.kv_cache = view",
    "utils.py:868-880",
], col=VIOLET, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 50, XS[1] - 2, BY + 50, VIOLET)
d.arrow(XS[2] - 2, BY + 50, XS[1] + BW + 2, BY + 50, VIOLET)
d.arrow_label((XS[1] + BW + XS[2]) / 2, BY + 50, "bind", VIOLET)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 196
d.panel(y1, P2_H, 2, "What does the runner set for each step?",
        "set_forward_context installs a ForwardContext for the duration of the model call.")
BY = y1 + 70
d.box(XS[0], BY, BW, 100, "execute_model", [
    "attn_metadata: name -> meta",
    "slot_mappings_by_layer:",
    "  name -> slot tensor",
    "batch_desc.cg_mode",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 100, "set_forward_context(...)", [
    "with ...:  model(**inputs)",
    "global _forward_context",
    "restored on exit",
    "forward_context.py:278",
], col=BLUE, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 100, "ForwardContext", [
    "no_compile_layers (static)",
    "attn_metadata, slot_mapping",
    "cudagraph_runtime_mode",
    "batch_descriptor, is_padding",
], col=BLUE, align="start", mono_size=11.5)
d.arrow(XS[0] + BW, BY + 50, XS[1] - 2, BY + 50, BLUE)
d.arrow(XS[1] + BW, BY + 50, XS[2] - 2, BY + 50, BLUE)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 220
d.panel(y2, P3_H, 3, "How does attention find its data?",
        "The custom ops get tensors plus the name; get_attention_context does three dict lookups.")
BY = y2 + 70
d.box(XS[0], BY, BW, 126, "LlamaAttention.forward", [
    "qkv_proj, split, RoPE",
    "self.attn(q, k, v)",
    "-> Attention.forward",
    "   reshape q, k, v",
    "   output = empty(...)",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[1], BY, BW, 126, "torch.ops.vllm.*", [
    "unified_kv_cache_update(",
    "  k, v, name)",
    "unified_attention_with_",
    "  output(q, k, v, out, name)",
    "opaque to torch.compile",
], col=EMERALD, align="start", mono_size=11.5)
d.box(XS[2], BY, BW, 126, "get_attention_context(name)", [
    "meta = ctx.attn_metadata[name]",
    "layer = no_compile_layers[name]",
    "kv = layer.kv_cache",
    "slot = ctx.slot_mapping[name]",
    "-> impl.do_kv_cache_update,",
], col=AMBER, align="start", mono_size=11.5)
d.text(XS[2] + 12, BY + 38 + 5 * 15, "   impl.forward", 11.5, INK, mono=True)
d.arrow(XS[0] + BW, BY + 59, XS[1] - 2, BY + 59, EMERALD)
d.arrow(XS[1] + BW, BY + 59, XS[2] - 2, BY + 59, AMBER)

y = y2 + P3_H + 24
d.legend(30, y, [(VIOLET, "start-up, once"), (BLUE, "every step, runner side"), (EMERALD, "every layer, model side"),
                 (AMBER, "lookup by name")])
y += 40
y = d.footnotes(30, y, [
    "In a FULL CUDA graph step no Python runs inside the model; the lookups ran once, at capture time.",
    "The name is a LayerName object on PyTorch 2.11 or newer with VLLM_USE_LAYERNAME=1 (default), else a str.",
])
d.save(height=y + 14)
