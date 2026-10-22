"""PyTorch for serving: eager vs compiled vs CUDA graph replay.

Used by: src/learn/pytorch-for-serving.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/models/llama.py:224-234              LlamaAttention.forward (qkv_proj, rotary, attn, o_proj)
  vllm/model_executor/layers/attention/attention.py:573-580  torch.ops.vllm.unified_attention_with_output
  vllm/utils/torch_utils.py:1056-1096                     vllm_lib = Library("vllm", "FRAGMENT"); direct_register_custom_op
  vllm/compilation/wrapper.py:168-175                     torch.compile(..., fullgraph=True, dynamic=False, backend=...)
  vllm/config/compilation.py:1123-1134                    VLLM_COMPILE -> VllmBackend
  vllm/config/compilation.py:766-769                      attention ops are the splitting ops
  vllm/compilation/backends.py:576                        split_graph()
  vllm/config/compilation.py:53-63                        CUDAGraphMode NONE / PIECEWISE / FULL / FULL_AND_PIECEWISE
  vllm/v1/worker/gpu/cudagraph_utils.py:523-553            FULL only for uniform decode batches up to the largest size
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Three ways to run the same forward pass",
         "one decoder layer of a Llama model  ·  CUDA platform  ·  default optimization level O2",
         params="eager  ->  torch.compile (Dynamo + Inductor, vLLM backend)  ->  CUDA graph replay")

BW = 300
X1, X2, X3 = 46, 400, 754
BH = 128

def three(y, a, b, c, cols, labels):
    d.box(X1, y, BW, BH, a[0], a[1], col=cols[0], align="start")
    d.box(X2, y, BW, BH, b[0], b[1], col=cols[1], align="start")
    d.box(X3, y, BW, BH, c[0], c[1], col=cols[2], align="start")
    d.arrow(X1 + BW, y + 50, X2 - 2, y + 50, cols[0])
    d.arrow(X2 + BW, y + 50, X3 - 2, y + 50, cols[1])
    if labels[0]:
        d.arrow_label((X1 + BW + X2) / 2, y + 50, labels[0], cols[0])
    if labels[1]:
        d.arrow_label((X2 + BW + X3) / 2, y + 50, labels[1], cols[1])

# ---------------------------------------------------------------- panel 1
y0 = 116
PH = 222
d.panel(y0, PH, 1, "Eager: Python runs forward() one op at a time",
        "Each op goes through the PyTorch dispatcher and starts its own kernel. The CPU work grows with the number of ops.")
three(y0 + 72,
      ("Python: LlamaAttention.forward", ["qkv, _ = self.qkv_proj(h)", "q, k = self.rotary_emb(...)",
                                          "o = self.attn(q, k, v)", "out, _ = self.o_proj(o)"]),
      ("PyTorch dispatcher", ["one call per op", "picks the CUDA kernel", "checks shapes and dtypes",
                              "on every forward pass"]),
      ("GPU", ["many small kernels", "a gap between two launches", "CPU time ~ number of ops"]),
      (INK, INK, INK), ("op", "launch"))

# ---------------------------------------------------------------- panel 2
y1 = y0 + PH + 20
d.panel(y1, PH, 2, "Compiled: Dynamo captures a graph, Inductor writes the kernels",
        "vLLM compiles with fullgraph=True and splits the graph at the attention custom op. Inductor compiles each piece.")
three(y1 + 72,
      ("Dynamo (capture)", ["reads the Python bytecode", "records the ops as an FX graph",
                            "fullgraph=True: a graph break", "  is an error"]),
      ("VllmBackend: split_graph()", ["split at the splitting ops,", "for example",
                                      "vllm::unified_attention_", "  with_output (opaque)"]),
      ("Inductor (codegen)", ["compiles each piece", "fuses ops into Triton kernels",
                              "fewer, larger kernels", "cache: ~/.cache/vllm/torch_compile_cache"]),
      (BLUE, VIOLET, BLUE), ("FX graph", "pieces"))

# ---------------------------------------------------------------- panel 3
y2 = y1 + PH + 20
d.panel(y2, PH, 3, "Graph replay: record the launches once, replay them each step",
        "At start-up vLLM captures CUDA graphs for fixed batch sizes. A step pads its batch to a captured size.")
three(y2 + 72,
      ("Capture (start-up)", ["run each captured size once", "record all kernel launches",
                              "pieces: PIECEWISE graphs", "whole model: FULL graphs"]),
      ("Dispatch (each step)", ["uniform 1-token decode batch", "  <= largest size: FULL",
                                "other batches: PIECEWISE", "too large: eager"]),
      ("GPU", ["one replay per graph", "attention runs between", "  PIECEWISE pieces",
               "almost no launch overhead"]),
      (EMERALD, EMERALD, EMERALD), ("graphs", "replay"))

y = y2 + PH + 26
d.legend(30, y, [(INK, "eager PyTorch"), (BLUE, "torch.compile"), (VIOLET, "vLLM split at custom ops"),
                 (EMERALD, "CUDA graphs")])
y += 40
y = d.footnotes(30, y, [
    "Default: optimization level O2 -> compilation mode VLLM_COMPILE and CUDA graph mode FULL_AND_PIECEWISE.",
    "--enforce-eager turns off both compilation and CUDA graphs. The attention backend can lower the graph mode.",
])
d.save(height=y + 14)
