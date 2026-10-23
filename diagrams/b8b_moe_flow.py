"""Expert parallel MoE: one MoE layer with DP = 2 and --enable-expert-parallel (EP size 2).

Used by: src/concepts/expert-parallel-moe.html
Sources (at the pinned commit in explainer.json):
  vllm/model_executor/layers/fused_moe/layer.py:92-147, 306-346, 415-470   FusedMoEFactory: router, RoutedExperts, MoERunner
  vllm/model_executor/layers/fused_moe/config.py:1198-1244   use_ep, ep_size = flattened TP across DP
  vllm/config/parallel.py:182-191   expert_placement_strategy "linear" (default)
  vllm/config/parallel.py:220-222   all2all_backend default allgather_reducescatter
  vllm/distributed/device_communicators/all2all.py:67-173   dispatch = all-gather, combine = reduce-scatter
  vllm/model_executor/layers/fused_moe/modular_kernel.py:51-84   [Router] → [Quantize-Dispatch] → [Experts] → [Combine]
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("One MoE layer with expert parallelism",
         "DP = 2, TP = 1, --enable-expert-parallel  →  EP size 2  ·  8 experts, top_k = 2  ·  linear placement",
         params="rank 0 holds experts 0-3  ·  rank 1 holds experts 4-7  ·  hidden size H  ·  rank 0 has 3 tokens, rank 1 has 2")

# ---------------------------------------------------------------- panel 1
y0 = 116
P1_H = 236
d.panel(y0, P1_H, 1, "Where do the tokens go?",
        "The router picks 2 experts per token; dispatch moves each token to the ranks that own them.")
BY = y0 + 70
BW, BH = 316, 140
d.box(46, BY, BW, BH, "1  Router (on each rank)", [
    "rank 0: x [3, H] → logits [3, 8]",
    "top-2 ids  t0:(1,5) t1:(2,3) t2:(6,7)",
    "rank 1: x [2, H]",
    "top-2 ids  t3:(0,4) t4:(4,5)",
    "+ top-2 weights [T, 2]",
], col=BLUE, align="start")
d.box(392, BY, BW, BH, "2  Dispatch (prepare)", [
    "default allgather_reducescatter:",
    "all-gather over the DP group",
    "→ every rank sees all 5 tokens",
    "DeepEP / NIXL-EP: send each token",
    "only to the ranks of its experts",
], col=AMBER, align="start")
d.box(738, BY, BW, BH, "3  Local experts", [
    "rank 0: experts 0-3 run on",
    "  t0(e1) t1(e2,e3) t3(e0)",
    "rank 1: experts 4-7 run on",
    "  t0(e5) t2(e6,e7) t3(e4) t4(e4,e5)",
    "expert_map: global id → local id",
], col=EMERALD, align="start")
d.arrow(362, BY + BH / 2, 390, BY + BH / 2, AMBER, sw=2)
d.arrow(708, BY + BH / 2, 736, BY + BH / 2, AMBER, sw=2)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P1_H + 22
P2_H = 200
d.panel(y1, P2_H, 2, "How do the results come back?",
        "Combine sums the weighted expert outputs of each token and returns them to the token's own rank.")
CY = y1 + 70
d.box(46, CY, BW, 104, "4  Combine (finalize)", [
    "default: reduce-scatter over DP",
    "→ rank 0 gets [3, H], rank 1 [2, H]",
    "each row = Σ weight · expert output",
], col=AMBER, align="start")
d.box(392, CY, BW, 104, "5  + shared experts (if any)", [
    "dense MLP on the local tokens,",
    "started before dispatch, may run",
    "on a second CUDA stream",
], col=VIOLET, align="start")
d.box(738, CY, BW, 104, "6  Layer output", [
    "rank 0: [3, H]   rank 1: [2, H]",
    "next: attention again, on each",
    "DP rank's own tokens only",
], col=BLUE, align="start")
d.arrow(362, CY + 52, 390, CY + 52, BLUE)
d.arrow(708, CY + 52, 736, CY + 52, BLUE)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P2_H + 22
P3_H = 214
d.panel(y2, P3_H, 3, "Which objects does FusedMoEFactory build?",
        "One MoERunner per MoE layer; the kernel is a prepare/finalize object plus an experts object.")
KY = y2 + 70
d.box(46, KY, BW, 116, "MoERunner", [
    "router: FusedMoERouter",
    "routed_experts: RoutedExperts",
    "shared_experts, gate (optional)",
    "forward → torch.ops.vllm.moe_forward",
], col=VIOLET, align="start")
d.box(392, KY, BW, 116, "FusedMoEPrepareAndFinalize", [
    "= dispatch + combine backend",
    "naive (AG/RS), DeepEP HT / LL / v2,",
    "NIXL-EP, MoRI, FlashInfer NVLink,",
    "MoonEP, passthrough",
], col=AMBER, align="start")
d.box(738, KY, BW, 116, "FusedMoEExperts", [
    "= the grouped expert GEMMs",
    "picked by an oracle per quant type:",
    "triton, flashinfer, cutlass,",
    "deep_gemm, marlin, aiter, …",
], col=EMERALD, align="start")
d.arrow(362, KY + 58, 390, KY + 58, INK)
d.text(723, KY + 64, "+", 20, INK, "bold", "middle")

y = y2 + P3_H + 24
d.legend(30, y, [(BLUE, "tokens on their own rank"), (AMBER, "communication"), (EMERALD, "expert compute"),
                 (VIOLET, "layer objects")])
y += 36
y = d.footnotes(30, y, [
    "Token ids, expert choices and sizes are an example. Without --enable-expert-parallel, the experts are split with TP",
    "  over all DP x TP ranks instead (ep_size = 1). Kernel internals are not shown.",
])
d.save(height=y + 14)
