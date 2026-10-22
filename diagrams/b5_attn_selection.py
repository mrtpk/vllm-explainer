"""How one attention layer gets its backend class, with four worked examples on CUDA.

Used by: src/concepts/attention-backend-selection.html
Sources (vLLM at the pinned commit 3e44c0d):
  vllm/model_executor/layers/attention/attention.py:350-360   Attention.__init__ calls get_attn_backend(use_mla=False)
  vllm/model_executor/layers/attention/mla_attention.py:514-521  MLAAttention calls it with use_mla=True, num_heads
  vllm/v1/attention/selector.py:109-203   get_attn_backend: AttentionSelectorConfig, block_size only if user set it,
                                          attention_config.backend / backend_per_kind
  vllm/v1/attention/selector.py:206-228   @cache _cached_get_attn_backend -> current_platform.get_attn_backend_cls
  vllm/platforms/cuda.py:430-449          explicit backend: validate only that one, ValueError if invalid
  vllm/platforms/cuda.py:84-181           _get_backend_priorities (MLA / non-MLA, per compute capability major)
  vllm/platforms/cuda.py:377-417          get_valid_backends: validate_configuration for each candidate
  vllm/platforms/cuda.py:471-485          no valid -> ValueError with every reason; else smallest priority
  vllm/v1/attention/backend.py:279-369    validate_configuration: the reasons
  vllm/v1/attention/backends/flash_attn.py:456-457   FLASH_ATTN needs capability >= 8.0
  vllm/v1/attention/backends/flashinfer.py:662-675   FLASHINFER head sizes 64/128/256/512, capability 8.0-12.1
  vllm/v1/attention/backends/mla/flashattn_mla.py:42-46, 69-70, 85-86  FLASH_ATTN_MLA: no fp8 KV, SM90 + FA3 only
  vllm/v1/attention/backends/mla/flashinfer_mla.py:220-221   FLASHINFER_MLA: capability major 10
  vllm/v1/attention/backends/fa_utils.py:338-339  MLA head dims 576 (qk) / 512 (v)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("How a layer gets its attention backend",
         "Runs in the worker while the model is built  ·  one answer per input set (@cache)  ·  CUDA platform",
         params="vllm/v1/attention/selector.py  ·  vllm/platforms/cuda.py  ·  vllm/v1/attention/backend.py")

LX = [40, 390, 740]
LW = 320
SEL, PLAT, OUT = BLUE, VIOLET, EMERALD

# ---- Panel 1: the call chain -------------------------------------------------------------------
y1 = 120
CH = 112
P1_H = 70 + CH + 24 + CH + 22
d.panel(y1, P1_H, 1, "Which code decides?",
        "The layer asks the selector; the selector asks current_platform; CUDA walks a priority list.")
r1 = y1 + 70
d.step_card(LX[0], r1, LW, CH, 1, "Layer __init__", SEL, tag="attention.py:350", rows=[
    ("in", "head_size, dtype, kv_cache_dtype"),
    ("", "use_mla, has_sink, attn_type,"),
    ("", "sliding window (MLA: use_sparse)"),
], key_w=24)
d.step_card(LX[1], r1, LW, CH, 2, "get_attn_backend", SEL, tag="selector.py:109", rows=[
    ("add", "block_size if --block-size"),
    ("", "kv connector, PCP/DCP, batch inv."),
    ("pick", "--attention-backend / per kind"),
], key_w=38)
d.step_card(LX[2], r1, LW, CH, 3, "Cached selector", SEL, tag="selector.py:206", rows=[
    ("@cache", "one call per input set"),
    ("call", "platform.get_attn_backend_cls"),
    ("then", "import the class   :227"),
], key_w=50)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r1 + 56, LX[i + 1] - 2, r1 + 56, SEL)

r2 = r1 + CH + 24
d.step_card(LX[0], r2, LW, CH, 4, "Explicit backend?", PLAT, tag="cuda.py:430-449", rows=[
    ("yes", "validate only that backend"),
    ("bad", "ValueError, no fallback"),
    ("no", "go to the priority list"),
], key_w=30)
d.step_card(LX[1], r2, LW, CH, 5, "Priority list", PLAT, tag="cuda.py:84-181", rows=[
    ("key", "use_mla, capability major,"),
    ("", "num_heads, kv dtype, mm prefix"),
    ("each", "validate_configuration()"),
], key_w=30)
d.step_card(LX[2], r2, LW, CH, 6, "Pick", OUT, tag="cuda.py:471-518", rows=[
    ("none", "ValueError + every reason"),
    ("else", "valid one with smallest index"),
    ("log", "\"Using X attention backend\""),
], key_w=38)
d.path_arrow(f"M {LX[2] + LW / 2:.0f},{r1 + CH + 2} L {LX[2] + LW / 2:.0f},{r1 + CH + 12} "
             f"L {LX[0] + LW / 2:.0f},{r1 + CH + 12} L {LX[0] + LW / 2:.0f},{r2 - 2}", PLAT)
for i in range(2):
    d.arrow(LX[i] + LW + 2, r2 + 56, LX[i + 1] - 2, r2 + 56, PLAT)

# ---- Panel 2: worked examples ----------------------------------------------------------------------
y2 = y1 + P1_H + 18
EW, EH = 505, 132
P2_H = 70 + EH + 16 + EH + 22
d.panel(y2, P2_H, 2, "Which backend wins? (default kv_cache_dtype auto, BF16)",
        "The first valid entry of the list wins. Sparse MLA entries fail with \"non-sparse not supported\".")
EX = [40, 555]
ey1 = y2 + 70
ey2 = ey1 + EH + 16
examples = [
    (EX[0], ey1, "Llama-3-8B on H100 (SM90)", "MHA/GQA, head_size 128", [
        "FLASH_ATTN, FLASHINFER, TRITON_ATTN,",
        "FLEX_ATTENTION, TURBOQUANT",
    ], "FLASH_ATTN"),
    (EX[1], ey1, "Llama-3-8B on B200 (SM100)", "MHA/GQA, head_size 128", [
        "FLASHINFER, FLASH_ATTN, TRITON_ATTN,",
        "FLEX_ATTENTION, TURBOQUANT",
    ], "FLASHINFER"),
    (EX[0], ey2, "DeepSeek-V3 (MLA) on H100", "use_mla, head_size 576 = 512 + 64", [
        "FLASH_ATTN_MLA, FLASHMLA, FLASHINFER_MLA,",
        "TRITON_MLA, + 3 sparse MLA entries",
    ], "FLASH_ATTN_MLA"),
    (EX[1], ey2, "DeepSeek-V3 (MLA) on B200", "use_mla, head_size 576", [
        "FLASHINFER_MLA, TOKENSPEED_MLA, CUTLASS_MLA,",
        "FLASH_ATTN_MLA, FLASHMLA, TRITON_MLA, + sparse",
    ], "FLASHINFER_MLA"),
]
for x, y, title, sub, lst, win in examples:
    d.rect(x, y, EW, EH, fill="#ffffff", stroke="#e5e7eb", rx=10, sw=1.2)
    d.text(x + 14, y + 22, title, 13.5, INK, "bold")
    d.text(x + 14, y + 40, sub, 11.5, MUTE, mono=True)
    d.text(x + 14, y + 62, "priority list:", 11.5, MUTE)
    for i, s in enumerate(lst):
        d.text(x + 14, y + 79 + i * 15, s, 11.5, INK, mono=True)
    d.text(x + 14, y + 121, "chosen:", 12, MUTE, "bold")
    d.chip(x + 76, y + 104, 190, win, OUT, h=26)

y = y2 + P2_H + 16
d.callout(30, y, 1040, "fp8 KV cache:",
          "on H100 the MLA example moves to FLASHMLA, because FLASH_ATTN_MLA accepts only auto/float16/bfloat16.",
          AMBER, h=34)
y += 50
d.legend(30, y, [(SEL, "selector (any platform)"), (PLAT, "CudaPlatform"), (OUT, "result")])
y += 34
y = d.footnotes(30, y, [
    "Assumes the vllm_flash_attn build has FA3 (SM90) / FA4 (SM100) and that FlashInfer and FlashMLA import;",
    "  an ImportError or OSError makes that backend invalid (cuda.py:405-409). Model sizes are from the model configs.",
    "Not shown: ROCm, CPU and XPU have their own get_attn_backend_cls; an out-of-tree platform returns any class path.",
])
d.save(height=y + 8)
