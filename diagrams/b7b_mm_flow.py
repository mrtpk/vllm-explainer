"""An image request through vLLM: media fetch -> processor -> placeholders -> encoder -> merged embeddings -> decoder.

Used by: src/concepts/multimodal-serving.html
Sources (at the pinned commit in explainer.json):
  vllm/renderers/chat_utils/chat_message.py:101-119            content part types (image_url, audio_url, video_url, ...)
  vllm/multimodal/media/connector.py:348-427                    MediaConnector: data/file/http URLs, allowed domains, local path
  vllm/multimodal/processing/processor.py:1811-1847             apply(): HF processor, prompt updates, placeholders, hashes
  vllm/multimodal/cache/factories.py:23-87                      processor cache type: lru (sender + engine receiver), shm, processor_only
  vllm/v1/core/encoder_cache_manager.py:19-70, 316-362          encoder cache in embeddings; budget = max(max_num_batched_tokens, item size)
  vllm/v1/core/kv_cache_utils.py:555-586                        prefix-cache extra key ("mm", hash, offset)
  vllm/v1/worker/gpu/mm/encoder_runner.py:81-108, 150-176, 207-308   encode, gather, merge
  vllm/v1/worker/gpu/model_states/default.py:98-120             execute_mm_encoder, gather_mm_embeddings, get_inputs_embeds
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("An image request through vLLM",
         "vllm serve with a vision-language model  ·  Model Runner V2  ·  one image_url part  ·  N = embeddings of the image (model-specific)",
         params="prompt = text tokens + N placeholder tokens  ·  encoder output [N, hidden] cached by mm_hash  ·  merged into inputs_embeds")

BW = 300
X1, X2, X3 = 46, 400, 754
BH = 136

# ---------------------------------------------------------------- panel 1
y0 = 124
P_H = 236
d.panel(y0, P_H, 1, "What does the API server do with the image?",
        "It loads the media, runs the model's processor, and puts N placeholder tokens into the prompt.")
R1 = y0 + 70
d.box(X1, R1, BW, BH, "Content part → MediaConnector", [
    "{\"type\": \"image_url\", ...}",
    "data: base64 | http(s) | file:",
    "http: --allowed-media-domains,",
    "  VLLM_IMAGE_FETCH_TIMEOUT 5 s",
    "file: --allowed-local-media-path",
    "media/connector.py:348-427",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "Processor (per model)", [
    "BaseMultiModalProcessor.apply",
    "HF processor → pixel_values",
    "prompt update: N placeholders",
    "mm_hash of the item",
    "limit: --limit-mm-per-prompt",
    "processing/processor.py:1811-1847",
], col=BLUE, align="start")
d.box(X3, R1, BW, BH, "Processor cache (sender)", [
    "--mm-processor-cache-gb 4",
    "type lru: hit → send hash only,",
    "  engine core keeps the data",
    "> 1 API server, or DP without an",
    "  external LB: processor_only",
    "cache/factories.py:23-87",
], col=AMBER, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, BLUE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P_H + 22
d.panel(y1, P_H, 2, "How does the engine core schedule the encoder?",
        "With the prompt chunk that covers the placeholders, if the encoder cache and the encoder budget have room.")
R2 = y1 + 70
d.box(X1, R2, BW, BH, "Request with mm_features", [
    "identifier = mm_hash",
    "mm_position: offset, length N",
    "data: pixel_values (or None",
    "  on a processor-cache hit)",
    "prefix-cache key:",
    "  (\"mm\", hash, offset in block)",
], col=VIOLET, align="start")
d.box(X2, R2, BW, BH, "EncoderCacheManager", [
    "unit: encoder embeddings",
    "size = max(max_num_batched_tokens,",
    "  largest item)",
    "hit by hash → no encoder run",
    "free entries evicted lazily",
    "encoder_cache_manager.py:316-362",
], col=VIOLET, align="start")
d.box(X3, R2, BW, BH, "SchedulerOutput", [
    "scheduled_encoder_inputs:",
    "  {req_id: [input index]}",
    "free_encoder_mm_hashes",
    "  (entries to drop on workers)",
    "encoder compute budget",
    "  = encoder cache size",
], col=VIOLET, align="start")
d.arrow(X1 + BW, R2 + BH / 2, X2 - 2, R2 + BH / 2, VIOLET)
d.arrow(X2 + BW, R2 + BH / 2, X3 - 2, R2 + BH / 2, VIOLET)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P_H + 22
d.panel(y2, P_H, 3, "How do the image embeddings reach the decoder?",
        "The worker encodes the image once, caches the output by hash, and splices it in at the placeholder positions.")
R3 = y2 + 70
d.box(X1, R3, BW, BH, "EncoderRunner (gpu/mm)", [
    "prepare_mm_inputs: skip cached",
    "execute_mm_encoder:",
    "  model.embed_multimodal(...)",
    "  batched per modality",
    "encoder_outputs[hash] = [N, hidden]",
    "encoder_runner.py:81-176",
], col=TEAL, align="start")
d.box(X2, R3, BW, BH, "gather_mm_embeddings", [
    "slice of the item that lies in",
    "  this step's token window",
    "is_mm_embed mask [num_tokens]",
    "decodes only: nothing to gather",
    "missing entry → RuntimeError",
    "encoder_runner.py:207-294",
], col=TEAL, align="start")
d.box(X3, R3, BW, BH, "Decoder forward", [
    "embed_input_ids(input_ids,",
    "  mm_embeds, is_multimodal)",
    "  → inputs_embeds",
    "text rows: token embeddings",
    "placeholder rows: image embeds",
    "encoder_runner.py:296-308",
], col=EMERALD, align="start")
d.arrow(X1 + BW, R3 + BH / 2, X2 - 2, R3 + BH / 2, TEAL)
d.arrow(X2 + BW, R3 + BH / 2, X3 - 2, R3 + BH / 2, TEAL)

y = y2 + P_H + 24
d.legend(30, y, [(BLUE, "API server"), (AMBER, "cache across processes"), (VIOLET, "engine core"),
                 (TEAL, "worker: encoder"), (EMERALD, "worker: decoder")])
y += 40
y = d.footnotes(30, y, [
    "N and the processor output depend on the model and the image size; the flow and the cache rules come from the cited lines.",
    "A request whose prompt is chunked runs the encoder with the chunk that first reaches the placeholders; later chunks reuse the cached output.",
])
d.save(height=y + 14)
