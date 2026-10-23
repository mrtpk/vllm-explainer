"""Generative vs pooling models: runner choice, model heads, and the HTTP endpoints of each task.

Used by: src/concepts/pooling-and-embeddings.html
Sources (at the pinned commit in explainer.json):
  vllm/config/model.py:86-88, 124-128, 163-169                 --runner / --convert options
  vllm/config/model.py:1283-1385, 2340-2357                    auto resolution by architecture suffix
  vllm/model_executor/model_loader/utils.py:324-333            convert embed / classify -> adapters
  vllm/model_executor/models/adapters.py:250-281, 307-378      as_embedding_model, as_seq_cls_model
  vllm/config/pooler.py:16-121                                 pooling types, dimensions, use_activation
  vllm/model_executor/layers/pooler/seqwise/heads.py:59-117    projector, matryoshka truncation, normalize
  vllm/v1/core/sched/scheduler.py:2263-2266                    pooling request stops at its first output
  vllm/entrypoints/pooling/factories.py:110-143                pooling routes per task
  vllm/entrypoints/pooling/utils.py:220-238                    scoring API rule
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svglib import AMBER, BLUE, EMERALD, TEAL, VIOLET, Diagram  # noqa: E402

d = Diagram(width=1100)
d.header("Generative and pooling models in vLLM",
         "One vLLM instance runs one runner type  ·  the pooling task decides the HTTP routes  ·  Model Runner V2",
         params="--runner auto | generate | pooling | draft  ·  --convert auto | none | embed | classify")

BW = 300
X1, X2, X3 = 46, 400, 754
BH = 136

# ---------------------------------------------------------------- panel 1
y0 = 124
P_H = 236
d.panel(y0, P_H, 1, "How does vLLM decide that a model pools?",
        "From --runner and --convert, else from the architecture name; convert wraps a generative class.")
R1 = y0 + 70
d.box(X1, R1, BW, BH, "--runner (default auto)", [
    "sentence-transformers config",
    "  → pooling",
    "else the registry decides",
    "else by name suffix:",
    "  ForCausalLM → generate",
    "model.py:1283-1329",
], col=BLUE, align="start")
d.box(X2, R1, BW, BH, "--convert (default auto)", [
    "ForSequenceClassification",
    "  → classify",
    "EmbeddingModel, *Model → embed",
    "pooling + causal LM arch → embed",
    "set convert → runner pooling",
    "model.py:1331-1385, 2340-2357",
], col=BLUE, align="start")
d.box(X3, R1, BW, BH, "Adapter class", [
    "embed → as_embedding_model(cls)",
    "  DispatchPooler.for_embedding",
    "classify → as_seq_cls_model(cls)",
    "  + score: ReplicatedLinear",
    "already pooling → unchanged",
    "adapters.py:250-378",
], col=VIOLET, align="start")
d.arrow(X1 + BW, R1 + BH / 2, X2 - 2, R1 + BH / 2, BLUE)
d.arrow(X2 + BW, R1 + BH / 2, X3 - 2, R1 + BH / 2, BLUE)

# ---------------------------------------------------------------- panel 2
y1 = y0 + P_H + 22
d.panel(y1, P_H, 2, "What differs at the end of the forward pass?",
        "A generative model samples one token per step; a pooling model reduces hidden states once and stops.")
R2 = y1 + 70
d.box(X1, R2, BW, BH, "Generative head", [
    "hidden → compute_logits",
    "  → Sampler → 1 token",
    "many steps per request",
    "logits processors, spec decode",
    "async scheduling on",
    "",
], col=AMBER, align="start")
d.box(X2, R2, BW, BH, "Pooler", [
    "sequence: LAST | MEAN | CLS",
    "  (model default LAST)",
    "token: ALL | STEP",
    "head: projector, dimensions",
    "  (matryoshka), normalize",
    "pooler.py:16-121",
], col=TEAL, align="start")
d.box(X3, R2, BW, BH, "One output, then done", [
    "pooler_output → request",
    "  FINISHED_STOPPED",
    "async scheduling off (default)",
    "no logits processors",
    "MEAN/CLS/STEP: chunked prefill",
    "  and prefix caching off (default)",
], col=TEAL, align="start")
d.arrow(X2 + BW, R2 + BH / 2, X3 - 2, R2 + BH / 2, TEAL)

# ---------------------------------------------------------------- panel 3
y2 = y1 + P_H + 22
d.panel(y2, P_H, 3, "Which HTTP routes does the server add?",
        "Generate routes for generative models; pooling routes chosen by the pooling task.")
R3 = y2 + 70
d.box(X1, R3, BW, BH, "Task generate", [
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses, ...",
    "",
    "no pooling routes",
    "",
], col=AMBER, align="start")
d.box(X2, R3, BW, BH, "Pooling task routes", [
    "/pooling  (any pooling task)",
    "/classify (classify)",
    "/v1/embeddings, /v2/embed",
    "  (embed)",
    "/v1/embeddings (token_embed)",
    "factories.py:110-143",
], col=EMERALD, align="start")
d.box(X3, R3, BW, BH, "Scoring routes", [
    "/score, /v1/score",
    "/rerank, /v1/rerank, /v2/rerank",
    "only for embed, token_embed,",
    "  or classify with",
    "  num_labels == 1",
    "pooling/utils.py:220-238",
], col=EMERALD, align="start")
d.arrow(X2 + BW, R3 + BH / 2, X3 - 2, R3 + BH / 2, EMERALD, dash="5 4")

y = y2 + P_H + 24
d.legend(30, y, [(BLUE, "config resolution"), (VIOLET, "model class"), (AMBER, "generative path"),
                 (TEAL, "pooling path"), (EMERALD, "pooling routes")])
y += 40
y = d.footnotes(30, y, [
    "The dashed arrow means \"also\": scoring routes are added next to the task routes when the rule holds.",
    "Chunked prefill and prefix caching rules are for decoder pooling models; bidirectional (encoder-only) models get neither.",
])
d.save(height=y + 14)
