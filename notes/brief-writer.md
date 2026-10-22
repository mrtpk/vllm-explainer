# Writer brief (shared by every batch writer of "vLLM Explained")

The supervisor's prompt gives you: the batch name, your pages and their scope, the visual (if any), and the
glossary terms you own. This file gives everything else. It is binding.

## Paths
- Explainer folder: /mnt/c/Users/tpesc/Downloads/test-harness/vllm-explainer (git repo; do not commit — the supervisor commits).
- Target repo (READ ONLY, pinned): /mnt/c/Users/tpesc/Downloads/test-harness/vllm at 3e44c0db26f3530d674d255dafabaedcf0e3d742. Never change it.
- Sibling site (for {{deepinfer}} recap links and as a model of depth): /mnt/c/Users/tpesc/Downloads/test-harness/deepinfer-explainer

## Read first, in this order
1. STYLE.md (all of it; §1.1 project rules; §4 templates; §5 tags; §6 ASD-STE100; §10 diagrams).
2. plan.md §1-2 (goal: proficiency in ONLINE serving with the LATEST vLLM; plugins; how vLLM works) and §12 (your batch).
3. instructions-svg.md (STYLE.md §10 overrides it: canvas 1100, min text 11).
4. notes/facts.md — shared facts. [FC ...] and [S] entries are verified; [W ...] are not. Still open the code and cite it yourself.
5. The model page of your page type (finished and fact-checked):
   concept `src/concepts/plugin-loading.html` · tour `src/tours/life-of-a-chat-request.html` ·
   module `src/code/plugins-init.html` · basics `src/learn/serving-metrics.html` · reference `src/reference/env-vars.html`.
6. skills/codebase-explainer/references/writing-and-accuracy.md and page-templates.md.
7. notes/survey/*.md — a guide with file:line hints. NOT a source: line numbers can be off.

## Content rules
- Reader: knows Python and the LLM basics of DeepInfer Explained. Start simple (one sentence, the idea, an analogy, a diagram); then a full Deep dive (types and shapes, data layouts, algorithm steps, edge cases, env vars and flags, a worked example with real numbers, line-by-line reading of key code).
- Every claim about the code: {{cite path:line}} or <x-code>. Open every cited line yourself. Use the narrowest correct range.
- No guesses. If the code does not answer a question, write it on the page as an open question and list it in your report.
- Latest code only: no V0; Model Runner V2 only (vllm/v1/worker/gpu/). Kernel internals are out of scope: explain only how vLLM selects and calls kernels.
- Basics that DeepInfer Explained teaches (basics-covered.md, STYLE.md §1.1): one-line recap + {{deepinfer <id> | title}} link. Never re-explain them.
- <x-simplified> for simplifications. ASD-STE100: max 25 words per sentence, simple words. Captions start with "Figure: ".
- Diagrams: one Python script per diagram with svglib, Diagram(width=1100), max 3 boxes across, max 3 panels, no text below 11; split big diagrams. Prefix = your batch (e.g. diagrams/b2a_*.py).
- Link to other pages with {{link <id>}}. Planned pages (in src/nav/*.json) that do not exist yet give warnings only.

## Accuracy lessons from the Phase 1 fact checks (30-40% of claims needed a fix)
1. DEFAULTS: never cite only a config dataclass default. Many real defaults are set later in vllm/engine/arg_utils.py (create_engine_config and helpers) or in VllmConfig.__post_init__: chunked prefill (arg_utils ~3031-3035 + model.py ~2229-2236), prefix caching (~3062-3063 + model.py ~2275-2291), max_num_seqs / max_num_batched_tokens (by usage context and GPU memory ~2961-2993, --performance-mode ~3171-3175), async scheduling (config/vllm.py ~1664-1698), CUDA graph mode (optimization level O2 → FULL_AND_PIECEWISE). Trace the value to where it is finally set.
2. CONDITIONS: write "only when ..." where the code has a condition. Do not write "always", "every", "all" unless you checked every path.
3. LINE RANGES: off-by-one is the most common defect. Re-open each cited range after you finish the page.
4. Known traps: the default engine loop is step_with_batch_queue (async scheduling), not EngineCore.step(); with TP=1 the worker runs inside the engine core process (uni executor); the default sampler is the FlashInfer fused top-k/top-p sampler when supported (Gumbel is the fallback); FULL CUDA graphs only for uniform 1-token decode batches up to the largest captured size; VLLM_PLUGINS does not filter vllm.logits_processors; the engine core only gives timestamps — the API server computes latency metrics.
5. Docs in docs/ are often stale. Cite code for behaviour. If docs disagree with code, say so on the page (it is FAQ material) and in your report.

## Ownership (parallel work)
- You own only: your page files, src/glossary/<batch>.json, diagrams/<batch>_*.py, and your visual assets/viz/<name>.js (if any; shared visual styles only appended to assets/viz/viz.css).
- Before adding a glossary term, search src/glossary/*.json. Do not redefine; reuse with {{term id}}. Report wrong terms instead of editing other files.
- Do NOT edit: build.py, assets/site.css, assets/site.js, nav files, other glossary files, notes/*, STYLE.md, plan.md, progress.md, other pages.
- Another agent may work at the same time: use `python3 build.py --preview` when another page is broken. Your pages must have 0 errors and no template or STE warnings (link-not-written-yet warnings are fine).
- Performance: both repos are on a slow Windows mount. Use `git -C ../vllm grep -n`, `sed -n 'a,bp'`; never cat or wc the whole tree.
- Browser check (optional but welcome): /tmp/claude-1000/vllm-foundation/shot.js works with the LD_LIBRARY_PATH set up in that folder (see its scripts). Use your own output folder /tmp/claude-1000/vllm-<batch>/.

## Report (final message only, no report files)
1. Pages written, diagrams, visual, glossary ids added.
2. Build result (errors and warnings for your pages).
3. NEW FACTS for notes/facts.md (each with file:line).
4. FACTS IN facts.md THAT ARE WRONG (with proof).
5. Claims you are less sure of (for the fact checker to check hard).
6. Open questions, docs-vs-code differences, cross-page problems.
