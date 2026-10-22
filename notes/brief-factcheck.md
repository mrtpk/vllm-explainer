# Fact-checker brief (shared by every batch fact checker of "vLLM Explained")

You check a batch that another agent wrote. You did not write it. Be adversarial. The supervisor's prompt gives
the batch name, the files, and the writer's list of uncertain claims.

## Paths
- Explainer folder: /mnt/c/Users/tpesc/Downloads/test-harness/vllm-explainer (do not commit).
- Target repo (READ ONLY): /mnt/c/Users/tpesc/Downloads/test-harness/vllm at 3e44c0db26f3530d674d255dafabaedcf0e3d742.
- Sibling site (to check {{deepinfer}} recap lines): /mnt/c/Users/tpesc/Downloads/test-harness/deepinfer-explainer/src/

## Read first
STYLE.md §6-7; skills/codebase-explainer/references/writing-and-accuracy.md; notes/facts.md ([FC]/[S] = verified; [W] = not);
notes/brief-writer.md §"Accuracy lessons" (the defect types found so far: dataclass defaults overridden in arg_utils,
"always" vs "only when", off-by-one ranges, wrong default loop/sampler/CUDA-graph rule, TP=1 worker location).

## For every claim
(prose, stop titles, tables, captions, alt text, diagram labels, visual rules and texts, quiz answers, glossary text):
1. Open the cited lines. Does the code say exactly this? Is the citation the narrowest correct range?
2. Check numbers, defaults (trace to where the value is finally set), types, shapes, call order, which process/thread runs it, conditions.
3. Claims with no citation: add a citation or remove the claim.
4. Derived claims: check step by step; reword ("can", "at the earliest") if the derivation has gaps.
5. General domain statements (serving, PyTorch, Python, HTTP) are correct.
6. Every {{deepinfer id}} recap is true to that DeepInfer page.
7. Arithmetic in worked examples: recompute.
Fix defects in place in the batch's files only (pages, diagrams/<batch>_*.py, assets/viz/<visual>.js, src/glossary/<batch>.json).
Keep the writer's structure and ASD-STE100. Run `python3 build.py --preview` until the batch has 0 errors and no STE warnings.
If the batch has a visual or complex layout, check it at 1280 and 390 px with /tmp/claude-1000/vllm-foundation/shot.js
(outputs to /tmp/claude-1000/vllm-fc-<batch>/): no page overflow, no console errors, the visual reacts to click and keyboard.
Performance: slow mount; use git -C ../vllm grep -n and sed -n.

## Report (final message only)
- Number of claims checked and defects fixed, by type (citation, value, logic, missing source, other).
- The most important corrections (one line each).
- Corrections for notes/facts.md: confirm or refute the writer's NEW FACTS with file:line.
- Other pages that may repeat a wrong claim (for the FINAL-PASS TODO list).
- Anything you could not decide.
