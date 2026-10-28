# Writer brief (shared by every batch writer of "<Site name>")

The supervisor's prompt gives you: the batch name, your pages and their scope, the visual (if any), and the
glossary terms you own. This file gives everything else. It is binding.

## Paths
- Explainer folder: <explainer path> (git repo; do not commit: the supervisor commits).
- Target repo (READ ONLY, pinned): <target path> at <full sha>. Never change it.
- Sibling site (optional, for recap links): <path or URL>

## Read first, in this order
1. STYLE.md (all of it).
2. plan.md §1-2 (goal and decisions) and the batch table (your batch).
3. instructions-svg.md (STYLE.md §10 overrides it: canvas 1100, min text 11).
4. notes/facts.md: [FC] and [S] entries are verified; [W] entries are not. Still open the code and cite it yourself.
5. The model page of your page type (finished and fact-checked): <concept> · <tour> · <module> · <basics> · <reference>.
6. skills/codebase-explainer/references/writing-and-accuracy.md and page-templates.md.
7. notes/survey/*.md: a guide with file:line hints. NOT a source.

## Content rules
- Reader: <what the reader knows>. Start simple (one sentence, the idea, a diagram); then a full Deep dive
  (types and shapes, data layouts, algorithm steps, edge cases, env vars and flags, a worked example with
  real numbers, line-by-line reading of key code).
- Every claim about the code: {{cite path:line}} or <x-code>. Open every cited line yourself. Narrowest range.
- No guesses. If the code does not answer a question, write it on the page as an open question and list it in your report.
- Scope: <in scope / out of scope>.
- Basics that the sibling site teaches (basics-covered.md): one-line recap + link. Never re-explain them.
- <x-simplified> for simplifications. Site text in <language style>. Captions start with "Figure: ".
- Diagrams: one script per diagram, Diagram(width=1100), max 3 boxes across, max 3 panels, no text below 11.
  Prefix = your batch (diagrams/<batch>_*.py).
- Literal tag syntax that is text ({{cite}}, <x-code>) must be escaped (&#123;&#123;cite&#125;&#125;).

## Accuracy lessons
(The supervisor adds each repeated defect type from the fact checks here, with the trap and the correct
place in the code. Start with these.)
1. DEFAULTS: trace each default to where it is finally set (argument parsing, post-init hooks, platform
   code). Do not cite only a config class field.
2. CONDITIONS: write "only when ..." where the code has a condition. No "always", "every", "all" unless you
   checked every path.
3. LINE RANGES: off-by-one is the most common defect. Open each cited range again after you finish the page.
4. COUNTS AND ORDER: "runs N times" or "before X" needs a check of every call site.
5. DOCS: docs are often older than the code. Cite the code. Report each docs-vs-code difference (FAQ material).

## Ownership (parallel work)
- You own only: your page files, src/glossary/<batch>.json, diagrams/<batch>_*.py, and your visual
  assets/viz/<name>.js (shared visual styles only appended to assets/viz/viz.css).
- Search src/glossary/*.json before you add a term. Do not redefine; reuse with {{term id}}.
- Do NOT edit: build.py, assets/site.css, assets/site.js, nav files, other glossary files, notes/*,
  STYLE.md, plan.md, progress.md, other pages.
- Use `python3 build.py --preview` when another page is broken. Your pages must have 0 errors and no
  template or language warnings.

## No subagents
Do NOT start subagents or forks. The user allows <N> running agents; you are one of them. Do all the work yourself.

## Claims that the code has a bug
Do NOT put a "this looks like a bug" claim on a page. Re-read the control flow twice (is the line inside or
after the loop or if? check the caller and the callee) and report it to the supervisor with exact lines.

## Report (final message only, no report files)
1. Pages written, diagrams, visual, glossary ids added.
2. Build result (errors and warnings for your pages).
3. NEW FACTS for notes/facts.md (each with file:line).
4. FACTS IN facts.md THAT ARE WRONG (with proof).
5. Claims you are less sure of (for the fact checker to check hard).
6. Open questions, docs-vs-code differences, cross-page problems.
