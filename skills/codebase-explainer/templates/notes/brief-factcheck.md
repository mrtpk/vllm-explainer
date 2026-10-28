# Fact-checker brief (shared by every batch fact checker of "<Site name>")

You check a batch that another agent wrote. You did not write it. Be adversarial. The supervisor's prompt
gives the batch name, the files, and the writer's list of uncertain claims.

## Paths
- Explainer folder: <explainer path> (do not commit).
- Target repo (READ ONLY): <target path> at <full sha>.
- Sibling site (optional, to check recap lines): <path or URL>

## Read first
STYLE.md §6-7; skills/codebase-explainer/references/writing-and-accuracy.md; notes/facts.md ([FC]/[S] =
verified; [W] = not); notes/brief-writer.md §"Accuracy lessons" (the defect types found so far).

## For every claim
(prose, stop titles, tables, captions, alt text, diagram labels, visual rules and texts, quiz answers, glossary text):
1. Open the cited lines. Does the code say exactly this? Is the citation the narrowest correct range?
2. Check numbers, defaults (trace to where the value is finally set), types, shapes, call order, which
   process or thread runs it, conditions.
3. Claims with no citation: add a citation or remove the claim.
4. Derived claims: check step by step; reword ("can", "at the earliest") if the derivation has gaps.
5. General domain statements are correct.
6. Every recap link to the sibling site is true to that page.
7. Arithmetic in worked examples: recompute.
Fix defects in place in the batch's files only (pages, diagrams/<batch>_*.py, assets/viz/<visual>.js,
src/glossary/<batch>.json). Keep the writer's structure and the language rules. Run
`python3 build.py --preview` until the batch has 0 errors and no language warnings.

## No subagents
Do NOT start subagents or forks. The user allows <N> running agents; you are one of them. Do all the work yourself.

## Claims that the code has a bug
Do NOT put a "this looks like a bug" claim on a page. Re-read the control flow twice (indentation: is the
line inside or after the loop or if? the caller and the callee) and report it to the supervisor with exact
lines. Fact checkers can be wrong too.

## Report (final message only)
- Number of claims checked and defects fixed, by type (citation, value, logic, missing source, other).
- The most important corrections (one line each).
- Corrections for notes/facts.md: confirm or refute the writer's NEW FACTS with file:line.
- Other pages that may repeat a wrong claim (for the FINAL-PASS TODO list).
- Anything you could not decide.
