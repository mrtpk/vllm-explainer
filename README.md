# vLLM Explained

A static website that teaches the latest [vLLM](https://github.com/vllm-project/vllm) codebase step by step.
vLLM is an inference and serving engine for large language models. The site focuses on online serving
with `vllm serve`, on how vLLM works inside, and on how plugins extend it.

The site is for a reader who knows Python and the LLM basics of its sibling site,
[DeepInfer Explained](https://explaindeepinfer.projects.thomaspaul.dev). Each page starts with a simple
explanation. Then a "Deep dive" gives the full detail for study: types and shapes, defaults and where they
are set, call order, process boundaries, log lines and edge cases.

> **Status:** complete. The site has 107 pages, 82 diagrams, 6 interactive visuals and 270 glossary terms.
> Every batch was checked against the code by a separate fact-check agent. See [`progress.md`](progress.md).

## Code version

The site explains one exact version of vLLM:

| | |
|---|---|
| Commit | [`3e44c0db26f3530d674d255dafabaedcf0e3d742`](https://github.com/vllm-project/vllm/tree/3e44c0db26f3530d674d255dafabaedcf0e3d742) |
| Date | 2026-10-22 |

All code excerpts and `file:line` links point to this commit. Line numbers can be different in newer versions.
The repository has no submodules.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 3e44c0db26f3530d674d255dafabaedcf0e3d742
```

## Read the site

Open `site/index.html` in a browser. The site works from disk (`file://`), with no server.
It needs an internet connection only for the Inter font. Without it, the site uses a system font.

| Part | Folder | Pages | What it contains |
|---|---|---|---|
| Learn the basics | `learn/` | 12 | vLLM-only basics: serving metrics, the OpenAI API, processes and IPC, entry points, torch.compile, prefix caching, TP/PP, quantized checkpoints, LoRA, HF checkpoints. Basics that DeepInfer Explained already teaches get a one-line recap and a link. |
| Concepts | `concepts/` | 48 | Architecture and engine · Plugins and extensibility (12 pages) · Model execution · Serving features · Scale and deployment · Operating vLLM (tuning, metrics, benchmarking, deployment, troubleshooting) |
| Guided tours | `tours/` | 7 | Life of a chat request · `vllm serve` start-up · One engine step · One forward pass · Life of a plugin · Build a toy plugin · One step across GPUs |
| Code map | `code/` | 31 | The folder tree and one page for each important module |
| Reference | `reference/` | 7 + glossary | Generated tables (env vars, `vllm serve` flags, config classes, Prometheus metrics), extension points, vLLM vs DeepInfer, FAQ (with a table of places where docs and code disagree) |

Interactive visuals: request stepper (Tour 1), scheduler simulator, prefix-cache explorer, plugin
start-up timeline, custom-op and attention-backend chooser, tuning playground.

Press `/` to search. Point to a dotted word to see its glossary definition.

## Scope

- **Latest code only.** V0 is gone (only aliases remain). Model Runner V2 is the default and the only
  runner explained; the Model Runner V1 fallback gets one cited note.
- **Out of scope:** kernel internals (`csrc/`, Triton/CUDA/CuTe/Helion kernel code). The pages explain how
  vLLM selects and calls kernels. The experimental Rust frontend and the per-family model packages in
  `vllm/models/` get one paragraph each.

## Build the site

Requirements:
- Python 3.11 or newer. The build uses only the standard library.
- A checkout of vLLM at the commit above, in a folder next to this one. For the `{{deepinfer}}` link check,
  also the DeepInfer explainer (optional; without it the build only warns):

```
parent/
  vllm/                   # git checkout at 3e44c0d (read only)
  vllm-explainer/         # this repository
  deepinfer-explainer/    # optional, for checking DeepInfer link ids
```

Commands:

```bash
cd vllm-explainer
python3 build.py                   # build site/ from src/
python3 build.py --strict          # also stop on warnings
python3 build.py --preview         # write site/ even with errors
python3 build.py --force-diagrams  # run all diagram scripts again
python3 tools/gen_env_vars.py      # regenerate a reference table (also gen_serve_flags, gen_config_classes, gen_metrics; --check to verify)
python3 tools/gen_folder_tree.py   # regenerate the code-map folder tree
```

The build reads code excerpts directly from `../vllm`, so each excerpt is exact. It stops with an error if
vLLM is at a different commit, or if a link, a code citation, a DeepInfer link or a glossary term is wrong.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Page sources (HTML fragments with a small header), navigation (`src/nav/`), glossary (`src/glossary/`) |
| `diagrams/` | One Python script per SVG diagram, and the shared helper `svglib.py` |
| `assets/` | Site CSS and JavaScript, and the interactive visuals (`assets/viz/`) |
| `tools/` | Generators for the reference tables (parse the vLLM source with `ast`; vLLM is not imported) |
| `notes/facts.md` | Facts checked in the code that all pages share, with the corrections found by fact checks |
| `notes/survey/` | The four codebase surveys made before planning (a guide, not a source) |
| `notes/brief-writer.md`, `notes/brief-factcheck.md` | The shared briefs for the writer and fact-checker agents |
| `notes/generated/` | JSON data written by the generators |
| `basics-covered.md` | What DeepInfer Explained already teaches, and the recap-and-link rule |
| `build.py` | The site generator |
| `site/` | The built site. Do not edit: the build writes it again each time |
| `STYLE.md` | The writer's guide: page templates, tags, writing rules, accuracy rules |
| `plan.md`, `progress.md` | The project plan and the progress log |
| `skills/codebase-explainer/` | The agent skill used to plan and build this site |

## How the content is made

- The pages are written in ASD-STE100 (Simplified Technical English): short sentences and simple words.
- Each claim about the code has a citation to a file and line at the pinned commit.
- After each batch of pages was written, a separate agent compared each claim with the code and corrected
  errors (typically 20–50 fixes per batch). A final pass made the pages consistent with each other.
- Docs in `docs/` are often older than the code. The pages follow the code, and the FAQ lists the
  differences that were found.

The pages were written with Claude Code (Anthropic) and reviewed against the source code.
vLLM is a project of the vLLM team. This site is not an official vLLM project.
