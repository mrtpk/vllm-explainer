#!/usr/bin/env python3
"""Generate the folder tree of src/code/index.html from the target repo at the pinned commit.

PATTERN for every reference generator (env vars, config fields, CLI flags, API tables):
  1. read the facts from the code (here: `git ls-tree` at the pinned commit, and line counts);
  2. merge them with hand-written text (here: DESCRIPTIONS below, one short sentence per folder);
  3. write HTML between markers in a page source:
       <!-- BEGIN GENERATED folder-tree --> ... <!-- END GENERATED folder-tree -->
  4. --check: exit 1 if the page is out of date (run it in the final pass).

Python 3 standard library only. Reads explainer.json for the repo path and the commit.

Usage:
  python3 tools/gen_folder_tree.py             # write the page
  python3 tools/gen_folder_tree.py --check     # only check
  python3 tools/gen_folder_tree.py --depth 3   # show deeper folders (default 2)
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "explainer.json").read_text(encoding="utf-8"))
REPO = (ROOT / CFG["repo_path"]).resolve()
SHA = CFG["sha"]
PAGE = ROOT / "src" / "code" / "index.html"
BEGIN, END = "<!-- BEGIN GENERATED folder-tree -->", "<!-- END GENERATED folder-tree -->"

# Hand-written text: one short sentence per folder (site language rules apply).
# Optional page link: "folder": ("Description.", "code/page-id")
DESCRIPTIONS: dict[str, str | tuple[str, str]] = {
    "": "The repository root.",
    # --- top level -------------------------------------------------------------------------------
    ".agents": "Skill files for AI coding agents that work on the vLLM repository. Not part of vLLM at run time.",
    ".buildkite": "CI pipelines and scripts for Buildkite (tests on GPU machines, image builds, benchmarks).",
    ".claude": "Links to the agent skills in .agents/, for Claude Code. Not part of vLLM at run time.",
    ".github": "GitHub settings: issue templates and GitHub Actions workflows.",
    "benchmarks": "Stand-alone benchmark scripts, most of them for single kernels. The serving benchmark is vllm bench (vllm/benchmarks/).",
    "cmake": "CMake helpers for the C++/CUDA build of csrc/ (out of scope).",
    "csrc": "C++ and CUDA kernels, compiled into vllm._C. Out of scope: the site explains only how vLLM calls them.",
    "docker": "Dockerfiles for each hardware type (CUDA, ROCm, CPU, ...) and container entry scripts.",
    "docs": "The user and design documentation (mkdocs). Useful, but the site cites the code for behaviour.",
    "examples": "Runnable examples: online serving clients, offline inference, features, deployment, observability.",
    "recipes": "Deployment recipes: one YAML file per Hugging Face model with tested vllm serve settings.",
    "requirements": "Python requirement files for each hardware type, for the build and for the tests.",
    "rust": "Experimental Rust OpenAI frontend. It talks to the Python engine core over the same ZMQ boundary. One paragraph only on this site.",
    "tests": "The test suite. tests/plugins/ has small example plugin packages; tests/v1/ tests the V1 engine.",
    "tools": "Developer tools: pre-commit hooks, profiling helpers, recipe tuning, EP kernel install scripts.",
    "vllm": "The Python package vllm: everything that runs when you call vllm serve.",
    # --- vllm/ -----------------------------------------------------------------------------------
    "vllm/assets": "Sample images, audio and video for tests and examples.",
    "vllm/benchmarks": "The vllm bench commands: serving, throughput and latency benchmarks.",
    "vllm/compilation": "torch.compile integration: the compile backend, graph splitting, fusion passes, CUDA graph wrappers.",
    "vllm/config": ("The @config dataclasses (ModelConfig, CacheConfig, ParallelConfig, ...) that make up VllmConfig.", "code/config-vllm"),
    "vllm/cute_utils": "Small helpers for CuTe DSL kernels (out of scope).",
    "vllm/device_allocator": "GPU memory allocators for sleep mode (cumem, xpumem) and the sleep-mode backend registry.",
    "vllm/distributed": ("Process groups and communicators (TP, PP, DP, EP), KV transfer connectors, EPLB, elastic EP.", "code/distributed-parallel-state"),
    "vllm/engine": ("EngineArgs and the CLI flags (arg_utils.py). The other files are aliases to the V1 engine classes.", "code/engine-arg-utils"),
    "vllm/entrypoints": ("The CLI (vllm serve, bench, ...), the OpenAI-compatible API server, other APIs and the offline LLM class.", "code/entrypoints-cli-serve"),
    "vllm/inputs": "Types for prompts and engine inputs.",
    "vllm/ir": "vLLM IR: ops declared with register_op. Each op has implementations (providers); a priority list, with defaults from the platform, picks one.",
    "vllm/kernels": "Kernel code: IR op providers (vllm_c, aiter, oink, triton) and the Helion kernel integration (out of scope inside).",
    "vllm/logging_utils": "Logging helpers: log format, request-input dump, uvicorn log setup, timing.",
    "vllm/lora": "LoRA adapter support: LoRA layers, the adapter manager, Punica kernel wrappers, the LoRA resolver registry.",
    "vllm/model_executor": ("Model code and the layers it uses: model definitions, linear and attention layers, quantization, CustomOp, model loaders.", "code/model-executor-custom-op"),
    "vllm/models": "New per-family model packages (for example deepseek_v4/) with nvidia/ and amd/ subfolders. One paragraph only on this site.",
    "vllm/multimodal": "Multimodal inputs: processors, the multimodal registry, input hashing and caches.",
    "vllm/parser": "Unified output parsers: one Parser object does reasoning and tool-call extraction for the serving layer.",
    "vllm/platforms": ("Platform classes (CUDA, ROCm, XPU, CPU, TPU) and current_platform with plugin detection.", "code/platforms-init"),
    "vllm/plugins": ("Plugin loading by entry-point group, plus endpoint, IO processor and LoRA resolver plugins.", "code/plugins-init"),
    "vllm/profiler": "Torch and layer-wise profiler helpers.",
    "vllm/ray": "Small helpers for Ray (lazy import, environment for Ray workers).",
    "vllm/reasoning": "Reasoning parsers that split model output into reasoning content and the answer.",
    "vllm/renderers": "Renderers: turn chat messages into a prompt (chat templates) and tokens, for the API server.",
    "vllm/snapshot": "Create, inspect and restore snapshots of an initialized engine (vllm snapshot).",
    "vllm/third_party": "Vendored third-party code (FlashMLA, flash linear attention, pynvml). Out of scope.",
    "vllm/tilelang_utils": "Helpers for TileLang kernels (out of scope).",
    "vllm/tokenizers": "Tokenizer wrappers (Hugging Face and model-specific) and detokenizer helpers.",
    "vllm/tool_parsers": "Tool-call parsers, one per model family, that read tool calls from the model output.",
    "vllm/tracing": "OpenTelemetry tracing helpers.",
    "vllm/transformers_utils": "Helpers around Hugging Face transformers: config loading, custom configs, repo files.",
    "vllm/triton_utils": "Helpers to import and configure Triton (out of scope inside kernels).",
    "vllm/usage": "Anonymous usage statistics (usage_lib.py).",
    "vllm/utils": "General helpers: argparse, import, network, system and torch utilities.",
    "vllm/v1": ("The V1 engine: AsyncLLM and EngineCore, the scheduler and KV cache manager, executors, workers and Model Runner V2.", "code/v1-engine-async-llm"),
    "vllm/vllm_flash_attn": "The Python interface of vLLM's FlashAttention build (out of scope inside).",
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-tree", "-r", "--name-only", SHA],
                         capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        sys.exit(f"git ls-tree failed: {out.stderr.strip()}")
    return [p for p in out.stdout.splitlines() if (REPO / p).is_file()]


def line_count(path: str) -> int:
    data = (REPO / path).read_bytes()
    if b"\0" in data[:4096]:
        return 0  # binary
    return data.count(b"\n") + (0 if not data or data.endswith(b"\n") else 1)


def build_tree(files: list[str], depth: int) -> dict:
    tree: dict = {"files": 0, "lines": 0, "dirs": {}}
    for f in files:
        n = line_count(f)
        node = tree
        node["files"] += 1
        node["lines"] += n
        for part in f.split("/")[:-1][:depth]:
            node = node["dirs"].setdefault(part, {"files": 0, "lines": 0, "dirs": {}})
            node["files"] += 1
            node["lines"] += n
    return tree


def render(node: dict, prefix: str) -> str:
    items = []
    for name in sorted(node["dirs"]):
        sub = node["dirs"][name]
        path = f"{prefix}{name}/"
        desc = DESCRIPTIONS.get(path.rstrip("/"), "")
        link = ""
        if isinstance(desc, tuple):
            desc, page = desc
            link = f" {{{{link {page} | page}}}}"
        cite = f"{{{{cite {path} | {html.escape(path)}}}}}"
        text = f" {html.escape(desc)}" if desc else ""
        children = render(sub, path) if sub["dirs"] else ""
        items.append(f'<li>{cite} <span class="ct-n">{sub["files"]:,} files · {sub["lines"]:,} lines</span>'
                     f"{text}{link}{children}</li>")
    return "<ul>" + "".join(items) + "</ul>" if items else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--depth", type=int, default=2)
    args = ap.parse_args()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != SHA:
        sys.exit(f"{REPO} is at {head[:12]}, but explainer.json pins {SHA[:12]}")
    tree = build_tree(tracked_files(), args.depth)
    body = (f'<p class="ct-total">{tree["files"]:,} tracked files, {tree["lines"]:,} lines '
            f"(text files only; counted with <code>git ls-tree</code> at the pinned commit).</p>\n"
            f'<div class="ct-tree">{render(tree, "")}</div>')
    page = PAGE.read_text(encoding="utf-8")
    if BEGIN not in page or END not in page:
        sys.exit(f"{PAGE}: markers {BEGIN} / {END} not found")
    start = page.index(BEGIN) + len(BEGIN)
    new = page[:start] + "\n" + body + "\n" + page[page.index(END):]
    if args.check:
        if new != page:
            print(f"{PAGE.relative_to(ROOT)} is out of date: run python3 tools/gen_folder_tree.py")
            return 1
        print("folder tree is up to date")
        return 0
    PAGE.write_text(new, encoding="utf-8")
    print(f"wrote {PAGE.relative_to(ROOT)}: {tree['files']} files, {tree['lines']} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
