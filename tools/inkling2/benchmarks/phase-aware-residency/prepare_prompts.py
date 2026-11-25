#!/usr/bin/env python3
"""Build complete source-review prompts from the mmap runtime snapshot."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "prompts"
SOURCE_REV = "e1538e9954a191d352cae0697b5908b0e477c913"
TOKENIZE_URL = os.environ.get("INK2_TOKENIZE_URL", "http://127.0.0.1:18090/tokenize")
SIZES = (1024, 4096, 8192, 16384)

SYSTEM = """You are reviewing a real CUDA inference runtime. Give a concise, technically grounded analysis. Treat the code below as a production system: preserve output correctness, long-context safety, prompt caching, and the mmap-backed CPU expert path. Do not invent measurements. Do not write or continue source code; answer the review task at the end."""

USER_TASK = """Task: Review this direct-checkpoint Inkling-2-Mini runtime before implementing phase-aware VRAM residency. The target design gives prefill the workspace it needs for large prompt chunks, then reuses released GPU pages for more permanently resident decode experts. Dense weights, current expert slots, KV state, CPU workers, and server conversation caching must remain valid. The checkpoint experts stay mmap-backed in host RAM; do not pin the whole checkpoint. The source excerpts are unmodified files from the mmap-backed runtime at commit e1538e995. They are selected excerpts, ordered by execution flow; omitted code is not available. Excerpts may end at a source line boundary."""

REVIEW_REQUEST = """Review request: Based only on the supplied source, identify allocation ownership and lifetimes, which GPU pages can move from prefill workspace to decode expert slots, when that move is safe, and what must happen to CUDA graphs and future KV growth. State what must remain unchanged for CPU fallback and prompt-cache continuation. Separate source facts from hypotheses, and list the runtime evidence needed to validate the design. If an excerpt does not provide enough information, say so. Do not claim benchmark results. Answer the review now."""

SECTIONS = (
    ("tools/inkling2/ink2_engine.cu", 2388, 2510, "CUDA allocation helpers and growable VMM ranges"),
    ("tools/inkling2/ink2_engine.cu", 2540, 2665, "Engine ownership and expert pool state"),
    ("tools/inkling2/ink2_engine.cu", 3250, 3370, "Expert placement and initial offload pool allocation"),
    ("tools/inkling2/ink2_engine.cu", 3373, 3655, "KV sizing, pool resize, refill, and context growth"),
    ("tools/inkling2/ink2_engine.cu", 4475, 4700, "Prefill buffer allocation, mapping, fitting, and release"),
    ("tools/inkling2/ink2_engine.cu", 4825, 5060, "Prefill expert staging and grouped execution"),
    ("tools/inkling2/ink2_engine.cu", 5165, 5368, "Small-batch target verification and attention workspace"),
    ("tools/inkling2/ink2_engine.cu", 5430, 5555, "Optional draft model and its persistent allocations"),
    ("tools/inkling2/ink2_engine.cu", 5628, 5675, "Prefill entry and graph capture"),
    ("tools/inkling2/ink2_engine.cu", 5970, 6035, "Runtime initialization and initial decode graph"),
    ("tools/inkling2/ink2_engine.cu", 6305, 6350, "Server request prefill completion and transition to decode"),
    ("tools/inkling2/ink2_engine.cu", 6468, 6505, "Standalone prompt path and post-prompt release"),
    ("tools/inkling2/ink2_grow_guard.cuh", 1, 120, "Prompt and KV growth admission guard"),
    ("tools/inkling2/ink2_session.cuh", 1, 320, "Conversation cache and state restore"),
    ("tools/inkling2/ink2_cpu.hpp", 1, 120, "CPU expert doorbell and worker interfaces"),
    ("tools/server/server-context.cpp", 1445, 1565, "Direct checkpoint server configuration"),
    ("tools/inkling2/README.md", 1, 90, "Runtime and server behavior"),
)


def source(path: str, start: int, end: int) -> list[str]:
    content = subprocess.check_output(
        ["git", "show", f"{SOURCE_REV}:{path}"], cwd=ROOT, text=True
    )
    return content.splitlines()[start - 1 : end]


def tokenize(text: str) -> list[int]:
    body = json.dumps(
        {"content": text, "add_special": False, "parse_special": True}
    ).encode()
    request = urllib.request.Request(
        TOKENIZE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)["tokens"]


def make_prompt(code_lines: int, sections: list[tuple[str, int, str, list[str]]]) -> str:
    parts = ["<|message_system|><|content_text|>", SYSTEM, "<|end_message|>"]
    parts.extend(("<|message_user|><|content_text|>", USER_TASK, "\n\nSource excerpts:\n"))
    remaining = code_lines
    for path, start, title, lines in sections:
        if remaining <= 0:
            break
        take = min(remaining, len(lines))
        code = "\n".join(lines[:take])
        parts.append(
            f"\nSource file: {path}, lines {start}-{start + take - 1}. {title}.\n"
            f"```cpp\n{code}\n```\n"
        )
        remaining -= take
    parts.extend(("\n", REVIEW_REQUEST, "<|end_message|><|message_model|><|content_text|>"))
    return "".join(parts)


def fit_prompt(target: int, sections: list[tuple[str, int, str, list[str]]]) -> tuple[str, list[int]]:
    max_lines = sum(len(lines) for _, _, _, lines in sections)
    low = 0
    high = max_lines
    best: tuple[str, list[int]] | None = None
    while low <= high:
        middle = (low + high) // 2
        prompt = make_prompt(middle, sections)
        tokens = tokenize(prompt)
        if len(tokens) <= target:
            best = (prompt, tokens)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise SystemExit(f"fixed instructions exceed target context of {target} tokens")
    return best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sections = [
        (path, start, title, source(path, start, end))
        for path, start, end, title in SECTIONS
    ]
    for target in SIZES:
        prompt, tokens = fit_prompt(target, sections)
        ids_path = OUT / f"context-{target}.ids"
        ids_path.write_text(",".join(map(str, tokens)) + "\n")
        text_path = OUT / f"context-{target}.txt"
        text_path.write_text(prompt)
        if target == max(SIZES):
            (OUT / "source-review-prompt.txt").write_text(prompt)
        print(f"target {target}, actual {len(tokens)} tokens -> {ids_path}")


if __name__ == "__main__":
    main()
