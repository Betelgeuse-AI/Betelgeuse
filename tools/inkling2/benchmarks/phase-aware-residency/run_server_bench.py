#!/usr/bin/env python3
"""Run repeatable completion benchmarks against a local direct-checkpoint server."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
RAW = HERE / "raw"


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.load(response)


def read_ids(path: Path) -> list[int]:
    return [int(value) for value in path.read_text().replace("\n", "").split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--label", default="baseline")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 4096, 8192, 16384])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    log_path = RAW / f"{args.label}-chunk-{args.chunk}.jsonl"
    summaries: dict[int, list[dict[str, float]]] = {}
    first_outputs: dict[int, str] = {}

    with log_path.open("w") as log:
        for context in args.contexts:
            ids = read_ids(PROMPTS / f"context-{context}.ids")
            actual_context = len(ids)
            decoded = post(base, "/detokenize", {"tokens": ids})["content"]
            roundtrip = post(base, "/tokenize", {"content": decoded})["tokens"]
            prompt_path = PROMPTS / f"benchmark-context-{context}.txt"
            prompt_path.write_text(decoded)
            if len(roundtrip) != actual_context:
                raise SystemExit(f"{context}: tokenizer roundtrip changed token count to {len(roundtrip)}")

            rows: list[dict[str, float]] = []
            for run in range(1, args.repeats + 1):
                body = {
                    "prompt": decoded,
                    "max_tokens": 512,
                    "temperature": 0.0,
                    "top_k": 0,
                    "top_p": 1.0,
                    "min_p": 0.0,
                    "ignore_eos": True,
                    "cache_prompt": True,
                    "stream": False,
                }
                result = post(base, "/v1/completions", body)
                timings = result["timings"]
                if timings["prompt_n"] != actual_context:
                    raise SystemExit(
                        f"target {context}: server evaluated {timings['prompt_n']} tokens, expected {actual_context}"
                    )
                output = result["choices"][0]["text"]
                details = result.get("usage", {}).get("prompt_tokens_details", {})
                cache_n = timings.get("cache_n", details.get("cached_tokens", 0))
                row = {
                    "target_context": context,
                    "context": actual_context,
                    "run": run,
                    "prompt_n": timings["prompt_n"],
                    "predicted_n": timings["predicted_n"],
                    "prompt_ms": timings["prompt_ms"],
                    "predicted_ms": timings["predicted_ms"],
                    "pp_s": timings["prompt_per_second"],
                    "tg_s": timings["predicted_per_second"],
                    "cache_prompt": 1.0,
                    "cache_n": cache_n,
                    "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
                    "output_matches_first": output == first_outputs.get(context, output),
                    "chunk": float(args.chunk),
                }
                first_outputs.setdefault(context, output)
                log.write(json.dumps({"metrics": row, "response": result}, separators=(",", ":")) + "\n")
                log.flush()
                rows.append(row)
                print(
                    f"chunk={args.chunk} target={context} actual={actual_context} run={run} "
                    f"prompt={row['pp_s']:.1f} pp/s decode={row['tg_s']:.2f} tok/s "
                    f"TTFT={row['prompt_ms'] / 1000:.2f}s cached={cache_n} generated={row['predicted_n']} "
                    f"repeat_match={row['output_matches_first']}"
                )
            summaries[context] = rows

    print(f"raw results: {log_path}")
    for context, rows in summaries.items():
        print(
            f"median context={context} chunk={args.chunk} "
            f"pp/s={statistics.median(row['pp_s'] for row in rows):.1f} "
            f"tok/s={statistics.median(row['tg_s'] for row in rows):.2f} "
            f"TTFT={statistics.median(row['prompt_ms'] for row in rows) / 1000:.2f}s"
        )


if __name__ == "__main__":
    main()
