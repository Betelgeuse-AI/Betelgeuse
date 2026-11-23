#!/usr/bin/env python3
"""Rebuild physical-layer route counts from strata-ink2 --record-routing output."""
import argparse
import struct
from pathlib import Path

NL, NE, NCALLS, TOPK = 24, 128, 48, 8
LOOP_BEGIN, NLOOP = 6, 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("route_bin", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--start", type=int, required=True, help="first generated position in the prompt+decode trace")
    ap.add_argument("--tokens", type=int, required=True, help="number of generated positions to aggregate")
    a = ap.parse_args()
    raw = a.route_bin.read_bytes()
    expected = NCALLS * a.context * TOPK * 4
    if len(raw) != expected:
        raise SystemExit(f"expected {expected} bytes, got {len(raw)}")
    route = struct.unpack(f"<{expected // 4}i", raw)
    counts = [[0] * NE for _ in range(NL)]
    for slot in range(NCALLS):
        layer = slot if slot < NL else LOOP_BEGIN + (slot - NL) % NLOOP
        base = slot * a.context * TOPK
        for pos in range(a.start, a.start + a.tokens):
            off = base + pos * TOPK
            for expert in route[off:off + TOPK]:
                if 0 <= expert < NE:
                    counts[layer][expert] += 1
    a.output.write_text("".join(f"{l} {e} {counts[l][e]}\n" for l in range(NL) for e in range(NE)))
    print(f"wrote {NL * NE} layer/expert counts to {a.output}")


if __name__ == "__main__":
    main()
