#!/usr/bin/env python3
"""Create packed MTP weight files and BF16 reconstruction checkpoints for acceptance tests."""

import argparse
import json
import mmap
import struct
from pathlib import Path

import numpy as np


def read_header(path):
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    return header, 8 + size


def entries_for(header, mode):
    packed = []
    recon = []
    shapes = {}
    for name, item in header.items():
        if name == "__metadata__":
            continue
        shape = item["shape"]
        quantize = item["dtype"] == "BF16" and len(shape) == 2 and (
            name.startswith("layers.") or name == "markov_head.markov_w1.weight")
        if not quantize:
            spec = (name, item["dtype"], shape, "copy", name)
            packed.append(spec)
            recon.append(spec)
            continue
        rows, cols = shape
        shapes[name] = shape
        if mode == "fp8":
            packed.append((name, "F8_E4M3", shape, "fp8", name))
            packed.append((name + ".__scale", "F32", [rows], "fp8_scale", name))
        else:
            padded = (cols + 127) // 128 * 128
            groups = (cols + 127) // 128
            packed.append((name, "U8", [rows, padded // 2], "int4", name))
            packed.append((name + ".__scale", "F16", [rows, groups], "int4_scale", name))
        recon.append((name, "BF16", shape, mode + "_recon", name))
    return packed, recon, shapes


def make_header(specs, metadata):
    out = {"__metadata__": metadata}
    offset = 0
    for name, dtype, shape, role, source in specs:
        count = int(np.prod(shape, dtype=np.int64))
        size = count * {"BF16": 2, "F16": 2, "F32": 4, "U8": 1, "F8_E4M3": 1}[dtype]
        out[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(out, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((-len(raw)) % 8)
    return raw


def write_prefix(path, raw_header):
    f = path.open("xb")
    f.write(struct.pack("<Q", len(raw_header)))
    f.write(raw_header)
    return f


def validate_file(path):
    header, data_start = read_header(path)
    spans = sorted(item["data_offsets"] for name, item in header.items() if name != "__metadata__")
    end = 0
    for begin, next_end in spans:
        if begin != end:
            raise ValueError(f"{path}: tensor data has a gap or overlap at {end}")
        end = next_end
    if path.stat().st_size != data_start + end:
        raise ValueError(f"{path}: expected {data_start + end} bytes, got {path.stat().st_size}")


def bf16_to_f32(raw, shape):
    bits = np.frombuffer(raw, dtype="<u2").reshape(shape).astype(np.uint32) << 16
    return bits.view(np.float32)


def f32_to_bf16_bytes(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    rounded = bits + 0x7FFF + ((bits >> 16) & 1)
    return (rounded >> 16).astype("<u2").tobytes()


def fp8_lut():
    out = np.empty(256, dtype=np.float32)
    for code in range(256):
        exp, mant = (code >> 3) & 15, code & 7
        value = np.nan if exp == 15 and mant == 7 else (mant * 2.0**-9 if exp == 0 else (1.0 + mant / 8.0) * 2.0**(exp - 7))
        out[code] = -value if code & 0x80 else value
    return out


def encode_fp8(values):
    a = np.abs(values)
    sign = np.signbit(values).astype(np.uint8) << 7
    small = a < 2.0**-6
    code = np.zeros(a.shape, dtype=np.uint8)
    sub = np.rint(a[small] * 512.0).astype(np.int16)
    code[small] = np.where(sub >= 8, 8, sub).astype(np.uint8)
    normal = ~small
    v = a[normal]
    exp2 = np.floor(np.log2(v)).astype(np.int16)
    mant = np.rint((v / np.exp2(exp2) - 1.0) * 8.0).astype(np.int16)
    exp = exp2 + 7
    carry = mant == 8
    exp += carry
    mant[carry] = 0
    exp = np.clip(exp, 1, 15)
    mant = np.minimum(mant, 6)
    code[normal] = ((exp << 3) | mant).astype(np.uint8)
    return code | sign


def quantized_matrix(raw, shape, mode):
    x = bf16_to_f32(raw, shape)
    rows, cols = shape
    if mode == "fp8":
        scale = np.max(np.abs(x), axis=1).astype(np.float32) / 448.0
        scale[scale == 0] = 1.0
        codes = encode_fp8(x / scale[:, None])
        dequant = fp8_lut()[codes] * scale[:, None]
        return codes.tobytes(), scale.astype("<f4").tobytes(), f32_to_bf16_bytes(dequant)
    padded = (cols + 127) // 128 * 128
    grouped = np.zeros((rows, padded), dtype=np.float32)
    grouped[:, :cols] = x
    groups = grouped.reshape(rows, padded // 128, 128)
    scale = np.max(np.abs(groups), axis=2) / 7.0
    scale[scale == 0] = 1.0
    scale16 = scale.astype("<f2")
    scale16[scale16 == 0] = np.nextafter(np.float16(0), np.float16(1))
    q = np.rint(groups / scale16.astype(np.float32)[:, :, None]).clip(-7, 7).astype(np.int8).reshape(rows, padded)
    qcode = q.astype(np.uint8) & 15
    packed = qcode[:, 0::2] | (qcode[:, 1::2] << 4)
    dequant = q.reshape(rows, padded).astype(np.float32).reshape(rows, padded // 128, 128)
    dequant *= scale16.astype(np.float32)[:, :, None]
    return packed.tobytes(), scale16.tobytes(), f32_to_bf16_bytes(dequant.reshape(rows, padded)[:, :cols])


def run(src, models_out, tests_out, mode):
    src_header, data_start = read_header(src)
    packed_specs, recon_specs, logical_shapes = entries_for(src_header, mode)
    quantized_names = set(logical_shapes)
    packed_path = models_out / ("mtp-fp8-e4m3.safetensors" if mode == "fp8" else "mtp-int4-g128.safetensors")
    recon_path = tests_out / mode / "mtp.safetensors"
    if packed_path.exists() or recon_path.exists():
        raise FileExistsError("refusing to overwrite an existing quantization result")
    packed_meta = {"format": "inkling-mtp-fp8-e4m3-rowwise-v1" if mode == "fp8" else "inkling-mtp-int4-symmetric-g128-v1",
                   "source": src.name, "logical_shapes": json.dumps(logical_shapes, separators=(",", ":"))}
    recon_meta = {"format": "bf16-reconstruction", "quantization": mode}
    packed_f = write_prefix(packed_path, make_header(packed_specs, packed_meta))
    recon_path.parent.mkdir(parents=True, exist_ok=True)
    recon_f = write_prefix(recon_path, make_header(recon_specs, recon_meta))
    with src.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        for name, item in src_header.items():
            if name == "__metadata__":
                continue
            begin, end = item["data_offsets"]
            raw = mm[data_start + begin:data_start + end]
            if name in quantized_names:
                codes, scales, restored = quantized_matrix(raw, item["shape"], mode)
                packed_f.write(codes)
                packed_f.write(scales)
                recon_f.write(restored)
            else:
                packed_f.write(raw)
                recon_f.write(raw)
    packed_f.close()
    recon_f.close()
    validate_file(packed_path)
    validate_file(recon_path)
    sidecar = models_out / ("mtp-int4-g128-logical-shapes.json" if mode == "int4" else "mtp-fp8-e4m3-logical-shapes.json")
    sidecar.write_text(json.dumps({"format": packed_meta["format"], "source": str(src), "logical_shapes": logical_shapes}, indent=2) + "\n")
    return packed_path, recon_path


def link_test_model(base_model, tests_out, mode, recon_path):
    model_dir = tests_out / mode
    target = base_model / "model-00001-of-00001.safetensors"
    link = model_dir / "model-00001-of-00001.safetensors"
    if link.exists() or link.is_symlink():
        raise FileExistsError("refusing to replace test model link")
    link.symlink_to(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path.home() / "Models/Inkling-2-Mini")
    parser.add_argument("--models-out", type=Path, default=Path.home() / "Models/Inkling-2-Mini-Quantized")
    parser.add_argument("--tests-out", type=Path, default=Path.home() / "fluxaeros-data/inkling2-dspark-2026-11-23/quant-test")
    args = parser.parse_args()
    args.models_out.mkdir(parents=True, exist_ok=True)
    for mode in ("fp8", "int4"):
        packed, recon = run(args.model / "mtp.safetensors", args.models_out, args.tests_out, mode)
        link_test_model(args.model, args.tests_out, mode, recon)
        print(f"{mode}: packed={packed} ({packed.stat().st_size} bytes), test MTP={recon}")


if __name__ == "__main__":
    main()
