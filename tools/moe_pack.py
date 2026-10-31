#!/usr/bin/env python3

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import struct
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
from gguf import GGUFEndian, GGUFReader, GGUFWriter


BUNDLE_DENSE = "Qwen3.8-Flash-Next-Dense.gguf"
BUNDLE_PLE = "Qwen3.8-Flash-Next-PLE.gguf"
EXPERT_PATTERN = re.compile(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight")


def expert_layout(model):
    reader = GGUFReader(model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    layers = sorted({int(match.group(1)) for name in tensors
                     if (match := EXPERT_PATTERN.fullmatch(name))})
    if not layers or layers != list(range(len(layers))):
        raise ValueError("requires contiguous expert layers starting at zero in the first GGUF shard")
    layout = []
    offset = 0
    experts = None
    for layer in layers:
        names = [f"blk.{layer}.ffn_{role}_exps.weight" for role in ("gate", "up", "down")]
        if any(name not in tensors for name in names):
            raise ValueError(f"layer {layer}: requires separate gate, up and down tensors in this shard")
        roles = [tensors[name] for name in names]
        if any(len(tensor.shape) != 3 for tensor in roles):
            raise ValueError(f"layer {layer}: expert tensors must have three dimensions")
        count = int(roles[0].shape[2])
        if count < 1 or any(int(tensor.shape[2]) != count for tensor in roles):
            raise ValueError(f"layer {layer}: inconsistent expert counts")
        if experts is not None and count != experts:
            raise ValueError("all layers must use the same expert count")
        experts = count
        if roles[0].tensor_type != roles[1].tensor_type or tuple(roles[0].shape) != tuple(roles[1].shape):
            raise ValueError(f"layer {layer}: gate and up must have the same shape and type")
        if tuple(roles[2].shape[:2]) != tuple(reversed(roles[0].shape[:2])):
            raise ValueError(f"layer {layer}: incompatible down projection shape")
        if any(tensor.n_bytes % count for tensor in roles):
            raise ValueError(f"layer {layer}: expert byte size is not integral")
        sizes = [tensor.n_bytes // count for tensor in roles]
        blob = sum(sizes)
        if blob <= 0 or blob > 256 * 1024 * 1024:
            raise ValueError(f"layer {layer}: unsupported expert byte size {blob}")
        layout.append({"layer": layer, "offset": offset, "blob": blob, "sizes": sizes,
                       "source_offsets": [tensor.data_offset for tensor in roles],
                       "types": [int(tensor.tensor_type) for tensor in roles]})
        offset += count * blob
    return layout, experts, offset


def index_text(layout):
    header = "# native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off\n"
    return header + "".join(" ".join(str(value) for value in (
        row["layer"], row["types"][0], row["types"][2], row["offset"], row["blob"], *row["source_offsets"]
    )) + "\n" for row in layout)


def read_expert(source, row, expert):
    parts = []
    for offset, size in zip(row["source_offsets"], row["sizes"]):
        source.seek(offset + expert * size)
        data = source.read(size)
        if len(data) != size:
            raise OSError(f"short source read: layer {row['layer']}, expert {expert}")
        parts.append(data)
    return b"".join(parts)


def create_pack(model, output):
    layout, experts, total = expert_layout(model)
    padded = (total + 4095) // 4096 * 4096
    output.mkdir(parents=True, exist_ok=True)
    names = ("experts.bin", "expert-pack.json", "native_experts.txt")
    if any((output / name).exists() for name in names):
        raise FileExistsError("output pack already exists; use --verify or choose an empty directory")
    if shutil.disk_usage(output).free < padded + 1024 * 1024:
        raise OSError(f"not enough free space: need at least {padded / 2**30:.2f} GiB")
    with tempfile.TemporaryDirectory(prefix=".moe-pack-", dir=output) as temporary:
        temporary = Path(temporary)
        digest = hashlib.sha256()
        with model.open("rb") as source, (temporary / "experts.bin").open("xb") as target:
            for row in layout:
                for expert in range(experts):
                    data = read_expert(source, row, expert)
                    target.write(data)
                    digest.update(data)
                print(f"packed layer {row['layer'] + 1}/{len(layout)}", flush=True)
            padding = bytes(padded - total)
            target.write(padding)
            digest.update(padding)
            target.flush()
            os.fsync(target.fileno())
        (temporary / "native_experts.txt").write_text(index_text(layout), encoding="ascii")
        metadata = {"format": "native-experts-v3", "source": model.name, "source_bytes": model.stat().st_size,
                    "layers": len(layout), "experts_per_layer": experts, "bytes": padded,
                    "sha256": digest.hexdigest(), "weights": "unaltered source bytes"}
        (temporary / "expert-pack.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        published = []
        try:
            for name in names:
                destination = output / name
                os.link(temporary / name, destination)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink()
            raise
    return metadata


def verify_pack(model, output, verify_all=False):
    layout, experts, total = expert_layout(model)
    if (output / "experts.bin").stat().st_size != (total + 4095) // 4096 * 4096:
        raise ValueError("pack byte size does not match the model")
    actual = [line.split()[:8] for line in (output / "native_experts.txt").read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    expected = [line.split() for line in index_text(layout).splitlines()[1:]]
    if actual != expected:
        raise ValueError("pack index does not match the model")
    selected = range(experts) if verify_all else sorted({0, experts // 2, experts - 1})
    checked = 0
    with model.open("rb") as source, (output / "experts.bin").open("rb") as packed:
        for row in layout:
            for expert in selected:
                packed.seek(row["offset"] + expert * row["blob"])
                if packed.read(row["blob"]) != read_expert(source, row, expert):
                    raise ValueError(f"weight mismatch: layer {row['layer']}, expert {expert}")
                checked += 1
    return {"verified_experts": checked, "total_experts": len(layout) * experts, "all": verify_all}


def copy_tensor_bytes(source, target, offset, size):
    target.flush()
    destination = target.tell()
    copied = 0
    if hasattr(os, "copy_file_range"):
        while copied < size:
            try:
                count = os.copy_file_range(source.fileno(), target.fileno(), min(size - copied, 64 * 1024 * 1024),
                                           offset + copied, destination + copied)
            except OSError as error:
                if error.errno not in (errno.EXDEV, errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                    raise
                break
            if not count:
                raise OSError("short source read while copying dense tensor")
            copied += count
    source.seek(offset + copied)
    target.seek(destination + copied)
    while copied < size:
        data = source.read(min(size - copied, 64 * 1024 * 1024))
        if not data:
            raise OSError("short source read while copying dense tensor")
        target.write(data)
        copied += len(data)


def reuse_file(source, target):
    try:
        os.link(source, target)
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
        target.symlink_to(source.resolve(strict=True))


def write_stripped_dense(shard, reader, output):
    writer = GGUFWriter(output, reader.fields["general.architecture"].contents(), endianess=reader.endianess)
    writer.data_alignment = int(reader.alignment)
    writer.kv_data[0].clear()
    for key, field in reader.fields.items():
        if key not in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            writer.add_key_value(key, field.contents(), field.types[0], field.types[1] if len(field.types) > 1 else None)
    writer.add_bool("betelgeuse.stripped_experts", True)
    writer.add_string("betelgeuse.ple_file", BUNDLE_PLE)
    writer.add_uint32("betelgeuse.expert_alignment", 4096)
    dense = [tensor for tensor in reader.tensors if not EXPERT_PATTERN.fullmatch(tensor.name)]
    if not dense:
        raise ValueError("first shard has no dense tensors")
    tensors = dense + [tensor for tensor in reader.tensors if EXPERT_PATTERN.fullmatch(tensor.name)]
    offsets = {}
    offset = 0
    for tensor in tensors:
        offsets[tensor.name] = offset
        offset += writer.ggml_pad(tensor.n_bytes, writer.data_alignment)
        writer.add_tensor_info(tensor.name, tuple(reversed(tensor.shape)), np.dtype(np.float32), tensor.n_bytes,
                               raw_dtype=tensor.tensor_type)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        target = writer.fout[0]
        byte_order = "<" if reader.endianess == GGUFEndian.LITTLE else ">"
        # Keep descriptors contiguous so the standard GGUF parser accepts metadata-only loading.
        for tensor in tensors:
            for part in tensor.field.parts[:-1]:
                target.write(part.tobytes())
            target.write(struct.pack(byte_order + "Q", offsets[tensor.name]))
        writer.write_padding(target, target.tell())
        with shard.open("rb") as source:
            for tensor in dense:
                copy_tensor_bytes(source, target, tensor.data_offset, tensor.n_bytes)
                writer.write_padding(target, tensor.n_bytes)
        target.flush()
        os.fsync(target.fileno())
    finally:
        writer.close()


def write_aligned_pack(shard, output, layout, experts, existing_pack=None):
    source_path = shard if existing_pack is None else existing_pack / "experts.bin"
    with source_path.open("rb") as source, output.open("xb") as target:
        for row in layout:
            padding = bytes((-row["blob"]) % 4096)
            for expert in range(experts):
                if existing_pack is None:
                    data = read_expert(source, row, expert)
                else:
                    source.seek(row["offset"] + expert * row["blob"])
                    data = source.read(row["blob"])
                    if len(data) != row["blob"]:
                        raise OSError(f"short pack read: layer {row['layer']}, expert {expert}")
                target.write(data)
                target.write(padding)
            print(f"aligned layer {row['layer'] + 1}/{len(layout)}", flush=True)
        target.flush()
        os.fsync(target.fileno())


def create_bundle(shard1, shard2, output, existing_pack=None):
    """Preserve dense/tokenizer bytes and reuse verified companions without modifying sources."""
    shard1 = shard1.resolve(strict=True)
    shard2 = shard2.resolve(strict=True)
    reader = GGUFReader(shard1)
    ple = GGUFReader(shard2)
    layout, experts, total = expert_layout(shard1)
    for index, source in enumerate((reader, ple)):
        for key, expected in (("split.no", index), ("split.count", 2),
                              ("split.tensors.count", len(reader.tensors) + len(ple.tensors))):
            field = source.fields.get(key)
            if field is None or field.contents() != expected:
                raise ValueError(f"shard {index + 1}: incompatible {key}")
    if {tensor.name for tensor in ple.tensors} != {"per_layer_token_embd.weight"}:
        raise ValueError("second shard must contain only per_layer_token_embd.weight")
    if any(tensor.name == "per_layer_token_embd.weight" for tensor in reader.tensors):
        raise ValueError("PLE tensor is duplicated in the first shard")
    if existing_pack is not None:
        existing_pack = existing_pack.resolve(strict=True)
        verify_pack(shard1, existing_pack)
    names = (BUNDLE_DENSE, "experts.bin", BUNDLE_PLE)
    output.mkdir(parents=True, exist_ok=True)
    if any(os.path.lexists(output / name) for name in names):
        raise FileExistsError("bundle output already exists; choose an empty directory")
    with tempfile.TemporaryDirectory(prefix=".betelgeuse-", dir=output) as temporary:
        temporary = Path(temporary)
        write_stripped_dense(shard1, reader, temporary / BUNDLE_DENSE)
        if existing_pack is not None and all(row["blob"] % 4096 == 0 for row in layout):
            reuse_file(existing_pack / "experts.bin", temporary / "experts.bin")
        else:
            write_aligned_pack(shard1, temporary / "experts.bin", layout, experts, existing_pack)
        if existing_pack is None:
            shutil.copyfile(shard2, temporary / BUNDLE_PLE)
        else:
            reuse_file(shard2, temporary / BUNDLE_PLE)
        published = []
        try:
            for name in names:
                source = temporary / name
                destination = output / name
                if source.is_symlink():
                    destination.symlink_to(source.readlink())
                else:
                    os.link(source, destination)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink()
            raise
    total = sum((row["blob"] + 4095) // 4096 * 4096 * experts for row in layout)
    return {"dense_bytes": (output / BUNDLE_DENSE).stat().st_size, "expert_bytes": total,
            "ple_bytes": shard2.stat().st_size, "layers": len(layout), "experts_per_layer": experts,
            "reuse": existing_pack is not None}


def main():
    parser = argparse.ArgumentParser(description="Pack separate routed experts without changing quantized bytes.")
    parser.add_argument("--model", type=Path, required=True, help="first GGUF shard, containing all routed experts")
    parser.add_argument("--output", type=Path, required=True, help="pack directory; existing packs are never overwritten")
    parser.add_argument("--shard2", type=Path, help="PLE shard; produce a stripped three-file bundle instead of a pack")
    parser.add_argument("--existing-pack", type=Path, help="sample-verify an existing pack; pad legacy blobs if needed, otherwise hardlink; reuse unchanged PLE (cross-filesystem links are symlinks)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="inspect layout and disk requirements without writing")
    mode.add_argument("--verify", action="store_true", help="compare three experts per layer against the GGUF")
    mode.add_argument("--verify-all", action="store_true", help="compare every expert against the GGUF")
    args = parser.parse_args()
    if args.existing_pack is not None and args.shard2 is None:
        parser.error("--existing-pack requires --shard2")
    if args.shard2 is not None and (args.dry_run or args.verify or args.verify_all):
        parser.error("--shard2 cannot be combined with pack inspection modes")
    try:
        model = args.model.resolve(strict=True)
        if args.shard2 is not None:
            result = create_bundle(model, args.shard2, args.output, args.existing_pack)
        elif args.dry_run:
            layout, experts, total = expert_layout(model)
            result = {"layers": len(layout), "experts_per_layer": experts, "bytes": (total + 4095) // 4096 * 4096}
        elif args.verify or args.verify_all:
            result = verify_pack(model, args.output, args.verify_all)
        else:
            result = create_pack(model, args.output)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
