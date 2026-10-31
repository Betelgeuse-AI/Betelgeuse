import importlib.util
import errno
import json
from pathlib import Path
import tempfile
import unittest

from unittest import mock
import numpy as np

spec = importlib.util.spec_from_file_location("moe_pack", Path(__file__).resolve().parents[1] / "tools/moe_pack.py")
moe_pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(moe_pack)
from gguf import GGUFReader, GGUFWriter


class BundleHeader(GGUFReader):
    def _build_tensors(self, start_offs, fields):
        self.descriptors = {bytes(field.parts[1]).decode(): field for field in fields}


class TestMoePack(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.gguf"
        self.output = self.root / "pack"

    def write_model(self, omit=None, wrong_shape=False, unaligned=False, split=False):
        writer = GGUFWriter(self.model, "qwen4exp")
        if split:
            writer.add_uint16("split.no", 0)
            writer.add_uint16("split.count", 2)
            writer.add_int32("split.tensors.count", 9)
            writer.add_array("tokenizer.ggml.tokens", ["one", "two", "three"])
            writer.add_tensor("token_embd.weight", np.arange(32, dtype=np.float32).reshape(4, 8))
            writer.add_tensor("blk.0.ffn_gate_exps.bias", np.arange(4, dtype=np.float32))
        for layer in range(2):
            dtype = np.float32 if layer == 0 else np.float16
            for role in ("gate", "up", "down"):
                if (layer, role) == omit:
                    continue
                shape = (4, 64, 32) if role == "down" else (4, 32, 64)
                if unaligned:
                    shape = (4, 8, 4) if role == "down" else (4, 4, 8)
                if wrong_shape and role == "down":
                    shape = (4, 32, 64)
                values = np.arange(np.prod(shape), dtype=dtype).reshape(shape) + layer
                writer.add_tensor(f"blk.{layer}.ffn_{role}_exps.weight", values)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_round_trip(self):
        self.write_model()
        metadata = moe_pack.create_pack(self.model, self.output)
        self.assertEqual(metadata["layers"], 2)
        self.assertEqual(metadata["experts_per_layer"], 4)
        self.assertEqual(moe_pack.verify_pack(self.model, self.output)["verified_experts"], 6)
        self.assertEqual(moe_pack.verify_pack(self.model, self.output, True)["verified_experts"], 8)
        self.assertEqual(json.loads((self.output / "expert-pack.json").read_text()), metadata)

    def test_no_overwrite(self):
        self.write_model()
        moe_pack.create_pack(self.model, self.output)
        original = (self.output / "experts.bin").read_bytes()
        with self.assertRaises(FileExistsError):
            moe_pack.create_pack(self.model, self.output)
        self.assertEqual((self.output / "experts.bin").read_bytes(), original)

    def test_corrupt_pack(self):
        self.write_model()
        moe_pack.create_pack(self.model, self.output)
        with (self.output / "experts.bin").open("r+b") as target:
            target.write(b"\xff")
        with self.assertRaisesRegex(ValueError, "weight mismatch"):
            moe_pack.verify_pack(self.model, self.output)

    def test_truncated_pack(self):
        self.write_model()
        moe_pack.create_pack(self.model, self.output)
        (self.output / "experts.bin").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "byte size"):
            moe_pack.verify_pack(self.model, self.output)

    def test_incomplete_layer(self):
        self.write_model(omit=(1, "up"))
        with self.assertRaisesRegex(ValueError, "requires separate"):
            moe_pack.create_pack(self.model, self.output)
        self.assertFalse(self.output.exists())

    def test_shape_mismatch(self):
        self.write_model(wrong_shape=True)
        with self.assertRaisesRegex(ValueError, "projection shape"):
            moe_pack.expert_layout(self.model)

    def test_bundle_pads_unaligned_legacy_experts(self):
        self.write_model(unaligned=True, split=True)
        ple = self.write_ple()
        moe_pack.create_pack(self.model, self.output)
        bundle = self.root / "bundle"
        result = moe_pack.create_bundle(self.model, ple, bundle, self.output)
        layout, experts, _ = moe_pack.expert_layout(self.model)
        self.assertEqual(result["expert_bytes"], sum((row["blob"] + 4095) // 4096 * 4096 * experts for row in layout))
        self.assertFalse((bundle / "experts.bin").samefile(self.output / "experts.bin"))
        with self.model.open("rb") as source, (bundle / "experts.bin").open("rb") as packed:
            for row in layout:
                for expert in range(experts):
                    self.assertEqual(packed.tell() % 4096, 0)
                    self.assertEqual(packed.read(row["blob"]), moe_pack.read_expert(source, row, expert))
                    padding = (-row["blob"]) % 4096
                    self.assertEqual(packed.read(padding), bytes(padding))
        self.assertEqual(moe_pack.verify_pack(self.model, self.output, True)["verified_experts"], 8)


    def write_ple(self):
        path = self.root / "ple.gguf"
        writer = GGUFWriter(path, "qwen4exp")
        writer.add_uint16("split.no", 1)
        writer.add_uint16("split.count", 2)
        writer.add_int32("split.tensors.count", 9)
        writer.add_tensor("per_layer_token_embd.weight", np.arange(32, dtype=np.float32).reshape(4, 8))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        return path

    def test_stripped_bundle_reuses_sources_and_preserves_dense(self):
        self.write_model(split=True)
        ple = self.write_ple()
        moe_pack.create_pack(self.model, self.output)
        bundle = self.root / "bundle"
        moe_pack.create_bundle(self.model, ple, bundle, self.output)
        self.assertEqual({path.name for path in bundle.iterdir()},
                         {moe_pack.BUNDLE_DENSE, "experts.bin", moe_pack.BUNDLE_PLE})
        self.assertTrue((bundle / "experts.bin").samefile(self.output / "experts.bin"))
        self.assertTrue((bundle / moe_pack.BUNDLE_PLE).samefile(ple))
        source = GGUFReader(self.model)
        dense_path = bundle / moe_pack.BUNDLE_DENSE
        header = BundleHeader(dense_path)
        self.assertTrue(header.fields["betelgeuse.stripped_experts"].contents())
        self.assertEqual(header.fields["betelgeuse.ple_file"].contents(), moe_pack.BUNDLE_PLE)
        for key, field in source.fields.items():
            if key != "GGUF.kv_count":
                self.assertEqual(header.fields[key].contents(), field.contents())
        tensors = {tensor.name: tensor for tensor in source.tensors}
        expected_offset = 0
        for name, descriptor in header.descriptors.items():
            self.assertEqual(int(descriptor.parts[-1][0]), expected_offset)
            expected_offset += (tensors[name].n_bytes + header.alignment - 1) // header.alignment * header.alignment
        with dense_path.open("rb") as stripped:
            for tensor in source.tensors:
                descriptor = header.descriptors[tensor.name]
                self.assertEqual([part.tobytes() for part in descriptor.parts[:-1]],
                                 [part.tobytes() for part in tensor.field.parts[:-1]])
                offset = header.data_offset + int(descriptor.parts[-1][0])
                if moe_pack.EXPERT_PATTERN.fullmatch(tensor.name):
                    self.assertGreaterEqual(offset, dense_path.stat().st_size)
                else:
                    self.assertLessEqual(offset + tensor.n_bytes, dense_path.stat().st_size)
                    stripped.seek(offset)
                    self.assertEqual(stripped.read(tensor.n_bytes), tensor.data.tobytes())
        with self.assertRaises(FileExistsError):
            moe_pack.create_bundle(self.model, ple, bundle, self.output)
        self.assertEqual(len(GGUFReader(self.model).tensors), 8)

    def test_bundle_rejects_incompatible_companion_and_pack(self):
        self.write_model(split=True)
        ple = self.write_ple()
        moe_pack.create_pack(self.model, self.output)
        bundle = self.root / "bundle"
        with self.assertRaisesRegex(ValueError, "split.tensors.count"):
            moe_pack.create_bundle(self.model, self.model, bundle, self.output)
        with (self.output / "experts.bin").open("r+b") as packed:
            packed.truncate(4096)
        with self.assertRaisesRegex(ValueError, "byte size"):
            moe_pack.create_bundle(self.model, ple, bundle, self.output)
        self.assertFalse(bundle.exists())

    def test_cross_filesystem_reuse_and_link_errors(self):
        source = self.root / "source"
        source.write_bytes(b"unchanged")
        target = self.root / "target"
        with mock.patch.object(moe_pack.os, "link", side_effect=OSError(errno.EXDEV, "cross-device")):
            moe_pack.reuse_file(source, target)
        self.assertTrue(target.is_symlink())
        self.assertTrue(target.samefile(source))
        with mock.patch.object(moe_pack.os, "link", side_effect=PermissionError(errno.EACCES, "denied")):
            with self.assertRaises(PermissionError):
                moe_pack.reuse_file(source, self.root / "denied")
        self.assertFalse((self.root / "denied").exists())


if __name__ == "__main__":
    unittest.main()
