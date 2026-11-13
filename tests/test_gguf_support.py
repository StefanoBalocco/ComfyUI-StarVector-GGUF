"""Offline tests: no ComfyUI, GPU, external files or model downloads required."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

from package_alias import GGUFType as DType, fake_tensor
from starvector_node.gguf_support import inspect_tensor_index, inspect_local_gguf, read_embedded_config, read_embedded_assets


def tensor(name, dtype=DType.Q4_0):
    return fake_tensor(name, tensor_type=dtype)


class GGUFSupportTests(unittest.TestCase):
    def test_full_pack_reports_all_components(self):
        items = [
            tensor("model.image_encoder.visual_encoder.conv1.weight"),
            tensor("model.image_projection.c_fc.weight"),
            tensor("model.svg_transformer.transformer.transformer.wte.weight", DType.F16),
        ]
        self.assertEqual(inspect_tensor_index(items), 3)

    def test_rejects_diffusion_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "full StarVector GGUF"):
            inspect_tensor_index([tensor("diffusion_model.input_blocks.0.weight")])

    def test_rejects_decoder_only_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "full StarVector GGUF"):
            inspect_tensor_index([
                tensor("model.svg_transformer.transformer.transformer.wte.weight")
            ])

    def test_embedded_config_parsing(self):
        field = SimpleNamespace(contents=lambda: '{"hidden_size": 2048, "image_encoder_type": "clip"}')
        reader = SimpleNamespace(get_field=lambda key: field if key == "starvector.config_json" else None)
        self.assertEqual(read_embedded_config(reader)["hidden_size"], 2048)

        for raw in ("not-json", "[]", "null"):
            with self.subTest(raw=raw):
                broken = SimpleNamespace(contents=lambda raw=raw: raw)
                reader = SimpleNamespace(get_field=lambda key: broken)
                with self.assertRaisesRegex(ValueError, "starvector.config_json"):
                    read_embedded_config(reader)
        with self.assertRaisesRegex(ValueError, "starvector.config_json"):
            read_embedded_config(SimpleNamespace(get_field=lambda key: None))

    def test_reads_embedded_tokenizer_metadata(self):
        payload = '{"version":"1.0","model":{"type":"BPE"}}'
        fields = {
            "starvector.embedded_file.tokenizer.json": SimpleNamespace(contents=lambda: payload),
            "starvector.embedded_file.tokenizer_config.json": SimpleNamespace(contents=lambda: '{}'),
        }
        reader = SimpleNamespace(get_field=lambda key: fields.get(key))
        self.assertEqual(read_embedded_assets(reader)["tokenizer.json"], payload)

    def test_missing_or_incomplete_tokenizer_rejected_even_with_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "tokenizer.json").write_text("{}")
            for fields in (
                {},
                {"starvector.embedded_file.vocab.json": SimpleNamespace(contents=lambda: '{"a":0}')},
            ):
                reader = SimpleNamespace(get_field=lambda key: fields.get(key))
                with self.subTest(fields=fields):
                    with self.assertRaisesRegex(ValueError, "Invalid StarVector GGUF.*embedded tokenizer"):
                        read_embedded_assets(reader)

    def test_rejects_invalid_embedded_tokenizer_contents(self):
        for contents in ("", bytes([255]), 10):
            reader = SimpleNamespace(get_field=lambda key: (
                SimpleNamespace(contents=lambda: contents)
                if key == "starvector.embedded_file.tokenizer.json" else None
            ))
            with self.subTest(contents=contents):
                with self.assertRaisesRegex(ValueError, "Invalid StarVector GGUF"):
                    read_embedded_assets(reader)

    def test_gguf_is_opened_once_and_index_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            path.touch()
            tensors = [
                tensor("model.image_encoder.visual_encoder.conv1.weight"),
                tensor("model.image_projection.c_fc.weight"),
                tensor("model.svg_transformer.transformer.transformer.wte.weight"),
            ]
            config_field = SimpleNamespace(contents=lambda: '{"hidden_size": 2048}')
            reader = SimpleNamespace(
                tensors=tensors,
                get_field=lambda key: (
                    config_field if key == "starvector.config_json" else
                    SimpleNamespace(contents=lambda: '{"model":{"type":"BPE"}}')
                    if key == "starvector.embedded_file.tokenizer.json" else None
                ),
            )
            calls = []
            gguf = ModuleType("gguf")
            def fake_reader(filename, mode):
                calls.append((filename, mode))
                return reader
            gguf.GGUFReader = fake_reader
            with patch.dict(sys.modules, {"gguf": gguf}):
                ref = inspect_local_gguf(path)

            # Copy-on-write, so the SoA repack can wrap the mapping without writing the file.
            self.assertEqual(calls, [(str(path), "c")])
            self.assertIs(ref.tensor_index[tensors[0].name], tensors[0])
            self.assertEqual(len(ref.tensor_index), 3)
            self.assertEqual(ref.model_config, {"hidden_size": 2048})
            self.assertIn("tokenizer.json", ref.model_assets)

    def test_rejects_duplicate_names_in_shared_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            path.touch()
            gguf = ModuleType("gguf")
            config_field = SimpleNamespace(contents=lambda: '{"hidden_size": 2048}')
            gguf.GGUFReader = lambda filename, mode: SimpleNamespace(
                get_field=lambda key: (
                    config_field if key == "starvector.config_json" else
                    SimpleNamespace(contents=lambda: '{"model":{"type":"BPE"}}')
                    if key == "starvector.embedded_file.tokenizer.json" else None
                ),
                tensors=[
                tensor("model.image_encoder.same.weight"),
                tensor("model.image_encoder.same.weight"),
                tensor("model.image_projection.c_fc.weight"),
                tensor("model.svg_transformer.transformer.transformer.wte.weight"),
            ])
            with patch.dict(sys.modules, {"gguf": gguf}):
                with self.assertRaisesRegex(ValueError, "Duplicate tensor"):
                    inspect_local_gguf(path)


if __name__ == "__main__":
    unittest.main()
