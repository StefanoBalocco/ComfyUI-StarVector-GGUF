"""CPU-only checks of the GGUF exporter's per-tensor precision choice."""
import importlib.util
from pathlib import Path
import sys
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "starvector_safetensors_to_gguf.py"
spec = importlib.util.spec_from_file_location("_starvector_converter_test", SCRIPT)
module = importlib.util.module_from_spec(spec)
# The exporter's dataclass resolves its own module while being created.
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TensorTargetTests(unittest.TestCase):
    def target(self, name, shape, quantization, dtype="F16"):
        return module.tensor_target(shape, dtype, quantization, name=name)

    def test_low_bit_keeps_vision_and_embeddings_q8(self):
        for name, shape in (
            ("model.image_encoder.visual_encoder.transformer.resblocks.0.attn.in_proj_weight", (3072, 1024)),
            ("model.image_encoder.visual_encoder.positional_embedding", (257, 1024)),
            ("model.image_projection.c_fc.weight", (2048, 1024)),
            ("model.svg_transformer.transformer.transformer.wte.weight", (49156, 2048)),
            ("model.svg_transformer.transformer.transformer.wpe.weight", (8192, 2048)),
            ("model.svg_transformer.transformer.model.embed_tokens.weight", (49152, 4608)),
        ):
            self.assertEqual(self.target(name, shape, "q4_0"), "Q8_0", name)

    def test_low_bit_quantizes_decoder_blocks(self):
        name = "model.svg_transformer.transformer.transformer.h.0.mlp.c_fc.weight"
        self.assertEqual(self.target(name, (8192, 2048), "q4_0"), "Q4_0")

    def test_q8_and_unquantized_tensors_unchanged(self):
        self.assertEqual(self.target("model.image_projection.c_fc.weight", (2048, 1024), "q8_0"), "Q8_0")
        self.assertEqual(self.target("model.image_encoder.ln_vision.weight", (1024,), "q4_0", "F32"), "F32")
        self.assertEqual(self.target("model.image_encoder.visual_encoder.conv1.weight", (1024, 3, 14, 14), "q4_0"), "F16")


if __name__ == "__main__":
    unittest.main()
