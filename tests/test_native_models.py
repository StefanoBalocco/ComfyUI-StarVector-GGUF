"""Check the native StarVector architectures against Transformers with the same weights.

CPU-only: no Hub downloads, checkpoint files or ComfyUI environment required.
"""
import unittest

import torch

from package_alias import fake_comfy_attention, fake_tensor


class _MinimalSVGTokenizer:
    """Enough tokenizer API to exercise the native generation loop."""
    eos_token_id = 63

    def encode(self, text):
        return [1] if text == "<svg" else [2, 3]

    def decode(self, ids):
        return "<svg" + " ".join(str(token) for token in ids[1:])


def _store(state_dict, prefix):
    from starvector_node.runtime_shared import TensorStore
    return TensorStore({
        prefix + name: fake_tensor(prefix + name, value.numpy())
        for name, value in state_dict.items() if name != "lm_head.weight"
    })


class NativeModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # CI has no full ComfyUI installation; mock its attention entry point.
        cls.enterClassContext(fake_comfy_attention())

    def _check_decoder(self, reference, native, steps):
        """Native prefill, cached decode and beam reordering against Transformers."""
        from starvector_node.runtime_shared import generate_tokens
        embeddings = torch.randn((1, 5, 32))
        with torch.inference_mode():
            expected = reference(inputs_embeds=embeddings.expand(2, -1, -1), use_cache=True)
            decoding, logits = native.prefill(embeddings, 2, 5 + len(steps))
            self.assertTrue(torch.allclose(logits, expected.logits[:, -1], atol=1e-5))
            cache = expected.past_key_values
            for tokens, parents in steps:
                cache.reorder_cache(torch.tensor(parents))
                decoding.reorder(parents)
                expected = reference(input_ids=torch.tensor(tokens)[:, None], past_key_values=cache, use_cache=True)
                cache = expected.past_key_values
                logits = decoding.step(torch.tensor(tokens))
                self.assertTrue(torch.allclose(logits, expected.logits[:, -1], atol=1e-5))
        svg = generate_tokens(native, _MinimalSVGTokenizer(), torch.zeros((1, 4, 32)), max_new_tokens=3, seed=123)
        self.assertTrue(svg.startswith("<svg"))

    def test_native_gpt_bigcode_matches_transformers(self):
        from transformers import GPTBigCodeConfig, GPTBigCodeForCausalLM
        from starvector_node.runtime_1b import GPTBigCode
        from starvector_node.runtime_shared import DECODER_PREFIX

        torch.manual_seed(0)
        reference = GPTBigCodeForCausalLM(GPTBigCodeConfig(
            vocab_size=64, n_positions=64, n_embd=32, n_layer=2,
            n_head=4, n_inner=64, multi_query=True, _attn_implementation="sdpa",
        )).eval()
        native = GPTBigCode(_store(reference.state_dict(), DECODER_PREFIX), {
            "hidden_size": 32, "num_attention_heads": 4, "num_hidden_layers": 2,
            "max_position_embeddings": 64, "n_inner": 64}, 64).eval()
        self._check_decoder(reference, native, (([4, 9], [0, 1]), ([7, 7], [1, 1]), ([3, 5], [1, 0])))

    def test_native_starcoder2_matches_transformers(self):
        from transformers import Starcoder2Config, Starcoder2ForCausalLM
        from starvector_node.runtime_8b import Starcoder2
        from starvector_node.runtime_shared import DECODER_PREFIX

        torch.manual_seed(0)
        # Decoding past the sliding window of 6 tokens checks the window mask.
        reference = Starcoder2ForCausalLM(Starcoder2Config(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
            rope_parameters={"rope_type": "default", "rope_theta": 1000.0}, sliding_window=6,
            _attn_implementation="sdpa",
        )).eval()
        native = Starcoder2(_store(reference.state_dict(), DECODER_PREFIX), {
            "hidden_size": 32, "num_attention_heads": 4, "num_kv_heads": 2, "num_hidden_layers": 2,
            "intermediate_size": 64, "max_position_embeddings": 64, "rope_theta": 1000.0, "sliding_window": 6}, 64).eval()
        self._check_decoder(reference, native, (([4, 9], [0, 1]), ([7, 7], [1, 1]), ([3, 5], [1, 0]), ([8, 2], [0, 0]), ([6, 6], [0, 1])))

    def test_native_siglip_matches_transformers(self):
        from transformers import SiglipVisionConfig, SiglipVisionModel
        from starvector_node.runtime_8b import SiglipVision
        from starvector_node.runtime_shared import VISION_PREFIX

        torch.manual_seed(0)
        reference = SiglipVisionModel(SiglipVisionConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            image_size=32, patch_size=16, layer_norm_eps=1e-6, hidden_act="gelu_pytorch_tanh",
            vision_use_head=False, _attn_implementation="sdpa",
        )).eval()
        native = SiglipVision(_store(reference.state_dict(), VISION_PREFIX + "visual_encoder."), 32,
                              width=32, heads=4, layers=2, inner=64, patch=16).eval()
        pixels = torch.rand(1, 3, 32, 32)
        with torch.inference_mode():
            expected = reference(pixel_values=pixels).last_hidden_state
            self.assertTrue(torch.allclose(native(pixels), expected, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
