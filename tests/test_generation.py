"""Offline tests of the native StarVector SVG token generation loop."""
import unittest

import torch

from package_alias import fake_comfy_attention
from starvector_node import runtime_shared
from starvector_node.runtime_shared import _make_causal_mask, comfy_attention, generate_tokens


class TinyTokenizer:
    eos_token_id = 99

    def encode(self, text):
        return {"<svg": [1, 2], "</svg>": [3, 4]}[text]

    def decode(self, ids):
        return "<svg" + "".join(f" {token}" for token in ids[2:])


class ScriptDecoder:
    """Makes `script[step]` (then its last token) the dominant next token of every row."""

    def __init__(self, script, context=40, vocab=128):
        self.script, self.context, self.vocab = script, context, vocab
        self.calls = []

    def embed(self, ids):
        return torch.zeros((*ids.shape, 4))

    def prefill(self, embeddings, rows, capacity):
        self.calls.append(("prefill", embeddings.shape[1], rows, capacity))
        self.step_index = 0
        return self, self._logits(rows)

    def _logits(self, rows):
        logits = torch.zeros((rows, self.vocab))
        # Dominant even after the 3.1 repetition penalty and temperature.
        logits[:, self.script[min(self.step_index, len(self.script) - 1)]] = 200.
        return logits

    def step(self, tokens):
        self.calls.append(("step", tokens.tolist()))
        self.step_index += 1
        return self._logits(tokens.shape[0])

    def reorder(self, parents):
        self.calls.append(("reorder", list(parents)))


class GenerationTests(unittest.TestCase):
    def test_generates_until_svg_end(self):
        for beams in (1, 2, 8):
            with self.subTest(beams=beams):
                decoder = ScriptDecoder([7, 8, 3, 4, 9])
                rng_before = torch.get_rng_state()
                svg = generate_tokens(decoder, TinyTokenizer(), torch.zeros(1, 8, 4), max_new_tokens=16, num_beams=beams)
                self.assertEqual(svg, "<svg 7 8 3 4")
                self.assertEqual(decoder.calls[0], ("prefill", 10, beams, 26))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_generation_emits_incremental_progress(self):
        events = []
        generate_tokens(ScriptDecoder([7, 3, 4]), TinyTokenizer(), torch.zeros(1, 8, 4),
                        max_new_tokens=8, num_beams=1, progress_callback=events.append)
        self.assertEqual(events, [1, 2, 3])

    def test_context_cap(self):
        decoder = ScriptDecoder([7])
        with self.assertLogs(runtime_shared.__name__, level="INFO") as captured:
            svg = generate_tokens(decoder, TinyTokenizer(), torch.zeros(1, 34, 4), max_new_tokens=4096, num_beams=1)
        self.assertIn("requested max_new_tokens=4096, effective max_new_tokens=4", "\n".join(captured.output))
        self.assertEqual(svg, "<svg 7 7 7 7")
        self.assertEqual(decoder.calls[0], ("prefill", 36, 1, 40))

    def test_loop_guard_stops_repeated_output(self):
        script = list(range(5, 69)) * 20
        for beams in (1, 2):
            with self.subTest(beams=beams):
                with self.assertLogs(runtime_shared.__name__, level="WARNING") as captured:
                    svg = generate_tokens(ScriptDecoder(script, context=2000), TinyTokenizer(), torch.zeros(1, 8, 4),
                                          max_new_tokens=1000, num_beams=beams)
                generated = len(svg.split()) - 1
                self.assertGreaterEqual(generated, 64 * 3)
                self.assertLess(generated, 64 * 3 + 16)
                self.assertIn("loop guard", "\n".join(captured.output))
                svg = generate_tokens(ScriptDecoder(script, context=2000), TinyTokenizer(), torch.zeros(1, 8, 4),
                                      max_new_tokens=1000, num_beams=beams, loop_guard=False)
                self.assertEqual(len(svg.split()) - 1, 1000)

    def test_invalid_beam_count(self):
        for beams in (0, 9, 1.5, "2"):
            with self.subTest(beams=beams):
                with self.assertRaisesRegex(ValueError, "num_beams"):
                    generate_tokens(ScriptDecoder([7]), TinyTokenizer(), torch.zeros(1, 8, 4),
                                    max_new_tokens=8, num_beams=beams)

    def test_context_overflow(self):
        with self.assertRaisesRegex(ValueError, "context size"):
            generate_tokens(ScriptDecoder([7]), TinyTokenizer(), torch.zeros(1, 39, 4), 4)

    def test_invalid_sampling_settings(self):
        for temperature, top_p in [(0, 0.9), (-0.5, 0.9), (1, 0), (1, 1.1)]:
            with self.subTest(temperature=temperature, top_p=top_p):
                with self.assertRaises(ValueError):
                    generate_tokens(ScriptDecoder([7]), TinyTokenizer(), torch.zeros(1, 8, 4),
                                    max_new_tokens=8, temperature=temperature, top_p=top_p)

    def test_processors_match_transformers(self):
        from transformers import RepetitionPenaltyLogitsProcessor, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
        torch.manual_seed(0)
        scores = torch.randn(3, 200) * 4
        history = torch.tensor([[5, 9, 9], [0, 1, 2], [7, 150, 199]])
        seen = torch.zeros_like(scores, dtype=torch.bool).scatter(1, history, True)
        for keep in (1, 2):
            expected = RepetitionPenaltyLogitsProcessor(runtime_shared.REPETITION_PENALTY)(history, scores.clone())
            for processor in (TemperatureLogitsWarper(1.5), TopKLogitsWarper(runtime_shared.TOP_K, min_tokens_to_keep=keep),
                              TopPLogitsWarper(0.9, min_tokens_to_keep=keep)):
                expected = processor(history, expected)
            actual = runtime_shared._warp(runtime_shared._penalize(scores, seen), 1.5, 0.9, keep)
            self.assertTrue(torch.equal(torch.isinf(actual), torch.isinf(expected)))
            self.assertTrue(torch.allclose(actual[torch.isfinite(actual)], expected[torch.isfinite(expected)]))

    def test_attention_causality(self):
        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
        seen = []
        with fake_comfy_attention(seen):
            output = comfy_attention(q, k, v, causal=True)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0], q.shape[1])
        self.assertIs(seen[0][2], False)
        self.assertTrue(torch.allclose(output.transpose(1, 2), expected, atol=1e-6))
    def test_causal_mask_with_cache(self):
        mask = _make_causal_mask(torch.ones(1, 1, 3, 4), torch.ones(1, 1, 7, 4))
        self.assertEqual(tuple(mask.shape), (1, 1, 3, 7))
        self.assertEqual(mask[0, 0, 0, 4].item(), 0)
        self.assertLess(mask[0, 0, 0, 5].item(), -1000)
        self.assertEqual(mask[0, 0, 2, 6].item(), 0)


if __name__ == "__main__":
    unittest.main()
