"""Shared runtime pieces: GGUF tensor store, tied output head and embedded tokenizer."""
import json
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn

from package_alias import fake_tensor
from starvector_node.quantization import GGUFEmbedding, GGUFLinear, tied_head
from starvector_node.runtime_shared import TensorStore, load_local_tokenizer


def tokenizer_assets(eos="<|endoftext|>"):
    tokenizer = Tokenizer(models.WordLevel({"<|endoftext|>": 0, "<svg": 1, "</svg>": 2, "a": 3, "[UNK]": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.add_special_tokens(["<|endoftext|>"])
    return {"tokenizer.json": tokenizer.to_str(), "tokenizer_config.json": json.dumps({"eos_token": eos})}


class TensorStoreTests(unittest.TestCase):
    def test_reports_missing_tensors(self):
        with self.assertRaisesRegex(ValueError, "Missing StarVector tensor: nope"):
            TensorStore({}).get("nope")

    def test_tied_head_shares_embedding_parameters(self):
        item = fake_tensor("root.wte.weight", np.arange(9 * 32, dtype=np.float32).reshape(9, 32))
        model = nn.Module()
        model.wte = TensorStore({item.name: item}).embedding("root.wte", 9, 32)
        model.lm_head = tied_head(model.wte)
        self.assertIsInstance(model.wte, GGUFEmbedding)
        self.assertIsInstance(model.lm_head, GGUFLinear)
        self.assertIs(model.lm_head.weight, model.wte.weight)
        self.assertEqual(len(list(model.parameters())), 1)
        x = torch.randn((2, 32))
        self.assertTrue(torch.allclose(model.lm_head(x), torch.nn.functional.linear(x, torch.from_numpy(item.data)), rtol=1e-5, atol=1e-4))


class EmbeddedTokenizerTests(unittest.TestCase):
    def test_encodes_decodes_and_skips_special_tokens(self):
        for eos in ("<|endoftext|>", {"content": "<|endoftext|>"}):
            with self.subTest(eos=eos):
                tokenizer = load_local_tokenizer(tokenizer_assets(eos), 5)
                self.assertEqual(tokenizer.eos_token_id, 0)
                self.assertEqual(tokenizer.encode("<svg a"), [1, 3])
                self.assertEqual(tokenizer.decode([1, 3, 0]), "<svg a")

    def test_rejects_vocabulary_mismatch(self):
        with self.assertRaisesRegex(ValueError, "decoder requires 6"):
            load_local_tokenizer(tokenizer_assets(), 6)

    def test_rejects_missing_invalid_or_eos_less_tokenizer(self):
        for invalid in ({}, None, {"vocab.json": "{}"}, {"tokenizer.json": "{"}, tokenizer_assets("<missing>")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "Invalid StarVector GGUF"):
                    load_local_tokenizer(invalid, 5)


if __name__ == "__main__":
    unittest.main()
