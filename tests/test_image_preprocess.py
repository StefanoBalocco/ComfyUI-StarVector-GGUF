"""StarVector image preprocessing: CLIP square padding and SigLIP resize."""
import unittest

import torch

import package_alias  # noqa: F401
from starvector_node.runtime_1b import CLIP_MEAN, CLIP_STD, preprocess_image
from starvector_node.runtime_8b import preprocess_siglip
from starvector_node.runtime_shared import image_to_pil


class PreprocessTests(unittest.TestCase):
    def test_1b_pads_to_white_square_and_rounds_bytes(self):
        image = torch.full((1, 10, 20, 3), 1.9 / 255.0)
        result = preprocess_image(image)
        self.assertEqual(tuple(result.shape), (1, 3, 224, 224))
        self.assertAlmostEqual(float(result[0, 0, 112, 112]), (2.0 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0], places=5)
        self.assertAlmostEqual(float(result[0, 0, 0, 112]), (1.0 - CLIP_MEAN[0]) / CLIP_STD[0], places=5)

    def test_8b_resizes_without_padding_and_rounds_bytes(self):
        result = preprocess_siglip(torch.full((1, 12, 24, 3), 1.9 / 255.0))
        self.assertEqual(tuple(result.shape), (1, 3, 384, 384))
        self.assertTrue(torch.allclose(result, torch.full_like(result, (2.0 / 255.0 - 0.5) / 0.5), atol=1e-5))

    def test_alpha_is_composited_on_white_and_batches_rejected(self):
        image = torch.zeros((1, 2, 2, 4))
        self.assertEqual(image_to_pil(image).getpixel((0, 0)), (255, 255, 255))
        with self.assertRaisesRegex(ValueError, "one ComfyUI IMAGE"):
            image_to_pil(torch.zeros((2, 2, 2, 3)))


if __name__ == "__main__":
    unittest.main()
