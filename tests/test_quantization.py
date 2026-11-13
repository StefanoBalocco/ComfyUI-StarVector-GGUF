"""Small reference arithmetic for packed GGUF Q4_0/Q5_0/Q8_0 weights."""
import struct
import unittest

import numpy as np
import torch

from package_alias import GGUFType as QType, fake_tensor
from starvector_node import quantization as quant
from starvector_node.quantization import (
    dequantize, dequantize_soa, graph_io, pack_gguf_soa, tensor_payload,
    GGUFConv2d, GGUFLinear, GGUFEmbedding,
)


def fake(name, qtype, shape, raw):
    return fake_tensor(name, np.frombuffer(raw, dtype=np.uint8).copy(), qtype, shape)


class QuantizationTests(unittest.TestCase):
    def test_q8_0_decode_blocks(self):
        raw = struct.pack('<e', 0.5) + bytes(range(32))
        result = dequantize(torch.tensor(list(raw), dtype=torch.uint8), 'Q8_0', (1, 32), torch.float32)
        self.assertTrue(torch.allclose(result[0], torch.arange(32).float() * 0.5))

    def test_q4_0_unpack_nibble_halves(self):
        raw = struct.pack('<e', 1.0) + bytes([0xF0] * 16)
        result = dequantize(torch.tensor(list(raw), dtype=torch.uint8), 'Q4_0', (1, 32), torch.float32)
        self.assertTrue(torch.equal(result[0, :16], torch.full((16,), -8.)))
        self.assertTrue(torch.equal(result[0, 16:], torch.full((16,), 7.)))

    def test_q5_0_fifth_bits(self):
        # Low nibbles 0..15 then 0; fifth bits set for the second half only.
        raw = struct.pack('<e', 1.0) + struct.pack('<I', 0xFFFF0000) + bytes(range(16))
        result = dequantize(torch.tensor(list(raw), dtype=torch.uint8), 'Q5_0', (1, 32), torch.float32)
        self.assertTrue(torch.equal(result[0, :16], torch.arange(16).float() - 16))
        self.assertTrue(torch.equal(result[0, 16:], torch.zeros(16)))

    def test_q8_linear(self):
        raw = struct.pack('<e', 0.25) + bytes([4] * 32)
        weight = fake('test.weight', QType.Q8_0, (1, 32), raw)
        layer = GGUFLinear(32, 1)
        layer.set_gguf('weight', weight)
        x = torch.ones((1, 32))
        self.assertTrue(torch.allclose(layer(x), torch.tensor([[32.]])))
        self.assertEqual(layer.weight.dtype, torch.uint8)
        self.assertEqual(layer.weight_scales.dtype, torch.uint8)
        self.assertEqual(
            layer.weight.numel() + layer.weight_scales.numel(), len(raw))

    def test_quant_embedding_selects_only_requested_rows(self):
        raw = (struct.pack('<e', 1.0) + bytes([0x99] * 16)) + (struct.pack('<e', 2.0) + bytes([0x99] * 16))
        weights = fake('tokens.weight', QType.Q4_0, (2, 32), raw)
        emb = GGUFEmbedding(2, 32)
        emb.set_gguf('weight', weights)
        x = emb(torch.tensor([[1, 0, 1]]))
        self.assertEqual(tuple(x.shape), (1, 3, 32))
        self.assertTrue(torch.equal(x[0, 0], torch.full((32,), 2.)))
        self.assertTrue(torch.equal(x[0, 1], torch.ones(32)))


    def _packed_matrix(self, qtype, n, k):
        rng = np.random.default_rng(321)
        blocks = n * k // 32
        scales = rng.uniform(0.02, 0.22, size=blocks).astype(np.float16)
        if qtype == 'Q4_0':
            signed = rng.integers(-8, 8, size=(blocks, 32), dtype=np.int8)
            unsigned = (signed.astype(np.int16) + 8).astype(np.uint8)
            payload = unsigned[:, :16] | (unsigned[:, 16:] << 4)
            kind = QType.Q4_0
        elif qtype == 'Q5_0':
            unsigned = rng.integers(0, 32, size=(blocks, 32)).astype(np.uint32)
            fifth = ((unsigned >> 4) << np.arange(32, dtype=np.uint32)).sum(axis=1).astype('<u4')
            nibbles = (unsigned[:, :16] & 15) | ((unsigned[:, 16:] & 15) << 4)
            payload = np.concatenate((fifth.view(np.uint8).reshape(blocks, 4), nibbles.astype(np.uint8)), axis=1)
            kind = QType.Q5_0
        else:
            signed = rng.integers(-127, 128, size=(blocks, 32), dtype=np.int8)
            payload = signed.view(np.uint8)
            kind = QType.Q8_0
        raw = b''.join(
            struct.pack('<e', float(scales[i])) + payload[i].tobytes()
            for i in range(blocks)
        )
        return fake('test.weight', kind, (n, k), raw)

    def test_quantized_linear_matches_dequantized_reference(self):
        for qtype in ('Q4_0', 'Q5_0', 'Q8_0'):
            for shape in ((1, 9, 64), (2, 33, 192), (8, 17, 256), (16, 13, 64)):
                with self.subTest(qtype=qtype, shape=shape):
                    m, n, k = shape
                    layer = GGUFLinear(k, n)
                    original = self._packed_matrix(qtype, n, k)
                    layer.set_gguf('weight', original)
                    bias = np.linspace(-0.5, 0.5, n, dtype=np.float32)
                    layer.set_gguf('bias', fake_tensor('test.bias', bias.copy()))
                    x = torch.randn(m, k, dtype=torch.float32)
                    unpacked = dequantize(
                        tensor_payload(original), qtype, (n, k),
                        torch.float32)
                    self.assertTrue(torch.equal(
                        layer.unpack('weight', torch.float32, 'cpu'), unpacked))
                    self.assertEqual(
                        layer.weight.numel() + layer.weight_scales.numel(),
                        tensor_payload(original).numel())
                    expected = torch.nn.functional.linear(
                        x, unpacked, layer.bias)
                    self.assertTrue(torch.allclose(layer(x), expected, rtol=2e-5, atol=2e-4))

                    if torch.cuda.is_available():
                        gpu_layer = layer.to('cuda')
                        self.assertEqual(gpu_layer.weight_scales.dtype,
                                         torch.uint8)
                        for dtype in (torch.float16, torch.float32):
                            x_gpu = x.to(device='cuda', dtype=dtype)
                            reference = torch.nn.functional.linear(
                                x_gpu, unpacked.to('cuda', dtype=dtype),
                                gpu_layer.unpack('bias', dtype, x_gpu.device))
                            out = gpu_layer(x_gpu)
                            self.assertTrue(torch.allclose(
                                out, reference, atol=0.06, rtol=0.06),
                                f'SoA forward {qtype} {shape} {dtype} mismatch')
                            if dtype == torch.float16:
                                if quant.triton is not None:
                                    fused = quant._fused_soa_linear(
                                        x_gpu, gpu_layer.weight,
                                        gpu_layer.weight_scales,
                                        gpu_layer._specs['weight'],
                                        gpu_layer.unpack('bias', dtype, x_gpu.device))
                                    self.assertIsNotNone(
                                        fused, 'SoA GEMV/GEMM unexpectedly fell back')
                                    self.assertTrue(torch.allclose(
                                        fused, reference, atol=0.06, rtol=0.06))

    def test_soa_payload_exact_and_dtype_stable(self):
        for qtype in ('Q4_0', 'Q5_0', 'Q8_0'):
            tensor = self._packed_matrix(qtype, 13, 128)
            q, scale_bits = pack_gguf_soa(tensor)
            reference = dequantize(
                tensor_payload(tensor), qtype, (13, 128), torch.float32)
            self.assertEqual(q.numel() + scale_bits.numel(),
                             tensor_payload(tensor).numel())
            self.assertTrue(torch.equal(
                dequantize_soa(q, scale_bits, qtype, (13, 128),
                               torch.float32), reference))
            layer = GGUFLinear(128, 13)
            layer.set_gguf('weight', tensor)
            layer.to(dtype=torch.bfloat16)
            self.assertEqual(layer.weight.dtype, torch.uint8)
            self.assertEqual(layer.weight_scales.dtype, torch.uint8)
            self.assertTrue(torch.equal(
                layer.unpack('weight', torch.float32, 'cpu'), reference))
            x = torch.randn(7, 128)
            self.assertTrue(torch.allclose(
                layer(x), torch.nn.functional.linear(x, reference),
                rtol=1e-5, atol=1e-4))

    def test_tied_soa_embedding_and_head_share_both_parameters(self):
        for qtype in ('Q4_0', 'Q8_0'):
            tensor = self._packed_matrix(qtype, 12, 128)
            emb = GGUFEmbedding(12, 128)
            emb.set_gguf('weight', tensor)
            head = GGUFLinear(128, 12)
            head.weight = emb.weight
            head.weight_scales = emb.weight_scales
            head._specs['weight'] = emb._specs['weight']
            self.assertIs(head.weight, emb.weight)
            self.assertIs(head.weight_scales, emb.weight_scales)
            model = torch.nn.Module()
            model.add_module('embed', emb)
            model.add_module('head', head)
            model.to(dtype=torch.float16)
            # Mirrors ComfyUI's stage dtype conversion and shared parameters.
            self.assertIs(head.weight, emb.weight)
            self.assertIs(head.weight_scales, emb.weight_scales)
            self.assertEqual(len(list(model.parameters())), 2)
            selected = torch.tensor([[0, 5, 5, 1, 10, 0, 8, 7, 3]])
            dense = dequantize(
                tensor_payload(tensor), qtype, (12, 128), torch.float32)
            self.assertTrue(torch.equal(emb(selected), dense[selected]))
            x = torch.randn(2, 128)
            self.assertTrue(torch.allclose(
                head(x), torch.nn.functional.linear(x, dense),
                rtol=1e-5, atol=1e-4))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_cuda_graph_capture_with_static_triton_io(self):
        if quant.triton is None:
            self.skipTest('Triton not installed')
        for qtype in ('Q4_0', 'Q5_0', 'Q8_0'):
            layer = GGUFLinear(256, 96)
            layer.set_gguf('weight', self._packed_matrix(qtype, 96, 256))
            layer = layer.to('cuda')
            x = torch.randn(2, 256, device='cuda', dtype=torch.float16)
            buffers = []
            with graph_io(buffers):
                layer(x)
            self.assertEqual(len(buffers), 1)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph), graph_io(buffers):
                captured = layer(x)
            # The capture writes into the warm-up's static output buffer.
            self.assertEqual(captured.data_ptr(), buffers[0][1].data_ptr())
            x.copy_(torch.randn_like(x))
            graph.replay()
            reference = torch.nn.functional.linear(x.float(), layer.unpack('weight', torch.float32, 'cuda'))
            self.assertTrue(torch.allclose(captured.float(), reference, atol=0.05, rtol=0.02), qtype)

    def test_conv_weights_stay_floating_point(self):
        conv = GGUFConv2d(stride=2)
        conv.set_gguf('weight', fake_tensor('patch.weight', np.ones((2, 3, 2, 2), np.float32)))
        conv.set_gguf('bias', fake_tensor('patch.bias', np.array([1, 2], np.float32)))
        output = conv(torch.ones(1, 3, 2, 2))
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.equal(output.flatten(), torch.tensor([13., 14.])))

    def test_short_embedding_decode_avoids_unique_sort(self):
        from unittest.mock import patch
        raw = (struct.pack('<e', 1.0) + bytes([0x99] * 16)) * 3
        weights = fake('tokens.weight', QType.Q4_0, (3, 32), raw)
        emb = GGUFEmbedding(3, 32)
        emb.set_gguf('weight', weights)
        with patch.object(torch, 'unique', side_effect=AssertionError('unnecessary unique')):
            output = emb(torch.tensor([[2, 1, 2]]))
        self.assertEqual(tuple(output.shape), (1, 3, 32))
        self.assertTrue(torch.equal(output, torch.ones_like(output)))

    def test_long_embedding_prefill_deduplicates(self):
        raw = (struct.pack('<e', 1.0) + bytes([0x99] * 16)) * 3
        emb = GGUFEmbedding(3, 32)
        emb.set_gguf('weight', fake('tokens.weight', QType.Q4_0, (3, 32), raw))
        out = emb(torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 2]]))
        self.assertEqual(tuple(out.shape), (1, 9, 32))
        self.assertTrue(torch.equal(out, torch.ones_like(out)))


if __name__ == '__main__':
    unittest.main()
