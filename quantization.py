"""StarVector GGUF ops: compact SoA parameters and low-batch Triton GEMV."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_TYPES = {"Q4_0", "Q5_0", "Q8_0", "F16", "F32", "BF16"}
# Bytes of one GGUF block of 32 weights: an FP16 scale followed by the payload.
BLOCK_BYTES = {"Q4_0": 18, "Q5_0": 22, "Q8_0": 34}
_LOG = logging.getLogger(__name__)

def type_name(tensor):
    return tensor.tensor_type.name

def tensor_shape(tensor):
    return tuple(int(x) for x in reversed(tensor.shape))

def tensor_payload(tensor):
    qtype = type_name(tensor)
    if qtype not in SUPPORTED_TYPES:
        raise ValueError(f"Unsupported GGUF type {qtype}: {tensor.name}")
    data = np.array(tensor.data, copy=True)
    if qtype in BLOCK_BYTES or qtype == "BF16":
        data = data.view(np.uint8).reshape(-1)
    return torch.from_numpy(data)


def _block_values(payload, qtype):
    """Integer weights [blocks, 32] of GGUF block payloads (scales excluded)."""
    if qtype == "Q8_0":
        return payload.contiguous().view(torch.int8).float()
    if qtype == "Q4_0":
        return torch.cat((payload & 15, payload >> 4), -1).float() - 8
    # Q5_0: 4 bytes holding the fifth bit of each of the 32 weights, read as one
    # little-endian int32, then the Q4_0-style nibbles. Viewing bytes as int32
    # needs contiguous memory starting at a multiple of 4 bytes, and blocks are
    # 20 or 22 bytes apart, so those 4 bytes of each block are copied first.
    high = payload[:, :4].clone(memory_format=torch.contiguous_format).view(torch.int32)
    bits = (high >> torch.arange(32, device=payload.device, dtype=torch.int32)) & 1
    low = payload[:, 4:]
    return (torch.cat((low & 15, low >> 4), -1).int() | (bits << 4)).float() - 16


def dequantize(data, qtype, shape, dtype):
    if qtype in ("F16", "F32"):
        return data.reshape(shape).to(dtype)
    if qtype == "BF16":
        return data.reshape(-1).view(torch.bfloat16).reshape(shape).to(dtype)
    if qtype not in BLOCK_BYTES:
        raise ValueError(f"Unsupported GGUF quantization: {qtype}")
    block_bytes = BLOCK_BYTES[qtype]
    n = 1
    for dim in shape:
        n *= dim
    if n % 32 or data.numel() != n // 32 * block_bytes:
        raise ValueError(f"Invalid {qtype} payload for shape {shape}")
    blocks = data.contiguous().view(torch.uint8).reshape(-1, block_bytes)
    scales = blocks[:, :2].contiguous().view(torch.float16).float().reshape(-1, 1)
    return (_block_values(blocks[:, 2:], qtype) * scales).reshape(shape).to(dtype)

def pack_gguf_soa(tensor):
    """Reorder a GGUF Q4_0/Q5_0/Q8_0 matrix on CPU into its final Q/scales layout.

    The parameters contain only the final payload and FP16 scale *bits*;
    ComfyUI's ordinary CPU/GPU offloading transfers these tensors without
    ever allocating a second native GGUF weight in VRAM.
    """
    qtype = type_name(tensor)
    shape = tensor_shape(tensor)
    if qtype not in BLOCK_BYTES or len(shape) != 2 or shape[1] % 32:
        raise ValueError(f"Expected rank-2 GGUF Q4_0/Q5_0/Q8_0 matrix: {tensor.name}")
    block_bytes = BLOCK_BYTES[qtype]
    blocks = shape[0] * (shape[1] // 32)
    src = torch.from_numpy(np.asarray(tensor.data).view(np.uint8))
    if src.numel() != blocks * block_bytes:
        raise ValueError(f"Invalid packed GGUF payload: {tensor.name}")
    raw = src.reshape(blocks, block_bytes)
    # Store scale as raw bytes to preserve FP16 bits even if ComfyUI calls
    # model.to(dtype=float32/bfloat16). No native AoS copy survives. Torch's
    # multithreaded copies run ~2x faster than NumPy's here; with even block
    # sizes each leading FP16 scale is one int16 column.
    q = raw[:, 2:].clone(memory_format=torch.contiguous_format).reshape(-1)
    scales = raw.view(torch.int16)[:, 0].clone(memory_format=torch.contiguous_format).view(torch.uint8)
    return q, scales


def dequantize_soa(qbytes, scale_bits, qtype, shape, dtype):
    """Decode the Q/scales representation on CPU or GPU."""
    if qtype not in BLOCK_BYTES:
        raise ValueError(f"Unsupported SoA quantization: {qtype}")
    n = 1
    for dim in shape:
        n *= dim
    width = BLOCK_BYTES[qtype] - 2
    if (n % 32 or qbytes.numel() != (n // 32) * width or
            scale_bits.numel() != (n // 32) * 2):
        raise ValueError(f"Invalid SoA payload for shape {shape}")
    scale = scale_bits.contiguous().view(torch.float16).float().reshape(-1, 1)
    return (_block_values(qbytes.reshape(-1, width), qtype) * scale).reshape(shape).to(dtype)


@dataclass(frozen=True)
class PackedSpec:
    qtype: str
    shape: tuple[int, ...]

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

# Triton kernel code of each SoA quantization.
_QTYPE_CODE = {"Q4_0": 4, "Q5_0": 5, "Q8_0": 8}
# SoA GEMV launch parameters (bn, blocks, num_warps) per quantization and
# decode rows, from `tools/tune_kernels.py --sweep` on an RTX 3060.
SOA_GEMV_CONFIG = {
    ("Q4_0", 1): (4, 32, 4), ("Q4_0", 2): (1, 32, 2),
    ("Q5_0", 1): (4, 16, 2), ("Q5_0", 2): (8, 16, 2),
    ("Q8_0", 1): (1, 32, 2), ("Q8_0", 2): (1, 16, 2),
}

if triton is not None:
    @triton.jit
    def _soa_mm(X, Q, S, B, Y, M: tl.constexpr, N: tl.constexpr,
                K: tl.constexpr, QTYPE: tl.constexpr,
                HAS_BIAS: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, BK: tl.constexpr):
        """Quantized SoA GEMM for prefill and the encoders (M > 2)."""
        rows = tl.program_id(0) * BM + tl.arange(0, BM)
        outs = tl.program_id(1) * BN + tl.arange(0, BN)
        lanes = tl.arange(0, BK)
        width: tl.constexpr = 16 if QTYPE == 4 else (20 if QTYPE == 5 else 32)
        blocks_per_row: tl.constexpr = K // 32
        accum = tl.full((BM, BN), 0, tl.float32)
        for tile in range(tl.cdiv(K, BK)):
            indices = tile * BK + lanes
            valid = (outs[:, None] < N) & (indices[None, :] < K)
            offset = outs[:, None] * blocks_per_row + indices[None, :] // 32
            scale = tl.load(S + offset, valid, other=0).to(tl.float32)
            within = indices[None, :] % 32
            if QTYPE == 4:
                qbyte = tl.load(Q + offset * width + within % 16, valid, other=0)
                values = ((qbyte >> ((within // 16) * 4)) & 15).to(tl.float32) - 8.
            elif QTYPE == 5:
                qbyte = tl.load(Q + offset * width + 4 + within % 16, valid, other=0)
                high = tl.load(Q + offset * width + within // 8, valid, other=0)
                values = (((qbyte >> ((within // 16) * 4)) & 15) |
                          (((high >> (within % 8)) & 1) << 4)).to(tl.float32) - 16.
            else:
                qbyte = tl.load(Q + offset * width + within, valid, other=0)
                values = qbyte.to(tl.int8, bitcast=True).to(tl.float32)
            weights = (values * scale).to(tl.float16)
            activation = tl.load(
                X + rows[:, None] * K + indices[None, :],
                (rows[:, None] < M) & (indices[None, :] < K),
                other=0).to(tl.float16)
            accum += tl.dot(activation, tl.trans(weights))
        if HAS_BIAS:
            bias = tl.load(B + outs, outs < N, other=0).to(tl.float32)
            accum += bias[None, :]
        tl.store(Y + rows[:, None] * N + outs[None, :], accum,
                 (rows[:, None] < M) & (outs[None, :] < N))

    @triton.jit
    def _soa_gemv(X, Q, S, Y, B,
                  M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                  QTYPE: tl.constexpr, BN: tl.constexpr, BLOCKS: tl.constexpr,
                  HAS_BIAS: tl.constexpr):
        """Read fully aligned contiguous Q bytes and separately packed FP16 scales."""
        rows = tl.program_id(0) * BN + tl.arange(0, BN)
        blocks = tl.arange(0, BLOCKS)
        lanes = tl.arange(0, 32 if QTYPE == 8 else 16)
        # Q5_0 SoA blocks are a little-endian uint32 of fifth bits plus 16 nibble bytes.
        width: tl.constexpr = 32 if QTYPE == 8 else (20 if QTYPE == 5 else 16)
        nibbles: tl.constexpr = 4 if QTYPE == 5 else 0
        nblocks: tl.constexpr = K // 32
        total0 = tl.full((BN, BLOCKS), 0., tl.float32)
        if M == 2:
            total1 = tl.full((BN, BLOCKS), 0., tl.float32)
        for tile in range(tl.cdiv(nblocks, BLOCKS)):
            bid = tile * BLOCKS + blocks
            valid = bid < nblocks
            mask = (rows[:, None] < N) & valid[None, :]
            bid_flat = rows[:, None] * nblocks + bid[None, :]
            scale = tl.load(S + bid_flat, mask, other=0).to(tl.float32)
            raw = tl.load(Q + bid_flat[:, :, None] * width + nibbles + lanes[None, None, :],
                          mask[:, :, None], other=0)
            idx = bid[:, None] * 32 + lanes[None, :]
            if QTYPE == 8:
                q = raw.to(tl.int8, bitcast=True).to(tl.float32)
                x0 = tl.load(X + idx, valid[:, None], other=0).to(tl.float32)
                total0 += tl.sum(q * x0[None, :, :], 2) * scale
                if M == 2:
                    x1 = tl.load(X + K + idx, valid[:, None], other=0).to(tl.float32)
                    total1 += tl.sum(q * x1[None, :, :], 2) * scale
            else:
                if QTYPE == 5:
                    shifts = tl.arange(0, 4)
                    fifth = tl.sum(tl.load(Q + bid_flat[:, :, None] * width + shifts[None, None, :],
                                           mask[:, :, None], other=0).to(tl.int32) << (shifts * 8)[None, None, :], 2)
                    qlo = ((raw & 15).to(tl.int32) | (((fifth[:, :, None] >> lanes[None, None, :]) & 1) << 4)).to(tl.float32) - 16.
                    qhi = ((raw >> 4).to(tl.int32) | (((fifth[:, :, None] >> (lanes[None, None, :] + 16)) & 1) << 4)).to(tl.float32) - 16.
                else:
                    qlo = (raw & 15).to(tl.float32) - 8.
                    qhi = (raw >> 4).to(tl.float32) - 8.
                x0lo = tl.load(X + idx, valid[:, None], other=0).to(tl.float32)
                x0hi = tl.load(X + idx + 16, valid[:, None], other=0).to(tl.float32)
                total0 += tl.sum(qlo * x0lo[None, :, :] + qhi * x0hi[None, :, :], 2) * scale
                if M == 2:
                    x1lo = tl.load(X + K + idx, valid[:, None], other=0).to(tl.float32)
                    x1hi = tl.load(X + K + idx + 16, valid[:, None], other=0).to(tl.float32)
                    total1 += tl.sum(qlo * x1lo[None, :, :] + qhi * x1hi[None, :, :], 2) * scale
        out0 = tl.sum(total0, 1)
        if HAS_BIAS:
            bias = tl.load(B + rows, rows < N, other=0).to(tl.float32)
            out0 += bias
        tl.store(Y + rows, out0, rows < N)
        if M == 2:
            out1 = tl.sum(total1, 1)
            if HAS_BIAS:
                out1 += bias
            tl.store(Y + N + rows, out1, rows < N)


def soa_gemv(x: torch.Tensor, q: torch.Tensor, s: torch.Tensor,
             out: torch.Tensor, qtype: str, *, bn: int = 4,
             blocks: int = 16, num_warps: int = 4, bias=None) -> None:
    """GEMV for the separately packed Q/scale layout, supporting M=1/2."""
    if qtype not in ("Q4_0", "Q5_0", "Q8_0"):
        raise ValueError(f"Unsupported GGUF type: {qtype}")
    if x.ndim != 2 or out.ndim != 2 or x.shape[0] not in (1, 2):
        raise ValueError("SOA kernel requires rank-2 input with M=1/2")
    m, k = x.shape
    n = out.shape[1]
    if k <= 0 or k % 32 or n <= 0 or out.shape[0] != m:
        raise ValueError("Invalid matrix dimensions")
    if bn not in (1, 2, 4, 8, 16) or blocks not in (4, 8, 16, 32, 64):
        raise ValueError("Unsupported GEMV tile size")
    if num_warps not in (2, 4, 8):
        raise ValueError("num_warps must be 2, 4 or 8")
    if x.device.type != "cuda" or q.device != x.device or s.device != x.device or out.device != x.device:
        raise ValueError("Expected CUDA tensors on the same device")
    if x.dtype != torch.float16 or out.dtype != torch.float16 or q.dtype != torch.uint8 or s.dtype != torch.float16:
        raise ValueError("Expected FP16 input/output/scales and uint8 Q bytes")
    if not (x.is_contiguous() and q.is_contiguous() and s.is_contiguous() and out.is_contiguous()):
        raise ValueError("Expected contiguous tensors")
    if bias is not None and (bias.shape != (n,) or
                             bias.device != x.device or
                             bias.dtype != torch.float16 or
                             not bias.is_contiguous()):
        raise ValueError("SoA GEMV bias must be contiguous FP16 on input device")
    total = n * (k // 32)
    if q.numel() != total * (BLOCK_BYTES[qtype] - 2) or s.numel() != total:
        raise ValueError("Incorrect repacked weight size")
    _soa_gemv[(triton.cdiv(n, bn),)](
        x, q, s, out, bias if bias is not None else q,
        m, n, k, _QTYPE_CODE[qtype],
        bn, blocks, bias is not None, num_warps=num_warps)


# Triton checks pointer arguments with cuPointerGetAttribute, which rejects
# memory allocated while a CUDA graph is captured with cudaMallocAsync (the
# ComfyUI default). A decode graph therefore runs its Triton linears on
# input/output buffers allocated by the eager warm-up of the same step.
_GRAPH_IO = ContextVar("starvector_graph_io", default=None)


@contextmanager
def graph_io(buffers):
    """Warm-up (empty `buffers`) allocates static Triton input/output pairs in
    call order; the CUDA graph capture that follows reuses them."""
    token = _GRAPH_IO.set((buffers, not buffers, [0]))
    try:
        yield
    finally:
        _GRAPH_IO.reset(token)


def _fused_soa_linear(x, qbytes, scale_bits, spec, bias):
    """Dispatch the SoA GEMV for 1/2 rows and the SoA GEMM for more rows."""
    if (triton is None or x.device.type != "cuda" or
            os.getenv("STARVECTOR_DISABLE_TRITON") == "1" or
            x.dtype != torch.float16 or x.numel() % spec.shape[1]):
        return None
    nrows = x.numel() // spec.shape[1]
    n, k = spec.shape
    try:
        inp = x.reshape(nrows, k)
        io = _GRAPH_IO.get()
        if io is None:
            inp = inp.contiguous()
            out = torch.empty((nrows, n), dtype=inp.dtype, device=inp.device)
        else:
            buffers, warmup, calls = io
            if warmup:
                buffers.append((torch.empty((nrows, k), dtype=inp.dtype, device=inp.device),
                                torch.empty((nrows, n), dtype=inp.dtype, device=inp.device)))
            static_in, out = buffers[calls[0]]
            calls[0] += 1
            inp = static_in.copy_(inp)
        scales = scale_bits.view(torch.float16)
        if nrows in (1, 2):
            bn, blocks, warps = SOA_GEMV_CONFIG[spec.qtype, nrows]
            soa_gemv(inp, qbytes, scales, out, spec.qtype,
                     bn=bn, blocks=blocks, num_warps=warps, bias=bias)
        else:
            # The GEMV is tuned for M=1/2; prefill and vision blocks use the
            # SoA GEMM on the same arrays.
            _soa_mm[(triton.cdiv(nrows, 16), triton.cdiv(n, 32))](
                inp, qbytes, scales, bias if bias is not None else qbytes,
                out, nrows, n, k, _QTYPE_CODE[spec.qtype],
                bias is not None, 16, 32, 32, num_warps=4)
        return out.reshape(*x.shape[:-1], n)
    except Exception as exc:
        _LOG.warning("StarVector SoA kernel failed; using tiled fallback: %s", exc)
        return None


class GGUFWeights(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("weight", None)
        self.register_parameter("weight_scales", None)
        self.register_parameter("bias", None)
        self._specs: dict[str, PackedSpec] = {}

    def set_gguf(self, key, tensor):
        if key not in ("weight", "bias"):
            raise ValueError("Unexpected weight key")
        qtype, shape = type_name(tensor), tensor_shape(tensor)
        payload = tensor_payload(tensor)
        if qtype == "BF16":
            # Floating point once, so the model dtype cast applies at load time:
            # a per-call BF16 decode would allocate inside decode CUDA graphs.
            payload, qtype = dequantize(payload, qtype, shape, torch.float32), "F32"
        self.register_parameter(key, nn.Parameter(payload, requires_grad=False))
        self._specs[key] = PackedSpec(qtype, shape)

    def _set_soa_weight(self, tensor):
        if type_name(tensor) not in BLOCK_BYTES:
            return False
        q, scales = pack_gguf_soa(tensor)
        self.register_parameter(
            "weight", nn.Parameter(q, requires_grad=False))
        self.register_parameter(
            "weight_scales", nn.Parameter(scales, requires_grad=False))
        self._specs["weight"] = PackedSpec(type_name(tensor), tensor_shape(tensor))
        return True

    def unpack(self, key, dtype, device):
        data = getattr(self, key)
        if data is None:
            return None
        spec = self._specs[key]
        if key == "weight" and self.weight_scales is not None:
            return dequantize_soa(
                data.to(device=device), self.weight_scales.to(device=device),
                spec.qtype, spec.shape, dtype)
        return dequantize(data.to(device=device), spec.qtype, spec.shape, dtype)

class GGUFLinear(GGUFWeights):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features

    def set_gguf(self, key, tensor):
        if key == "weight" and self._set_soa_weight(tensor):
            return
        super().set_gguf(key, tensor)

    def forward(self, x):
        spec = self._specs.get("weight")
        if spec is None:
            raise RuntimeError("GGUFLinear missing weight")
        if spec.shape != (self.out_features, self.in_features):
            raise ValueError(f"Unexpected weight shape {spec.shape}")
        bias = self.unpack("bias", x.dtype, x.device)
        if spec.qtype in BLOCK_BYTES:
            fused = _fused_soa_linear(
                x, self.weight, self.weight_scales, spec, bias)
            if fused is not None:
                return fused
            # CPU, non-FP16 inputs or missing Triton: dequantize small
            # slices of the resident SoA layout directly.
            rows_per_chunk = max(1, 2 * 1024 * 1024 // self.in_features)
            x_flat = x.reshape(-1, self.in_features)
            blocks_per_row = self.in_features // 32
            payload = self.weight.reshape(
                self.out_features, blocks_per_row * (BLOCK_BYTES[spec.qtype] - 2))
            scales = self.weight_scales.reshape(
                self.out_features, blocks_per_row * 2)
            outputs = []
            for start in range(0, self.out_features, rows_per_chunk):
                end = min(start + rows_per_chunk, self.out_features)
                weights = dequantize_soa(
                    payload[start:end].flatten(), scales[start:end].flatten(),
                    spec.qtype, (end - start, self.in_features), x.dtype)
                outputs.append(F.linear(
                    x_flat, weights, bias[start:end] if bias is not None else None))
            return torch.cat(outputs, -1).reshape(*x.shape[:-1], self.out_features)
        return F.linear(x, self.unpack("weight", x.dtype, x.device), bias)

class GGUFEmbedding(GGUFWeights):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.output_dtype = torch.float32

    def set_gguf(self, key, tensor):
        if key == "weight" and self._set_soa_weight(tensor):
            return
        super().set_gguf(key, tensor)

    def forward(self, ids):
        spec = self._specs.get("weight")
        if spec is None:
            raise RuntimeError("GGUFEmbedding missing weight")
        if spec.qtype in ("F16", "F32", "BF16"):
            return F.embedding(ids, self.unpack("weight", self.output_dtype, ids.device))
        if spec.shape[-1] % 32:
            raise ValueError("GGUF embedding hidden dimension must be divisible by 32")
        blocks_per_row = spec.shape[-1] // 32
        stride = blocks_per_row * (BLOCK_BYTES[spec.qtype] - 2)
        flat_ids = ids.reshape(-1)
        if flat_ids.numel() <= 8:
            # Avoid torch.unique/synchronization in token-by-token decoding.
            selected = flat_ids
            inverse = None
        else:
            selected, inverse = torch.unique(flat_ids, return_inverse=True)
        q = self.weight.reshape(spec.shape[0], stride).index_select(0, selected)
        scales = self.weight_scales.reshape(
            spec.shape[0], blocks_per_row * 2).index_select(0, selected)
        decoded = dequantize_soa(
            q.flatten(), scales.flatten(), spec.qtype,
            (selected.numel(), spec.shape[-1]), self.output_dtype)
        if inverse is not None:
            decoded = decoded.index_select(0, inverse)
        return decoded.reshape(*ids.shape, spec.shape[-1])

def tied_head(embedding):
    """Output projection sharing both GGUF parameter arrays of its embedding."""
    head = GGUFLinear(embedding.embedding_dim, embedding.num_embeddings)
    head.weight = embedding.weight
    head.weight_scales = embedding.weight_scales
    head._specs["weight"] = embedding._specs["weight"]
    return head

class GGUFConv2d(GGUFWeights):
    """Patch embedding convolution with stride equal to the patch size."""

    def __init__(self, stride):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        return F.conv2d(x, self.unpack("weight", x.dtype, x.device),
                        self.unpack("bias", x.dtype, x.device), stride=self.stride)
