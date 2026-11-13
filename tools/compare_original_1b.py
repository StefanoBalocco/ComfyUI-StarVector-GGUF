#!/usr/bin/env python3
"""Measure how far StarVector 1B GGUF files deviate from StarVector's FP16 checkpoint.

All models run through this node's runtime: the checkpoint's safetensors load
at full precision (cuBLAS linears) and the GGUFs use their quantized kernels.
The checkpoint samples an SVG for the image; each model is then teacher-forced
through those tokens with cached CUDA-graph decoding. The script reports the
image-prefix error and, over all tokens, the top-1 agreement and the KL
divergence of each GGUF's next-token distribution from the checkpoint's.

Run it with ComfyUI's Python on a CUDA GPU; ComfyUI supplies the attention.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))  # ComfyUI root, for its attention
package = ModuleType('starvector_node')
package.__path__ = [str(ROOT)]
sys.modules['starvector_node'] = package

from starvector_node.gguf_support import EMBEDDED_TOKENIZER_FILES, inspect_local_gguf  # noqa: E402
from starvector_node.runtime_1b import StarVector1B  # noqa: E402


class OriginalTensor:
    """A safetensors tensor with the name/type/shape/data interface of a GGUF tensor."""

    def __init__(self, path, name, shape, dtype):
        self.path, self.name = path, name
        self.shape = tuple(reversed(shape))  # GGUF dimension order
        self.tensor_type = SimpleNamespace(name=dtype)

    @property
    def data(self):
        with safe_open(str(self.path), framework='pt', device='cpu') as handle:
            return handle.get_tensor(self.name).numpy()


def original_reference(folder):
    """The original checkpoint directory with the attributes of a GGUFReference."""
    index = folder / 'model.safetensors.index.json'
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values()))
    else:
        shards = ['model.safetensors']
    tensors = {}
    for shard in shards:
        with safe_open(str(folder / shard), framework='pt', device='cpu') as handle:
            for name in handle.keys():
                meta = handle.get_slice(name)
                if meta.get_dtype() in ('F16', 'F32'):  # skips BatchNorm's int64 counter
                    tensors[name] = OriginalTensor(folder / shard, name, meta.get_shape(), meta.get_dtype())
    assets = {name: (folder / name).read_text(encoding='utf-8') for name in EMBEDDED_TOKENIZER_FILES if (folder / name).is_file()}
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    return SimpleNamespace(tensor_index=tensors, model_config=config, model_assets=assets)


def forced(model, image, ids):
    """Image prefix and next-token logits along `ids`, through prefill and cached decode."""
    with torch.inference_mode():
        prefix = model.prepare_image_prefix(image)
        embeddings = torch.cat([prefix, model.decoder.embed(torch.tensor([ids[:1]], device='cuda')).to(prefix.dtype)], 1)
        decoding, logits = model.decoder.prefill(embeddings, 1, embeddings.shape[1] + len(ids))
        rows = [logits[0].float().cpu()]
        for token in ids[1:-1]:
            rows.append(decoding.step(torch.tensor([token], device='cuda'))[0].float().cpu())
    return prefix[0].float().cpu(), torch.stack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, required=True, help='Original starvector-1b-im2svg directory (config.json, tokenizer, safetensors)')
    parser.add_argument('--gguf', type=Path, nargs='+', required=True, help='StarVector 1B GGUF files to compare with the original')
    parser.add_argument('--image', type=Path, required=True, help='Input image for the sampled SVG')
    parser.add_argument('--max-new-tokens', type=int, default=1000, help='Maximum new tokens the original model samples')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('CUDA is required')
    with Image.open(args.image) as source:
        image = torch.from_numpy(np.asarray(source.convert('RGB'), dtype=np.uint8).copy()).unsqueeze(0).float() / 255

    original = StarVector1B(original_reference(args.original), torch.float16).to('cuda')
    svg = original.generate_from_prefix(original.prepare_image_prefix(image), max_new_tokens=args.max_new_tokens, seed=args.seed, loop_guard=False)
    ids = original.tokenizer.encode(svg)
    reference_prefix, reference = forced(original, image, ids)
    print(f'original: sampled {len(ids)} SVG tokens', flush=True)
    del original
    gc.collect()
    torch.cuda.empty_cache()

    log_reference = torch.log_softmax(reference, -1)
    for path in args.gguf:
        model = StarVector1B(inspect_local_gguf(path), torch.float16).to('cuda')
        prefix, logits = forced(model, image, ids)
        prefix_error = ((prefix - reference_prefix).norm() / reference_prefix.norm()).item()
        top1 = (logits.argmax(-1) == reference.argmax(-1)).float().mean().item()
        kl = (log_reference.exp() * (log_reference - torch.log_softmax(logits, -1))).sum(-1)
        print(f'{path.name}: image prefix error {prefix_error:.4f}, top-1 agreement {top1:.4f}, '
              f'KL mean {kl.mean().item():.5f} max {kl.max().item():.4f}', flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
