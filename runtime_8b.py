"""StarVector 8B GGUF runtime (SigLIP + StarCoder2)."""
from __future__ import annotations

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .quantization import GGUFConv2d, tensor_shape, tied_head
from .runtime_shared import (ADAPTER_PREFIX, DECODER_PREFIX, VISION_PREFIX, CachedDecoding, StarVectorRuntimeBase,
                             comfy_attention, image_to_pil, normalize)


def _rotate(x, cos, sin):
    """Rotary embedding over rotated halves, as Transformers' Starcoder2."""
    first, second = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-second, first), dim=-1) * sin


class Starcoder2Block(nn.Module):
    """Starcoder2 grouped-query block that stores its keys/values in a fixed cache."""

    def __init__(self, store, name, hidden, heads, kv_heads, inner):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, hidden // heads
        self.input_layernorm = store.norm(name + '.input_layernorm', (hidden,))
        self.q_proj = store.linear(name + '.self_attn.q_proj', hidden, hidden)
        self.k_proj = store.linear(name + '.self_attn.k_proj', hidden, kv_heads * self.head_dim)
        self.v_proj = store.linear(name + '.self_attn.v_proj', hidden, kv_heads * self.head_dim)
        self.o_proj = store.linear(name + '.self_attn.o_proj', hidden, hidden)
        self.post_attention_layernorm = store.norm(name + '.post_attention_layernorm', (hidden,))
        self.c_fc = store.linear(name + '.mlp.c_fc', hidden, inner)
        self.c_proj = store.linear(name + '.mlp.c_proj', inner, hidden)

    def forward(self, x, cos, sin, keys, values, position=None, bias=None):
        """Prefill (position None) fills cache slots from 0 and attends causally
        through ComfyUI; decode writes slot `position` and attends the cache
        span of `keys`, with `bias` hiding slots outside the sliding window."""
        rows, length, hidden = x.shape
        h = self.input_layernorm(x)
        q = _rotate(self.q_proj(h).view(rows, length, self.heads, self.head_dim).transpose(1, 2), cos, sin)
        k = _rotate(self.k_proj(h).view(rows, length, self.kv_heads, self.head_dim).transpose(1, 2), cos, sin)
        v = self.v_proj(h).view(rows, length, self.kv_heads, self.head_dim).transpose(1, 2)
        if position is None:
            keys[:, :, :length] = k
            values[:, :, :length] = v
            out = comfy_attention(q, k, v, causal=True).reshape(rows, length, hidden)
        else:
            keys.index_copy_(2, position, k)
            values.index_copy_(2, position, v)
            # Each key/value head serves a contiguous group of query heads.
            group = self.heads // self.kv_heads
            scores = torch.baddbmm(bias.expand(rows * self.kv_heads, group, -1), q.reshape(rows * self.kv_heads, group, self.head_dim),
                                   keys.flatten(0, 1).transpose(1, 2), alpha=self.head_dim ** -0.5)
            out = torch.bmm(torch.softmax(scores, -1, dtype=torch.float32).to(x.dtype), values.flatten(0, 1)).view(rows, 1, hidden)
        x = x + self.o_proj(out)
        return x + self.c_proj(F.gelu(self.c_fc(self.post_attention_layernorm(x)), approximate='tanh'))


class Starcoder2(nn.Module):
    """StarVector-8B Starcoder2 decoder for the shared token generation loop."""

    def __init__(self, store, cfg, vocab_size):
        super().__init__()
        prefix = DECODER_PREFIX + 'model.'
        hidden, heads = int(cfg['hidden_size']), int(cfg['num_attention_heads'])
        self.kv_heads = int(cfg.get('num_kv_heads', heads))
        self.head_dim = hidden // heads
        self.context = int(cfg.get('max_position_embeddings', cfg.get('max_length_train', 16000)))
        # bigcode/starcoder2-7b values: StarVector's own config omits them.
        self.rope_theta = float(cfg.get('rope_theta', 1e6))
        self.window = int(cfg.get('sliding_window', 4096))
        inner = int(cfg.get('intermediate_size', 4 * hidden))
        self.embed_tokens = store.embedding(prefix + 'embed_tokens', vocab_size, hidden)
        self.layers = nn.ModuleList([Starcoder2Block(store, prefix + f'layers.{i}', hidden, heads, self.kv_heads, inner) for i in range(int(cfg['num_hidden_layers']))])
        self.norm = store.norm(prefix + 'norm', (hidden,))
        self.lm_head = tied_head(self.embed_tokens)

    def _rope(self, positions, dtype):
        inv_freq = 1. / (self.rope_theta ** (torch.arange(0, self.head_dim, 2, device=positions.device, dtype=torch.float32) / self.head_dim))
        angles = positions.float()[:, None] * inv_freq
        angles = torch.cat((angles, angles), -1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def embed(self, ids):
        return self.embed_tokens(ids)

    def prefill(self, embeddings, rows, capacity):
        """Encode the prompt once; its cache is shared by all `rows` decode rows.
        The prompt (576 image tokens and '<svg') is shorter than the sliding window."""
        length = embeddings.shape[1]
        keys = torch.zeros((len(self.layers), rows, self.kv_heads, capacity, self.head_dim), device=embeddings.device, dtype=embeddings.dtype)
        values = torch.zeros_like(keys)
        cos, sin = self._rope(torch.arange(length, device=embeddings.device), embeddings.dtype)
        x = embeddings
        for layer, layer_keys, layer_values in zip(self.layers, keys, values):
            x = layer(x, cos, sin, layer_keys, layer_values)
        logits = self.lm_head(self.norm(x[:, -1]))
        return CachedDecoding(self, keys, values, length), logits.expand(rows, -1)

    def decode(self, tokens, position, keys, values, span):
        """One token per row at `position`, attending the first `span` cache slots."""
        slots = torch.arange(span, device=keys.device)
        hidden_slots = (slots > position) | (slots <= position - self.window)
        bias = torch.zeros(span, device=keys.device, dtype=keys.dtype).masked_fill_(hidden_slots, float('-inf'))
        cos, sin = self._rope(position, keys.dtype)
        x = self.embed_tokens(tokens)[:, None]
        for layer, layer_keys, layer_values in zip(self.layers, keys, values):
            x = layer(x, cos, sin, layer_keys[:, :, :span], layer_values[:, :, :span], position, bias)
        return self.lm_head(self.norm(x[:, 0]))


class SiglipLayer(nn.Module):
    def __init__(self, store, name, width, heads, inner):
        super().__init__()
        self.heads = heads
        self.layer_norm1 = store.norm(name + '.layer_norm1', (width,), eps=1e-6)
        self.q_proj = store.linear(name + '.self_attn.q_proj', width, width)
        self.k_proj = store.linear(name + '.self_attn.k_proj', width, width)
        self.v_proj = store.linear(name + '.self_attn.v_proj', width, width)
        self.out_proj = store.linear(name + '.self_attn.out_proj', width, width)
        self.layer_norm2 = store.norm(name + '.layer_norm2', (width,), eps=1e-6)
        self.fc1 = store.linear(name + '.mlp.fc1', width, inner)
        self.fc2 = store.linear(name + '.mlp.fc2', inner, width)

    def forward(self, x):
        batch, length, width = x.shape
        h = self.layer_norm1(x)
        q, k, v = (projection(h).view(batch, length, self.heads, width // self.heads).transpose(1, 2)
                   for projection in (self.q_proj, self.k_proj, self.v_proj))
        # The SigLIP vision encoder always uses ComfyUI's selected attention.
        x = x + self.out_proj(comfy_attention(q, k, v).reshape(batch, length, width))
        return x + self.fc2(F.gelu(self.fc1(self.layer_norm2(x)), approximate='tanh'))


class SiglipVision(nn.Module):
    """SigLIP-large-patch16 vision tower up to its post-layernorm patch tokens."""

    def __init__(self, store, image_size=384, width=1024, heads=16, layers=24, inner=4096, patch=16):
        super().__init__()
        prefix = VISION_PREFIX + 'visual_encoder.'
        self.patch_embedding = GGUFConv2d(stride=patch)
        self.patch_embedding.set_gguf('weight', store.get(prefix + 'embeddings.patch_embedding.weight'))
        self.patch_embedding.set_gguf('bias', store.get(prefix + 'embeddings.patch_embedding.bias'))
        self.position_embedding = store.embedding(prefix + 'embeddings.position_embedding', (image_size // patch) ** 2, width)
        self.layers = nn.ModuleList([SiglipLayer(store, prefix + f'encoder.layers.{i}', width, heads, inner) for i in range(layers)])
        self.post_layernorm = store.norm(prefix + 'post_layernorm', (width,), eps=1e-6)

    def forward(self, pixels):
        x = self.patch_embedding(pixels).flatten(2).transpose(1, 2)
        x = x + self.position_embedding(torch.arange(x.shape[1], device=x.device))
        for layer in self.layers:
            x = layer(x)
        return self.post_layernorm(x)


class ProjectionAdapter8B(nn.Module):
    def __init__(self, store, hidden_size):
        super().__init__()
        self.c_fc = store.linear(ADAPTER_PREFIX + 'c_fc', 1024, 2048)
        self.c_proj = store.linear(ADAPTER_PREFIX + 'c_proj', 2048, hidden_size)
        # LayerNorm over all 576 (384 / 16 squared) image tokens jointly.
        self.norm = store.norm(ADAPTER_PREFIX + 'norm', (576, hidden_size))

    def forward(self, vision):
        x = self.c_fc(vision)
        x = x * torch.sigmoid(x)
        x = self.c_proj(x)
        return self.norm(x)


def preprocess_siglip(image):
    """ComfyUI IMAGE -> bicubic 384x384 stretch, SigLIP normalization."""
    return normalize(image_to_pil(image).resize((384, 384), Image.Resampling.BICUBIC), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class StarVector8B(StarVectorRuntimeBase):
    """StarVector 8B: SigLIP-384 encoder, LayerNorm adapter and StarCoder2 decoder."""

    def _build(self, store, cfg):
        if not str(cfg.get('image_encoder_type', '')).startswith('siglip') or int(cfg.get('image_size', 384)) != 384:
            raise ValueError('Embedded GGUF config is not a StarVector 8B SigLIP-384 model')
        vocab_size = tensor_shape(store.get(DECODER_PREFIX + 'model.embed_tokens.weight'))[0]
        self.decoder = Starcoder2(store, cfg, vocab_size)
        self.vision = SiglipVision(store)
        self.adapter = ProjectionAdapter8B(store, int(cfg['hidden_size']))

    def _preprocess_image(self, image):
        return preprocess_siglip(image)
