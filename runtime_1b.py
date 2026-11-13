"""StarVector 1B GGUF runtime (CLIP + GPTBigCode)."""
from __future__ import annotations

import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from .quantization import GGUFConv2d, GGUFLinear, tensor_shape, tied_head
from .runtime_shared import (ADAPTER_PREFIX, DECODER_PREFIX, VISION_PREFIX, CachedDecoding, StarVectorRuntimeBase,
                             comfy_attention, image_to_pil, normalize)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class BigCodeBlock(nn.Module):
    """GPTBigCode multi-query block that stores its keys/values in a fixed cache."""

    def __init__(self, store, name, hidden, heads, inner):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.ln_1 = store.norm(name + '.ln_1', (hidden,))
        self.c_attn = store.linear(name + '.attn.c_attn', hidden, hidden + 2 * self.head_dim)
        self.attn_proj = store.linear(name + '.attn.c_proj', hidden, hidden)
        self.ln_2 = store.norm(name + '.ln_2', (hidden,))
        self.c_fc = store.linear(name + '.mlp.c_fc', hidden, inner)
        self.c_proj = store.linear(name + '.mlp.c_proj', inner, hidden)

    def forward(self, x, keys, values, position=None, bias=None):
        """Prefill (position None) fills cache slots from 0 and attends causally
        through ComfyUI; decode writes slot `position` and attends the cache
        span of `keys`, with `bias` hiding the slots after `position`."""
        rows, length, hidden = x.shape
        q, k, v = self.c_attn(self.ln_1(x)).split((hidden, self.head_dim, self.head_dim), -1)
        if position is None:
            keys[:, :length] = k
            values[:, :length] = v
            q = q.view(rows, length, self.heads, self.head_dim).transpose(1, 2)
            out = comfy_attention(q, k[:, None], v[:, None], causal=True).reshape(rows, length, hidden)
        else:
            keys.index_copy_(1, position, k)
            values.index_copy_(1, position, v)
            # Multi-query: every head of a row attends the same key/value head.
            scores = torch.baddbmm(bias.expand(rows, self.heads, -1), q.view(rows, self.heads, self.head_dim),
                                   keys.transpose(1, 2), alpha=self.head_dim ** -0.5)
            out = torch.bmm(torch.softmax(scores, -1, dtype=torch.float32).to(x.dtype), values).view(rows, 1, hidden)
        x = x + self.attn_proj(out)
        return x + self.c_proj(F.gelu(self.c_fc(self.ln_2(x)), approximate='tanh'))


class GPTBigCode(nn.Module):
    """StarVector-1B GPTBigCode decoder for the shared token generation loop."""

    def __init__(self, store, cfg, vocab_size):
        super().__init__()
        if not cfg.get('multi_query', True) or cfg.get('activation_function', 'gelu_pytorch_tanh') != 'gelu_pytorch_tanh':
            raise NotImplementedError('Only multi-query GPTBigCode with gelu_pytorch_tanh is supported')
        prefix = DECODER_PREFIX + 'transformer.'
        hidden, heads = int(cfg['hidden_size']), int(cfg['num_attention_heads'])
        self.context = int(cfg.get('max_position_embeddings', 8192))
        self.head_dim = hidden // heads
        self.wte = store.embedding(prefix + 'wte', vocab_size, hidden)
        self.wpe = store.embedding(prefix + 'wpe', self.context, hidden)
        inner = int(cfg.get('n_inner') or 4 * hidden)
        self.h = nn.ModuleList([BigCodeBlock(store, prefix + f'h.{i}', hidden, heads, inner) for i in range(int(cfg['num_hidden_layers']))])
        self.ln_f = store.norm(prefix + 'ln_f', (hidden,))
        self.lm_head = tied_head(self.wte)

    def embed(self, ids):
        return self.wte(ids)

    def prefill(self, embeddings, rows, capacity):
        """Encode the prompt once; its cache is shared by all `rows` decode rows."""
        length = embeddings.shape[1]
        keys = torch.zeros((len(self.h), rows, capacity, self.head_dim), device=embeddings.device, dtype=embeddings.dtype)
        values = torch.zeros_like(keys)
        x = embeddings + self.wpe(torch.arange(length, device=embeddings.device))
        for block, block_keys, block_values in zip(self.h, keys, values):
            x = block(x, block_keys, block_values)
        logits = self.lm_head(self.ln_f(x[:, -1]))
        return CachedDecoding(self, keys, values, length), logits.expand(rows, -1)

    def decode(self, tokens, position, keys, values, span):
        """One token per row at `position`, attending the first `span` cache slots."""
        slots = torch.arange(span, device=keys.device)
        bias = torch.zeros(span, device=keys.device, dtype=keys.dtype).masked_fill_(slots > position, float('-inf'))
        x = (self.wte(tokens) + self.wpe(position))[:, None]
        for block, block_keys, block_values in zip(self.h, keys, values):
            x = block(x, block_keys[:, :span], block_values[:, :span], position, bias)
        return self.lm_head(self.ln_f(x[:, 0]))


class CLIPAttention(nn.Module):
    def __init__(self, store, name, width, heads):
        super().__init__()
        self.heads = heads
        self.in_proj = GGUFLinear(width, width * 3)
        self.in_proj.set_gguf('weight', store.get(name + '.in_proj_weight'))
        self.in_proj.set_gguf('bias', store.get(name + '.in_proj_bias'))
        self.out_proj = store.linear(name + '.out_proj', width, width)

    def forward(self, x):
        batch, sequence, width = x.shape
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        def heads(value):
            return value.reshape(batch, sequence, self.heads, width // self.heads).transpose(1, 2)
        q, k, v = map(heads, (q, k, v))
        # The CLIP vision encoder always uses ComfyUI's selected attention.
        return self.out_proj(comfy_attention(q, k, v).reshape(batch, sequence, width))


class CLIPBlock(nn.Module):
    def __init__(self, store, name, width, heads):
        super().__init__()
        self.ln_1 = store.norm(name + '.ln_1', (width,))
        self.attn = CLIPAttention(store, name + '.attn', width, heads)
        self.ln_2 = store.norm(name + '.ln_2', (width,))
        self.mlp = nn.Module()
        self.mlp.add_module('c_fc', store.linear(name + '.mlp.c_fc', width, width * 4))
        self.mlp.add_module('c_proj', store.linear(name + '.mlp.c_proj', width * 4, width))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        z = self.mlp.c_fc(self.ln_2(x))
        z = z * torch.sigmoid(1.702 * z)
        return x + self.mlp.c_proj(z)


class CLIPVision(nn.Module):
    def __init__(self, store):
        super().__init__()
        prefix = VISION_PREFIX + 'visual_encoder.'
        width, patch = 1024, 14
        self.conv1 = GGUFConv2d(stride=patch)
        self.conv1.set_gguf('weight', store.get(prefix + 'conv1.weight'))
        self.class_embedding = nn.Parameter(store.dense(prefix + 'class_embedding'), requires_grad=False)
        self.positional_embedding = nn.Parameter(store.dense(prefix + 'positional_embedding'), requires_grad=False)
        self.ln_pre = store.norm(prefix + 'ln_pre', (width,))
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList([CLIPBlock(store, prefix + f'transformer.resblocks.{i}', width, 16) for i in range(23)])
        self.ln_vision = store.norm(VISION_PREFIX + 'ln_vision', (width,))
        if self.positional_embedding.shape != ((224 // patch) ** 2 + 1, width):
            raise ValueError('CLIP positional embedding does not match 224x224 inputs')

    def forward(self, image):
        x = self.conv1(image).flatten(2).transpose(1, 2)
        class_token = self.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(image.shape[0], -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x = self.ln_pre(x + self.positional_embedding.to(x.dtype))
        for block in self.transformer.resblocks:
            x = block(x)
        return self.ln_vision(x)


class ProjectionAdapter(nn.Module):
    def __init__(self, store, hidden):
        super().__init__()
        self.c_fc = store.linear(ADAPTER_PREFIX + 'c_fc', 1024, 2048)
        self.c_proj = store.linear(ADAPTER_PREFIX + 'c_proj', 2048, hidden)
        norm_name = ADAPTER_PREFIX + 'norm.'
        self.register_buffer('running_mean', store.dense(norm_name + 'running_mean'))
        self.register_buffer('running_var', store.dense(norm_name + 'running_var'))
        self.register_buffer('norm_weight', store.dense(norm_name + 'weight'))
        self.register_buffer('norm_bias', store.dense(norm_name + 'bias'))
        if self.running_mean.numel() != 257:
            raise ValueError('Expected a 257-channel BatchNorm adapter for StarVector 1B')

    def forward(self, vision_embeddings):
        x = self.c_fc(vision_embeddings)
        x = x * torch.sigmoid(x)
        x = self.c_proj(x)
        return F.batch_norm(x, self.running_mean, self.running_var, self.norm_weight, self.norm_bias, training=False, eps=1e-5)


def preprocess_image(image):
    """ComfyUI IMAGE -> white-padded square, bicubic 224x224, CLIP normalization."""
    pil = image_to_pil(image)
    side = max(pil.size)
    canvas = Image.new('RGB', (side, side), (255, 255, 255))
    canvas.paste(pil, ((side - pil.width) // 2, (side - pil.height) // 2))
    return normalize(canvas.resize((224, 224), Image.Resampling.BICUBIC), CLIP_MEAN, CLIP_STD)


class StarVector1B(StarVectorRuntimeBase):
    """StarVector 1B: CLIP ViT-L/14 encoder, BatchNorm adapter and GPTBigCode decoder."""

    def _build(self, store, cfg):
        if cfg.get('image_encoder_type') != 'clip' or int(cfg.get('image_size', 224)) != 224:
            raise ValueError('Embedded GGUF config is not a StarVector 1B CLIP-224 model')
        vocab_size = tensor_shape(store.get(DECODER_PREFIX + 'transformer.wte.weight'))[0]
        self.decoder = GPTBigCode(store, cfg, vocab_size)
        self.vision = CLIPVision(store)
        self.adapter = ProjectionAdapter(store, int(cfg['hidden_size']))

    def _preprocess_image(self, image):
        return preprocess_image(image)
