"""Code shared by the StarVector 1B and 8B runtimes.

GGUF tensor access, ComfyUI attention, image conversion, the embedded
tokenizer, the native sampling loop, CUDA-graph KV-cache decoding and the
runtime base class. runtime_1b.py and runtime_8b.py hold only the genuine
CLIP/GPTBigCode versus SigLIP/StarCoder2 differences.
"""
from __future__ import annotations

import json
import logging
from math import sqrt
from time import perf_counter

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer
from torch import nn

from .quantization import GGUFLinear, GGUFEmbedding, dequantize, graph_io, tensor_payload, tensor_shape, type_name


DECODER_PREFIX = 'model.svg_transformer.transformer.'
VISION_PREFIX = 'model.image_encoder.'
ADAPTER_PREFIX = 'model.image_projection.'


def _make_causal_mask(q, k):
    qlen, klen = q.shape[-2], k.shape[-2]
    if qlen <= 1:
        return None
    query_positions = torch.arange(klen - qlen, klen, device=q.device)
    key_positions = torch.arange(klen, device=q.device)
    valid = key_positions[None, :] <= query_positions[:, None]
    return torch.where(valid, 0.0, torch.finfo(q.dtype).min).to(q.dtype)[None, None, :, :]

def comfy_attention(q, k, v, causal=False):
    """[B,H,T,D] queries over [B,Hkv,S,D] keys/values through ComfyUI's attention; returns [B,T,H,D]."""
    mask = _make_causal_mask(q, k) if causal else None
    # Imported at call time, so the tests can substitute a stand-in.
    from comfy.ldm.modules.attention import optimized_attention
    # SageAttention's INT8 Q/K quantization measurably corrupts both the CLIP
    # image prefix and the decoder logits, so opt out of low precision attention.
    output = optimized_attention(
        q, k, v, heads=q.shape[1], mask=mask,
        skip_reshape=True, skip_output_reshape=True, enable_gqa=q.shape[1] != k.shape[1], scale=1.0 / sqrt(q.shape[-1]),
        low_precision_attention=False)
    return output.transpose(1, 2).contiguous()


class TensorStore:
    """Build GGUF layers from the tensors of an opened GGUF file, by name."""

    def __init__(self, tensors):
        self.tensors = tensors

    def get(self, name):
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise ValueError(f'Missing StarVector tensor: {name}') from exc

    def dense(self, name, dtype=torch.float32):
        tensor = self.get(name)
        return dequantize(tensor_payload(tensor), type_name(tensor), tensor_shape(tensor), dtype)

    def linear(self, name, in_features, out_features):
        result = GGUFLinear(in_features, out_features)
        weight = self.get(name + '.weight')
        if tensor_shape(weight) != (out_features, in_features):
            raise ValueError(f'Unexpected shape for {name}.weight: {tensor_shape(weight)}, expected {(out_features, in_features)}')
        result.set_gguf('weight', weight)
        result.set_gguf('bias', self.get(name + '.bias'))
        return result

    def embedding(self, name, num_embeddings, embedding_dim):
        result = GGUFEmbedding(num_embeddings, embedding_dim)
        weight = self.get(name + '.weight')
        if tensor_shape(weight) != (num_embeddings, embedding_dim):
            raise ValueError(f'Unexpected shape for {name}.weight: {tensor_shape(weight)}, expected {(num_embeddings, embedding_dim)}')
        result.set_gguf('weight', weight)
        return result

    def norm(self, name, shape, eps=1e-5):
        layer = nn.LayerNorm(shape, eps=eps)
        weight = self.dense(name + '.weight')
        bias = self.dense(name + '.bias')
        if tuple(layer.weight.shape) != tuple(weight.shape) or tuple(layer.bias.shape) != tuple(bias.shape):
            raise ValueError(f'Invalid layer norm shape: {name}')
        layer.weight = nn.Parameter(weight, requires_grad=False)
        layer.bias = nn.Parameter(bias, requires_grad=False)
        return layer


def image_to_pil(image):
    """The single image of a ComfyUI IMAGE [1,H,W,3/4] as RGB, alpha composited on white."""
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[-1] not in (3, 4):
        raise ValueError('Expected one ComfyUI IMAGE [1,H,W,3/4]')
    data = image[0].detach().cpu().float().clamp(0, 1).numpy()
    rgb = data[..., :3]
    if data.shape[-1] == 4:
        rgb = rgb * data[..., 3:4] + (1.0 - data[..., 3:4])
    return Image.fromarray((rgb * 255).round().astype(np.uint8), mode='RGB')


def normalize(pil, mean, std):
    """RGB PIL image -> normalized [1,3,H,W] float tensor."""
    values = torch.from_numpy(np.asarray(pil).copy()).float().permute(2, 0, 1).unsqueeze(0) / 255
    return (values - torch.tensor(mean).reshape(1, 3, 1, 1)) / torch.tensor(std).reshape(1, 3, 1, 1)


class SVGTokenizer:
    """The GGUF-embedded tokenizer.json, through the `tokenizers` library."""

    def __init__(self, tokenizer, eos_token_id):
        self.tokenizer, self.eos_token_id = tokenizer, eos_token_id

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


def load_local_tokenizer(embedded_assets, vocab_size):
    """Build the tokenizer from the embedded GGUF metadata."""
    if not isinstance(embedded_assets, dict) or "tokenizer.json" not in embedded_assets:
        raise ValueError("Invalid StarVector GGUF: embedded tokenizer.json is missing")
    try:
        tokenizer = Tokenizer.from_str(embedded_assets["tokenizer.json"])
        eos = json.loads(embedded_assets.get("tokenizer_config.json", "{}")).get("eos_token")
    # tokenizers raises plain Exception for malformed tokenizer.json.
    except Exception as exc:
        raise ValueError("Invalid StarVector GGUF: embedded tokenizer cannot be loaded") from exc
    eos_token_id = tokenizer.token_to_id(eos["content"] if isinstance(eos, dict) else str(eos))
    if eos_token_id is None:
        raise ValueError(f"Invalid StarVector GGUF: embedded tokenizer has no EOS token {eos!r}")
    if tokenizer.get_vocab_size(with_added_tokens=True) != vocab_size:
        raise ValueError(
            f"Invalid StarVector GGUF: embedded tokenizer vocabulary has "
            f"{tokenizer.get_vocab_size(with_added_tokens=True)} tokens, decoder requires {vocab_size}"
        )
    return SVGTokenizer(tokenizer, eos_token_id)


# Transformers' sampling settings of the StarVector quickstart.
TOP_K = 50
REPETITION_PENALTY = 3.1
LENGTH_PENALTY = -1.0
# Loop guard: a trailing window of tokens that already occurred this many times.
LOOP_WINDOW = 64
LOOP_REPEATS = 3


def _repeating(tokens):
    """True when the last LOOP_WINDOW tokens occur LOOP_REPEATS times."""
    if len(tokens) < LOOP_WINDOW * LOOP_REPEATS:
        return False
    tail = tokens[-LOOP_WINDOW:]
    count = 0
    for start in range(len(tokens) - LOOP_WINDOW + 1):
        if tokens[start] == tail[0] and tokens[start:start + LOOP_WINDOW] == tail:
            count += 1
            if count == LOOP_REPEATS:
                return True
    return False


def _penalize(scores, seen):
    """Transformers' RepetitionPenaltyLogitsProcessor over the generated tokens."""
    penalized = torch.where(scores < 0, scores * REPETITION_PENALTY, scores / REPETITION_PENALTY)
    return torch.where(seen, penalized, scores)


def _warp(scores, temperature, top_p, keep):
    """Transformers' temperature, top-k and top-p warpers, in that order."""
    if temperature != 1.0:
        scores = scores / temperature
    threshold = torch.topk(scores, max(TOP_K, keep))[0][..., -1:]
    scores = scores.masked_fill(scores < threshold, float('-inf'))
    if top_p < 1.0:
        ordered, order = torch.sort(scores, descending=False)
        remove = ordered.softmax(-1).cumsum(-1) <= 1 - top_p
        remove[..., -keep:] = False
        scores = scores.masked_fill(remove.scatter(-1, order, remove), float('-inf'))
    return scores


def _ends(tokens, stop_ids, eos_id):
    return tokens[-1] == eos_id or tokens[-len(stop_ids):] == stop_ids


def _sample(decoder, embeddings, max_new_tokens, temperature, top_p, stop_ids, eos_id, loop_guard, on_step):
    """Transformers' multinomial sampling of one sequence."""
    decoding, logits = decoder.prefill(embeddings, 1, embeddings.shape[1] + max_new_tokens)
    seen = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    tokens = []
    for step in range(max_new_tokens):
        scores = _warp(_penalize(logits.float(), seen), temperature, top_p, 1)
        token = torch.multinomial(scores.softmax(-1), 1)[:, 0]
        seen[token] = True
        tokens.append(int(token))
        on_step(step + 1)
        if _ends(tokens, stop_ids, eos_id):
            return tokens, 'stop'
        if loop_guard and step % 16 == 15 and _repeating(tokens):
            return tokens, 'repetition'
        if step + 1 < max_new_tokens:
            logits = decoding.step(token)
    return tokens, 'token_limit'


def _beam_sample(decoder, embeddings, max_new_tokens, temperature, top_p, beams, stop_ids, eos_id, loop_guard, on_step):
    """Transformers' beam sampling with early_stopping=True for one prompt.

    Each step samples 2 * beams continuations without replacement from all
    beams. Only the first `beams` samples may finish; the best unfinished
    samples continue. Finished scores use the length penalty.
    """
    keep = 2 * beams
    decoding, logits = decoder.prefill(embeddings, beams, embeddings.shape[1] + max_new_tokens)
    device, vocab = logits.device, logits.shape[-1]
    running = [[] for _ in range(beams)]
    running_scores = [0.] + [-1e9] * (beams - 1)
    seen = torch.zeros((beams, vocab), dtype=torch.bool, device=device)
    finished = []
    improvable = True
    for step in range(max_new_tokens):
        log_probs = _warp(_penalize(torch.log_softmax(logits.float(), -1), seen), temperature, top_p, 2)
        scores = (log_probs + torch.tensor(running_scores, device=device)[:, None]).view(1, -1)
        probs = scores.softmax(-1)
        # Transformers would fail when fewer than `keep` continuations remain
        # possible (a dead or far behind beam): sample only the possible ones.
        picks = torch.multinomial(probs, min(keep, int(torch.count_nonzero(probs))))
        picked = scores.gather(1, picks)[0].tolist()
        parents, tokens = zip(*(divmod(pick, vocab) for pick in picks[0].tolist()))
        candidates = [running[parent] + [token] for parent, token in zip(parents, tokens)]
        last = step + 1 == max_new_tokens
        hits = [last or _ends(candidate, stop_ids, eos_id) for candidate in candidates]
        if improvable and len(finished) < beams:
            finished += [(picked[j] / (step + 1) ** LENGTH_PENALTY, candidates[j]) for j in range(min(beams, len(picked))) if hits[j]]
            finished = sorted(finished, key=lambda item: item[0], reverse=True)[:beams]
        live = [score - 1e9 * hit for score, hit in zip(picked, hits)]
        order = sorted(range(len(picked)), key=live.__getitem__, reverse=True)[:beams]
        running_scores = [live[j] for j in order] + [-1e9] * (beams - len(order))
        order += [order[0]] * (beams - len(order))
        running = [candidates[j] for j in order]
        on_step(step + 1)
        worst = min(score for score, _ in finished) if len(finished) == beams else -1e9
        improvable = improvable and running_scores[0] / (step + 1) ** LENGTH_PENALTY > worst
        if not improvable or len(finished) == beams or all(hits):
            break
        if loop_guard and step % 16 == 15 and _repeating(running[0]):
            return (finished[0][1] if finished else running[0]), 'repetition'
        selected = torch.tensor([parents[j] for j in order], device=device)
        next_tokens = torch.tensor([tokens[j] for j in order], device=device)
        seen = seen[selected]
        seen[torch.arange(beams, device=device), next_tokens] = True
        decoding.reorder([parents[j] for j in order])
        logits = decoding.step(next_tokens)
    if not finished:
        return running[0], 'token_limit'
    best = finished[0][1]
    return best, ('token_limit' if len(best) == max_new_tokens and not _ends(best, stop_ids, eos_id) else 'stop')


def generate_tokens(decoder, tokenizer, prefix, max_new_tokens=4096,
                    temperature=1.5, top_p=0.9, seed=0, num_beams=2,
                    loop_guard=True, progress_callback=None):
    """Generate SVG with StarVector's original sampling, natively on `decoder`.

    `decoder` provides `context`, `embed(ids)` and `prefill(embeddings, rows,
    capacity)`, which returns a decoding object with `step(tokens)` and
    `reorder(parents)` plus the first logits. One sequence is returned.
    """
    if prefix.ndim != 3 or prefix.shape[0] != 1:
        raise ValueError("Expected one image embedding sequence [1,T,H]")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if temperature <= 0:
        raise ValueError("Sampling requires temperature > 0")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not isinstance(num_beams, int) or not 1 <= num_beams <= 8:
        raise ValueError("num_beams must be an integer between 1 and 8")

    device = prefix.device
    prompt_ids = tokenizer.encode("<svg")
    stop_ids = tokenizer.encode("</svg>")
    if not prompt_ids or not stop_ids:
        raise ValueError("Tokenizer cannot encode SVG control tokens")

    context = decoder.context
    prefix_length = prefix.shape[1] + len(prompt_ids)
    available = context - prefix_length
    if available < 1:
        raise ValueError(
            f"Generation exceeds context size {context}: "
            f"prefix={prefix.shape[1]}, prompt={len(prompt_ids)}"
        )
    # max_new_tokens is a ceiling; an architecture's context limit is absolute.
    requested_new_tokens = int(max_new_tokens)
    max_new_tokens = min(requested_new_tokens, available)
    logger = logging.getLogger(__name__)
    logger.info("StarVector: context=%d, image prefix=%d, SVG prompt=%d, requested max_new_tokens=%d, effective max_new_tokens=%d, top_k=%d, repetition_penalty=%s, loop_guard=%s", context, prefix.shape[1], len(prompt_ids), requested_new_tokens, max_new_tokens, TOP_K, REPETITION_PENALTY, loop_guard)
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    embeddings = torch.cat((prefix, decoder.embed(prompt).to(prefix.dtype)), dim=1)
    on_step = progress_callback if progress_callback is not None else (lambda step: None)
    # fork_rng preserves the caller's random state across invocations.
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == "cuda" else []
    started = perf_counter()
    with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        if num_beams == 1:
            generated, reason = _sample(decoder, embeddings, max_new_tokens, float(temperature), float(top_p), stop_ids, tokenizer.eos_token_id, loop_guard, on_step)
        else:
            generated, reason = _beam_sample(decoder, embeddings, max_new_tokens, float(temperature), float(top_p), num_beams, stop_ids, tokenizer.eos_token_id, loop_guard, on_step)

    # The image prefix has no text tokens: restore the textual '<svg' prompt.
    svg = tokenizer.decode(prompt_ids + generated)
    if not svg.startswith("<svg"):
        raise RuntimeError("Generated text does not start with the SVG prompt")
    elapsed = perf_counter() - started
    has_terminator = svg.rstrip().endswith("</svg>")
    logger.info(
        "StarVector: decoded %d/%d tokens in %.2fs (%.2f tokens/s); "
        "beams=%d, stop=%s, SVG terminator at end=%s",
        len(generated), max_new_tokens, elapsed,
        len(generated) / max(elapsed, 1e-9),
        num_beams, reason, has_terminator,
    )
    if reason == "repetition":
        logger.warning("StarVector: stopped after %d tokens because the model kept repeating the same output (loop guard)", len(generated))
    elif not has_terminator:
        logger.warning(
            "StarVector: decoder stopped without a final </svg> (reason=%s, "
            "generated=%d/%d tokens); output tail=%r",
            reason, len(generated), max_new_tokens, svg[-160:],
        )
    return svg


class CachedDecoding:
    """KV cache and CUDA-graph decode steps of one generation call.

    `keys`/`values` hold the decode rows at dim 1 and cache slots at dim -2.
    The model's `decode(tokens, position, keys, values, span)` computes one
    step; the cache is zero-initialized because decode attends whole spans
    with masked tails, and stale NaN/Inf bits would survive the mask.
    """

    def __init__(self, model, keys, values, length):
        self.model, self.keys, self.values, self.length = model, keys, values, length
        self.tokens = torch.zeros(keys.shape[1], dtype=torch.long, device=keys.device)
        self.position = torch.full((1,), length, dtype=torch.long, device=keys.device)
        self.graphs = {}

    def _decode(self, span):
        return self.model.decode(self.tokens, self.position, self.keys, self.values, span)

    def step(self, tokens):
        # Attend a power-of-two prefix of the cache: one CUDA graph per span.
        span = min(self.keys.shape[-2], max(512, 1 << self.length.bit_length()))
        self.tokens.copy_(tokens)
        if self.keys.device.type != 'cuda':
            logits = self._decode(span)
        else:
            if span not in self.graphs:
                buffers = []
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), graph_io(buffers):
                    self._decode(span)  # compiles kernels; rewrites only the slot of this step
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode='thread_local'), graph_io(buffers):
                    output = self._decode(span)
                # The graph addresses `buffers` directly: keep them with it.
                self.graphs[span] = graph, output, buffers
            graph, logits, _ = self.graphs[span]
            graph.replay()
        self.position += 1
        self.length += 1
        return logits

    def reorder(self, parents):
        """Copy the cache of each row's parent beam into rows that changed parent."""
        rows = [row for row, parent in enumerate(parents) if parent != row]
        if rows:
            sources = [parents[row] for row in rows]
            self.keys[:, rows, ..., :self.length, :] = self.keys[:, sources, ..., :self.length, :]
            self.values[:, rows, ..., :self.length, :] = self.values[:, sources, ..., :self.length, :]


class StarVectorRuntimeBase(nn.Module):
    """Construction and inference steps identical for the 1B and 8B runtimes.

    `reference` provides tensor_index, model_config and model_assets, like
    gguf_support.GGUFReference. Subclasses build `decoder`, `vision` and
    `adapter` in `_build(store, config)` and convert ComfyUI images in
    `_preprocess_image`.
    """

    def __init__(self, reference, dtype):
        super().__init__()
        self.dtype = dtype
        self._build(TensorStore(reference.tensor_index), reference.model_config)
        self.tokenizer = load_local_tokenizer(reference.model_assets, self.decoder.lm_head.out_features)
        for module in self.modules():
            if isinstance(module, GGUFEmbedding):
                module.output_dtype = dtype
        self.to(dtype=dtype)
        self.eval()

    @torch.inference_mode()
    def prepare_image_prefix(self, image):
        device = next(self.vision.parameters()).device
        pixels = self._preprocess_image(image).to(device=device, dtype=self.dtype)
        return self.adapter(self.vision(pixels))

    @torch.inference_mode()
    def generate_from_prefix(self, prefix, max_new_tokens=4096, temperature=1.5, top_p=0.9, seed=0, num_beams=2, loop_guard=True, progress_callback=None):
        device = next(self.decoder.parameters()).device
        return generate_tokens(self.decoder, self.tokenizer, prefix.to(device=device, dtype=self.dtype), max_new_tokens, temperature, top_p, seed, num_beams=num_beams, loop_guard=loop_guard, progress_callback=progress_callback)
