#!/usr/bin/env python3
"""Convert a StarVector Safetensors directory into a self-contained quantized GGUF.

Reads StarVector's starvector-{1b,8b}-im2svg directories and writes their
tensors under the same keys. Each tensor's role and GGUF block layout select its
quantization; normalization parameters, biases and unquantized tensors keep
their F16/F32/BF16 type. The GGUF embeds the architecture, tokenizer and image
normalization that ComfyUI-StarVector-GGUF reads, plus the SHA-256 and byte
length of each input file.

Runs on the CPU with gguf, safetensors, numpy, torch and tokenizers. The metadata
follows ComfyUI-StarVector-GGUF's convention.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, processors

# Fields referenced by the StarVector 1B/8B construction paths. Keep
# exactly these source config keys when present; the runtime applies defaults
# to the remaining optional settings.
CONFIG_KEYS = {
    "1b": (
        "image_encoder_type", "image_size", "hidden_size",
        "num_hidden_layers", "num_attention_heads", "starcoder_model_name",
        "max_position_embeddings", "multi_query", "n_inner",
        "activation_function",
    ),
    "8b": (
        "image_encoder_type", "image_size", "hidden_size",
        "num_hidden_layers", "num_attention_heads", "num_kv_heads",
        "intermediate_size", "max_position_embeddings", "max_length_train",
        "rope_theta", "rms_norm_eps", "hidden_act",
    ),
}
# 1B embeds its tokenizer.json and these small metadata files as they are. 8B
# builds one tokenizer.json from its GPT-2 BPE files, plus a tokenizer_config.json
# holding its special tokens.
TOKENIZER_SUPPORT_FILES_1B = (
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"
)
MAX_TEXT_BYTES = 16 * 1024 * 1024
PRESERVED_DTYPES = {"F16", "F32", "BF16"}
BYTES_PER_DTYPE = {"F16": 2, "F32": 4, "BF16": 2}
# GGML block sizes and packed bytes, per ggml-org/llama.cpp's GGML_QUANT_SIZES.
QUANTIZATION_SPECS = {
    "q4_0": ("Q4_0", 32, 18),
    "q5_0": ("Q5_0", 32, 22),
    "q8_0": ("Q8_0", 32, 34),
}
QUANTIZED_TYPES = frozenset(spec[0] for spec in QUANTIZATION_SPECS.values())


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shard: Path
    shape: tuple[int, ...]
    source_dtype: str
    target_dtype: str
    byte_count: int


def find_shards(folder: Path) -> tuple[dict[str, Path], list[Path]]:
    """Use the official model index if present; reject stray/unmapped tensors."""
    index_path = folder / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid Safetensors weight_map in {index_path}")
        mapping = {}
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                raise ValueError("Non-string tensor name or shard in weight_map")
            if Path(filename).name != filename or not filename.endswith(".safetensors"):
                raise ValueError(f"Unsafe/non-Safetensors shard filename: {filename}")
            path = folder / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            mapping[name] = path
        return mapping, sorted(set(mapping.values()))

    shards = sorted(folder.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No *.safetensors files in {folder}")
    mapping = {}
    for path in shards:
        with safe_open(path, framework="pt", device="cpu") as sf:
            for name in sf.keys():
                if name in mapping:
                    raise ValueError(f"Duplicate tensor {name} in {path} and {mapping[name]}")
                mapping[name] = path
    return mapping, shards


def tensor_role(name: str, shape: tuple[int, ...]) -> str:
    """Identify weights whose GGUF layout is directly usable by the runtime.

    Normalization includes the StarVector-8B adapter's two-dimensional norm
    parameters. Conv2d's GGUF tensor must keep its original four-dimensional
    shape for the current ComfyUI runtime; this requires an aligned kernel
    width, irrespective of whether the flattened kernel would be aligned.
    """
    leaf = name.rsplit(".", 1)[-1].lower()
    parent = name.rsplit(".", 1)[0].rsplit(".", 1)[-1].lower()

    if (leaf.endswith("bias") or leaf in {
        "running_mean", "running_var", "num_batches_tracked",
        "inv_freq", "position_ids", "attention_mask", "causal_mask",
    }):
        return "parameter"

    # LayerNorm, RMSNorm, BatchNorm, and StarVector's projection adapter.
    # The adapter has norm.weight shaped [tokens, hidden_size], so rank
    # alone would incorrectly classify it as a quantizable matrix.
    if ("norm" in parent or parent == "ln" or parent.startswith("ln_")
            or parent == "bn" or parent.startswith("bn_")
            or (parent.startswith("bn") and parent[2:].isdigit())):
        return "normalization"

    if len(shape) == 4 and (leaf.endswith("weight") or not name):
        return "convolution"

    if len(shape) == 2 and (leaf.endswith("weight") or leaf in {
        "positional_embedding", "position_embedding", "position_embeddings",
        "token_embedding", "token_embeddings", "embedding", "embeddings",
    } or not name):
        return "matrix"

    return "parameter"


def tensor_target(shape: tuple[int, ...], source_dtype: str,
                  quantization: str = "q8_0", *, name: str = "") -> str:
    """Choose the requested quantization for eligible weight layouts."""
    target, block_size, _ = QUANTIZATION_SPECS[quantization]
    if source_dtype not in PRESERVED_DTYPES:
        raise ValueError(f"Unsupported tensor dtype {source_dtype} (shape {shape})")

    role = tensor_role(name, shape)
    if role in ("matrix", "convolution") and shape[-1] % block_size == 0:
        # Below 8 bits, the vision tower/adapter and the token/position
        # embeddings (the token embedding is also the tied output head) stay
        # Q8_0: Q4_0 there distorts the image prefix by 15-20%.
        if target != "Q8_0" and (name.startswith(("model.image_encoder.", "model.image_projection."))
                                 or name.endswith(("wte.weight", "wpe.weight", "embed_tokens.weight"))):
            return "Q8_0"
        return target
    return source_dtype


def inventory_model(folder: Path, quantization: str = "q8_0") -> tuple[list[TensorPlan], list[Path]]:
    mapping, shards = find_shards(folder)
    plans = []
    seen = set()
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as sf:
            for name in sorted(sf.keys()):
                if name not in mapping or mapping[name] != shard:
                    raise ValueError(f"Tensor {name} in {shard} not mapped by model index")
                if name in seen:
                    raise ValueError(f"Duplicate tensor {name}")
                seen.add(name)
                # PyTorch BatchNorm's training update counter has no inference
                # role and GGUF inference doesn't consume it.
                if name.endswith(".num_batches_tracked"):
                    continue
                sl = sf.get_slice(name)
                shape = tuple(sl.get_shape())
                dtype = sl.get_dtype()
                target = tensor_target(shape, dtype, quantization, name=name)
                count = prod(shape)
                if target in QUANTIZED_TYPES:
                    _, block_size, block_bytes = next(
                        spec for spec in QUANTIZATION_SPECS.values() if spec[0] == target)
                    byte_count = count // block_size * block_bytes
                else:
                    byte_count = count * BYTES_PER_DTYPE[target]
                plans.append(TensorPlan(name, shard, shape, dtype, target, byte_count))
    if seen != set(mapping):
        raise ValueError(f"Unmapped/missing tensor keys: {sorted(set(mapping) - seen)[:5]}")
    return plans, shards


def validate_family(plans: list[TensorPlan], family: str) -> None:
    names = {p.name for p in plans}
    required = {
        "model.image_encoder.": "vision encoder",
        "model.image_projection.": "projection adapter",
        "model.svg_transformer.transformer.": "SVG decoder",
    }
    for prefix, component in required.items():
        if not any(name.startswith(prefix) for name in names):
            raise ValueError(f"Missing {component} weights ({prefix}*)")
    expected = (
        "model.svg_transformer.transformer.transformer.wte.weight"
        if family == "1b" else
        "model.svg_transformer.transformer.model.embed_tokens.weight"
    )
    if expected not in names:
        raise ValueError(f"Selected --family {family} does not match the decoder tensors")


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _build_8b_tokenizer_assets(
    folder: Path, used_files: set[Path] | None = None
) -> dict[str, bytes]:
    """Build the 8B's tokenizer.json from its original GPT-2 BPE files.

    vocab.json and merges.txt become a byte-level BPE tokenizer, the added
    tokens in tokenizer_config.json keep their original IDs and flags, and the
    result is checked against vocab.json and added_tokens.json using only the
    serialized tokenizer.
    """
    for filename in ("vocab.json", "merges.txt", "tokenizer_config.json",
                     "special_tokens_map.json", "added_tokens.json"):
        path = folder / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        if used_files is not None:
            used_files.add(path)

    vocab = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
    config = json.loads((folder / "tokenizer_config.json").read_text(encoding="utf-8"))
    special_map = json.loads((folder / "special_tokens_map.json").read_text(encoding="utf-8"))
    original_added = json.loads((folder / "added_tokens.json").read_text(encoding="utf-8"))
    if not isinstance(original_added, dict):
        raise ValueError("Original added_tokens.json must be a token-to-ID mapping")

    merges = [tuple(line.split(" ")) for line in
              (folder / "merges.txt").read_text(encoding="utf-8").splitlines()[1:] if line]
    tokenizer = Tokenizer(models.BPE(vocab, merges, continuing_subword_prefix="", end_of_word_suffix=""))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=bool(config.get("add_prefix_space", False)))
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.TemplateProcessing(single="$A", pair="$A $B:1")
    for token_id, entry in sorted(config["added_tokens_decoder"].items(), key=lambda item: int(item[0])):
        token = AddedToken(
            entry["content"], lstrip=entry["lstrip"], rstrip=entry["rstrip"],
            normalized=entry["normalized"], single_word=entry["single_word"],
            special=entry["special"],
        )
        if entry["special"]:
            tokenizer.add_special_tokens([token])
        else:
            tokenizer.add_tokens([token])

    compact_config = {}
    for name in ("bos_token", "eos_token", "unk_token", "pad_token"):
        value = special_map[name]
        compact_config[name] = value["content"] if isinstance(value, dict) else value

    tokenizer_bytes = tokenizer.to_str().encode("utf-8")
    generated = Tokenizer.from_str(tokenizer_bytes.decode("utf-8"))
    for token, token_id in original_added.items():
        if (not isinstance(token, str) or not isinstance(token_id, int)
                or generated.token_to_id(token) != token_id):
            raise ValueError(f"Original added-token ID not preserved for {token!r}")
    for token_id, entry in config["added_tokens_decoder"].items():
        if generated.token_to_id(entry["content"]) != int(token_id):
            raise ValueError(f"Added-token ID not preserved for {entry['content']!r}")
    if generated.get_vocab_size(with_added_tokens=True) != len(
            set(vocab) | {entry["content"] for entry in config["added_tokens_decoder"].values()}):
        raise ValueError("Generated 8B tokenizer vocabulary size differs from vocab.json plus added tokens")
    for sample in ("<svg", "</svg>"):
        ids = generated.encode(sample, add_special_tokens=False).ids
        if generated.decode(ids, skip_special_tokens=True) != sample:
            raise ValueError(f"Generated 8B tokenizer does not round-trip {sample!r}")

    if len(tokenizer_bytes) > MAX_TEXT_BYTES:
        raise ValueError("Generated tokenizer.json exceeds the allowed asset size")
    return {
        "tokenizer.json": tokenizer_bytes,
        "tokenizer_config.json": _json_bytes(compact_config),
    }


def read_assets(
    folder: Path, family: str, used_files: set[Path] | None = None
) -> dict[str, bytes]:
    """Return only the model config, canonical tokenizer and required metadata.

    Both 1B and 8B outputs carry a tokenizer.json; neither includes a redundant
    copy of the BPE vocab/merges. 8B's tokenizer.json is built from the original
    distribution, with a verified unchanged token/ID mapping.
    """
    def get(name: str, required: bool = False) -> bytes | None:
        path = folder / name
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return None
        data = path.read_bytes()
        if len(data) > MAX_TEXT_BYTES:
            raise ValueError(f"Asset {path} exceeds {MAX_TEXT_BYTES} bytes")
        data.decode("utf-8")
        if used_files is not None:
            used_files.add(path)
        return data

    original_config = json.loads(get("config.json", required=True))
    if not isinstance(original_config, dict):
        raise ValueError("config.json must contain a JSON object")
    for key in ("image_encoder_type", "hidden_size", "num_hidden_layers",
                "num_attention_heads"):
        if key not in original_config:
            raise ValueError(f"Source config.json missing {key}")

    config = {key: original_config[key] for key in CONFIG_KEYS[family]
              if key in original_config}
    assets = {"config.json": _json_bytes(config)}

    if family == "1b":
        assets["tokenizer.json"] = get("tokenizer.json", required=True)
        for name in TOKENIZER_SUPPORT_FILES_1B:
            value = get(name)
            if value is not None:
                assets[name] = value
    else:
        assets.update(_build_8b_tokenizer_assets(folder, used_files))
        # 8B's image preprocessing is the only extra runtime asset, and is
        # reduced to the normalization parameters differing from defaults.
        merged = {}
        for filename in ("preprocessor_config.json", "processor_config.json"):
            raw = get(filename)
            if raw is not None:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError(f"Invalid {filename}: expected JSON object")
                merged.update(data)
        normalized = {key: merged[key] for key in ("image_mean", "image_std")
                      if key in merged and merged[key] != [0.5, 0.5, 0.5]}
        if normalized:
            assets["preprocessor_config.json"] = _json_bytes(normalized)
    return assets


def write_runtime_metadata(writer, assets: dict[str, bytes]) -> None:
    """Write the embedded StarVector configuration and inference assets."""
    writer.add_string("starvector.config_json", assets["config.json"].decode("utf-8"))
    for name, data in assets.items():
        if name != "config.json":
            writer.add_string(f"starvector.embedded_file.{name}", data.decode("utf-8"))


def source_file_manifest(folder: Path, files: set[Path]) -> dict[str, tuple[str, int]]:
    """SHA-256 and exact byte length of each original file read during export.

    Stream the Safetensors shards to keep memory usage independent of file size.
    Generated tokenizer data is represented by the original input files from
    which it was constructed.
    """
    manifest = {}
    for path in sorted(files):
        path = path.resolve(strict=True)
        if path.parent != folder:
            raise ValueError(f"Source file outside model directory: {path}")
        digest = sha256()
        byte_count = 0
        with path.open("rb") as source:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
        if path.name in manifest:
            raise ValueError(f"Duplicate source filename: {path.name}")
        manifest[path.name] = (digest.hexdigest(), byte_count)
    return manifest


def write_source_file_metadata(writer, manifest: dict[str, tuple[str, int]]) -> None:
    """Record the fingerprint and length of every conversion input in the GGUF."""
    for filename, (digest, byte_count) in sorted(manifest.items()):
        writer.add_string(f"starvector.file_sha256.{filename}", digest)
        writer.add_uint64(f"starvector.file_bytes.{filename}", byte_count)


def _shape_for_gguf(plan: TensorPlan) -> tuple[int, ...]:
    """The GGUF writer takes the packed *byte* shape for quantized tensors."""
    if plan.target_dtype in QUANTIZED_TYPES:
        _, block_size, block_bytes = next(
            spec for spec in QUANTIZATION_SPECS.values() if spec[0] == plan.target_dtype
        )
        return (*plan.shape[:-1], plan.shape[-1] // block_size * block_bytes)
    return plan.shape


def _convert_tensor(sf, plan: TensorPlan, gguf, chunk_mib: int) -> np.ndarray:
    if plan.target_dtype in QUANTIZED_TYPES:
        result = np.empty(_shape_for_gguf(plan), dtype=np.uint8)
        sl = sf.get_slice(plan.name)
        # Cap float32 chunk memory: the output buffer is the packed tensor.
        elements_per_outer_slice = prod(plan.shape[1:])
        step = max(1, chunk_mib * 1024 * 1024 // (elements_per_outer_slice * 4))
        for i in range(0, plan.shape[0], step):
            block = sl[i:i + step].to(dtype=torch.float32)
            data = np.ascontiguousarray(block.numpy())
            result[i:i + step] = gguf.quants.quantize(data, gguf.GGMLQuantizationType[plan.target_dtype])
        return result

    tensor = sf.get_tensor(plan.name)
    if plan.target_dtype == "BF16":
        # NumPy cannot represent bfloat16 natively. Reinterpret the original
        # 16-bit words as uint16, so every BF16 bit is preserved exactly.
        # GGUFWriter receives raw_dtype=BF16 along with this uint16 payload.
        return np.ascontiguousarray(tensor.view(torch.uint16).numpy())
    np_dtype = {"F16": np.float16, "F32": np.float32}[plan.target_dtype]
    return np.ascontiguousarray(tensor.to(dtype={"F16": torch.float16, "F32": torch.float32}[plan.target_dtype]).numpy(), dtype=np_dtype)


def check_quantizer(gguf, quantization: str) -> None:
    """Validate that the installed gguf library can encode this GGML type."""
    dtype_name, block_size, packed_bytes = QUANTIZATION_SPECS[quantization]
    try:
        dtype = gguf.GGMLQuantizationType[dtype_name]
        gguf_block = gguf.GGML_QUANT_SIZES[dtype]
        if tuple(gguf_block) != (block_size, packed_bytes):
            raise RuntimeError(
                f"GGUF {dtype_name} block geometry differs: {gguf_block}"
            )
        sample = np.linspace(-1, 1, num=block_size, dtype=np.float32).reshape(1, -1)
        encoded = gguf.quants.quantize(sample, dtype)
    except (AttributeError, KeyError, NotImplementedError) as exc:
        raise RuntimeError(f"Installed gguf package cannot quantize {dtype_name}. Install a gguf release supporting this quantizer.") from exc
    if encoded.dtype != np.uint8 or encoded.nbytes != packed_bytes:
        raise RuntimeError(
            f"Unexpected {dtype_name} quantizer output: "
            f"{encoded.dtype}, {encoded.nbytes} bytes (expected {packed_bytes})"
        )


def convert(folder: Path, output: Path, family: str,
            chunk_mib: int, dry_run: bool = False,
            quantization: str = "q8_0"):
    if quantization not in QUANTIZATION_SPECS:
        raise ValueError(f"Unknown quantization {quantization!r}")
    folder = folder.expanduser().resolve(strict=True)
    if not folder.is_dir():
        raise NotADirectoryError(folder)
    output = output.expanduser().resolve()
    if output.exists() and not dry_run:
        raise FileExistsError(f"Output exists; refusing to overwrite: {output}")
    used_files: set[Path] = set()
    assets = read_assets(folder, family, used_files)
    plans, shards = inventory_model(folder, quantization)
    validate_family(plans, family)
    used_files.update(shards)
    index = folder / "model.safetensors.index.json"
    if index.is_file():
        used_files.add(index)
    counts = Counter(p.target_dtype for p in plans)
    estimated = sum(p.byte_count for p in plans)
    print(f"StarVector-{family.upper()} {quantization.upper()} input: {folder}")
    print(f"Original Safetensors shards: {len(shards)}, tensors: {len(plans)}")
    print(f"Tensor format counts: {dict(counts)}")
    print(f"Approx. output payload: {estimated / 1024**3:.2f} GiB (+ GGUF metadata)")
    print("Embedded assets: " + ", ".join(assets))
    print(f"Embedded metadata payload: {sum(len(v) for v in assets.values()):,} bytes")
    print("Source files: " + ", ".join(sorted(path.name for path in used_files)))
    target_type = QUANTIZATION_SPECS[quantization][0]
    if counts[target_type] == 0:
        raise ValueError(f"No quantizable matrices found for {target_type}")
    # Also validate the installed quantizer during dry-run.
    import gguf
    check_quantizer(gguf, quantization)
    if dry_run:
        return

    print("Computing source SHA-256 digests and byte lengths...", flush=True)
    manifest = source_file_manifest(folder, used_files)
    print(f"Source file fingerprints: {len(manifest)}", flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output), arch="starvector")
    try:
        # GGUFWriter automatically writes general.architecture=starvector.
        # Tensor descriptors encode each quantized/F16/F32/BF16 type. HF model
        # relation and licensing belong to the model card.
        write_runtime_metadata(writer, assets)
        write_source_file_metadata(writer, manifest)

        for p in plans:
            target_type = gguf.GGMLQuantizationType[p.target_dtype]
            dtype = np.dtype(np.uint8 if p.target_dtype in QUANTIZED_TYPES else {
                "F16": np.float16, "F32": np.float32, "BF16": np.uint16,
            }[p.target_dtype])
            writer.add_tensor_info(p.name, _shape_for_gguf(p), dtype, p.byte_count,
                                   raw_dtype=target_type)

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_ti_data_to_file()
        position = 0
        for shard in shards:
            shard_plans = [p for p in plans if p.shard == shard]
            with safe_open(shard, framework="pt", device="cpu") as sf:
                for p in shard_plans:
                    position += 1
                    print(f"[{position}/{len(plans)}] {p.name}: {p.source_dtype} -> {p.target_dtype}", flush=True)
                    packed = _convert_tensor(sf, p, gguf, chunk_mib)
                    if packed.nbytes != p.byte_count:
                        raise RuntimeError(f"Unexpected packed size for {p.name}: "
                                           f"{packed.nbytes} != {p.byte_count}")
                    writer.write_tensor_data(packed)
                    del packed
        writer.close()
    except BaseException:
        writer.close()
        output.unlink(missing_ok=True)
        raise

    reader = gguf.GGUFReader(str(output))
    if len(reader.tensors) != len(plans):
        raise RuntimeError("Output tensor count differs from source inventory")
    if reader.get_field("starvector.config_json").contents() != assets["config.json"].decode("utf-8"):
        raise RuntimeError("Embedded config verification failed")
    for name, raw in assets.items():
        if name != "config.json":
            field = reader.get_field(f"starvector.embedded_file.{name}")
            if field is None or field.contents() != raw.decode("utf-8"):
                raise RuntimeError(f"Embedded asset verification failed: {name}")
    for name, (digest, byte_count) in manifest.items():
        sha_field = reader.get_field(f"starvector.file_sha256.{name}")
        length_field = reader.get_field(f"starvector.file_bytes.{name}")
        if sha_field is None or sha_field.contents() != digest:
            raise RuntimeError(f"Source SHA-256 metadata mismatch: {name}")
        if length_field is None or length_field.contents() != byte_count:
            raise RuntimeError(f"Source byte-length metadata mismatch: {name}")
    for before, after in zip(plans, reader.tensors):
        if before.name != after.name or before.target_dtype != after.tensor_type.name:
            raise RuntimeError(f"Tensor index/type mismatch: {before.name} vs {after.name}")
    print(f"SUCCESS: verified {len(plans)} tensors, assets, and {len(manifest)} source files -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path,
                        help="Original StarVector Safetensors directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path of the new GGUF file")
    parser.add_argument("--family", required=True, choices=("1b", "8b"), help="Model family")
    parser.add_argument("--quantization", choices=tuple(QUANTIZATION_SPECS),
                        default="q8_0", help="Quantization of eligible matrices (default: q8_0)")
    parser.add_argument("--chunk-mib", type=int, default=32,
                        help="Largest float32 chunk, in MiB, held during quantization")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the inputs and report tensor types only")
    args = parser.parse_args()
    if args.chunk_mib < 1:
        parser.error("--chunk-mib must be >= 1")
    convert(args.model_dir, args.output, args.family, args.chunk_mib, args.dry_run,
            args.quantization)


if __name__ == "__main__":
    main()
