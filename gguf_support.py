"""Local GGUF discovery and structural validation for StarVector."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
import json

REQUIRED_COMPONENTS = {
    "vision": "model.image_encoder.",
    "adapter": "model.image_projection.",
    "decoder": "model.svg_transformer.transformer.",
}
# The runtime reads the tokenizer and its EOS token; other embedded files are ignored.
EMBEDDED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
EMBEDDED_ASSET_PREFIX = "starvector.embedded_file."
MAX_EMBEDDED_ASSET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class GGUFReference:
    """A validated StarVector GGUF: its memory-mapped tensors by name, embedded config and tokenizer files."""

    path: Path
    tensor_index: dict = field(repr=False)
    model_config: dict = field(repr=False)
    model_assets: dict[str, str] = field(repr=False)


def inspect_tensor_index(tensors: Iterable) -> int:
    """Count GGUF tensors and check every StarVector component is present."""
    components = {key: 0 for key in REQUIRED_COMPONENTS}
    count = 0
    for tensor in tensors:
        count += 1
        for component, prefix in REQUIRED_COMPONENTS.items():
            if tensor.name.startswith(prefix):
                components[component] += 1
                break

    missing = [key for key, value in components.items() if not value]
    if missing:
        raise ValueError(
            "This is not a full StarVector GGUF pack: missing tensor groups "
            + ", ".join(missing)
            + ". A GGUF containing only an LLM or a diffusion UNet is unsupported."
        )
    return count


def read_embedded_config(reader) -> dict:
    """Parse the original model config stored inside StarVector's GGUF."""
    field = reader.get_field("starvector.config_json")
    if field is None:
        raise ValueError(
            "This StarVector GGUF has no starvector.config_json metadata. "
            "The model architecture cannot be constructed from this export."
        )
    raw = field.contents()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError("starvector.config_json must be a JSON string")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in GGUF starvector.config_json") from exc
    if not isinstance(config, dict):
        raise ValueError("GGUF starvector.config_json must contain a JSON object")
    return config


def read_embedded_assets(reader) -> dict[str, str]:
    """Read required self-contained tokenizer assets directly from GGUF metadata."""
    assets = {}
    for filename in EMBEDDED_TOKENIZER_FILES:
        field = reader.get_field(EMBEDDED_ASSET_PREFIX + filename)
        if field is None:
            continue
        contents = field.contents()
        if isinstance(contents, bytes):
            try:
                contents = contents.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"Invalid StarVector GGUF: embedded {filename} is not UTF-8"
                ) from exc
        if not isinstance(contents, str):
            raise ValueError(f"Invalid StarVector GGUF: embedded {filename} is not a string")
        if not contents or len(contents.encode("utf-8")) > MAX_EMBEDDED_ASSET_BYTES:
            raise ValueError(f"Invalid StarVector GGUF: embedded {filename} is empty or too large")
        assets[filename] = contents

    if "tokenizer.json" not in assets:
        raise ValueError(
            "Invalid StarVector GGUF: embedded tokenizer metadata is missing "
            "(expected starvector.embedded_file.tokenizer.json)"
        )
    return assets


def inspect_local_gguf(path: Path) -> GGUFReference:
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError("Expected a local .gguf file")
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        raise RuntimeError(
            "GGUF support requires the 'gguf' package installed in ComfyUI's Python environment."
        ) from exc

    # GGUFReader memory-maps the tensor data. Copy-on-write gives torch.from_numpy
    # a writable mapping whose writes stay in memory.
    reader = GGUFReader(str(path), "c")
    config = read_embedded_config(reader)
    assets = read_embedded_assets(reader)
    count = inspect_tensor_index(reader.tensors)
    tensor_index = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensor_index) != count:
        raise ValueError("Duplicate tensor names in GGUF checkpoint")
    return GGUFReference(path=path, tensor_index=tensor_index, model_config=config, model_assets=assets)
