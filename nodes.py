"""GGUF StarVector 1B/8B ComfyUI nodes."""
from __future__ import annotations
import logging
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import comfy.model_management as mm
import comfy.model_patcher
import folder_paths
import torch
from comfy.utils import ProgressBar
from comfy_api.latest import io
from torch import nn

from .gguf_support import inspect_local_gguf
from .runtime_1b import StarVector1B
from .runtime_8b import StarVector8B
from .runtime_shared import DECODER_PREFIX
from .svg_recovery import is_complete_svg, recover_svg_markup, trace_svg
from .svg_refine import refine_svg

_LOG = logging.getLogger(__name__)
# extra_model_paths.yaml has already been loaded by ComfyUI when custom
# nodes are imported. Preserve its registered starvector paths and priority.
# Also allow the default ComfyUI models/starvector folder as a fallback.
MODEL_DIRECTORY = Path(folder_paths.models_dir) / "starvector"
folder_paths.add_model_folder_path("starvector", str(MODEL_DIRECTORY))
folder_paths.folder_names_and_paths["starvector"][1].add(".gguf")
# Most paths refine_paths adds; refine usually ends earlier, once the badly drawn regions are covered.
REFINE_PATH_LIMIT = 16
# The decoder token embedding tells the two architectures apart.
RUNTIMES = {
    DECODER_PREFIX + "transformer.wte.weight": StarVector1B,
    DECODER_PREFIX + "model.embed_tokens.weight": StarVector8B,
}


def _resolve_gguf(gguf_name):
    path = folder_paths.get_full_path("starvector", gguf_name)
    if path is None:
        raise FileNotFoundError(f"No GGUF file {gguf_name!r} in ComfyUI's registered starvector model paths: {folder_paths.get_folder_paths('starvector')}")
    path = Path(path).resolve(strict=True)
    roots = [Path(root).resolve() for root in folder_paths.get_folder_paths("starvector")]
    if not any(path.is_relative_to(root) for root in roots):
        raise ValueError("GGUF path outside registered model folder")
    return path


def _save_diagnostic(markup):
    """Keep the unmodified model output in ComfyUI's temp directory for diagnosis."""
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", prefix="starvector-invalid-", suffix=".svg.txt", dir=folder_paths.get_temp_directory(), delete=False) as stream:
            stream.write(markup)
            return stream.name
    except OSError as exc:
        _LOG.warning("StarVector: cannot save invalid SVG diagnostic: %s", exc)
        return None


def _svg_markup(svg):
    """Decode the native ComfyUI SVG documents with getvalue(), which keeps their stream positions."""
    if not isinstance(svg, io.SVG.Type):
        raise TypeError("Expected a ComfyUI native SVG object")
    if not svg.data:
        raise ValueError("SVG batch must contain at least one document")
    return [stream.getvalue().decode("utf-8") for stream in svg.data]


class _Stage(nn.Module):
    """Encoder or decoder as a separate ComfyUI residency stage, so the
    encoder can be evicted when the decoder needs VRAM."""

    def __init__(self, inner, dtype):
        super().__init__()
        self.inner = inner
        self.stage_dtype = dtype

    def get_dtype(self):
        return self.stage_dtype


class StarVectorGGUFLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gguf_name": (folder_paths.get_filename_list("starvector"),),
            "device": (["auto", "cpu"], {"default": "auto", "tooltip": "auto uses ComfyUI's device; cpu requires float32."}),
            "precision": (["auto", "float32", "float16", "bfloat16"], {"default": "auto", "tooltip": "auto selects float16 on GPU and float32 on CPU."}),
        }}

    RETURN_TYPES = ("STARVECTOR",)
    RETURN_NAMES = ("starvector",)
    FUNCTION = "load"
    CATEGORY = "StarVector/GGUF"

    def load(self, gguf_name, device, precision):
        started = perf_counter()
        ref = inspect_local_gguf(_resolve_gguf(gguf_name))
        runtime = next((runtime for name, runtime in RUNTIMES.items() if name in ref.tensor_index), None)
        if runtime is None:
            raise ValueError(f"{ref.path.name} is not a StarVector GPTBigCode/StarCoder2 GGUF")
        target = torch.device("cpu") if device == "cpu" else mm.get_torch_device()
        dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        dtype = (torch.float32 if target.type == "cpu" else torch.float16) if precision == "auto" else dtype_map[precision]
        if target.type == "cpu" and dtype != torch.float32:
            raise ValueError("CPU inference requires float32")
        instance = runtime(ref, dtype)
        vision = _Stage(nn.ModuleDict({"vision": instance.vision, "adapter": instance.adapter}), dtype)
        patchers = {
            name: comfy.model_patcher.ModelPatcher(stage, load_device=target, offload_device=torch.device("cpu"))
            for name, stage in (("vision_patcher", vision), ("decoder_patcher", _Stage(instance.decoder, dtype)))
        }
        _LOG.info("StarVector: loaded %s (%d tensors) as %s in %.2fs", ref.path.name, len(ref.tensor_index), runtime.__name__, perf_counter() - started)
        return ({"instance": instance, **patchers},)


class StarVectorGGUFImageToSVG:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "starvector": ("STARVECTOR",),
            "image": ("IMAGE",),
            "max_new_tokens": ("INT", {"default": 4096, "min": 1, "max": 16000, "tooltip": "Maximum new tokens; the model's remaining context also caps it."}),
            "temperature": ("FLOAT", {"default": 1.5, "min": 0.05, "max": 2., "step": 0.05}),
            "top_p": ("FLOAT", {"default": 0.9, "min": 0.05, "max": 1., "step": 0.05}),
            "num_beams": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1, "tooltip": "1 samples one sequence; 2 or more use beam sampling, with more memory per beam."}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "loop_guard": ("BOOLEAN", {"default": True, "tooltip": "Stop when the model repeats the same output."}),
            "on_failure": (["none", "recover", "vtracer"], {"default": "vtracer", "tooltip": "Action when StarVector's SVG is incomplete (loop guard, max_new_tokens) or malformed: none raises an error, recover repairs the partial markup, vtracer traces the input image."}),
            "refine_steps": ("INT", {"default": 100, "min": 0, "max": 10000, "tooltip": "Gradient-descent steps (LIVE-style) of each fit of the SVG to the input image: the initial fit, and one more after each added path. 0 turns refining off; 300 fits closer. Applies to complete StarVector SVGs and to VTracer's traces. Runs on the CUDA GPU, or on the CPU more slowly."}),
            "refine_paths": ("BOOLEAN", {"default": True, "tooltip": "Add paths, one at a time, where the SVG still differs most from the image, each followed by refine_steps steps. Up to 16 paths: about 10 on shaded illustrations, 0-1 on flat icons. Works with refine_steps above 0."}),
        }}

    RETURN_TYPES = ("SVG",)
    RETURN_NAMES = ("svg",)
    FUNCTION = "generate"
    CATEGORY = "StarVector/GGUF"

    def generate(self, starvector, image, max_new_tokens, temperature=1.5, top_p=0.9, seed=0, num_beams=2, loop_guard=True, on_failure="vtracer", refine_steps=100, refine_paths=True):
        instance = starvector["instance"]
        if image.ndim != 4 or image.shape[0] < 1 or image.shape[-1] not in (3, 4):
            raise ValueError("Expected a non-empty ComfyUI IMAGE batch [B,H,W,3/4]")

        # Progress advances with every encoded image, decoded token and refine
        # step, so ComfyUI sees a multi-minute run working. The bar holds one
        # unit per encoded image, then per image its tokens and refine steps.
        new_paths = REFINE_PATH_LIMIT if refine_paths else 0
        budget = max_new_tokens + refine_steps * (new_paths + 1)
        progress = ProgressBar(image.shape[0] * (1 + budget))
        progress.update_absolute(0)
        # Encode each image separately (both runtimes expect B=1). Keep only
        # the compact image prefixes in CPU RAM before switching GPU stages.
        vision_started = perf_counter()
        mm.load_models_gpu([starvector["vision_patcher"]], force_full_load=True)
        prefixes = []
        for i in range(image.shape[0]):
            prefixes.append(instance.prepare_image_prefix(image[i:i + 1]).cpu())
            progress.update_absolute(i + 1)

        _LOG.info("StarVector: encoded %d image(s) in %.2fs", image.shape[0], perf_counter() - vision_started)
        mm.load_models_gpu([starvector["decoder_patcher"]], force_full_load=True)
        documents = []
        for index, prefix in enumerate(prefixes):
            offset = len(prefixes) + index * budget
            generation_started = perf_counter()
            markup = instance.generate_from_prefix(
                prefix, max_new_tokens, temperature, top_p, seed, num_beams=num_beams,
                loop_guard=loop_guard, progress_callback=lambda step, offset=offset: (
                    progress.update_absolute(offset + min(step, max_new_tokens))
                ),
            )
            elapsed = perf_counter() - generation_started
            refinable = is_complete_svg(markup)
            if refinable:
                valid_markup = markup
                _LOG.info("StarVector: generated valid SVG %d/%d in %.2fs (%d characters)", index + 1, len(prefixes), elapsed, len(valid_markup))
            else:
                # The unmodified output is saved for diagnosis.
                diagnostic = _save_diagnostic(markup)
                if on_failure == "none":
                    raise RuntimeError(f"StarVector did not complete SVG {index + 1}/{len(prefixes)} after {elapsed:.2f}s (loop guard, max_new_tokens or malformed markup); original saved={diagnostic}. Set on_failure to recover or vtracer to return an SVG anyway.")
                if on_failure == "vtracer":
                    valid_markup = trace_svg(image[index].cpu().numpy(), compact=refine_steps > 0)
                    refinable = True
                    _LOG.warning("StarVector: SVG %d/%d incomplete after %.2fs; traced the input with VTracer instead; original saved=%s", index + 1, len(prefixes), elapsed, diagnostic)
                else:
                    valid_markup, recovery, diagnostic_reason = recover_svg_markup(markup)
                    if recovery == "repaired":
                        _LOG.warning("StarVector: recovered malformed SVG %d/%d after %.2fs using Beautiful Soup + CairoSVG svg2svg; original error=%s; original saved=%s", index + 1, len(prefixes), elapsed, diagnostic_reason, diagnostic)
                    else:
                        _LOG.error("StarVector: generated SVG %d/%d cannot be recovered after %.2fs; returning EMPTY SVG; reason=%s; original saved=%s", index + 1, len(prefixes), elapsed, diagnostic_reason, diagnostic)
            if refine_steps and refinable:
                # Refine steps fill their own part of the bar, even when the SVG ended early.
                progress.update_absolute(offset + max_new_tokens)
                refine_started = perf_counter()
                valid_markup = refine_svg(valid_markup, image[index].to(mm.get_torch_device()), refine_steps, new_paths, lambda: progress.update(1))
                _LOG.info("StarVector: refined SVG %d/%d in %.2fs", index + 1, len(prefixes), perf_counter() - refine_started)
            progress.update_absolute(offset + budget)
            documents.append(BytesIO(valid_markup.encode("utf-8")))

        # Native ComfyUI SVG batch, one document per input image, in order.
        return (io.SVG.Type(documents),)


class StarVectorGGUFTextPreview:
    """Show SVG markup read-only in the node; pass the same SVG socket through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"svg": ("SVG",)}}

    RETURN_TYPES = ("SVG",)
    RETURN_NAMES = ("svg",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "StarVector/GGUF"

    def preview(self, svg):
        return {"ui": {"svg": _svg_markup(svg)}, "result": (svg,)}


class StarVectorGGUFPreviewSVG:
    """Client-side SVG preview that passes the SVG socket through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "svg": ("SVG",),
            "background": (
                ["white", "black", "gray", "transparent"],
                {"default": "white", "tooltip": "Preview background; changing it updates the preview at once."},
            ),
        }}

    RETURN_TYPES = ("SVG",)
    RETURN_NAMES = ("svg",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "StarVector/GGUF"

    def preview(self, svg, background="white"):
        return {
            "ui": {
                "svg": _svg_markup(svg),
                "background": [background],
            },
            "result": (svg,),
        }


NODE_CLASS_MAPPINGS = {
    "StarVectorGGUFLoader": StarVectorGGUFLoader,
    "StarVectorGGUFImageToSVG": StarVectorGGUFImageToSVG,
    "StarVectorGGUFTextPreview": StarVectorGGUFTextPreview,
    "StarVectorGGUFPreviewSVG": StarVectorGGUFPreviewSVG,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StarVectorGGUFLoader": "Load StarVector GGUF",
    "StarVectorGGUFImageToSVG": "Image to SVG",
    "StarVectorGGUFTextPreview": "SVG Text Preview",
    "StarVectorGGUFPreviewSVG": "SVG Image Preview",
}
