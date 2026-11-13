"""Handle incomplete StarVector output: repair it, or trace the input with VTracer.

Repair follows StarVector's reference implementation: Beautiful Soup rebuilds
the XML tree and CairoSVG's svg2svg() writes it back as normalized SVG. Only
malformed output goes through this lossy conversion; the ComfyUI caller saves
the unmodified markup.
"""
from __future__ import annotations

from io import BytesIO
from xml.etree import ElementTree

import cairosvg
import numpy as np
import vtracer
from bs4 import BeautifulSoup
from PIL import Image


EMPTY_SVG = "<svg></svg>"
# VTracer's defaults are corner_threshold=60, length_threshold=4, splice_threshold=45.
COMPACT_GEOMETRY = {"corner_threshold": 90, "length_threshold": 8, "splice_threshold": 60}


def _validate_svg(markup: str) -> None:
    root = ElementTree.fromstring(markup)
    if root.tag.rsplit("}", 1)[-1] != "svg":
        raise ValueError(f"Generated document has no SVG root: {root.tag}")


def is_complete_svg(markup: str) -> bool:
    """False for output cut by the loop guard or max_new_tokens, or otherwise malformed."""
    try:
        _validate_svg(markup)
    except (ElementTree.ParseError, ValueError):
        return False
    return True


def trace_svg(pixels, compact=False) -> str:
    """VTracer color tracing of one RGB/RGBA image [H,W,C] with values in [0, 1].

    Three decimals of the pixel coordinates are invisible and avoid float
    artifacts such as 343.03999999999996 in VTracer's default output.
    `compact` fits fewer, longer curves: about 20% smaller and slightly less
    accurate, which a following refine more than recovers.
    """
    data = (np.clip(pixels, 0, 1) * 255).round().astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(data).save(buffer, "PNG")
    geometry = COMPACT_GEOMETRY if compact else {}
    return vtracer.convert_raw_image_to_svg(buffer.getvalue(), img_format="png", path_precision=3, **geometry)


def recover_svg_markup(markup: str) -> tuple[str, str, str | None]:
    """Return (valid SVG, status, diagnostic).

    status is 'original', 'repaired', or 'empty'. A repair can change vector
    geometry or drop unfinished attributes. Both repaired and empty outcomes
    must be logged and their unmodified input retained by the caller.
    """
    try:
        _validate_svg(markup)
        return markup, "original", None
    except (ElementTree.ParseError, ValueError) as exc:
        original_error = str(exc)

    try:
        if "<!DOCTYPE" in markup.upper() or "<!ENTITY" in markup.upper():
            raise ValueError("DOCTYPE/ENTITY declarations are not supported")

        # StarVector's clean_svg pipeline: Beautiful Soup reconstructs the XML
        # tree, CairoSVG re-emits valid SVG vectors. unsafe=False blocks external
        # files, network resources and XML entities.
        reconstructed = BeautifulSoup(markup, "xml").prettify()
        converted = cairosvg.svg2svg(bytestring=reconstructed.encode("utf-8"), unsafe=False).decode("utf-8")
        # Match upstream clean_svg's removal of the optional XML declaration.
        converted = "\n".join(line for line in converted.split("\n") if not line.lstrip().startswith("<?xml"))
        _validate_svg(converted)
        return converted, "repaired", original_error
    except Exception as exc:
        return EMPTY_SVG, "empty", f"{original_error}; recovery failed: {type(exc).__name__}: {exc}"
