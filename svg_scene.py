"""Capture the drawable geometry of an arbitrary SVG as device-space cubic Beziers.

CairoSVG already resolves CSS classes, inheritance, <use>, transforms, viewBox,
units and arcs. A surface subclass records what it asks cairo to fill, stroke and
clip, so the result is exactly what CairoSVG would draw.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cairocffi
import numpy as np
from cairosvg.parser import Tree
from cairosvg.surface import PNGSurface
from cairosvg.url import parse_url


@dataclass
class Shape:
    subpaths: list[np.ndarray]
    closed: list[bool]
    # Per subpath, per cubic segment: drawn as a straight line, its control
    # points at 1/3 and 2/3 of the way.
    lines: list[np.ndarray]
    fill: tuple[float, float, float, float] | None
    fill_rule: str
    stroke: tuple[float, float, float, float] | None
    stroke_width: float
    clip: list[list[np.ndarray]]
    # "miter" or "round"; bevel joins are kept as round.
    stroke_join: str = "miter"
    miter_limit: float = 4.0
    stroke_cap: str = "butt"


@dataclass
class Scene:
    width: int
    height: int
    shapes: list[Shape]
    unsupported: list[str] = field(default_factory=list)


def _device_path(context):
    """Current path of a cairocffi context as (cubic subpaths, closed flags,
    straight-segment flags) in device space."""
    context.save()
    context.identity_matrix()
    path = context.copy_path()
    context.restore()
    subpaths, closed, lines = [], [], []
    points, straight = [], []

    def finish(is_closed):
        if len(points) > 1:
            subpaths.append(np.array(points, np.float32))
            closed.append(is_closed)
            lines.append(np.array(straight, bool))

    for kind, coords in path:
        if kind == cairocffi.PATH_MOVE_TO:
            finish(False)
            points, straight = [coords], []
        elif kind == cairocffi.PATH_LINE_TO:
            a = np.array(points[-1])
            b = np.array(coords)
            points += [tuple(a + (b - a) / 3), tuple(a + (b - a) * 2 / 3), coords]
            straight.append(True)
        elif kind == cairocffi.PATH_CURVE_TO:
            points += [coords[0:2], coords[2:4], coords[4:6]]
            straight.append(False)
        else:
            finish(True)
            points, straight = [], []
    finish(False)
    return subpaths, closed, lines


class _Recorder:
    """Stands in for the cairocffi Context of a CairoSVG surface and records paint operations."""

    def __init__(self, context, scene):
        self.context = context
        self.scene = scene
        self.recording = False
        self.text_depth = 0
        self.clips = []
        self.saved_clips = []
        self.group_starts = []
        self.group_done = None
        self.pending = None

    def __getattr__(self, name):
        return getattr(self.context, name)

    def _unsupported(self, reason):
        if reason not in self.scene.unsupported:
            self.scene.unsupported.append(reason)

    def _paint(self, kind):
        """RGBA of the solid source, or None when it is transparent or not representable."""
        source = self.context.get_source()
        if isinstance(source, cairocffi.SolidPattern):
            rgba = source.get_rgba()
            return rgba if rgba[3] > 0 else None
        if isinstance(source, cairocffi.Gradient):
            self._unsupported(f"gradient {kind}")
        else:
            self._unsupported(f"pattern {kind}")
        return None

    def _opaque_source(self):
        source = self.context.get_source()
        return not isinstance(source, cairocffi.SolidPattern) or source.get_rgba()[3] > 0

    def _record_fill(self):
        if not self.recording or self.text_depth or not self._opaque_source():
            return
        subpaths, closed, lines = _device_path(self.context)
        if not subpaths:
            return
        rule = "evenodd" if self.context.get_fill_rule() == cairocffi.FILL_RULE_EVEN_ODD else "nonzero"
        shape = Shape(subpaths, closed, lines, self._paint("fill"), rule, None, 0.0, list(self.clips))
        self.scene.shapes.append(shape)
        self.pending = shape

    def fill(self):
        self._record_fill()
        self.pending = None
        self.context.fill()

    def fill_preserve(self):
        self._record_fill()
        self.context.fill_preserve()

    def _record_stroke(self):
        if not self.recording or self.text_depth or not self._opaque_source():
            return
        subpaths, closed, lines = _device_path(self.context)
        if not subpaths:
            return
        if self.context.get_dash()[0]:
            self._unsupported("dashed stroke")
        xx, yx, xy, yy, _, _ = self.context.get_matrix()
        width = self.context.get_line_width() * math.sqrt(abs(xx * yy - xy * yx))
        stroke = self._paint("stroke")
        shape = self.pending
        if shape is None or shape.closed != closed or len(shape.subpaths) != len(subpaths) or not all(map(np.array_equal, shape.subpaths, subpaths)):
            shape = Shape(subpaths, closed, lines, None, "nonzero", None, 0.0, list(self.clips))
            self.scene.shapes.append(shape)
        shape.stroke = stroke
        shape.stroke_width = width
        shape.stroke_join = "miter" if self.context.get_line_join() == cairocffi.LINE_JOIN_MITER else "round"
        shape.miter_limit = self.context.get_miter_limit()
        shape.stroke_cap = {cairocffi.LINE_CAP_ROUND: "round", cairocffi.LINE_CAP_SQUARE: "square"}.get(self.context.get_line_cap(), "butt")
        self.pending = None

    def stroke(self):
        self._record_stroke()
        self.context.stroke()

    def stroke_preserve(self):
        self._record_stroke()
        self.context.stroke_preserve()

    def _clip_region(self):
        if self.recording:
            subpaths, _, _ = _device_path(self.context)
            self.clips.append(subpaths)

    def clip(self):
        self._clip_region()
        self.context.clip()

    def clip_preserve(self):
        self._clip_region()
        self.context.clip_preserve()

    def save(self):
        self.saved_clips.append(list(self.clips))
        self.context.save()

    def restore(self):
        self.clips = self.saved_clips.pop()
        self.context.restore()

    def push_group(self):
        self.group_starts.append(len(self.scene.shapes))
        self.pending = None
        self.context.push_group()

    def pop_group_to_source(self):
        self.group_done = self.group_starts.pop()
        self.pending = None
        self.context.pop_group_to_source()

    def paint_with_alpha(self, alpha):
        if self.group_done is None:
            self._unsupported("image")
        else:
            for shape in self.scene.shapes[self.group_done:]:
                shape.fill = shape.fill and shape.fill[:3] + (shape.fill[3] * alpha,)
                shape.stroke = shape.stroke and shape.stroke[:3] + (shape.stroke[3] * alpha,)
            self.group_done = None
        self.context.paint_with_alpha(alpha)

    def mask_surface(self, *args):
        self.group_done = None
        self._unsupported("mask")
        self.context.mask_surface(*args)


class _CaptureSurface(PNGSurface):
    def __init__(self, *args, scene, **kwargs):
        self.scene = scene
        super().__init__(*args, **kwargs)

    @property
    def context(self):
        return self._recorder

    @context.setter
    def context(self, context):
        self._recorder = _Recorder(context, self.scene)

    def draw(self, node):
        self._recorder.recording = True
        if parse_url(node.get("filter")).fragment in self.filters:
            self._recorder._unsupported("filter")
        if parse_url(node.get("mask")).fragment in self.masks:
            self._recorder._unsupported("mask")
        text = node.tag in ("text", "tspan", "textPath")
        if text:
            self._recorder._unsupported("text")
            self._recorder.text_depth += 1
        super().draw(node)
        if text:
            self._recorder.text_depth -= 1


def capture_svg(markup: str, width: int, height: int) -> Scene:
    scene = Scene(width, height, [])
    tree = Tree(bytestring=markup.encode(), unsafe=False)
    _CaptureSurface(tree, None, 96, None, None, None, 1, width, height, scene=scene)
    return scene


def _number(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _path_data(subpaths, closed, lines) -> str:
    parts = []
    for points, is_closed, straight in zip(subpaths, closed, lines):
        parts.append("M" + " ".join(map(_number, points[0])))
        for i, line in zip(range(1, len(points), 3), straight):
            if line:
                parts.append("L" + " ".join(map(_number, points[i + 2])))
            else:
                parts.append("C" + " ".join(_number(v) for p in points[i:i + 3] for v in p))
        if is_closed:
            parts.append("Z")
    return " ".join(parts)


def _color(rgba) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in rgba[:3])


def scene_to_svg(scene: Scene) -> str:
    clip_ids = {}
    body = []
    for shape in scene.shapes:
        attrs = [f'd="{_path_data(shape.subpaths, shape.closed, shape.lines)}"']
        if shape.fill is None:
            attrs.append('fill="none"')
        else:
            attrs.append(f'fill="{_color(shape.fill)}"')
            if shape.fill[3] < 1:
                attrs.append(f'fill-opacity="{_number(shape.fill[3])}"')
        if shape.fill_rule == "evenodd":
            attrs.append('fill-rule="evenodd"')
        if shape.stroke is not None:
            attrs.append(f'stroke="{_color(shape.stroke)}"')
            if shape.stroke[3] < 1:
                attrs.append(f'stroke-opacity="{_number(shape.stroke[3])}"')
            attrs.append(f'stroke-width="{_number(shape.stroke_width)}"')
            if shape.stroke_join == "round":
                attrs.append('stroke-linejoin="round"')
            elif shape.miter_limit != 4:
                attrs.append(f'stroke-miterlimit="{_number(shape.miter_limit)}"')
            if shape.stroke_cap != "butt":
                attrs.append(f'stroke-linecap="{shape.stroke_cap}"')
        element = f'<path {" ".join(attrs)}/>'
        for region in shape.clip:
            data = _path_data(region, [True] * len(region), [np.zeros(len(subpath) // 3, bool) for subpath in region])
            index = clip_ids.setdefault(data, len(clip_ids))
            element = f'<g clip-path="url(#c{index})">{element}</g>'
        body.append(element)
    defs = "".join(f'<clipPath id="c{i}"><path d="{data}"/></clipPath>' for data, i in clip_ids.items())
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{scene.width}" height="{scene.height}" viewBox="0 0 {scene.width} {scene.height}">'
            f'<defs>{defs}</defs>{"".join(body)}</svg>')
