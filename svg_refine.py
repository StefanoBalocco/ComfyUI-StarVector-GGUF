"""Fit an SVG to a raster image by gradient descent, in the spirit of LIVE.

Every shape the SVG draws (captured through CairoSVG, see svg_scene) is
rasterized by a differentiable rasterizer: per pixel, the distance to each
paint layer's flattened outline and its winding number give an anti-aliased
coverage, and layers are composited source-over in paint order. Triton kernels
compute it on CUDA; elsewhere PyTorch computes the same values and gradients
from the pixels near each outline. Adam then moves the Bezier control points
and adjusts colors and stroke widths to match the image. As in LIVE (Ma et al.,
CVPR 2022), new paths can be added one at a time on top, where the error
remains largest, each followed by a joint optimization of all paths.
"""
from __future__ import annotations

import logging
import math
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
import vtracer
from PIL import Image
from scipy import ndimage, sparse

from .svg_scene import Scene, Shape, capture_svg, scene_to_svg

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

_LOG = logging.getLogger(__name__)
TILE = 16
# Outside a layer's padded bounding box: distance of a pixel that no outline reaches.
FAR = 1.0e4
# Flattening: samples per cubic segment, about one edge per 2 px of control polygon.
SAMPLE_COUNTS = (1, 2, 4, 8, 16)
FILL_NONZERO, FILL_EVENODD, STROKE = 0, 1, 2
POINT_LR, COLOR_LR, WIDTH_LR = 0.25, 0.01, 0.1
# Longest side of the raster the shapes are fitted on, in pixels; the refined
# SVG is written in these units.
RESOLUTION = 512
# A stroke corner whose tangent turns by less than about 1 degree is smooth;
# miter joins apply to sharper corners.
SMOOTH_COS = 0.9998
# New paths start from connected regions of one quantized target color
# (LEVELS per channel) whose summed RGB error exceeds SEED_ERROR.
SEED_ERROR = 0.1
LEVELS = 4
MIN_SEED_PIXELS = 16

if triton is not None:
    @triton.jit
    def _coverage_forward(edges, cuts, squares, widths, offsets, boxes, modes, cover, dist, index, E, H, W, tiles_x, FAR: tl.constexpr, TILE: tl.constexpr):
        """Box-filtered coverage of each pixel by one layer and the edge its
        gradient goes to. A fill (nonzero or even-odd winding) also keeps its
        signed distance to the outline (positive inside). A stroke is the union
        of the bands around its edges; near a butt or square open end, they are
        cut where the first (cuts[0]) or last (cuts[1]) edge of the line starts
        or ends, or half a width beyond for square caps."""
        tile = tl.program_id(0)
        layer = tl.program_id(1)
        i = tl.arange(0, TILE * TILE)
        tx = (tile % tiles_x) * TILE
        ty = (tile // tiles_x) * TILE
        x = tx + i % TILE
        y = ty + i // TILE
        valid = (x < W) & (y < H)
        out = layer * H * W + y * W + x
        px = x.to(tl.float32) + 0.5
        py = y.to(tl.float32) + 0.5
        mode = tl.load(modes + layer)
        half = tl.load(widths + layer) * 0.5
        best = tl.full((TILE * TILE,), 3.0e38, tl.float32)
        closest = tl.full((TILE * TILE,), -1, tl.int32)
        winding = tl.zeros((TILE * TILE,), tl.int32)
        covered = tl.zeros((TILE * TILE,), tl.float32)
        signed = tl.zeros((TILE * TILE,), tl.float32) + tl.where(mode == 2, FAR, -FAR)
        if ((tx + TILE >= tl.load(boxes + layer * 4)) & (tx <= tl.load(boxes + layer * 4 + 2))
                & (ty + TILE >= tl.load(boxes + layer * 4 + 1)) & (ty <= tl.load(boxes + layer * 4 + 3))):
            if mode == 2:
                extend = tl.where(tl.load(squares + layer) != 0, half, 0.0)
                for e in range(tl.load(offsets + layer), tl.load(offsets + layer + 1)):
                    ax = tl.load(edges + e * 4)
                    ay = tl.load(edges + e * 4 + 1)
                    dx = tl.load(edges + e * 4 + 2) - ax
                    dy = tl.load(edges + e * 4 + 3) - ay
                    rx = px - ax
                    ry = py - ay
                    along = (rx * dx + ry * dy) / tl.maximum(dx * dx + dy * dy, 1e-12)
                    t = tl.minimum(tl.maximum(along, 0.0), 1.0)
                    d2 = (rx - t * dx) * (rx - t * dx) + (ry - t * dy) * (ry - t * dy)
                    # The end edges of a cut line continue straight up to the cuts.
                    first = tl.load(cuts + e)
                    last = tl.load(cuts + E + e)
                    t = tl.where(first == e, tl.minimum(along, 1.0), t)
                    t = tl.where(last == e, tl.maximum(along, t), t)
                    d = tl.sqrt((rx - t * dx) * (rx - t * dx) + (ry - t * dy) * (ry - t * dy))
                    band = tl.maximum(tl.minimum(d + half, 0.5) - tl.maximum(d - half, -0.5), 0.0)
                    fx = tl.load(edges + tl.maximum(first, 0) * 4)
                    fy = tl.load(edges + tl.maximum(first, 0) * 4 + 1)
                    fdx = tl.load(edges + tl.maximum(first, 0) * 4 + 2) - fx
                    fdy = tl.load(edges + tl.maximum(first, 0) * 4 + 3) - fy
                    reach = ((px - fx) * fdx + (py - fy) * fdy) / tl.maximum(tl.sqrt(fdx * fdx + fdy * fdy), 1e-6) + extend
                    band *= tl.where(first >= 0, tl.minimum(tl.maximum(reach + 0.5, 0.0), 1.0), 1.0)
                    lx = tl.load(edges + tl.maximum(last, 0) * 4 + 2)
                    ly = tl.load(edges + tl.maximum(last, 0) * 4 + 3)
                    ldx = lx - tl.load(edges + tl.maximum(last, 0) * 4)
                    ldy = ly - tl.load(edges + tl.maximum(last, 0) * 4 + 1)
                    reach = ((lx - px) * ldx + (ly - py) * ldy) / tl.maximum(tl.sqrt(ldx * ldx + ldy * ldy), 1e-6) + extend
                    band *= tl.where(last >= 0, tl.minimum(tl.maximum(reach + 0.5, 0.0), 1.0), 1.0)
                    better = (band > covered) | ((band == covered) & (d2 < best))
                    covered = tl.where(better, band, covered)
                    best = tl.where(better, d2, best)
                    closest = tl.where(better, e, closest)
                signed = tl.sqrt(best)
            else:
                for e in range(tl.load(offsets + layer), tl.load(offsets + layer + 1)):
                    ax = tl.load(edges + e * 4)
                    ay = tl.load(edges + e * 4 + 1)
                    by = tl.load(edges + e * 4 + 3)
                    dx = tl.load(edges + e * 4 + 2) - ax
                    dy = by - ay
                    rx = px - ax
                    ry = py - ay
                    t = tl.minimum(tl.maximum((rx * dx + ry * dy) / tl.maximum(dx * dx + dy * dy, 1e-12), 0.0), 1.0)
                    qx = rx - t * dx
                    qy = ry - t * dy
                    d2 = qx * qx + qy * qy
                    closer = d2 < best
                    best = tl.where(closer, d2, best)
                    closest = tl.where(closer, e, closest)
                    cross = dx * ry - rx * dy
                    winding += ((ay <= py) & (by > py) & (cross > 0)).to(tl.int32)
                    winding -= ((ay > py) & (by <= py) & (cross < 0)).to(tl.int32)
                inside = tl.where(mode == 1, (winding & 1) != 0, winding != 0)
                d = tl.sqrt(best)
                signed = tl.where(inside, d, -d)
                covered = tl.minimum(tl.maximum(signed + 0.5, 0.0), 1.0)
        tl.store(cover + out, covered, mask=valid)
        tl.store(dist + out, signed, mask=valid)
        tl.store(index + out, closest, mask=valid)

    @triton.jit
    def _cut_gradient(edges, grad_edges, cut, start: tl.constexpr, g, px, py, m):
        """Gradient of a cut's reach (from the start or before the end of edge
        `cut`) into that edge's end points."""
        ax = tl.load(edges + cut * 4, mask=m, other=0.0)
        ay = tl.load(edges + cut * 4 + 1, mask=m, other=0.0)
        dx = tl.load(edges + cut * 4 + 2, mask=m, other=1.0) - ax
        dy = tl.load(edges + cut * 4 + 3, mask=m, other=1.0) - ay
        length = tl.maximum(tl.sqrt(dx * dx + dy * dy), 1e-6)
        nx = dx / length
        ny = dy / length
        # reach = (p - a).n at the start, |b - a| - (p - a).n at the end; w is
        # the part of p - a normal to the edge, over |b - a|.
        along = (px - ax) * nx + (py - ay) * ny
        wx = (px - ax - along * nx) / length
        wy = (py - ay - along * ny) / length
        if start:
            tl.atomic_add(grad_edges + cut * 4, g * (-nx - wx), mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 1, g * (-ny - wy), mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 2, g * wx, mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 3, g * wy, mask=m)
        else:
            tl.atomic_add(grad_edges + cut * 4, g * wx, mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 1, g * wy, mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 2, g * (nx - wx), mask=m)
            tl.atomic_add(grad_edges + cut * 4 + 3, g * (ny - wy), mask=m)

    @triton.jit
    def _coverage_backward(edges, cuts, squares, widths, modes, dist, index, grad_cover, grad_edges, grad_widths, E, H, W, tiles_x, TILE: tl.constexpr):
        tile = tl.program_id(0)
        layer = tl.program_id(1)
        i = tl.arange(0, TILE * TILE)
        x = (tile % tiles_x) * TILE + i % TILE
        y = (tile // tiles_x) * TILE + i // TILE
        valid = (x < W) & (y < H)
        out = layer * H * W + y * W + x
        e = tl.load(index + out, mask=valid, other=-1)
        g = tl.load(grad_cover + out, mask=valid, other=0.0)
        m = (e >= 0) & (g != 0)
        ax = tl.load(edges + e * 4, mask=m, other=0.0)
        ay = tl.load(edges + e * 4 + 1, mask=m, other=0.0)
        dx = tl.load(edges + e * 4 + 2, mask=m, other=1.0) - ax
        dy = tl.load(edges + e * 4 + 3, mask=m, other=1.0) - ay
        px = x.to(tl.float32) + 0.5
        py = y.to(tl.float32) + 0.5
        rx = px - ax
        ry = py - ay
        along = (rx * dx + ry * dy) / tl.maximum(dx * dx + dy * dy, 1e-12)
        t = tl.minimum(tl.maximum(along, 0.0), 1.0)
        # The box filter's derivative is 1 within half a pixel of an outline,
        # hence 0 at every pixel center when outlines lie on the pixel grid.
        # Back-propagate the 2 px tent filter's instead: same total along a
        # straight outline, never zero next to it.
        if tl.load(modes + layer) == 2:
            half = tl.load(widths + layer) * 0.5
            extend = tl.where(tl.load(squares + layer) != 0, half, 0.0)
            first = tl.load(cuts + e, mask=m, other=-1)
            last = tl.load(cuts + E + e, mask=m, other=-1)
            t = tl.where(first == e, tl.minimum(along, 1.0), t)
            t = tl.where(last == e, tl.maximum(along, t), t)
            d = tl.sqrt((rx - t * dx) * (rx - t * dx) + (ry - t * dy) * (ry - t * dy))
            band = tl.maximum(tl.minimum(d + half, 0.5) - tl.maximum(d - half, -0.5), 0.0)
            outer = tl.maximum(1 - tl.abs(half - d), 0.0)
            inner = tl.maximum(1 - tl.abs(half + d), 0.0)
            cut_first = m & (first >= 0)
            fx = tl.load(edges + first * 4, mask=cut_first, other=0.0)
            fy = tl.load(edges + first * 4 + 1, mask=cut_first, other=0.0)
            fdx = tl.load(edges + first * 4 + 2, mask=cut_first, other=1.0) - fx
            fdy = tl.load(edges + first * 4 + 3, mask=cut_first, other=0.0) - fy
            reach_start = ((px - fx) * fdx + (py - fy) * fdy) / tl.maximum(tl.sqrt(fdx * fdx + fdy * fdy), 1e-6) + extend
            cut_last = m & (last >= 0)
            lx = tl.load(edges + last * 4 + 2, mask=cut_last, other=1.0)
            ly = tl.load(edges + last * 4 + 3, mask=cut_last, other=0.0)
            ldx = lx - tl.load(edges + last * 4, mask=cut_last, other=0.0)
            ldy = ly - tl.load(edges + last * 4 + 1, mask=cut_last, other=0.0)
            reach_end = ((lx - px) * ldx + (ly - py) * ldy) / tl.maximum(tl.sqrt(ldx * ldx + ldy * ldy), 1e-6) + extend
            cut_start = tl.where(cut_first, tl.minimum(tl.maximum(reach_start + 0.5, 0.0), 1.0), 1.0)
            cut_end = tl.where(cut_last, tl.minimum(tl.maximum(reach_end + 0.5, 0.0), 1.0), 1.0)
            g_start = tl.where(cut_first, g * band * cut_end * tl.maximum(1 - tl.abs(reach_start), 0.0), 0.0)
            g_end = tl.where(cut_last, g * band * cut_start * tl.maximum(1 - tl.abs(reach_end), 0.0), 0.0)
            g_d = g * cut_start * cut_end * (inner - outer)
            g_half = g * cut_start * cut_end * (outer + inner) + tl.where(tl.load(squares + layer) != 0, g_start + g_end, 0.0)
            tl.atomic_add(grad_widths + layer, tl.sum(tl.where(m, 0.5 * g_half, 0.0), 0))
            _cut_gradient(edges, grad_edges, first, True, g_start, px, py, cut_first & (g_start != 0))
            _cut_gradient(edges, grad_edges, last, False, g_end, px, py, cut_last & (g_end != 0))
        else:
            signed = tl.load(dist + out, mask=m, other=0.0)
            g_d = tl.where(signed >= 0, g, -g) * tl.maximum(1 - tl.abs(signed), 0.0)
        qx = t * dx - rx
        qy = t * dy - ry
        scale = g_d / tl.maximum(tl.sqrt(qx * qx + qy * qy), 1e-6)
        tl.atomic_add(grad_edges + e * 4, scale * qx * (1 - t), mask=m)
        tl.atomic_add(grad_edges + e * 4 + 1, scale * qy * (1 - t), mask=m)
        tl.atomic_add(grad_edges + e * 4 + 2, scale * qx * t, mask=m)
        tl.atomic_add(grad_edges + e * 4 + 3, scale * qy * t, mask=m)

    @triton.jit
    def _kite_geometry(points, widths, corners, layers, limits, c, C, m):
        """Unit tangents arriving at (a) and leaving (b) each corner and their
        norms, the outer side of the turn, the half width signed toward it, the
        miter scale k of the tip and whether k varies (miter limit holds)."""
        in_head = tl.load(corners + C + c, mask=m, other=0) * 2
        in_tail = tl.load(corners + 2 * C + c, mask=m, other=0) * 2
        out_head = tl.load(corners + 3 * C + c, mask=m, other=0) * 2
        out_tail = tl.load(corners + 4 * C + c, mask=m, other=0) * 2
        ax = tl.load(points + in_head, mask=m, other=1.0) - tl.load(points + in_tail, mask=m, other=0.0)
        ay = tl.load(points + in_head + 1, mask=m, other=0.0) - tl.load(points + in_tail + 1, mask=m, other=0.0)
        bx = tl.load(points + out_head, mask=m, other=1.0) - tl.load(points + out_tail, mask=m, other=0.0)
        by = tl.load(points + out_head + 1, mask=m, other=0.0) - tl.load(points + out_tail + 1, mask=m, other=0.0)
        la = tl.maximum(tl.sqrt(ax * ax + ay * ay), 1e-3)
        lb = tl.maximum(tl.sqrt(bx * bx + by * by), 1e-3)
        ax, ay, bx, by = ax / la, ay / la, bx / lb, by / lb
        cross = ax * by - ay * bx
        cos = ax * bx + ay * by
        side = tl.where(cross > 0, -1.0, tl.where(cross < 0, 1.0, 0.0))
        half = side * 0.5 * tl.load(widths + tl.load(layers + c, mask=m, other=0), mask=m, other=0.0)
        miter = (1 + cos) * 0.5 >= tl.load(limits + c, mask=m, other=1.0)
        k = tl.where(miter, 1 / tl.maximum(1 + cos, 1e-6), 0.5)
        return ax, ay, bx, by, la, lb, side, half, k, miter & (1 + cos > 1e-6)

    @triton.jit
    def _kite_forward(points, widths, corners, layers, limits, ring, C, BLOCK: tl.constexpr):
        """Corner, outer offsets and miter tip (or bevel midpoint) of each miter join."""
        c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        ax, ay, bx, by, la, lb, _, half, k, _ = _kite_geometry(points, widths, corners, layers, limits, c, C, m)
        v = tl.load(corners + c, mask=m, other=0) * 2
        vx = tl.load(points + v, mask=m, other=0.0)
        vy = tl.load(points + v + 1, mask=m, other=0.0)
        # Left normals (-y, x) scaled by the signed half width.
        tl.store(ring + c * 8, vx, mask=m)
        tl.store(ring + c * 8 + 1, vy, mask=m)
        tl.store(ring + c * 8 + 2, vx - half * ay, mask=m)
        tl.store(ring + c * 8 + 3, vy + half * ax, mask=m)
        tl.store(ring + c * 8 + 4, vx - half * k * (ay + by), mask=m)
        tl.store(ring + c * 8 + 5, vy + half * k * (ax + bx), mask=m)
        tl.store(ring + c * 8 + 6, vx - half * by, mask=m)
        tl.store(ring + c * 8 + 7, vy + half * bx, mask=m)

    @triton.jit
    def _kite_backward(points, widths, corners, layers, limits, grad_ring, grad_points, grad_widths, C, BLOCK: tl.constexpr):
        c = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = c < C
        ax, ay, bx, by, la, lb, side, half, k, varying = _kite_geometry(points, widths, corners, layers, limits, c, C, m)
        g0x = tl.load(grad_ring + c * 8, mask=m, other=0.0)
        g0y = tl.load(grad_ring + c * 8 + 1, mask=m, other=0.0)
        g1x = tl.load(grad_ring + c * 8 + 2, mask=m, other=0.0)
        g1y = tl.load(grad_ring + c * 8 + 3, mask=m, other=0.0)
        g2x = tl.load(grad_ring + c * 8 + 4, mask=m, other=0.0)
        g2y = tl.load(grad_ring + c * 8 + 5, mask=m, other=0.0)
        g3x = tl.load(grad_ring + c * 8 + 6, mask=m, other=0.0)
        g3y = tl.load(grad_ring + c * 8 + 7, mask=m, other=0.0)
        # Normals na = (-ay, ax), nb = (-by, bx); first = v + half na,
        # second = v + half nb, tip = v + half k (na + nb).
        gnax = half * (g1x + k * g2x)
        gnay = half * (g1y + k * g2y)
        gnbx = half * (g3x + k * g2x)
        gnby = half * (g3y + k * g2y)
        tip = g2x * (-ay - by) + g2y * (ax + bx)
        g_half = g1x * -ay + g1y * ax + g3x * -by + g3y * bx + k * tip
        # k = 1 / (1 + a.b) within the miter limit.
        g_cos = tl.where(varying, -k * k * half * tip, 0.0)
        gax = gnay + g_cos * bx
        gay = -gnax + g_cos * by
        gbx = gnby + g_cos * ax
        gby = -gnbx + g_cos * ay
        # Through the normalization of a and b (linear below the 1e-3 floor).
        da = tl.where(la > 1e-3, ax * gax + ay * gay, 0.0)
        db = tl.where(lb > 1e-3, bx * gbx + by * gby, 0.0)
        gax, gay = (gax - ax * da) / la, (gay - ay * da) / la
        gbx, gby = (gbx - bx * db) / lb, (gby - by * db) / lb
        v = tl.load(corners + c, mask=m, other=0) * 2
        tl.atomic_add(grad_points + v, g0x + g1x + g2x + g3x, mask=m)
        tl.atomic_add(grad_points + v + 1, g0y + g1y + g2y + g3y, mask=m)
        in_head = tl.load(corners + C + c, mask=m, other=0) * 2
        in_tail = tl.load(corners + 2 * C + c, mask=m, other=0) * 2
        out_head = tl.load(corners + 3 * C + c, mask=m, other=0) * 2
        out_tail = tl.load(corners + 4 * C + c, mask=m, other=0) * 2
        tl.atomic_add(grad_points + in_head, gax, mask=m)
        tl.atomic_add(grad_points + in_head + 1, gay, mask=m)
        tl.atomic_add(grad_points + in_tail, -gax, mask=m)
        tl.atomic_add(grad_points + in_tail + 1, -gay, mask=m)
        tl.atomic_add(grad_points + out_head, gbx, mask=m)
        tl.atomic_add(grad_points + out_head + 1, gby, mask=m)
        tl.atomic_add(grad_points + out_tail, -gbx, mask=m)
        tl.atomic_add(grad_points + out_tail + 1, -gby, mask=m)
        tl.atomic_add(grad_widths + tl.load(layers + c, mask=m, other=0), 0.5 * side * g_half, mask=m)

    @triton.jit
    def _composite_forward(alpha, colors, out, N, HW, BLOCK: tl.constexpr):
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < HW
        r = tl.zeros((BLOCK,), tl.float32) + 1.0
        g = r
        b = r
        for n in range(N):
            a = tl.load(alpha + n * HW + p, mask=m, other=0.0)
            r += a * (tl.load(colors + n * 3) - r)
            g += a * (tl.load(colors + n * 3 + 1) - g)
            b += a * (tl.load(colors + n * 3 + 2) - b)
        tl.store(out + p, r, mask=m)
        tl.store(out + HW + p, g, mask=m)
        tl.store(out + 2 * HW + p, b, mask=m)

    @triton.jit
    def _composite_backward(alpha, colors, grad_out, weight, grad_alpha, N, HW, BLOCK: tl.constexpr):
        """Gradients of source-over compositing: `weight` first holds the transmittance
        of the layers above each layer, then the layer's visible alpha."""
        p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = p < HW
        gr = tl.load(grad_out + p, mask=m, other=0.0)
        gg = tl.load(grad_out + HW + p, mask=m, other=0.0)
        gb = tl.load(grad_out + 2 * HW + p, mask=m, other=0.0)
        above = tl.zeros((BLOCK,), tl.float32) + 1.0
        for k in range(N):
            n = N - 1 - k
            tl.store(weight + n * HW + p, above, mask=m)
            above *= 1 - tl.load(alpha + n * HW + p, mask=m, other=0.0)
        r = tl.zeros((BLOCK,), tl.float32) + 1.0
        g = r
        b = r
        for n in range(N):
            a = tl.load(alpha + n * HW + p, mask=m, other=0.0)
            above = tl.load(weight + n * HW + p, mask=m, other=0.0)
            cr = tl.load(colors + n * 3)
            cg = tl.load(colors + n * 3 + 1)
            cb = tl.load(colors + n * 3 + 2)
            tl.store(grad_alpha + n * HW + p, above * (gr * (cr - r) + gg * (cg - g) + gb * (cb - b)), mask=m)
            tl.store(weight + n * HW + p, a * above, mask=m)
            r += a * (cr - r)
            g += a * (cg - g)
            b += a * (cb - b)


class _LayerCoverage(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edges, cuts, squares, widths, offsets, boxes, modes, height, width):
        layers = modes.numel()
        cover = torch.empty((layers, height, width), device=edges.device)
        dist = torch.empty_like(cover)
        index = torch.empty((layers, height, width), device=edges.device, dtype=torch.int32)
        tiles_x = triton.cdiv(width, TILE)
        _coverage_forward[(tiles_x * triton.cdiv(height, TILE), layers)](edges, cuts, squares, widths, offsets, boxes, modes, cover, dist, index, len(edges), height, width, tiles_x, FAR=FAR, TILE=TILE)
        ctx.save_for_backward(edges, cuts, squares, widths, modes, dist, index)
        return cover

    @staticmethod
    def backward(ctx, grad_cover):
        edges, cuts, squares, widths, modes, dist, index = ctx.saved_tensors
        layers, height, width = dist.shape
        grad_edges = torch.zeros_like(edges)
        grad_widths = torch.zeros_like(widths)
        tiles_x = triton.cdiv(width, TILE)
        _coverage_backward[(tiles_x * triton.cdiv(height, TILE), layers)](edges, cuts, squares, widths, modes, dist, index, grad_cover.contiguous(), grad_edges, grad_widths, len(edges), height, width, tiles_x, TILE=TILE)
        return grad_edges, None, None, grad_widths, None, None, None, None, None


class _MiterKites(torch.autograd.Function):
    """[C,4,2] corner, outer offsets and miter tip of each miter join."""

    @staticmethod
    def forward(ctx, points, widths, corners, layers, limits):
        count = corners.shape[1]
        ring = torch.empty((count, 4, 2), device=points.device)
        _kite_forward[(triton.cdiv(count, 128),)](points, widths, corners, layers, limits, ring, count, BLOCK=128)
        ctx.save_for_backward(points, widths, corners, layers, limits)
        return ring

    @staticmethod
    def backward(ctx, grad_ring):
        points, widths, corners, layers, limits = ctx.saved_tensors
        count = corners.shape[1]
        grad_points = torch.zeros_like(points)
        grad_widths = torch.zeros_like(widths)
        _kite_backward[(triton.cdiv(count, 128),)](points, widths, corners, layers, limits, grad_ring.contiguous(), grad_points, grad_widths, count, BLOCK=128)
        return grad_points, grad_widths, None, None, None


class _Flatten(torch.autograd.Function):
    """Product with a constant sparse matrix, whose transpose is prebuilt for the backward."""

    @staticmethod
    def forward(ctx, points, matrix, transposed):
        ctx.transposed = transposed
        return matrix @ points

    @staticmethod
    def backward(ctx, grad):
        return ctx.transposed @ grad, None, None


class _Composite(torch.autograd.Function):
    """Source-over of solid-color layers with per-pixel alpha onto white, the
    background StarVector's preprocessing composites images on."""

    @staticmethod
    def forward(ctx, alpha, colors):
        layers, height, width = alpha.shape
        out = torch.empty((3, height, width), device=alpha.device)
        _composite_forward[(triton.cdiv(height * width, 256),)](alpha, colors, out, layers, height * width, BLOCK=256)
        ctx.save_for_backward(alpha, colors)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        alpha, colors = ctx.saved_tensors
        layers, height, width = alpha.shape
        weight = torch.empty_like(alpha)
        grad_alpha = torch.empty_like(alpha)
        grad_out = grad_out.contiguous()
        _composite_backward[(triton.cdiv(height * width, 256),)](alpha, colors, grad_out, weight, grad_alpha, layers, height * width, BLOCK=256)
        return grad_alpha, weight.view(layers, -1) @ grad_out.view(3, -1).T


def _near_pixels(edges, pad, height, width):
    """(edge, y, x) of every pixel whose center lies in an edge's bounding box grown by `pad`."""
    start = ((torch.minimum(edges[:, :2], edges[:, 2:]) - pad[:, None] - 0.5).ceil().long()).clamp(min=0)
    stop = torch.minimum((torch.maximum(edges[:, :2], edges[:, 2:]) + pad[:, None] - 0.5).floor().long() + 1, torch.tensor([width, height], device=edges.device))
    size = (stop - start).clamp(min=0)
    counts = size[:, 0] * size[:, 1]
    edge = torch.repeat_interleave(torch.arange(len(edges), device=edges.device), counts)
    local = torch.arange(len(edge), device=edges.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    return edge, start[edge, 1] + local // size[edge, 0], start[edge, 0] + local % size[edge, 0]


def _first_by_rank(key, *ranks):
    """Mask of the one pair per key with the lexicographically smallest ranks."""
    group = torch.unique(key, return_inverse=True)[1]
    keep = torch.ones_like(key, dtype=torch.bool)
    for rank in ranks:
        top = torch.finfo(rank.dtype).max if rank.is_floating_point() else torch.iinfo(rank.dtype).max
        rank = rank.masked_fill(~keep, top)
        keep &= rank == torch.full((len(key),), top, dtype=rank.dtype, device=key.device).scatter_reduce(0, group, rank, "amin")[group]
    return keep


def _winding(edges, edge_layer, layers, height, width):
    """Winding number of every pixel center per layer: each edge crossing a
    pixel row counts for the pixels left of the crossing, +1 upward, -1 downward."""
    ax, ay, bx, by = edges.unbind(1)
    first = (torch.minimum(ay, by) - 0.5).ceil().long().clamp(0, height)
    counts = ((torch.maximum(ay, by) - 0.5).ceil().long().clamp(0, height) - first).clamp(min=0)
    edge = torch.repeat_interleave(torch.arange(len(edges), device=edges.device), counts)
    row = first[edge] + torch.arange(len(edge), device=edges.device) - torch.repeat_interleave(counts.cumsum(0) - counts, counts)
    cross = ax[edge] + (row + 0.5 - ay[edge]) * (bx[edge] - ax[edge]) / (by[edge] - ay[edge])
    sign = torch.where(by[edge] > ay[edge], 1, -1).int()
    base = (edge_layer[edge] * height + row) * (width + 1)
    steps = torch.zeros(layers * height * (width + 1), dtype=torch.int32, device=edges.device)
    steps.index_add_(0, base, sign)
    steps.index_add_(0, base + (cross - 0.5).ceil().long().clamp(0, width), -sign)
    return steps.view(layers, height, width + 1).cumsum(2, dtype=torch.int32)[..., :width]


def _stroke_terms(edges, cuts, half, extend, e, px, py):
    """Per pixel and stroke edge, as in _coverage_forward: the band before the
    cuts at butt or square open ends, each cut's reach and factor."""
    ax, ay, bx, by = edges[e].unbind(1)
    dx, dy, rx, ry = bx - ax, by - ay, px - ax, py - ay
    along = (rx * dx + ry * dy) / (dx * dx + dy * dy).clamp(min=1e-12)
    t = along.clamp(0, 1)
    d2 = (rx - t * dx).square() + (ry - t * dy).square()
    first, last = cuts[0, e].long(), cuts[1, e].long()
    t = torch.where(first == e, along.clamp(max=1), t)
    t = torch.where(last == e, torch.maximum(along, t), t)
    d = ((rx - t * dx).square() + (ry - t * dy).square()).sqrt()
    band = ((d + half).clamp(max=0.5) - (d - half).clamp(min=-0.5)).clamp(min=0)
    fx, fy, fbx, fby = edges[first.clamp(min=0)].unbind(1)
    reach_start = ((px - fx) * (fbx - fx) + (py - fy) * (fby - fy)) / ((fbx - fx).square() + (fby - fy).square()).sqrt().clamp(min=1e-6) + extend
    lax, lay, lx, ly = edges[last.clamp(min=0)].unbind(1)
    reach_end = ((lx - px) * (lx - lax) + (ly - py) * (ly - lay)) / ((lx - lax).square() + (ly - lay).square()).sqrt().clamp(min=1e-6) + extend
    cut_start = torch.where(first >= 0, (reach_start + 0.5).clamp(0, 1), 1.0)
    cut_end = torch.where(last >= 0, (reach_end + 0.5).clamp(0, 1), 1.0)
    return dx, dy, rx, ry, t, d, d2, band, first, last, reach_start, reach_end, cut_start, cut_end


def _add_edge_gradient(grad_edges, edge, g_d, t, dx, dy, rx, ry):
    """Gradient `g_d` of each pixel's distance to its edge, into the edge's end points."""
    qx, qy = t * dx - rx, t * dy - ry
    scale = g_d / (qx * qx + qy * qy).sqrt().clamp(min=1e-6)
    grad_edges.index_add_(0, edge, torch.stack([scale * qx * (1 - t), scale * qy * (1 - t), scale * qx * t, scale * qy * t], 1))


def _add_cut_gradient(edges, grad_edges, cut, start, g, px, py):
    """As _cut_gradient: gradient of a cut's reach into the end points of edge `cut`."""
    keep = (cut >= 0) & (g != 0)
    cut, g, px, py = cut[keep], g[keep, None], px[keep], py[keep]
    ax, ay, bx, by = edges[cut].unbind(1)
    length = ((bx - ax).square() + (by - ay).square()).sqrt().clamp(min=1e-6)
    nx, ny = (bx - ax) / length, (by - ay) / length
    along = (px - ax) * nx + (py - ay) * ny
    wx, wy = (px - ax - along * nx) / length, (py - ay - along * ny) / length
    grad = torch.stack([-nx - wx, -ny - wy, wx, wy] if start else [wx, wy, nx - wx, ny - wy], 1)
    grad_edges.index_add_(0, cut, g * grad)


class _NearCoverage(torch.autograd.Function):
    """_LayerCoverage's coverage and gradients in PyTorch, for the CPU and other
    devices. Only pixels near an outline need distances: a fill's coverage is
    its inside test beyond half a pixel and its gradient vanishes beyond one;
    a stroke reaches half its width, plus its square cap and that pixel."""

    @staticmethod
    def forward(ctx, edges, cuts, squares, widths, modes, edge_layer, height, width):
        layers = len(modes)
        stroke = modes[edge_layer] == STROKE
        fills = (~stroke).nonzero()[:, 0]
        winding = _winding(edges[fills], edge_layer[fills], layers, height, width)
        winding[modes == FILL_EVENODD] &= 1
        inside = (winding != 0).view(-1)
        cover = inside.to(edges.dtype)
        e, y, x = _near_pixels(edges[fills], edges.new_ones(len(fills)), height, width)
        e = fills[e]
        ax, ay, bx, by = edges[e].unbind(1)
        dx, dy, rx, ry = bx - ax, by - ay, x + 0.5 - ax, y + 0.5 - ay
        t = ((rx * dx + ry * dy) / (dx * dx + dy * dy).clamp(min=1e-12)).clamp(0, 1)
        d2 = (rx - t * dx).square() + (ry - t * dy).square()
        key = (edge_layer[e] * height + y) * width + x
        nearest = _first_by_rank(key, d2, e)
        fill_key, fill_edge = key[nearest], e[nearest]
        signed = torch.where(inside[fill_key], d2[nearest].sqrt(), -d2[nearest].sqrt())
        cover[fill_key] = (signed + 0.5).clamp(0, 1)
        strokes = stroke.nonzero()[:, 0]
        half = widths[edge_layer[strokes]] * 0.5
        e, y, x = _near_pixels(edges[strokes], half + torch.where(squares[edge_layer[strokes]] != 0, half, 0.0) + 2, height, width)
        e = strokes[e]
        half = widths[edge_layer[e]] * 0.5
        *_, d2, band, _, _, _, _, cut_start, cut_end = _stroke_terms(edges, cuts, half, torch.where(squares[edge_layer[e]] != 0, half, 0.0), e, x + 0.5, y + 0.5)
        band = band * cut_start * cut_end
        key = (edge_layer[e] * height + y) * width + x
        best = _first_by_rank(key, -band, d2, e)
        stroke_key, stroke_edge = key[best], e[best]
        cover[stroke_key] = band[best]
        ctx.save_for_backward(edges, cuts, squares, widths, edge_layer, fill_key, fill_edge, signed, stroke_key, stroke_edge)
        ctx.size = height, width
        return cover.view(layers, height, width)

    @staticmethod
    def backward(ctx, grad_cover):
        edges, cuts, squares, widths, edge_layer, fill_key, fill_edge, signed, stroke_key, stroke_edge = ctx.saved_tensors
        height, width = ctx.size
        grad_cover = grad_cover.reshape(-1)
        grad_edges = torch.zeros_like(edges)
        grad_widths = torch.zeros_like(widths)
        # The 2 px tent filter's derivative, as in _coverage_backward.
        g = grad_cover[fill_key]
        px, py = fill_key % width + 0.5, fill_key // width % height + 0.5
        ax, ay, bx, by = edges[fill_edge].unbind(1)
        dx, dy, rx, ry = bx - ax, by - ay, px - ax, py - ay
        t = ((rx * dx + ry * dy) / (dx * dx + dy * dy).clamp(min=1e-12)).clamp(0, 1)
        _add_edge_gradient(grad_edges, fill_edge, torch.where(signed >= 0, g, -g) * (1 - signed.abs()).clamp(min=0), t, dx, dy, rx, ry)
        g = grad_cover[stroke_key]
        px, py = stroke_key % width + 0.5, stroke_key // width % height + 0.5
        layer = edge_layer[stroke_edge]
        half = widths[layer] * 0.5
        square = squares[layer] != 0
        dx, dy, rx, ry, t, d, _, band, first, last, reach_start, reach_end, cut_start, cut_end = _stroke_terms(edges, cuts, half, torch.where(square, half, 0.0), stroke_edge, px, py)
        outer = (1 - (half - d).abs()).clamp(min=0)
        inner = (1 - (half + d).abs()).clamp(min=0)
        g_start = torch.where(first >= 0, g * band * cut_end * (1 - reach_start.abs()).clamp(min=0), 0.0)
        g_end = torch.where(last >= 0, g * band * cut_start * (1 - reach_end.abs()).clamp(min=0), 0.0)
        g = g * cut_start * cut_end
        grad_widths.index_add_(0, layer, 0.5 * (g * (outer + inner) + torch.where(square, g_start + g_end, 0.0)))
        _add_cut_gradient(edges, grad_edges, first, True, g_start, px, py)
        _add_cut_gradient(edges, grad_edges, last, False, g_end, px, py)
        _add_edge_gradient(grad_edges, stroke_edge, g * (inner - outer), t, dx, dy, rx, ry)
        return grad_edges, None, None, grad_widths, None, None, None, None


def _kites(points, widths, corners, layers, limits):
    """_MiterKites in PyTorch: [C,4,2] corner, outer offsets and miter tip (or bevel midpoint)."""
    vertex, in_head, in_tail, out_head, out_tail = points.index_select(0, corners.reshape(-1)).view(5, -1, 2)
    tangents = F.normalize(torch.stack([in_head, out_head]) - torch.stack([in_tail, out_tail]), dim=2, eps=1e-3)
    normals = tangents.flip(2) * tangents.new_tensor([-1.0, 1.0])
    cos = (tangents[0] * tangents[1]).sum(1, keepdim=True)
    offsets = normals * (-torch.sign((normals[0] * tangents[1]).sum(1, keepdim=True)) * widths.index_select(0, layers)[:, None] / 2)
    first, second = vertex + offsets
    tip = torch.where((1 + cos) / 2 >= limits[:, None], vertex + offsets.sum(0) / (1 + cos).clamp(min=1e-6), (first + second) / 2)
    return torch.stack([vertex, first, tip, second], 1)


class _TorchComposite(torch.autograd.Function):
    """_Composite in PyTorch: each layer's visible alpha is its alpha times the
    transmittance of the layers above it, and white shows through all of them."""

    @staticmethod
    def forward(ctx, alpha, colors):
        layers, height, width = alpha.shape
        alpha = alpha.reshape(layers, -1)
        through = (1 - alpha).flip(0).cumprod(0).flip(0)
        above = torch.cat([through[1:], through.new_ones(1, height * width)])
        visible = alpha * above
        ctx.save_for_backward(alpha, colors, above, visible)
        ctx.size = height, width
        return (colors.T @ visible + through[0]).view(3, height, width)

    @staticmethod
    def backward(ctx, grad_out):
        alpha, colors, above, visible = ctx.saved_tensors
        grad_out = grad_out.reshape(3, -1)
        toward = colors @ grad_out
        # grad_out . composite below each layer, built bottom-up from white.
        below = grad_out.sum(0)
        grad_alpha = torch.empty_like(alpha)
        for layer in range(len(alpha)):
            torch.mul(above[layer], toward[layer] - below, out=grad_alpha[layer])
            below = torch.lerp(below, toward[layer], alpha[layer])
        return grad_alpha.view(-1, *ctx.size), visible @ grad_out.T


class _Plan:
    """Flattening and paint-layer plan of a scene: which control points make which
    polyline vertices, and which vertex pairs are the edges of each fill or stroke."""

    def __init__(self, scene, device):
        self.triton = triton is not None and torch.device(device).type == "cuda"
        # Each polyline vertex is a fixed Bernstein combination of 4 control points
        # (a subpath's endpoint is its last control point).
        controls, line_starts, columns, weights, rings = [], [], [], [], []
        base = vertex = 0
        for shape in scene.shapes:
            shape_rings = []
            for subpath, closed, straight in zip(shape.subpaths, shape.closed, shape.lines):
                controls.append(subpath)
                line_starts.append(base + 3 * np.flatnonzero(straight))
                lengths = np.linalg.norm(np.diff(subpath, axis=0), axis=1).reshape(-1, 3).sum(1)
                counts = np.asarray(SAMPLE_COUNTS)[np.minimum(np.searchsorted(2 * np.asarray(SAMPLE_COUNTS), lengths), len(SAMPLE_COUNTS) - 1)]
                segment = np.repeat(np.arange(len(counts)), counts)
                t = (np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)) / counts[segment]
                columns += [base + 3 * segment[:, None] + np.arange(4), np.full((1, 4), base + len(subpath) - 1)]
                weights += [np.stack([(1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t ** 2, t ** 3], 1), np.array([[1.0, 0, 0, 0]])]
                shape_rings.append((np.arange(vertex, vertex + counts.sum() + 1), closed, subpath, base))
                vertex += counts.sum() + 1
                base += len(subpath)
            rings.append(shape_rings)
        points = np.concatenate(controls)
        columns, weights = np.concatenate(columns), np.concatenate(weights)
        # The control points of a straight segment stay at 1/3 and 2/3 of its
        # endpoints: only the other points are parameters, so it stays straight.
        starts = np.concatenate(line_starts)
        free = np.ones(len(points), bool)
        free[starts + 1] = free[starts + 2] = False
        index = np.cumsum(free) - 1
        dependent = np.concatenate([starts + 1, starts + 1, starts + 2, starts + 2])
        rows = np.concatenate([np.flatnonzero(free), dependent])
        cols = np.concatenate([index[free], index[starts], index[starts + 3], index[starts], index[starts + 3]])
        values = np.concatenate([np.ones(free.sum()), np.full(len(starts), 2 / 3), np.full(len(starts), 1 / 3), np.full(len(starts), 1 / 3), np.full(len(starts), 2 / 3)])
        self.expand = sparse.csr_matrix((values, (rows, cols)), shape=(len(points), free.sum()))
        self.points = torch.from_numpy(points[free]).to(device)

        edge_starts, edge_ends, cuts, offsets, modes, colors, opacity, widths, clips = [], [], [], [0], [], [], [], [], []
        squares = []
        # Initial polyline vertices, to find the edges near open stroke ends.
        vertices = (weights[:, :, None] * points[columns]).sum(1)
        corners, corner_limits, miter_owners = [], [], []
        count = 0
        for shape, shape_rings in zip(scene.shapes, rings):
            for mode, paint in ((FILL_EVENODD if shape.fill_rule == "evenodd" else FILL_NONZERO, shape.fill), (STROKE, shape.stroke)):
                if paint is None:
                    continue
                for ids, closed, _, _ in shape_rings:
                    if mode != STROKE or closed:
                        edge_starts.append(ids)
                        edge_ends.append(np.roll(ids, -1))
                        cuts.append(np.full((2, len(ids)), -1, np.int32))
                    else:
                        edge_starts.append(ids[:-1])
                        edge_ends.append(ids[1:])
                        cuts.append(np.full((2, len(ids) - 1), -1, np.int32))
                        if shape.stroke_cap != "round":
                            # Round caps are the bands' own ends. Butt and square ends
                            # cut the edges within about a width of them.
                            lengths = np.linalg.norm(np.diff(vertices[ids], axis=0), axis=1)
                            reach = shape.stroke_width + 2
                            cuts[-1][0, np.cumsum(lengths) - lengths < reach] = count
                            cuts[-1][1, lengths[::-1].cumsum()[::-1] - lengths < reach] = count + len(lengths) - 1
                    count += len(edge_starts[-1])
                if mode == STROKE and shape.stroke_join == "miter":
                    layer_corners = np.concatenate([base + _corners(subpath, closed) for _, closed, subpath, base in shape_rings])
                    if len(layer_corners):
                        corners.append(layer_corners)
                        corner_limits.append(np.full(len(layer_corners), shape.miter_limit))
                        miter_owners.append(len(modes))
                offsets.append(count)
                squares.append(mode == STROKE and shape.stroke_cap == "square")
                modes.append(mode)
                colors.append(paint[:3])
                opacity.append(paint[3])
                widths.append(shape.stroke_width if mode == STROKE else 0.0)
                clips.append(shape.clip)
        # Edge endpoints as one constant sparse matrix over the free control points.
        ids = np.concatenate(edge_starts + edge_ends)
        flatten = (sparse.csr_matrix((weights[ids].reshape(-1), (np.repeat(np.arange(len(ids)), 4), columns[ids].reshape(-1))), shape=(len(ids), len(points))) @ self.expand).tocoo()
        values = torch.from_numpy(flatten.data).float()
        self.flatten = torch.sparse_coo_tensor(np.stack([flatten.row, flatten.col]), values, flatten.shape, check_invariants=True).coalesce().to(device)
        self.flatten_t = torch.sparse_coo_tensor(np.stack([flatten.col, flatten.row]), values, flatten.shape[::-1], check_invariants=True).coalesce().to(device)
        # Miter joins: one more fill layer per miter-joined stroke, made of a
        # 4-edge quadrilateral per corner (see _miter_kites).
        self.layers = len(modes)
        self.miter_owners = torch.tensor(miter_owners, device=device, dtype=torch.int64)
        if corners:
            counts = [len(layer_corners) for layer_corners in corners]
            self.corners = torch.from_numpy(np.concatenate(corners).T.copy()).to(device)
            self.corner_layer = torch.from_numpy(np.repeat(miter_owners, counts)).to(device)
            self.corner_limit = torch.from_numpy(1 / np.concatenate(corner_limits) ** 2).float().to(device)
            full = self.expand.tocoo()
            values = torch.from_numpy(full.data).float()
            self.expand_points = torch.sparse_coo_tensor(np.stack([full.row, full.col]), values, full.shape, check_invariants=True).coalesce().to(device)
            self.expand_points_t = torch.sparse_coo_tensor(np.stack([full.col, full.row]), values, full.shape[::-1], check_invariants=True).coalesce().to(device)
            offsets += (count + 4 * np.cumsum(counts)).tolist()
            modes += [FILL_NONZERO] * len(counts)
            cuts.append(np.full((2, 4 * sum(counts)), -1, np.int32))
            squares += [False] * len(counts)
        self.cuts = torch.from_numpy(np.concatenate(cuts, 1)).to(device)
        self.squares = torch.tensor(squares, device=device, dtype=torch.int32)
        self.edge_layer = torch.repeat_interleave(torch.arange(len(modes), device=device), torch.from_numpy(np.diff(offsets)).to(device))
        self.offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
        self.modes = torch.tensor(modes, device=device, dtype=torch.int32)
        self.colors = torch.tensor(colors, device=device, dtype=torch.float32)
        self.widths = torch.tensor(widths, device=device, dtype=torch.float32)
        self.scale = torch.tensor(opacity, device=device, dtype=torch.float32)[:, None, None]
        if any(clips):
            self.scale = self.scale * _clip_coverage(clips, scene.width, scene.height, device)
        self.size = (scene.height, scene.width)

    def _miter_kites(self, points, widths):
        """Miter join of each stroke corner as a quadrilateral: the corner, the
        two outer offsets of the stroke and the miter tip. Beyond the miter
        limit it collapses to the bevel triangle, inside the round join."""
        full = _Flatten.apply(points, self.expand_points, self.expand_points_t)
        ring = (_MiterKites.apply if self.triton else _kites)(full, widths, self.corners, self.corner_layer, self.corner_limit)
        return torch.cat([ring, ring.roll(-1, 1)], 2).reshape(-1, 4)

    def coverage(self, points, widths):
        edges = _Flatten.apply(points, self.flatten, self.flatten_t).view(2, -1, 2).transpose(0, 1).reshape(-1, 4)
        if len(self.miter_owners):
            edges = torch.cat([edges, self._miter_kites(points, widths)])
            widths = torch.cat([widths, widths.new_zeros(len(self.miter_owners))])
        if self.triton:
            ends = edges.detach().view(-1, 2, 2)
            layer = self.edge_layer[:, None].expand(-1, 2)
            pad = (widths.detach() / 2 + 1)[:, None]
            boxes = torch.cat([
                torch.full((len(self.modes), 2), FAR, device=edges.device).scatter_reduce(0, layer, ends.amin(1), "amin") - pad,
                torch.full((len(self.modes), 2), -FAR, device=edges.device).scatter_reduce(0, layer, ends.amax(1), "amax") + pad,
            ], 1)
            cover = _LayerCoverage.apply(edges, self.cuts, self.squares, widths, self.offsets, boxes, self.modes, *self.size)
        else:
            cover = _NearCoverage.apply(edges, self.cuts, self.squares, widths, self.modes, self.edge_layer, *self.size)
        if len(self.miter_owners):
            # A miter join adds its quadrilateral to the round-joined stroke band.
            cover = cover[:self.layers].index_put((self.miter_owners,), torch.maximum(cover[self.miter_owners], cover[self.layers:]))
        return cover

    def render(self, points, colors, widths):
        return (_Composite if self.triton else _TorchComposite).apply(self.coverage(points, widths) * self.scale, colors)

    def write_back(self, scene, points, colors, widths):
        points = (self.expand @ points.detach().cpu().numpy()).astype(np.float32)
        colors, widths = colors.detach().cpu().tolist(), widths.detach().cpu().tolist()
        base = layer = 0
        for shape in scene.shapes:
            for i, subpath in enumerate(shape.subpaths):
                shape.subpaths[i] = points[base:base + len(subpath)].copy()
                base += len(subpath)
            if shape.fill is not None:
                shape.fill = (*colors[layer], shape.fill[3])
                layer += 1
            if shape.stroke is not None:
                shape.stroke = (*colors[layer], shape.stroke[3])
                shape.stroke_width = widths[layer]
                layer += 1


def _corners(subpath, closed):
    """Corners of a stroked subpath where its tangent turns, as rows of control
    point indices: the vertex, then head and tail of the tangent arriving at it
    and of the tangent leaving it."""
    last = len(subpath) - 1

    def distinct(fixed, candidates):
        return next((i for i in candidates if np.linalg.norm(subpath[i] - subpath[fixed]) > 1e-3), candidates[-1])

    def arriving(end):
        return end, distinct(end, (end - 1, end - 2, end - 3))

    def leaving(start):
        return distinct(start, (start + 1, start + 2, start + 3)), start

    corners = [(end, *arriving(end), *leaving(end)) for end in range(3, last, 3)]
    if closed and np.linalg.norm(subpath[last] - subpath[0]) > 1e-3:
        # Around the implicit closing line from the last point to the start.
        corners += [(last, *arriving(last), 0, last), (0, 0, last, *leaving(0))]
    elif closed:
        corners.append((0, *arriving(last), *leaving(0)))
    corners = np.array(corners, dtype=np.int64).reshape(-1, 5)
    arrive = subpath[corners[:, 1]] - subpath[corners[:, 2]]
    leave = subpath[corners[:, 3]] - subpath[corners[:, 4]]
    cos = (arrive * leave).sum(1) / np.maximum(np.linalg.norm(arrive, axis=1) * np.linalg.norm(leave, axis=1), 1e-12)
    return corners[cos < SMOOTH_COS]


def _clip_coverage(clips, width, height, device):
    """Fixed alpha factor of each layer's clip regions (intersection of nonzero fills)."""
    regions, layer_regions = {}, []
    for layer_clips in clips:
        keys = [b"".join(subpath.tobytes() for subpath in region) for region in layer_clips]
        for key, region in zip(keys, layer_clips):
            regions.setdefault(key, (len(regions), region))
        layer_regions.append([regions[key][0] for key in keys])
    shapes = [Shape(list(region), [True] * len(region), [np.zeros(len(subpath) // 3, bool) for subpath in region], (0.0, 0.0, 0.0, 1.0), "nonzero", None, 0.0, []) for _, region in regions.values()]
    plan = _Plan(Scene(width, height, shapes), device)
    region_alpha = plan.coverage(plan.points, plan.widths)
    factors = torch.ones((len(clips), height, width), device=device)
    for factor, indices in zip(factors, layer_regions):
        for index in indices:
            factor *= region_alpha[index]
    return factors


def _optimize(scene, target, steps, progress):
    """Adam on all control points, colors and stroke widths; keeps the best step."""
    plan = _Plan(scene, target.device)
    points = plan.points.clone().requires_grad_()
    colors = plan.colors.clone().requires_grad_()
    widths = plan.widths.clone().requires_grad_()
    optimizer = torch.optim.Adam([
        {"params": [points], "lr": POINT_LR},
        {"params": [colors], "lr": COLOR_LR},
        {"params": [widths], "lr": WIDTH_LR},
    ], fused=True)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - 0.9 * step / steps)
    best = [points.detach().clone(), colors.detach().clone(), widths.detach().clone()]
    best_loss = torch.full((), math.inf, device=target.device)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        image = plan.render(points, colors, widths)
        loss = (image - target).square().mean()
        loss.backward()
        # Compare the losses on the device: the CPU queues the next steps while
        # the GPU computes, where reading each loss into Python would wait for it.
        improved = loss.detach() < best_loss
        best_loss = torch.minimum(best_loss, loss.detach())
        for kept, current in zip(best, (points, colors, widths)):
            kept.copy_(torch.where(improved, current.detach(), kept))
        optimizer.step()
        schedule.step()
        colors.detach().clamp_(0, 1)
        widths.detach().clamp_(min=0)
        progress()
    plan.write_back(scene, *best)
    return plan.render(*best).detach()


def _add_path(scene, target, image):
    """New path on top of the scene, traced from the connected region of one
    quantized target color with the largest error, colored like the target
    there. False when no region qualifies."""
    error = (image - target).abs().sum(0).cpu().numpy()
    levels = (target * LEVELS).long().clamp(max=LEVELS - 1)
    codes = ((levels[0] * LEVELS + levels[1]) * LEVELS + levels[2]).cpu().numpy()
    seeds = error > SEED_ERROR
    best_mass, region = 0.0, None
    for code in np.unique(codes[seeds]):
        labels, count = ndimage.label(seeds & (codes == code))
        index = np.arange(1, count + 1)
        masses = np.where(ndimage.sum(seeds, labels, index) >= MIN_SEED_PIXELS, ndimage.sum(error, labels, index), 0)
        if masses.max() > best_mass:
            best_mass, region = masses.max(), labels == np.argmax(masses) + 1
    if region is None:
        return False
    buffer = BytesIO()
    Image.fromarray(np.where(region, 0, 255).astype(np.uint8)).save(buffer, "PNG")
    traced = capture_svg(vtracer.convert_raw_image_to_svg(buffer.getvalue(), img_format="png", colormode="binary", path_precision=3), scene.width, scene.height)
    subpaths = [subpath for shape in traced.shapes for subpath in shape.subpaths]
    if not subpaths:
        return False
    lines = [straight for shape in traced.shapes for straight in shape.lines]
    color = target[:, torch.from_numpy(region).to(target.device)].mean(1).tolist()
    scene.shapes.append(Shape(subpaths, [True] * len(subpaths), lines, (*color, 1.0), "nonzero", None, 0.0, []))
    return True


def refine_svg(markup, image, steps, new_paths, progress=lambda: None, resolution=RESOLUTION):
    """Fit `markup` to `image` ([H,W,3|4] floats in [0,1]) at `resolution`
    pixels on the longest side, adding up to `new_paths` paths. Runs on the
    image's CUDA device, otherwise on the CPU. Returns the refined SVG, or
    `markup` unchanged when CairoSVG draws something the rasterizer cannot
    represent (gradients, text, images...)."""
    if image.device.type != "cuda":
        image = image.cpu()
    height, width = image.shape[:2]
    scale = resolution / max(height, width)
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    scene = capture_svg(markup, size[1], size[0])
    if scene.unsupported:
        _LOG.warning("StarVector: SVG not refined, it uses %s", ", ".join(scene.unsupported))
        return markup
    with torch.inference_mode(False), torch.enable_grad():
        rgb = image[..., :3] * image[..., 3:] + (1 - image[..., 3:]) if image.shape[-1] == 4 else image
        target = F.interpolate(rgb.permute(2, 0, 1)[None].float(), size=size, mode="bilinear", antialias=True)[0].clamp(0, 1)
        rendered = _optimize(scene, target, steps, progress) if scene.shapes else torch.ones_like(target)
        for _ in range(new_paths):
            if not _add_path(scene, target, rendered):
                break
            rendered = _optimize(scene, target, steps, progress)
    return scene_to_svg(scene)
