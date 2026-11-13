"""Tests for the differentiable SVG rasterizer and the LIVE-style refinement."""
import io
import unittest

import cairosvg
import numpy as np
import torch
from PIL import Image

import package_alias  # noqa: F401
from starvector_node import svg_refine
from starvector_node.svg_refine import _Plan, _corners, refine_svg
from starvector_node.svg_scene import capture_svg, scene_to_svg

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
needs_cuda = unittest.skipUnless(torch.cuda.is_available() and svg_refine.triton is not None, "needs CUDA and Triton")


def raster(markup):
    png = cairosvg.svg2png(bytestring=markup.encode(), output_width=64, output_height=64, background_color="white")
    return torch.from_numpy(np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32) / 255).permute(2, 0, 1)


STAR = '<polygon points="32,6 39,25 58,25 43,37 49,57 32,45 15,57 21,37 6,25 25,25" fill="#f0b400" stroke="#203040" stroke-width="3"/>'
TRIANGLE = '<polygon points="6,32 58,24 58,40" fill="none" stroke="#aa2200" stroke-width="5"{}/>'


def line(a, b):
    a, b = np.array(a, np.float32), np.array(b, np.float32)
    return [a, a + (b - a) / 3, a + 2 * (b - a) / 3, b]


def polyline(*vertices):
    subpath = line(vertices[0], vertices[1])
    for a, b in zip(vertices[1:], vertices[2:]):
        subpath += line(a, b)[1:]
    return np.array(subpath, np.float32)


CASES = {
        "filled square": '<rect x="10" y="12" width="40" height="36" fill="#cc3322"/>',
        "even-odd ring": '<path d="M5 5H55V55H5Z M20 20H40V40H20Z" fill="#224488" fill-rule="evenodd"/>',
        "open polyline": '<polyline points="8,50 20,10 36,44 56,14" fill="none" stroke="#118833" stroke-width="5"/>',
        "closed stroke": '<rect x="12" y="14" width="38" height="30" fill="none" stroke="#880088" stroke-width="6"/>',
        "miter star": STAR,
        "bevelled sharp triangle": TRIANGLE.format(""),
        "mitered sharp triangle": TRIANGLE.format(' stroke-miterlimit="20"'),
        "open zigzag": '<polyline points="6,50 20,14 34,50 48,14 58,40" fill="none" stroke="#118833" stroke-width="6"/>',
        "butt line": '<line x1="10.3" y1="20.6" x2="52.7" y2="38.2" stroke="#118833" stroke-width="10"/>',
        "square line": '<line x1="10.3" y1="20.6" x2="52.7" y2="38.2" stroke="#118833" stroke-width="10" stroke-linecap="square"/>',
        "round line": '<line x1="10.3" y1="20.6" x2="52.7" y2="38.2" stroke="#118833" stroke-width="10" stroke-linecap="round"/>',
        "butt zigzag": '<polyline points="6,50 20,14 34,50 48,14 58,40" fill="none" stroke="#118833" stroke-width="12"/>',
        "square zigzag": '<polyline points="6,50 20,14 34,50 48,14 58,40" fill="none" stroke="#118833" stroke-width="12" stroke-linecap="square"/>',
        "butt curve": '<path d="M8 50C10 10 50 10 56 50" fill="none" stroke="#880088" stroke-width="10"/>',
        "translucent fills": '<rect width="40" height="40" fill="#ff0000" fill-opacity="0.5"/><rect x="20" y="20" width="40" height="40" fill="#0000ff" fill-opacity="0.6"/>',
        "partial clip": '<defs><clipPath id="c"><circle cx="24" cy="32" r="18"/></clipPath></defs><rect width="64" height="40" fill="red" clip-path="url(#c)"/>',
}


def square(x):
    return SVG + f'<rect x="{x}" y="14" width="24" height="30" fill="#cc3322"/></svg>'


class RefineSVGTests:
    def test_rasterizer_matches_cairosvg(self):
        for name, body in CASES.items():
            with self.subTest(name):
                scene = capture_svg(SVG + body + "</svg>", 64, 64)
                plan = _Plan(scene, self.device)
                rendered = plan.render(plan.points, plan.colors, plan.widths).cpu()
                self.assertLess((rendered - raster(scene_to_svg(scene))).abs().mean().item(), 0.01)

    def test_gradient_matches_finite_difference(self):
        torch.manual_seed(0)
        markup = SVG + '<path d="M10 12L50 14L30 50Z" fill="#cc3322"/><path d="M8 52L28 40L56 56" fill="none" stroke="#2244aa" stroke-width="4"/></svg>'
        plan = _Plan(capture_svg(markup, 64, 64), self.device)
        # Drawn on the CPU, so that every device checks the same target.
        target = torch.rand(3, 64, 64).to(self.device)
        base = [plan.points, plan.colors, plan.widths]

        def loss(*params):
            return (plan.render(*params) - target).square().sum()

        params = [p.clone().requires_grad_() for p in base]
        grads = torch.autograd.grad(loss(*params), params)
        # Along the gradient: a random direction can be nearly orthogonal to
        # it, and the relative error of a near-zero derivative is meaningless.
        for scale in (0.05, 0.1):
            steps = [scale * g / g.norm() for g in grads]
            analytic = sum((g * s).sum() for g, s in zip(grads, steps)).item()
            plus, minus = (loss(*[p + sign * s for p, s in zip(base, steps)]) for sign in (1, -1))
            numeric = ((plus - minus) / 2).item()
            self.assertLess(abs(analytic - numeric) / abs(numeric), 0.05)

    def test_gradient_with_caps_matches_finite_difference(self):
        torch.manual_seed(0)
        target = torch.rand(3, 64, 64, device=self.device)
        for cap in ("butt", "square"):
            with self.subTest(cap):
                plan = _Plan(capture_svg(SVG + f'<polyline points="8,50 24,14 40,48 56,16" fill="none" stroke="#2244aa" stroke-width="8" stroke-linecap="{cap}"/></svg>', 64, 64), self.device)
                base = [plan.points, plan.colors, plan.widths]

                def loss(*params):
                    return (plan.render(*params) - target).square().sum()

                params = [p.clone().requires_grad_() for p in base]
                grads = torch.autograd.grad(loss(*params), params)
                steps = [0.1 * g / g.norm() for g in grads]
                analytic = sum((g * s).sum() for g, s in zip(grads, steps)).item()
                plus, minus = (loss(*[p + sign * s for p, s in zip(base, steps)]) for sign in (1, -1))
                numeric = ((plus - minus) / 2).item()
                self.assertLess(abs(analytic - numeric) / abs(numeric), 0.05)

    def test_cuts_mark_edges_near_open_stroke_ends(self):
        def stroke_plan(body):
            return _Plan(capture_svg(SVG + body + "</svg>", 64, 64), self.device)

        open_line = '<polyline points="4,10 60,10 60,54" fill="none" stroke="black" stroke-width="4"{}/>'
        plan = stroke_plan(open_line.format(""))
        first, end = plan.offsets[0].item(), plan.offsets[1].item() - 1
        cuts = plan.cuts.cpu()
        self.assertEqual(cuts[0, first].item(), first)
        self.assertEqual(cuts[1, end].item(), end)
        self.assertEqual(cuts[0, first + 1].item(), first)
        self.assertEqual(cuts[0, first + 2].item(), -1)
        self.assertEqual(cuts[1, end - 1].item(), end)
        self.assertEqual(cuts[1, end - 3].item(), -1)
        middle = first + 20
        self.assertEqual(cuts[:, middle].tolist(), [-1, -1])
        self.assertEqual(plan.squares[0].item(), 0)
        self.assertEqual(stroke_plan(open_line.format(' stroke-linecap="square"')).squares[0].item(), 1)
        for name, body in (("round", open_line.format(' stroke-linecap="round"')), ("closed", '<polygon points="8,8 56,8 32,56" fill="none" stroke="black" stroke-width="4"/>')):
            with self.subTest(name):
                self.assertTrue((stroke_plan(body).cuts == -1).all())

    def test_refine_fits_line_length(self):
        full = SVG + '<line x1="8" y1="32" x2="56" y2="32" stroke="#118833" stroke-width="12"/></svg>'
        short = full.replace('x2="56"', 'x2="41"')
        image = raster(full).permute(1, 2, 0).to(self.device)
        with torch.inference_mode():
            refined = refine_svg(short, image, 200, 0, resolution=64)
        self.assertNotIn("stroke-linecap", refined)
        self.assertLess((raster(refined) - raster(full)).square().mean() * 10, (raster(short) - raster(full)).square().mean())

    def test_refine_fits_shifted_pixel_aligned_square(self):
        with torch.inference_mode():
            image = raster(square(18)).permute(1, 2, 0).to(self.device)
            refined = refine_svg(square(14), image, 200, 0, resolution=64)
        target = raster(square(18))
        before = (raster(square(14)) - target).square().mean()
        after = (raster(refined) - target).square().mean()
        self.assertLess(after * 10, before)

    def test_straight_segments_stay_straight(self):
        star = SVG + '<polygon points="32,6 39,25 58,25 43,37 49,57 32,45 15,57 21,37 6,25 25,25" fill="#f0b400"/></svg>'
        image = raster(star.replace("32,6 ", "32,2 ")).permute(1, 2, 0).to(self.device)
        refined = refine_svg(star, image, 100, 0, resolution=64)
        self.assertNotIn("C", refined.split('d="')[1].split('"')[0])
        self.assertLess((raster(refined) - raster(star.replace("32,6 ", "32,2 "))).abs().mean(), (raster(star) - raster(star.replace("32,6 ", "32,2 "))).abs().mean())

    def test_refine_keeps_sharp_miter_tips(self):
        star = SVG + STAR + "</svg>"
        moved = star.replace("32,6 ", "32,3 ").replace("6,25 25,25", "3,27 25,25")
        image = raster(moved).permute(1, 2, 0).to(self.device)
        with torch.inference_mode():
            refined = refine_svg(star, image, 150, 0, resolution=64)
        self.assertNotIn("stroke-linejoin", refined)
        self.assertLess((raster(refined) - raster(moved)).abs().mean() * 5, (raster(star) - raster(moved)).abs().mean())

    def test_new_paths_from_empty_document(self):
        disk = '<circle cx="32" cy="32" r="18" fill="#2266cc"/></svg>'
        target = raster(SVG + disk)
        image = target.permute(1, 2, 0).to(self.device)
        refined = refine_svg(SVG[:-1] + "/>", image, 50, 2, resolution=64)
        self.assertIn("<path", refined)
        self.assertLess((raster(refined) - target).square().mean() * 4, (torch.ones_like(target) - target).square().mean())

    def test_unsupported_content_is_returned_unchanged(self):
        markup = SVG + '<defs><linearGradient id="g"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="64" height="64" fill="url(#g)"/></svg>'
        with self.assertLogs("starvector_node.svg_refine", "WARNING"):
            self.assertEqual(refine_svg(markup, torch.ones(64, 64, 3, device=self.device), 5, 0, resolution=64), markup)

    def test_progress_called_once_per_step(self):
        calls = []
        refine_svg(square(14), raster(square(18)).permute(1, 2, 0).to(self.device), 7, 0, lambda: calls.append(1), resolution=64)
        self.assertEqual(len(calls), 7)


class RefineSVGCPUTests(RefineSVGTests, unittest.TestCase):
    device = "cpu"


@needs_cuda
class RefineSVGCUDATests(RefineSVGTests, unittest.TestCase):
    device = "cuda"


@needs_cuda
class TritonMatchesTorchTests(unittest.TestCase):
    def test_miter_kites_match_torch(self):
        torch.manual_seed(0)
        markup = SVG + STAR + TRIANGLE.format("") + '<path d="M8 40C20 10 36 10 40 40L56 52" fill="none" stroke="#2244aa" stroke-width="4"/></svg>'
        plan = _Plan(capture_svg(markup, 64, 64), "cuda")
        points = (plan.points + 0.3 * torch.randn_like(plan.points)).requires_grad_()
        widths = plan.widths.clone().requires_grad_()
        weight = torch.randn(len(plan.corners.T), 4, 2, device="cuda")
        results = []
        for kites in (svg_refine._MiterKites.apply, svg_refine._kites):
            full = svg_refine._Flatten.apply(points, plan.expand_points, plan.expand_points_t)
            out = kites(full, widths, plan.corners, plan.corner_layer, plan.corner_limit)
            grads = torch.autograd.grad((out * weight).sum(), [points, widths])
            results.append((out.detach(), *grads))
        for name, ours, expected in zip(("kites", "points", "widths"), *results):
            self.assertLess((ours - expected).abs().max().item(), 1e-4, name)

    def test_render_matches_torch(self):
        for name, body in CASES.items():
            with self.subTest(name):
                torch.manual_seed(0)
                scene = capture_svg(SVG + body + "</svg>", 64, 64)
                plans = {device: _Plan(scene, device) for device in ("cuda", "cpu")}
                self.assertTrue(plans["cuda"].triton)
                self.assertFalse(plans["cpu"].triton)
                noise = 0.2 * torch.randn_like(plans["cpu"].points)
                target = torch.rand(3, 64, 64)
                outputs, grads = {}, {}
                for device, plan in plans.items():
                    params = [(plan.points + noise.to(device)).requires_grad_(), plan.colors.clone().requires_grad_(), plan.widths.clone().requires_grad_()]
                    rendered = plan.render(*params)
                    grads[device] = torch.autograd.grad((rendered - target.to(device)).square().sum(), params)
                    outputs[device] = rendered.detach().cpu()
                self.assertLess((outputs["cuda"] - outputs["cpu"]).abs().max().item(), 1e-5)
                for parameter, triton_grad, torch_grad in zip(("points", "colors", "widths"), grads["cuda"], grads["cpu"]):
                    norm = triton_grad.norm().item()
                    if norm > 0:
                        self.assertLess((triton_grad.cpu() - torch_grad).norm().item() / norm, 1e-3, parameter)


class CornerTests(unittest.TestCase):
    def test_corner_counts(self):
        square = polyline((0, 0), (10, 0), (10, 10), (0, 10), (0, 0))
        zigzag = polyline((0, 0), (10, 5), (0, 10), (10, 15))
        open_square = polyline((0, 0), (10, 0), (10, 10), (0, 10))
        circle = np.array([(10, 0), (10, 5.523), (5.523, 10), (0, 10), (-5.523, 10), (-10, 5.523), (-10, 0), (-10, -5.523), (-5.523, -10), (0, -10), (5.523, -10), (10, -5.523), (10, 0)], np.float32)
        self.assertEqual(len(_corners(square, True)), 4)
        self.assertEqual(len(_corners(zigzag, False)), 2)
        self.assertEqual(len(_corners(circle, True)), 0)
        closing = _corners(open_square, True)
        self.assertEqual(len(closing), 4)
        self.assertEqual(sorted(closing[:, 0].tolist()), [0, 3, 6, 9])
        self.assertEqual(closing[closing[:, 0] == 9, 3:].tolist(), [[0, 9]])


if __name__ == "__main__":
    unittest.main()
