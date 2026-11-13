"""Offline tests for the CairoSVG-based scene capture."""
import io
import unittest

import cairosvg
import numpy as np
from PIL import Image

import package_alias  # noqa: F401
from starvector_node.svg_scene import capture_svg, scene_to_svg

SVG = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '


def raster(markup):
    png = cairosvg.svg2png(bytestring=markup.encode(), output_width=64, output_height=64, background_color="white")
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32) / 255


class SVGSceneTests(unittest.TestCase):
    def assertRoundTrip(self, markup):
        diff = np.abs(raster(markup) - raster(scene_to_svg(capture_svg(markup, 64, 64)))).mean()
        self.assertLess(diff, 0.01)

    def test_viewbox_transform_and_style_class(self):
        markup = SVG + 'viewBox="0 0 10 10"><style>.a{fill:#ff0000}</style><g transform="translate(2 0)"><rect class="a" width="4" height="2"/></g></svg>'
        scene = capture_svg(markup, 64, 64)
        self.assertEqual(len(scene.shapes), 1)
        shape = scene.shapes[0]
        self.assertEqual(shape.fill, (1.0, 0.0, 0.0, 1.0))
        self.assertIsNone(shape.stroke)
        self.assertEqual(shape.clip, [])
        self.assertTrue(shape.closed[0])
        np.testing.assert_allclose(shape.subpaths[0].min(0), [12.8, 0], atol=0.01)
        np.testing.assert_allclose(shape.subpaths[0].max(0), [38.4, 12.8], atol=0.01)
        self.assertEqual(scene.unsupported, [])
        self.assertRoundTrip(markup)

    def test_basic_shapes_round_trip(self):
        cases = {
            "rounded rect": '<rect x="4" y="4" width="50" height="40" rx="8" ry="5" fill="#336699"/>',
            "circle": '<circle cx="30" cy="30" r="20" fill="green"/>',
            "arc path": '<path d="M10 30 A 20 20 0 0 1 50 30 L 30 55 z" fill="orange"/>',
            "evenodd": '<path d="M5 5H55V55H5Z M20 20H40V40H20Z" fill="black" fill-rule="evenodd"/>',
            "use": '<defs><rect id="r" width="20" height="20" fill="purple"/></defs><use xlink:href="#r" x="10" y="15"/>',
        }
        for name, body in cases.items():
            with self.subTest(name):
                markup = SVG + 'viewBox="0 0 64 64">' + body + "</svg>"
                scene = capture_svg(markup, 64, 64)
                self.assertTrue(scene.shapes)
                self.assertEqual(scene.unsupported, [])
                self.assertRoundTrip(markup)
        scene = capture_svg(SVG + 'viewBox="0 0 64 64">' + cases["evenodd"] + "</svg>", 64, 64)
        self.assertEqual(scene.shapes[0].fill_rule, "evenodd")
        self.assertEqual(len(scene.shapes[0].subpaths), 2)

    def test_fill_and_stroke_make_one_shape_with_device_width(self):
        markup = SVG + 'viewBox="0 0 32 32"><polygon points="4,4 28,6 16,28" fill="#00ff00" fill-opacity="0.5" stroke="#0000ff" stroke-width="2"/></svg>'
        scene = capture_svg(markup, 64, 64)
        self.assertEqual(len(scene.shapes), 1)
        shape = scene.shapes[0]
        self.assertEqual(shape.fill, (0.0, 1.0, 0.0, 0.5))
        self.assertEqual(shape.stroke, (0.0, 0.0, 1.0, 1.0))
        self.assertAlmostEqual(shape.stroke_width, 4.0, places=4)
        self.assertEqual(shape.subpaths[0].shape, (7, 2))
        self.assertRoundTrip(markup)

    def test_group_opacity_multiplies_alpha(self):
        markup = SVG + 'viewBox="0 0 64 64"><g opacity="0.5"><g opacity="0.5"><rect width="40" height="40" fill="#ff0000"/></g><rect x="20" y="20" width="40" height="40" fill="#0000ff" fill-opacity="0.8"/></g></svg>'
        scene = capture_svg(markup, 64, 64)
        self.assertEqual(scene.unsupported, [])
        self.assertAlmostEqual(scene.shapes[0].fill[3], 0.25)
        self.assertAlmostEqual(scene.shapes[1].fill[3], 0.4)
        self.assertRoundTrip(markup)

    def test_clip_paths(self):
        full = '<clipPath id="c"><rect width="64" height="64"/></clipPath>'
        part = '<clipPath id="c"><circle cx="20" cy="32" r="18"/></clipPath>'
        for name, clip in (("full canvas", full), ("partial", part)):
            with self.subTest(name):
                markup = SVG + f'viewBox="0 0 64 64"><defs>{clip}</defs><g clip-path="url(#c)"><rect width="64" height="40" fill="red"/></g><rect x="30" y="40" width="30" height="20" fill="blue"/></svg>'
                scene = capture_svg(markup, 64, 64)
                self.assertEqual(len(scene.shapes[0].clip), 1)
                self.assertEqual(scene.shapes[1].clip, [])
                self.assertEqual(scene.unsupported, [])
                self.assertRoundTrip(markup)

    def test_unsupported_features_are_reported(self):
        cases = {
            "gradient fill": '<defs><linearGradient id="g"><stop offset="0" stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="64" height="64" fill="url(#g)"/>',
            "dashed stroke": '<path d="M5 5H60" stroke="black" stroke-dasharray="4 2"/>',
            "text": '<text x="5" y="30">hi</text>',
            "filter": '<filter id="f"><feOffset dx="2"/></filter><rect width="20" height="20" filter="url(#f)"/>',
        }
        for reason, body in cases.items():
            with self.subTest(reason):
                scene = capture_svg(SVG + 'viewBox="0 0 64 64">' + body + "</svg>", 64, 64)
                self.assertIn(reason, scene.unsupported)

    def test_serialization_format(self):
        markup = SVG + 'viewBox="0 0 64 64"><defs><clipPath id="c"><rect width="32" height="64"/></clipPath></defs><g clip-path="url(#c)"><path d="M0 0H3" fill="none" stroke="#102030" stroke-opacity="0.5" stroke-width="2.5"/><rect width="2" height="2" fill="red"/></g></svg>'
        svg = scene_to_svg(capture_svg(markup, 64, 64))
        self.assertTrue(svg.startswith('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'))
        self.assertEqual(svg.count("<clipPath"), 1)
        self.assertEqual(svg.count('<g clip-path="url(#c0)">'), 2)
        self.assertIn('fill="none" stroke="#102030" stroke-opacity="0.5" stroke-width="2.5"', svg)
        # Straight segments stay lines, curves stay cubic.
        self.assertIn('d="M0 0 L3 0"', svg)
        self.assertIn('d="M0 0 L2 0 L2 2 L0 2 Z"', svg)
        scene = capture_svg(SVG + 'viewBox="0 0 64 64"><path d="M0 0L8 0C8 4 4 8 0 8Z"/></svg>', 64, 64)
        self.assertEqual(scene.shapes[0].lines[0].tolist(), [True, False])
        self.assertIn("M0 0 L8 0 C8 4 4 8 0 8 Z", scene_to_svg(scene))

    def test_stroke_join_capture(self):
        cases = {
            "default": ("", "miter", 4.0),
            "round": (' stroke-linejoin="round"', "round", 4.0),
            "bevel": (' stroke-linejoin="bevel"', "round", 4.0),
            "limit": (' stroke-miterlimit="10"', "miter", 10.0),
        }
        for name, (attrs, join, limit) in cases.items():
            for tag in ("polyline", "polygon"):
                with self.subTest(name, tag=tag):
                    scene = capture_svg(SVG + f'viewBox="0 0 64 64"><{tag} points="8,50 30,10 52,50" fill="none" stroke="black" stroke-width="4"{attrs}/></svg>', 64, 64)
                    self.assertEqual((scene.shapes[0].stroke_join, scene.shapes[0].miter_limit), (join, limit))

    def test_stroke_join_serialization(self):
        for name, join, limit, expected in (("default", "miter", 4.0, ()), ("limit", "miter", 10.0, ('stroke-miterlimit="10"',)), ("round", "round", 4.0, ('stroke-linejoin="round"',))):
            with self.subTest(name):
                scene = capture_svg(SVG + 'viewBox="0 0 64 64"><polyline points="8,50 30,10 52,50" fill="none" stroke="black" stroke-width="4"/></svg>', 64, 64)
                scene.shapes[0].stroke_join, scene.shapes[0].miter_limit = join, limit
                svg = scene_to_svg(scene)
                self.assertNotIn("stroke-linecap", svg)
                for attr in ("stroke-linejoin", "stroke-miterlimit"):
                    self.assertEqual(attr in svg, any(e.startswith(attr) for e in expected))
                for attr in expected:
                    self.assertIn(attr, svg)

    def test_stroke_cap_capture_and_serialization(self):
        for name, attrs, cap, expected in (("default", "", "butt", None), ("round", ' stroke-linecap="round"', "round", 'stroke-linecap="round"'), ("square", ' stroke-linecap="square"', "square", 'stroke-linecap="square"')):
            with self.subTest(name):
                scene = capture_svg(SVG + f'viewBox="0 0 64 64"><polyline points="8,50 30,10 52,50" fill="none" stroke="black" stroke-width="4"{attrs}/></svg>', 64, 64)
                self.assertEqual(scene.shapes[0].stroke_cap, cap)
                svg = scene_to_svg(scene)
                if expected is None:
                    self.assertNotIn("stroke-linecap", svg)
                else:
                    self.assertIn(expected, svg)
                self.assertRoundTrip(SVG + f'viewBox="0 0 64 64"><polyline points="8,50 30,10 52,50" fill="none" stroke="black" stroke-width="4"{attrs}/></svg>')


if __name__ == "__main__":
    unittest.main()
