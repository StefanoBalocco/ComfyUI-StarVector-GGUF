"""Offline tests for the upstream Beautiful Soup -> CairoSVG SVG-to-SVG recovery."""
import unittest
from xml.etree import ElementTree

import package_alias  # noqa: F401
from starvector_node.svg_recovery import EMPTY_SVG, recover_svg_markup


class SVGRecoveryTests(unittest.TestCase):
    def test_valid_svg_passes_through_byte_for_byte(self):
        source = '<svg xmlns="http://www.w3.org/2000/svg"><rect width="5" height="5"/></svg>'
        self.assertEqual(recover_svg_markup(source), (source, "original", None))

    def test_truncated_group_and_svg_are_recovered_using_cairosvg(self):
        source = '<svg viewBox="0 0 100 100"><g><path d="M 0 0 L 10 10" /></g>'
        result, status, error = recover_svg_markup(source)
        self.assertEqual(status, "repaired", error)
        self.assertIsNotNone(error)
        root = ElementTree.fromstring(result)
        self.assertEqual(root.tag.rsplit("}", 1)[-1], "svg")
        self.assertTrue(any(tag.tag.rsplit("}", 1)[-1] == "path" for tag in root.iter()))
        self.assertIn("xmlns", result)

    def test_incomplete_attribute_can_be_lost_but_recovery_returns_svg(self):
        source = '<svg width="100" height="100"><path d="M 0 0 L 10'
        result, status, error = recover_svg_markup(source)
        self.assertEqual(status, "repaired", error)
        self.assertIsNotNone(error)
        self.assertEqual(ElementTree.fromstring(result).tag.rsplit("}", 1)[-1], "svg")

    def test_svg2svg_preserves_vector_format(self):
        source = '<svg width="100" height="100"><rect width="30" height="30">'
        result, status, error = recover_svg_markup(source)
        self.assertEqual(status, "repaired", error)
        self.assertIn("<svg", result)
        self.assertTrue(any(tag.tag.rsplit("}", 1)[-1] == "path" for tag in ElementTree.fromstring(result).iter()))
        self.assertNotIn("data:image/png", result)

    def test_non_svg_root_falls_back_to_empty(self):
        result, status, error = recover_svg_markup("<html><svg/></html>")
        self.assertEqual((result, status), (EMPTY_SVG, "empty"))
        self.assertIsNotNone(error)

    def test_unrecoverable_garbage_returns_empty_svg(self):
        result, status, error = recover_svg_markup("not XML at all")
        self.assertEqual((result, status), (EMPTY_SVG, "empty"))
        self.assertIn("recovery failed", error)

    def test_missing_svg_size_uses_upstream_fallback(self):
        result, status, error = recover_svg_markup("<svg><g>")
        self.assertEqual((result, status), (EMPTY_SVG, "empty"))
        self.assertIn("SVG size is undefined", error)

    def test_namespaced_valid_svg_is_preserved(self):
        source = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
        self.assertEqual(recover_svg_markup(source), (source, "original", None))

    def test_doctype_is_not_processed(self):
        source = '<!DOCTYPE svg [<!ENTITY ext SYSTEM "file:///etc/passwd">]><svg width="100" height="100">&ext;'
        result, status, error = recover_svg_markup(source)
        self.assertEqual((result, status), (EMPTY_SVG, "empty"))
        self.assertIn("DOCTYPE", error)


if __name__ == "__main__":
    unittest.main()
