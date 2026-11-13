"""ComfyUI node tests with stand-ins for folder_paths, comfy and comfy_api."""
from io import BytesIO
from pathlib import Path
import importlib
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import torch

import package_alias  # noqa: F401


class NativeSVG:
    """Offline stand-in with exactly the ComfyUI SVG(data: list[BytesIO]) contract."""

    def __init__(self, data):
        self.data = data


class SVGNodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        paths = types.ModuleType("folder_paths")
        paths.models_dir = str(root / "models")
        temporary = root / "temp"
        temporary.mkdir()
        paths.get_temp_directory = lambda: str(temporary)
        # Simulate ComfyUI having already read a starvector entry from
        # extra_model_paths.yaml before importing the custom node.
        configured = str(root / "shared-models" / "starvector")
        paths.folder_names_and_paths = {
            "starvector": ([configured], set())
        }
        def add_model_folder_path(name, directory, is_default=False):
            if name not in paths.folder_names_and_paths:
                paths.folder_names_and_paths[name] = ([directory], set())
            else:
                directories, _ = paths.folder_names_and_paths[name]
                if directory not in directories:
                    if is_default:
                        directories.insert(0, directory)
                    else:
                        directories.append(directory)
        paths.add_model_folder_path = add_model_folder_path
        paths.get_folder_paths = lambda name: paths.folder_names_and_paths[name][0][:]
        self.paths = paths
        self.configured_directory = configured
        pkg = types.ModuleType("comfy_api")
        pkg.__path__ = []
        latest = types.ModuleType("comfy_api.latest")
        latest.io = types.SimpleNamespace(SVG=types.SimpleNamespace(Type=NativeSVG))

        self.calls, self.progress_events = [], []
        management = types.ModuleType("comfy.model_management")
        management.load_models_gpu = lambda models, force_full_load=False: self.calls.append((models, force_full_load))
        management.get_torch_device = lambda: torch.device("cpu")
        patcher = types.ModuleType("comfy.model_patcher")
        patcher.ModelPatcher = lambda model, **options: types.SimpleNamespace(model=model, **options)
        utils = types.ModuleType("comfy.utils")
        events = self.progress_events

        class FakeProgressBar:
            def __init__(self, total):
                events.append(("init", total))

            def update_absolute(self, step):
                events.append(("step", step))

            def update(self, amount):
                events.append(("update", amount))
        utils.ProgressBar = FakeProgressBar
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        comfy.model_management, comfy.model_patcher, comfy.utils = management, patcher, utils

        modules = {
            "folder_paths": paths,
            "comfy": comfy,
            "comfy.model_management": management,
            "comfy.model_patcher": patcher,
            "comfy.utils": utils,
            "comfy_api": pkg,
            "comfy_api.latest": latest,
        }
        prior = {name: sys.modules.get(name) for name in modules}
        sys.modules.update(modules)

        def restore():
            sys.modules.pop("starvector_node.nodes", None)
            for key, previous in prior.items():
                if previous is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = previous

        self.addCleanup(restore)
        # A fresh import registers the model folder with this test's folder_paths.
        sys.modules.pop("starvector_node.nodes", None)
        self.nodes = importlib.import_module("starvector_node.nodes")
        # Refine is on by default; its own tests cover it, these check the node around it.
        self.nodes.refine_svg = lambda markup, *args: markup
        self.text = '<svg xmlns="http://www.w3.org/2000/svg"><circle r="5"/></svg>'
        self.svg = NativeSVG([BytesIO(self.text.encode("utf-8"))])

    def test_model_paths_use_comfyui_registry_and_preserve_extra_config(self):
        roots, extensions = self.paths.folder_names_and_paths["starvector"]
        self.assertEqual(roots, [
            self.configured_directory,
            str(Path(self.paths.models_dir) / "starvector"),
        ])
        self.assertIn(".gguf", extensions)

    def test_loader_resolves_path_and_rejects_unknown_architecture(self):
        folder = Path(self.configured_directory)
        folder.mkdir(parents=True, exist_ok=True)
        model = folder / "model.gguf"
        model.touch()
        self.paths.get_full_path = lambda category, name: str(model)
        ref = types.SimpleNamespace(path=model, tensor_index={"model.svg_transformer.transformer.lm_head.weight": object()})
        with patch.object(self.nodes, "inspect_local_gguf", return_value=ref) as inspect:
            with self.assertRaisesRegex(ValueError, "not a StarVector"):
                self.nodes.StarVectorGGUFLoader().load("model.gguf", device="auto", precision="auto")
        inspect.assert_called_once_with(model)

    def test_exactly_four_public_nodes(self):
        nodes = self.nodes.NODE_CLASS_MAPPINGS
        self.assertEqual(set(nodes), {
            "StarVectorGGUFLoader",
            "StarVectorGGUFImageToSVG",
            "StarVectorGGUFTextPreview",
            "StarVectorGGUFPreviewSVG",
        })
        self.assertEqual(nodes["StarVectorGGUFLoader"].RETURN_TYPES, ("STARVECTOR",))
        self.assertEqual(nodes["StarVectorGGUFLoader"].RETURN_NAMES, ("starvector",))
        converter = nodes["StarVectorGGUFImageToSVG"]
        self.assertEqual(converter.RETURN_TYPES, ("SVG",))
        self.assertEqual(converter.INPUT_TYPES()["required"]["starvector"], ("STARVECTOR",))
        required = converter.INPUT_TYPES()["required"]
        self.assertEqual(required["max_new_tokens"][1]["default"], 4096)
        self.assertEqual(required["temperature"][1]["default"], 1.5)
        self.assertGreater(required["temperature"][1]["min"], 0)
        self.assertEqual(required["top_p"][1]["default"], 0.9)
        self.assertEqual(required["num_beams"][0], "INT")
        self.assertEqual(required["num_beams"][1]["default"], 2)
        self.assertEqual(required["num_beams"][1]["min"], 1)
        self.assertEqual(required["num_beams"][1]["max"], 8)
        self.assertEqual(required["loop_guard"][0], "BOOLEAN")
        self.assertTrue(required["loop_guard"][1]["default"])
        self.assertNotIn("parallel_processing", required)

    def _make_converter(self):
        class Engine:
            def __init__(self):
                self.image_batches = []
                self.generated = []
            def prepare_image_prefix(self, image):
                self.image_batches.append(image.shape)
                return image[:, 0:1, 0:1, 0:1].reshape(1, 1, 1)
            def generate_from_prefix(self, prefix, max_new_tokens, temperature, top_p, seed, num_beams=2, loop_guard=True, progress_callback=None):
                self.generated.append((prefix.device.type, max_new_tokens, seed, num_beams, loop_guard))
                progress_callback(1)
                progress_callback(2)
                value = int(prefix.reshape(-1)[0].item())
                return '<svg data-image="' + str(value) + '"/>'

        engine = Engine()
        engine.progress_events = self.progress_events
        loader = {
            "instance": engine,
            "vision_patcher": "vision",
            "decoder_patcher": "decoder",
        }
        return self.nodes.StarVectorGGUFImageToSVG(), loader, engine, self.calls

    def test_native_converter_output(self):
        node, loader, engine, calls = self._make_converter()
        image = torch.ones((1, 2, 2, 3), dtype=torch.float32)
        native = node.generate(loader, image=image, max_new_tokens=32, refine_steps=0)[0]
        self.assertIsInstance(native, NativeSVG)
        self.assertEqual(len(native.data), 1)
        self.assertIsInstance(native.data[0], BytesIO)
        self.assertEqual(native.data[0].getvalue(), b'<svg data-image="1"/>')
        self.assertEqual(engine.image_batches, [(1, 2, 2, 3)])
        self.assertEqual(engine.generated, [("cpu", 32, 0, 2, True)])
        # One unit for the encoded image, then 32 for its tokens.
        self.assertEqual(engine.progress_events, [
            ("init", 33), ("step", 0), ("step", 1), ("step", 2), ("step", 3), ("step", 33)
        ])
        self.assertEqual(calls, [(["vision"], True), (["decoder"], True)])

    def test_image_batch_generates_ordered_svg_batch(self):
        node, loader, engine, calls = self._make_converter()
        images = torch.zeros((3, 2, 2, 3), dtype=torch.float32)
        images[0] = 1
        images[1] = 2
        images[2] = 3
        native = node.generate(loader, image=images, max_new_tokens=256, seed=42, num_beams=8, loop_guard=False, refine_steps=0)[0]
        self.assertIsInstance(native, NativeSVG)
        self.assertEqual(len(native.data), 3)
        self.assertEqual([stream.getvalue().decode("utf-8") for stream in native.data], [
            '<svg data-image="1"/>', '<svg data-image="2"/>', '<svg data-image="3"/>'
        ])
        self.assertEqual(engine.image_batches, [(1, 2, 2, 3)] * 3)
        self.assertEqual(engine.generated, [("cpu", 256, 42, 8, False)] * 3)
        self.assertEqual(engine.progress_events, [
            ("init", 771), ("step", 0),
            ("step", 1), ("step", 2), ("step", 3),
            ("step", 4), ("step", 5), ("step", 259),
            ("step", 260), ("step", 261), ("step", 515),
            ("step", 516), ("step", 517), ("step", 771),
        ])
        # Vision stage is loaded once for the whole batch, then decoder once.
        self.assertEqual(calls, [(["vision"], True), (["decoder"], True)])

    def test_returns_empty_svg_and_saves_raw_markup_when_recovery_fails(self):
        node, loader, engine, calls = self._make_converter()
        raw = '<svg><path d="M0 0"'
        engine.generate_from_prefix = lambda *args, **kwargs: raw
        with self.assertLogs(level="ERROR") as captured:
            native = node.generate(loader, image=torch.ones((1, 2, 2, 3)), max_new_tokens=32, on_failure="recover")[0]
        self.assertEqual(native.data[0].getvalue(), b'<svg></svg>')
        saved = list(Path(self.paths.get_temp_directory()).glob("starvector-invalid-*.svg.txt"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_text(encoding="utf-8"), raw)
        self.assertTrue(any("returning EMPTY SVG" in msg for msg in captured.output))
        self.assertEqual(calls, [(["vision"], True), (["decoder"], True)])

    def test_repairs_truncated_svg_and_keeps_original_for_diagnosis(self):
        node, loader, engine, calls = self._make_converter()
        raw = '<svg viewBox="0 0 100 100"><g><path d="M 0 0 L 10 10"/></g>'
        engine.generate_from_prefix = lambda *args, **kwargs: raw
        with self.assertLogs(level="WARNING") as captured:
            native = node.generate(loader, image=torch.ones((1, 2, 2, 3)), max_new_tokens=32, on_failure="recover")[0]
        from xml.etree import ElementTree
        markup = native.data[0].getvalue().decode("utf-8")
        root = ElementTree.fromstring(markup)
        self.assertEqual(root.tag.rsplit("}", 1)[-1], "svg")
        self.assertTrue(any(tag.tag.rsplit("}", 1)[-1] == "path" for tag in root.iter()))
        saved = list(Path(self.paths.get_temp_directory()).glob("starvector-invalid-*.svg.txt"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_text(encoding="utf-8"), raw)
        self.assertTrue(any("recovered malformed SVG" in msg for msg in captured.output))

    def test_incomplete_svg_raises_with_none_and_keeps_original(self):
        node, loader, engine, calls = self._make_converter()
        raw = '<svg viewBox="0 0 10 10"><path d="M0 0'
        engine.generate_from_prefix = lambda *args, **kwargs: raw
        self.assertEqual(node.INPUT_TYPES()["required"]["on_failure"][0], ["none", "recover", "vtracer"])
        self.assertEqual(node.INPUT_TYPES()["required"]["on_failure"][1]["default"], "vtracer")
        with self.assertRaisesRegex(RuntimeError, "did not complete SVG 1/1.*on_failure"):
            node.generate(loader, image=torch.ones((1, 2, 2, 3)), max_new_tokens=32, on_failure="none")
        saved = list(Path(self.paths.get_temp_directory()).glob("starvector-invalid-*.svg.txt"))
        self.assertEqual([path.read_text(encoding="utf-8") for path in saved], [raw])

    def test_incomplete_svg_traced_with_vtracer(self):
        node, loader, engine, calls = self._make_converter()
        engine.generate_from_prefix = lambda *args, **kwargs: '<svg><path d="M0 0'
        image = torch.ones((1, 16, 16, 3))
        image[0, 4:12, 4:12] = torch.tensor([0.8, 0.1, 0.1])
        with self.assertLogs(level="WARNING") as captured:
            native = node.generate(loader, image=image, max_new_tokens=32, on_failure="vtracer")[0]
        from xml.etree import ElementTree
        root = ElementTree.fromstring(native.data[0].getvalue())
        self.assertEqual(root.tag.rsplit("}", 1)[-1], "svg")
        self.assertTrue(any(tag.tag.rsplit("}", 1)[-1] == "path" for tag in root.iter()))
        self.assertTrue(any("traced the input with VTracer" in msg for msg in captured.output))

    def test_refine_applies_to_starvector_and_vtracer_svg_only(self):
        node, loader, engine, calls = self._make_converter()
        required = node.INPUT_TYPES()["required"]
        self.assertEqual(required["refine_steps"][1]["default"], 100)
        self.assertEqual(required["refine_paths"][0], "BOOLEAN")
        self.assertIs(required["refine_paths"][1]["default"], True)
        seen = []

        def fake_refine(markup, image, steps, new_paths, progress):
            progress()
            seen.append((markup[:4], tuple(image.shape), steps, new_paths))
            return "<svg refined=''/>"

        image = torch.ones((1, 16, 16, 3))
        image[0, 4:12, 4:12] = torch.tensor([0.8, 0.1, 0.1])
        with patch.object(self.nodes, "refine_svg", fake_refine):
            native = node.generate(loader, image=image, max_new_tokens=32, refine_steps=5, refine_paths=True)[0]
            self.assertEqual(native.data[0].getvalue(), b"<svg refined=''/>")
            self.assertEqual(engine.progress_events[:2], [("init", 1 + 32 + 5 * 17), ("step", 0)])
            # Refine starts at the end of the token part, however early the SVG ended.
            refine_start = engine.progress_events.index(("step", 1 + 32))
            self.assertEqual(engine.progress_events[refine_start + 1], ("update", 1))
            engine.generate_from_prefix = lambda *args, **kwargs: '<svg><path d="M0 0'
            with self.assertLogs(level="WARNING"):
                node.generate(loader, image=image, max_new_tokens=32, on_failure="vtracer", refine_steps=5, refine_paths=False)
                # A repaired partial SVG is not refined.
                node.generate(loader, image=image, max_new_tokens=32, on_failure="recover", refine_steps=5)
        self.assertEqual(seen, [("<svg", (16, 16, 3), 5, 16), ("<?xm", (16, 16, 3), 5, 0)])

    def test_vtracer_fallback_is_compact_only_before_refine(self):
        node, loader, engine, calls = self._make_converter()
        engine.generate_from_prefix = lambda *args, **kwargs: '<svg><path d="M0 0'
        traced = []
        with patch.object(self.nodes, "trace_svg", lambda pixels, compact: traced.append(compact) or "<svg/>"), \
                patch.object(self.nodes, "refine_svg", lambda markup, *args: markup), self.assertLogs(level="WARNING"):
            node.generate(loader, image=torch.ones((1, 2, 2, 3)), max_new_tokens=32, on_failure="vtracer", refine_steps=0)
            node.generate(loader, image=torch.ones((1, 2, 2, 3)), max_new_tokens=32, on_failure="vtracer", refine_steps=5)
        self.assertEqual(traced, [False, True])

    def test_image_batch_rejects_empty_input(self):
        node, loader, engine, calls = self._make_converter()
        with self.assertRaisesRegex(ValueError, "non-empty"):
            node.generate(loader, image=torch.zeros((0, 2, 2, 3)), max_new_tokens=10)
        self.assertEqual(calls, [])

    def test_text_preview_passes_same_object_and_does_not_seek(self):
        original = self.svg.data[0]
        original.seek(9)
        response = self.nodes.StarVectorGGUFTextPreview().preview(self.svg)
        self.assertIs(response["result"][0], self.svg)
        self.assertEqual(response["ui"]["svg"], [self.text])
        self.assertEqual(original.tell(), 9)

    def test_graphic_preview_preserves_native_svg(self):
        node = self.nodes.StarVectorGGUFPreviewSVG()
        params = node.INPUT_TYPES()["required"]
        self.assertEqual(set(params), {"svg", "background"})
        self.assertEqual(params["background"][1]["default"], "white")
        original = self.svg.data[0]
        original.seek(4)
        response = node.preview(self.svg, "transparent")
        self.assertIs(response["result"][0], self.svg)
        self.assertEqual(response["ui"]["svg"], [self.text])
        self.assertNotIn("width", response["ui"])
        self.assertNotIn("height", response["ui"])
        self.assertEqual(response["ui"]["background"], ["transparent"])
        self.assertEqual(original.tell(), 4)

    def test_batch_of_native_svg_documents(self):
        self.svg.data.append(BytesIO(b"<svg/>"))
        response = self.nodes.StarVectorGGUFTextPreview().preview(self.svg)
        self.assertEqual(response["ui"]["svg"], [self.text, "<svg/>"])
        response = self.nodes.StarVectorGGUFPreviewSVG().preview(self.svg)
        self.assertEqual(response["ui"]["svg"], [self.text, "<svg/>"])

    def test_reject_non_native_svg(self):
        with self.assertRaisesRegex(TypeError, "native SVG"):
            self.nodes.StarVectorGGUFTextPreview().preview({"svg_string": self.text})


if __name__ == "__main__":
    unittest.main()
