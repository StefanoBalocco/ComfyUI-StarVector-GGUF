"""Register the custom node folder as package `starvector_node` for the tests.

The modules use package-relative imports; tests import them as
`starvector_node.<module>` without a ComfyUI installation. Also holds the
fakes shared by the tests: GGUF tensors and the ComfyUI attention module.
"""
import sys
import types
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if "starvector_node" not in sys.modules:
    package = types.ModuleType("starvector_node")
    package.__path__ = [str(ROOT)]
    sys.modules["starvector_node"] = package


class GGUFType(Enum):
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q5_0 = 6
    Q8_0 = 8


def fake_tensor(name, data=None, tensor_type=GGUFType.F32, shape=None):
    """GGUF tensor as the reader returns it: `shape` (default: data.shape) is stored reversed."""
    if shape is None:
        shape = () if data is None else data.shape
    return SimpleNamespace(name=name, tensor_type=tensor_type, shape=tuple(reversed(shape)), data=data)


@contextmanager
def fake_comfy_attention(calls=None):
    """Context replacing comfy.ldm.modules.attention with a torch SDPA wrapper.

    Only this sys.modules entry is swapped: restoring a whole copy of sys.modules would also
    drop extension modules imported meanwhile (numpy.fft, torch caches), which cannot load twice.

    Each call appends (heads, mask, low_precision_attention) to `calls` if given.
    """
    backend = types.ModuleType("comfy.ldm.modules.attention")

    def optimized(q, k, v, heads, mask=None, **options):
        if calls is not None:
            calls.append((heads, mask, options.get("low_precision_attention")))
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=options.get("scale"),
            enable_gqa=options.get("enable_gqa", False),
        )
    backend.optimized_attention = optimized
    previous = sys.modules.get(backend.__name__)
    sys.modules[backend.__name__] = backend
    try:
        yield
    finally:
        if previous is None:
            del sys.modules[backend.__name__]
        else:
            sys.modules[backend.__name__] = previous
