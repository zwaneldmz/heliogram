"""heliogram -- optical context compression for self-hosted VLMs.

Measures whether one ViT patch (~1 vision token) can reliably carry more than one text token's
worth of data (~6 bits for base64), by defining a patch-aligned symbolic codec, a corruption
suite mirroring real serving pipelines, and a model-free evaluation harness.

Public API (see spec/format-v0.1.md for the exact pinned format):

    encode(data, palette=8, patch_size=14, nsym=32, seed=0) -> PIL.Image
    decode_pixels(img, palette=8, patch_size=14, nsym=32) -> bytes   # reference, no model
    decode(img, ..., decoder=None) -> bytes                          # decoder plug point
    get_palette(palette) -> list[(r, g, b)]                         # deterministic palette
    VLMDecoder                                                       # Phase-2 stub, raises NotImplementedError

Phase-2 (GPU) scaffold -- see the README's "Phase 2 (GPU)" section:

    write_dataset(...) / generate_examples(...)   # heliogram.dataset: synthetic training pairs,
                                                    # CPU-only, no model required
    QwenVLDecoder(model=..., processor=...)        # heliogram.vlm: decoder plug point for a
                                                    # fine-tuned VLM; raises without a real model
    zero_shot_symbol_error(model, processor, ...)  # heliogram.vlm: measures a STOCK model's
                                                    # symbol error -- only with a real model

DATA HONESTY: nothing above the Phase-2 line has been run against a real model in this repo (no
GPU here). `import heliogram` never requires torch/transformers/peft/bitsandbytes -- see
heliogram/vlm.py's module docstring for the lazy-import boundary.

Apache-2.0. All data used by this project is synthetic and seed-deterministic.
"""

from .codec import (
    CODEC_VERSION,
    PATCH_SIZE,
    VALID_PALETTES,
    HeliogramDecodeError,
    VLMDecoder,
    decode,
    decode_pixels,
    encode,
    get_palette,
)
from .dataset import (
    Example,
    generate_examples,
    iter_manifest,
    symbols_to_target,
    target_to_symbols,
    write_dataset,
)
# Gate-independent instruments (the durable value -- see README "Instruments") and the
# patch-size verifier (makes codec.PATCH_SIZE auditable against a real model config). All of
# these keep torch/transformers lazy exactly like heliogram.vlm, so importing them here does
# NOT break the "import heliogram never pulls in torch" invariant -- FrozenEncoderHandle's
# torch use is local to its methods, and heliogram.instruments' model paths raise without a
# real model rather than importing one. The `instruments` subpackage is re-exported so
# `import heliogram; heliogram.instruments.foreign_tile.guard(...)` resolves.
from . import instruments
from .encoder import FrozenEncoderHandle
from .patchsize import (
    KNOWN_PATCH_SIZES,
    PatchSizeReport,
    known_patch_size,
    verify_patch_size,
)

try:
    # heliogram.vlm's own top-level imports are as light as heliogram.codec's (no
    # torch/transformers at module scope -- those are lazy inside its methods/functions), so
    # this should always succeed. It is still wrapped in a try/except, per the Phase-2 scaffold
    # contract: `import heliogram` must never require torch/transformers/peft/bitsandbytes, even
    # if heliogram.vlm's import surface grows to need a guard in the future.
    from .vlm import QwenVLDecoder, zero_shot_symbol_error
except ImportError:  # pragma: no cover
    QwenVLDecoder = None  # type: ignore[assignment]
    zero_shot_symbol_error = None  # type: ignore[assignment]

__all__ = [
    "CODEC_VERSION",
    "PATCH_SIZE",
    "VALID_PALETTES",
    "HeliogramDecodeError",
    "VLMDecoder",
    "decode",
    "decode_pixels",
    "encode",
    "get_palette",
    "Example",
    "generate_examples",
    "iter_manifest",
    "symbols_to_target",
    "target_to_symbols",
    "write_dataset",
    "QwenVLDecoder",
    "zero_shot_symbol_error",
    "instruments",
    "FrozenEncoderHandle",
    "KNOWN_PATCH_SIZES",
    "PatchSizeReport",
    "known_patch_size",
    "verify_patch_size",
]

__version__ = "0.1.0"
