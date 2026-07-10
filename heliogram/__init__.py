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
]

__version__ = "0.1.0"
