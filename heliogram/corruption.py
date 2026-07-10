"""heliogram.corruption -- composable, model-free image corruptions for robustness testing.

Every function takes a PIL.Image and returns a PIL.Image of the SAME size, so a decoder that
assumes a fixed patch grid can be pointed directly at the result, and so corruptions compose.
"""

from __future__ import annotations

import io
from typing import Callable, Sequence, Tuple, Union

from PIL import Image

__all__ = ["resize_roundtrip", "jpeg_compress", "crop_pad", "compose"]


def resize_roundtrip(img: Image.Image, scale: float = 0.95) -> Image.Image:
    """Downscale by `scale` then upscale back to the original size, both bilinear.

    Simulates the resolution loss of an intermediate resize step in a real serving pipeline.
    `scale` in (0, 1]; e.g. 0.95 == a 5% shrink before restoring the original dimensions.
    """
    if not (0 < scale <= 1):
        raise ValueError("scale must be in (0, 1]")
    w, h = img.size
    small_w, small_h = max(1, round(w * scale)), max(1, round(h * scale))
    shrunk = img.resize((small_w, small_h), Image.BILINEAR)
    return shrunk.resize((w, h), Image.BILINEAR)


def jpeg_compress(img: Image.Image, quality: int = 85) -> Image.Image:
    """Re-encode through JPEG at `quality` (70-95 is the realistic serving range) and decode it
    back to a PIL image of the same size."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def crop_pad(
    img: Image.Image,
    dx: int = 2,
    dy: int = 2,
    fill: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Shift the canvas content by (dx, dy) px: crop that many px off the bottom/right edges and
    pad the top/left with `fill`, keeping the original image size (a slight misalignment).
    """
    w, h = img.size
    canvas = Image.new(img.mode, (w, h), fill)
    canvas.paste(img, (dx, dy))
    return canvas


Corruption = Union[Callable[[Image.Image], Image.Image], Tuple[Callable, dict]]


def compose(img: Image.Image, corruptions: Sequence[Corruption]) -> Image.Image:
    """Apply a sequence of corruptions to `img` in order. Each item is either a bare
    `fn(img) -> img` callable or a `(fn, kwargs)` tuple."""
    out = img
    for item in corruptions:
        if isinstance(item, tuple):
            fn, kwargs = item
            out = fn(out, **kwargs)
        else:
            out = item(out)
    return out
