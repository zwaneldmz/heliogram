"""heliogram.corruption -- composable, model-free image corruptions for robustness testing.

Every function takes a PIL.Image and returns a PIL.Image of the SAME size, so a decoder that
assumes a fixed patch grid can be pointed directly at the result, and so corruptions compose.

ONE DOCUMENTED EXCEPTION to the same-size contract: `qwen_smart_resize` below. It emulates the
target model's OWN image preprocessing (Qwen2/2.5-VL's `smart_resize`), whose entire effect is
that the model sees a DIFFERENT-SIZED image than the one you encoded -- resizing back to the
original size would erase exactly the corruption being measured. Downstream measurement code
(`heliogram.harness._run_cell`) reads the corrupted image at whatever size the corruption
returned, the same way the model would.
"""

from __future__ import annotations

import io
import math
from typing import Callable, Sequence, Tuple, Union

from PIL import Image

__all__ = [
    "resize_roundtrip",
    "jpeg_compress",
    "crop_pad",
    "compose",
    "QWEN_PATCH_FACTOR",
    "QWEN_DEFAULT_MIN_PIXELS",
    "QWEN_DEFAULT_MAX_PIXELS",
    "QWEN_GENEROUS_MAX_PIXELS",
    "qwen_smart_resize_dims",
    "qwen_smart_resize",
]


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


# Qwen2-VL / Qwen2.5-VL preprocessing constants, mirrored from
# transformers.models.qwen2_vl.image_processing_qwen2_vl (verified against transformers 5.13.0;
# see tests/test_smart_resize.py, which compares qwen_smart_resize_dims against the installed
# transformers implementation directly whenever transformers+torchvision are importable).
QWEN_PATCH_FACTOR = 28  # ViT patch_size (14) * spatial merge_size (2): the snap unit
QWEN_DEFAULT_MIN_PIXELS = 56 * 56  # 3,136 -- transformers' smart_resize signature default
QWEN_DEFAULT_MAX_PIXELS = 14 * 14 * 4 * 1280  # 1,003,520 -- ditto (== processor longest_edge)
# "Generous" bound for the operator-controlled case: large enough that no heliogram grid in this
# project's sweep (largest: ~29.5M px at palette=2/16KB) is EVER above it... is what we'd like,
# but Qwen's ViT cost scales with pixel count, so a realistic operator ceiling matters more than
# a vacuous one. 16M px matches scripts/run_probe.py's identity bound and comfortably covers
# every palette>=8 grid in the sweep; the palette=2/4 16KB grids exceed even this (they are
# huge because 1 bit/patch needs many patches) and are downscaled -- reported as measured.
QWEN_GENEROUS_MAX_PIXELS = 16_000_000


def qwen_smart_resize_dims(
    height: int,
    width: int,
    factor: int = QWEN_PATCH_FACTOR,
    min_pixels: int = QWEN_DEFAULT_MIN_PIXELS,
    max_pixels: int = QWEN_DEFAULT_MAX_PIXELS,
) -> Tuple[int, int]:
    """Exact mirror of `transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize`
    (transformers 5.13.0): returns the (height, width) Qwen2/2.5-VL's image processor will
    actually feed the vision tower. Both outputs are multiples of `factor`; the total pixel
    count is clamped into [min_pixels, max_pixels]; aspect ratio is approximately preserved.

    Mirrored (a ~15-line pure function) rather than imported because importing the transformers
    implementation requires torch+torchvision -- far too heavy for this pillow/numpy-only module
    -- and the equivalence is pinned by tests/test_smart_resize.py whenever those packages ARE
    installed, so drift against a future transformers version is detectable, not silent.
    """
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            "absolute aspect ratio must be smaller than 200, got "
            f"{max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qwen_smart_resize(
    img: Image.Image,
    factor: int = QWEN_PATCH_FACTOR,
    min_pixels: int = QWEN_DEFAULT_MIN_PIXELS,
    max_pixels: int = QWEN_GENEROUS_MAX_PIXELS,
    resample: int = Image.BICUBIC,
) -> Image.Image:
    """Apply the resize Qwen2/2.5-VL's OWN image processor applies before the vision tower ever
    sees a pixel -- the one corruption an in-scope (self-hosted, controls-preprocessing) operator
    CANNOT opt out of: `smart_resize` snapping both pixel dimensions to multiples of `factor`
    (28 = 14px patch x 2x2 merge), plus the pixel-budget clamp.

    THE DOCUMENTED EXCEPTION to this module's same-size contract (see module docstring): the
    returned image is the size the MODEL sees, which is the whole point. Identity (returns `img`
    unchanged) exactly when both dimensions are already multiples of `factor` and the pixel
    count is inside [min_pixels, max_pixels] -- i.e. for every even-patch-grid image
    `heliogram.codec.encode(..., align=2)` (or `heliogram.dataset`'s generators, which use it)
    emits, provided the pixel budget is met.

    Two bound presets matter (see the constants above):
      - `max_pixels=QWEN_GENEROUS_MAX_PIXELS` (this function's default): the operator-controlled
        case -- min/max_pixels are processor-constructor arguments an in-scope operator can (and
        should) widen, as scripts/run_probe.py does, so what remains is the UNAVOIDABLE 28px
        snap. This is the honest "best case an operator can arrange".
      - `max_pixels=QWEN_DEFAULT_MAX_PIXELS` (1,003,520 px): what a NAIVE operator gets from the
        stock processor defaults -- any grid over ~1MP is downscaled wholesale. The harness
        measures this separately (`qwen_smart_resize_1mp` in heliogram.harness.CORRUPTIONS).

    `resample` defaults to bicubic, matching the transformers processor's default (resample=3).
    """
    h_bar, w_bar = qwen_smart_resize_dims(
        img.height, img.width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels
    )
    if (h_bar, w_bar) == (img.height, img.width):
        return img
    return img.resize((w_bar, h_bar), resample)


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
