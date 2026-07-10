"""heliogram.baselines -- reference points to compare the codec's bits/patch against.

(a) base64_bits_per_token: the "cost" of shipping bytes as base64 text tokens (~6 bits/token
    analytically; accepts a real tokenizer for a measured value).
(b) rendered_text_density: a model-free, geometric estimate of how densely typeset text packs
    into the SAME patch grid unit the codec uses. The *true* bits/patch for rendered text needs
    actual OCR by an un-fine-tuned VLM -- that is Phase 2 work and is not done here.
"""

from __future__ import annotations

import base64
import math
import random
from dataclasses import dataclass
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "Base64Baseline",
    "base64_bits_per_token",
    "RenderedTextDensity",
    "rendered_text_density",
]


@dataclass
class Base64Baseline:
    bits_per_token: float
    note: str


def base64_bits_per_token(
    tokenizer: object = None,
    sample_bytes: Optional[bytes] = None,
    seed: int = 0,
) -> Base64Baseline:
    """~6 bits/token analytic baseline for base64 text, or a real measurement if given a tokenizer.

    base64 uses a 64-symbol alphabet (log2(64) = 6 bits/char), and common tokenizers emit
    roughly one token per base64 character, so ~6 bits/token is the standard analytic estimate.

    If `tokenizer` is given (any object exposing `.encode(str) -> Sequence[int]`, e.g. a
    HuggingFace tokenizer, or a plain callable `str -> Sequence[int]`), this instead *measures*
    bits/token directly: it base64-encodes `sample_bytes` (4096 deterministic pseudo-random bytes
    by default, seeded by `seed`) and divides the original bit count by the resulting token count.
    """
    if sample_bytes is None:
        rng = random.Random(seed)
        sample_bytes = bytes(rng.getrandbits(8) for _ in range(4096))

    if tokenizer is None:
        return Base64Baseline(
            bits_per_token=6.0,
            note=(
                "analytic: base64 alphabet size 64 -> log2(64)=6 bits/char; ~1 char/token for "
                "typical BPE tokenizers on base64 streams. Pass a real tokenizer for a measured "
                "value."
            ),
        )

    b64_text = base64.b64encode(sample_bytes).decode("ascii")
    token_ids = tokenizer.encode(b64_text) if hasattr(tokenizer, "encode") else tokenizer(b64_text)
    n_tokens = len(token_ids)
    if n_tokens == 0:
        raise ValueError("tokenizer produced zero tokens for the base64 sample")
    bits = len(sample_bytes) * 8
    return Base64Baseline(
        bits_per_token=bits / n_tokens,
        note=(
            f"measured: {n_tokens} tokens for {len(sample_bytes)} bytes ({bits} bits) of "
            "base64 text via the provided tokenizer"
        ),
    )


@dataclass
class RenderedTextDensity:
    image: Image.Image
    patches_used: int
    chars_per_patch: float
    bits_per_patch: float
    text_len: int
    note: str


def rendered_text_density(payload: bytes, patch_size: int = 14) -> RenderedTextDensity:
    """Typeset `payload` (base64-encoded, monospace) into an image using the SAME patch-size
    unit the codec uses, and measure a purely GEOMETRIC (model-free) density: how many patches
    of typeset text it takes to hold the payload, and the bits/patch that implies if every
    character were perfectly legible.

    This does NOT run OCR and is not a measurement of what an actual VLM can read off the image
    -- that requires the un-fine-tuned VLM's OCR accuracy, which is Phase 2 work (out of scope
    here, no GPU). Treat bits_per_patch here as an upper-bound / packing-density baseline only.
    """
    text = base64.b64encode(payload).decode("ascii")
    font = ImageFont.load_default()
    bbox = font.getbbox("M")
    char_w = max(1, bbox[2] - bbox[0])
    char_h = max(1, bbox[3] - bbox[1])

    # square-ish canvas, sized in whole patches, that roughly fits len(text) characters
    target_w_px = max(
        patch_size,
        math.ceil(math.sqrt(len(text)) * char_w / patch_size) * patch_size,
    )
    chars_per_line = max(1, target_w_px // char_w)
    n_lines = max(1, math.ceil(len(text) / chars_per_line))
    target_h_px = max(patch_size, math.ceil(n_lines * char_h / patch_size) * patch_size)

    img = Image.new("RGB", (target_w_px, target_h_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for line_idx in range(n_lines):
        line = text[line_idx * chars_per_line : (line_idx + 1) * chars_per_line]
        draw.text((0, line_idx * char_h), line, fill=(0, 0, 0), font=font)

    patches_w = target_w_px // patch_size
    patches_h = target_h_px // patch_size
    patches_used = patches_w * patches_h
    chars_per_patch = len(text) / patches_used
    bits_per_patch = chars_per_patch * 6.0  # base64: 6 bits/char

    return RenderedTextDensity(
        image=img,
        patches_used=patches_used,
        chars_per_patch=chars_per_patch,
        bits_per_patch=bits_per_patch,
        text_len=len(text),
        note=(
            "geometric/model-free: measures typeset packing density only, assumes perfect "
            "legibility. Real bits/patch for rendered text needs OCR accuracy from an "
            "un-fine-tuned VLM (Phase 2, out of scope here)."
        ),
    )
