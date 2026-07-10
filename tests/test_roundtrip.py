"""Pytest suite for heliogram codec v0.1: clean roundtrip exactness, determinism, and calibration
recovery under a mild global color shift. Assert-based, no fixtures.
"""

import io
import json

import numpy as np
from PIL import Image

from heliogram import decode_pixels, encode, get_palette
from heliogram.codec import VALID_PALETTES

PAYLOADS = [
    b"",
    b"a",
    b"hello, heliogram!",
    json.dumps(
        {"id": 42, "name": "heliogram", "tags": ["vlm", "codec"], "active": True}
    ).encode("utf-8"),
    bytes(range(256)) * 2,  # 512 bytes; exercises reedsolo's internal chunking
]


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_clean_roundtrip_exact():
    for palette in VALID_PALETTES:
        for payload in PAYLOADS:
            img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
            recovered = decode_pixels(img, palette=palette, patch_size=14, nsym=32)
            assert recovered == payload, (
                f"roundtrip mismatch for palette={palette} payload_len={len(payload)}"
            )


def test_determinism_same_args_identical_png_bytes():
    payload = b"determinism check: heliogram v0.1"
    for palette in VALID_PALETTES:
        img1 = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        img2 = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        assert _png_bytes(img1) == _png_bytes(img2)


def test_determinism_different_payload_differs():
    img1 = encode(b"payload one", palette=8)
    img2 = encode(b"payload two", palette=8)
    assert _png_bytes(img1) != _png_bytes(img2)


def test_calibration_recovery_under_mild_global_color_shift():
    shift = np.array([6, -4, 5], dtype=np.int16)
    for palette in VALID_PALETTES:
        payload = f"calibration test palette={palette}".encode("utf-8")
        img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        arr = np.asarray(img, dtype=np.int16)
        shifted = np.clip(arr + shift, 0, 255).astype(np.uint8)
        shifted_img = Image.fromarray(shifted)
        recovered = decode_pixels(shifted_img, palette=palette, patch_size=14, nsym=32)
        assert recovered == payload, f"calibration recovery failed for palette={palette}"


def test_palette_deterministic_and_separable():
    for palette in VALID_PALETTES:
        colors_a = get_palette(palette)
        colors_b = get_palette(palette)
        assert colors_a == colors_b
        assert len(set(colors_a)) == palette  # all colors distinct
