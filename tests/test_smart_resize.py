"""Tests for heliogram.corruption's qwen_smart_resize -- the target model's own preprocessing,
mirrored as a corruption.

Three layers, in order of authority:
 1. Equivalence against the INSTALLED transformers implementation (skipped when
    transformers/torchvision aren't installed -- they are heavy GPU-path deps, not part of the
    base install): qwen_smart_resize_dims must agree with
    transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize on every probed input.
    This is the drift detector: if a future transformers changes smart_resize, this fails loudly
    instead of the mirror silently measuring a stale algorithm.
 2. Pinned behavioral facts that do NOT need transformers: identity on 28px-aligned images
    within the pixel budget, snap-resampling on odd-patch-dimension images, wholesale downscale
    above max_pixels.
 3. Harness integration: the two CORRUPTIONS entries exist, and a size-changed corrupted image
    is measured at its own size (no resize-back) -- pinned by measuring a real cell.
"""

from __future__ import annotations

import math

import pytest
from PIL import Image

from heliogram.codec import PATCH_SIZE, encode
from heliogram.corruption import (
    QWEN_DEFAULT_MAX_PIXELS,
    QWEN_DEFAULT_MIN_PIXELS,
    QWEN_GENEROUS_MAX_PIXELS,
    QWEN_PATCH_FACTOR,
    qwen_smart_resize,
    qwen_smart_resize_dims,
)


def _transformers_smart_resize():
    """The installed transformers implementation, or None if transformers/torchvision (its
    import-time deps) aren't installed in this environment."""
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    except Exception:
        return None
    return smart_resize


# A spread of (height, width) covering: 28-aligned grids (identity candidates), odd-patch
# dimensions (snap), tall-thin calibration-row-forced grids (palette=256 shapes), and
# above-budget grids (downscale). All within the aspect-ratio<200 bound.
_PROBE_DIMS = [
    (28, 28),
    (224, 224),
    (1050, 3584),   # 75x256 patches: odd height -- the palette=256/16KB sweep shape
    (280, 3584),    # 20x256 patches: exactly 1,003,520 px, the default budget boundary
    (294, 3598),    # both dims odd patch counts
    (5432, 5432),   # 388x388 patches (~29.5M px): above even the generous budget
    (2464, 2464),
    (28, 3584),     # 2x256 patches: the smallest palette=256 grid
    (98, 98),
    (56, 4088),     # aspect ratio 73, tall-thin but legal
]


@pytest.mark.parametrize("bounds", [
    (QWEN_DEFAULT_MIN_PIXELS, QWEN_DEFAULT_MAX_PIXELS),
    (QWEN_DEFAULT_MIN_PIXELS, QWEN_GENEROUS_MAX_PIXELS),
    (28 * 28, 16_000_000),
])
def test_dims_match_installed_transformers(bounds):
    reference = _transformers_smart_resize()
    if reference is None:
        pytest.skip("transformers (or its torch/torchvision deps) not installed")
    min_pixels, max_pixels = bounds
    for h, w in _PROBE_DIMS:
        ours = qwen_smart_resize_dims(h, w, min_pixels=min_pixels, max_pixels=max_pixels)
        theirs = reference(h, w, factor=QWEN_PATCH_FACTOR,
                           min_pixels=min_pixels, max_pixels=max_pixels)
        assert ours == tuple(theirs), (
            f"qwen_smart_resize_dims drifted from transformers' smart_resize at "
            f"(h={h}, w={w}, min={min_pixels}, max={max_pixels}): ours={ours}, "
            f"transformers={tuple(theirs)}"
        )


def test_identity_on_aligned_grid_within_budget():
    """An even-patch-grid image (both pixel dims multiples of 28) under the pixel budget passes
    through untouched -- the exact object, not just equal pixels (identity is what
    pad_to_even_patch_grid + generous bounds buys, per the README's scope argument)."""
    img = Image.new("RGB", (3584, 280), (10, 20, 30))  # 256x20 patches, exactly 1,003,520 px
    out = qwen_smart_resize(img, max_pixels=QWEN_DEFAULT_MAX_PIXELS)
    assert out is img  # 1,003,520 == max_pixels: not ABOVE the budget, so identity


def test_snap_on_odd_patch_dimension():
    """A grid with an odd patch-count dimension (pixel dim an odd multiple of 14) is resampled
    to the nearest 28px multiple -- the mandatory snap no operator setting avoids."""
    img = Image.new("RGB", (3584, 1050), (10, 20, 30))  # 256x75 patches; 1050 = 75*14, odd rows
    out = qwen_smart_resize(img, max_pixels=QWEN_GENEROUS_MAX_PIXELS)
    assert out.size != img.size
    assert out.width % QWEN_PATCH_FACTOR == 0 and out.height % QWEN_PATCH_FACTOR == 0
    assert out.size == (3584, 1064)  # round(1050/28)=38 -> 1064: one full patch-row taller


def test_downscale_above_max_pixels():
    """Above the pixel budget the whole grid is downscaled -- both dims shrink, stay 28-aligned,
    and the result is under budget."""
    img = Image.new("RGB", (5432, 5432), (10, 20, 30))  # 388x388 patches, ~29.5M px
    out = qwen_smart_resize(img, max_pixels=QWEN_DEFAULT_MAX_PIXELS)
    assert out.width < img.width and out.height < img.height
    assert out.width % QWEN_PATCH_FACTOR == 0 and out.height % QWEN_PATCH_FACTOR == 0
    assert out.width * out.height <= QWEN_DEFAULT_MAX_PIXELS


def test_aspect_ratio_guard_matches_transformers():
    with pytest.raises(ValueError, match="aspect ratio"):
        qwen_smart_resize_dims(28, 28 * 250)


def test_harness_has_both_smart_resize_entries():
    from heliogram.harness import CORRUPTIONS

    assert "qwen_smart_resize" in CORRUPTIONS
    assert "qwen_smart_resize_1mp" in CORRUPTIONS


def test_harness_measures_size_changed_image_without_resize_back():
    """The palette=8 default-payload grid is 16x16 patches (224x224 px, 28-aligned, tiny) -- so
    qwen_smart_resize is the identity there and decode must SUCCEED, while an artificial
    odd-height grid must FAIL through the same _run_cell path (no resize-back hiding it)."""
    from heliogram import harness

    # Identity case: aligned grid -> smart_resize is a no-op -> decode succeeds.
    cell = harness._run_cell(
        palette=8,
        corruption_name="qwen_smart_resize",
        corruption_fn=harness.CORRUPTIONS["qwen_smart_resize"],
        n_trials=1,
        payload_size=48,
        subpatch=1,
    )
    assert cell.decode_success_rate == 1.0

    # Misaligned case: 75-row palette=256/16KB-shaped grid via a raw encode -- the snap moves
    # every data row off the 14px lattice the symbols were painted on. Payload must be RANDOM:
    # a constant payload paints near-uniform data rows that misalignment cannot corrupt (verified
    # -- an all-zeros 16KB payload survives this exact resample with 0 symbol errors).
    import random

    rng = random.Random(0)
    img = encode(bytes(rng.getrandbits(8) for _ in range(16384)), palette=256, nsym=32)
    assert (img.height // PATCH_SIZE) % 2 == 1  # precondition: odd patch rows (75)
    resized = harness.CORRUPTIONS["qwen_smart_resize"](img)
    assert resized.size != img.size  # the corruption really changed the size

    from heliogram.codec import HeliogramDecodeError, decode_pixels

    with pytest.raises(HeliogramDecodeError):
        decode_pixels(resized, palette=256, nsym=32)


def test_align2_encode_is_smart_resize_proof_where_unaligned_fails():
    """encode(align=2) is the encode-side fix for the mandatory 28px snap: the exact payload
    that fails decode after qwen_smart_resize when encoded normally (odd patch rows) must
    round-trip BIT-EXACTLY through the same corruption when encoded with align=2 -- and with
    no decoder flag, since an aligned grid is an ordinary v0.1 grid with more padding."""
    import random

    from heliogram.codec import HeliogramDecodeError, decode_pixels

    rng = random.Random(0)
    payload = bytes(rng.getrandbits(8) for _ in range(16384))

    unaligned = encode(payload, palette=256, nsym=32)
    assert (unaligned.height // PATCH_SIZE) % 2 == 1  # the hazard case: odd patch rows
    with pytest.raises(HeliogramDecodeError):
        decode_pixels(qwen_smart_resize(unaligned), palette=256, nsym=32)

    aligned = encode(payload, palette=256, nsym=32, align=2)
    assert aligned.width % (2 * PATCH_SIZE) == 0 and aligned.height % (2 * PATCH_SIZE) == 0
    resized = qwen_smart_resize(aligned)
    assert resized is aligned  # identity: smart_resize has nothing to snap
    assert decode_pixels(resized, palette=256, nsym=32) == payload


def test_align_default_is_byte_identical_to_prior_releases():
    """align=1 (the default) must not perturb a single output byte -- the pinned-hash
    determinism tests elsewhere cover the default path; this pins that passing align=1
    explicitly is the same thing, and that an ALIGNED image still decodes with no flag."""
    payload = bytes(range(48))
    assert (
        encode(payload, palette=8).tobytes()
        == encode(payload, palette=8, align=1).tobytes()
    )
    from heliogram.codec import decode_pixels

    aligned = encode(payload, palette=8, align=2)
    assert decode_pixels(aligned, palette=8) == payload


def test_align_rejects_invalid():
    from heliogram.codec import compute_grid

    with pytest.raises(ValueError, match="align"):
        compute_grid(100, 8, align=0)


def test_generous_budget_covers_every_palette_ge8_sweep_grid():
    """The docstring claim 'QWEN_GENEROUS_MAX_PIXELS comfortably covers every palette>=8 grid in
    the sweep' is a checkable fact about grid math -- check it, so the constant can't silently
    rot if SWEEP_PAYLOAD_SIZES grows."""
    from heliogram.harness import NSYM, SWEEP_PAYLOAD_SIZES, SUBPATCHES, _grid_stats

    for palette in (8, 16, 32, 64, 128, 256):
        for subpatch in SUBPATCHES:
            for payload_size in SWEEP_PAYLOAD_SIZES:
                g = _grid_stats(payload_size, palette, NSYM, subpatch)
                pixels = (g.width * PATCH_SIZE) * (g.height * PATCH_SIZE)
                assert pixels <= QWEN_GENEROUS_MAX_PIXELS, (
                    f"palette={palette}, subpatch={subpatch}, payload={payload_size}: "
                    f"{pixels} px exceeds QWEN_GENEROUS_MAX_PIXELS"
                )
