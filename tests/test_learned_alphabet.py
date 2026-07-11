"""Pytest suite for heliogram.instruments.learned_alphabet (CPU, model-free palette search) and
heliogram.encoder (the lazy-GPU frozen-encoder plug point) -- handoff M4/A12.

Assert-based, no fixtures/frameworks beyond plain pytest.raises/subprocess/capsys, matching the
rest of this repo's test idiom. Like tests/test_phase2_scaffold.py and tests/test_patchsize.py,
part of this file's job is proving the CPU-only import boundary holds for BOTH new modules:
neither heliogram.encoder nor heliogram.instruments.learned_alphabet may pull in torch/
transformers merely by being imported.

Search-based tests below intentionally use small `iters`/`palette_size` combinations to keep
this file's wall-clock cost low (each `optimize_palette` call costs roughly
`(iters * 4 + 1) * len(corruptions)` small encode/corrupt/classify cycles -- see
heliogram/instruments/learned_alphabet.py's DEFAULT_ITERS docstring) while still exercising the
real search end to end, not a mocked-out shortcut.

Where this file pins a MEASURED outcome (an exact `symbol_error`/`baseline_symbol_error` value,
or `improved`'s value for a specific palette/seed/iters combination): if this ever flips, that is
a real behavior change (to `symbol_error_rate`, `_paint_candidate_tile`, `optimize_palette`'s
search loop, `DEFAULT_SEARCH_CORRUPTIONS`, or `heliogram.codec` itself) -- update the assertion
AND re-derive the number by actually re-running `optimize_palette`, don't just silently loosen or
delete the check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from heliogram.codec import PATCH_SIZE, decode_pixels, encode, extract_symbols, get_palette
from heliogram.encoder import FrozenEncoderHandle, _to_pixel_tensor
from heliogram.instruments.learned_alphabet import (
    DEFAULT_ITERS,
    DEFAULT_SEARCH_CORRUPTIONS,
    LearnedPalette,
    _paint_candidate_tile,
    _search_payloads,
    build_parser,
    compare_to_handcrafted,
    format_comparison,
    main,
    optimize_palette,
    symbol_error_rate,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- import-time boundary -------------------------------------------------------------------


def test_import_learned_alphabet_and_encoder_do_not_pull_in_torch():
    """The exact invariant the task's VERIFY step checks: importing either new module must
    never require torch/transformers, even transitively. Subprocess-isolated: see
    tests/conftest.py."""
    from tests.conftest import assert_import_stays_torch_free

    assert_import_stays_torch_free(
        "heliogram.instruments.learned_alphabet", "heliogram.encoder"
    )


# --- heliogram.encoder: FrozenEncoderHandle -------------------------------------------------


def test_frozen_encoder_handle_encode_pixels_requires_a_model():
    handle = FrozenEncoderHandle(model=None)
    with pytest.raises(RuntimeError, match="learned_alphabet"):
        handle.encode_pixels([[[0, 0, 0]]])


def test_frozen_encoder_handle_embed_with_pixel_grad_requires_a_model():
    """The task's explicit ask: FrozenEncoderHandle(model=None).embed_with_pixel_grad(...)
    raises RuntimeError."""
    handle = FrozenEncoderHandle(model=None)
    with pytest.raises(RuntimeError, match="learned_alphabet"):
        handle.embed_with_pixel_grad([[[0, 0, 0]]])


def test_frozen_encoder_handle_default_constructor_has_no_model():
    handle = FrozenEncoderHandle()
    assert handle.model is None
    assert handle.processor is None
    assert handle.device is None


def test_frozen_encoder_handle_stores_constructor_args():
    sentinel_model, sentinel_processor = object(), object()
    handle = FrozenEncoderHandle(model=sentinel_model, processor=sentinel_processor, device="cpu")
    assert handle.model is sentinel_model
    assert handle.processor is sentinel_processor
    assert handle.device == "cpu"


class _FakeTorch:
    """Stand-in for the `torch` module exposing only what `_to_pixel_tensor` calls
    (`from_numpy`), so the plain-numpy shape/dtype/scaling logic can be tested WITHOUT torch
    installed -- exactly the "ordinary, testable-without-a-model Python" heliogram/encoder.py's
    module docstring describes for this function."""

    @staticmethod
    def from_numpy(arr):
        return arr  # identity: let the test inspect the raw numpy array directly


def test_to_pixel_tensor_single_pil_image_shape_and_scaling():
    img = Image.new("RGB", (4, 6), (255, 0, 128))
    out = _to_pixel_tensor(img, _FakeTorch)
    assert out.shape == (1, 3, 6, 4)  # N, C, H, W
    assert out.dtype.name == "float32"
    assert out.max() <= 1.0 and out.min() >= 0.0
    # red channel (255) -> 1.0, green (0) -> 0.0
    assert out[0, 0, 0, 0] == pytest.approx(1.0)
    assert out[0, 1, 0, 0] == pytest.approx(0.0)


def test_to_pixel_tensor_batch_of_arrays():
    import numpy as np

    tiles = [np.zeros((5, 5, 3), dtype=np.uint8), np.full((5, 5, 3), 255, dtype=np.uint8)]
    out = _to_pixel_tensor(tiles, _FakeTorch)
    assert out.shape == (2, 3, 5, 5)
    assert out[0].max() == pytest.approx(0.0)
    assert out[1].min() == pytest.approx(1.0)


def test_to_pixel_tensor_rejects_bad_shape():
    import numpy as np

    with pytest.raises(ValueError):
        _to_pixel_tensor(np.zeros((5, 5)), _FakeTorch)  # missing channel dim


# --- symbol_error_rate: the search objective ------------------------------------------------


def test_symbol_error_rate_rejects_wrong_length_colors():
    with pytest.raises(ValueError, match="palette_size=8"):
        symbol_error_rate(get_palette(8)[:-1], palette_size=8)


def test_symbol_error_rate_rejects_invalid_palette_size():
    with pytest.raises(ValueError):
        symbol_error_rate(get_palette(8), palette_size=3)  # 3 is not in VALID_PALETTES


def test_symbol_error_rate_rejects_empty_corruptions():
    with pytest.raises(ValueError, match="corruptions"):
        symbol_error_rate(get_palette(8), palette_size=8, corruptions={})


def test_symbol_error_rate_deterministic_for_fixed_seed():
    colors = get_palette(16)
    e1 = symbol_error_rate(colors, palette_size=16, seed=0)
    e2 = symbol_error_rate(colors, palette_size=16, seed=0)
    assert e1 == e2


def test_symbol_error_rate_zero_for_handcrafted_small_palette():
    """MEASURED (seed=0, DEFAULT_SEARCH_CORRUPTIONS): the handcrafted get_palette(16) baseline
    has zero symbol error under this module's default corruption subset -- small hue-only
    palettes already survive resize/JPEG-85/JPEG-70/crop-pad at this severity on the reference
    pixel decoder. If this ever flips, that's a real change to DEFAULT_SEARCH_CORRUPTIONS or the
    codec/corruption suite -- re-derive, don't just delete."""
    assert symbol_error_rate(get_palette(16), palette_size=16, seed=0) == pytest.approx(0.0)


def test_symbol_error_rate_accepts_a_single_corruption_subset():
    """`corruptions` genuinely restricts which corruption(s) are measured -- a single-entry dict
    only runs that one corruption, not the whole default suite."""
    single = {"jpeg_q70": DEFAULT_SEARCH_CORRUPTIONS["jpeg_q70"]}
    # Should not raise, and should be a valid probability.
    err = symbol_error_rate(get_palette(32), palette_size=32, corruptions=single, seed=0)
    assert 0.0 <= err <= 1.0


def test_default_search_corruptions_names_and_order_are_pinned():
    """Regression pin on the exact corruption subset/order this module searches against by
    default -- changing this is a real behavior change to the search (see module docstring)."""
    assert list(DEFAULT_SEARCH_CORRUPTIONS) == [
        "clean",
        "resize_5pct",
        "jpeg_q85",
        "jpeg_q70",
        "crop_pad_2px",
    ]


# --- _search_payloads: deterministic payload generation for the search ----------------------


def test_search_payloads_deterministic_and_fixed_count():
    p1 = _search_payloads(seed=0)
    p2 = _search_payloads(seed=0)
    assert p1 == p2
    assert len(p1) >= 1
    assert all(isinstance(p, bytes) for p in p1)


def test_search_payloads_differ_across_seeds():
    assert _search_payloads(seed=0) != _search_payloads(seed=1)


# --- _paint_candidate_tile: the "repaint a real encode() image" technique -------------------


def test_paint_candidate_tile_with_identical_colors_reproduces_original_pixels():
    """Repainting with the SAME colors encode() already used must reproduce byte-identical
    pixels -- a sanity check on the repaint mechanism itself before trusting it for anything
    else."""
    payload = b"paint-candidate-tile sanity check"
    palette = 16
    clean = encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=32, seed=0, subpatch=1)
    width, height, symbols = extract_symbols(clean, palette=palette, patch_size=PATCH_SIZE)
    repainted = _paint_candidate_tile(
        clean, get_palette(palette), width, height, PATCH_SIZE, symbols
    )
    assert list(clean.getdata()) == list(repainted.getdata())


def test_paint_candidate_tile_with_distinct_colors_still_fully_decodes_when_clean():
    """THE key technical trick this module depends on, proven directly (not just asserted in
    the module docstring): repainting a real encode() image with an ARBITRARY distinct color
    list (here: get_palette's own colors, reversed -- nothing to do with the codec's own
    mapping) and then calling heliogram.codec.decode_pixels DIRECTLY and UNMODIFIED on the
    (uncorrupted) result still recovers the exact original payload. This is only possible
    because extract_symbols recovers its calibration colors from row 0's actual pixel content
    (never from get_palette as anything but an unreachable fallback) -- see the module
    docstring's "KEY TECHNICAL TRICK" section."""
    payload = b"the key technical trick, proven end to end"
    palette = 16
    clean = encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=32, seed=0, subpatch=1)
    width, height, symbols = extract_symbols(clean, palette=palette, patch_size=PATCH_SIZE)

    reversed_colors = list(reversed(get_palette(palette)))
    assert reversed_colors != get_palette(palette)  # genuinely a different mapping
    repainted = _paint_candidate_tile(clean, reversed_colors, width, height, PATCH_SIZE, symbols)

    decoded = decode_pixels(repainted, palette=palette, patch_size=PATCH_SIZE, nsym=32)
    assert decoded == payload


# --- optimize_palette: determinism -----------------------------------------------------------


def test_optimize_palette_deterministic_under_fixed_seed():
    """The task's explicit ask: optimize_palette is deterministic under a fixed seed (identical
    colors twice). Uses a small, cheap (palette_size=16, iters=10) config -- determinism is a
    property of the algorithm/seeding, not of any particular config, so a cheap one suffices."""
    r1 = optimize_palette(palette_size=16, seed=0, iters=10)
    r2 = optimize_palette(palette_size=16, seed=0, iters=10)
    assert r1.colors == r2.colors
    assert r1 == r2  # every field, not just colors


def test_optimize_palette_different_seeds_can_diverge():
    """Sanity check that `seed` actually drives the search (not silently ignored): at a palette
    size with room to improve (see the pinned-improvement test below), two different seeds visit
    different coordinates and can land on different learned colors."""
    r1 = optimize_palette(palette_size=32, seed=0, iters=20)
    r2 = optimize_palette(palette_size=32, seed=1, iters=20)
    assert r1.colors != r2.colors


def test_optimize_palette_rejects_invalid_palette_size():
    with pytest.raises(ValueError):
        optimize_palette(palette_size=3)


# --- optimize_palette / compare_to_handcrafted: the M4 DoD, both directions, no cherry-picking


def test_optimize_palette_never_worse_than_baseline_by_construction():
    """The "guarantee by construction" documented in optimize_palette's docstring: because the
    search only ever accepts a strictly improving move and starts from the baseline itself,
    symbol_error can never exceed baseline_symbol_error, for ANY palette size -- checked here
    across several sizes, not just the one pinned-improvement case below."""
    for palette_size in (8, 16, 32, 64):
        result = optimize_palette(palette_size=palette_size, seed=0, iters=10)
        assert result.symbol_error <= result.baseline_symbol_error + 1e-12
        assert result.improved == (result.symbol_error < result.baseline_symbol_error - 1e-12)


def test_compare_to_handcrafted_pins_measured_improvement_for_palette_32():
    """MEASURED (palette_size=32, seed=0, iters=40, DEFAULT_SEARCH_CORRUPTIONS): the search DOES
    find a strictly better palette than get_palette(32) here -- baseline_symbol_error
    0.011458... vs. learned symbol_error 0.000520... . This is the M4 Definition of Done: a
    symbol-error number for the learned code, reported beside the handcrafted baseline, honestly
    measured (not cherry-picked -- see the companion "no improvement" test below for a palette
    size where this does NOT happen). If this ever flips, that's a real change to the search/
    codec/corruption suite -- re-derive, don't just delete.
    """
    result = compare_to_handcrafted(palette_size=32, seed=0, iters=40)
    assert isinstance(result, LearnedPalette)
    assert result.palette_size == 32
    assert result.baseline_symbol_error == pytest.approx(0.011458333333333333)
    assert result.symbol_error == pytest.approx(0.0005208333333333333)
    assert result.improved is True
    assert len(result.colors) == 32
    assert "REDUCES" in result.note


def test_compare_to_handcrafted_honest_when_no_improvement_found():
    """MEASURED (palette_size=16, seed=0, iters=10, DEFAULT_SEARCH_CORRUPTIONS): the handcrafted
    baseline is ALREADY error-free here, so there is no room for the search to improve --
    `improved` is honestly False, not silently omitted or reframed. DATA-HONESTY style: this
    test exists specifically so a not-beaten result is asserted, not cherry-picked away. If this
    ever flips (the baseline stops being error-free under this suite, or the search finds an
    improving move where none existed), that's a real change -- re-derive, don't just delete.
    """
    result = compare_to_handcrafted(palette_size=16, seed=0, iters=10)
    assert result.baseline_symbol_error == pytest.approx(0.0)
    assert result.symbol_error == pytest.approx(0.0)
    assert result.improved is False
    assert "did NOT improve" in result.note


def test_compare_to_handcrafted_is_a_thin_alias_for_optimize_palette():
    a = optimize_palette(palette_size=8, seed=0, iters=5)
    b = compare_to_handcrafted(palette_size=8, seed=0, iters=5)
    assert a == b


def test_optimize_palette_threads_custom_corruptions_through():
    custom = {"jpeg_q70": DEFAULT_SEARCH_CORRUPTIONS["jpeg_q70"]}
    result = optimize_palette(palette_size=8, corruptions=custom, seed=0, iters=5)
    assert result.corruptions == ["jpeg_q70"]


# --- LearnedPalette: plain dataclass ----------------------------------------------------------


def test_learned_palette_is_a_plain_dataclass_with_expected_fields():
    result = LearnedPalette(
        palette_size=8,
        colors=[(0, 0, 0)] * 8,
        seed=0,
        corruptions=["clean"],
        symbol_error=0.1,
        baseline_symbol_error=0.2,
        improved=True,
        note="test note",
    )
    assert result.palette_size == 8
    assert len(result.colors) == 8
    assert result.seed == 0
    assert result.corruptions == ["clean"]
    assert result.symbol_error == 0.1
    assert result.baseline_symbol_error == 0.2
    assert result.improved is True
    assert result.note == "test note"


# --- format_comparison / CLI -------------------------------------------------------------------


def test_format_comparison_mentions_key_fields():
    """Constructs a LearnedPalette directly (no search needed) so this test is instant."""
    result = LearnedPalette(
        palette_size=8,
        colors=list(get_palette(8)),
        seed=0,
        corruptions=["clean", "jpeg_q70"],
        symbol_error=0.125,
        baseline_symbol_error=0.25,
        improved=True,
        note="example note text",
    )
    text = format_comparison(result)
    assert "palette_size=8" in text
    assert "0.1250" in text
    assert "0.2500" in text
    assert "yes" in text.lower()
    assert "example note text" in text


def test_build_parser_has_sane_defaults_and_does_not_require_flags():
    parser = build_parser()
    args = parser.parse_args([])  # unlike heliogram.patchsize, nothing is required here
    assert args.palette_sizes == [8, 16]
    assert args.iters == DEFAULT_ITERS
    assert args.seed == 0


def test_main_runs_end_to_end_and_prints_comparison(capsys):
    rc = main(["--palette-sizes", "8", "--iters", "5", "--seed", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "palette_size=8" in out
    assert "improved over handcrafted baseline" in out


def test_cli_help_does_not_require_torch():
    result = subprocess.run(
        [sys.executable, "-m", "heliogram.instruments.learned_alphabet", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--palette-sizes" in result.stdout
