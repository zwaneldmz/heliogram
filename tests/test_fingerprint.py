"""Pytest suite for heliogram.instruments.fingerprint (handoff M6, A10: "encoder id / silent-swap
probe" -- a CPU-simulatable blind-swap test, a byproduct of the capacity sweep).

Assert-based, no fixtures/frameworks beyond plain pytest.raises, matching the rest of this
repo's test idiom. Like tests/test_foreign_tile.py's equivalent check for
heliogram.instruments.foreign_tile, this file proves the CPU-only import boundary holds for
heliogram.instruments.fingerprint too -- nothing in this module (or its tests) ever needs
torch/transformers.

Several tests below pin ACTUAL measured numbers (the reference signature, and the
reference-vs-swapped distance) rather than just bounds-checking. Per this project's DATA HONESTY
rule: if any of these ever flip, that is a real behavior change (to fingerprint()'s trial/
corruption math, swapped_palette_encode's construction, or the underlying codec/corruption
modules) -- update the assertion AND re-derive the number by actually re-running fingerprint(),
don't just silently loosen or delete the check.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from PIL import Image

from heliogram.codec import PATCH_SIZE, VALID_PALETTES, VALID_SUBPATCHES, encode, extract_symbols
from heliogram.instruments.fingerprint import (
    DEFAULT_FINGERPRINT_CORRUPTIONS,
    DEFAULT_SWAP_THRESHOLD,
    Fingerprint,
    build_parser,
    detect_swap,
    distance,
    fingerprint,
    format_fingerprint,
    main,
    swapped_palette_encode,
)

# --- import-time boundary -------------------------------------------------------------------


def test_import_fingerprint_does_not_pull_in_torch():
    from tests.conftest import assert_import_stays_torch_free

    assert_import_stays_torch_free("heliogram.instruments.fingerprint")


# --- Fingerprint: plain dataclass -------------------------------------------------------------


def test_fingerprint_dataclass_fields():
    fp = Fingerprint(
        palette=8,
        subpatch=1,
        payload_size=48,
        corruptions=["clean", "jpeg_q70"],
        signature={"clean": 0.0, "jpeg_q70": 0.01},
        note="n",
    )
    assert fp.palette == 8
    assert fp.corruptions == ["clean", "jpeg_q70"]
    assert fp.signature["jpeg_q70"] == 0.01


def test_fingerprints_compare_by_value():
    kwargs = dict(palette=8, subpatch=1, payload_size=48, corruptions=["clean"], note="n")
    a = Fingerprint(signature={"clean": 0.0}, **kwargs)
    b = Fingerprint(signature={"clean": 0.0}, **kwargs)
    c = Fingerprint(signature={"clean": 0.1}, **kwargs)
    assert a == b
    assert a != c


# --- fingerprint(): defaults reproduce decode_pixels' own signature ---------------------------


def test_fingerprint_default_corruptions_are_the_documented_subset():
    assert list(DEFAULT_FINGERPRINT_CORRUPTIONS) == [
        "clean",
        "resize_5pct",
        "jpeg_q85",
        "jpeg_q70",
        "combined",
    ]


def test_fingerprint_returns_dataclass_with_expected_shape():
    fp = fingerprint(palette=8, payload_size=64, trials=2)
    assert isinstance(fp, Fingerprint)
    assert fp.palette == 8
    assert fp.subpatch == 1
    assert fp.payload_size == 64
    assert fp.corruptions == list(DEFAULT_FINGERPRINT_CORRUPTIONS)
    assert set(fp.signature) == set(DEFAULT_FINGERPRINT_CORRUPTIONS)
    assert "CHANNEL/DECODER" in fp.note
    assert "VLM" in fp.note


def test_fingerprint_clean_and_resize_5pct_are_exactly_zero_at_defaults():
    """At the codec's own defaults (encode_fn=encode, decode_symbols_fn=extract_symbols), truth
    and observed are read off literally the same clean image for "clean" (a no-op corruption),
    so symbol_error_rate must be EXACTLY 0.0, not just close -- and this codec/corruption
    combination happens to also fully survive a 5% resize round-trip at palette=64 (see the next
    test's pinned numbers)."""
    fp = fingerprint()  # module defaults: palette=64, payload_size=512, trials=5, seed=0
    assert fp.signature["clean"] == 0.0
    assert fp.signature["resize_5pct"] == 0.0


def test_fingerprint_default_signature_is_pinned():
    """MEASURED (palette=64, subpatch=1, payload_size=512, trials=5, seed=0, the module's own
    defaults): this exact signature. If this ever flips (a change to encode/extract_symbols/
    corruption primitives, or to fingerprint()'s own trial/corruption math), that is a real
    behavior change -- re-run fingerprint() and update these pinned numbers, don't just loosen
    the bounds."""
    fp = fingerprint()
    assert fp.signature["clean"] == pytest.approx(0.0)
    assert fp.signature["resize_5pct"] == pytest.approx(0.0)
    assert fp.signature["jpeg_q85"] == pytest.approx(0.0033653846153846156)
    assert fp.signature["jpeg_q70"] == pytest.approx(0.04591346153846154)
    assert fp.signature["combined"] == pytest.approx(0.03028846153846154)


def test_fingerprint_signature_values_are_fractions_in_unit_interval():
    fp = fingerprint(palette=16, payload_size=64, trials=2)
    for rate in fp.signature.values():
        assert 0.0 <= rate <= 1.0


def test_fingerprint_deterministic_across_separate_calls():
    """Two independent fingerprint() calls with identical arguments must produce byte-identical
    signatures -- no unseeded randomness anywhere in this module (see module docstring)."""
    kwargs = dict(palette=32, payload_size=128, trials=3, seed=7)
    a = fingerprint(**kwargs)
    b = fingerprint(**kwargs)
    assert a.signature == b.signature
    assert a.corruptions == b.corruptions


def test_fingerprint_different_seed_can_change_signature():
    """Sanity check that `seed` actually drives the measurement (not silently ignored) -- mirrors
    test_optimize_palette_different_seeds_can_diverge in tests/test_learned_alphabet.py, a direct
    `!=` with no escape hatch.

    The previous version of this test used palette=8, payload_size=32, trials=3 and asserted
    `a.signature != b.signature or a.corruptions == b.corruptions`. Since `corruptions` never
    varies with `seed` (both calls always use DEFAULT_FINGERPRINT_CORRUPTIONS here), the right
    side of that OR was always True, making the whole assertion a tautology -- it would still
    pass even if `seed` were silently dropped inside fingerprint()'s payload construction.
    Confirmed by direct measurement: fingerprint(palette=8, payload_size=32, trials=3, seed=0) vs.
    seed=99 produces byte-identical all-0.0 signatures (palette=8's wide inter-symbol spacing
    shrugs off this module's mild corruption envelope regardless of payload content) -- exactly
    the coincidental tie the old escape hatch was silently papering over.

    MEASURED: at palette=32, payload_size=32, trials=2, seed=0 vs. seed=99 diverge on
    jpeg_q85/jpeg_q70/combined (different random payloads corrupt differently), so a plain `!=`
    holds with no hedging. If this ever ties, that's either a real regression (seed no longer
    threaded into random_payload()/_search) or a genuine behavior change to the corruption/codec
    primitives -- re-verify by hand (don't just delete the check) and pick a fresh seed pair or
    update this comment.
    """
    a = fingerprint(palette=32, payload_size=32, trials=2, seed=0)
    b = fingerprint(palette=32, payload_size=32, trials=2, seed=99)
    assert a.signature != b.signature


def test_fingerprint_custom_corruptions_subset():
    fp = fingerprint(palette=8, payload_size=32, trials=2, corruptions={"clean": lambda img: img})
    assert fp.corruptions == ["clean"]
    assert fp.signature == {"clean": 0.0}


# --- fingerprint(): validation ------------------------------------------------------------------


def test_fingerprint_rejects_invalid_palette():
    with pytest.raises(ValueError):
        fingerprint(palette=3)  # not in VALID_PALETTES


def test_fingerprint_rejects_invalid_subpatch():
    with pytest.raises(ValueError):
        fingerprint(subpatch=3)  # not in VALID_SUBPATCHES


def test_fingerprint_rejects_trials_below_one():
    with pytest.raises(ValueError, match="trials"):
        fingerprint(trials=0)


def test_fingerprint_rejects_empty_corruptions():
    with pytest.raises(ValueError, match="corruptions"):
        fingerprint(corruptions={})


# --- distance(): L1/L2 over shared corruption keys ------------------------------------------


def test_distance_identical_fingerprints_is_zero():
    fp = fingerprint(palette=8, payload_size=32, trials=2)
    assert distance(fp, fp) == 0.0
    assert distance(fp, fp, metric="l1") == 0.0


def test_distance_l1_vs_l2():
    a = Fingerprint(
        palette=8, subpatch=1, payload_size=1, corruptions=["x", "y"],
        signature={"x": 0.0, "y": 0.0}, note="n",
    )
    b = Fingerprint(
        palette=8, subpatch=1, payload_size=1, corruptions=["x", "y"],
        signature={"x": 0.3, "y": 0.4}, note="n",
    )
    assert distance(a, b, metric="l1") == pytest.approx(0.7)
    assert distance(a, b, metric="l2") == pytest.approx(0.5)  # 3-4-5 triangle


def test_distance_only_considers_shared_keys():
    a = Fingerprint(
        palette=8, subpatch=1, payload_size=1, corruptions=["x", "y"],
        signature={"x": 0.0, "y": 0.5}, note="n",
    )
    b = Fingerprint(
        palette=8, subpatch=1, payload_size=1, corruptions=["x", "z"],
        signature={"x": 0.0, "z": 0.9}, note="n",
    )
    assert distance(a, b) == 0.0  # only "x" is shared, and it matches on both sides


def test_distance_raises_on_no_shared_keys():
    a = Fingerprint(
        palette=8, subpatch=1, payload_size=1, corruptions=["x"], signature={"x": 0.0}, note="n"
    )
    b = Fingerprint(
        palette=8, subpatch=1, payload_size=1, corruptions=["y"], signature={"y": 0.0}, note="n"
    )
    with pytest.raises(ValueError, match="no common corruption key"):
        distance(a, b)


def test_distance_rejects_unknown_metric():
    fp = fingerprint(palette=8, payload_size=32, trials=2)
    with pytest.raises(ValueError, match="metric"):
        distance(fp, fp, metric="bogus")


# --- detect_swap(): the blind-swap Definition of Done -----------------------------------------


def test_detect_swap_false_for_identical_config():
    """Reference vs. a FRESH fingerprint() call with identical config: must not be flagged --
    this is one half of the blind-swap DoD (no false alarm on the trusted pipeline)."""
    reference = fingerprint()
    identical = fingerprint()  # same defaults, computed independently
    assert distance(reference, identical) == 0.0
    assert detect_swap(reference, identical) is False


def test_detect_swap_true_for_deliberately_swapped_encoder():
    """THE Definition of Done this module targets (see its module docstring): a deliberately
    swapped encoder (shuffled palette mapping) must be flagged.

    MEASURED (palette=64, subpatch=1, payload_size=512, trials=5, seed=0, default
    DEFAULT_SWAP_THRESHOLD=0.2, default shuffle_seed=1337): distance(reference, swapped) ==
    2.1303966481434524 (L2), roughly 10x the threshold -- comfortably flagged. If this ever
    flips, that is a real behavior change -- re-run fingerprint(encode_fn=swapped_palette_encode)
    and update this pinned number, don't silently loosen or delete the check.
    """
    reference = fingerprint()
    swapped = fingerprint(encode_fn=swapped_palette_encode)

    d = distance(reference, swapped)
    assert d == pytest.approx(2.1303966481434524)
    assert d > DEFAULT_SWAP_THRESHOLD
    assert detect_swap(reference, swapped) is True
    assert detect_swap(reference, swapped, threshold=DEFAULT_SWAP_THRESHOLD) is True


def test_detect_swap_true_for_swapped_decoder_too():
    """The seam works both ways (see fingerprint()'s docstring): swapping decode_symbols_fn for
    a reader that disagrees with the canonical mapping must ALSO be flagged, not just a swapped
    encoder."""

    def offset_by_one(img, palette=64, patch_size=PATCH_SIZE, subpatch=1):
        w, h, symbols = extract_symbols(
            img, palette=palette, patch_size=patch_size, subpatch=subpatch
        )
        return w, h, [(s + 1) % palette for s in symbols]

    reference = fingerprint()
    swapped_decoder = fingerprint(decode_symbols_fn=offset_by_one)
    assert distance(reference, swapped_decoder) > DEFAULT_SWAP_THRESHOLD
    assert detect_swap(reference, swapped_decoder) is True


def test_detect_swap_respects_custom_threshold():
    reference = fingerprint()
    swapped = fingerprint(encode_fn=swapped_palette_encode)
    d = distance(reference, swapped)
    assert detect_swap(reference, swapped, threshold=d + 1.0) is False  # raise the bar past it
    assert detect_swap(reference, swapped, threshold=d - 1e-6) is True  # lower it just below


def test_detect_swap_deterministic_under_fixed_seed():
    """Full pipeline determinism: rebuilding reference/swapped fingerprints from scratch with the
    same seed always yields the same swap verdict and (up to floating point) the same distance."""
    ref1 = fingerprint(palette=16, payload_size=64, trials=3, seed=2)
    swap1 = fingerprint(palette=16, payload_size=64, trials=3, seed=2, encode_fn=swapped_palette_encode)
    ref2 = fingerprint(palette=16, payload_size=64, trials=3, seed=2)
    swap2 = fingerprint(palette=16, payload_size=64, trials=3, seed=2, encode_fn=swapped_palette_encode)

    assert distance(ref1, swap1) == distance(ref2, swap2)
    assert detect_swap(ref1, swap1) == detect_swap(ref2, swap2) is True


# --- swapped_palette_encode(): the deliberately-swapped encoder helper ------------------------


def test_swapped_palette_encode_preserves_image_geometry():
    """Same patch grid / pixel dimensions as a canonical encode() call -- the swap only changes
    DATA cell colors, never the geometry, so corruption functions (which assume a fixed grid)
    still apply cleanly."""
    payload = b"geometry check payload"
    standard = encode(payload, palette=16, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
    swapped = swapped_palette_encode(payload, palette=16, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
    assert standard.size == swapped.size


def test_swapped_palette_encode_leaves_calibration_row_untouched():
    payload = b"calibration row check"
    standard = encode(payload, palette=16, patch_size=PATCH_SIZE, subpatch=1)
    swapped = swapped_palette_encode(payload, palette=16, patch_size=PATCH_SIZE, subpatch=1)
    std_arr = np.array(standard.convert("RGB"))
    swap_arr = np.array(swapped.convert("RGB"))
    assert (std_arr[:PATCH_SIZE] == swap_arr[:PATCH_SIZE]).all()  # row 0 identical
    assert not (std_arr[PATCH_SIZE:] == swap_arr[PATCH_SIZE:]).all()  # data rows differ


def test_swapped_palette_encode_deterministic_for_fixed_shuffle_seed():
    payload = b"determinism check"
    a = swapped_palette_encode(payload, palette=32, shuffle_seed=1337)
    b = swapped_palette_encode(payload, palette=32, shuffle_seed=1337)
    assert list(a.getdata()) == list(b.getdata())


def test_swapped_palette_encode_different_shuffle_seed_differs():
    payload = b"different mapping check"
    a = swapped_palette_encode(payload, palette=32, shuffle_seed=1337)
    b = swapped_palette_encode(payload, palette=32, shuffle_seed=42)
    assert list(a.getdata()) != list(b.getdata())


def test_swapped_palette_encode_supports_every_valid_subpatch():
    for subpatch in VALID_SUBPATCHES:
        payload = b"subpatch sweep"
        standard = encode(payload, palette=8, subpatch=subpatch)
        swapped = swapped_palette_encode(payload, palette=8, subpatch=subpatch)
        assert standard.size == swapped.size


def test_swapped_palette_encode_rejects_invalid_subpatch():
    with pytest.raises(ValueError):
        swapped_palette_encode(b"x", palette=8, subpatch=3)


def test_swapped_palette_encode_ignores_seed_like_encode_does():
    """`seed` is accepted only for encode_fn call-signature compatibility (mirrors
    heliogram.codec.encode's own reserved `seed` parameter) -- it must not change the output."""
    payload = b"seed is a no-op"
    a = swapped_palette_encode(payload, palette=16, seed=0)
    b = swapped_palette_encode(payload, palette=16, seed=999)
    assert list(a.getdata()) == list(b.getdata())


def test_swapped_palette_encode_usable_as_a_drop_in_encode_fn():
    """The whole point: swapped_palette_encode's call signature is interchangeable with
    heliogram.codec.encode's, so it plugs directly into fingerprint()'s encode_fn= seam."""
    fp = fingerprint(encode_fn=swapped_palette_encode, palette=8, payload_size=32, trials=2)
    assert isinstance(fp, Fingerprint)


# --- format_fingerprint() ----------------------------------------------------------------------


def test_format_fingerprint_mentions_palette_and_every_corruption():
    fp = fingerprint(palette=8, payload_size=32, trials=2)
    text = format_fingerprint(fp)
    assert "palette:      8" in text
    for name in fp.corruptions:
        assert name in text
    assert "note:" in text


# --- CLI: build_parser / main -------------------------------------------------------------------


def test_build_parser_has_sane_defaults_and_needs_no_required_flags():
    parser = build_parser()
    args = parser.parse_args([])  # nothing required -- fingerprint()'s own defaults apply
    assert args.palette == 64
    assert args.subpatch == 1
    assert args.trials == 5


def test_build_parser_rejects_invalid_palette_choice():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--palette", "3"])  # not in VALID_PALETTES


def test_main_demonstrates_blind_swap_dod(capsys):
    """End-to-end CLI demo: reference vs. identical-config (not flagged) and reference vs.
    swapped-encoder (flagged) -- the exact two cases the module docstring promises. main()
    returns 0 exactly when both behave as documented (see its own self-check)."""
    rc = main(["--palette", "8", "--payload-size", "64", "--trials", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Reference fingerprint" in out
    assert "Identical-config fingerprint" in out
    assert "Swapped-encoder fingerprint" in out
    assert "detect_swap = False" in out
    assert "detect_swap = True" in out
    assert "not a VLM measurement" in out


def test_main_self_check_fails_loudly_if_threshold_hides_a_real_swap(capsys):
    """If --threshold is set absurdly high, the swapped-encoder case is no longer flagged --
    main()'s own self-check must catch that and return 1, not silently report success."""
    rc = main(
        ["--palette", "8", "--payload-size", "64", "--trials", "2", "--threshold", "1000"]
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "UNEXPECTED" in err


def test_main_self_check_fails_loudly_if_threshold_flags_the_identical_case(capsys):
    """A threshold below 0 flags even the identical-config comparison (distance 0.0 > any
    negative number) -- main() must report that as a failure too."""
    rc = main(["--palette", "8", "--payload-size", "64", "--trials", "2", "--threshold", "-1"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "UNEXPECTED" in err


def test_cli_help_does_not_require_torch_and_lists_flags():
    parser = build_parser()
    help_text = parser.format_help()
    assert "--palette" in help_text
    assert "--threshold" in help_text
    from tests.conftest import assert_import_stays_torch_free

    assert_import_stays_torch_free("heliogram.instruments.fingerprint")
