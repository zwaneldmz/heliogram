"""Pytest suite for heliogram.instruments.foreign_tile (handoff M6, I2/B10: the pre-ingest guard
that must exist BEFORE any learned-alphabet capability work, guardrail #4).

Assert-based, no fixtures/frameworks beyond plain pytest.raises, matching the rest of this
repo's test idiom. Like tests/test_phase2_scaffold.py's equivalent check for heliogram.vlm, this
file proves the CPU-only import boundary holds for heliogram.instruments.foreign_tile too --
nothing in this module (or its tests) ever needs torch/transformers.

The evaluate_detector(build_eval_set(...)) test below pins the ACTUAL measured TPR/FPR this
detector achieves against a deterministic, seeded eval set (see build_eval_set's own docstring
for exactly what counts as a positive/negative). Per this project's DATA HONESTY rule: if this
number ever flips, that is a real behavior change (to patch_structure_score, DEFAULT_VAR_THRESHOLD,
is_foreign's decode-attempt logic, or build_eval_set's construction) -- update the assertion AND
re-derive the number by actually re-running evaluate_detector, don't just silently loosen or
delete the check.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from PIL import Image

from heliogram.codec import PATCH_SIZE, VALID_PALETTES, HeliogramDecodeError, decode_pixels, encode
from heliogram.dataset import random_payload
from heliogram.instruments.foreign_tile import (
    DEFAULT_VAR_THRESHOLD,
    AllowListEntry,
    DetectorEval,
    ForeignTileVerdict,
    build_eval_set,
    evaluate_detector,
    guard,
    is_foreign,
    patch_structure_score,
)

DEFAULT_ALLOWLIST = (AllowListEntry(palette=8), AllowListEntry(palette=16))

# --- import-time boundary -------------------------------------------------------------------


def test_import_foreign_tile_does_not_pull_in_torch():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


# --- patch_structure_score: clean tiles vs. natural images --------------------------------------


def test_patch_structure_score_zero_for_clean_heliogram_tile():
    """A freshly-encoded, uncorrupted heliogram tile is exactly solid-per-patch -- score must be
    (numerically) zero, regardless of palette."""
    for palette in (8, 16, 32, 64):
        img = encode(random_payload(0, 48), palette=palette, patch_size=PATCH_SIZE, nsym=32)
        assert patch_structure_score(img) == pytest.approx(0.0, abs=1e-9)


def test_patch_structure_score_large_for_uniform_noise():
    rng = np.random.RandomState(0)
    arr = rng.randint(0, 256, size=(168, 168, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    assert patch_structure_score(img) > DEFAULT_VAR_THRESHOLD


def test_patch_structure_score_too_small_image_is_infinite():
    tiny = Image.new("RGB", (5, 5), (0, 0, 0))  # smaller than one PATCH_SIZE=14 cell
    assert patch_structure_score(tiny, patch_size=PATCH_SIZE) == float("inf")


# --- is_foreign: the three core cases named in the handoff --------------------------------------


def test_non_allowlisted_heliogram_tile_is_foreign():
    """A real heliogram tile encoded with a palette NOT on the allow-list must be flagged."""
    img = encode(random_payload(1, 64), palette=64, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
    verdict = is_foreign(img, DEFAULT_ALLOWLIST)
    assert isinstance(verdict, ForeignTileVerdict)
    assert verdict.is_patch_structured is True
    assert verdict.is_foreign is True
    assert verdict.matched_allowlist_entry is None


def test_allowlisted_heliogram_tile_is_not_foreign():
    """A real heliogram tile encoded with an allow-listed config must NOT be flagged, and the
    verdict must report which allow-list entry matched."""
    img = encode(random_payload(2, 64), palette=8, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
    verdict = is_foreign(img, DEFAULT_ALLOWLIST)
    assert verdict.is_patch_structured is True
    assert verdict.is_foreign is False
    assert verdict.matched_allowlist_entry == AllowListEntry(palette=8)


def test_allowlisted_tile_matches_regardless_of_entry_order():
    img = encode(random_payload(3, 64), palette=16, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
    verdict = is_foreign(img, DEFAULT_ALLOWLIST)
    assert verdict.is_foreign is False
    assert verdict.matched_allowlist_entry == AllowListEntry(palette=16)


def test_seeded_natural_image_is_not_foreign_and_not_patch_structured():
    rng = np.random.RandomState(7)
    arr = rng.randint(0, 256, size=(168, 168, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    verdict = is_foreign(img, DEFAULT_ALLOWLIST)
    assert verdict.is_patch_structured is False
    assert verdict.is_foreign is False


def test_is_foreign_patch_grid_reflects_image_dimensions():
    img = encode(random_payload(4, 32), palette=8, patch_size=PATCH_SIZE, nsym=32)
    verdict = is_foreign(img, DEFAULT_ALLOWLIST)
    expected = (img.width // PATCH_SIZE, img.height // PATCH_SIZE)
    assert verdict.patch_grid == expected


def test_is_foreign_rejects_only_heliogram_decode_error_others_propagate():
    """A malformed AllowListEntry (invalid palette, a caller bug) must propagate loudly rather
    than being swallowed as "doesn't decode" -- is_foreign only catches HeliogramDecodeError."""
    img = encode(random_payload(5, 32), palette=8, patch_size=PATCH_SIZE, nsym=32)
    bad_allowlist = [AllowListEntry(palette=3)]  # 3 is not in VALID_PALETTES
    with pytest.raises(ValueError):
        is_foreign(img, bad_allowlist)


# --- guard(): the intended pre-ingest wrapper ---------------------------------------------------


def test_guard_is_a_thin_wrapper_around_is_foreign():
    foreign_img = encode(random_payload(6, 48), palette=64, patch_size=PATCH_SIZE, nsym=32)
    trusted_img = encode(random_payload(7, 48), palette=8, patch_size=PATCH_SIZE, nsym=32)
    assert guard(foreign_img, DEFAULT_ALLOWLIST) is True
    assert guard(trusted_img, DEFAULT_ALLOWLIST) is False


def test_guard_forwards_kwargs_to_is_foreign():
    """Proves var_threshold= actually reaches is_foreign (not silently dropped) by forcing each
    boundary case: a negative threshold makes EVERY score "above threshold" (score is always
    >= 0.0), so even a clean tile reads as not-patch-structured; a huge threshold makes even
    pure noise read as "patch structured", pushing it into (and failing) the decode-attempt
    stage instead of being waved through as natural."""
    rng = np.random.RandomState(0)
    noise = Image.fromarray(rng.randint(0, 256, size=(168, 168, 3), dtype=np.uint8))

    tile = encode(random_payload(8, 48), palette=8, patch_size=PATCH_SIZE, nsym=32)
    assert is_foreign(tile, DEFAULT_ALLOWLIST, var_threshold=-1.0).is_patch_structured is False

    assert guard(noise, DEFAULT_ALLOWLIST) is False  # default threshold: correctly not flagged
    assert guard(noise, DEFAULT_ALLOWLIST, var_threshold=1e9) is True  # forced "structured"
    huge_verdict = is_foreign(noise, DEFAULT_ALLOWLIST, var_threshold=1e9)
    assert huge_verdict.is_patch_structured is True
    assert huge_verdict.is_foreign is True  # patch-structured (forced) but decodes under nothing


# --- AllowListEntry / ForeignTileVerdict / DetectorEval: plain dataclasses ----------------------


def test_allowlist_entry_defaults_mirror_encode_defaults():
    entry = AllowListEntry()
    assert entry.palette == 8
    assert entry.patch_size == PATCH_SIZE
    assert entry.subpatch == 1


def test_allowlist_entries_compare_by_value():
    assert AllowListEntry(palette=8) == AllowListEntry(palette=8)
    assert AllowListEntry(palette=8) != AllowListEntry(palette=16)


# --- evaluate_detector / build_eval_set: the actual DoD -----------------------------------------


def test_build_eval_set_is_deterministic():
    p1, n1, a1 = build_eval_set(seed=0)
    p2, n2, a2 = build_eval_set(seed=0)
    assert len(p1) == len(p2) and len(n1) == len(n2)
    assert a1 == a2
    for imga, imgb in zip(p1, p2):
        assert list(imga.getdata()) == list(imgb.getdata())
    for imga, imgb in zip(n1, n2):
        assert list(imga.getdata()) == list(imgb.getdata())


def test_build_eval_set_different_seeds_differ():
    p1, _, _ = build_eval_set(seed=0)
    p2, _, _ = build_eval_set(seed=1)
    # not every image need differ, but the whole sequence should not be byte-identical
    same = all(
        list(a.getdata()) == list(b.getdata())
        for a, b in zip(p1, p2)
        if a.size == b.size
    )
    assert not same


def test_build_eval_set_positives_are_all_flagged_foreign_by_construction():
    """Every positive build_eval_set produces (both the plain off-allowlist ones and the harder
    learned-alphabet-like ones) must independently verify as is_foreign=True -- this is a
    consistency check on the construction itself, ahead of the aggregate TPR/FPR pinned below."""
    positives, _, allowlist = build_eval_set(seed=0)
    for img in positives:
        assert is_foreign(img, allowlist).is_foreign is True


def test_build_eval_set_negatives_are_all_not_flagged_by_construction():
    _, negatives, allowlist = build_eval_set(seed=0)
    for img in negatives:
        assert is_foreign(img, allowlist).is_foreign is False


def test_build_eval_set_without_learned_alphabet_has_fewer_positives():
    with_hard, _, _ = build_eval_set(seed=0, include_learned_alphabet=True)
    without_hard, _, _ = build_eval_set(seed=0, include_learned_alphabet=False)
    assert len(without_hard) < len(with_hard)


def test_build_eval_set_respects_custom_allowlist_without_reusing_its_palettes():
    """If a caller's own allowlist already covers the default off-allowlist palettes (32, 64),
    build_eval_set must pick a genuinely different palette for its "plain" positives rather than
    silently reusing an allow-listed (trusted) palette as a supposedly-foreign one."""
    custom_allowlist = [AllowListEntry(palette=32), AllowListEntry(palette=64)]
    positives, negatives, allowlist = build_eval_set(seed=0, allowlist=custom_allowlist)
    for img in positives:
        assert is_foreign(img, allowlist).is_foreign is True
    for img in negatives:
        assert is_foreign(img, allowlist).is_foreign is False


def test_evaluate_detector_returns_dataclass_with_expected_fields():
    positives, negatives, allowlist = build_eval_set(seed=0)
    result = evaluate_detector(positives, negatives, allowlist)
    assert isinstance(result, DetectorEval)
    assert result.n_positive == len(positives)
    assert result.n_negative == len(negatives)
    assert result.threshold == DEFAULT_VAR_THRESHOLD


def test_evaluate_detector_tpr_at_bounded_fpr_against_learned_alphabet_tiles():
    """THE Definition of Done this module targets (see its module docstring): TPR at a bounded
    FPR, measured against learned-alphabet-like hard positives, not just easy ones.

    MEASURED (seed=0, DEFAULT_VAR_THRESHOLD, default allowlist {palette=8, palette=16}):
    tpr=1.0, fpr=0.0 over 8 positives (4 plain off-allowlist + 4 learned-alphabet-like hard
    positives) and 16 negatives (12 natural-ish + 4 allow-listed trusted tiles). Bars from the
    task: TPR >= 0.9, FPR <= 0.1 -- both cleared with margin. If this ever flips (a code change
    to patch_structure_score, DEFAULT_VAR_THRESHOLD, is_foreign, or build_eval_set's
    construction), that is a real behavior change -- re-run evaluate_detector(*build_eval_set())
    and update BOTH the bars-check below and this docstring's numbers, don't silently delete it.
    """
    positives, negatives, allowlist = build_eval_set(seed=0)
    result = evaluate_detector(positives, negatives, allowlist)

    assert result.tpr >= 0.9
    assert result.fpr <= 0.1
    # Pin the exact observed numbers too (not just the bar), per this project's data-honesty
    # convention of asserting a measured outcome exactly, not just "within bounds".
    assert result.tpr == pytest.approx(1.0)
    assert result.fpr == pytest.approx(0.0)


def test_evaluate_detector_tpr_fpr_are_fractions_in_unit_interval():
    positives, negatives, allowlist = build_eval_set(seed=0)
    result = evaluate_detector(positives, negatives, allowlist)
    assert 0.0 <= result.tpr <= 1.0
    assert 0.0 <= result.fpr <= 1.0


def test_evaluate_detector_empty_positives_or_negatives_do_not_divide_by_zero():
    _, negatives, allowlist = build_eval_set(seed=0)
    result = evaluate_detector([], negatives, allowlist)
    assert result.tpr == 0.0
    assert result.n_positive == 0


# --- sanity: build_eval_set's off-allowlist palettes are real VALID_PALETTES values -------------


def test_build_eval_set_default_allowlist_uses_valid_palettes():
    _, _, allowlist = build_eval_set(seed=0)
    for entry in allowlist:
        assert entry.palette in VALID_PALETTES


def test_hard_positives_actually_fail_to_decode_under_every_allowlist_entry():
    """Direct proof of the mechanism (not just the end-to-end is_foreign verdict): a
    learned-alphabet-like hard positive must raise on decode_pixels for EVERY allow-listed
    entry, individually -- this is what makes it "foreign" rather than "structured but
    accidentally trusted"."""
    positives, _, allowlist = build_eval_set(seed=0, include_learned_alphabet=True)
    # the last positives in the list are the hard (learned-alphabet-like) ones -- see
    # build_eval_set's construction order (plain off-allowlist first, then hard positives).
    hard_positives = positives[-4:]
    for img in hard_positives:
        for entry in allowlist:
            with pytest.raises(HeliogramDecodeError):
                decode_pixels(
                    img, palette=entry.palette, patch_size=entry.patch_size, nsym=32,
                    subpatch=entry.subpatch,
                )
