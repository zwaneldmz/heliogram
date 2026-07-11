"""Pytest suite for heliogram.instruments.saliency (handoff M6, A11: "recoverable bits by patch
position" -- a near-free byproduct of the M3 capacity sweep, CPU/model-free).

Assert-based, no fixtures/frameworks beyond plain pytest, matching the rest of this repo's test
idiom. Like tests/test_foreign_tile.py and tests/test_patchsize.py, this file's import-time test
proves heliogram.instruments.saliency never needs torch/transformers merely to be imported or to
run its (no real model) code path -- it measures heliogram.codec's reference pixel decoder only.

test_crop_pad_shifted_column_has_higher_error_than_interior_columns pins a MEASURED, geometrically
-grounded outcome (see heliogram.instruments.saliency's own module docstring for the derivation,
and this test's docstring below for the honest accounting of what drives its exact size): if this
ever flips, that is a real behavior change (to heliogram.codec's grid-sizing/padding math, or to
heliogram.instruments.saliency's own per-position aggregation) -- re-run the comparison and update
BOTH the assertion and this docstring's numbers, don't silently loosen or delete the check.
"""

from __future__ import annotations


import numpy as np
import pytest

from heliogram.corruption import crop_pad, jpeg_compress
from heliogram.instruments.saliency import (
    SaliencyMap,
    format_saliency,
    position_error_map,
    summarize,
)

# --- import-time boundary ------------------------------------------------------------------------


def test_import_saliency_does_not_pull_in_torch():
    from tests.conftest import assert_import_stays_torch_free

    assert_import_stays_torch_free("heliogram.instruments.saliency")


# --- SaliencyMap shape / structure -----------------------------------------------------------


def test_map_shape_equals_height_by_width():
    m = position_error_map(palette=8, payload_size=128, subpatch=1, trials=2, seed=0)
    assert isinstance(m, SaliencyMap)
    assert m.error_by_position.shape == (m.height, m.width)
    assert m.palette == 8
    assert m.subpatch == 1
    assert m.trials == 2


def test_shape_holds_for_subpatch_2_too():
    m = position_error_map(palette=16, payload_size=256, subpatch=2, trials=2, seed=0)
    assert m.error_by_position.shape == (m.height, m.width)  # one cell per PATCH, not per sub-cell


def test_row0_is_nan_calibration_rows_1plus_are_real_bounded_rates():
    """Row 0 (calibration -- see heliogram.codec's module docstring) carries no comparable
    symbol of its own; SaliencyMap documents this as NaN, never a fabricated 0.0 or an average
    that silently mixes in a not-really-a-symbol row."""
    m = position_error_map(palette=8, payload_size=128, subpatch=1, trials=2, seed=0)
    assert np.all(np.isnan(m.error_by_position[0, :]))
    assert not np.any(np.isnan(m.error_by_position[1:, :]))
    assert np.all(m.error_by_position[1:, :] >= 0.0)
    assert np.all(m.error_by_position[1:, :] <= 1.0)


def test_trials_must_be_at_least_one():
    with pytest.raises(ValueError, match="trials"):
        position_error_map(trials=0)


# --- determinism -----------------------------------------------------------------------------


def test_deterministic_under_fixed_seed():
    m1 = position_error_map(palette=16, payload_size=256, subpatch=1, trials=3, seed=7)
    m2 = position_error_map(palette=16, payload_size=256, subpatch=1, trials=3, seed=7)
    assert np.array_equal(m1.error_by_position, m2.error_by_position, equal_nan=True)
    assert m1.corruption == m2.corruption
    assert (m1.width, m1.height) == (m2.width, m2.height)


def test_different_seed_can_change_the_map():
    """Not every position need differ, but the WHOLE map should not be byte-identical across two
    unrelated seeds for a corruption known to produce nonzero, payload-dependent error (an
    above-half-patch-threshold crop_pad shift -- see module docstring's "cliff" finding: which
    positions coincidentally match truth there depends on the specific random payload)."""
    kwargs = dict(
        palette=64,
        corruption=lambda img: crop_pad(img, dx=8, dy=8),
        payload_size=1024,
        subpatch=1,
        trials=3,
    )
    m1 = position_error_map(seed=0, **kwargs)
    m2 = position_error_map(seed=999, **kwargs)
    assert not np.array_equal(m1.error_by_position, m2.error_by_position, equal_nan=True)


# --- the "cliff, not a gradient" finding (see module docstring) --------------------------------


def test_small_shift_below_half_patch_threshold_gives_zero_error_everywhere():
    """One half of the module docstring's "cliff, not a gradient" finding: a crop_pad shift at
    or below patch_size // 2 (7px here) moves every sampled center -- calibration AND data alike
    -- within its own original solid-color patch. Zero symbol error, everywhere, no exceptions.
    This also pins that heliogram.harness's own realistic corruption suite (crop_pad_2px, max 6px
    even in its diagnostic stress suite) never meaningfully exercises this codec's crop_pad
    failure mode at all -- consistent with RESULTS.md's absorption note."""
    m = position_error_map(
        palette=64,
        corruption=lambda img: crop_pad(img, dx=6, dy=6),
        corruption_name="crop_pad_6px",
        payload_size=1024,
        subpatch=1,
        trials=3,
        seed=0,
    )
    assert np.all(m.error_by_position[1:, :] == 0.0)


def test_crop_pad_shifted_column_has_higher_error_than_interior_columns():
    """THE real, checkable geometric property this test file is named for.

    MECHANISM (see heliogram.instruments.saliency's own module docstring for the full
    derivation): heliogram.codec.encode's calibration row (row 0) is ALWAYS full patch_size-px
    patches regardless of `subpatch` -- only DATA cells are subdivided into k x k sub-cells. That
    gives calibration a patch_size // 2 = 7px shift tolerance and, at subpatch=2 with
    patch_size=14, data sub-cells only a (patch_size // 2) // 2 = 3px tolerance. dx=4 (> 3, <= 7)
    shifts data sub-cells while leaving calibration fully, correctly recoverable, so a
    misclassified sub-cell is compared against the TRUE palette, not a self-corrupted one. In
    that window, DATA COLUMN 0 has no valid left neighbor at all -- only crop_pad's fill color,
    classified against the still-correct palette -- and so can never benefit from a
    shifted-but-real neighbor coincidentally matching truth, unlike every other column, which
    reads a REAL (if displaced) neighboring symbol that occasionally does match.

    HONEST accounting of what drives the SIZE of the measured gap (not just its direction):
    heliogram.codec.encode pads unused grid capacity with symbol 0 (see that function's
    docstring), and for THIS payload size that padding lands in the last data row -- where two
    neighboring zero-padded cells trivially "match" under the column shift. Column 0 is never
    part of that padding, so it can never collect that particular bonus either. Both facts (column
    0's structural inability to read a valid neighbor, AND this payload size's own padding
    location) push the SAME direction; disentangling them is not needed to check the property
    the module docstring claims -- see test_small_shift_below_half_patch_threshold_gives_
    zero_error_everywhere above for proof this is not simply "any crop_pad always does this".

    MEASURED (seed=0, palette=64, subpatch=2, payload_size=1024B, nsym=32, patch_size=14,
    trials=5, crop_pad(dx=4, dy=2)): column-0 mean error ~0.943, interior (columns 10..53 of 64)
    mean error ~0.853 -- column 0 higher by a wide, comfortably-reproducible margin (>0.05, not a
    hair's breadth). heliogram.instruments.saliency.summarize's own crop_pad-specific check
    agrees. If this ever flips, that's a real change (to the grid math or the padding scheme) --
    re-derive, don't silently loosen this.
    """
    m = position_error_map(
        palette=64,
        corruption=lambda img: crop_pad(img, dx=4, dy=2),
        corruption_name="crop_pad_4px",
        payload_size=1024,
        subpatch=2,
        trials=5,
        seed=0,
    )
    data = m.error_by_position[1:, :]
    edge_col0 = float(data[:, 0].mean())
    interior = float(data[:, 10:-10].mean())
    assert edge_col0 > interior  # strictly higher -- see docstring for the measured margin
    assert edge_col0 - interior > 0.05  # a real, comfortably-sized gap, not sampling noise

    s = summarize(m)
    assert s["crop_pad_left_edge_elevated"] is True
    assert s["crop_pad_left_col_mean_error"] > s["crop_pad_col_interior_mean_error"]


# --- summarize() ---------------------------------------------------------------------------------


def test_summarize_worst_position_and_rate_match_the_map():
    m = position_error_map(
        palette=64,
        corruption=lambda img: crop_pad(img, dx=4, dy=2),
        payload_size=1024,
        subpatch=2,
        trials=5,
        seed=0,
    )
    s = summarize(m)
    row, col = s["worst_position"]
    assert 1 <= row <= m.height - 1  # a real DATA row, never the calibration row (0)
    assert 0 <= col <= m.width - 1
    assert s["worst_rate"] == pytest.approx(float(m.error_by_position[row, col]))
    assert s["worst_rate"] == pytest.approx(float(np.nanmax(m.error_by_position)))
    assert 0.0 <= s["mean_error"] <= 1.0


def test_summarize_omits_crop_pad_keys_for_non_crop_pad_corruption():
    m = position_error_map(
        palette=8,
        corruption=lambda img: jpeg_compress(img, quality=70),
        corruption_name="jpeg_q70",
        payload_size=128,
        trials=2,
        seed=0,
    )
    s = summarize(m)
    assert "crop_pad_left_edge_elevated" not in s
    assert "crop_pad_top_edge_elevated" not in s
    assert {"mean_error", "worst_position", "worst_rate", "edge_mean_error", "interior_mean_error"} <= s.keys()


def test_summarize_includes_crop_pad_keys_when_corruption_name_says_so():
    m = position_error_map(
        palette=8,
        corruption=lambda img: crop_pad(img, dx=2, dy=2),
        corruption_name="crop_pad_2px",
        payload_size=128,
        trials=2,
        seed=0,
    )
    s = summarize(m)
    assert "crop_pad_left_edge_elevated" in s
    assert "crop_pad_top_edge_elevated" in s
    assert isinstance(s["crop_pad_left_edge_elevated"], bool)
    assert isinstance(s["crop_pad_top_edge_elevated"], bool)


def test_summarize_raises_on_a_map_with_no_data_rows():
    """height < 2 (only a calibration row, no data) would leave summarize() nothing to reduce --
    it must raise rather than silently return NaNs / an empty result."""
    m = position_error_map(palette=8, payload_size=128, trials=1, seed=0)
    starved = SaliencyMap(
        palette=m.palette,
        subpatch=m.subpatch,
        corruption=m.corruption,
        patch_size=m.patch_size,
        width=m.width,
        height=1,
        trials=m.trials,
        error_by_position=m.error_by_position[:1, :],  # calibration row only
        note="synthetic: no data rows",
    )
    with pytest.raises(ValueError, match="data rows"):
        summarize(starved)


# --- format_saliency -------------------------------------------------------------------------


def test_format_saliency_returns_readable_string_with_headline_numbers():
    m = position_error_map(palette=8, payload_size=128, trials=2, seed=0)
    text = format_saliency(m)
    assert isinstance(text, str)
    assert "palette=8" in text
    assert "mean error" in text.lower()
    assert "worst position" in text.lower()


def test_format_saliency_includes_crop_pad_line_only_for_crop_pad():
    m_crop = position_error_map(
        palette=8, corruption=lambda img: crop_pad(img, dx=2, dy=2),
        corruption_name="crop_pad_2px", payload_size=128, trials=2, seed=0,
    )
    m_jpeg = position_error_map(
        palette=8, corruption=lambda img: jpeg_compress(img, quality=70),
        corruption_name="jpeg_q70", payload_size=128, trials=2, seed=0,
    )
    assert "crop_pad check" in format_saliency(m_crop)
    assert "crop_pad check" not in format_saliency(m_jpeg)
