"""Pytest suite for heliogram.instruments.bayes_bound (Group E: "is large-palette-under-JPEG
information-theoretically dead, or just too dumb a decoder to read it?").

Assert-based, no fixtures/frameworks beyond plain pytest, matching the rest of this repo's test
idiom. Like tests/test_saliency.py/test_fingerprint.py/test_foreign_tile.py, this file's
import-time test proves heliogram.instruments.bayes_bound never needs torch/transformers merely
to be imported or to run its (no real model) code path -- everything it measures comes from
heliogram.codec's reference pixel decoder machinery and heliogram.harness's own corruption suite,
reused, never a VLM.

Kept FAST (this is the "fast CPU subset", not the full default sweep the module's own CLI runs):
every `bayes_bound_cell`/`run` call below uses `n_images=2` (the minimum: 1 train + 1 test image)
and small (<=256B) payloads, so the whole file runs in well under a minute -- see
heliogram.instruments.bayes_bound's own module docstring / CLI for the full default sweep this
file does NOT attempt to reproduce.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from heliogram.codec import RS_NSIZE, encode, extract_symbols, PATCH_SIZE
from heliogram.corruption import jpeg_compress
from heliogram.dataset import random_payload
from heliogram.harness import CORRUPTIONS, NSYM
from heliogram.instruments.bayes_bound import (
    BayesBoundCell,
    DEFAULT_CORRUPTION_NAMES,
    DEFAULT_PALETTES,
    GaussianOracle,
    RS_BUDGET,
    bayes_bound_cell,
    build_parser,
    calibration_nn_whole_patch,
    fit_gaussian_oracle,
    format_table,
    main,
    predict_gaussian_oracle,
    run,
    whole_patch_means,
)

# --- import-time boundary ------------------------------------------------------------------------


def test_import_bayes_bound_does_not_pull_in_torch():
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


# --- RS_BUDGET -------------------------------------------------------------------------------


def test_rs_budget_matches_the_documented_formula():
    """floor(nsym/2)/RS_NSIZE with nsym=heliogram.harness.NSYM(=32) -> 16/255 ~= 0.0627, the
    exact figure the module docstring/task motivation both cite -- pinned here independently of
    the module's own internal computation (recomputed from the public NSYM/RS_NSIZE constants,
    not by re-reading RS_BUDGET back at itself)."""
    expected = (NSYM // 2) / RS_NSIZE
    assert RS_BUDGET == pytest.approx(expected)
    assert RS_BUDGET == pytest.approx(16 / 255)
    assert 0.0 < RS_BUDGET < 1.0


def test_default_corruption_names_are_real_harness_corruptions():
    """DEFAULT_CORRUPTION_NAMES must be reused keys of heliogram.harness.CORRUPTIONS, never a
    locally-redefined JPEG quality/composition -- see the module docstring's DATA HONESTY note."""
    for name in DEFAULT_CORRUPTION_NAMES:
        assert name in CORRUPTIONS
    assert DEFAULT_PALETTES  # non-empty, sane default sweep dimension


# --- whole_patch_means -------------------------------------------------------------------------


def test_whole_patch_means_shape_and_row0_is_calibration():
    payload = random_payload(0, 64)
    img = encode(payload, palette=8, patch_size=PATCH_SIZE, nsym=32, seed=0, subpatch=1)
    width, height, means = whole_patch_means(img, PATCH_SIZE)
    assert means.shape == (height, width, 3)
    # row 0 is the calibration row: every column i's mean should be very close to one of
    # heliogram.codec.get_palette(8)'s colors (clean image, no corruption -- exact solid patches).
    from heliogram.codec import get_palette

    colors = np.array(get_palette(8), dtype=np.float64)
    for i in range(width):
        dist = np.min(np.sum((colors - means[0, i]) ** 2, axis=1))
        assert dist < 1e-6  # clean patch, whole-patch mean == the solid color exactly


def test_whole_patch_means_rejects_too_small_image():
    from PIL import Image

    tiny = Image.new("RGB", (4, 4), (0, 0, 0))
    with pytest.raises(ValueError, match="too small"):
        whole_patch_means(tiny, PATCH_SIZE)


# --- calibration_nn_whole_patch (Classifier 1) ------------------------------------------------


def test_calibration_nn_whole_patch_shape_matches_extract_symbols():
    payload = random_payload(1, 128)
    img = encode(payload, palette=16, patch_size=PATCH_SIZE, nsym=32, seed=0, subpatch=1)
    w1, h1, truth = extract_symbols(img, palette=16, patch_size=PATCH_SIZE, subpatch=1)
    w2, h2, symbols = calibration_nn_whole_patch(img, 16, PATCH_SIZE)
    assert (w1, h1) == (w2, h2)
    assert len(symbols) == len(truth)


def test_calibration_nn_whole_patch_is_exact_on_a_clean_image():
    """On a CLEAN (uncorrupted) image every patch is a single solid color, so whole-patch mean ==
    center pixel == the exact palette color -- Classifier 1 must recover the ground truth exactly,
    zero error, same as decode_pixels/extract_symbols on a clean image."""
    payload = random_payload(2, 256)
    img = encode(payload, palette=64, patch_size=PATCH_SIZE, nsym=32, seed=0, subpatch=1)
    _, _, truth = extract_symbols(img, palette=64, patch_size=PATCH_SIZE, subpatch=1)
    _, _, observed = calibration_nn_whole_patch(img, 64, PATCH_SIZE)
    assert observed == truth


# --- GaussianOracle (Classifier 2) --------------------------------------------------------------


def test_gaussian_oracle_perfectly_separates_well_separated_synthetic_classes():
    """Sanity-checks the Mahalanobis-distance math itself, independent of any heliogram image:
    three classes with means far apart relative to a small pooled variance must classify their
    own (noisy) training-distribution samples correctly -- if this fails, the oracle's fit/predict
    arithmetic itself is broken, not just imprecise on real corrupted pixels."""
    palette = 4  # must be a valid heliogram.codec palette size (VALID_PALETTES)
    rng = np.random.RandomState(0)
    means = np.array(
        [[10.0, 10.0, 10.0], [90.0, 90.0, 90.0], [170.0, 170.0, 170.0], [250.0, 250.0, 250.0]]
    )
    train_feats = []
    labels = []
    feats = []
    for c in range(palette):
        pts = means[c] + rng.normal(scale=2.0, size=(20, 3))
        feats.append(pts)
        labels.extend([c] * 20)
    features = np.concatenate(feats, axis=0)
    train_feats.append((features, labels))

    oracle = fit_gaussian_oracle(train_feats, palette)
    assert isinstance(oracle, GaussianOracle)
    assert oracle.n_missing_classes == 0
    assert oracle.n_train_patches == 20 * palette

    test_feats = []
    test_labels = []
    for c in range(palette):
        pts = means[c] + rng.normal(scale=2.0, size=(10, 3))
        test_feats.append(pts)
        test_labels.extend([c] * 10)
    predicted = predict_gaussian_oracle(oracle, np.concatenate(test_feats, axis=0))
    errors = sum(1 for p, t in zip(predicted, test_labels) if p != t)
    assert errors == 0  # well-separated classes, small noise -- must be exact


def test_fit_gaussian_oracle_missing_class_falls_back_to_get_palette_color():
    """A symbol value that never appears in the training split gets its class_means entry set to
    heliogram.codec.get_palette's clean reference color -- documented explicitly as a
    conservative (never over-optimistic) fallback, see fit_gaussian_oracle's docstring."""
    from heliogram.codec import get_palette

    palette = 4
    # Only classes 0 and 1 ever appear in this synthetic training set -- 2 and 3 are missing.
    features = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [250.0, 250.0, 250.0]])
    labels = [0, 0, 1]
    oracle = fit_gaussian_oracle([(features, labels)], palette)
    assert oracle.n_missing_classes == 2
    colors = np.array(get_palette(palette), dtype=np.float64)
    assert np.allclose(oracle.class_means[2], colors[2])
    assert np.allclose(oracle.class_means[3], colors[3])
    assert oracle.class_counts[2] == 0 and oracle.class_counts[3] == 0


# --- bayes_bound_cell / run (fast subset) -------------------------------------------------------


def test_bayes_bound_cell_structure_and_sanity_bounds():
    """The fast-subset smoke test: palette=32, one corruption (jpeg_q85), n_images=2, a small
    payload -- must finish quickly and produce sane, bounded numbers."""
    cell = bayes_bound_cell(32, "jpeg_q85", n_images=2, payload_size=128, seed=0)
    assert isinstance(cell, BayesBoundCell)
    assert cell.palette == 32
    assert cell.corruption == "jpeg_q85"
    assert cell.n_images == 2
    assert cell.n_train_images + cell.n_test_images == cell.n_images
    assert cell.n_train_images >= 1 and cell.n_test_images >= 1

    for rate in (
        cell.center_pixel_nn_error,
        cell.whole_patch_nn_error,
        cell.gaussian_oracle_error,
        cell.best_error,
    ):
        assert 0.0 <= rate <= 1.0

    assert cell.rs_budget == pytest.approx(RS_BUDGET)
    assert cell.best_error == pytest.approx(
        min(cell.center_pixel_nn_error, cell.whole_patch_nn_error, cell.gaussian_oracle_error)
    )
    assert cell.best_classifier in ("center_pixel_nn", "whole_patch_nn", "gaussian_oracle")
    if cell.best_error < RS_BUDGET:
        assert cell.verdict == "information present -- a better reader could work"
    else:
        assert cell.verdict == "information likely destroyed at this operating point"
    assert isinstance(cell.note, str) and cell.note


def test_bayes_bound_cell_requires_at_least_two_images():
    with pytest.raises(ValueError, match="n_images"):
        bayes_bound_cell(32, "jpeg_q85", n_images=1, payload_size=64)
    with pytest.raises(ValueError, match="n_images"):
        bayes_bound_cell(32, "jpeg_q85", n_images=0, payload_size=64)


def test_bayes_bound_cell_rejects_unknown_corruption_name():
    with pytest.raises(ValueError, match="CORRUPTIONS"):
        bayes_bound_cell(32, "not_a_real_corruption", n_images=2, payload_size=64)


def test_bayes_bound_cell_deterministic_under_fixed_seed():
    c1 = bayes_bound_cell(32, "jpeg_q85", n_images=2, payload_size=128, seed=3)
    c2 = bayes_bound_cell(32, "jpeg_q85", n_images=2, payload_size=128, seed=3)
    assert c1 == c2


def test_run_produces_one_cell_per_palette_x_corruption_pair():
    palettes = [32, 64]
    corruptions = ["jpeg_q85", "jpeg_q70"]
    cells = run(
        palettes=palettes,
        corruption_names=corruptions,
        n_images=2,
        payload_size=64,
        seed=0,
    )
    assert len(cells) == len(palettes) * len(corruptions)
    seen = {(c.palette, c.corruption) for c in cells}
    assert seen == {(p, name) for p in palettes for name in corruptions}


# --- the "whole-patch beats center-pixel under JPEG" premise ------------------------------------


def test_whole_patch_error_materially_lower_than_center_pixel_under_jpeg_q70_at_palette_256():
    """THE targeted assertion: this instrument's whole point is that averaging over the FULL
    patch dominates decode_pixels' single-center-pixel sample under pure JPEG quantization/
    chroma-subsampling noise (see module docstring). At palette=256 (the densest, most fragile
    palette) under jpeg_q70 (the realistic-envelope corruption RESULTS.md already measures
    decode_pixels failing badly at), a single small image must show the whole-patch classifier
    strictly beating the center-pixel one -- tolerant (strictly less, not a fixed margin), per the
    work-order's own instruction, since the exact gap is sampling-noise-sensitive at n=1 image but
    the DIRECTION is not (see this module's own measured default sweep for the same effect at
    larger, averaged sample sizes)."""
    payload = random_payload(0, 256)
    clean = encode(payload, palette=256, patch_size=PATCH_SIZE, nsym=32, seed=0, subpatch=1)
    _, _, truth = extract_symbols(clean, palette=256, patch_size=PATCH_SIZE, subpatch=1)
    corrupted = jpeg_compress(clean, quality=70)

    _, _, center_observed = extract_symbols(
        corrupted, palette=256, patch_size=PATCH_SIZE, subpatch=1
    )
    _, _, whole_observed = calibration_nn_whole_patch(corrupted, 256, PATCH_SIZE)

    n = min(len(truth), len(center_observed), len(whole_observed))
    center_err = sum(1 for i in range(n) if truth[i] != center_observed[i]) / n
    whole_err = sum(1 for i in range(n) if truth[i] != whole_observed[i]) / n

    assert whole_err < center_err  # strictly lower -- the instrument's own premise, demonstrated


# --- the honest "translation reversal" this module's own docstring documents -------------------


def test_whole_patch_error_can_be_worse_than_center_pixel_under_translation():
    """The flip side, pinned so it cannot silently regress into being hidden/forgotten: a small
    crop_pad shift (well inside decode_pixels' own half-patch tolerance -- see
    heliogram.instruments.saliency's module docstring on the same "cliff" mechanism) leaves
    center-pixel sampling completely unaffected but corrupts whole-patch mean everywhere, because
    every patch's edge pixels now belong to a neighboring patch. This is exactly why
    BayesBoundCell.best_error/verdict are computed over ALL THREE classifiers (center-pixel NN
    included), not just the two whole-patch ones -- see this module's own docstring's "MEASURED,
    HONEST REVERSAL" section."""
    from heliogram.corruption import crop_pad

    payload = random_payload(0, 1024)
    clean = encode(payload, palette=32, patch_size=PATCH_SIZE, nsym=32, seed=0, subpatch=1)
    _, _, truth = extract_symbols(clean, palette=32, patch_size=PATCH_SIZE, subpatch=1)
    shifted = crop_pad(clean, dx=2, dy=2)

    _, _, center_observed = extract_symbols(shifted, palette=32, patch_size=PATCH_SIZE, subpatch=1)
    _, _, whole_observed = calibration_nn_whole_patch(shifted, 32, PATCH_SIZE)

    n = min(len(truth), len(center_observed), len(whole_observed))
    center_err = sum(1 for i in range(n) if truth[i] != center_observed[i]) / n
    whole_err = sum(1 for i in range(n) if truth[i] != whole_observed[i]) / n

    assert center_err == 0.0  # a 2px shift is far below patch_size//2's tolerance -- see saliency
    assert whole_err > center_err  # whole-patch mean is NOT robust to translation, unlike JPEG


# --- format_table ----------------------------------------------------------------------------


def test_format_table_contains_headers_rs_budget_and_every_cell():
    cells = run(
        palettes=[32], corruption_names=["jpeg_q85"], n_images=2, payload_size=64, seed=0
    )
    text = format_table(cells)
    assert "RS_BUDGET" in text
    assert "palette" in text and "corruption" in text and "verdict" in text
    assert "32" in text
    assert "jpeg_q85" in text


# --- CLI: build_parser / main ------------------------------------------------------------------


def test_build_parser_has_sane_defaults_and_needs_no_required_flags():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.palettes == list(DEFAULT_PALETTES)
    assert args.corruptions == list(DEFAULT_CORRUPTION_NAMES)
    assert args.n_images >= 2
    assert args.out is None


def test_main_runs_end_to_end_fast_subset_and_prints_table(capsys):
    rc = main(
        [
            "--palettes",
            "32",
            "--corruptions",
            "jpeg_q85",
            "--n-images",
            "2",
            "--payload-size",
            "64",
            "--seed",
            "0",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "RS_BUDGET" in out
    assert "MANDATORY CAVEAT" in out


def test_main_rejects_unknown_corruption_name(capsys):
    rc = main(["--palettes", "32", "--corruptions", "not_a_real_corruption"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown name" in err


def test_main_writes_out_file_when_requested(tmp_path):
    out_path = tmp_path / "bayes_bound.md"
    rc = main(
        [
            "--palettes",
            "32",
            "--corruptions",
            "jpeg_q85",
            "--n-images",
            "2",
            "--payload-size",
            "64",
            "--out",
            str(out_path),
        ]
    )
    assert rc == 0
    assert out_path.exists()
    content = out_path.read_text()
    assert "RS_BUDGET" in content
