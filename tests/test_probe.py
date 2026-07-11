"""Tests for heliogram.probe -- the CPU-testable half of the Step-0 frozen-encoder linear probe.

Everything here runs without torch/transformers/GPU. The crown jewel is the end-to-end
synthetic-encoder test at the bottom: it encodes REAL heliogram grids, computes merged-token
"embeddings" with a deterministic stand-in encoder that mimics exactly the raster token order
scripts/run_probe.py assumes of the real tower, and checks the probe reads the symbols back at
~zero error -- which exercises merged_token_labels' ordering contract against the actual codec
layout, not just against itself."""

from __future__ import annotations

import math

import numpy as np
import pytest

from heliogram.codec import PATCH_SIZE, encode, extract_symbols
from heliogram.dataset import pad_to_even_patch_grid
from heliogram.probe import (
    ProbeCellReport,
    evaluate_cell,
    fit_linear_probe,
    format_report,
    merged_token_labels,
    rs_symbol_error_budget,
)


# --- rs_symbol_error_budget ---------------------------------------------------------------------


def test_rs_budget_matches_hand_computed_value():
    # floor(32/2)=16 correctable bytes per 255-byte chunk
    assert rs_symbol_error_budget(32) == pytest.approx(16 / 255)
    assert rs_symbol_error_budget(16) == pytest.approx(8 / 255)


# --- merged_token_labels ------------------------------------------------------------------------


def test_merged_token_labels_hand_computed_small_grid():
    """4x4 patch grid (even), subpatch=1: 12 data symbols over rows 1..3. Merged 2x2 grid has
    4 tokens; hand-derive every position. Data symbol index for patch (row, col), row>=1, is
    (row-1)*width + col."""
    width, height = 4, 4
    symbols = list(range(100, 112))  # 12 distinct sentinels
    labels = merged_token_labels(width, height, symbols, merge=2)
    assert labels.shape == (4, 4)
    # token 0 covers patches (0,0),(0,1),(1,0),(1,1): row 0 is calibration -> -1, -1
    assert labels[0].tolist() == [-1, -1, 100, 101]
    # token 1 covers (0,2),(0,3),(1,2),(1,3)
    assert labels[1].tolist() == [-1, -1, 102, 103]
    # token 2 covers rows 2-3, cols 0-1: symbols (2-1)*4+0.. and (3-1)*4+0..
    assert labels[2].tolist() == [104, 105, 108, 109]
    # token 3 covers rows 2-3, cols 2-3
    assert labels[3].tolist() == [106, 107, 110, 111]


def test_merged_token_labels_rejects_odd_grid_and_bad_lengths():
    with pytest.raises(ValueError, match="divisible by merge"):
        merged_token_labels(3, 4, [0] * 9, merge=2)
    with pytest.raises(ValueError, match="expected width"):
        merged_token_labels(4, 4, [0] * 5, merge=2)


# --- fit_linear_probe ---------------------------------------------------------------------------


def _synthetic_cell(palette: int, n_tokens: int, dim: int, seed: int, noise: float):
    """Synthetic embeddings that DO linearly encode 4 labels per token: one one-hot block per
    position, mixed through a fixed random projection, plus Gaussian noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, palette, size=(n_tokens, 4))
    onehot = np.zeros((n_tokens, 4 * palette))
    for k in range(4):
        onehot[np.arange(n_tokens), k * palette + y[:, k]] = 1.0
    proj = np.random.default_rng(1234).normal(size=(4 * palette, dim))  # fixed across splits
    X = onehot @ proj + noise * rng.normal(size=(n_tokens, dim))
    return X, y


def test_probe_learns_linearly_separable_synthetic_embeddings():
    palette = 16
    X_tr, y_tr = _synthetic_cell(palette, 1500, 96, seed=0, noise=0.05)
    X_te, y_te = _synthetic_cell(palette, 500, 96, seed=1, noise=0.05)
    fit = fit_linear_probe(X_tr, y_tr, X_te, y_te, n_classes=palette, seed=0, epochs=40)
    assert fit.symbol_error < 0.05
    assert fit.train_symbol_error < 0.05
    assert fit.n_test_positions == 500 * 4


def test_probe_scores_near_chance_on_pure_noise():
    """Labels carry no signal -> error must sit near chance (1 - 1/P), NOT near zero: the probe
    must be incapable of inventing separability that is not in the embeddings."""
    palette = 16
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(1200, 64))
    y_tr = rng.integers(0, palette, size=(1200, 4))
    X_te = rng.normal(size=(400, 64))
    y_te = rng.integers(0, palette, size=(400, 4))
    fit = fit_linear_probe(X_tr, y_tr, X_te, y_te, n_classes=palette, seed=0, epochs=20)
    chance = 1 - 1 / palette
    assert fit.symbol_error > 0.75 * chance


def test_probe_ignores_negative_labels_and_is_deterministic():
    palette = 8
    X_tr, y_tr = _synthetic_cell(palette, 600, 48, seed=2, noise=0.05)
    X_te, y_te = _synthetic_cell(palette, 200, 48, seed=3, noise=0.05)
    y_tr[:, 0] = -1  # entire position 0 excluded from training
    y_te[:, 0] = -1
    a = fit_linear_probe(X_tr, y_tr, X_te, y_te, n_classes=palette, seed=7, epochs=30)
    b = fit_linear_probe(X_tr, y_tr, X_te, y_te, n_classes=palette, seed=7, epochs=30)
    assert a.symbol_error == b.symbol_error  # deterministic
    assert math.isnan(a.per_position_error[0])  # no valid test labels at position 0
    assert a.n_test_positions == 200 * 3
    assert a.symbol_error < 0.05  # remaining positions still learned


def test_probe_raises_on_all_invalid_train_labels():
    X = np.zeros((10, 4))
    y = np.full((10, 4), -1)
    with pytest.raises(ValueError, match="no valid"):
        fit_linear_probe(X, y, X, y, n_classes=4)


# --- verdicts / report --------------------------------------------------------------------------


def test_cell_report_verdict_thresholds():
    palette = 128
    X_tr, y_tr = _synthetic_cell(palette, 2000, 128, seed=4, noise=0.02)
    X_te, y_te = _synthetic_cell(palette, 600, 128, seed=5, noise=0.02)
    cell = evaluate_cell(palette, "clean", X_tr, y_tr, X_te, y_te, seed=0, epochs=40)
    assert cell.rs_budget == pytest.approx(16 / 255)
    assert cell.chance_error == pytest.approx(1 - 1 / 128)
    assert cell.fit.symbol_error <= cell.rs_budget
    assert cell.verdict.startswith("BELOW RS BUDGET")

    report = format_report([cell], model_id="synthetic-test-encoder")
    assert "synthetic-test-encoder" in report
    assert "BELOW RS BUDGET" in report
    assert f"| {palette} | clean |" in report


def test_near_chance_verdict_mentions_token_order_check():
    fit_like = evaluate_cell(
        16,
        "clean",
        np.random.default_rng(0).normal(size=(800, 32)),
        np.random.default_rng(1).integers(0, 16, size=(800, 4)),
        np.random.default_rng(2).normal(size=(300, 32)),
        np.random.default_rng(3).integers(0, 16, size=(300, 4)),
        epochs=15,
    )
    assert "chance" in fit_like.verdict
    assert "token-order" in fit_like.verdict


# --- end-to-end with real codec grids and a synthetic raster-order encoder ----------------------


def _synthetic_tower(img, palette, merge=2, noise=0.05, seed=0):
    """Stand-in for the frozen vision tower: for each merged token (raster order over the merged
    grid -- the SAME order scripts/run_probe.py assumes of Qwen's merger), emit the concatenated
    nearest-palette one-hot of its merge*merge patches' mean RGB, plus noise. One-hot (rather
    than raw RGB) keeps the test about the ORDERING CONTRACT, not about linear separability of
    the HSV hue ring (raw-RGB 16-way separation converges too slowly for a fast unit test).
    Features are still derived purely from IMAGE PIXELS in raster order, so if
    merged_token_labels' order contract disagreed with this raster order, feature/label pairs
    would scramble and the probe below would score at chance -- the contract stays pinned end to
    end against the real codec layout."""
    from heliogram.codec import get_palette

    colors = np.asarray(get_palette(palette), dtype=np.float64)
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    h_p = img.height // PATCH_SIZE
    w_p = img.width // PATCH_SIZE
    feats = []
    for mr in range(h_p // merge):
        for mc in range(w_p // merge):
            parts = []
            for dr in range(merge):
                for dc in range(merge):
                    r0 = (mr * merge + dr) * PATCH_SIZE
                    c0 = (mc * merge + dc) * PATCH_SIZE
                    mean = arr[r0 : r0 + PATCH_SIZE, c0 : c0 + PATCH_SIZE].mean(axis=(0, 1))
                    nearest = int(((colors - mean) ** 2).sum(axis=1).argmin())
                    onehot = np.zeros(palette)
                    onehot[nearest] = 1.0
                    parts.append(onehot)
            feats.append(np.concatenate(parts))
    X = np.stack(feats)
    if noise:
        X = X + noise * np.random.default_rng(seed).normal(size=X.shape)
    return X


def _real_grid_cell(palette, seeds, noise=0.05):
    xs, ys = [], []
    for s in seeds:
        payload = bytes(np.random.default_rng(s).integers(0, 256, size=96, dtype=np.uint8))
        img = pad_to_even_patch_grid(encode(payload, palette=palette), PATCH_SIZE, palette)
        w, h = img.width // PATCH_SIZE, img.height // PATCH_SIZE
        _, _, symbols = extract_symbols(img, palette=palette)
        ys.append(merged_token_labels(w, h, symbols))
        xs.append(_synthetic_tower(img, palette, noise=noise, seed=s))
    return np.concatenate(xs), np.concatenate(ys)


def test_end_to_end_real_grids_synthetic_encoder_reads_symbols_back():
    """Real encode() grids + the raster-order synthetic tower: the probe must read the symbols
    back at ~zero error. This is the CPU rehearsal of exactly what scripts/run_probe.py does on
    a GPU -- it validates the label/token ordering contract against the real codec, so a real
    run that scores at chance on clean images indicts the TOWER's token order (or the
    preprocessing), not this bookkeeping."""
    palette = 16
    X_tr, y_tr = _real_grid_cell(palette, seeds=(0, 1, 2))
    X_te, y_te = _real_grid_cell(palette, seeds=(10, 11))
    fit = fit_linear_probe(X_tr, y_tr, X_te, y_te, n_classes=palette, seed=0, epochs=60)
    assert fit.symbol_error < 0.01
    assert fit.n_test_positions > 0
