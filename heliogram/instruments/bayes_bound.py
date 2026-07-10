"""heliogram.instruments.bayes_bound -- is large-palette-under-JPEG information-theoretically
dead, or just too dumb a decoder to read it?

WHY THIS MODULE EXISTS: `heliogram.codec.decode_pixels` samples exactly ONE pixel -- the
geometric center -- of every `patch_size` x `patch_size` patch, and classifies that single pixel
by nearest-neighbor RGB distance against calibration colors recovered the same way (also from a
single center-pixel sample per row-0 calibration patch). That is a real, cheap, model-free
reference decoder -- but it is also, self-evidently, a LOWER BOUND on what the CHANNEL can carry:
it throws away every pixel in a patch except one, and classifies with the simplest possible rule.
RESULTS.md / `heliogram.codec`'s own DATA HONESTY note measure this exact decoder FAILING at
`palette` in {128, 256} under `jpeg_q70` (symbol error 0.28-0.49, see that module's docstring),
against a Reed-Solomon correction budget of roughly 6.3% (see RS_BUDGET below) -- a large gap.
That measurement, by itself, does NOT prove no decoder -- a fine-tuned VLM included -- can read
those palettes under JPEG: it only proves THIS decoder cannot. Before betting a GPU fine-tune
(Phase 2) on closing that gap, it is worth asking a cheaper, CPU-only question first: is the
color information corruption destroys even IN PRINCIPLE recoverable from the pixels that
survive, or is it gone, full stop, no matter how good the reader is?

THE INSTRUMENT: this module estimates the error rate of (near-)OPTIMAL classifiers operating on
WHOLE-PATCH statistics (not `decode_pixels`' single sampled pixel) of the SAME corrupted images.
Whole-patch mean RGB dominates a center-pixel sample under PURE JPEG quantization/subsampling
noise for a purely geometric reason: JPEG's 8x8 DCT block quantization and chroma subsampling
both introduce noise that varies WITHIN a patch (block-grid misalignment against the patch grid,
subsampled-chroma bleed near patch boundaries, ringing near hard color edges) -- a single sampled
pixel can land squarely in the worst spot that noise produces, while the patch's mean over all
`patch_size**2` pixels averages that noise down substantially (central-limit-theorem intuition,
not a proof -- corruption noise is not i.i.d. across a patch, so the real reduction is smaller
than a naive sqrt(N) bound, but still large and, as this module's own tests measure directly,
real). If even an approximate-Bayes classifier over these whole-patch statistics still sits far
above the RS budget, the color information is very likely physically destroyed by this corruption
at this operating point -- not merely mis-read by a dumb decoder -- and the Phase-2 "fine-tune a
VLM to read P=256 under JPEG" bet is closed a priori for that (palette, corruption) cell. If it
drops below budget, the opposite conclusion holds: `decode_pixels` was simply too crude, the
information is there, and a better reader (a VLM's own, presumably richer, patch features
included) has real room to recover it.

MEASURED, HONEST REVERSAL (read before trusting "whole-patch always wins" as a blanket claim --
it does not, and this module's own default sweep proves it): the geometric argument above is
about QUANTIZATION noise specifically. It says nothing about TRANSLATION. `heliogram.harness`'s
own `combined` corruption composes resize + JPEG q70 + a 2px `crop_pad` -- and a `crop_pad` shift,
however small, moves the WHOLE canvas, which means the pixels nearest a patch's edge, on the side
the shift comes from, are no longer that patch's own color at all -- they are the neighboring
patch's color (or crop_pad's fill), bled in by exactly `dx`/`dy` px. Averaging over the FULL patch
necessarily includes those bled-in edge pixels, so whole-patch mean is NOT robust to translation
the way it is to quantization noise -- quite the opposite: this module's own measurements (see the
default sweep's `combined` rows) show whole-patch mean error rates of 30-85%, dramatically WORSE
than `decode_pixels`' own center-pixel sample, which stays comfortably inside a patch (away from
every boundary) and is therefore completely UNAFFECTED by any shift up to `patch_size // 2` px --
a 2px `crop_pad` is nowhere close to that threshold (see `heliogram.instruments.saliency`'s module
docstring for the same "cliff, not a gradient" finding about `crop_pad` on the reference decoder).
This is a real, measured, and important finding this instrument's own numbers surface, not a
convenient footnote: it is exactly why `bayes_bound_cell`'s `best_error`/`verdict` below are
computed over ALL THREE measured error rates -- `decode_pixels`' own center-pixel NN included --
not just the two whole-patch-feature classifiers this module was built to add. Excluding
center-pixel from "best" would let this module report "information destroyed" for a
(palette, corruption) cell where the reference decoder ITSELF, already running in this repo today,
demonstrably beats the RS budget -- an indefensible verdict this module's own `combined` sweep
rows would otherwise produce; see `BayesBoundCell`'s docstring for how this is handled.

TWO CLASSIFIERS, BOTH OVER WHOLE-PATCH FEATURES (mean RGB over the FULL `patch_size` x
`patch_size` patch -- see above for why this dominates a center-pixel sample under JPEG):

  1. CALIBRATION-NN ORACLE (`calibration_nn_whole_patch`) -- EXACTLY `decode_pixels`' own
     strategy (recover each palette color from row 0, nearest-neighbor classify data patches
     against the recovered colors), upgraded from center-pixel sampling to whole-patch means at
     BOTH steps. Needs no training data or held-out split: like `decode_pixels`, it calibrates
     from the same image's own row 0, nothing more.
  2. APPROX-BAYES ORACLE (`fit_gaussian_oracle` / `predict_gaussian_oracle`) -- a per-class
     Gaussian model (per-class mean, POOLED diagonal covariance shared across classes, UNIFORM
     class prior -- a defensible simplification given RS-coded payload bytes are close to
     uniformly distributed, and one that avoids estimating per-class priors from a training split
     that may see very few examples of some classes at the largest palettes) fit on LABELED
     corrupted patches from a TRAINING split of the images, evaluated on a disjoint, HELD-OUT
     split. This approximates the Bayes-optimal classifier for the (assumed) Gaussian-per-class,
     shared-diagonal-covariance model of the corrupted whole-patch-mean channel -- the closest
     this CPU-only, model-free instrument can get to an actual Bayes-error estimate without a
     real density estimator.

DATA HONESTY: every feature and every ground-truth label here comes from
`heliogram.codec.encode`/`extract_symbols` and `heliogram.harness.CORRUPTIONS` -- the SAME
channel/corruption primitives the rest of this repo measures against, reused (never redefined:
the JPEG qualities and the "combined" resize+JPEG+crop composition below are pulled directly out
of `heliogram.harness.CORRUPTIONS` by name, not re-specified here). There is no
torch/transformers import anywhere in this file, at module scope or otherwise -- both classifiers
are plain numpy (nearest-neighbor / a per-class Gaussian discriminant), nothing here could produce
a number that looks like a VLM measurement. Ground truth for every patch always comes from
`extract_symbols` on the CLEAN, pre-corruption image (exact by construction -- the same convention
`heliogram.harness`/`heliogram.dataset`/every other instrument in this package uses), never
re-derived from a corrupted image.

MANDATORY CAVEAT (read before treating an above-budget Gaussian-oracle result as a proof of
impossibility): both classifiers here see exactly THREE numbers per patch -- the whole-patch mean
R, G, B -- and nothing else. A real VLM's vision encoder sees the FULL spatial pixel grid,
including cross-patch context (neighboring patches, learned features far richer than a per-patch
RGB mean, and whatever the model's own pretraining taught it about JPEG artifacts specifically).
An above-budget Gaussian-oracle result is therefore STRONG evidence that this corruption destroys
color information at this operating point for any per-patch-summary-statistic reader -- but it is
NOT an absolute proof that no reader whatsoever (a VLM with full spatial context included) could
do better by using information this instrument's 3-number-per-patch feature representation
discards. Treat "information likely destroyed" as "the cheap-oracle case for giving up here is
strong," not as "mathematically certain for every possible reader."

RS_BUDGET: `floor(nsym/2) / RS_NSIZE` with `nsym` = `heliogram.harness.NSYM` (32, the codec's own
default parity count) and `RS_NSIZE` = `heliogram.codec.RS_NSIZE` (255, reedsolo's GF(256)
codeword size) -- `16/255 ~= 0.0627`, the same "~6.3%" figure this project already cites (see
`heliogram.benefit.rs_error_correction_capacity`, which computes the identical `nsym // 2`
correctable-byte-count this budget is built from) as roughly the point past which Reed-Solomon can
no longer correct the errors a corrupted channel introduces. HONEST UNIT CAVEAT: this budget is
stated in RS byte-errors per 255-byte codeword, while the error rates this module reports are
per-SYMBOL (a `log2(palette)`-bit unit, not necessarily one byte) classification error rates --
comparing the two treats a symbol error and a byte error as roughly interchangeable, which is the
same approximation this project's own headline motivation numbers already make (see
`heliogram.codec`'s own P=128/256-under-`jpeg_q70` DATA HONESTY note) -- close enough to be a
useful gate, not an exact byte-for-byte equivalence.

CLI (`python3 -m heliogram.instruments.bayes_bound`): runs the default sweep (palette in
{32, 64, 128, 256} x corruption in {jpeg_q85, jpeg_q70, combined}, `n_images=6`,
`payload_size=1024`B) and prints a markdown table; `--out FILE` also writes the same table to
disk. Defaults are sized to finish in a few minutes on CPU -- see `DEFAULT_PALETTES` /
`DEFAULT_CORRUPTION_NAMES` / `DEFAULT_N_IMAGES` below; pass `--n-images` lower for a faster
(noisier) run.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..codec import (
    PATCH_SIZE,
    RS_NSIZE,
    VALID_PALETTES,
    bits_per_symbol,
    encode,
    extract_symbols,
    get_palette,
)
from ..dataset import random_payload
from ..harness import CORRUPTIONS, NSYM

__all__ = [
    "RS_BUDGET",
    "DEFAULT_PALETTES",
    "DEFAULT_CORRUPTION_NAMES",
    "DEFAULT_N_IMAGES",
    "DEFAULT_PAYLOAD_SIZE",
    "whole_patch_means",
    "calibration_nn_whole_patch",
    "GaussianOracle",
    "fit_gaussian_oracle",
    "predict_gaussian_oracle",
    "BayesBoundCell",
    "bayes_bound_cell",
    "run",
    "format_table",
    "build_parser",
    "main",
]

# floor(nsym/2)/RS_NSIZE -- see module docstring's "RS_BUDGET" section for the unit caveat.
# 16/255 ~= 0.062745... for the codec/harness's own default nsym=32.
RS_BUDGET: float = (NSYM // 2) / RS_NSIZE

DEFAULT_PALETTES: Tuple[int, ...] = (32, 64, 128, 256)
# Reused, not redefined -- these must be keys already present in heliogram.harness.CORRUPTIONS
# (see module docstring's DATA HONESTY section). "combined" is that dict's resize 5% + JPEG q70 +
# 2px crop/pad composition -- the project's own worst-realistic-case corruption.
DEFAULT_CORRUPTION_NAMES: Tuple[str, ...] = ("jpeg_q85", "jpeg_q70", "combined")
DEFAULT_N_IMAGES = 6
DEFAULT_PAYLOAD_SIZE = 1024

# Numerical floor for the Gaussian oracle's pooled per-channel variance (squared 0-255 RGB
# units), guarding against a channel dimension that happens to show ~zero within-class variance
# in a small training split (most likely at the largest palettes, where some classes may get very
# few -- even a single -- training samples) blowing up that dimension's Mahalanobis contribution
# to a degenerate near-infinity. This is a numerical safeguard only, not a claim about the true
# corruption distribution -- see `fit_gaussian_oracle`'s docstring.
_MIN_POOLED_VAR = 1.0


def whole_patch_means(
    img: Image.Image, patch_size: int = PATCH_SIZE
) -> Tuple[int, int, np.ndarray]:
    """(width, height, means): means[row, col] is the mean RGB (float64, 3-vector) over the FULL
    `patch_size` x `patch_size` patch at that grid position -- row 0 is the calibration row,
    exactly like `heliogram.codec.extract_symbols`' own row/col convention, INCLUDED here (not
    just the data rows) so `calibration_nn_whole_patch` can read row 0's whole-patch calibration
    colors straight out of the same array.

    THE key departure from `heliogram.codec.extract_symbols`: that function samples one pixel
    (the patch center) per patch; this function averages over every pixel the patch has. See the
    module docstring for why that matters under JPEG. Crops to the largest exact `patch_size`
    multiple, mirroring `extract_symbols`' implicit floor division -- any partial trailing row/
    column of pixels is excluded, never padded or resized.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    img_h, img_w = arr.shape[0], arr.shape[1]
    width = img_w // patch_size
    height = img_h // patch_size
    if width < 1 or height < 2:
        raise ValueError(f"image too small for patch_size={patch_size}: {img.size}")
    cropped = arr[: height * patch_size, : width * patch_size]
    # (height, patch_size, width, patch_size, 3) -> mean-pool each patch_size x patch_size cell.
    grid = cropped.reshape(height, patch_size, width, patch_size, 3)
    means = grid.mean(axis=(1, 3))  # (height, width, 3)
    return width, height, means


def calibration_nn_whole_patch(
    img: Image.Image, palette: int, patch_size: int = PATCH_SIZE
) -> Tuple[int, int, List[int]]:
    """Classifier 1 (the "calibration-NN oracle", see module docstring): EXACTLY
    `heliogram.codec.extract_symbols`' own algorithm -- recover each palette index's reference
    color as the mean of that index's row-0 samples, then nearest-neighbor classify every data
    patch against the recovered colors -- but with `whole_patch_means` (mean RGB over the FULL
    patch) standing in for `extract_symbols`' single sampled center pixel, at BOTH the
    calibration-recovery step and the data-classification step.

    No training data or held-out split needed: like `decode_pixels`, this calibrates from the
    SAME image's own row 0, nothing more -- it is a per-image, unsupervised (in the train/test
    sense) upgrade of the reference decoder's own strategy, not a learned model.

    Returns (width, height, symbols) with the same shape/ordering convention as
    `heliogram.codec.extract_symbols` (data symbols in row-major order, `subpatch=1` only -- this
    module's whole-patch feature never needs to resolve sub-patch structure, and the codec's own
    calibration row is always full `patch_size`-px patches regardless of subpatch anyway).

    Raises ValueError for the same "image too small" case `whole_patch_means` raises for, or if
    `palette` is not a valid `heliogram.codec` palette size.
    """
    bits_per_symbol(palette)  # validates palette is in VALID_PALETTES
    width, height, means = whole_patch_means(img, patch_size)

    sums = np.zeros((palette, 3), dtype=np.float64)
    counts = np.zeros(palette, dtype=np.int64)
    for i in range(width):
        c = i % palette
        sums[c] += means[0, i]
        counts[c] += 1

    fallback = np.array(get_palette(palette), dtype=np.float64)
    have_sample = counts > 0
    recovered = np.where(
        have_sample[:, None], sums / np.maximum(counts, 1)[:, None], fallback
    )

    data_means = means[1:, :, :].reshape(-1, 3)
    dists = ((data_means[:, None, :] - recovered[None, :, :]) ** 2).sum(axis=2)
    symbols = dists.argmin(axis=1).astype(int).tolist()
    return width, height, symbols


@dataclass
class GaussianOracle:
    """A fitted approx-Bayes oracle (Classifier 2, see module docstring): per-class mean whole-
    patch-mean RGB (`class_means`, shape `(palette, 3)`) and one POOLED diagonal covariance shared
    across every class (`pooled_var`, shape `(3,)`) -- i.e. Linear Discriminant Analysis restricted
    to a diagonal shared covariance, which (under a uniform class prior, see `predict_gaussian_
    oracle`) reduces to nearest-centroid classification under a per-channel-reweighted (Mahalanobis)
    distance.

    `class_counts`: how many TRAINING patches actually landed in each class -- `n_missing_classes`
    (classes with `class_counts[c] == 0`) is reported so a caller can see, honestly, how much of
    `class_means` is a real measurement vs. a `heliogram.codec.get_palette` fallback (see
    `fit_gaussian_oracle`'s docstring for what that fallback means and why it can only ever make
    this oracle's measured error rate WORSE, never artificially better).
    """

    palette: int
    class_means: np.ndarray
    pooled_var: np.ndarray
    class_counts: np.ndarray
    n_missing_classes: int
    n_train_patches: int


def fit_gaussian_oracle(
    train_features: Sequence[Tuple[np.ndarray, Sequence[int]]], palette: int
) -> GaussianOracle:
    """Fit `GaussianOracle` from a list of `(features, labels)` pairs -- one pair per TRAINING
    image, `features` an `(n_data_patches, 3)` array of whole-patch-mean RGB (from a CORRUPTED
    image, see `whole_patch_means`) and `labels` the matching ground-truth symbols (from
    `extract_symbols` on that image's CLEAN counterpart -- exact by construction, never re-derived
    from the corrupted image).

    `class_means[c]` is the empirical mean feature vector of every training patch labeled `c`;
    `pooled_var` is the WITHIN-class variance (sum of squared deviations from each patch's own
    class mean, summed across every class and divided by degrees of freedom `n_train_patches -
    n_classes_seen`), pooled across all classes -- the standard shared-diagonal-covariance
    estimate LDA uses, floored at `_MIN_POOLED_VAR` (see that constant's comment) for numerical
    safety only.

    HONEST fallback: a symbol value that never appears in `train_features` at all (possible at the
    largest palettes with a small training split -- see `BayesBoundCell`'s `n_train_images`) gets
    `class_means[c]` set to `heliogram.codec.get_palette(palette)[c]` -- the CLEAN, uncorrupted
    reference color for that index, which the fitted model was never actually shown looking
    corrupted. This can only ever make `predict_gaussian_oracle`'s measured error rate for that
    class WORSE than a real training sample would have (a clean reference color is, if anything, an
    easier target to confuse with a neighboring corrupted class than a properly-fit corrupted
    mean would be) -- i.e. this oracle's reported error rate is a conservative (not
    over-optimistic) approximation of the true Bayes error whenever `n_missing_classes > 0`.
    """
    bits_per_symbol(palette)  # validates palette is in VALID_PALETTES
    sums = np.zeros((palette, 3), dtype=np.float64)
    sqsums = np.zeros((palette, 3), dtype=np.float64)
    counts = np.zeros(palette, dtype=np.int64)

    for features, labels in train_features:
        labels_arr = np.asarray(labels, dtype=np.int64)
        feats_arr = np.asarray(features, dtype=np.float64)
        np.add.at(sums, labels_arr, feats_arr)
        np.add.at(sqsums, labels_arr, feats_arr ** 2)
        np.add.at(counts, labels_arr, 1)

    have_sample = counts > 0
    fallback = np.array(get_palette(palette), dtype=np.float64)
    class_means = np.where(
        have_sample[:, None], sums / np.maximum(counts, 1)[:, None], fallback
    )

    within_ss = np.zeros(3, dtype=np.float64)
    for c in range(palette):
        if counts[c] > 0:
            within_ss += sqsums[c] - counts[c] * class_means[c] ** 2

    n_train_patches = int(counts.sum())
    n_classes_seen = int(have_sample.sum())
    dof = max(n_train_patches - n_classes_seen, 1)
    pooled_var = np.maximum(within_ss / dof, _MIN_POOLED_VAR)

    return GaussianOracle(
        palette=palette,
        class_means=class_means,
        pooled_var=pooled_var,
        class_counts=counts,
        n_missing_classes=int((~have_sample).sum()),
        n_train_patches=n_train_patches,
    )


def predict_gaussian_oracle(oracle: GaussianOracle, features: np.ndarray) -> List[int]:
    """Classify `features` (an `(n, 3)` array of whole-patch-mean RGB) against `oracle`: minimize
    the shared-diagonal-covariance Mahalanobis distance `sum_channel((x - class_means[c])**2 /
    pooled_var)` over classes `c` -- the Bayes-optimal decision rule for the fitted
    Gaussian-per-class/shared-diagonal-covariance/UNIFORM-prior model (see `GaussianOracle`'s
    docstring; the uniform-prior simplification means the `-0.5*sum(log(pooled_var))` normalizing
    term, being identical for every class under a SHARED covariance, cancels out of the argmin and
    is correctly omitted here)."""
    features = np.asarray(features, dtype=np.float64)
    diffs = features[:, None, :] - oracle.class_means[None, :, :]
    mdist = (diffs ** 2 / oracle.pooled_var[None, None, :]).sum(axis=2)
    return mdist.argmin(axis=1).astype(int).tolist()


@dataclass
class BayesBoundCell:
    """One (palette, corruption) cell's measured result. `center_pixel_nn_error`,
    `whole_patch_nn_error`, and `gaussian_oracle_error` are all measured on the SAME held-out
    test-image set (`n_test_images` of the `n_images` encoded for this cell -- see
    `bayes_bound_cell`'s docstring for the train/test split) so the three numbers are directly,
    apples-to-apples comparable.

    `best_error`/`best_classifier` report whichever of ALL THREE measured error rates is lowest --
    `decode_pixels`' own center-pixel NN strategy INCLUDED, not just the two whole-patch-feature
    classifiers this module adds. This is deliberate, not an oversight: see the module docstring's
    "MEASURED, HONEST REVERSAL" section -- under corruption that includes a translational shift
    (e.g. `combined`'s `crop_pad`), whole-patch mean features can be dramatically WORSE than a
    single center-pixel sample, so a verdict built only from the two whole-patch classifiers could
    report "information destroyed" for a cell where the reference decoder running in this repo
    TODAY already beats the RS budget -- an indefensible result. `verdict` is derived from
    `best_error` against `RS_BUDGET`: information is judged "present" if ANY of the three measured
    classifiers -- naive or oracle -- beats the budget, since that is the actual question this
    instrument exists to answer (is the information gone, full stop, or just poorly read by
    whichever specific method happens to be in front of you).
    """

    palette: int
    corruption: str
    n_images: int
    n_train_images: int
    n_test_images: int
    payload_size: int
    patch_size: int
    center_pixel_nn_error: float
    whole_patch_nn_error: float
    gaussian_oracle_error: float
    rs_budget: float
    best_error: float
    best_classifier: str
    verdict: str
    note: str


def _symbol_error(truth: Sequence[int], observed: Sequence[int]) -> Tuple[int, int]:
    """(errors, total) over the shared prefix of `truth`/`observed` -- mirrors
    `heliogram.harness._run_cell`'s / every other instrument's identical length-mismatch guard."""
    n = min(len(truth), len(observed))
    errors = sum(1 for i in range(n) if truth[i] != observed[i])
    return errors, n


def bayes_bound_cell(
    palette: int,
    corruption_name: str,
    n_images: int = DEFAULT_N_IMAGES,
    payload_size: int = DEFAULT_PAYLOAD_SIZE,
    patch_size: int = PATCH_SIZE,
    nsym: int = NSYM,
    seed: int = 0,
) -> BayesBoundCell:
    """Measure one (palette, corruption) cell (see module docstring for the full method).

    `n_images` (>= 2, required -- see below) images are `heliogram.codec.encode()`d at
    `subpatch=1` with distinct deterministic payloads (`heliogram.dataset.random_payload(seed + i,
    payload_size)`) and corrupted via `heliogram.harness.CORRUPTIONS[corruption_name]` (reused,
    not redefined -- raises `ValueError` for a name not present in that dict). Ground-truth
    symbols for every image come from `extract_symbols` on the CLEAN image, exact by construction.

    TRAIN/TEST SPLIT: the first `n_train = max(1, n_images // 2)` images train the Gaussian oracle
    (`fit_gaussian_oracle`); the remaining `n_images - n_train` (`n_test`, always >= 1 since
    `n_images >= 2` is required) images are HELD OUT -- `center_pixel_nn_error`,
    `whole_patch_nn_error`, and `gaussian_oracle_error` are all measured on this SAME held-out set
    (see `BayesBoundCell`'s docstring for why: apples-to-apples comparability across the three
    numbers, even though `calibration_nn_whole_patch`/`extract_symbols` do not themselves need a
    training split -- they calibrate from each test image's own row 0, same as `decode_pixels`).

    Raises `ValueError` if `n_images < 2` (need at least one training image and one held-out test
    image) or if `corruption_name` is not a key in `heliogram.harness.CORRUPTIONS`.
    """
    if n_images < 2:
        raise ValueError(
            f"n_images must be >= 2 (need >=1 training image and >=1 held-out test image), "
            f"got {n_images!r}"
        )
    if corruption_name not in CORRUPTIONS:
        raise ValueError(
            f"corruption_name must be a key of heliogram.harness.CORRUPTIONS "
            f"({sorted(CORRUPTIONS)}), got {corruption_name!r}"
        )
    corruption_fn = CORRUPTIONS[corruption_name]

    corrupted_imgs: List[Image.Image] = []
    truths: List[List[int]] = []
    for i in range(n_images):
        payload = random_payload(seed + i, payload_size)
        clean = encode(
            payload, palette=palette, patch_size=patch_size, nsym=nsym, seed=0, subpatch=1
        )
        _, _, truth = extract_symbols(clean, palette=palette, patch_size=patch_size, subpatch=1)
        corrupted = corruption_fn(clean)
        if corrupted.size != clean.size:  # corruptions are documented size-stable, guard anyway
            corrupted = corrupted.resize(clean.size)
        corrupted_imgs.append(corrupted)
        truths.append(truth)

    n_train = max(1, n_images // 2)
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, n_images))

    train_features: List[Tuple[np.ndarray, Sequence[int]]] = []
    for i in train_idx:
        _, _, means = whole_patch_means(corrupted_imgs[i], patch_size)
        data_means = means[1:, :, :].reshape(-1, 3)
        train_features.append((data_means, truths[i]))
    oracle = fit_gaussian_oracle(train_features, palette)

    center_errors = center_total = 0
    whole_errors = whole_total = 0
    gauss_errors = gauss_total = 0
    for i in test_idx:
        truth = truths[i]
        corrupted = corrupted_imgs[i]

        _, _, observed_center = extract_symbols(
            corrupted, palette=palette, patch_size=patch_size, subpatch=1
        )
        e, n = _symbol_error(truth, observed_center)
        center_errors += e
        center_total += n

        _, _, observed_whole = calibration_nn_whole_patch(corrupted, palette, patch_size)
        e, n = _symbol_error(truth, observed_whole)
        whole_errors += e
        whole_total += n

        _, _, means = whole_patch_means(corrupted, patch_size)
        data_means = means[1:, :, :].reshape(-1, 3)
        predicted = predict_gaussian_oracle(oracle, data_means)
        e, n = _symbol_error(truth, predicted)
        gauss_errors += e
        gauss_total += n

    center_error_rate = (center_errors / center_total) if center_total else 0.0
    whole_error_rate = (whole_errors / whole_total) if whole_total else 0.0
    gauss_error_rate = (gauss_errors / gauss_total) if gauss_total else 0.0

    # Best over ALL THREE measured classifiers, decode_pixels' own center-pixel NN included -- see
    # BayesBoundCell's docstring / the module docstring's "MEASURED, HONEST REVERSAL" section for
    # why excluding center-pixel here would be indefensible (whole-patch features can be far worse
    # than a center-pixel sample under translational corruption, e.g. `combined`'s crop_pad).
    candidates: Dict[str, float] = {
        "center_pixel_nn": center_error_rate,
        "whole_patch_nn": whole_error_rate,
        "gaussian_oracle": gauss_error_rate,
    }
    best_classifier = min(candidates, key=lambda name: candidates[name])
    best_error = candidates[best_classifier]

    if best_error < RS_BUDGET:
        verdict = "information present -- a better reader could work"
    else:
        verdict = "information likely destroyed at this operating point"

    note = (
        f"palette={palette} corruption={corruption_name!r} n_images={n_images} "
        f"(train={n_train}, test={len(test_idx)}) payload_size={payload_size}B: "
        f"center-pixel NN (decode_pixels' own strategy)={center_error_rate:.4f}, "
        f"whole-patch calibration-NN={whole_error_rate:.4f}, "
        f"approx-Bayes (Gaussian) oracle={gauss_error_rate:.4f}, RS_BUDGET={RS_BUDGET:.4f}. "
        f"Best of all three: {best_classifier} ({best_error:.4f}) -> {verdict}. "
        "See module docstring's MANDATORY CAVEAT (a real VLM sees full spatial context, not just "
        "a 3-number-per-patch whole-patch mean or a single sampled pixel -- an above-budget "
        "result here is strong but not absolute proof of impossibility) and its MEASURED, HONEST "
        "REVERSAL section (whole-patch mean features can be dramatically WORSE than a "
        "center-pixel sample under corruption that includes a translational shift, which is why "
        "best_error/verdict are computed over all three classifiers, not just the two whole-patch "
        "ones)."
    )

    return BayesBoundCell(
        palette=palette,
        corruption=corruption_name,
        n_images=n_images,
        n_train_images=n_train,
        n_test_images=len(test_idx),
        payload_size=payload_size,
        patch_size=patch_size,
        center_pixel_nn_error=center_error_rate,
        whole_patch_nn_error=whole_error_rate,
        gaussian_oracle_error=gauss_error_rate,
        rs_budget=RS_BUDGET,
        best_error=best_error,
        best_classifier=best_classifier,
        verdict=verdict,
        note=note,
    )


def run(
    palettes: Sequence[int] = DEFAULT_PALETTES,
    corruption_names: Sequence[str] = DEFAULT_CORRUPTION_NAMES,
    n_images: int = DEFAULT_N_IMAGES,
    payload_size: int = DEFAULT_PAYLOAD_SIZE,
    patch_size: int = PATCH_SIZE,
    nsym: int = NSYM,
    seed: int = 0,
) -> List[BayesBoundCell]:
    """Run `bayes_bound_cell` over every (palette, corruption_name) pair, in order (palette outer,
    corruption inner) -- `len(palettes) * len(corruption_names)` cells total. See
    `bayes_bound_cell`'s docstring for what each cell measures and for the ValueError cases
    (`n_images < 2`, an unknown `corruption_name`)."""
    cells: List[BayesBoundCell] = []
    for palette in palettes:
        for name in corruption_names:
            cells.append(
                bayes_bound_cell(
                    palette,
                    name,
                    n_images=n_images,
                    payload_size=payload_size,
                    patch_size=patch_size,
                    nsym=nsym,
                    seed=seed,
                )
            )
    return cells


def format_table(cells: Sequence[BayesBoundCell]) -> str:
    """Markdown table: one row per cell, columns matching the module docstring's "Report per
    (palette, corruption)" requirement -- decode_pixels' own center-pixel NN error shown beside
    both whole-patch classifiers and the RS budget line, plus the verdict derived from the best
    (lowest-error) of all three classifiers, center-pixel NN included (see `BayesBoundCell`'s
    docstring and the module docstring's "MEASURED, HONEST REVERSAL" section for why)."""
    lines = [
        f"RS_BUDGET = floor({NSYM}/2)/{RS_NSIZE} = {RS_BUDGET:.4f} "
        f"({RS_BUDGET * 100:.2f}%) -- see module docstring for the byte-vs-symbol unit caveat.",
        "",
        "| palette | corruption | n (train/test) | center-pixel NN err (decode_pixels) | "
        "whole-patch cal-NN err | approx-Bayes (Gaussian) err | RS budget | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c.palette} | {c.corruption} | {c.n_train_images}/{c.n_test_images} | "
            f"{c.center_pixel_nn_error:.4f} | {c.whole_patch_nn_error:.4f} | "
            f"{c.gaussian_oracle_error:.4f} | {c.rs_budget:.4f} | {c.verdict} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--palettes",
        type=int,
        nargs="+",
        default=list(DEFAULT_PALETTES),
        choices=list(VALID_PALETTES),
        help=f"palette sizes to sweep (default: {list(DEFAULT_PALETTES)})",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        nargs="+",
        default=list(DEFAULT_CORRUPTION_NAMES),
        help=f"heliogram.harness.CORRUPTIONS keys to sweep (default: "
        f"{list(DEFAULT_CORRUPTION_NAMES)}); any of {sorted(CORRUPTIONS)}",
    )
    parser.add_argument(
        "--n-images", type=int, default=DEFAULT_N_IMAGES,
        help=f"images per cell, >= 2 (default: {DEFAULT_N_IMAGES})",
    )
    parser.add_argument(
        "--payload-size", type=int, default=DEFAULT_PAYLOAD_SIZE,
        help=f"synthetic payload size in bytes (default: {DEFAULT_PAYLOAD_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--out", type=str, default=None, help="also write the markdown table to this file"
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    bad = [name for name in args.corruptions if name not in CORRUPTIONS]
    if bad:
        print(
            f"error: --corruptions has unknown name(s) {bad}, must be a subset of "
            f"{sorted(CORRUPTIONS)}",
            file=sys.stderr,
        )
        return 2

    cells = run(
        palettes=args.palettes,
        corruption_names=args.corruptions,
        n_images=args.n_images,
        payload_size=args.payload_size,
        seed=args.seed,
    )
    table = format_table(cells)
    print(table)
    print()
    print(
        "MANDATORY CAVEAT: these classifiers see only a 3-number whole-patch-mean RGB feature "
        "per patch, not a real VLM vision encoder's full spatial context -- an above-budget "
        "result is strong evidence, not an absolute proof of impossibility for every possible "
        "reader. See this module's own docstring."
    )
    if args.out:
        Path(args.out).write_text(table + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
