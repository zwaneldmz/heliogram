"""heliogram.instruments.saliency -- per-grid-position recoverability under corruption (handoff
M6, A11: "recoverable bits by patch position" -- a near-free byproduct of the M3 capacity sweep).

WHY THIS MODULE EXISTS: heliogram.harness's capacity sweep (see that module and RESULTS.md)
reports ONE symbol_error_rate per (palette, subpatch, payload_size, corruption) cell -- a single
scalar averaged over the WHOLE patch grid. That number can hide a lot: maybe every patch position
is equally fragile, or maybe (as the intuitive "shifted content should hurt the edges more" guess
suggests) some grid POSITIONS are far more fragile than others. This module answers that directly,
at near-zero extra cost over what the sweep already computes -- the same encode/corrupt/
extract_symbols calls heliogram.harness._run_cell makes, just kept PER POSITION instead of reduced
to one scalar.

DATA HONESTY (read this first -- it is the whole point of this repo's rule): every measurement in
this module comes from `heliogram.codec.encode`/`extract_symbols`, the model-free REFERENCE PIXEL
DECODER (pixel-center sampling + nearest-neighbor color classification, no learning, no model of
any kind involved anywhere). This module measures how well THE CHANNEL survives corruption at each
grid position, NOT what a VLM would perceive there. There is no torch/transformers import anywhere
in this file, at module scope or otherwise -- nothing here could produce a number that looks like
a model measurement.

WHAT THE MEASURED DATA ACTUALLY SHOWS (an honest, somewhat counter-intuitive finding -- read this
before trusting a "hot spot" on one of these maps as a smooth gradient): translation (crop_pad)
corruption on THIS codec is a CLIFF, not a gradient. Row 0 of every image is the CALIBRATION row
(see heliogram.codec's module docstring); extract_symbols recovers each color's reference RGB from
that row before classifying any data cell. crop_pad shifts the WHOLE canvas -- calibration and data
alike -- by the same (dx, dy), so:
  - dx, dy both <= patch_size // 2 ("half"): every sampled center, calibration or data, still
    lands inside its own original solid-color patch. Zero symbol error, EVERYWHERE. There is no
    smooth "a little worse near the edge" regime below this threshold -- below it, classification
    is entirely unaffected, full stop.
  - dx or dy > half: calibration and data both shift by the same whole number of patches, so the
    recovered reference palette itself becomes a relabeling of the true one (predicted symbol =
    whatever patch is now sampled instead of the true one) -- a GLOBAL effect that lands on every
    position about equally hard, not specifically the positions nearest the shifted-in fill. Column
    0 (which needs a nonexistent "column -1" neighbor and so is forced to read the corruption's
    fill color) turns out to be statistically indistinguishable from an interior column here, once
    averaged over enough random payloads -- both hover close to the (palette-1)/palette
    "random guess" baseline. Which one looks a little better or worse in a short run is sampling
    noise (or, for the first few grid cells, an artifact of heliogram.codec's fixed CODEC_VERSION
    header byte), not a reproducible geometric law -- see tests/test_saliency.py, which measured
    this directly across dozens of (palette, payload_size, seed) combinations before settling on
    the construction below.

A genuine, LARGE, and reproducible position effect DOES show up once `subpatch` (k) > 1, because
`encode`'s calibration row is ALWAYS full patch_size-px patches regardless of `subpatch` (see that
function's docstring) -- only DATA cells are subdivided into a k x k grid of sub-cells. That gives
calibration and data DIFFERENT corruption thresholds: patch_size // 2 ("half") for calibration,
(patch_size // k) // 2 ("sub_half") for data. Choosing dx/dy strictly between sub_half and half
shifts data sub-cells while leaving calibration fully intact and correctly recoverable, so a
misclassified sub-cell is compared against the TRUE palette, not a self-corrupted, degenerate one.
In that window the leftmost sub-cell column has no valid left neighbor at all -- only the
corruption's fill color, classified against the still-correct palette -- and so can never benefit
from the redundancy (natural repeats, or heliogram.codec's own zero-padding of unused grid
capacity, see encode()'s docstring) that lets a shifted-but-real neighbor occasionally happen to
match truth. tests/test_saliency.py pins the exact, measured, reproducible result of this
construction and documents, honestly, what drives its size (partly genuine edge geometry, partly
this specific payload size's own zero-padding) -- see that file for the full accounting.

CLI (`python3 -m heliogram.instruments.saliency`): prints saliency-map summaries for a couple of
(palette, corruption) configs from a small default sweep. Model-free, cheap, safe to run anywhere.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..codec import PATCH_SIZE, encode, extract_symbols
from ..corruption import crop_pad, jpeg_compress
from ..dataset import random_payload

__all__ = [
    "SaliencyMap",
    "position_error_map",
    "summarize",
    "format_saliency",
    "build_parser",
    "main",
]


def _jpeg_q70(img: Image.Image) -> Image.Image:
    """Default corruption probe for position_error_map/main(): JPEG q70 -- this project's own
    "worst realistic corruption" (see heliogram.harness.CORRUPTIONS' `jpeg_q70` entry), and the
    one most likely to leave a nonzero, informative error map instead of the exact all-zero map
    every corruption produces below crop_pad's half-patch threshold (see module docstring)."""
    return jpeg_compress(img, quality=70)


def _corruption_label(fn: Callable[[Image.Image], Image.Image], override: Optional[str]) -> str:
    """Best-effort human-readable label for a corruption callable, used for SaliencyMap.corruption
    (a display string, not the callable itself -- kept plain so a SaliencyMap is trivially
    printable/comparable without holding a live function reference). Prefers an explicit
    `corruption_name=` override; otherwise reads the function's own `__name__` (stripping a
    leading underscore, e.g. this module's own `_jpeg_q70` -> "jpeg_q70"); falls back to
    `repr(fn)` for an anonymous lambda, which at least prints something instead of crashing."""
    if override is not None:
        return override
    name = getattr(fn, "__name__", None)
    if not name or name == "<lambda>":
        return repr(fn)
    return name.lstrip("_")


@dataclass
class SaliencyMap:
    """One (palette, subpatch, corruption, payload_size) config's per-grid-position error map.

    `error_by_position` is a `(height, width)` numpy array of MEAN symbol-error rate over
    `trials` random payloads, one entry per PATCH grid position (matching the `(width, height)`
    `heliogram.codec.extract_symbols` reports -- NOT a sub-cell grid: for `subpatch` k > 1, each
    entry is the mean error over that patch's k*k sub-cells). Row 0 is the CALIBRATION row (see
    heliogram.codec's module docstring) -- it carries no comparable "symbol" of its own (`encode`
    paints it from `get_palette`, not from the payload), so this module EXCLUDES it explicitly:
    `error_by_position[0, :]` is `np.nan`, never a fabricated 0.0 or an average that would
    silently mix in a not-really-a-symbol row. Rows `1 .. height-1` are real per-DATA-position
    error rates.

    `corruption` is a display label (see `_corruption_label`), not the callable itself.
    """

    palette: int
    subpatch: int
    corruption: str
    patch_size: int
    width: int
    height: int
    trials: int
    error_by_position: np.ndarray
    note: str


def position_error_map(
    palette: int = 64,
    corruption: Callable[[Image.Image], Image.Image] = _jpeg_q70,
    corruption_name: Optional[str] = None,
    payload_size: int = 1024,
    subpatch: int = 1,
    patch_size: int = PATCH_SIZE,
    nsym: int = 32,
    trials: int = 5,
    seed: int = 0,
) -> SaliencyMap:
    """For `trials` random payloads (seeded, deterministic), encode() a clean image, apply
    `corruption`, and compare `extract_symbols(clean)` (ground truth -- exact by construction,
    the same trick heliogram.harness/heliogram.dataset use) against `extract_symbols(corrupted)`
    at every DATA position, accumulating a per-position error rate. Mirrors
    heliogram.harness._run_cell's clean-vs-corrupted extract_symbols comparison exactly; the only
    difference is that this function keeps the per-position detail _run_cell reduces to one
    scalar (symbol_error_rate).

    Deterministic: `trial_seed = seed + trial` for `trial` in `range(trials)` drives
    `heliogram.dataset.random_payload` (the same deterministic payload construction the rest of
    this repo uses), and `encode`/`corruption` are themselves deterministic -- fixed arguments
    always produce a byte-identical `error_by_position`.

    `subpatch` k > 1 subdivides each DATA patch into a k x k grid of solid-color sub-cells (see
    heliogram.codec.encode's docstring); each `error_by_position` entry is the MEAN error over
    that patch's k*k sub-cells, so the returned map always has one row/col per PATCH position,
    never per sub-cell, regardless of `subpatch`. See this module's docstring for why `subpatch`
    matters here: it is what lets a shift corrupt data without also corrupting calibration.
    """
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials!r}")

    k = subpatch
    width: Optional[int] = None
    height: Optional[int] = None
    error_sum: Optional[np.ndarray] = None  # (height-1, width) accumulator, data rows only

    for trial in range(trials):
        trial_seed = seed + trial
        payload = random_payload(trial_seed, payload_size)
        clean_img = encode(
            payload, palette=palette, patch_size=patch_size, nsym=nsym, seed=0, subpatch=subpatch
        )
        corrupted_img = corruption(clean_img)
        if corrupted_img.size != clean_img.size:  # corruptions are documented to preserve size,
            corrupted_img = corrupted_img.resize(clean_img.size)  # guard a caller-supplied one anyway

        w, h, truth = extract_symbols(
            clean_img, palette=palette, patch_size=patch_size, subpatch=subpatch
        )
        _, _, observed = extract_symbols(
            corrupted_img, palette=palette, patch_size=patch_size, subpatch=subpatch
        )
        if width is None:
            width, height = w, h
            error_sum = np.zeros((height - 1, width), dtype=np.float64)

        truth_arr = np.asarray(truth, dtype=np.int64).reshape(height - 1, width, k, k)
        obs_arr = np.asarray(observed, dtype=np.int64).reshape(height - 1, width, k, k)
        per_patch_err = (truth_arr != obs_arr).astype(np.float64).mean(axis=(2, 3))
        error_sum += per_patch_err

    assert width is not None and height is not None and error_sum is not None  # trials >= 1

    error_by_position = np.full((height, width), np.nan, dtype=np.float64)
    error_by_position[1:, :] = error_sum / trials

    label = _corruption_label(corruption, corruption_name)
    return SaliencyMap(
        palette=palette,
        subpatch=subpatch,
        corruption=label,
        patch_size=patch_size,
        width=width,
        height=height,
        trials=trials,
        error_by_position=error_by_position,
        note=(
            f"per-position symbol error rate over {trials} trial(s); palette={palette}, "
            f"subpatch={subpatch}, corruption={label!r}, payload_size={payload_size}B. Row 0 is "
            "the calibration row (NaN, excluded -- see SaliencyMap's docstring); rows 1.. are "
            "real per-DATA-position error rates from the model-free reference pixel decoder "
            "(extract_symbols) -- this measures the channel, not a VLM."
        ),
    )


def summarize(m: SaliencyMap) -> Dict[str, object]:
    """Aggregate a SaliencyMap into headline numbers: overall mean error, the single worst
    position (highest error rate) and its rate, an edge-vs-interior comparison (mean error over
    the DATA region's own border -- its first/last data row and first/last column -- vs.
    everywhere else), and, only when `m.corruption` looks like a crop_pad-style corruption
    (a plain, case-insensitive substring match on the name -- this module has no other way to
    know what corruption produced a given map), separate top-row and left-column comparisons
    against the interior. Each of those is an HONEST, data-driven boolean computed from
    `m.error_by_position` itself -- never hardcoded True -- see this module's docstring and
    tests/test_saliency.py for when (and why) these are, and are not, actually elevated.
    """
    data = m.error_by_position[1:, :]  # exclude the NaN calibration row
    if data.size == 0:
        raise ValueError("SaliencyMap has no data rows to summarize (height < 2)")

    mean_error = float(np.nanmean(data))
    flat_idx = int(np.argmax(data))  # data has no NaNs of its own (only row 0 was excluded above)
    worst_row_in_data, worst_col = np.unravel_index(flat_idx, data.shape)
    worst_rate = float(data[worst_row_in_data, worst_col])
    # +1: report the position in error_by_position's own (height, width) coordinates, where row 0
    # is the excluded calibration row -- worst_row_in_data is an index into `data` (rows 1.. of
    # error_by_position), so the caller-facing row is offset by 1.
    worst_position: Tuple[int, int] = (int(worst_row_in_data) + 1, int(worst_col))

    border = np.zeros(data.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    interior_mask = ~border
    edge_mean = float(data[border].mean())
    interior_mean = float(data[interior_mask].mean()) if interior_mask.any() else float("nan")

    result: Dict[str, object] = {
        "mean_error": mean_error,
        "worst_position": worst_position,
        "worst_rate": worst_rate,
        "edge_mean_error": edge_mean,
        "interior_mean_error": interior_mean,
        "edge_minus_interior": edge_mean - interior_mean,
    }

    if "crop_pad" in m.corruption.lower():
        # Axis-appropriate comparisons, not the doubly-restricted 4-sided `interior_mean` above:
        # crop_pad's dx shifts COLUMNS, so "is the left column worse" is a question about columns
        # (compare against a representative middle BAND of columns, across every row -- including
        # whichever row heliogram.codec's own zero-padding of unused capacity happens to land in,
        # since that padding is a real part of this config's data, not something to exclude just
        # because it also happens to be the last row); dy shifts ROWS the same way. Quartering
        # each axis (a plain middle-half band, not a fixed pixel margin) keeps this well-defined
        # across the whole `width`/`height` range VALID_PALETTES implies (as small as 2).
        n_rows, n_cols = data.shape
        col_margin = max(1, n_cols // 4)
        row_margin = max(1, n_rows // 4)
        col_interior = data[:, col_margin:-col_margin] if n_cols > 2 * col_margin else data
        row_interior = data[row_margin:-row_margin, :] if n_rows > 2 * row_margin else data

        left_col_mean = float(data[:, 0].mean())
        top_row_mean = float(data[0, :].mean())
        col_interior_mean = float(col_interior.mean())
        row_interior_mean = float(row_interior.mean())
        result["crop_pad_left_col_mean_error"] = left_col_mean
        result["crop_pad_top_row_mean_error"] = top_row_mean
        result["crop_pad_col_interior_mean_error"] = col_interior_mean
        result["crop_pad_row_interior_mean_error"] = row_interior_mean
        result["crop_pad_left_edge_elevated"] = bool(left_col_mean > col_interior_mean)
        result["crop_pad_top_edge_elevated"] = bool(top_row_mean > row_interior_mean)

    return result


def format_saliency(m: SaliencyMap) -> str:
    """Plain-text pretty-printer: a header line plus summarize()'s headline numbers. Does not
    print the full `error_by_position` grid (that can be `width` columns wide -- up to 256 for
    palette=256 -- which would not fit a terminal usefully); use `m.error_by_position` directly
    for that, or a heatmap plotting tool outside this repo's CPU-only scope."""
    s = summarize(m)
    lines = [
        f"palette={m.palette} subpatch={m.subpatch} corruption={m.corruption!r} "
        f"trials={m.trials} grid={m.width}x{m.height} (w x h; row 0 = calibration, excluded)",
        f"mean error (data positions): {s['mean_error']:.4f}",
        f"worst position (row,col)={s['worst_position']}  rate={s['worst_rate']:.4f}",
        f"edge (data-region border) mean error: {s['edge_mean_error']:.4f}   "
        f"interior mean error: {s['interior_mean_error']:.4f}   "
        f"edge-interior: {s['edge_minus_interior']:+.4f}",
    ]
    if "crop_pad_left_edge_elevated" in s:
        lines.append(
            f"crop_pad check -- left col mean={s['crop_pad_left_col_mean_error']:.4f} "
            f"(elevated vs interior: {s['crop_pad_left_edge_elevated']})   "
            f"top row mean={s['crop_pad_top_row_mean_error']:.4f} "
            f"(elevated vs interior: {s['crop_pad_top_edge_elevated']})"
        )
    lines.append(m.note)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--trials", type=int, default=5, help="random payloads per config (default: 5)"
    )
    parser.add_argument("--seed", type=int, default=0, help="base seed (default: 0)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Runs a small, cheap default sweep and prints a SaliencyMap summary for each config.
    Model-free (encode/extract_symbols only, via position_error_map) -- safe and fast to run
    anywhere, no GPU required. Covers both this module's headline findings: a general-purpose
    palette/corruption probe (jpeg_q70, subpatch=1), and the subpatch=2 crop_pad construction
    that demonstrates a genuine, reproducible position effect (see module docstring)."""
    args = build_parser().parse_args(argv)

    configs: List[Tuple[int, str, Callable[[Image.Image], Image.Image], int, int]] = [
        (64, "jpeg_q70", _jpeg_q70, 1, 1024),
        (64, "crop_pad_4px", lambda img: crop_pad(img, dx=4, dy=2), 2, 1024),
    ]
    for palette, name, fn, subpatch, payload_size in configs:
        m = position_error_map(
            palette=palette,
            corruption=fn,
            corruption_name=name,
            payload_size=payload_size,
            subpatch=subpatch,
            trials=args.trials,
            seed=args.seed,
        )
        print(format_saliency(m))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
