"""heliogram.instruments.learned_alphabet -- CPU, model-free palette-color search (handoff M4,
A12's CPU-runnable flavor).

WHY THIS MODULE EXISTS: handoff M4/A12 asks for "a learned alphabet" -- optimizing symbol
colors instead of trusting `heliogram.codec.get_palette`'s hand-picked HSV scheme. There are two
completely different things that phrase could mean, and this repo has no GPU, so only one of
them can actually run here:

  1. Optimize the PALETTE COLORS against the model-free reference pixel classifier
     (`heliogram.codec.decode_pixels`/`extract_symbols`) under the existing corruption suite --
     a plain, seeded, CPU-only search. This is what THIS module does, end to end, and it is REAL
     and runnable in this repo's CPU-only environment.
  2. Optimize actual TILE PIXELS against a real, frozen ViT/VLM vision encoder's own embedding
     space via backpropagated pixel gradients -- a categorically bigger, GPU-bound question.
     That plug point is `heliogram.encoder.FrozenEncoderHandle`, which cannot run here (no GPU)
     and raises rather than fabricating a result. It is a SEPARATE module for exactly that
     reason: nothing in this file touches it, imports it, or depends on it.

DATA HONESTY (read this loudly, before trusting any number this module prints): every function
below measures THE CHANNEL -- `heliogram.codec.decode_pixels`'s underlying classifier
(`extract_symbols`, nearest-neighbor RGB distance) applied to images built and corrupted with
`heliogram.codec.encode`/`heliogram.corruption`'s own machinery -- never a VLM, never a learned
embedding of any kind. There is no model, no torch/transformers import at any scope, anywhere in
this file. "Learned" in this module's name means "found by a plain seeded numerical search over
RGB triples," not "learned by a neural network." Mirroring `heliogram/codec.py`'s own
subpatch>1 caveat ("a geometric upper bound on the channel, not a demonstrated model
capability"): every improvement this module measures is a GEOMETRIC/COLORIMETRIC ceiling on
*this specific nearest-neighbor pixel classifier*, not a capability claim about any real vision-
language model. A real ViT patch embedding may respond to entirely different color statistics
than Euclidean RGB distance (e.g. hue vs. luma weighting baked into its own pretraining data,
or invariances/sensitivities nearest-neighbor RGB distance does not share) -- colors that help
`decode_pixels` here could plausibly do nothing, or even hurt, a real VLM's own patch
classification. That open question is exactly what `heliogram.encoder.FrozenEncoderHandle`
exists for, and it is explicitly out of reach in this CPU-only slice.

THE KEY TECHNICAL TRICK (why this module never needs to edit `heliogram/codec.py`, and never
needs to reimplement `extract_symbols`'s nearest-neighbor classifier from scratch): a candidate
color list can be measured by (a) building a REAL `heliogram.codec.encode()` image (so grid
geometry, calibration-row layout, and symbol assignment all come straight from the real codec,
never re-derived), (b) repainting every patch's pixels with the candidate colors instead of
`get_palette(palette)`'s colors (`_paint_candidate_tile` below -- the same "build a real encode()
image, then repaint pixels" technique `heliogram.instruments.foreign_tile._shuffled_alphabet_tile`
already uses elsewhere in this package), and then (c) calling `heliogram.codec.extract_symbols`
(or `decode_pixels`) DIRECTLY and unmodified on the result, with the SAME `palette=` it was built
with. Step (c) works correctly for ANY row-0 colors, not just `get_palette(palette)`'s, because
`extract_symbols` recovers its per-index calibration color as the MEAN of that index's own row-0
patch pixels (`heliogram/codec.py`'s `recovered = sums / counts`) rather than trusting
`get_palette(palette)` -- `get_palette(palette)` only appears there as a fallback for a palette
index with zero row-0 samples, and `heliogram.codec.compute_grid` guarantees `width >= palette`
for every grid it builds, so every index 0..palette-1 appears at least once in row 0 regardless
of which actual colors are painted there. That fallback path is therefore mathematically
unreachable for any image this module builds -- confirmed directly in
tests/test_learned_alphabet.py, not just asserted here. This is what lets `symbol_error_rate`
below reuse `extract_symbols` (and `_paint_candidate_tile` reuse `encode`/`get_palette`) exactly
as they already exist, with zero RS/framing/palette/corruption logic duplicated anywhere in this
file.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..codec import PATCH_SIZE, bits_per_symbol, encode, extract_symbols, get_palette
from ..corruption import crop_pad, jpeg_compress, resize_roundtrip
from ..dataset import random_payload

__all__ = [
    "DEFAULT_SEARCH_CORRUPTIONS",
    "DEFAULT_ITERS",
    "LearnedPalette",
    "symbol_error_rate",
    "optimize_palette",
    "compare_to_handcrafted",
    "format_comparison",
    "build_parser",
    "main",
]

# A representative SUBSET of the realistic corruption envelope used everywhere else in this repo
# (heliogram.corruption's own primitives, at the SAME severities heliogram.dataset.
# DEFAULT_CORRUPTIONS/heliogram.harness.CORRUPTIONS use: resize scale=0.95, JPEG q85/q70, crop/pad
# 2px) -- not the full 8-entry suite, to keep each candidate-color evaluation cheap enough for a
# search loop that calls it hundreds of times (see DEFAULT_ITERS below). "clean" (a no-op) is
# included deliberately, not just for realism: it is what keeps the search from ever drifting two
# palette colors close enough together to collide (a collision would spike even the CLEAN error,
# since two symbols would then classify identically with no corruption involved at all, so any
# search step that caused one is rejected by the same acceptance rule that rejects a worse
# corrupted-error step -- see optimize_palette's docstring). jpeg_q70 (and, at larger payloads,
# jpeg_q85) are specifically the corruptions RESULTS.md/heliogram.codec document as where the
# reference pixel decoder actually starts failing at large palettes -- included here so the
# search has real pressure to respond to, not an arbitrarily easy suite.
DEFAULT_SEARCH_CORRUPTIONS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": lambda img: img,
    "resize_5pct": lambda img: resize_roundtrip(img, scale=0.95),
    "jpeg_q85": lambda img: jpeg_compress(img, quality=85),
    "jpeg_q70": lambda img: jpeg_compress(img, quality=70),
    "crop_pad_2px": lambda img: crop_pad(img, dx=2, dy=2),
}

# ponytail: this is THE tunable ceiling on search thoroughness vs. wall-clock cost -- each
# iteration evaluates up to 4 trial palettes (two step sizes x two directions, see
# optimize_palette), and each trial evaluation runs _SEARCH_N_PAYLOADS payloads x
# len(corruptions) encode/corrupt/classify cycles. 60 iterations x 4 trials x 2 payloads x 5
# corruptions = up to 2400 small (patch_size=14, palette_size<=64, <=~100B payload) encode/
# classify cycles, empirically well under a second in this repo's CPU environment (see
# tests/test_learned_alphabet.py) even at the larger palette sizes VALID_PALETTES offers. Raise
# this for a more thorough (slower) search; the search is a plain greedy coordinate descent (see
# optimize_palette), so more iterations can only ever match or improve the result, never worsen
# it.
DEFAULT_ITERS = 60

_SEARCH_PAYLOAD_SIZE_BYTES = 64  # modest synthetic payload -- keeps each evaluated tile small
_SEARCH_N_PAYLOADS = 2  # payloads averaged per evaluation -- a little noise reduction, still cheap
_STEPS = (48, -48, 16, -16)  # coarse-then-fine single-channel deltas tried every iteration
_NSYM = 32  # Reed-Solomon parity, matching every other module's default (codec/dataset/harness)


@dataclass
class LearnedPalette:
    """One `optimize_palette`/`compare_to_handcrafted` result.

    `colors`: the learned palette (length `palette_size`), i.e. what a search starting from
    `heliogram.codec.get_palette(palette_size)` converged to. `symbol_error`: its measured mean
    symbol-error-rate (see `symbol_error_rate`) under `corruptions` (the corruption NAMES used,
    in the order they were evaluated). `baseline_symbol_error`: `get_palette(palette_size)`'s own
    measured error under the IDENTICAL corruptions/payloads/seed -- so the two numbers are
    directly comparable, not measured under different conditions. `improved`: whether the search
    actually found a strictly lower-error palette than the handcrafted baseline (see
    `optimize_palette`'s docstring for why this can legitimately be False -- not every palette
    size/corruption mix has room to improve on an already-good baseline). `note`: a plain-English
    summary of the measured relationship, written either way (DATA HONESTY: no cherry-picking --
    see `optimize_palette`).
    """

    palette_size: int
    colors: List[Tuple[int, int, int]]
    seed: int
    corruptions: List[str]
    symbol_error: float
    baseline_symbol_error: float
    improved: bool
    note: str


def _search_payloads(seed: int) -> List[bytes]:
    """Deterministic, fixed-size set of synthetic payloads for measuring `symbol_error_rate`
    under a given search `seed`. The SAME payload set is reused for every candidate-color
    evaluation (and the baseline evaluation) within one `optimize_palette` call, so candidate-vs-
    candidate and candidate-vs-baseline comparisons are apples-to-apples -- letting the payload
    set drift between evaluations would make the hill-climb's accept/reject decisions
    (`optimize_palette`) noise, not signal. Reuses `heliogram.dataset.random_payload` (the same
    deterministic payload construction the rest of this repo's harness/dataset code uses) rather
    than inventing a second RNG scheme."""
    return [
        random_payload(seed * 1009 + i, _SEARCH_PAYLOAD_SIZE_BYTES)
        for i in range(_SEARCH_N_PAYLOADS)
    ]


def _paint_candidate_tile(
    clean_img: Image.Image,
    colors: Sequence[Tuple[int, int, int]],
    width: int,
    height: int,
    patch_size: int,
    symbols: Sequence[int],
) -> Image.Image:
    """Repaint a real `heliogram.codec.encode()` image's patches with `colors` instead of
    whatever `heliogram.codec.get_palette` colors it was actually painted with -- the same
    "build a real encode() image, then repaint pixels" technique
    `heliogram.instruments.foreign_tile._shuffled_alphabet_tile` already uses, reimplemented
    self-contained here (not imported -- that helper is private to its own module, and paints a
    PERMUTATION of `get_palette`'s colors, not an arbitrary candidate list) since `codec.py`
    cannot be edited in this slice and does not expose a "paint with a custom color list" entry
    point.

    Row 0 (calibration) gets `colors[i % palette_size]` per patch `i`, exactly mirroring
    `encode`'s own row-0 loop; data patches get `colors[symbols[idx]]` in the same row-major
    order `encode`/`extract_symbols` use. `width`/`height`/`symbols` are meant to come from
    `extract_symbols(clean_img, ...)` on the SAME clean image -- ground truth read straight off
    the codec, never re-derived. `subpatch=1` only (one symbol per data patch) -- the
    VLM-meaningful regime this whole module targets, matching `heliogram.dataset.
    DEFAULT_SUBPATCHES`; see the module docstring's subpatch>1 caveat cross-reference.
    """
    palette_size = len(colors)
    arr = np.array(clean_img.convert("RGB"))  # mutable copy
    for i in range(width):
        arr[0:patch_size, i * patch_size : (i + 1) * patch_size] = colors[i % palette_size]
    idx = 0
    for r in range(1, height):
        y0 = r * patch_size
        for c in range(width):
            x0 = c * patch_size
            arr[y0 : y0 + patch_size, x0 : x0 + patch_size] = colors[symbols[idx]]
            idx += 1
    return Image.fromarray(arr)


def symbol_error_rate(
    colors: Sequence[Tuple[int, int, int]],
    palette_size: int,
    corruptions: Optional[Dict[str, Callable[[Image.Image], Image.Image]]] = None,
    seed: int = 0,
    patch_size: int = PATCH_SIZE,
) -> float:
    """THE objective this module searches over: mean fraction of DATA symbols
    `heliogram.codec.extract_symbols` classifies WRONG for a candidate `colors` palette, averaged
    over a deterministic payload set (see `_search_payloads`) and every corruption in
    `corruptions` (default `DEFAULT_SEARCH_CORRUPTIONS`).

    For each payload: builds a real `heliogram.codec.encode()` image at `palette_size` (ground
    truth read via `extract_symbols` off that CLEAN image -- exact by construction, same
    convention as `heliogram.dataset`/`heliogram.harness`), repaints it with `colors`
    (`_paint_candidate_tile`), then for each corruption: applies it, calls `extract_symbols`
    AGAIN (unmodified, same `palette_size`) on the corrupted-and-candidate-colored image, and
    counts symbol mismatches against ground truth. See the module docstring's "KEY TECHNICAL
    TRICK" section for why calling the real `extract_symbols` here (rather than a hand-rolled
    reclassifier) is correct for an arbitrary `colors` list, not just `get_palette`'s own.

    This measures `heliogram.codec.decode_pixels`'s underlying classifier ONLY -- see the module
    docstring's DATA HONESTY section; it is not, and cannot be, a statement about any VLM.

    Raises ValueError if `len(colors) != palette_size` or `palette_size` is not a valid palette
    size, or if `corruptions` is given but empty.
    """
    bits_per_symbol(palette_size)  # validates palette_size is in VALID_PALETTES
    if len(colors) != palette_size:
        raise ValueError(
            f"colors must have exactly palette_size={palette_size} entries, got {len(colors)}"
        )
    if corruptions is None:
        corruptions = DEFAULT_SEARCH_CORRUPTIONS
    if not corruptions:
        raise ValueError("corruptions must be a non-empty mapping of name -> corruption fn")

    total_errors = 0
    total_symbols = 0
    for payload in _search_payloads(seed):
        clean = encode(
            payload, palette=palette_size, patch_size=patch_size, nsym=_NSYM, seed=0, subpatch=1
        )
        width, height, truth = extract_symbols(
            clean, palette=palette_size, patch_size=patch_size, subpatch=1
        )
        candidate_clean = _paint_candidate_tile(clean, colors, width, height, patch_size, truth)

        for corrupt_fn in corruptions.values():
            corrupted = corrupt_fn(candidate_clean)
            if corrupted.size != candidate_clean.size:  # corruptions are documented size-stable,
                corrupted = corrupted.resize(candidate_clean.size)  # but guard anyway
            _, _, observed = extract_symbols(
                corrupted, palette=palette_size, patch_size=patch_size, subpatch=1
            )
            n = min(len(truth), len(observed))
            total_errors += sum(1 for i in range(n) if truth[i] != observed[i])
            total_errors += abs(len(truth) - len(observed))
            total_symbols += len(truth)

    return (total_errors / total_symbols) if total_symbols else 0.0


def _clip255(value: int) -> int:
    return max(0, min(255, value))


def optimize_palette(
    palette_size: int = 16,
    corruptions: Optional[Dict[str, Callable[[Image.Image], Image.Image]]] = None,
    iters: int = DEFAULT_ITERS,
    seed: int = 0,
    patch_size: int = PATCH_SIZE,
) -> LearnedPalette:
    """Seeded coordinate-descent search over RGB triples, starting from
    `heliogram.codec.get_palette(palette_size)`, minimizing `symbol_error_rate` (equivalently:
    maximizing post-corruption separability) under `corruptions` (default
    `DEFAULT_SEARCH_CORRUPTIONS`).

    ALGORITHM: each of `iters` iterations picks one (palette index, RGB channel) coordinate at
    random (via `random.Random(seed)`, so the whole sequence of coordinates visited is
    deterministic) and tries perturbing it by each of `_STEPS` (coarse +-48 and fine +-16, both
    directions); if any trial's `symbol_error_rate` is STRICTLY lower than the current best, the
    best-improving trial is accepted and becomes the new starting point for the next iteration,
    else nothing changes that iteration. This is plain greedy hill-climbing/coordinate descent
    (as named in the handoff), not a claim of global optimality -- with a modest, documented
    iteration budget (see `DEFAULT_ITERS`'s `# ponytail:` note) it is a cheap local-search
    improvement pass over the handcrafted starting point, not an exhaustive search.

    DETERMINISM: the accept/reject sequence depends only on `seed` (which drives both the
    coordinate/step choices AND, via `_search_payloads`, which synthetic payloads are measured)
    plus the deterministic `encode`/`extract_symbols`/corruption functions -- calling this twice
    with identical arguments always returns byte-identical `colors` (see
    tests/test_learned_alphabet.py).

    GUARANTEE BY CONSTRUCTION: because the search only ever accepts a STRICTLY improving move
    (never a worsening or even a lateral one) and starts from the handcrafted baseline itself,
    the returned `symbol_error` can never exceed `baseline_symbol_error` for the SAME
    `corruptions`/`seed` -- `improved` (`symbol_error < baseline_symbol_error`) can legitimately
    still be False, though, whenever no single-coordinate perturbation the search tried actually
    lowered the mean error within its iteration budget (e.g. the baseline is already
    error-free under `corruptions`, which happens for small palettes under a mild-enough
    corruption mix -- see `LearnedPalette.note`, written honestly either way, and
    tests/test_learned_alphabet.py's own no-cherry-picking assertion). This guarantee is about
    the AGGREGATE objective actually searched over (the mean across every corruption in
    `corruptions`) -- it says nothing about any ONE corruption in isolation, since a move can
    trade a little error on one corruption for a bigger reduction on another; call
    `symbol_error_rate` directly with a single-entry `corruptions` dict to check any one
    corruption on its own.
    """
    bits_per_symbol(palette_size)  # validates palette_size is in VALID_PALETTES
    if corruptions is None:
        corruptions = DEFAULT_SEARCH_CORRUPTIONS
    corruption_names = list(corruptions)

    rng = random.Random(seed)
    baseline_colors = get_palette(palette_size)
    baseline_error = symbol_error_rate(
        baseline_colors, palette_size, corruptions, seed=seed, patch_size=patch_size
    )

    current = [list(c) for c in baseline_colors]
    current_error = baseline_error

    for _ in range(iters):
        color_idx = rng.randrange(palette_size)
        channel_idx = rng.randrange(3)
        best_trial: Optional[List[List[int]]] = None
        best_trial_error = current_error
        for delta in _STEPS:
            trial = [list(c) for c in current]
            trial[color_idx][channel_idx] = _clip255(trial[color_idx][channel_idx] + delta)
            trial_colors = [tuple(c) for c in trial]
            trial_error = symbol_error_rate(
                trial_colors, palette_size, corruptions, seed=seed, patch_size=patch_size
            )
            if trial_error < best_trial_error:
                best_trial_error = trial_error
                best_trial = trial
        if best_trial is not None:
            current = best_trial
            current_error = best_trial_error

    learned_colors = [tuple(c) for c in current]
    improved = current_error < baseline_error - 1e-12

    if improved:
        note = (
            f"learned palette REDUCES mean symbol-error-rate vs. the handcrafted "
            f"get_palette({palette_size}) baseline under corruptions={corruption_names} "
            f"(seed={seed}, {iters} coordinate-descent iterations): "
            f"{baseline_error:.4f} -> {current_error:.4f}. This measures "
            "heliogram.codec.decode_pixels/extract_symbols (the CPU pixel decoder) ONLY -- "
            "see this module's DATA HONESTY section; it is not a VLM result."
        )
    else:
        note = (
            f"learned palette did NOT improve over the handcrafted get_palette({palette_size}) "
            f"baseline under corruptions={corruption_names} (seed={seed}, {iters} "
            f"coordinate-descent iterations): both measured at {baseline_error:.4f} mean "
            "symbol-error-rate (no single-coordinate perturbation tried in this search lowered "
            "it within the iteration budget -- see DEFAULT_ITERS/`iters` to search harder). "
            "This measures heliogram.codec.decode_pixels/extract_symbols (the CPU pixel "
            "decoder) ONLY -- see this module's DATA HONESTY section; it is not a VLM result."
        )

    return LearnedPalette(
        palette_size=palette_size,
        colors=learned_colors,
        seed=seed,
        corruptions=corruption_names,
        symbol_error=current_error,
        baseline_symbol_error=baseline_error,
        improved=improved,
        note=note,
    )


def compare_to_handcrafted(
    palette_size: int = 16,
    corruptions: Optional[Dict[str, Callable[[Image.Image], Image.Image]]] = None,
    iters: int = DEFAULT_ITERS,
    seed: int = 0,
    patch_size: int = PATCH_SIZE,
) -> LearnedPalette:
    """The M4 Definition of Done: a symbol-error/bits-per-patch-flavored number for the learned
    code, reported beside the handcrafted `get_palette(palette_size)` baseline, both measured
    identically (same `corruptions`, payload set, and `seed`) so they are directly comparable.
    A thin, documented alias for `optimize_palette` (which already computes the baseline as part
    of its own search) -- see that function's docstring for the algorithm, determinism, and the
    "guarantee by construction" note on what `improved=False` can legitimately mean. See
    `format_comparison` for a pretty-printed bits/patch-style summary of the result."""
    return optimize_palette(
        palette_size=palette_size,
        corruptions=corruptions,
        iters=iters,
        seed=seed,
        patch_size=patch_size,
    )


def format_comparison(result: LearnedPalette) -> str:
    """Plain-text pretty-printer for a `LearnedPalette`, used by both the CLI and anyone
    debugging interactively. Reports an APPROXIMATE bits/patch figure
    (`log2(palette_size) * (1 - symbol_error_rate)`) alongside the raw symbol-error numbers --
    this is a simple proxy (symbol accuracy scaled by the raw per-symbol bit width), NOT
    `heliogram.harness`'s own `bits_per_patch` metric (which additionally accounts for the
    calibration-row/Reed-Solomon overhead and requires a full RS-verified decode_success_rate,
    neither of which this module measures) -- do not conflate the two numbers."""
    import math

    bpp_baseline = math.log2(result.palette_size) * (1 - result.baseline_symbol_error)
    bpp_learned = math.log2(result.palette_size) * (1 - result.symbol_error)
    lines = [
        f"palette_size={result.palette_size}  seed={result.seed}  "
        f"corruptions={result.corruptions}",
        f"  handcrafted get_palette({result.palette_size}):  "
        f"symbol_error={result.baseline_symbol_error:.4f}  "
        f"(~{bpp_baseline:.2f} bits/patch proxy)",
        f"  learned palette:                  "
        f"symbol_error={result.symbol_error:.4f}  (~{bpp_learned:.2f} bits/patch proxy)",
        f"  improved over handcrafted baseline: {'yes' if result.improved else 'no'}",
        f"  note: {result.note}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--palette-sizes",
        type=int,
        nargs="+",
        default=[8, 16],
        help="palette size(s) to search + compare against get_palette (default: 8 16); larger "
        "sizes cost more per search iteration (bigger patch grids) -- see DEFAULT_ITERS",
    )
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_ITERS, help=f"search iterations (default: {DEFAULT_ITERS})"
    )
    parser.add_argument("--seed", type=int, default=0, help="search seed (default: 0)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    for palette_size in args.palette_sizes:
        result = compare_to_handcrafted(palette_size, iters=args.iters, seed=args.seed)
        print(format_comparison(result))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
