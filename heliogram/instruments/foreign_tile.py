"""heliogram.instruments.foreign_tile -- pre-ingest guard: flag a heliogram-like payload that is
NOT on a trusted allow-list, before it ever reaches a model.

WHY THIS MODULE EXISTS (handoff M6, I2/B10 -- "detector before capability", guardrail #4): this
instrument must exist BEFORE any learned-alphabet capability work does, not after. If a future
fine-tuned VLM (or anyone else) ever learns to read heliogram-style patch grids, the very first
thing that needs to be true is that a pipeline can tell "a trusted heliogram payload I meant to
encode" apart from "a heliogram-shaped payload that snuck in from somewhere else" -- a malicious
image crafted to look like patch-structured data, or a payload encoded with a config nobody
approved. `guard()` below is the intended pre-ingest call for that: run it on an image BEFORE
handing that image to any decoder or model, and reject/quarantine anything it flags.

DATA HONESTY (read this first): every function in this module measures THE CHANNEL -- pixel
statistics (`patch_structure_score`, plain numpy) and `heliogram.codec.decode_pixels`/
`extract_symbols` (the model-free reference pixel decoder) -- never a VLM. There is no model
anywhere in this file, no torch/transformers import at any scope, and nothing here could be
mistaken for a claim about what a real vision-language model would perceive. "Detector" in this
module's name means "structural/decode-based classifier", not "learned classifier" -- see the
`# ponytail:` note in `is_foreign` for exactly where a real learned classifier would eventually
replace this heuristic, and why that upgrade is not needed yet.

DESIGN (model-free, structural -- no learning needed): a genuine heliogram tile has a distinctive
structural signature that a natural image essentially never has by accident: it tiles evenly
into `patch_size`-px cells, and every cell is a near-SOLID color (very low within-cell pixel
variance) -- calibration row included, since row 0 is exactly as solid-per-patch as the data rows
below it. A natural photo, by contrast, has real texture: adjacent pixels within almost any
`patch_size` x `patch_size` window differ by a lot, so its per-patch variance is large. That gap
is what `patch_structure_score` measures, and it is `is_foreign`'s first, cheap filter: only an
image that clears "looks patch-structured at all" ever proceeds to the (much more expensive and
much more decisive) second check -- does this patch-structured image actually decode under one
of the caller's trusted (`palette`, `patch_size`, `subpatch`) configs? A payload that is
patch-structured but decodes under NONE of them is exactly what this module calls "foreign": a
heliogram-shaped thing nobody approved.

Resolving the handoff's "hard-negatives: natural images + learned-alphabet tiles" phrase, in this
module's own terms: NEGATIVES here means images the guard must NOT flag -- natural images (must
not false-positive, i.e. bounds FPR) and allow-listed trusted tiles (also must not false-positive
-- a guard that blocks its own trusted traffic is useless). POSITIVES means images the guard MUST
catch -- heliogram-like payloads outside the trusted set, including the hard case the handoff
specifically calls out: "learned-alphabet" tiles, which still look exactly as patch-aligned and
solid as a trusted tile (so the cheap first filter alone cannot reject them) but use a
non-standard palette/mapping, so no allow-listed config actually decodes them. The Definition of
Done this module targets is exactly the handoff's: **TPR at a bounded FPR, measured against
learned-alphabet-like hard positives, not just easy ones** -- see `build_eval_set` for how those
hard positives are constructed and `evaluate_detector` for how TPR/FPR are computed.

Everything below is seeded and deterministic (same seed -> identical positives/negatives/scores
every run) -- no unseeded randomness anywhere in this file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..codec import (
    PATCH_SIZE,
    VALID_PALETTES,
    HeliogramDecodeError,
    decode_pixels,
    encode,
    extract_symbols,
    get_palette,
)
from ..dataset import random_payload

__all__ = [
    "DEFAULT_VAR_THRESHOLD",
    "AllowListEntry",
    "ForeignTileVerdict",
    "DetectorEval",
    "patch_structure_score",
    "is_foreign",
    "guard",
    "evaluate_detector",
    "build_eval_set",
]

# Tunable ceiling for patch_structure_score (see that function's docstring for units: mean
# per-patch color variance, i.e. mean over patches of Var(R)+Var(G)+Var(B) pooled across the
# patch's pixels, roughly averaged per channel by numpy's .var()). Empirically (see this
# module's tests and the scratch measurements that picked this number): a clean heliogram tile of
# any palette/subpatch scores EXACTLY 0.0 (every cell is a single solid RGB value, no
# antialiasing); mild corruption (resize +-5%, JPEG q95-q70) pushes that into roughly the
# 100-450 range depending on palette size; uniform-noise or ordinary textured/high-frequency
# synthetic images score in the thousands. 250.0 sits in the gap between "clean or mildly
# corrupted heliogram tile" and "has real texture", with a mild bias toward giving corrupted
# heliogram tiles the benefit of the doubt (proceeding to the decisive decode-attempt stage
# rather than being waved through as "natural" outright). This is a single global scalar, not a
# learned boundary -- see the `# ponytail:` note in `is_foreign` for its known failure modes.
DEFAULT_VAR_THRESHOLD = 250.0


@dataclass
class AllowListEntry:
    """One trusted heliogram encoding config. An allow-list is simply a sequence of these --
    `is_foreign` tries `heliogram.codec.decode_pixels` under every entry in turn. Defaults mirror
    `heliogram.codec.encode`'s own defaults (palette=8, patch_size=heliogram.codec.PATCH_SIZE,
    subpatch=1), so a bare `AllowListEntry()` is "the default encode() config" -- the single most
    common trusted case."""

    palette: int = 8
    patch_size: int = PATCH_SIZE
    subpatch: int = 1


@dataclass
class ForeignTileVerdict:
    """Result of one `is_foreign` call.

    `is_patch_structured`: did the image clear the cheap patch/variance filter at all? False
    means "looks like a natural image" -- `is_foreign` is always False in that case (see module
    docstring). `matched_allowlist_entry`: which `AllowListEntry` successfully decoded the image,
    if any (None if the image wasn't patch-structured, or was patch-structured but matched
    nothing). `patch_grid`: (width, height) in patches at the `patch_size` the structural check
    used (NOT necessarily the matched entry's own patch_size, since that is only known once a
    match is found). `mean_cell_variance`: the raw `patch_structure_score` value, so a caller can
    see how close to the threshold a verdict was, not just the boolean side of it.
    """

    is_foreign: bool
    is_patch_structured: bool
    matched_allowlist_entry: Optional[AllowListEntry]
    patch_grid: Tuple[int, int]
    mean_cell_variance: float
    note: str


def _patch_grid_dims(img: Image.Image, patch_size: int) -> Tuple[int, int]:
    return img.width // patch_size, img.height // patch_size


def patch_structure_score(img: Image.Image, patch_size: int = PATCH_SIZE) -> float:
    """Mean per-patch color variance across the WHOLE patch tiling (row 0's calibration patches
    included -- they are exactly as solid-per-patch as the data rows in a real heliogram tile, so
    folding them in only reinforces the signal, never dilutes it).

    ~0.0 for a genuine heliogram tile: every `patch_size` x `patch_size` cell is one solid RGB
    value (`heliogram.codec.encode`'s entire pixel-painting scheme), so the within-cell variance
    of a clean, uncorrupted tile is exactly zero. LARGE for a natural photo, where adjacent
    pixels within almost any `patch_size`-px window differ meaningfully. See DEFAULT_VAR_THRESHOLD
    above for the empirically-picked cutoff and its known failure modes.

    Crops to the largest exact `patch_size` multiple (mirrors `heliogram.codec.extract_symbols`'
    implicit floor division) -- any partial trailing row/column of pixels is simply excluded,
    never padded or resized. Returns `float("inf")` if the image is smaller than one
    `patch_size` x `patch_size` cell in either dimension (cannot be tiled at all) -- `is_foreign`
    treats that the same as "clearly not patch-structured".
    """
    width, height = _patch_grid_dims(img, patch_size)
    if width < 1 or height < 1:
        return float("inf")
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    cropped = arr[: height * patch_size, : width * patch_size]
    # (height, patch_size, width, patch_size, 3) -> group every pixel by which patch it's in.
    grid = cropped.reshape(height, patch_size, width, patch_size, 3).transpose(0, 2, 1, 3, 4)
    patches = grid.reshape(height * width, patch_size * patch_size, 3)
    return float(patches.var(axis=1).mean())


def is_foreign(
    img: Image.Image,
    allowlist: Sequence[AllowListEntry],
    patch_size: int = PATCH_SIZE,
    var_threshold: float = DEFAULT_VAR_THRESHOLD,
    nsym: int = 32,
) -> ForeignTileVerdict:
    """The core three-step guard logic (see module docstring for the reasoning behind each step):

    (a) `patch_structure_score(img, patch_size)` above `var_threshold` -> NOT patch-structured ->
        `is_foreign=False`. This looks like a natural image; let it through without even trying
        to decode it (decode attempts are the expensive, allow-list-sized part of this check).
    (b) Patch-structured: try `heliogram.codec.decode_pixels` under EVERY `allowlist` entry's
        (palette, patch_size, subpatch), in order, catching only `HeliogramDecodeError` (per
        `decode_pixels`' documented contract -- any other exception means a malformed
        `AllowListEntry`, e.g. an invalid palette, and is a caller bug that should propagate
        loudly, not be swallowed here). The first entry that decodes without raising wins:
        `is_foreign=False`, `matched_allowlist_entry` set to that entry -- a trusted tile.
    (c) Patch-structured but NO allow-list entry decodes it -> `is_foreign=True`: a heliogram-like
        payload structurally present in the image, encoded under a config nobody approved.

    `nsym` is shared across every allow-list entry's decode attempt (mirroring
    `heliogram.codec.decode_pixels`' own default of 32) -- it is not part of `AllowListEntry`
    itself, matching the handoff's specified `AllowListEntry` shape (palette, patch_size,
    subpatch only). Pass the `nsym` your trusted pipeline actually uses if it differs from 32.
    """
    score = patch_structure_score(img, patch_size=patch_size)
    patch_grid = _patch_grid_dims(img, patch_size)

    # ponytail: a single global variance threshold is the known ceiling here. It cannot tell a
    # very-low-texture natural image (a smooth gradient, a flat sky, a solid-color banner) from a
    # heliogram tile any better than "which number is bigger" -- see this module's docstring and
    # tests for the natural-image constructions actually validated against this threshold, which
    # deliberately avoid that adversarial edge case rather than claim robustness to it. It is
    # also not robust to aggressive corruption: a crop/pad shift in particular creates a few
    # very-high-variance border patches that can swing the MEAN score well past the threshold
    # even for an otherwise-solid tile. If false-positive/false-negative rate ever matters more
    # than this handles in a real pipeline: train a classifier (even a small one -- a handful of
    # structural features would likely beat one hand-tuned global scalar) instead of pushing this
    # single cutoff further. See evaluate_detector/build_eval_set for how to measure this
    # heuristic's actual TPR/FPR before trusting it anywhere real.
    if score > var_threshold:
        return ForeignTileVerdict(
            is_foreign=False,
            is_patch_structured=False,
            matched_allowlist_entry=None,
            patch_grid=patch_grid,
            mean_cell_variance=score,
            note=(
                f"mean within-cell variance {score:.1f} is above var_threshold={var_threshold:.1f}"
                " -- looks like a natural image, not patch-structured; not flagged."
            ),
        )

    for entry in allowlist:
        try:
            decode_pixels(
                img,
                palette=entry.palette,
                patch_size=entry.patch_size,
                nsym=nsym,
                subpatch=entry.subpatch,
            )
        except HeliogramDecodeError:
            continue
        return ForeignTileVerdict(
            is_foreign=False,
            is_patch_structured=True,
            matched_allowlist_entry=entry,
            patch_grid=patch_grid,
            mean_cell_variance=score,
            note=(
                f"patch-structured (variance {score:.1f}) and decodes cleanly under allow-listed "
                f"entry palette={entry.palette}/patch_size={entry.patch_size}/"
                f"subpatch={entry.subpatch} -- trusted, not flagged."
            ),
        )

    return ForeignTileVerdict(
        is_foreign=True,
        is_patch_structured=True,
        matched_allowlist_entry=None,
        patch_grid=patch_grid,
        mean_cell_variance=score,
        note=(
            f"patch-structured (variance {score:.1f}) but does not decode under any of the "
            f"{len(allowlist)} allow-listed config(s) -- a heliogram-like payload outside the "
            "trusted set; FLAGGED as foreign."
        ),
    )


def guard(img: Image.Image, allowlist: Sequence[AllowListEntry], **kw: object) -> bool:
    """Thin wrapper over `is_foreign` -- the intended pre-ingest call. Usage:
    `if guard(img, allowlist): reject_or_quarantine(img)  # before it ever reaches a model`.
    `**kw` forwards to `is_foreign` (`patch_size=`, `var_threshold=`, `nsym=`)."""
    return is_foreign(img, allowlist, **kw).is_foreign


@dataclass
class DetectorEval:
    """Aggregate result of `evaluate_detector`. `tpr` (true positive rate) is the fraction of
    `positives` correctly flagged foreign; `fpr` (false positive rate) is the fraction of
    `negatives` incorrectly flagged foreign. `threshold` records the `var_threshold` actually
    used (whatever `evaluate_detector` was called with, defaulting to DEFAULT_VAR_THRESHOLD)."""

    tpr: float
    fpr: float
    n_positive: int
    n_negative: int
    threshold: float
    note: str


def evaluate_detector(
    positives: Sequence[Image.Image],
    negatives: Sequence[Image.Image],
    allowlist: Sequence[AllowListEntry],
    **kw: object,
) -> DetectorEval:
    """Measure `is_foreign`'s TPR/FPR against a labeled set of images.

    `positives`: images that SHOULD be flagged foreign (heliogram-like payloads outside
    `allowlist`, including learned-alphabet-like hard cases -- see `build_eval_set`). TPR = the
    fraction of these `is_foreign` actually flags.

    `negatives`: images that should NOT be flagged (natural images + allow-listed trusted tiles).
    FPR = the fraction of these `is_foreign` incorrectly flags.

    `**kw` forwards to every `is_foreign` call (`patch_size=`, `var_threshold=`, `nsym=`) --
    `threshold` in the returned DetectorEval reflects whatever `var_threshold` was actually used.
    """
    threshold = float(kw.get("var_threshold", DEFAULT_VAR_THRESHOLD))  # type: ignore[arg-type]
    n_pos = len(positives)
    n_neg = len(negatives)
    true_positives = sum(1 for img in positives if is_foreign(img, allowlist, **kw).is_foreign)
    false_positives = sum(1 for img in negatives if is_foreign(img, allowlist, **kw).is_foreign)
    tpr = (true_positives / n_pos) if n_pos else 0.0
    fpr = (false_positives / n_neg) if n_neg else 0.0
    return DetectorEval(
        tpr=tpr,
        fpr=fpr,
        n_positive=n_pos,
        n_negative=n_neg,
        threshold=threshold,
        note=(
            f"TPR={tpr:.3f} ({true_positives}/{n_pos} positives correctly flagged foreign -- "
            "heliogram tiles outside the allow-list, incl. learned-alphabet-like hard cases), "
            f"FPR={fpr:.3f} ({false_positives}/{n_neg} negatives incorrectly flagged -- natural "
            "images + allow-listed trusted tiles). See build_eval_set's docstring for exactly "
            "what counts as a positive/negative here."
        ),
    )


def _shuffled_alphabet_tile(
    payload: bytes, palette: int, patch_size: int, nsym: int, seed: int
) -> Image.Image:
    """Build a "learned-alphabet-like" hard positive: a real `heliogram.codec.encode()` image
    (so it is genuinely patch-aligned and solid-per-cell, clearing `patch_structure_score`
    exactly like a trusted tile would) whose DATA cells are then repainted through a seeded,
    deterministic permutation of `heliogram.codec.get_palette(palette)` -- row 0's calibration
    row is left untouched. This stands in for "a model learned a non-standard color/symbol
    mapping for the same palette": to a decoder that calibrates purely from row 0 (exactly what
    `heliogram.codec.decode_pixels` does), the data cells now look like they were written with a
    different, unknown mapping, so nearest-neighbor classification is wrong for most cells and
    Reed-Solomon cannot correct that many errors -- `decode_pixels` reliably raises
    `HeliogramDecodeError` for this construction (see this module's tests). `subpatch=1` only,
    same as `heliogram.dataset.DEFAULT_SUBPATCHES` -- the VLM-meaningful regime this whole
    instrument cares about.

    Reuses `encode`/`extract_symbols`/`get_palette` throughout -- no RS/framing/palette logic is
    reimplemented here, only pixels are repainted after the fact.
    """
    clean = encode(payload, palette=palette, patch_size=patch_size, nsym=nsym, seed=0, subpatch=1)
    width, height, symbols = extract_symbols(
        clean, palette=palette, patch_size=patch_size, subpatch=1
    )
    colors = get_palette(palette)
    perm = list(range(palette))
    random.Random(seed).shuffle(perm)  # deterministic, seeded permutation

    arr = np.array(clean.convert("RGB"))  # mutable copy
    idx = 0
    for r in range(1, height):  # data rows only -- row 0 (calibration) is left standard
        y0 = r * patch_size
        for c in range(width):
            x0 = c * patch_size
            arr[y0 : y0 + patch_size, x0 : x0 + patch_size] = colors[perm[symbols[idx]]]
            idx += 1
    return Image.fromarray(arr)


def _natural_image(seed: int, size: int = PATCH_SIZE * 12) -> Image.Image:
    """One deterministic, seeded synthetic "natural-ish" image: rotates between uniform random
    noise and higher-spatial-frequency patterns (a sinusoid, or a fine per-pixel modulo
    gradient), all chosen to have clearly HIGH within-cell variance at `PATCH_SIZE`, unlike a
    very-low-frequency smooth gradient (which can score deceptively close to a heliogram tile --
    see DEFAULT_VAR_THRESHOLD's docstring and the `# ponytail:` note in `is_foreign`; that
    adversarial edge case is this heuristic's known, documented ceiling, not something this eval
    set claims to cover). `size` defaults to 12 patches square, big enough for a stable variance
    estimate."""
    rng = np.random.RandomState(seed)
    kind = seed % 3
    if kind == 0:
        arr = rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    elif kind == 1:
        freq = 6 + (seed % 4)  # spatial frequency high enough to vary a lot within one patch
        yy, xx = np.mgrid[0:size, 0:size]
        base = (128 + 127 * np.sin(2 * np.pi * freq * xx / size)).astype(np.uint8)
        arr = np.stack([base, np.roll(base, 5, axis=0), np.roll(base, 11, axis=1)], axis=-1)
    else:
        base = (np.arange(size) * 9) % 256  # fine per-pixel modulo gradient, high frequency
        g = np.tile(base, (size, 1)).astype(np.int64)
        arr = np.stack([g, g.T, (g + g.T) % 256], axis=-1).astype(np.uint8)
    return Image.fromarray(arr)


def build_eval_set(
    seed: int = 0,
    allowlist: Optional[Sequence[AllowListEntry]] = None,
    n_natural: int = 12,
    include_learned_alphabet: bool = True,
) -> Tuple[List[Image.Image], List[Image.Image], List[AllowListEntry]]:
    """Deterministic (positives, negatives, allowlist) triple for `evaluate_detector`, per this
    module's Definition of Done: **TPR at a bounded FPR, measured against learned-alphabet-like
    hard positives** (see module docstring).

    `allowlist` defaults to two trusted configs -- `AllowListEntry(palette=8)` and
    `AllowListEntry(palette=16)` (both at `patch_size=PATCH_SIZE`, `subpatch=1`) -- and is always
    returned alongside `positives`/`negatives` so a caller can pass this function's own output
    straight into `evaluate_detector`/`is_foreign` without reconstructing it.

    **positives** (heliogram tiles a guard MUST catch -- `is_foreign` should be True for all of
    these):
      - "plain" off-allowlist tiles: real `encode()` output at palette sizes NOT on `allowlist`
        (32 and 64 by default), varied payload sizes/seeds. The easy case: even the palette SIZE
        alone already differs from every allow-listed entry.
      - "learned-alphabet-like" hard positives (only if `include_learned_alphabet`, the default):
        `_shuffled_alphabet_tile` at palette sizes that ARE on `allowlist` (8 and 16) -- still
        patch-aligned and solid-per-cell (clears the cheap structural filter exactly like a
        trusted tile), but internally inconsistent with its own row-0 calibration, so it decodes
        under NO allow-list entry. This is the handoff's difficult case, and the reason
        `is_foreign`'s decode-attempt stage (not just the palette-size/structural checks) exists
        at all.

    **negatives** (images a guard must NOT flag -- `is_foreign` should be False for all of these):
      - `n_natural` synthetic natural-ish images (`_natural_image`: seeded noise/high-frequency
        patterns) -- bounds FPR against ordinary, non-heliogram input.
      - allow-listed trusted tiles: real `encode()` output at each `allowlist` entry's own config
        -- bounds FPR against the guard rejecting its own legitimate traffic.

    Every image here is built from a single `seed` argument (no other source of randomness),
    so fixed `seed` (and the other arguments) always yields byte-identical positives/negatives.
    """
    if allowlist is None:
        allowlist = [AllowListEntry(palette=8), AllowListEntry(palette=16)]
    allowlist = list(allowlist)
    allowed_palettes = {entry.palette for entry in allowlist}
    # Preferred order (32, 64) first, matching the default allowlist's {8, 16} -- falls through
    # to every other VALID_PALETTES size in case a caller's own allowlist already covers 32/64,
    # so this never silently reuses an allow-listed palette as an "off-allowlist" positive (that
    # would defeat the whole point: an allow-listed palette is trusted, not foreign, by
    # definition). Takes up to 2; degrades to fewer (never zero unless `allowlist` somehow covers
    # every VALID_PALETTES size, an intentionally-unhandled degenerate case).
    preferred_order = (32, 64) + tuple(p for p in VALID_PALETTES if p not in (32, 64))
    off_allowlist_palettes = [p for p in preferred_order if p not in allowed_palettes][:2]

    positives: List[Image.Image] = []
    i = 0
    for palette in off_allowlist_palettes:
        for _ in range(2):  # 2 seeds/payloads per off-allowlist palette
            payload = random_payload(seed + i, 32 + 16 * (i % 3))
            positives.append(
                encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
            )
            i += 1
    if include_learned_alphabet:
        for entry in allowlist:
            for _ in range(2):  # 2 seeds/payloads per allow-listed palette, hard case
                payload = random_payload(seed + i, 32 + 16 * (i % 3))
                positives.append(
                    _shuffled_alphabet_tile(
                        payload, entry.palette, entry.patch_size, nsym=32, seed=seed + i
                    )
                )
                i += 1

    negatives: List[Image.Image] = [_natural_image(seed + j) for j in range(n_natural)]
    j = n_natural
    for entry in allowlist:
        for _ in range(2):  # 2 trusted tiles per allow-listed entry
            payload = random_payload(seed + j, 32 + 16 * (j % 3))
            negatives.append(
                encode(
                    payload,
                    palette=entry.palette,
                    patch_size=entry.patch_size,
                    nsym=32,
                    subpatch=entry.subpatch,
                )
            )
            j += 1

    return positives, negatives, allowlist
