"""heliogram.instruments.fingerprint -- encoder/decoder identity probe: does a heliogram
channel's per-corruption symbol-error SIGNATURE still match the one we trust?

WHY THIS MODULE EXISTS (handoff M6, A10 -- "encoder id / silent-swap probe", a byproduct of the
capacity sweep): heliogram.harness's sweep already measures symbol_error_rate per (palette,
subpatch, payload_size, corruption) cell for the ONE reference encode/decode pipeline
(encode + decode_pixels). That per-corruption error VECTOR is not just a throughput number -- it
is also a characteristic SIGNATURE of the specific encode/decode pipeline that produced it: a
different color palette mapping, or a different pixel classifier, generally produces a
measurably different vector, even holding (palette, subpatch, payload_size) fixed.
`fingerprint()` below packages that observation into a comparison primitive: compute a reference
signature once (from a trusted encoder/decoder pair), then compare it against a freshly observed
signature from -- nominally -- "the same" pipeline. If the pipeline was silently swapped for a
different one (a different encoder implementation, a different palette mapping, a tampered
decoder), the observed signature diverges from the reference by far more than ordinary
run-to-run noise, and `detect_swap()` flags it. This is the CPU-simulatable blind-swap test the
handoff asks for.

DATA HONESTY (read this first, same rule as every other module in this package): every function
here measures THE CHANNEL -- `heliogram.codec.encode`/`extract_symbols` (or, when the seam below
is used to substitute something else, whatever CPU-only encode/decode callables are handed to
it) -- never a VLM. There is no model, no torch/transformers import, anywhere in this file.
"Encoder id" here means "which model-free encode/decode pipeline produced this image", not "which
VLM read it": the REAL, per-VLM fingerprint (does a fine-tuned model's own characteristic error
pattern shift when ITS weights are silently swapped for a different checkpoint?) is Phase 2 work,
gated on GPU access, and is not attempted anywhere in this repo -- see heliogram/vlm.py. This
module simulates the *mechanics* of that blind-swap test on the one thing measurable without a
GPU: the codec's own model-free encode/decode pipeline.

DESIGN: `fingerprint()` is built around a SEAM -- `encode_fn` (default `heliogram.codec.encode`)
and `decode_symbols_fn` (default `heliogram.codec.extract_symbols`, the same classifier
`decode_pixels` uses internally) are both plain, swappable callables, not hardcoded calls. Ground
truth for every trial, however, is ALWAYS read off the CANONICAL `encode()`/`extract_symbols()`
pair, independent of whichever `encode_fn`/`decode_symbols_fn` are under test -- this is what
makes a swap show up at all. If either seam is quietly replaced with something that no longer
agrees with the canonical codec's own symbol mapping, the observed-vs-canonical-truth error rate
jumps -- even on a "clean" (uncorrupted) image -- in a way ordinary corruption-driven noise never
does when the real pipeline is matched to itself. `swapped_palette_encode()` builds exactly that
"deliberately swapped encoder" for a blind test: a real `encode()` image (so it is geometrically
identical -- same patch grid, same calibration row) whose DATA cells alone are repainted through
a seeded, deterministic permutation of the palette -- a stand-in for "this image came from an
encoder using an unknown, non-standard color/symbol mapping" (the same technique
`heliogram.instruments.foreign_tile._shuffled_alphabet_tile` uses for its own, differently-scoped
guard test).

Everything below is seeded and deterministic (same seed -> identical signatures every run) -- no
unseeded randomness anywhere in this file.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..codec import (
    PATCH_SIZE,
    VALID_PALETTES,
    VALID_SUBPATCHES,
    bits_per_symbol,
    encode,
    extract_symbols,
    get_palette,
)
from ..corruption import compose, crop_pad, jpeg_compress, resize_roundtrip
from ..dataset import random_payload

__all__ = [
    "DEFAULT_FINGERPRINT_CORRUPTIONS",
    "DEFAULT_SWAP_THRESHOLD",
    "EncodeFn",
    "DecodeSymbolsFn",
    "Fingerprint",
    "fingerprint",
    "distance",
    "detect_swap",
    "swapped_palette_encode",
    "format_fingerprint",
    "build_parser",
    "main",
]

EncodeFn = Callable[..., Image.Image]
DecodeSymbolsFn = Callable[..., Tuple[int, int, List[int]]]
CorruptionFn = Callable[[Image.Image], Image.Image]

# A SUBSET of heliogram.corruption's realistic-serving-pipeline envelope (see
# heliogram.harness.CORRUPTIONS / heliogram.dataset.DEFAULT_CORRUPTIONS for the full 8-name suite
# this deliberately narrows) -- five representative severities (no corruption, a mild resize, two
# JPEG qualities, and the composed worst case) are enough to characterize a per-corruption
# signature without paying for the full sweep on every fingerprint() call. Deliberately named and
# defined separately from heliogram.dataset.DEFAULT_CORRUPTIONS / heliogram.harness.CORRUPTIONS
# (rather than importing either) -- this is a narrower, fingerprint-specific default built
# directly from heliogram.corruption's primitives, not a copy of the full envelope.
DEFAULT_FINGERPRINT_CORRUPTIONS: Dict[str, CorruptionFn] = {
    "clean": lambda img: img,
    "resize_5pct": lambda img: resize_roundtrip(img, scale=0.95),
    "jpeg_q85": lambda img: jpeg_compress(img, quality=85),
    "jpeg_q70": lambda img: jpeg_compress(img, quality=70),
    "combined": lambda img: compose(
        img,
        [
            (resize_roundtrip, {"scale": 0.95}),
            (jpeg_compress, {"quality": 70}),
            (crop_pad, {"dx": 2, "dy": 2}),
        ],
    ),
}

# See distance()/detect_swap()'s docstrings for the full reasoning; short version: two
# fingerprint() calls with IDENTICAL (encode_fn, decode_symbols_fn, palette, subpatch,
# payload_size, corruptions, seed) are byte-for-byte deterministic (see module docstring), so
# their distance is EXACTLY 0.0 -- there is no run-to-run noise to budget for at all. A genuinely
# swapped encoder (see swapped_palette_encode) instead disagrees with the canonical ground truth
# on most DATA cells at EVERY corruption severity, including "clean", pushing every signature
# entry towards (palette-1)/palette (>=0.98 for this module's own default palette=64) rather than
# just one entry. DEFAULT_SWAP_THRESHOLD sits far below that and far above the exact-0.0 matched
# case, separating the two DoD cases with a wide margin without being tuned to either one
# specifically. MEASURED (see tests/test_fingerprint.py, which pins the actual numbers): at this
# module's own defaults, distance(reference, identical) == 0.0 and distance(reference,
# swapped-palette) is roughly an order of magnitude above this threshold.
DEFAULT_SWAP_THRESHOLD = 0.2


def _check_subpatch(patch_size: int, subpatch: int) -> None:
    if subpatch not in VALID_SUBPATCHES:
        raise ValueError(f"subpatch must be one of {VALID_SUBPATCHES}, got {subpatch!r}")
    if patch_size % subpatch != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be evenly divisible by subpatch ({subpatch})"
        )


@dataclass
class Fingerprint:
    """One encode/decode pipeline's characteristic per-corruption symbol-error SIGNATURE.

    `signature` maps corruption name -> symbol_error_rate (fraction of DATA sub-cells whose
    `decode_symbols_fn`-classified value differs from the CANONICAL codec's own ground truth for
    the same payload -- see `fingerprint()`'s docstring for exactly what "canonical" means and
    why). `corruptions` is the ordered list of corruption names actually measured (the keys of
    `signature`, kept separately so a caller can inspect coverage without sorting a dict).
    """

    palette: int
    subpatch: int
    payload_size: int
    corruptions: List[str]
    signature: Dict[str, float]
    note: str


def fingerprint(
    encode_fn: EncodeFn = encode,
    decode_symbols_fn: DecodeSymbolsFn = extract_symbols,
    palette: int = 64,
    corruptions: Optional[Dict[str, CorruptionFn]] = None,
    payload_size: int = 512,
    subpatch: int = 1,
    patch_size: int = PATCH_SIZE,
    nsym: int = 32,
    trials: int = 5,
    seed: int = 0,
) -> Fingerprint:
    """Measure `encode_fn`/`decode_symbols_fn`'s characteristic per-corruption symbol-error
    signature at a fixed (palette, subpatch, payload_size).

    This is the seam (handoff M6, A10 -- see module docstring): `encode_fn` (default
    `heliogram.codec.encode`) and `decode_symbols_fn` (default `heliogram.codec.extract_symbols`,
    the same classifier `decode_pixels` uses internally) are both swappable, independently. Ground
    truth for every trial, however, is ALWAYS read via the CANONICAL `encode()`/`extract_symbols()`
    pair -- never via `encode_fn`/`decode_symbols_fn` themselves -- so a signature only comes out
    "clean" (near-zero error on mild/no corruption) when whatever `encode_fn`/`decode_symbols_fn`
    measure actually agrees with the real codec's own symbol mapping. Default args therefore
    reproduce `decode_pixels`' own signature exactly (the same per-corruption `symbol_error_rate`
    `heliogram.harness._run_cell` computes): with both seams at their defaults, the "observed"
    pipeline IS the canonical one, so they agree everywhere corruption doesn't genuinely destroy
    information.

    For each of `trials` seeded random payloads (`heliogram.dataset.random_payload(seed + i,
    payload_size)`): build the canonical reference image (`encode(...)`) and read its ground-truth
    symbols (`extract_symbols(...)`, exact by construction on a clean image -- see
    `heliogram.codec.extract_symbols`'s docstring); separately build the image under test
    (`encode_fn(...)`, same payload/palette/subpatch/patch_size/nsym); then, for every corruption
    in `corruptions` (default `DEFAULT_FINGERPRINT_CORRUPTIONS`, a subset of
    `heliogram.corruption`'s realistic envelope), apply it to the image under test and classify
    the result with `decode_symbols_fn`. `signature[name]` is the resulting mismatch rate against
    canonical ground truth, pooled over every trial and every DATA sub-cell (mismatched-length
    symbol lists are handled the same way `heliogram.harness._run_cell` does: compared only over
    their shared prefix, `min(len(truth), len(observed))`).

    Raises ValueError for an invalid `palette`/`subpatch`/`patch_size` combination (via
    `heliogram.codec.bits_per_symbol` / this module's own subpatch check), for `trials < 1`, or
    for an empty `corruptions` mapping.
    """
    bits_per_symbol(palette)  # validates palette is in VALID_PALETTES
    _check_subpatch(patch_size, subpatch)
    if trials < 1:
        raise ValueError(f"trials must be >= 1, got {trials!r}")
    if corruptions is None:
        corruptions = DEFAULT_FINGERPRINT_CORRUPTIONS
    if not corruptions:
        raise ValueError("corruptions must be a non-empty mapping of name -> corruption fn")

    names = list(corruptions)
    error_counts: Dict[str, int] = {name: 0 for name in names}
    total_counts: Dict[str, int] = {name: 0 for name in names}

    for trial in range(trials):
        payload = random_payload(seed + trial, payload_size)

        # Ground truth is ALWAYS the canonical codec, regardless of encode_fn/decode_symbols_fn
        # under test -- see module/function docstring for why this is what makes a swap visible.
        ref_img = encode(
            payload, palette=palette, patch_size=patch_size, nsym=nsym, seed=0, subpatch=subpatch
        )
        _, _, truth = extract_symbols(
            ref_img, palette=palette, patch_size=patch_size, subpatch=subpatch
        )

        test_img = encode_fn(
            payload, palette=palette, patch_size=patch_size, nsym=nsym, seed=0, subpatch=subpatch
        )

        for name in names:
            corrupted = corruptions[name](test_img)
            if corrupted.size != test_img.size:  # corruption fns are documented to preserve
                corrupted = corrupted.resize(test_img.size)  # size; guard anyway (mirrors
                # heliogram.harness._run_cell's identical guard)
            _, _, observed = decode_symbols_fn(
                corrupted, palette=palette, patch_size=patch_size, subpatch=subpatch
            )
            n = min(len(truth), len(observed))
            error_counts[name] += sum(1 for i in range(n) if truth[i] != observed[i])
            total_counts[name] += n

    signature = {
        name: (error_counts[name] / total_counts[name]) if total_counts[name] else 0.0
        for name in names
    }

    return Fingerprint(
        palette=palette,
        subpatch=subpatch,
        payload_size=payload_size,
        corruptions=names,
        signature=signature,
        note=(
            "CHANNEL/DECODER fingerprint only -- no VLM or model of any kind was involved (see "
            f"module docstring). encode_fn={getattr(encode_fn, '__name__', repr(encode_fn))}, "
            f"decode_symbols_fn={getattr(decode_symbols_fn, '__name__', repr(decode_symbols_fn))}, "
            f"{trials} trial(s)/corruption, seed={seed}."
        ),
    )


def distance(a: Fingerprint, b: Fingerprint, metric: str = "l2") -> float:
    """L1 (sum of absolute differences) or L2 (Euclidean norm, the default) distance between two
    fingerprints' signatures, computed over the corruption keys the two SHARE -- so comparing
    fingerprints built from slightly different `corruptions` mappings still works, over whatever
    overlap they have. Raises ValueError if `a`/`b` share no corruption key at all, or if `metric`
    is not "l1"/"l2"."""
    shared = sorted(set(a.signature) & set(b.signature))
    if not shared:
        raise ValueError(
            f"fingerprints share no common corruption key to compare: a has "
            f"{sorted(a.signature)}, b has {sorted(b.signature)}"
        )
    diffs = [a.signature[name] - b.signature[name] for name in shared]
    if metric == "l1":
        return sum(abs(d) for d in diffs)
    if metric == "l2":
        return math.sqrt(sum(d * d for d in diffs))
    raise ValueError(f"metric must be 'l1' or 'l2', got {metric!r}")


def detect_swap(
    reference: Fingerprint, observed: Fingerprint, threshold: float = DEFAULT_SWAP_THRESHOLD
) -> bool:
    """True when `observed`'s signature has drifted from `reference`'s by more than `threshold`
    (L2 distance by default, see `distance()`) -- i.e. the encoder/decoder pipeline that produced
    `observed` most likely differs from the one that produced `reference`. See
    DEFAULT_SWAP_THRESHOLD's comment for how that default was picked.

    This is a detector, not a proof: it flags a signature that no longer matches the trusted one,
    the same way `heliogram.instruments.foreign_tile.guard` flags a payload that doesn't decode
    under any trusted allow-list entry -- both are structural, model-free checks, not a claim
    about identifying WHICH different encoder/decoder is now in use, only THAT the channel's
    fingerprint no longer matches what was trusted.
    """
    return distance(reference, observed) > threshold


def swapped_palette_encode(
    data: bytes,
    palette: int = 8,
    patch_size: int = PATCH_SIZE,
    nsym: int = 32,
    seed: int = 0,
    subpatch: int = 1,
    shuffle_seed: int = 1337,
) -> Image.Image:
    """Deliberately-swapped encoder for the blind-swap test (see module docstring): builds a real
    `heliogram.codec.encode()` image -- so it is geometrically identical to a trusted tile, same
    patch grid, same calibration row -- then repaints its DATA cells only through a seeded,
    deterministic permutation of `heliogram.codec.get_palette(palette)`. Row 0 (calibration) is
    left untouched, so a canonical decoder still recovers the "right" RGB value for each color
    index; it is exactly the DATA cells' color<->symbol mapping that no longer agrees with the
    canonical one -- a stand-in for "this image was produced by an encoder using an unknown,
    non-standard color/symbol assignment" (the same construction
    `heliogram.instruments.foreign_tile._shuffled_alphabet_tile` uses for its own guard test,
    reimplemented here as a plain `encode_fn`-shaped function -- same call signature as
    `heliogram.codec.encode` -- rather than that module's internal helper, and generalized to
    every `subpatch` in `VALID_SUBPATCHES` rather than just 1).

    `shuffle_seed` (independent of `seed`, which -- like `heliogram.codec.encode`'s own `seed`
    parameter -- is accepted only for call-signature compatibility with the `encode_fn` seam and
    has no effect on the output) controls the permutation: the same `(palette, shuffle_seed)`
    always produces the same swapped mapping, and a different `shuffle_seed` produces a different
    (still deterministic) one. Reuses `encode`/`extract_symbols`/`get_palette` throughout -- no
    RS/framing/palette logic is reimplemented, only pixels are repainted after the fact.
    """
    del seed  # reserved for encode_fn call-signature compatibility only, see docstring
    _check_subpatch(patch_size, subpatch)
    clean = encode(
        data, palette=palette, patch_size=patch_size, nsym=nsym, seed=0, subpatch=subpatch
    )
    width, height, symbols = extract_symbols(
        clean, palette=palette, patch_size=patch_size, subpatch=subpatch
    )
    colors = get_palette(palette)
    perm = list(range(palette))
    random.Random(shuffle_seed).shuffle(perm)  # deterministic, seeded permutation

    arr = np.array(clean.convert("RGB"))  # mutable copy
    k = subpatch
    sub = patch_size // k
    idx = 0
    for r in range(1, height):  # data rows only -- row 0 (calibration) is left standard
        y0 = r * patch_size
        for c in range(width):
            x0 = c * patch_size
            for sr in range(k):  # sub-cell rows, row-major within the patch (matches encode())
                for sc in range(k):  # sub-cell cols
                    yy0 = y0 + sr * sub
                    xx0 = x0 + sc * sub
                    arr[yy0 : yy0 + sub, xx0 : xx0 + sub] = colors[perm[symbols[idx]]]
                    idx += 1
    return Image.fromarray(arr)


def format_fingerprint(fp: Fingerprint) -> str:
    """Plain-text pretty-printer for a Fingerprint, used by both the CLI and anyone debugging
    interactively."""
    lines = [
        f"palette:      {fp.palette}",
        f"subpatch:     {fp.subpatch}",
        f"payload_size: {fp.payload_size}B",
        "signature (symbol_error_rate per corruption):",
    ]
    for name in fp.corruptions:
        lines.append(f"  {name:<16} {fp.signature[name]:.4f}")
    lines.append(f"note: {fp.note}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--palette",
        type=int,
        default=64,
        choices=VALID_PALETTES,
        help="palette size to fingerprint (default: 64)",
    )
    parser.add_argument(
        "--subpatch",
        type=int,
        default=1,
        choices=VALID_SUBPATCHES,
        help="sub-patch density k (default: 1)",
    )
    parser.add_argument(
        "--payload-size",
        type=int,
        default=512,
        help="synthetic payload size in bytes (default: 512)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="random payloads averaged per corruption (default: 5)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SWAP_THRESHOLD,
        help=f"detect_swap L2 distance threshold (default: {DEFAULT_SWAP_THRESHOLD})",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Demonstrates the blind-swap DoD end to end: a reference fingerprint vs. a fresh
    identical-config fingerprint (must NOT flag a swap), and vs. a deliberately swapped-palette
    encoder's fingerprint (MUST flag a swap). Returns 0 if both checks behave as expected, 1
    otherwise -- a self-check, not just a demo (same philosophy as heliogram.harness's own
    self-consistency section)."""
    args = build_parser().parse_args(argv)
    common = dict(
        palette=args.palette,
        subpatch=args.subpatch,
        payload_size=args.payload_size,
        trials=args.trials,
        seed=args.seed,
    )

    reference = fingerprint(**common)
    print("=== Reference fingerprint (decode_pixels' own channel signature) ===")
    print(format_fingerprint(reference))

    identical = fingerprint(**common)
    d_same = distance(reference, identical)
    flagged_same = detect_swap(reference, identical, threshold=args.threshold)
    print("\n=== Identical-config fingerprint (fresh run, same args) ===")
    print(format_fingerprint(identical))
    print(f"distance(reference, identical) = {d_same:.6f}  ->  detect_swap = {flagged_same}")

    swapped = fingerprint(encode_fn=swapped_palette_encode, **common)
    d_swap = distance(reference, swapped)
    flagged_swap = detect_swap(reference, swapped, threshold=args.threshold)
    print("\n=== Swapped-encoder fingerprint (shuffled palette mapping) ===")
    print(format_fingerprint(swapped))
    print(f"distance(reference, swapped) = {d_swap:.6f}  ->  detect_swap = {flagged_swap}")

    print(
        "\nDATA HONESTY: every number above comes from extract_symbols, the model-free pixel "
        "classifier decode_pixels itself uses -- a channel/decoder fingerprint, not a VLM "
        "measurement. See the module docstring for the Phase-2, per-VLM fingerprint this stands "
        "in for."
    )

    ok = (not flagged_same) and flagged_swap
    if not ok:
        print(
            "\nUNEXPECTED: identical-config should not be flagged and the swapped encoder "
            "should be -- see above.",
            file=sys.stderr,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
