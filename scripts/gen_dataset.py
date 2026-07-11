#!/usr/bin/env python3
"""scripts/gen_dataset.py -- CLI over heliogram.dataset: generate a synthetic Phase-2 training
set (PNG images + JSONL manifest) for QLoRA fine-tuning (see scripts/train_qlora.py).

No GPU/torch/transformers needed -- this only calls into heliogram.dataset/heliogram.codec/
heliogram.corruption (pillow/numpy/reedsolo only, the same base dependencies as the rest of
Phase 1). Safe to run in this repo's CPU-only pytest environment; it is what generates the data
scripts/train_qlora.py's curriculum stages consume.

Every image/target pair this writes is ground truth by construction (heliogram.codec.encode()
writes a known symbol grid; heliogram.codec.extract_symbols() reads it back off the clean,
EVEN-PATCH-GRID-PADDED image before any corruption augmentation is applied) -- see
heliogram/dataset.py's module docstring.

PROCESSOR RESIZE HAZARD (D4 of the Phase-2 scaffold review -- see heliogram/dataset.py's
"PROCESSOR RESIZE HAZARD" module-docstring note and `pad_to_even_patch_grid`'s own docstring for
the full argument): every image this script writes has BOTH patch-grid dimensions (width/height
in `--patch-size`-px units) padded to even, so its pixel dimensions are already exact multiples
of `patch_size * 2` -- the alignment Qwen2-VL/Qwen2.5-VL's image processor's `smart_resize` step
requires to be a no-op. This is not optional and has no flag to disable it: an odd patch-count
dimension is silently resampled OFF the heliogram symbol lattice the moment the image reaches the
processor (see `scripts/train_qlora.py`'s `_assert_processor_alignment`, which asserts this
guarantee actually held by the time an image gets there).

DEFAULTS (Slice C retarget -- see heliogram/dataset.py's module docstring "THE BET" paragraph):
`--palettes` now defaults to `DEFAULT_PALETTES` (64, 128, 256 -- where decode_pixels is MEASURED
to clean-decode exactly but FAIL under JPEG q70/q85, see RESULTS.md), and `--corruption-prob`
now defaults to `RECOMMENDED_TRAINING_CORRUPTION_PROB` (0.5) instead of 0.0 -- a bare invocation
with no flags now generates large-palette, corruption-augmented data by default, since that is
what this project's actual Phase-2 bet needs training data for. Pass `--palettes
2,4,8,16,32,64,128,256 --corruption-prob 0.0` to reproduce the old "every palette, clean-only"
default.

Usage:
    python scripts/gen_dataset.py --out data/phase2_train --n 2000 --seed 0
    python scripts/gen_dataset.py --out data/phase2_val   --n 200  --seed 1
    python scripts/gen_dataset.py --out data/phase2_small --n 50 --palettes 2,4 --payload-sizes 16 --corruption-prob 0.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/gen_dataset.py` from anywhere without an editable install:
# `python <path>` sets sys.path[0] to the script's own directory (scripts/), not the repo root,
# so `import heliogram` would otherwise fail unless the package happens to be pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heliogram.codec import PATCH_SIZE, VALID_PALETTES, VALID_SUBPATCHES  # noqa: E402
from heliogram.dataset import (  # noqa: E402
    DEFAULT_PALETTES,
    DEFAULT_PAYLOAD_SIZES,
    DEFAULT_SUBPATCHES,
    RECOMMENDED_TRAINING_CORRUPTION_PROB,
    write_dataset,
)


def _int_list(text: str) -> list:
    return [int(x) for x in text.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="output directory (images/ + manifest.jsonl)"
    )
    parser.add_argument(
        "--n", type=int, default=1000, help="number of examples to generate (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed; fully determines the dataset for fixed --n and ranges (default: 0)",
    )
    parser.add_argument(
        "--palettes",
        type=_int_list,
        default=list(DEFAULT_PALETTES),
        help=f"comma-separated palette sizes to sample from, subset of {VALID_PALETTES} "
        f"(default: {list(DEFAULT_PALETTES)} -- the large-palette-under-corruption bet, see "
        "heliogram/dataset.py's module docstring; pass e.g. --palettes "
        f"{','.join(str(p) for p in VALID_PALETTES)} for the full range)",
    )
    parser.add_argument(
        "--subpatches",
        type=_int_list,
        default=list(DEFAULT_SUBPATCHES),
        help=f"comma-separated subpatch (k) values to sample from, subset of {VALID_SUBPATCHES} "
        "(default: 1 -- the VLM-meaningful regime; subpatch>1 is a pixel-decoder-only "
        "geometric ceiling, see heliogram/codec.py's DATA HONESTY note)",
    )
    parser.add_argument(
        "--payload-sizes",
        type=_int_list,
        default=list(DEFAULT_PAYLOAD_SIZES),
        help=f"comma-separated payload sizes in bytes to sample from (default: "
        f"{list(DEFAULT_PAYLOAD_SIZES)})",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help=f"patch size in px (default: {PATCH_SIZE})",
    )
    parser.add_argument(
        "--nsym", type=int, default=32, help="Reed-Solomon parity bytes (default: 32)"
    )
    parser.add_argument(
        "--corruption-prob",
        type=float,
        default=RECOMMENDED_TRAINING_CORRUPTION_PROB,
        help="probability of applying one randomly chosen corruption (from "
        "heliogram.dataset.DEFAULT_CORRUPTIONS, mirroring heliogram.harness's realistic "
        "envelope) to each example as augmentation; 0.0 disables augmentation entirely "
        f"(default: {RECOMMENDED_TRAINING_CORRUPTION_PROB} -- corruption augmentation ON by "
        "default, since learning to classify a big palette THROUGH corruption is this "
        "project's retargeted bet, see heliogram/dataset.py's module docstring)",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        help="image file extension/format passed to PIL.Image.save (default: png)",
    )
    return parser


def main(argv: list = None) -> int:
    args = build_parser().parse_args(argv)

    bad_palettes = [p for p in args.palettes if p not in VALID_PALETTES]
    if bad_palettes:
        print(
            f"error: --palettes contains invalid value(s) {bad_palettes}, must be a subset of "
            f"{VALID_PALETTES}",
            file=sys.stderr,
        )
        return 2
    bad_subpatches = [s for s in args.subpatches if s not in VALID_SUBPATCHES]
    if bad_subpatches:
        print(
            f"error: --subpatches contains invalid value(s) {bad_subpatches}, must be a subset "
            f"of {VALID_SUBPATCHES}",
            file=sys.stderr,
        )
        return 2

    manifest_path = write_dataset(
        args.out,
        args.n,
        palettes=args.palettes,
        subpatches=args.subpatches,
        payload_sizes=args.payload_sizes,
        patch_size=args.patch_size,
        nsym=args.nsym,
        seed=args.seed,
        corruption_prob=args.corruption_prob,
        image_format=args.image_format,
    )
    print(f"wrote {args.n} examples to {args.out} (manifest: {manifest_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
