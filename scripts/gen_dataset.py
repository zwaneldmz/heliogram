#!/usr/bin/env python3
"""scripts/gen_dataset.py -- CLI over heliogram.dataset: generate a synthetic Phase-2 training
set (PNG images + JSONL manifest) for QLoRA fine-tuning (see scripts/train_qlora.py).

No GPU/torch/transformers needed -- this only calls into heliogram.dataset/heliogram.codec/
heliogram.corruption (pillow/numpy/reedsolo only, the same base dependencies as the rest of
Phase 1). Safe to run in this repo's CPU-only pytest environment; it is what generates the data
scripts/train_qlora.py's curriculum stages consume.

Every image/target pair this writes is ground truth by construction (heliogram.codec.encode()
writes a known symbol grid; heliogram.codec.extract_symbols() reads it back off the clean image
before any corruption augmentation is applied) -- see heliogram/dataset.py's module docstring.

Usage:
    python scripts/gen_dataset.py --out data/phase2_train --n 2000 --seed 0
    python scripts/gen_dataset.py --out data/phase2_val   --n 200  --seed 1 --corruption-prob 0.5
    python scripts/gen_dataset.py --out data/phase2_small --n 50 --palettes 2,4 --payload-sizes 16
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
    DEFAULT_PAYLOAD_SIZES,
    DEFAULT_SUBPATCHES,
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
        default=list(VALID_PALETTES),
        help=f"comma-separated palette sizes to sample from, subset of {VALID_PALETTES} "
        "(default: all)",
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
        default=0.0,
        help="probability of applying one randomly chosen corruption (from "
        "heliogram.dataset.DEFAULT_CORRUPTIONS, mirroring heliogram.harness's realistic "
        "envelope) to each example as augmentation; 0.0 disables augmentation entirely "
        "(default: 0.0)",
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
