"""heliogram.dataset -- synthetic (image -> target symbol string) training pairs for Phase 2.

DATA HONESTY / scope: this module only *prepares training data*. It never loads, runs, or
requires a model of any kind -- every function here works with pillow/numpy/reedsolo alone (the
same base dependencies as `heliogram.codec`), so `import heliogram.dataset` (and `import
heliogram`, which re-exports it) is always safe in a CPU-only, no-torch environment. Nothing in
this module produces a number that could be mistaken for a VLM result; it produces bytes (PNG
files) and ground-truth strings, full stop.

Ground truth comes from the codec, not from any labeling process: for a synthetic payload,
`heliogram.codec.encode()` writes a known, exact grid of symbols, and
`heliogram.codec.extract_symbols()` reads that same grid straight back off the CLEAN
(pre-corruption) image -- which is exact by construction, the same trick
`heliogram.harness` uses to compute `symbol_error_rate`. So every `target` string this module
emits is *definitionally* correct for its image, with zero hand-labeling and zero ambiguity.

Target string encoding: each symbol (an integer in `[0, palette)`, palette in
`heliogram.codec.VALID_PALETTES`, so always <= 256) is rendered as one character from
`SYMBOL_ALPHABET`, a single fixed 256-character alphabet reused across every palette size. Its
first 64 characters are the standard base64 alphabet (index order), unchanged, so any target
string written before the palette range grew past 64 still decodes identically and
`SYMBOL_ALPHABET[:P]` is still a valid set of "base-P digits" for every P in VALID_PALETTES; the
remaining characters are the next printable, non-whitespace code points. Keeping the vocabulary
identical regardless of which palette a given example used is useful for a model trained across
the whole palette range, and makes round-tripping trivial: `symbols_to_target`/
`target_to_symbols` are exact inverses (see their docstrings). This is one reasonable "compact
string" choice, not a pinned wire format -- unlike `spec/format-v0.1.md`, nothing downstream of
`heliogram.codec` depends on this exact string shape.

Corruption augmentation (optional, `corruption_prob` > 0) draws from a small suite built on
`heliogram.corruption`'s primitives, intentionally mirroring the realistic envelope
`heliogram.harness.CORRUPTIONS` measures against (resize +-3-5%, JPEG q70-95, slight crop/pad,
and their composition) so that training augmentation matches what the Phase-1 harness actually
measured -- not a wider or narrower range invented independently. It is defined locally here
(rather than imported from `heliogram.harness`) because this module is scoped to import from
`heliogram.corruption` directly, per the Phase-2 scaffold's module boundaries; `heliogram/
harness.py` itself is out of scope to edit or depend on in this slice.

THE BET this module's defaults now target (Slice C retarget -- see scripts/train_qlora.py's
curriculum for where this plays out): `heliogram.codec` and RESULTS.md already MEASURE that
`decode_pixels` (the pixel decoder) clean-decodes `palette=128/256` exactly, byte-for-byte, but
FAILS to decode them at every tested payload size once the image goes through JPEG q70 (and, at
larger payloads, even JPEG q85) -- nearest-neighbor RGB classification simply cannot separate
that many colors once chroma subsampling has eroded hue differences (see `codec.py`'s
`get_palette` docstring and `spec/format-v0.1.md` section 2a). That gap -- not sub-patch
geometry -- is this project's actual Phase-2 bet: can a LEARNED reader classify a big color
palette (`P` in `{64, 128, 256}`) correctly through the same realistic corruption where a naive
nearest-neighbor classifier cannot? A patch's dominant color is a coarse, whole-patch visual
feature a ViT patch embedding could plausibly encode even after JPEG/resize blur it slightly --
a categorically different (and more plausible) ask than resolving `subpatch>1` spatial
structure *smaller* than one ViT patch, which stays a documented, secondary, PIXEL-DECODER-ONLY
geometric ceiling (see `codec.py`'s DATA HONESTY note and `spec/format-v0.1.md` section 6a) and
is explicitly NOT what `DEFAULT_PALETTES`/`DEFAULT_SUBPATCHES` below are tuned toward. Realizing
this bet is exactly what would make the README's Bar C (token crossover: fewer total patches
than base64 tokens, from ~3-13KB payloads at `palette=256/128`) usable end to end instead of a
clean-channel-only accounting fact. Nothing here measures whether a VLM actually can do this --
that is `scripts/train_qlora.py` plus a real GPU run, not this module -- but this module's
DEFAULTS (which palettes/subpatches its own generation functions emphasize when a caller
doesn't override them) are picked to produce training data for exactly that question, not
spread thin across every palette in `VALID_PALETTES` uniformly as before this retarget.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from PIL import Image

from .codec import (
    PATCH_SIZE,
    VALID_PALETTES,
    VALID_SUBPATCHES,
    bits_per_symbol,
    encode,
    extract_symbols,
)
from .corruption import compose, crop_pad, jpeg_compress, resize_roundtrip

__all__ = [
    "SYMBOL_ALPHABET",
    "DEFAULT_CORRUPTIONS",
    "DEFAULT_PALETTES",
    "DEFAULT_PAYLOAD_SIZES",
    "DEFAULT_SUBPATCHES",
    "RECOMMENDED_TRAINING_CORRUPTION_PROB",
    "Example",
    "symbols_to_target",
    "target_to_symbols",
    "random_payload",
    "generate_examples",
    "write_dataset",
    "iter_manifest",
]

# 256 distinct, non-whitespace characters -- covers every symbol value (0..255) for every palette
# in VALID_PALETTES (max palette is 256) with exactly one character per symbol, no separator
# needed. The FIRST 64 characters are the standard base64 alphabet, unchanged, so (a) any
# manifest.jsonl target string written when palettes were <= 64 still decodes identically, and (b)
# SYMBOL_ALPHABET[:P] is still a valid set of "base-P digits" for every P in VALID_PALETTES.
# Characters past index 63 are the next printable, non-whitespace Unicode code points (ASCII
# punctuation, then Latin-1, then Latin Extended-A). Two invariants every character must hold:
#   - non-whitespace: heliogram.vlm._extract_symbol_string strips whitespace from a raw VLM
#     response via str.split(), so a whitespace-classified alphabet character would be silently
#     dropped and corrupt parsing. isprintable() is False for every whitespace char, so it
#     guarantees this (the explicit `not ch.isspace()` clause below documents the requirement).
#   - distinct: target_to_symbols builds a char->index lookup dict, which requires uniqueness.
_BASE64_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits + "+/"


def _build_symbol_alphabet(size: int = 256) -> str:
    """Base64 alphabet first (indices 0..63, frozen for backward compatibility), then the next
    printable, non-whitespace code points until `size` distinct characters are collected."""
    chars = list(_BASE64_ALPHABET)
    seen = set(chars)
    cp = 0x21  # first printable ASCII, skipping the C0 control chars and space (0x20)
    while len(chars) < size:
        ch = chr(cp)
        if ch.isprintable() and not ch.isspace() and ch not in seen:
            chars.append(ch)
            seen.add(ch)
        cp += 1
    return "".join(chars)


SYMBOL_ALPHABET: str = _build_symbol_alphabet()
assert len(SYMBOL_ALPHABET) == 256  # sanity: one char per symbol for the largest palette (256)
assert len(set(SYMBOL_ALPHABET)) == 256  # all distinct (target_to_symbols' lookup needs this)
assert SYMBOL_ALPHABET[:64] == _BASE64_ALPHABET  # base64 prefix frozen for back-compat

# Mirrors heliogram.harness.CORRUPTIONS' realistic-serving-pipeline envelope (resize +-3-5%,
# JPEG q70-95, slight crop/pad, and their composition) -- see the module docstring for why this
# is defined here rather than imported from heliogram.harness. Kept numerically identical on
# purpose: training augmentation should match what RESULTS.md actually measures, not drift from
# it. "clean" (a no-op) is always included so corruption_prob < 1.0 has a real chance of no-op.
DEFAULT_CORRUPTIONS: Dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean": lambda img: img,
    "resize_3pct": lambda img: resize_roundtrip(img, scale=0.97),
    "resize_5pct": lambda img: resize_roundtrip(img, scale=0.95),
    "jpeg_q95": lambda img: jpeg_compress(img, quality=95),
    "jpeg_q85": lambda img: jpeg_compress(img, quality=85),
    "jpeg_q70": lambda img: jpeg_compress(img, quality=70),
    "crop_pad_2px": lambda img: crop_pad(img, dx=2, dy=2),
    "combined": lambda img: compose(
        img,
        [
            (resize_roundtrip, {"scale": 0.95}),
            (jpeg_compress, {"quality": 70}),
            (crop_pad, {"dx": 2, "dy": 2}),
        ],
    ),
}

# THE large-palette-under-corruption bet (see module docstring's "THE BET" paragraph): P in
# {64, 128, 256} is exactly where RESULTS.md/heliogram.codec measure decode_pixels to clean-
# decode exactly but FAIL under realistic JPEG q70/q85 -- the gap a learned reader would need to
# close. This is `generate_examples`/`write_dataset`'s default `palettes=`; callers can still
# pass palettes=VALID_PALETTES (or any other subset) for the full range, e.g. for an ablation
# against the smaller palettes decode_pixels already handles fine under the realistic
# corruption envelope (see RESULTS.md's self-consistency section).
DEFAULT_PALETTES: Tuple[int, ...] = (64, 128, 256)
# Spans a cheap warm-up tier (16/48/128B) through the low-KB range where the token-crossover
# benefit (README's Bar C) starts to show up for the largest palettes (RESULTS.md: palette=256
# crosses below base64 token count around ~3055B) -- widened from the original (16, 48, 128) by
# adding 1024B so a default-configuration training run doesn't only ever see payloads far below
# the regime the whole benefit claim is about. Deliberately stops short of the 4096/16384B
# tiers heliogram.harness's own sweep uses (those cost far more patches/pixels per training
# example); scripts/train_qlora.py's curriculum stages do reach into that range explicitly.
DEFAULT_PAYLOAD_SIZES: Tuple[int, ...] = (16, 48, 128, 1024)
DEFAULT_SUBPATCHES: Tuple[int, ...] = (1,)  # the VLM-meaningful regime; see codec.py's DATA
# HONESTY note on subpatch>1 being a pixel-decoder-only geometric ceiling -- callers who
# deliberately want to explore subpatch>1 training data can still pass subpatches=(1, 2). This
# axis is deliberately NOT part of the Slice C retarget (see module docstring's "THE BET"
# paragraph): the bet is learned big-palette color classification under corruption, not
# sub-patch geometry, which stays this project's documented secondary/geometric-ceiling axis.

# NOT `generate_examples`/`write_dataset`'s own `corruption_prob` parameter default (that stays
# 0.0 below, on purpose, for backward-compatible/least-surprise library-function behavior --
# existing callers of either function get clean-only data unless they explicitly ask for
# augmentation). This is what `scripts/gen_dataset.py`'s CLI and `scripts/train_qlora.py`'s
# curriculum stages default to instead, since THOSE are the places actually expressing this
# project's retargeted training bet (see module docstring): corruption augmentation ON by
# default for anyone generating a dataset meant for training, not just calling the library
# function directly with no arguments.
RECOMMENDED_TRAINING_CORRUPTION_PROB: float = 0.5


def symbols_to_target(symbols: Sequence[int], palette: int) -> str:
    """Render a symbol sequence (ints in `[0, palette)`) as a compact string, one character per
    symbol via `SYMBOL_ALPHABET`. Exact inverse of `target_to_symbols`. Raises ValueError if any
    symbol value is out of range for `palette` (or `palette` itself is invalid)."""
    bits_per_symbol(palette)  # validates palette is in VALID_PALETTES
    out_chars: List[str] = []
    for s in symbols:
        if not (0 <= s < palette):
            raise ValueError(f"symbol value {s!r} out of range for palette={palette}")
        out_chars.append(SYMBOL_ALPHABET[s])
    return "".join(out_chars)


def target_to_symbols(target: str, palette: int) -> List[int]:
    """Parse a target string produced by `symbols_to_target` back into a symbol list. Raises
    ValueError on any character not in `SYMBOL_ALPHABET`, or on a decoded value out of range for
    `palette` (which happens if `target` was produced with a different, larger palette)."""
    bits_per_symbol(palette)  # validates palette is in VALID_PALETTES
    lookup = {ch: i for i, ch in enumerate(SYMBOL_ALPHABET)}
    symbols: List[int] = []
    for ch in target:
        if ch not in lookup:
            raise ValueError(f"character {ch!r} is not in SYMBOL_ALPHABET")
        value = lookup[ch]
        if value >= palette:
            raise ValueError(f"decoded symbol value {value} out of range for palette={palette}")
        symbols.append(value)
    return symbols


def random_payload(seed: int, size: int) -> bytes:
    """Deterministic pseudo-random payload of `size` bytes from `seed` alone (same construction
    `heliogram.harness._random_payload` uses, duplicated here rather than imported since it is a
    private helper of that module)."""
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


@dataclass
class Example:
    """One synthetic training example. `image` is what a decoder (pixel or VLM) would actually
    see -- corrupted if `corruption != "clean"`. `target`/`payload` are always ground truth read
    from the CLEAN pre-corruption image, never re-derived from the (possibly corrupted)
    `image` itself."""

    image: Image.Image
    target: str
    payload: bytes
    palette: int
    subpatch: int
    patch_size: int
    nsym: int
    seed: int
    corruption: str


def generate_examples(
    n: int,
    palettes: Sequence[int] = DEFAULT_PALETTES,
    subpatches: Sequence[int] = DEFAULT_SUBPATCHES,
    payload_sizes: Sequence[int] = DEFAULT_PAYLOAD_SIZES,
    patch_size: int = PATCH_SIZE,
    nsym: int = 32,
    seed: int = 0,
    corruptions: Optional[Dict[str, Callable[[Image.Image], Image.Image]]] = None,
    corruption_prob: float = 0.0,
) -> Iterator[Example]:
    """Deterministically generate `n` synthetic (image, target) examples: for each, pick a
    (palette, subpatch, payload_size) uniformly at random (seeded), build a random payload,
    encode() it, read the ground-truth symbols off the CLEAN image via extract_symbols(), then
    -- with probability `corruption_prob` (default 0.0: augmentation OFF unless explicitly
    requested) -- apply one randomly chosen corruption from `corruptions` (default
    DEFAULT_CORRUPTIONS) to the image the example actually carries. The `target` string always
    reflects the clean image regardless of whether the returned `image` was corrupted.

    `palettes` defaults to `DEFAULT_PALETTES` (`(64, 128, 256)`) -- the large-palette-under-
    corruption bet this module's defaults now target, see the module docstring's "THE BET"
    paragraph -- not the full `VALID_PALETTES` range. Pass `palettes=VALID_PALETTES` (or any
    other subset) explicitly for the old "every palette, uniformly" behavior.

    Deterministic: fixed (n, seed, palettes, subpatches, payload_sizes, corruption_prob,
    corruptions) always yields the same sequence of examples, because a single `random.Random
    (seed)` instance drives every random choice below in a fixed order, and `encode()` /
    the corruption functions are themselves deterministic (no hidden randomness).

    Curriculum note (for scripts/train_qlora.py): this function is stage-agnostic -- it just
    draws uniformly from whatever `palettes`/`subpatches`/`payload_sizes`/`corruption_prob` it's
    given. A curriculum (low density/clean first, wider palette + corruption later) is built by
    calling this repeatedly with narrower ranges per stage, not by anything in this function --
    see `scripts/train_qlora.py`'s `build_curriculum()`, retargeted (Slice C) to concentrate its
    later stages on exactly `DEFAULT_PALETTES` with corruption turned on, per the bet above.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n!r}")
    palettes = list(palettes)
    subpatches = list(subpatches)
    payload_sizes = list(payload_sizes)
    if not palettes or not subpatches or not payload_sizes:
        raise ValueError("palettes, subpatches, and payload_sizes must all be non-empty")
    bad_palettes = [p for p in palettes if p not in VALID_PALETTES]
    if bad_palettes:
        raise ValueError(
            f"palettes contains invalid value(s) {bad_palettes}, must be a subset of "
            f"{VALID_PALETTES}"
        )
    bad_subpatches = [s for s in subpatches if s not in VALID_SUBPATCHES]
    if bad_subpatches:
        raise ValueError(
            f"subpatches contains invalid value(s) {bad_subpatches}, must be a subset of "
            f"{VALID_SUBPATCHES}"
        )
    if not (0.0 <= corruption_prob <= 1.0):
        raise ValueError(f"corruption_prob must be in [0, 1], got {corruption_prob!r}")

    if corruptions is None:
        corruptions = DEFAULT_CORRUPTIONS
    # Corruption draws exclude "clean" (a no-op is already what happens when corruption_prob
    # doesn't fire); if the caller's own `corruptions` dict has no non-clean entries, that's a
    # config error worth surfacing rather than silently always choosing "clean".
    non_clean_names = [name for name in corruptions if name != "clean"]
    if corruption_prob > 0.0 and not non_clean_names:
        raise ValueError("corruption_prob > 0 but `corruptions` has no non-'clean' entries")

    rng = random.Random(seed)
    for i in range(n):
        palette = rng.choice(palettes)
        subpatch = rng.choice(subpatches)
        payload_size = rng.choice(payload_sizes)
        trial_seed = seed + i

        payload = random_payload(trial_seed, payload_size)
        clean_img = encode(
            payload, palette=palette, patch_size=patch_size, nsym=nsym, seed=0, subpatch=subpatch
        )
        _, _, truth_symbols = extract_symbols(
            clean_img, palette=palette, patch_size=patch_size, subpatch=subpatch
        )
        target = symbols_to_target(truth_symbols, palette)

        if corruption_prob > 0.0 and rng.random() < corruption_prob:
            corruption_name = rng.choice(non_clean_names)
        else:
            corruption_name = "clean"
        out_img = corruptions[corruption_name](clean_img)
        if out_img.size != clean_img.size:  # corruption fns are documented to preserve size,
            out_img = out_img.resize(clean_img.size)  # but guard against a caller-supplied one

        yield Example(
            image=out_img,
            target=target,
            payload=payload,
            palette=palette,
            subpatch=subpatch,
            patch_size=patch_size,
            nsym=nsym,
            seed=trial_seed,
            corruption=corruption_name,
        )


def write_dataset(
    out_dir: Union[Path, str],
    n: int,
    *,
    palettes: Sequence[int] = DEFAULT_PALETTES,
    subpatches: Sequence[int] = DEFAULT_SUBPATCHES,
    payload_sizes: Sequence[int] = DEFAULT_PAYLOAD_SIZES,
    patch_size: int = PATCH_SIZE,
    nsym: int = 32,
    seed: int = 0,
    corruptions: Optional[Dict[str, Callable[[Image.Image], Image.Image]]] = None,
    corruption_prob: float = 0.0,
    image_format: str = "png",
) -> Path:
    """Materialize `n` examples from `generate_examples(...)` to disk under `out_dir`: one image
    per example in `out_dir/images/`, plus `out_dir/manifest.jsonl` with one JSON object per
    line. Each record always has at least the four keys the Phase-2 scaffold contract requires
    -- `image_path` (relative to `out_dir`), `palette`, `subpatch`, `target` -- plus a few extra
    fields (`payload_size`, `patch_size`, `nsym`, `corruption`, `seed`) kept for reproducibility
    and debugging. Returns the manifest path. Deterministic for fixed (n, seed, and the other
    generate_examples arguments) -- see that function's docstring, including its note on
    `palettes` defaulting to `DEFAULT_PALETTES` rather than the full `VALID_PALETTES` range.
    """
    out_dir = Path(out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    examples = generate_examples(
        n,
        palettes=palettes,
        subpatches=subpatches,
        payload_sizes=payload_sizes,
        patch_size=patch_size,
        nsym=nsym,
        seed=seed,
        corruptions=corruptions,
        corruption_prob=corruption_prob,
    )

    width = max(1, len(str(max(n - 1, 0))))
    with open(manifest_path, "w") as manifest:
        for i, ex in enumerate(examples):
            image_name = f"{i:0{width}d}.{image_format}"
            ex.image.save(images_dir / image_name)
            record = {
                "image_path": str(Path("images") / image_name),
                "palette": ex.palette,
                "subpatch": ex.subpatch,
                "target": ex.target,
                "payload_size": len(ex.payload),
                "patch_size": ex.patch_size,
                "nsym": ex.nsym,
                "corruption": ex.corruption,
                "seed": ex.seed,
            }
            manifest.write(json.dumps(record) + "\n")
    return manifest_path


def iter_manifest(manifest_path: Union[Path, str]) -> Iterator[Dict[str, object]]:
    """Read a `manifest.jsonl` written by `write_dataset` back into records, resolving
    `image_path` to an absolute path (write_dataset stores it relative to the manifest's own
    directory, e.g. "images/000.png"). Pure stdlib (json + pathlib) -- no heavy deps, safe to
    import/call eagerly from anywhere, including scripts/train_qlora.py's dataset-building step.
    """
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["image_path"] = str(base_dir / record["image_path"])
            yield record
