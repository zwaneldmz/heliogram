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

PROCESSOR RESIZE HAZARD (why `generate_examples`/`write_dataset` emit only 28px-aligned images):
Qwen2-VL/Qwen2.5-VL's own image processor calls a `smart_resize` step that snaps input pixel
dimensions to a multiple of `patch_size * merge_size` (14px * 2 = 28px for Qwen2.5-VL) BEFORE
the vision tower ever sees the image -- this is unconditional processor behavior, not something
a prompt or fine-tune can opt out of (it is now also a measured corruption row:
`heliogram.harness.CORRUPTIONS["qwen_smart_resize"]`). A grid whose patch-count width or height
is ODD has a pixel dimension that is an odd multiple of 14, i.e. NOT a multiple of 28 -- the
processor's `smart_resize` then resamples that dimension onto a different pixel grid than the
one `encode()` painted, an UNCONTROLLED corruption introduced by merely feeding the image to the
model. `generate_examples` fixes this at the source with `heliogram.codec.encode(..., align=2)`:
the grid is rounded up to even patch dimensions BEFORE layout, so every emitted image's pixel
dimensions are exact multiples of 28 and `smart_resize` (given matching `min_pixels`/
`max_pixels`, see `scripts/train_qlora.py`'s `_identity_pixel_bounds`) is the identity
transform -- and, unlike the earlier post-hoc `pad_to_even_patch_grid` construction this
replaced, an align=2 image is an ordinary v0.1 grid that `decode_pixels` round-trips bit-exactly
in every case (post-hoc COLUMN padding broke round-trip; see `pad_to_even_patch_grid`'s
docstring, kept for callers that must pad an EXISTING image's pixels).

PROMPT/OUTPUT-CONTRACT UNIFICATION (D5(b)/D5(d) of the Phase-2 scaffold review): `build_prompt`,
`format_output_text`, and `parse_output_text` below are the ONLY prompt-wording and
target<->response-text conversion functions in this repo. `heliogram.vlm.QwenVLDecoder` (both
its live-inference prompt and its response parsing) and `scripts/train_qlora.py` (its training-
target construction) both call these exact functions rather than building their own wording --
before this unification, `scripts/train_qlora.py`'s training prompt and
`heliogram.vlm.QwenVLDecoder._build_prompt`'s inference prompt were two independently-written
strings that happened to describe the same task; since a fine-tuned model is prompt-brittle (its
weights were updated against ONE specific wording), any drift between the training-time and
inference-time prompt is a silent capability tax paid at inference time for no reason. The output
contract itself also changed here (see `parse_output_text`'s docstring): row-per-line plain text,
no fenced code block, because `SYMBOL_ALPHABET` deliberately includes the backtick character
(index 89, only reachable once `palette` > 64) and a fence-based contract cannot tell a
legitimate run of backtick DATA symbols apart from the model's own closing fence.
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from .codec import (
    PATCH_SIZE,
    VALID_PALETTES,
    VALID_SUBPATCHES,
    bits_per_symbol,
    encode,
    extract_symbols,
    get_palette,
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
    "pad_to_even_patch_grid",
    "n_data_cells",
    "build_prompt",
    "format_output_text",
    "parse_output_text",
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


def n_data_cells(width: int, height: int, subpatch: int = 1) -> int:
    """Number of DATA symbol cells in a `width` x `height` PATCH grid (row 0 is always the
    calibration row, excluded) at the given `subpatch` (k x k sub-cells per data patch). A tiny,
    single-source-of-truth helper -- `build_prompt` (what it tells the model to output),
    `heliogram.vlm.expected_max_new_tokens` (how many tokens to budget for that same output), and
    `heliogram.vlm.teacher_forced_symbol_accuracy` (how many target positions to score) all call
    this exact function so the three can never silently drift apart from each other."""
    return width * (height - 1) * subpatch * subpatch


def build_prompt(palette: int, width: int, height: int, subpatch: int = 1) -> str:
    """THE canonical heliogram-grid transcription instruction prompt (D5(b) of the Phase-2
    scaffold review) -- see the module docstring's "PROMPT/OUTPUT-CONTRACT UNIFICATION" note.
    `heliogram.vlm.QwenVLDecoder._build_prompt` (live inference) and `scripts/train_qlora.py`'s
    training-target construction both call this exact function; neither builds its own wording.

    `width`/`height` are PATCH grid dimensions (as returned by `heliogram.codec.extract_symbols`,
    or `img.width // patch_size` / `img.height // patch_size`), so the model is told exactly how
    many rows/characters to produce -- see `n_data_cells`.

    OUTPUT CONTRACT: row-per-line plain text, NO fenced code block (see `format_output_text` /
    `parse_output_text`'s docstrings for why: `SYMBOL_ALPHABET` deliberately includes the
    backtick character, so a fence-based contract cannot distinguish a legitimate run of backtick
    DATA symbols from the model's own closing fence -- this was an actual parsing hazard in an
    earlier version of this prompt, see D5(d) of the Phase-2 scaffold review).
    """
    bits_per_symbol(palette)  # validates palette is in VALID_PALETTES
    if width < 1 or height < 2:
        raise ValueError(f"width/height must describe at least one data row, got {(width, height)}")
    alphabet_prefix = SYMBOL_ALPHABET[:palette]
    row_len = width * subpatch * subpatch
    subcell_note = (
        f" Each data patch is itself subdivided into a {subpatch}x{subpatch} grid of solid-color "
        "sub-cells (top-left, top-right, ... row-major); read those sub-cells in row-major order "
        "before moving to the next patch -- each LINE of your output still corresponds to one "
        "row of DATA PATCHES, not one row of sub-cells."
        if subpatch > 1
        else ""
    )
    return (
        "This image is a heliogram-encoded data grid. Row 0 (the top row) is a CALIBRATION row "
        f"cycling through a {palette}-color palette, in color-index order 0..{palette - 1}. "
        f"Every row below it is DATA: each cell is one solid color from that same {palette}-color "
        "palette." + subcell_note + " Read the data cells in row-major order (left to right, top "
        "to bottom) and classify each one's color against the calibration row's colors to get "
        f"its color index (0..{palette - 1}). For a cell with color index i, its character is "
        f'position i (0-indexed) of this exact string: "{alphabet_prefix}". Output exactly '
        f"{height - 1} lines, one line per data-patch row, each line holding exactly {row_len} "
        "characters (that row's cells' characters, in order, with no separators). Output nothing "
        "else: no explanation, no code fence, no leading or trailing whitespace on any line."
    )


def format_output_text(target: str, width: int, subpatch: int = 1) -> str:
    """Convert a compact `target` string (one character per symbol, see `symbols_to_target`) into
    the row-per-line plain-text form that IS the model's training/inference output contract (see
    `build_prompt` / the module docstring's "PROMPT/OUTPUT-CONTRACT UNIFICATION" note): one line
    per DATA-PATCH ROW (`len(target) // (width * subpatch * subpatch)` of them, i.e. `height - 1`
    from the original `width` x `height` patch grid), each line holding exactly
    `width * subpatch * subpatch` characters.

    This is a pure RE-CHUNKING (insert `"\\n"` every `width * subpatch * subpatch` characters),
    never a reordering: `target`'s character order already IS row-major over data-patch rows,
    then (for `subpatch` > 1) row-major over each patch's sub-cells within that row -- exactly
    what one "line" of `width * subpatch * subpatch` consecutive characters covers. Exact inverse
    (up to the whitespace `parse_output_text` strips): `parse_output_text`.

    Raises ValueError if `width * subpatch * subpatch` does not evenly divide `len(target)` --
    which would mean `target` was not actually produced for a grid of this `width`/`subpatch`.
    """
    row_len = width * subpatch * subpatch
    if row_len <= 0:
        raise ValueError(f"width * subpatch * subpatch must be positive, got {row_len}")
    if len(target) % row_len != 0:
        raise ValueError(
            f"target length ({len(target)}) is not a multiple of width*subpatch*subpatch "
            f"({row_len}) -- cannot split into equal-length rows; target was not produced for a "
            f"grid with width={width}, subpatch={subpatch}"
        )
    return "\n".join(target[i : i + row_len] for i in range(0, len(target), row_len))


def parse_output_text(text: str) -> str:
    """Inverse of `format_output_text`: strip ALL whitespace (spaces and newlines alike) and
    return the resulting compact target string, ready for `target_to_symbols`.

    This is deliberately the ENTIRE parsing logic -- no fenced-code-block detection, no other
    heuristic. `heliogram.vlm._extract_symbol_string` (the response-parsing entry point used by
    `QwenVLDecoder`/`zero_shot_symbol_error`) delegates to this exact function, so training-target
    construction (`scripts/train_qlora.py`) and inference-time response parsing
    (`heliogram.vlm`) go through the identical operation rather than two independent
    reimplementations of "strip whitespace". See `build_prompt`'s docstring for why the output
    contract has no fence to strip: `SYMBOL_ALPHABET` includes the backtick character, so a
    fence-detecting parser cannot tell a legitimate run of backtick DATA symbols apart from the
    model's own closing fence -- row-per-line plain text sidesteps the ambiguity entirely instead
    of trying to resolve it, and newlines double as resync anchors a fence never provided.
    """
    return "".join(text.split())


def random_payload(seed: int, size: int) -> bytes:
    """Deterministic pseudo-random payload of `size` bytes from `seed` alone (same construction
    `heliogram.harness._random_payload` uses, duplicated here rather than imported since it is a
    private helper of that module)."""
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


def pad_to_even_patch_grid(img: Image.Image, patch_size: int, palette: int) -> Image.Image:
    """Pad `img` (a freshly `encode()`'d, still-clean heliogram grid) so its patch-grid
    dimensions (`img.width // patch_size`, `img.height // patch_size`) are BOTH even -- see the
    module docstring's "PROCESSOR RESIZE HAZARD" note for why this matters (Qwen's image
    processor snaps pixel dimensions to a multiple of `patch_size * 2`, and an odd patch-count
    dimension is exactly the case where that snap resamples the image off its own symbol
    lattice). No-op (returns `img` itself, unchanged) when both dimensions are already even.

    NO LONGER `generate_examples`' construction: dataset generation now aligns the grid BEFORE
    layout via `heliogram.codec.encode(..., align=2)` (see the module docstring), which has no
    decode-roundtrip caveat at all. This function is kept for callers that must pad an EXISTING
    image's pixels (an image they did not encode themselves, or one whose layout is already
    fixed) -- with the documented COLUMN-padding tradeoff below.

    HOW: extends the image by one extra column (if width is odd) and/or one extra row (if height
    is odd), each exactly `patch_size` px, filled with LEGITIMATE calibration/data colors (never
    an arbitrary placeholder), so a fresh `extract_symbols(padded_img, ...)` call reads a fully
    coherent, self-consistent grid:
      - row 0 (the calibration row)'s extra column, if any, continues the calibration cycle --
        `get_palette(palette)[new_column_index % palette]`, exactly what `encode()` itself would
        have painted had the grid naturally been one column wider -- so row 0 stays a correct
        calibration row of its new (even) width, and `extract_symbols`' calibration-color
        recovery (which groups columns by `i % palette`) is not biased by an inconsistent entry.
      - every other newly-added cell (the extra data row's cells, and the extra column's cells in
        every data row) is painted `get_palette(palette)[0]` (symbol 0) -- a fixed, deterministic,
        documented choice. These cells never correspond to real payload bytes: `encode()` already
        zero-pads any unused DATA capacity within its OWN computed grid (see `codec.encode`'s
        docstring), and this padding is a further, DATASET-GENERATION-ONLY extension strictly
        beyond that, purely to satisfy the processor's alignment requirement -- a model
        mis-transcribing them has zero bearing on payload recovery.

    GROUND TRUTH for the padded cells is never hand-asserted here: `generate_examples` re-derives
    `target` by calling `extract_symbols` on the RETURNED (padded) image, the same
    "definitionally correct by construction" trick this module uses everywhere else (see module
    docstring) -- this function only makes pixel-level painting decisions consistent with
    `heliogram.codec.get_palette`; it has no notion of RS/framing.

    DATA HONESTY / KNOWN TRADEOFF: adding a ROW is provably safe for `decode_pixels` byte-for-byte
    roundtrip -- the new row's symbols are read strictly AFTER every pre-existing (real-or-
    already-zero-padded) symbol in row-major order, i.e. as a pure suffix beyond what
    `decode_pixels`' RS-decode step (`stream[:ecc_len]`) ever looks at. Adding a COLUMN is NOT
    safe: `extract_symbols` reads a FIXED width for every row, so an inserted column places one
    extra symbol at the END OF EVERY ROW, in the MIDDLE of the overall row-major symbol stream --
    shifting bit alignment for any payload long enough to span multiple rows (verified: RS decode
    fails outright on a column-padded image carrying real content, since a shifted bitstream is
    not merely "corrupted", it is a structurally different byte sequence). This is fine and
    expected for THIS module's actual use (`Example.image` is generated for VLM/processor
    consumption, never for `decode_pixels`) but is worth knowing if a padded `Example.image` is
    reused for something else: only images whose width was ALREADY even (no column added) still
    support a `decode_pixels` roundtrip after padding.

    `subpatch` is not a parameter here because it does not need to be: row 0 (calibration) is
    always full `patch_size` x `patch_size` patches regardless of `subpatch` (see
    `codec.encode`'s own contract), and this function operates purely on whole-patch columns/rows
    -- it never looks inside a patch's sub-cells, so it is subpatch-agnostic by construction.
    """
    width = img.width // patch_size
    height = img.height // patch_size
    if width % 2 == 0 and height % 2 == 0:
        return img

    colors = get_palette(palette)
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    if width % 2 == 1:
        extra_col = np.empty((height * patch_size, patch_size, 3), dtype=np.uint8)
        extra_col[0:patch_size, :] = colors[width % palette]  # row 0: continue calibration cycle
        extra_col[patch_size:, :] = colors[0]  # data rows: fixed symbol 0
        arr = np.concatenate([arr, extra_col], axis=1)
        width += 1

    if height % 2 == 1:
        extra_row = np.empty((patch_size, width * patch_size, 3), dtype=np.uint8)
        extra_row[:, :] = colors[0]  # a new DATA row -- fixed symbol 0 throughout
        arr = np.concatenate([arr, extra_row], axis=0)
        height += 1

    return Image.fromarray(arr)


@dataclass
class Example:
    """One synthetic training example. `image` is what a decoder (pixel or VLM) would actually
    see -- corrupted if `corruption != "clean"`, and always patch-grid-even (see
    `pad_to_even_patch_grid`) even when `corruption == "clean"`. `target`/`payload` are always
    ground truth read from the CLEAN, EVEN-PADDED pre-corruption image, never re-derived from
    the (possibly corrupted) `image` itself."""

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
    encode() it, PAD it to an even patch-grid width/height (`pad_to_even_patch_grid` -- see the
    module docstring's "PROCESSOR RESIZE HAZARD" note), read the ground-truth symbols off that
    CLEAN, EVEN-PADDED image via extract_symbols(), then -- with probability `corruption_prob`
    (default 0.0: augmentation OFF unless explicitly requested) -- apply one randomly chosen
    corruption from `corruptions` (default DEFAULT_CORRUPTIONS) to the image the example actually
    carries. The `target` string always reflects the clean, even-padded image regardless of
    whether the returned `image` was corrupted.

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
        # align=2 makes encode() itself emit an even patch-grid width/height (pixel dims are
        # multiples of 28 = patch*merge), so Qwen's smart_resize is the identity on every image
        # this generator emits -- see the module docstring's "PROCESSOR RESIZE HAZARD" note.
        # This replaced the earlier encode-then-pad_to_even_patch_grid construction: grid
        # alignment BEFORE layout keeps the image an ordinary, decode_pixels-round-trippable
        # v0.1 grid in every case (post-hoc COLUMN padding broke round-trip, a documented
        # tradeoff pad_to_even_patch_grid still carries for callers that need pixel-level
        # padding of an existing image), and it is exactly what a production encoder targeting
        # a 2x2-merge VLM should ship (heliogram.codec.encode(..., align=2)) -- so training
        # images and deployment images are now the SAME construction, not merely equivalent.
        # `target` below is read off this aligned image, so it always matches exactly what a
        # downstream VLM prompt (build_prompt) will ask for.
        clean_img = encode(
            payload,
            palette=palette,
            patch_size=patch_size,
            nsym=nsym,
            seed=0,
            subpatch=subpatch,
            align=2,
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
