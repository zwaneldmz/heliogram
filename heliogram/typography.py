"""heliogram.typography -- CPU-only GEOMETRIC de-risk gate for the "dense typeset glyphs" pivot.

CONTEXT (see NOTES.md and README.md): heliogram's solid-color-block codec is measured dead as a
compression scheme -- its per-patch net ceiling is `log2(256) * 223/255 ~= 6.996 bits/patch`
(spec/format-v0.1.md, README.md), strictly below the measured text-context bars for shipping
bytes as text in a VLM's own context (`heliogram/data/text_baselines.json`: base64 = 8.096
bits/token, ascii85 = 8.374 bits/token -- ascii85, not base64, is the STRONGEST measured
encoding and therefore the honest bar). The proposed pivot -- following DeepSeek-OCR/Glyph
(NOTES.md section 1) -- is to render the payload as DENSE TYPESET GLYPHS instead of solid-color
blocks, betting that a VLM tower's pretrained OCR competence preserves rendered text where it
does not preserve arbitrary color symbols.

THIS MODULE ANSWERS THE CHEAP QUESTION FIRST, BEFORE ANY GPU IS SPENT: what is the GEOMETRIC
bits/patch of typeset text at the same 14px patch grid the codec uses, honestly accounting for
Reed-Solomon ECC and framing overhead -- and does it even clear the ~8.4 bits/token text bar IN
PRINCIPLE (assuming perfect legibility)? If it cannot, the pivot is dead cheaply, no GPU needed.
If it can, the pivot survives to a later (GPU, out of scope here) readability test of whether an
actual un-fine-tuned VLM can OCR glyphs this dense.

DATA HONESTY -- READ THIS BEFORE TRUSTING ANY NUMBER BELOW: every measurement in this module is
GEOMETRIC PACKING DENSITY ONLY. It assumes PERFECT legibility -- that every glyph, however small,
is read back exactly. No OCR runs here, no model is loaded, nothing about a real VLM's ability to
resolve small glyphs through its own ViT patch tokenizer is measured. This makes every bits/patch
number here an UPPER BOUND on what a real reader could extract, never a lower bound and never a
capability claim. Passing the bars checked here is NECESSARY for the pivot to be worth a GPU
readability test, but nowhere near SUFFICIENT -- and because shrinking the font indefinitely
trivially increases geometric density without limit (this is a model-free measurement; it knows
nothing about a legibility floor), "does SOME font size clear the bar" is close to a foregone
conclusion. The interesting number this module actually surfaces is HOW SMALL the font must get to
clear each bar, since that determines how hard the (unmeasured, Phase-2, GPU) readability test
downstream will be.

Measures two variants per font size, both rendered as ascii85 text (base64.a85encode -- the
STRONGEST measured text encoding per heliogram.baselines, not base64):

  raw   ascii85(payload) rendered directly, NO error correction. This is the fair, apples-to-
        apples comparison against the measured TEXT-TOKEN bars (base64/ascii85 bits/token):
        those bars are also the cost of shipping raw text with no added ECC -- a text token
        chain has no legibility risk once tokenized, so it needs none.
  rs    the payload is FIRST framed exactly as heliogram.codec.encode does (version byte +
        4-byte big-endian length + payload, then reedsolo.RSCodec(nsym).encode(...) at the same
        nsym=32 the codec sweep uses), and THAT ecc-protected byte stream is what gets
        ascii85-rendered. This is the fair, apples-to-apples comparison against the COLOR
        CODEC's 6.996 net ceiling, which already bakes in the identical nsym=32 RS overhead --
        a real glyph channel would need the same kind of ECC to be trustworthy once legibility
        is imperfect, exactly the same reason the color codec carries it.

bits/patch is always `payload_bits / total_patches` -- the ORIGINAL payload's bits (never the
inflated ecc byte count) divided by every 14px patch the rendered image occupies, INCLUDING
margins/padding needed to round the canvas up to a whole number of patches in each dimension --
exactly the "TRUE PAYLOAD DENSITY" convention heliogram.harness._bits_per_patch_on_success uses
for the color codec (payload_bytes*8/total_patches, calibration row and padding both counted
against the format, never credited as free capacity). See that function's docstring for the
historical bug (~3x overstatement) this convention exists to avoid repeating here.
"""

from __future__ import annotations

import argparse
import base64
import math
import random
import struct
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont
from reedsolo import RSCodec

from .baselines import MeasuredTextBaselines, load_measured_text_baselines
from .codec import CODEC_VERSION, PATCH_SIZE

__all__ = [
    "COLOR_CODEC_NET_CEILING_BITS_PER_PATCH",
    "DEFAULT_FONT_SIZES_PX",
    "DEFAULT_PAYLOAD_SIZE",
    "DEFAULT_NSYM",
    "MONOSPACE_FONT_CANDIDATES",
    "TypesetDensity",
    "TypographyRow",
    "ReferenceBars",
    "load_reference_bars",
    "render_typeset_density",
    "sweep_typography",
    "build_parser",
    "main",
]

# log2(256) * 223/255 -- the color codec's measured *net* per-patch ceiling as payload size grows
# without bound (README.md "Headline"/"Capacity" sections, spec/format-v0.1.md section on the
# capacity limit): 223/255 is exactly `(RS_NSIZE - nsym) / RS_NSIZE` at nsym=32, i.e. the
# asymptotic RS parity overhead fraction once chunking amortizes fully. Pinned here as a literal,
# not re-derived, because it is a PUBLISHED measured number this module compares against, not a
# quantity this module computes itself -- see tests/test_typography.py for a cross-check that it
# matches `log2(256) * 223/255` to the stated precision.
COLOR_CODEC_NET_CEILING_BITS_PER_PATCH = 6.996

# Representative font sizes (px, FreeType "size" argument -- roughly nominal em height) spanning
# "about one glyph's advance width per 14px patch" (14px) down to a deliberately illegible-to-a-
# human-eye size (4px) that nonetheless packs many ascii85 characters per patch geometrically --
# see this module's DATA HONESTY note on why "does some size clear the bar" is not the
# interesting question; the per-size table is.
DEFAULT_FONT_SIZES_PX: Tuple[int, ...] = (14, 12, 10, 8, 6, 4)

DEFAULT_PAYLOAD_SIZE = 4096  # bytes; matches heliogram.harness.SWEEP_PAYLOAD_SIZES' mid tier
DEFAULT_NSYM = 32            # matches heliogram.codec.encode's default / heliogram.harness.NSYM

# Deterministic, real monospace TrueType font files searched in order. A monospace font is not
# optional here: the chars-per-line packing math below assumes every glyph has the SAME advance
# width (measured once via `font.getlength("M")`), which only a genuinely monospace font
# guarantees -- Pillow's own bundled `ImageFont.load_default(size=N)` font is proportional (its
# 'i' is roughly a third the width of its 'M'), so it is deliberately NOT used as a fallback here:
# silently packing proportional glyphs under a monospace assumption would overstate chars/line
# for narrow-heavy alphabets and understate it for wide-heavy ones -- exactly the kind of silent
# error this project's honesty culture forbids (see heliogram.baselines' RuntimeError-not-
# fabricated-fallback convention for measure_base64_baseline/measure_text_encoding_baselines,
# the same contract this module follows for font loading).
MONOSPACE_FONT_CANDIDATES: Tuple[str, ...] = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Courier_New.ttf",
    "/Library/Fonts/Courier New.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "C:\\Windows\\Fonts\\consola.ttf",
)

_font_cache: dict = {}


def _load_monospace_font(font_size_px: int) -> Tuple[ImageFont.FreeTypeFont, str]:
    """Load the first available candidate monospace TrueType font at `font_size_px`, verifying it
    is actually monospace (two visually dissimilar-width glyphs, 'i' and 'M', must measure the
    same advance width) before trusting it for the packing math below.

    Raises RuntimeError -- never silently substitutes a proportional font or a fabricated width
    -- if no candidate file exists on this system, or if the file that does exist turns out not
    to be monospace. Results are memoized per (path, size) for the CLI sweep's sake.
    """
    cache_key = font_size_px
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    tried = []
    for path in MONOSPACE_FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, font_size_px)
        except OSError:
            tried.append(path)
            continue
        w_i = font.getlength("i")
        w_m = font.getlength("M")
        if abs(w_i - w_m) > 0.01:
            tried.append(f"{path} (not monospace: 'i'={w_i} vs 'M'={w_m})")
            continue
        _font_cache[cache_key] = (font, path)
        return font, path

    raise RuntimeError(
        "heliogram.typography needs a real monospace TrueType font to measure typeset packing "
        f"density honestly (chars-per-line packing assumes uniform glyph advance width) -- none "
        f"of the candidate paths worked: {tried}. Install one (e.g. `apt install fonts-dejavu-"
        "core` for DejaVu Sans Mono) or add a working path to "
        "heliogram.typography.MONOSPACE_FONT_CANDIDATES. This deliberately does NOT fall back to "
        "Pillow's bundled proportional default font under a 'monospace' label -- see this "
        "function's docstring."
    )


def _rs_frame(payload: bytes, nsym: int) -> bytes:
    """Frame `payload` EXACTLY as heliogram.codec.encode does before RS-protecting it: version
    byte + 4-byte big-endian length + payload, then reedsolo.RSCodec(nsym).encode(...). Reusing
    this exact framing (not a simplified stand-in) is what makes the 'rs' variant an
    apples-to-apples ECC comparison against the color codec's own nsym=32 overhead."""
    message = bytes([CODEC_VERSION]) + struct.pack(">I", len(payload)) + bytes(payload)
    return bytes(RSCodec(nsym).encode(message))


def _layout_canvas(
    text_len: int, char_w: float, line_h: int, patch_size: int, align: int = 1
) -> Tuple[int, int, int, int]:
    """Size a square-ish canvas (in whole `align * patch_size`-px multiples, so margins/padding
    needed to round up are counted -- never cropped away) that fits `text_len` monospace
    characters of advance width `char_w` and line height `line_h`, and return
    (chars_per_line, n_lines, canvas_w_px, canvas_h_px).

    `align` (default 1: round to whole `patch_size`-px multiples, the original convention shared
    with heliogram.baselines.rendered_text_density) rounds BOTH canvas dimensions up to a multiple
    of `align * patch_size` px. `align=2` yields dimensions that are multiples of `2 * patch_size`
    = 28px at the default 14px patch -- exactly the snap unit of Qwen2/2.5-VL's `smart_resize`, so
    a canvas laid out with `align=2` is fed to the model as the IDENTITY transform (no
    uncontrolled resample onto a different pixel grid before the ViT reads it), the same reason
    heliogram.codec.encode grew its own `align` parameter. This is REQUIRED, not cosmetic, for the
    readability measurement in heliogram.ocr_eval: without it every rendered OCR image hits
    smart_resize and the model grades a resampled (blurred/rescaled) image, not the rendering (see
    that module + scripts/run_typography_ocr.py's identity guard). The extra padding patches are
    counted against bits/patch honestly, exactly as the color codec's align padding is.
    """
    unit = align * patch_size
    target_w_px = max(
        unit,
        math.ceil(math.sqrt(max(text_len, 1)) * char_w / unit) * unit,
    )
    chars_per_line = max(1, int(target_w_px // char_w))
    n_lines = max(1, math.ceil(text_len / chars_per_line))
    target_h_px = max(unit, math.ceil(n_lines * line_h / unit) * unit)
    return chars_per_line, n_lines, target_w_px, target_h_px


@dataclass
class TypesetDensity:
    """One (font_size, ecc variant) geometric measurement. See module docstring's DATA HONESTY
    note: `bits_per_patch` is a model-free UPPER BOUND assuming perfect legibility, not a
    demonstrated reading capability."""

    font_size_px: int
    ecc_applied: bool
    nsym: Optional[int]
    payload_len: int
    rendered_len: int  # length (chars) of the ascii85 text actually typeset
    chars_per_line: int
    n_lines: int
    patches_wide: int
    patches_high: int
    total_patches: int
    chars_per_patch: float
    payload_bits: int
    bits_per_patch: float
    font_path: str
    image: Image.Image
    note: str


def render_typeset_density(
    payload: bytes,
    font_size_px: int,
    *,
    apply_rs: bool = False,
    nsym: int = DEFAULT_NSYM,
    patch_size: int = PATCH_SIZE,
    align: int = 1,
) -> TypesetDensity:
    """Typeset `payload` as ascii85 text (base64.a85encode -- the strongest measured text
    encoding, see module docstring) at `font_size_px` into an image sized on the `patch_size`
    grid, and report the GEOMETRIC (model-free, perfect-legibility) bits/patch.

    `align` (default 1: byte-identical geometry to prior releases -- every pinned bits/patch
    number in this module's tests is `align=1`) rounds the rendered canvas up to `align *
    patch_size`-px multiples in both dimensions -- see `_layout_canvas`. `align=2` makes the
    canvas a multiple of 28px so Qwen2/2.5-VL's `smart_resize` is the identity on it, which is
    what heliogram.ocr_eval passes for the readability measurement (an unaligned canvas is
    silently resampled by the processor before the model reads it). The extra padding patches
    lower bits/patch honestly, exactly as the color codec's `align=2` padding does.

    `apply_rs=False` (default): typesets `ascii85(payload)` directly -- NO error correction. Fair
    comparison against the measured text-TOKEN bars (base64/ascii85 bits/token), which likewise
    carry no ECC (a text token stream has no legibility risk).

    `apply_rs=True`: frames `payload` exactly as heliogram.codec.encode does (version + length +
    payload, then `reedsolo.RSCodec(nsym)`) and typesets `ascii85(ecc_bytes)` instead -- fair
    comparison against the color codec's own nsym=32-inclusive 6.996 bits/patch net ceiling.

    `bits_per_patch` is always `len(payload) * 8 / total_patches` -- the ORIGINAL payload's bits
    (RS parity bytes are framing overhead, never credited as payload) over EVERY patch the
    rendered canvas occupies, margins/padding included (see `_layout_canvas`). This can only ever
    go DOWN when `apply_rs=True` vs `False` for the same payload/font_size, since RS strictly
    inflates the rendered byte stream (and therefore `total_patches`) without changing the
    numerator.
    """
    stream = _rs_frame(payload, nsym) if apply_rs else bytes(payload)
    text = base64.a85encode(stream).decode("ascii")

    font, font_path = _load_monospace_font(font_size_px)
    char_w = font.getlength("M")
    ascent, descent = font.getmetrics()
    line_h = max(1, ascent + descent)

    chars_per_line, n_lines, canvas_w, canvas_h = _layout_canvas(
        len(text), char_w, line_h, patch_size, align
    )

    image = Image.new("L", (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(image)
    for line_idx in range(n_lines):
        line = text[line_idx * chars_per_line : (line_idx + 1) * chars_per_line]
        draw.text((0, line_idx * line_h), line, fill=0, font=font)

    patches_wide = canvas_w // patch_size
    patches_high = canvas_h // patch_size
    total_patches = patches_wide * patches_high
    payload_bits = len(payload) * 8

    return TypesetDensity(
        font_size_px=font_size_px,
        ecc_applied=apply_rs,
        nsym=nsym if apply_rs else None,
        payload_len=len(payload),
        rendered_len=len(text),
        chars_per_line=chars_per_line,
        n_lines=n_lines,
        patches_wide=patches_wide,
        patches_high=patches_high,
        total_patches=total_patches,
        chars_per_patch=len(text) / total_patches,
        payload_bits=payload_bits,
        bits_per_patch=payload_bits / total_patches,
        font_path=font_path,
        image=image,
        note=(
            "geometric/model-free: assumes perfect legibility, an UPPER BOUND only -- see "
            "heliogram.typography's module docstring. "
            + (
                f"RS-framed (nsym={nsym}) before rendering -- ECC-honest comparison against the "
                "color codec's own nsym-inclusive net ceiling."
                if apply_rs
                else "raw text, NO error correction -- comparison against the measured "
                "text-token bars, which likewise carry none."
            )
        ),
    )


@dataclass
class ReferenceBars:
    """The three reference lines this pivot must be measured against (see module docstring)."""

    color_codec_net_ceiling: float
    base64_bits_per_token: Optional[float]
    ascii85_bits_per_token: Optional[float]
    note: str


def load_reference_bars() -> ReferenceBars:
    """Load the base64/ascii85 measured text bars via heliogram.baselines.
    load_measured_text_baselines(), degrading gracefully (never fabricating a number) if
    heliogram/data/text_baselines.json is absent -- same soft-read contract as every other
    consumer of that file in this repo (heliogram.harness in particular)."""
    measured: Optional[MeasuredTextBaselines] = load_measured_text_baselines()
    if measured is None:
        return ReferenceBars(
            color_codec_net_ceiling=COLOR_CODEC_NET_CEILING_BITS_PER_PATCH,
            base64_bits_per_token=None,
            ascii85_bits_per_token=None,
            note=(
                "no measured text baseline found at heliogram/data/text_baselines.json -- only "
                f"the color codec's {COLOR_CODEC_NET_CEILING_BITS_PER_PATCH} bits/patch net "
                "ceiling is available as a reference line. Run `python -m heliogram.baselines "
                "--measure` (needs transformers + HF Hub access) to produce the text bars."
            ),
        )
    base64_e = measured.encodings.get("base64")
    ascii85_e = measured.encodings.get("ascii85")
    return ReferenceBars(
        color_codec_net_ceiling=COLOR_CODEC_NET_CEILING_BITS_PER_PATCH,
        base64_bits_per_token=base64_e.bits_per_token if base64_e else None,
        ascii85_bits_per_token=ascii85_e.bits_per_token if ascii85_e else None,
        note=measured.note,
    )


@dataclass
class TypographyRow:
    """One font size's raw-vs-RS comparison against all three reference bars. `beats_*` flags
    are computed against the RS-framed (ECC-honest) bits/patch, per this module's docstring:
    a real glyph channel needs the same kind of ECC the color codec carries to be trustworthy
    once legibility is imperfect, so the RS number -- not the no-ECC raw number -- is the fair
    headline figure. Raw is still reported alongside it for context (it upper-bounds the RS
    figure and shows how much of the gap, if any, is pure RS overhead vs geometric packing)."""

    font_size_px: int
    chars_per_patch_raw: float
    bits_per_patch_raw: float
    bits_per_patch_rs: float
    total_patches_raw: int
    total_patches_rs: int
    beats_color_codec_ceiling: bool
    beats_base64_bar: Optional[bool]
    beats_ascii85_bar: Optional[bool]


def sweep_typography(
    payload: bytes,
    font_sizes_px: Sequence[int] = DEFAULT_FONT_SIZES_PX,
    *,
    nsym: int = DEFAULT_NSYM,
    patch_size: int = PATCH_SIZE,
    bars: Optional[ReferenceBars] = None,
    align: int = 1,
) -> List[TypographyRow]:
    """Run `render_typeset_density` for both ecc variants at every font size in `font_sizes_px`,
    and compare each RS-framed bits/patch against `bars` (loaded via `load_reference_bars()` if
    not given). Returns one `TypographyRow` per font size, in the same order as `font_sizes_px`.

    `align` (default 1) is forwarded to `render_typeset_density` -- pass `align=2` to report the
    density of the SAME 28px-aligned canvas heliogram.ocr_eval actually feeds the model, so the
    "beats 8.374?" verdict reflects the image the model sees (identity under smart_resize), not a
    tighter unaligned canvas the processor would have resampled anyway.
    """
    if bars is None:
        bars = load_reference_bars()

    rows: List[TypographyRow] = []
    for size in font_sizes_px:
        raw = render_typeset_density(
            payload, size, apply_rs=False, patch_size=patch_size, align=align
        )
        rs = render_typeset_density(
            payload, size, apply_rs=True, nsym=nsym, patch_size=patch_size, align=align
        )
        rows.append(
            TypographyRow(
                font_size_px=size,
                chars_per_patch_raw=raw.chars_per_patch,
                bits_per_patch_raw=raw.bits_per_patch,
                bits_per_patch_rs=rs.bits_per_patch,
                total_patches_raw=raw.total_patches,
                total_patches_rs=rs.total_patches,
                beats_color_codec_ceiling=rs.bits_per_patch > bars.color_codec_net_ceiling,
                beats_base64_bar=(
                    rs.bits_per_patch > bars.base64_bits_per_token
                    if bars.base64_bits_per_token is not None
                    else None
                ),
                beats_ascii85_bar=(
                    rs.bits_per_patch > bars.ascii85_bits_per_token
                    if bars.ascii85_bits_per_token is not None
                    else None
                ),
            )
        )
    return rows


# --------------------------------------------------------------------------------------------
# CLI: `python -m heliogram.typography`
# --------------------------------------------------------------------------------------------


def _fmt_bool(b: Optional[bool]) -> str:
    if b is None:
        return "n/a"
    return "YES" if b else "no"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only, model-free GEOMETRIC de-risk gate for the 'dense typeset glyphs' pivot "
            "(see this module's docstring): reports geometric bits/patch of typeset ascii85 "
            "text at a sweep of font sizes, raw and RS-framed (nsym=32, matching the color "
            "codec), against the color codec's 6.996 bits/patch net ceiling and the measured "
            "text-token bars. Assumes PERFECT legibility -- an upper bound only, not a "
            "capability claim; see the module docstring's DATA HONESTY note."
        )
    )
    parser.add_argument(
        "--payload-size",
        type=int,
        default=DEFAULT_PAYLOAD_SIZE,
        help=f"synthetic payload size in bytes (default: {DEFAULT_PAYLOAD_SIZE})",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for the synthetic payload")
    parser.add_argument(
        "--font-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_FONT_SIZES_PX),
        help=f"font sizes (px) to sweep (default: {list(DEFAULT_FONT_SIZES_PX)})",
    )
    parser.add_argument("--nsym", type=int, default=DEFAULT_NSYM, help="RS parity bytes/chunk")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE, help="patch grid unit, px")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    rng = random.Random(args.seed)
    payload = bytes(rng.getrandbits(8) for _ in range(args.payload_size))

    bars = load_reference_bars()
    rows = sweep_typography(
        payload,
        args.font_sizes,
        nsym=args.nsym,
        patch_size=args.patch_size,
        bars=bars,
    )

    print(
        f"heliogram.typography -- GEOMETRIC (model-free, perfect-legibility) de-risk gate for "
        f"the dense-typeset-glyphs pivot"
    )
    print(f"payload: {args.payload_size} bytes (seed={args.seed}), patch_size={args.patch_size}px")
    print(f"reference: color codec net ceiling = {bars.color_codec_net_ceiling:.3f} bits/patch")
    if bars.ascii85_bits_per_token is not None:
        print(
            f"reference: measured ascii85 text-token bar = {bars.ascii85_bits_per_token:.3f} "
            "bits/token (strongest measured text encoding)"
        )
    if bars.base64_bits_per_token is not None:
        print(f"reference: measured base64 text-token bar = {bars.base64_bits_per_token:.3f} bits/token")
    if bars.base64_bits_per_token is None and bars.ascii85_bits_per_token is None:
        print(f"NOTE: {bars.note}")
    print()

    header = (
        f"{'font(px)':>8} | {'chars/patch(raw)':>17} | {'bits/patch(raw)':>16} | "
        f"{'bits/patch(RS)':>15} | {'beats 6.996?':>12} | {'beats 8.374?':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.font_size_px:>8} | {row.chars_per_patch_raw:>17.3f} | "
            f"{row.bits_per_patch_raw:>16.3f} | {row.bits_per_patch_rs:>15.3f} | "
            f"{_fmt_bool(row.beats_color_codec_ceiling):>12} | "
            f"{_fmt_bool(row.beats_ascii85_bar):>12}"
        )
    print()
    print(
        "columns are computed on the RS-framed (nsym=%d) bits/patch -- the ECC-honest number, "
        "fair against the color codec's own RS-inclusive ceiling and the harder of the two "
        "bars to clear; raw (no-ECC) bits/patch is shown alongside for context only." % args.nsym
    )

    any_beats_ceiling = any(r.beats_color_codec_ceiling for r in rows)
    any_beats_ascii85 = any(r.beats_ascii85_bar for r in rows if r.beats_ascii85_bar is not None)
    if not any_beats_ceiling:
        verdict = (
            "VERDICT: typeset glyphs do NOT geometrically clear even the color codec's own "
            "6.996 bits/patch ceiling at any swept font size -- the pivot is DEAD, cheaply, "
            "with no GPU spent."
        )
    elif bars.ascii85_bits_per_token is not None and not any_beats_ascii85:
        verdict = (
            "VERDICT: typeset glyphs geometrically clear the color codec's 6.996 bits/patch "
            "ceiling at some swept font size, but NOT the measured ~8.374 bits/token text bar -- "
            "the pivot beats the codec it would replace but does not beat the thing it needs to "
            "beat to matter economically. Dead as stated; would need denser packing than swept "
            "here to survive."
        )
    else:
        verdict = (
            "VERDICT: typeset glyphs geometrically clear BOTH the color codec's 6.996 bits/patch "
            "ceiling and the measured ~8.374 bits/token text bar at some swept font size -- this "
            "is a GEOMETRIC UPPER BOUND ONLY (perfect legibility assumed, no OCR run). The pivot "
            "survives this cheap CPU gate and is a candidate for the (unmeasured, GPU-gated) "
            "Phase-2 readability test; passing here is necessary, nowhere near sufficient."
        )
    print()
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
