"""heliogram.codec -- the pinned heliogram codec v0.1: byte<->symbol packing over a patch grid.

See spec/format-v0.1.md for the full mini-RFC. This module implements exactly that format:

- PATCH_SIZE px solid-color blocks are the symbol unit (default 14px, ~1 ViT patch/token).
- Palette size P in {2,4,8,16,32,64,128,256}; bits_per_symbol = log2(P). Colors are
  deterministic: P<=16 uses evenly spaced HSV hues at full saturation/value, exactly as in the
  original v0.1 release (unchanged numbers). P>16 (32, 64, 128, 256) additionally tiles more
  VALUE levels over that same 16-hue wheel, since hue alone stops being separable at that
  density -- see get_palette's docstring. Same P always yields the same colors, no RNG.
  DATA HONESTY: P=128/256 clean-decode exactly on this reference pixel decoder (all colors
  distinct, nearest-neighbor classification is exact on an uncorrupted image) but are MEASURED
  to FAIL under realistic JPEG q70 compression on this same decoder -- chroma subsampling erodes
  hue separation faster than 256-way nearest-neighbor can tolerate. That is a fact about this
  pixel decoder, not a capability claim about a learned (VLM) reader -- see spec/format-v0.1.md
  section 2a and tests/test_roundtrip.py's pinned known-failure tests.
- Row 0 of the patch grid is a CALIBRATION ROW: patch i = palette[i % P]. This lets a decoder
  recover the actual on-image RGB for every palette color after corruption, before classifying
  data patches against those recovered colors (nearest-neighbor). Row 0 is always full patches,
  regardless of `subpatch` below.
- All patches after row 0 are DATA patches, filled row-major. With `subpatch` (k, default 1) > 1,
  each data patch is itself subdivided into a k x k grid of solid-color sub-cells, each holding
  one symbol (k*k symbols/data patch instead of 1) -- a purely geometric density knob; k=1
  reproduces today's exact byte-for-byte output. DATA HONESTY note: decode_pixels can read
  sub-cells trivially because it samples exact pixel centers off a known grid; a real ViT/VLM
  image encoder tokenizes at its own fixed patch grid and likely CANNOT resolve structure below
  one patch. So subpatch>1 capacity numbers from this module are a geometric UPPER BOUND on the
  channel, not a demonstrated model capability -- see spec/format-v0.1.md section 6a.
- Payload framing: message = version(1B) + payload_len(4B, big-endian) + payload. The framed
  message is Reed-Solomon protected (reedsolo.RSCodec, `nsym` parity bytes per internal chunk).
  The resulting ecc bytes are bit-packed (MSB-first) into log2(P)-bit symbols and written
  row-major into data patches (and, for subpatch>1, row-major into each patch's sub-cells). Any
  data patches beyond what the ecc bytes need are padded with symbol 0. The grid is auto-sized
  to the smallest roughly-square shape that fits the required symbols, with width >= palette so
  every color appears at least once in the calibration row.

decode_pixels() is the reference, model-free decoder: sample each patch's (or sub-cell's) center
pixel, recover calibration colors from row 0, nearest-neighbor classify data patches, rebuild the
ecc bitstream, RS-decode, strip the header, and return the payload. It is deliberately the whole
channel: no semantic understanding, just raw color->symbol recovery. Phase 2 (out of scope here
-- no GPU available) replaces this classifier with a fine-tuned VLM; see VLMDecoder below.

Header recovery: `payload_len` (the 4-byte field above) is NOT trusted from the uncorrected
symbol stream -- it is always re-derived from the Reed-Solomon-CORRECTED message, with a
guess-and-one-retry fast path, a multi-chunk first-codeword recovery pass, and a bounded
single-chunk length scan as successive fallbacks before giving up. See decode_pixels' own
docstring for the exact three-pass order and the DATA HONESTY caveat on RS mis-correction risk
beyond its guaranteed-detection budget (this fixes what used to be a single point of failure
sitting outside the ECC guarantee -- see git history / the fix that added this note for details).
"""

from __future__ import annotations

import colorsys
import math
import struct
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from reedsolo import RSCodec

__all__ = [
    "CODEC_VERSION",
    "PATCH_SIZE",
    "RS_NSIZE",
    "VALID_PALETTES",
    "VALID_SUBPATCHES",
    "HeliogramDecodeError",
    "get_palette",
    "bits_per_symbol",
    "rs_encoded_length",
    "compute_grid",
    "extract_symbols",
    "encode",
    "decode_pixels",
    "decode",
    "VLMDecoder",
]

CODEC_VERSION = 1          # format version byte written into every framed message
PATCH_SIZE = 14            # default patch size in px (~1 ViT patch/token)
RS_NSIZE = 255             # reedsolo's GF(256) codeword size (its library default/max)
VALID_PALETTES = (2, 4, 8, 16, 32, 64, 128, 256)
VALID_SUBPATCHES = (1, 2)  # `subpatch` (k): k x k solid-color sub-cells per data patch


class HeliogramDecodeError(Exception):
    """Raised by decode_pixels when the recovered stream cannot be parsed or RS-decoded."""


def _check_palette(palette: int) -> None:
    if palette not in VALID_PALETTES:
        raise ValueError(f"palette must be one of {VALID_PALETTES}, got {palette!r}")


def _check_subpatch(patch_size: int, subpatch: int) -> None:
    if subpatch not in VALID_SUBPATCHES:
        raise ValueError(f"subpatch must be one of {VALID_SUBPATCHES}, got {subpatch!r}")
    if patch_size % subpatch != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be evenly divisible by subpatch ({subpatch})"
        )


def bits_per_symbol(palette: int) -> int:
    """log2(palette); palette is always a power of two in VALID_PALETTES."""
    _check_palette(palette)
    return palette.bit_length() - 1


def get_palette(palette: int) -> List[Tuple[int, int, int]]:
    """Deterministic, separable RGB palette for a given size. Same `palette` -> same colors.

    P <= 16: colors are evenly spaced HSV hues at full saturation/value: hue_i = i / palette,
    S=V=1.0. This is the original v0.1 scheme, byte-for-byte UNCHANGED for P in {2,4,8,16} --
    the reference values pinned in spec/format-v0.1.md stay valid.

    P > 16 (32, 64, 128, 256 -- the larger sizes in VALID_PALETTES): hue-only separation runs
    out at this density -- adjacent hues get close in RGB, and JPEG's chroma subsampling erodes
    hue (chroma) differences faster than brightness (luma) ones. So colors additionally vary in
    VALUE: the palette is `levels` value-levels (`levels = palette // 16`; 2 for P=32, 4 for
    P=64, 8 for P=128, 16 for P=256) of the same 16 hues used by P=16 (hue_i = i / 16), at S=1.0
    and value stepped evenly from 1.0 down to a floor of 0.4. This is a DESIGN CHOICE to add a
    second separation axis riding on luma (which JPEG preserves at full spatial resolution,
    unlike subsampled chroma) -- it is not empirically validated against corruption in this
    slice; see spec/format-v0.1.md section 2a. Still fully deterministic: no RNG, same P -> same
    colors, and all P colors are guaranteed distinct (see tests/test_roundtrip.py), including
    get_palette(256).

    Extending to P=128/256 reused this exact formula unchanged (only `levels` grows to 8/16) --
    the P<=64 branch and the P>16 branch's code path are untouched, so get_palette(P) for P in
    {2,4,8,16,32,64} is byte-identical to every prior release
    (tests/test_roundtrip.py::test_get_palette_le64_byte_identical_to_pinned_values pins this
    against SHA-256 digests captured from git history, not just a re-derivation of the current
    code). P=128/256 clean-decode exactly
    on decode_pixels (see tests/test_roundtrip.py) but are MEASURED to fail under JPEG q70 on
    this reference pixel decoder -- see the module docstring's DATA HONESTY note and
    spec/format-v0.1.md section 2a. Nobody has tested whether a learned (VLM) reader could do
    better; that is exactly the open Phase-2 question, not something this function's determinism
    settles either way.
    """
    _check_palette(palette)
    if palette <= 16:
        colors = []
        for i in range(palette):
            hue = i / palette
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors.append((round(r * 255), round(g * 255), round(b * 255)))
        return colors

    # P > 16: tile the same 16-way hue wheel across `levels` value levels so all P colors stay
    # distinct once hue alone runs out of separation. hues=16 keeps the brightest level
    # (value=1.0) identical to the P=16 hue set; levels = palette // hues (2 for 32, 4 for 64,
    # 8 for 128, 16 for 256).
    hues = 16
    levels = palette // hues
    v_min = 0.4
    colors = []
    for lvl in range(levels):
        value = 1.0 if levels == 1 else 1.0 - lvl * (1.0 - v_min) / (levels - 1)
        for h in range(hues):
            hue = h / hues
            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, value)
            colors.append((round(r * 255), round(g * 255), round(b * 255)))
    return colors


def rs_encoded_length(message_len: int, nsym: int, nsize: int = RS_NSIZE) -> int:
    """Total byte length of reedsolo.RSCodec(nsym).encode(message) for a message of this length.

    reedsolo chunks messages longer than (nsize - nsym) bytes and appends `nsym` parity bytes to
    *each* chunk, so for large messages the total ecc length is message_len + nsym * num_chunks,
    not simply message_len + nsym.
    """
    chunk_size = nsize - nsym
    if chunk_size <= 0:
        raise ValueError("nsym must be less than nsize")
    num_chunks = max(1, math.ceil(message_len / chunk_size))
    return message_len + num_chunks * nsym


def compute_grid(num_symbols: int, palette: int) -> Tuple[int, int]:
    """Smallest roughly-square (width, height) patch grid whose data patches (every row after
    row 0) can hold `num_symbols` symbols, with width >= palette so every calibration color
    appears at least once in row 0.

    This function itself has no notion of sub-patch packing (see `encode`'s `subpatch` param):
    `num_symbols` here means "count of things laid out row-major across data patches, one per
    patch". When sub-patch packing puts k*k symbols in each data patch, callers pass the
    DATA-PATCH count (`ceil(total_symbols / k**2)`), not the raw total-symbol count.
    """
    _check_palette(palette)
    width = max(palette, math.ceil(math.sqrt(max(num_symbols, 1))))
    data_rows = max(1, math.ceil(num_symbols / width))
    height = data_rows + 1  # +1 for the calibration row
    return width, height


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))  # MSB-first per byte


def _pack_symbols(ecc_bytes: bytes, bps: int) -> List[int]:
    """ecc bytes -> MSB-first bitstream -> zero-padded to a multiple of `bps` -> symbol values."""
    bits = _bytes_to_bits(ecc_bytes)
    pad = (-len(bits)) % bps
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    bits = bits.reshape(-1, bps)
    weights = 1 << np.arange(bps - 1, -1, -1)
    return (bits * weights).sum(axis=1).astype(int).tolist()


def _symbols_to_bytes(symbols: Sequence[int], bps: int) -> bytes:
    """Inverse of _pack_symbols: symbol values -> MSB-first bitstream -> bytes (trailing bits
    that don't fill a whole byte are dropped)."""
    if not symbols:
        return b""
    arr = np.array(symbols, dtype=np.uint32)
    shifts = np.arange(bps - 1, -1, -1)
    bits = ((arr[:, None] >> shifts[None, :]) & 1).astype(np.uint8).reshape(-1)
    nbytes = len(bits) // 8
    bits = bits[: nbytes * 8]
    if nbytes == 0:
        return b""
    return bytes(np.packbits(bits))  # MSB-first per byte


def encode(
    data: bytes,
    palette: int = 8,
    patch_size: int = PATCH_SIZE,
    nsym: int = 32,
    seed: int = 0,
    subpatch: int = 1,
) -> Image.Image:
    """Encode `data` as a heliogram codec v0.1 image. Deterministic: identical arguments always
    produce a pixel-identical image (the palette/layout have no randomness; `seed` is accepted
    for API stability / future use, e.g. dithering, but has no effect on v0.1 output) -- the PNG
    CONTAINER bytes may differ across Pillow versions even for pixel-identical output (Pillow's
    PNG encoder is not itself pinned by this project), so pin img.tobytes() (plus size/mode), not
    a PNG-byte hash, if you need a byte-for-byte determinism guard; see
    tests/test_roundtrip.py::test_subpatch1_output_unchanged_pinned_hash.

    `subpatch` (k, default 1) subdivides each DATA patch into a k x k grid of `patch_size/k`-px
    solid-color sub-cells, each carrying one symbol -- k*k symbols per data patch instead of 1.
    Sub-cells are laid out row-major within the patch (top-left, top-right, bottom-left,
    bottom-right for k=2). The calibration row (row 0) is unaffected: it always stays full
    palette[i % P] patches, exactly as for k=1. `patch_size` must be evenly divisible by `k`,
    and `k` must be in VALID_SUBPATCHES (1 or 2), else ValueError. k=1 reproduces today's exact
    byte-for-byte output (back-compat). See the module docstring's DATA HONESTY note: subpatch>1
    is a geometric upper bound the pixel decoder reads trivially, not a claim about what a real
    ViT-patch VLM encoder can resolve.
    """
    _check_palette(palette)
    _check_subpatch(patch_size, subpatch)
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    if len(data) > 0xFFFFFFFF:
        raise ValueError("payload too large for a 4-byte length header")
    del seed  # reserved, see docstring

    bps = bits_per_symbol(palette)
    message = bytes([CODEC_VERSION]) + struct.pack(">I", len(data)) + bytes(data)
    ecc_bytes = bytes(RSCodec(nsym).encode(message))

    symbols = _pack_symbols(ecc_bytes, bps)
    num_symbols = len(symbols)

    k = subpatch
    cells_per_patch = k * k
    data_patches_needed = math.ceil(num_symbols / cells_per_patch)
    width, height = compute_grid(data_patches_needed, palette)
    data_patches = width * (height - 1)
    capacity = data_patches * cells_per_patch
    symbols = symbols + [0] * (capacity - num_symbols)

    colors = get_palette(palette)
    img_arr = np.zeros((height * patch_size, width * patch_size, 3), dtype=np.uint8)

    for i in range(width):  # calibration row -- always full patches, unaffected by subpatch
        color = colors[i % palette]
        img_arr[0:patch_size, i * patch_size : (i + 1) * patch_size] = color

    sub = patch_size // k
    idx = 0
    for r in range(1, height):  # data rows, row-major
        y0 = r * patch_size
        for c in range(width):
            x0 = c * patch_size
            for sr in range(k):  # sub-cell rows, row-major within the patch
                for sc in range(k):  # sub-cell cols
                    yy0 = y0 + sr * sub
                    xx0 = x0 + sc * sub
                    img_arr[yy0 : yy0 + sub, xx0 : xx0 + sub] = colors[symbols[idx]]
                    idx += 1

    return Image.fromarray(img_arr)


def extract_symbols(
    img: Image.Image, palette: int = 8, patch_size: int = PATCH_SIZE, subpatch: int = 1
) -> Tuple[int, int, List[int]]:
    """Classify every data sub-cell (every row after the row-0 calibration row, subdivided k x k
    per `subpatch`=k) against colors recovered from row 0. Returns (width, height, symbols):
    width/height are PATCH grid dimensions (row 0's calibration patches are always full
    patch_size x patch_size cells, regardless of `subpatch`). `symbols` is the classified values
    (0..palette-1) in (data-patch row-major, then sub-cell row-major) order -- length
    width*(height-1) for subpatch=1 (one symbol/data patch, as in v0.1) or
    width*(height-1)*subpatch*subpatch for subpatch=2 (top-left/top-right/bottom-left/
    bottom-right order within each patch).

    This is the shared classification core of decode_pixels. It is also exposed on its own
    because it is useful for measuring raw channel/symbol error rate independent of the
    RS/framing layers (see heliogram.harness).
    """
    _check_palette(palette)
    _check_subpatch(patch_size, subpatch)
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    img_h, img_w = arr.shape[0], arr.shape[1]
    width = img_w // patch_size
    height = img_h // patch_size
    if width < 1 or height < 2:
        raise HeliogramDecodeError(f"image too small for patch_size={patch_size}: {img.size}")

    half = patch_size // 2

    def center(row: int, col: int) -> np.ndarray:
        y = row * patch_size + half
        x = col * patch_size + half
        return arr[y, x]

    sums = np.zeros((palette, 3), dtype=np.float64)
    counts = np.zeros(palette, dtype=np.int64)
    for i in range(width):
        c = i % palette
        sums[c] += center(0, i)
        counts[c] += 1

    fallback = np.array(get_palette(palette), dtype=np.float64)
    have_sample = counts > 0
    recovered = np.where(
        have_sample[:, None],
        sums / np.maximum(counts, 1)[:, None],
        fallback,
    )

    k = subpatch
    sub = patch_size // k
    sub_half = sub // 2

    def sub_center(row: int, col: int, sr: int, sc: int) -> np.ndarray:
        y = row * patch_size + sr * sub + sub_half
        x = col * patch_size + sc * sub + sub_half
        return arr[y, x]

    symbols: List[int] = []
    for r in range(1, height):
        for c in range(width):
            for sr in range(k):
                for sc in range(k):
                    px = sub_center(r, c, sr, sc)
                    dists = np.sum((recovered - px) ** 2, axis=1)
                    symbols.append(int(np.argmin(dists)))
    return width, height, symbols


def _attempt(stream: bytes, payload_len: int, nsym: int, depth: int = 0) -> Optional[bytes]:
    """Try to Reed-Solomon decode `stream` on the hypothesis that the framed message's payload is
    `payload_len` bytes long (message_len = 5 + payload_len). Returns the recovered payload on
    success, or None on ANY failure (never raises) -- `decode_pixels` is responsible for turning a
    final None (every strategy exhausted) into HeliogramDecodeError.

    This is the core of the F2 fix ("header outside the ECC guarantee"): `payload_len` is only
    ever used here to decide how many bytes to feed `RSCodec.decode` -- the header that actually
    gets trusted for the returned slice is always re-read from `decoded_message`, i.e. AFTER
    Reed-Solomon has had a chance to correct it, never from the possibly-corrupted raw `stream`.
    So corruption landing on the ~40-bit length field no longer kills the decode the way it used
    to when that field was parsed straight from the uncorrected stream and trusted outright.

    depth=0 (the normal entry, called with a tentative/guessed `payload_len`): if the CORRECTED
    header's length field disagrees with the `payload_len` this call was invoked with, that
    corrected value just survived Reed-Solomon and is itself trustworthy -- retry once (depth=1)
    using it. depth=1 never recurses again: if its own corrected header still disagrees with what
    it was called with, guessing further would not be principled (most likely the ecc_len this
    call used doesn't align with the true codeword boundary at all), so give up and return None
    and let `decode_pixels`'s other recovery passes (multi-chunk header recovery, bounded scan)
    try a different strategy instead of spinning on more guesses here.
    """
    message_len = 5 + payload_len
    ecc_len = rs_encoded_length(message_len, nsym)
    if len(stream) < ecc_len:
        return None
    try:
        decoded_message, _, _ = RSCodec(nsym).decode(bytes(stream[:ecc_len]))
    except Exception:  # reedsolo raises ReedSolomonError on uncorrectable input
        return None
    decoded_message = bytes(decoded_message)
    if len(decoded_message) < 5 or decoded_message[0] != CODEC_VERSION:
        return None
    hdr_len = struct.unpack(">I", decoded_message[1:5])[0]
    if hdr_len == payload_len:
        if len(decoded_message) < 5 + hdr_len:
            return None  # decoded shorter than its own declared length -- inconsistent, bail
        return decoded_message[5 : 5 + hdr_len]
    if depth == 0:
        return _attempt(stream, hdr_len, nsym, depth=1)
    return None


def decode_pixels(
    img: Image.Image,
    palette: int = 8,
    patch_size: int = PATCH_SIZE,
    nsym: int = 32,
    subpatch: int = 1,
) -> bytes:
    """Reference, model-free decoder. Sample each patch's (or, for subpatch=k>1, each sub-cell's)
    center pixel, recover calibration colors from row 0, nearest-neighbor classify data patches,
    rebuild the ecc bitstream, RS-decode, strip the header, and return the payload. Raises
    HeliogramDecodeError if the recovered stream cannot be parsed or RS-corrected by any of the
    strategies below.

    `subpatch` must match the value `img` was encoded with (see `encode`'s docstring); it is not
    a self-describing field in the byte stream.

    HEADER RECOVERY (fixes F2, "header outside the ECC guarantee"): the framing header's
    `payload_len` field (message[1:5], 4 bytes big-endian) used to be parsed straight from the
    UNCORRECTED recovered symbol stream and then trusted for the final payload slice -- meaning
    corruption landing on those ~40 bits (a tiny fraction of a multi-hundred-byte codeword) could
    kill an otherwise perfectly Reed-Solomon-correctable decode. That length field sat OUTSIDE the
    very ECC guarantee protecting everything around it: a single point of failure in a format
    whose whole point is surviving corruption. That is fixed now -- the header is always read from
    the RS-CORRECTED message, never trusted uncorrected, via three fallback passes tried in order
    (clean-path decodes still return from pass 1, byte-for-byte identical to before this fix):

      1. Parse a tentative `payload_len` from `stream[1:5]` AS-IS (today's fast path -- correct
         for the overwhelming majority of clean or lightly-corrupted images) and hand it to
         `_attempt` (depth=0). `_attempt` re-derives the trustworthy, RS-corrected header from the
         decode itself, and if that disagrees with the tentative guess, retries once (depth=1)
         using the corrected value before giving up -- see `_attempt`'s docstring. This alone
         recovers header corruption within Reed-Solomon's correction budget (up to
         floor(nsym/2) byte errors in the chunk covering the header) that would previously have
         been fatal even though the RS layer could trivially have fixed it.
      2. If pass 1 fails and the recovered stream is at least RS_NSIZE (255) bytes long -- i.e.
         large enough that reedsolo would have split it into multiple chunks -- RS-decode just the
         first RS_NSIZE-byte codeword in isolation. reedsolo always emits a full RS_NSIZE-byte
         first chunk for a multi-chunk message, so that chunk alone contains the whole 5-byte
         header, protected by its own chunk's parity independent of chunk 2+. Read `payload_len`
         from THAT corrected header and retry `_attempt` with it. This recovers header corruption
         severe enough that pass 1's guess-and-one-retry both landed on the wrong codeword
         boundary and failed outright.
      3. If that also fails, bounded-scan every `message_len` from 5 up to the largest value
         reedsolo would still treat as a single chunk (RS_NSIZE - nsym), RS-decoding
         `stream[:message_len + nsym]` at each candidate. A candidate is accepted ONLY if the
         decoded version byte equals CODEC_VERSION AND the decoded length field is exactly
         self-consistent with the `message_len` tried (i.e. equals `message_len - 5`) -- this is a
         last resort, not a guess dressed up as a certainty: trailing symbol-0 grid padding after
         the true codeword overwhelmingly fails RS decode outright at the wrong boundary, and the
         version+length double-check makes a false accept at a boundary RS happened not to reject
         astronomically unlikely.
      4. If every pass above fails, raise HeliogramDecodeError.

    DATA HONESTY: bounded-distance Reed-Solomon decoding (what every `RSCodec.decode` call above
    does) GUARANTEES correction only up to t = floor(nsym/2) byte errors per RS_NSIZE-byte chunk,
    and GUARANTEES detection (raising rather than returning anything) only up to nsym-t errors.
    Beyond nsym-t errors in a chunk, RS decoding can MIS-CORRECT: land on a different, wrong, but
    internally-consistent codeword and return wrong bytes without raising at all -- a property of
    the code, not a bug in this implementation, and exactly why pass 3 above insists on BOTH the
    version byte and the length-field self-consistency check rather than accepting the first
    candidate that merely fails to raise. See the module docstring for the same caveat applied to
    RS-protection generally: "detects corruption" / "raises rather than silently returning wrong
    bytes" claims elsewhere in this project hold only up to nsym-t errors, not unconditionally.
    """
    bps = bits_per_symbol(palette)
    _, _, symbols = extract_symbols(img, palette=palette, patch_size=patch_size, subpatch=subpatch)
    stream = bytes(_symbols_to_bytes(symbols, bps))

    if len(stream) < 5:
        raise HeliogramDecodeError("recovered stream shorter than the 5-byte framing header")

    # Pass 1: tentative length from the uncorrected stream, same starting guess as before this
    # fix -- the fast path that is correct for the overwhelming majority of clean or lightly
    # corrupted images. `_attempt` re-derives the trustworthy (RS-corrected) header itself and
    # retries once if the tentative guess disagrees with it (see _attempt's docstring above).
    tentative_len = struct.unpack(">I", stream[1:5])[0]
    result = _attempt(stream, tentative_len, nsym)
    if result is not None:
        return result

    # Pass 2a: multi-chunk header recovery. reedsolo chunks messages longer than
    # (RS_NSIZE - nsym) bytes into RS_NSIZE-byte codewords, so a multi-chunk message's first
    # chunk is always a full, independently-correctable RS_NSIZE bytes containing the whole
    # 5-byte header. RS-decoding just that first chunk in isolation recovers a header that was
    # too corrupted for pass 1 (which guessed from the raw bytes and retried once) to land on the
    # right codeword boundary at all.
    if len(stream) >= RS_NSIZE:
        try:
            first_chunk, _, _ = RSCodec(nsym).decode(bytes(stream[:RS_NSIZE]))
        except Exception:  # reedsolo raises ReedSolomonError on uncorrectable input
            first_chunk = None
        if first_chunk is not None:
            first_chunk = bytes(first_chunk)
            if len(first_chunk) >= 5:
                candidate_len = struct.unpack(">I", first_chunk[1:5])[0]
                result = _attempt(stream, candidate_len, nsym)
                if result is not None:
                    return result

    # Pass 2b: bounded single-chunk scan, last resort. Only message_len values small enough that
    # reedsolo would treat the whole message as ONE chunk are tried (message_len <= RS_NSIZE -
    # nsym), so this never guesses chunk boundaries for a multi-chunk message (pass 2a handles
    # those). A candidate is accepted only if BOTH the corrected version byte matches
    # CODEC_VERSION AND the corrected length field is exactly self-consistent with the
    # message_len tried -- see this function's docstring for why that double-check makes a false
    # accept astronomically unlikely rather than merely unlikely.
    for message_len in range(5, RS_NSIZE - nsym + 1):
        ecc_len = message_len + nsym
        if ecc_len > min(len(stream), RS_NSIZE):
            break
        try:
            decoded, _, _ = RSCodec(nsym).decode(bytes(stream[:ecc_len]))
        except Exception:  # reedsolo raises ReedSolomonError on uncorrectable input
            continue
        decoded = bytes(decoded)
        if (
            len(decoded) >= message_len
            and decoded[0] == CODEC_VERSION
            and struct.unpack(">I", decoded[1:5])[0] == message_len - 5
        ):
            return decoded[5:message_len]

    raise HeliogramDecodeError(
        "could not recover a valid framed message from the recovered pixel stream: neither the "
        "tentative header (pass 1), RS-corrected multi-chunk header recovery (pass 2a), nor a "
        "bounded single-chunk length scan (pass 2b) produced a self-consistent version+length "
        f"header (recovered stream length {len(stream)}B) -- image likely too corrupted or too "
        "small"
    )


DecoderFn = Callable[..., bytes]


def decode(
    img: Image.Image,
    palette: int = 8,
    patch_size: int = PATCH_SIZE,
    nsym: int = 32,
    subpatch: int = 1,
    decoder: Optional[DecoderFn] = None,
) -> bytes:
    """Decode an image to payload bytes using `decoder` (defaults to decode_pixels, the
    reference no-model decoder). This is the Phase-2 plug point: pass
    decoder=VLMDecoder(model=...) to swap in a fine-tuned VLM decoder once one exists.

    `subpatch` is forwarded to `decoder` like `palette`/`patch_size`/`nsym` (it must match what
    `img` was encoded with). The VLMDecoder stub accepts and ignores `subpatch`, so
    `decode(img, decoder=VLMDecoder())` reaches its NotImplementedError as documented.
    """
    if decoder is None:
        decoder = decode_pixels
    return decoder(img, palette=palette, patch_size=patch_size, nsym=nsym, subpatch=subpatch)


class VLMDecoder:
    """Phase-2 plug point -- OUT OF SCOPE for this CPU-only v0.1 release (no GPU available).

    Intended shape: wrap a small VLM fine-tuned to read the heliogram symbol grid directly from
    pixels, replacing decode_pixels' per-patch nearest-neighbor classifier with a learned,
    corruption-robust one. It would still be responsible for producing a symbol grid (or raw
    payload bytes) that feeds the same framing/RS layer used here.

    Calling an instance raises NotImplementedError -- there is no fine-tuned model in this repo.
    See spec/format-v0.1.md and the README's "Roadmap / Phase-2 boundary" section for the plan
    and its Decision Gate.
    """

    def __init__(self, model: object = None, **kwargs: object) -> None:
        self.model = model
        self._kwargs = kwargs

    def __call__(
        self,
        img: Image.Image,
        palette: int = 8,
        patch_size: int = PATCH_SIZE,
        nsym: int = 32,
        subpatch: int = 1,
    ) -> bytes:
        # subpatch accepted to match decode()'s decoder contract (decode forwards it
        # unconditionally); a real VLM decoder must handle it. The stub ignores it.
        del subpatch
        raise NotImplementedError(
            "VLMDecoder is a Phase-2 plug point: fine-tune a VLM (GPU required) to classify "
            "heliogram patches directly, then feed its output through the same RS/framing "
            "layer decode_pixels uses. Not implemented in this Phase-1, CPU-only release."
        )
