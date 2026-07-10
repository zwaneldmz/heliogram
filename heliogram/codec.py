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
    produce a byte-identical PNG (the palette/layout have no randomness; `seed` is accepted for
    API stability / future use, e.g. dithering, but has no effect on v0.1 output).

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
    HeliogramDecodeError if the recovered stream cannot be parsed or RS-corrected.

    `subpatch` must match the value `img` was encoded with (see `encode`'s docstring); it is not
    a self-describing field in the byte stream.
    """
    bps = bits_per_symbol(palette)
    _, _, symbols = extract_symbols(img, palette=palette, patch_size=patch_size, subpatch=subpatch)
    stream = _symbols_to_bytes(symbols, bps)

    if len(stream) < 5:
        raise HeliogramDecodeError("recovered stream shorter than the 5-byte framing header")

    payload_len = struct.unpack(">I", stream[1:5])[0]
    message_len = 5 + payload_len
    ecc_len = rs_encoded_length(message_len, nsym)
    if len(stream) < ecc_len:
        raise HeliogramDecodeError(
            f"recovered stream too short ({len(stream)}B) for the framed message "
            f"({ecc_len}B expected) -- image likely too corrupted or too small"
        )

    ecc_bytes = bytes(stream[:ecc_len])
    try:
        decoded_message, _, _ = RSCodec(nsym).decode(ecc_bytes)
    except Exception as exc:  # reedsolo raises ReedSolomonError on uncorrectable input
        raise HeliogramDecodeError(f"Reed-Solomon decode failed: {exc}") from exc

    decoded_message = bytes(decoded_message)
    if len(decoded_message) < message_len:
        raise HeliogramDecodeError("decoded message shorter than the expected framing")
    if decoded_message[0] != CODEC_VERSION:
        raise HeliogramDecodeError(
            f"unsupported/corrupted codec version byte {decoded_message[0]!r} "
            f"(expected {CODEC_VERSION})"
        )

    return decoded_message[5 : 5 + payload_len]


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
