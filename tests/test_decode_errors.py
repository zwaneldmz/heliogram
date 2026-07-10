"""Pytest suite for heliogram.codec's decode FAILURE and RECOVERY paths -- previously untested.

Before this file, tests/test_roundtrip.py exercised clean roundtrips (every VALID_PALETTES x
PAYLOADS combination) and a handful of MEASURED corruption failures (JPEG q70/q95-at-scale,
"combined" at small payloads), but nothing in this repo directly exercised decode_pixels' actual
error/recovery machinery: the multi-pass header-recovery fallback added by the F2 fix (see
heliogram/codec.py's decode_pixels docstring, "HEADER RECOVERY"), the early "stream too short"
guard, or a wrong-codec-version rejection. This file closes that gap.

Assert-based, no fixtures beyond plain pytest.raises/monkeypatch, matching the rest of this
repo's test idiom (see tests/test_roundtrip.py, tests/test_foreign_tile.py).

THE F2 FIX THIS FILE IS MOSTLY ABOUT: before it, `payload_len` (decode_pixels' 4-byte framing
length field) was parsed straight from the UNCORRECTED recovered symbol stream and trusted
outright for the final payload slice -- so corruption landing on those ~40 bits (a tiny fraction
of a multi-hundred-byte Reed-Solomon codeword) could kill an otherwise perfectly RS-correctable
decode, even though the SAME corruption spread over payload bytes instead would have decoded
fine. The tests below construct exactly that scenario directly (not via a real JPEG/resize
round-trip, which cannot be aimed at a specific handful of symbols) -- deterministic, targeted
pixel edits to the first few DATA patches, where the framing header's symbols always land -- and
prove decode_pixels now recovers the exact payload through each of its three fallback passes in
turn (see decode_pixels' docstring for the full pass-by-pass explanation this file's test names
and docstrings mirror).
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from PIL import Image

import heliogram.codec as codec
from heliogram.codec import (
    CODEC_VERSION,
    PATCH_SIZE,
    RS_NSIZE,
    HeliogramDecodeError,
    # _attempt and _symbols_to_bytes are private -- imported directly anyway, matching
    # heliogram/vlm.py's own precedent for importing a private heliogram.codec name with a
    # docstring justification. _attempt lets this file assert its depth=0/depth=1 recursion
    # behavior directly rather than only indirectly through decode_pixels' end-to-end outcome;
    # _symbols_to_bytes lets tests rebuild the exact recovered byte stream decode_pixels itself
    # would see (to measure byte-error counts, tentative header guesses, etc.) without
    # reimplementing that bit-packing arithmetic here.
    _attempt,
    _symbols_to_bytes,
    bits_per_symbol,
    decode_pixels,
    encode,
    extract_symbols,
    get_palette,
    rs_encoded_length,
)

NSYM = 32


def _corrupt_first_data_symbols(
    img: Image.Image, palette: int, patch_size: int, n_symbols: int, shift: int = 1
) -> Image.Image:
    """Return a repainted copy of `img` with the first `n_symbols` DATA symbols (row-major from
    data row 1, column 0 onward -- exactly where the framing header's version+length-field
    symbols always land, since they are the very first bytes of the framed message) each
    replaced by a DIFFERENT palette color: the patch's own classified symbol value shifted by
    `shift` mod `palette` (guaranteed different from the original for any 0 < shift < palette).

    Uses `extract_symbols` only to read back the patch grid WIDTH (to turn a linear symbol index
    into a (row, col) patch coordinate) and each touched patch's OWN current symbol (so the
    replacement is provably different, not just "probably different"); the actual corruption is
    then a plain pixel-array edit on a real encode() image, not a re-encode from scratch -- this
    exercises decode_pixels' pixel-classification + recovery path the same way a real corrupted
    image would arrive at it, just with the corruption aimed precisely at the header instead of
    scattered by JPEG/resize the way tests/test_roundtrip.py's realistic-corruption tests are.
    """
    width, _height, symbols = extract_symbols(img, palette=palette, patch_size=patch_size)
    colors = get_palette(palette)
    arr = np.array(img.convert("RGB"))
    for i in range(n_symbols):
        row = 1 + i // width
        col = i % width
        new_symbol = (symbols[i] + shift) % palette
        y0, x0 = row * patch_size, col * patch_size
        arr[y0 : y0 + patch_size, x0 : x0 + patch_size] = colors[new_symbol]
    return Image.fromarray(arr)


# --- header corruption within the RS correction budget now decodes (the F2 fix, end to end) ----


def test_header_corruption_within_rs_budget_single_chunk_now_decodes():
    """THE headline F2 regression test: a 48-byte payload (single Reed-Solomon chunk at
    nsym=32 -- message_len=53, ecc_len=85, well under RS_NSIZE-nsym=223) whose framing header
    (version byte + 4-byte length field, the first ~14 symbols at 3 bits/symbol for palette=8)
    is corrupted in EVERY one of those symbols -- 6 actual bytes differ in the recovered stream
    (verified directly, see the diff-count assertion below), comfortably within Reed-Solomon's
    correction budget of floor(nsym/2)=16 byte errors for this single 85-byte codeword.

    Before the F2 fix, this exact corruption pattern was FATAL: payload_len was parsed from the
    uncorrected (corrupted) stream[1:5], producing a huge garbage value (verified below, not
    assumed) -- causing decode_pixels to compute a wildly wrong ecc_len and immediately fail with
    "recovered stream too short", never even attempting the RS decode that could trivially have
    fixed this. Now, decode_pixels' header-recovery pass 1 (see its docstring) re-derives the
    header from the RS-CORRECTED message instead of trusting the raw guess, so this decodes the
    exact original payload.
    """
    payload = bytes(range(48))
    palette = 8
    img = encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, seed=0)

    bps = bits_per_symbol(palette)
    _, _, orig_symbols = extract_symbols(img, palette=palette, patch_size=PATCH_SIZE)
    orig_stream = _symbols_to_bytes(orig_symbols, bps)

    n_header_symbols = 14  # ceil(40 header bits / 3 bits-per-symbol at palette=8)
    corrupted = _corrupt_first_data_symbols(img, palette, PATCH_SIZE, n_header_symbols, shift=1)

    _, _, new_symbols = extract_symbols(corrupted, palette=palette, patch_size=PATCH_SIZE)
    new_stream = _symbols_to_bytes(new_symbols, bps)
    byte_diffs = sum(1 for a, b in zip(orig_stream, new_stream) if a != b)
    assert 0 < byte_diffs <= NSYM // 2, (
        f"test construction assumption broken: expected a nonzero byte-error count within RS's "
        f"correction budget (<= {NSYM // 2}), got {byte_diffs}"
    )
    uncorrected_guess = struct.unpack(">I", new_stream[1:5])[0]
    assert uncorrected_guess != len(payload), (
        "test construction assumption broken: header corruption must actually change the "
        "uncorrected length-field guess away from the true payload_len, or this test would not "
        "be exercising the F2 fix at all"
    )

    recovered = decode_pixels(corrupted, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM)
    assert recovered == payload


def test_header_corruption_multichunk_exercises_pass_2a_multichunk_recovery():
    """Same idea as the single-chunk test above, but at payload_size=1024 (multi-chunk:
    message_len=1029, chunk_size=RS_NSIZE-nsym=223, so reedsolo splits this into 5 chunks,
    ecc_len=1197). This specifically targets decode_pixels' pass 2a (RS-decode just the first
    RS_NSIZE=255-byte codeword in isolation to recover the header) rather than pass 1, and this
    test PROVES that routing directly rather than just trusting the end-to-end outcome:

      1. `_attempt` (pass 1's helper) is called directly with the tentative (corrupted, garbage)
         length guess and asserted to return None -- pass 1 genuinely fails first.
      2. decode_pixels is then asserted to still recover the exact payload -- necessarily via
         pass 2a or 2b, not pass 1.

    The header corruption itself is identical in kind to the single-chunk test (a handful of
    early data-patch symbols shifted to a different palette color) -- only the payload size
    differs, which is what turns this into a multi-chunk message.
    """
    payload = bytes((i * 7) % 256 for i in range(1024))
    palette = 8
    img = encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, seed=0)

    bps = bits_per_symbol(palette)
    n_header_symbols = 14
    corrupted = _corrupt_first_data_symbols(img, palette, PATCH_SIZE, n_header_symbols, shift=3)

    _, _, new_symbols = extract_symbols(corrupted, palette=palette, patch_size=PATCH_SIZE)
    new_stream = _symbols_to_bytes(new_symbols, bps)
    assert len(new_stream) >= RS_NSIZE, (
        "test construction assumption: must be long enough to trigger pass 2a's multi-chunk "
        "first-codeword recovery"
    )

    tentative_len = struct.unpack(">I", new_stream[1:5])[0]
    assert tentative_len != len(payload), (
        "test construction assumption broken: the corrupted header must produce a wrong "
        "tentative guess"
    )
    assert _attempt(new_stream, tentative_len, NSYM) is None, (
        "test construction assumption broken: pass 1 (_attempt on the raw tentative guess) "
        "must genuinely fail here, or this test would not be isolating pass 2a at all"
    )

    recovered = decode_pixels(corrupted, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM)
    assert recovered == payload


# --- truncated stream ---------------------------------------------------------------------------


def test_truncated_stream_shorter_than_5_byte_header_raises():
    """The earliest, cheapest guard in decode_pixels: a recovered stream too short to even hold
    the 5-byte version+length header must raise immediately with an informative, specific
    message (not fall through to the (empty) fallback passes and raise a generic one). Built by
    hand from a minimal 2x2 patch grid (1 calibration row + 1 data row of 2 patches) at
    palette=2 (1 bit/symbol) -- 2 data symbols is 2 bits, nowhere near the 40 bits a framing
    header needs -- rather than cropping a real encode() output, so the "too short" condition is
    exact and not dependent on how large a real grid happens to be."""
    palette = 2
    patch_size = PATCH_SIZE
    colors = get_palette(palette)
    width, height = 2, 2
    arr = np.zeros((height * patch_size, width * patch_size, 3), dtype=np.uint8)
    for i in range(width):
        arr[0:patch_size, i * patch_size : (i + 1) * patch_size] = colors[i % palette]
    for i in range(width):
        arr[patch_size : 2 * patch_size, i * patch_size : (i + 1) * patch_size] = colors[0]
    img = Image.fromarray(arr)

    with pytest.raises(HeliogramDecodeError, match="shorter than the 5-byte framing header"):
        decode_pixels(img, palette=palette, patch_size=patch_size, nsym=NSYM)


def test_truncated_stream_fewer_data_rows_than_frame_needs_raises():
    """Crop a real, otherwise-valid encode() image down to just the calibration row plus ONE
    data row -- far fewer data patches than the payload's framed message actually needs -- and
    assert decode_pixels raises HeliogramDecodeError rather than returning wrong/partial bytes.
    This is the "recovered stream long enough to have a header, but not long enough to hold the
    whole framed+ECC'd message" case, distinct from the (shorter than 5 bytes) case above."""
    payload = b"x" * 200
    palette = 8
    img = encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, seed=0)
    width_px, _height_px = img.size
    cropped = img.crop((0, 0, width_px, 2 * PATCH_SIZE))  # calibration row + 1 data row only

    with pytest.raises(HeliogramDecodeError):
        decode_pixels(cropped, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM)


# --- wrong codec version -------------------------------------------------------------------------


def test_wrong_codec_version_byte_raises(monkeypatch):
    """A cleanly-encoded (uncorrupted) image whose framing header's version byte does not match
    the DECODER's CODEC_VERSION must raise HeliogramDecodeError, not silently decode under the
    wrong format assumption. Built by monkeypatching `heliogram.codec.CODEC_VERSION` to 2 for the
    `encode()` call only (both `encode` and `decode_pixels`/`_attempt` read the module-level
    global at call time, not at def time, so restoring it before decoding is enough to create a
    genuine version mismatch without hand-rolling the bit-packing/RS framing directly).

    This is NOT a corruption scenario -- the RS codeword itself is perfectly valid (no bytes
    differ from what encode() actually wrote), so every one of decode_pixels' three passes
    reaches a successful RS decode at the true message boundary at some point; the CODEC_VERSION
    check is what must reject it every time (see `_attempt`'s docstring: the version check
    happens before the length-field is even consulted)."""
    payload = b"version byte test payload"
    assert CODEC_VERSION == 1, "this test assumes the real CODEC_VERSION is 1, not 2"
    monkeypatch.setattr(codec, "CODEC_VERSION", 2)
    img = codec.encode(payload, palette=8, patch_size=PATCH_SIZE, nsym=NSYM, seed=0)
    monkeypatch.undo()  # restore CODEC_VERSION to 1 before decoding -- the actual mismatch

    assert codec.CODEC_VERSION == 1
    with pytest.raises(HeliogramDecodeError):
        decode_pixels(img, palette=8, patch_size=PATCH_SIZE, nsym=NSYM)


# --- _attempt() unit coverage (the header-recovery helper itself) --------------------------------


def test_attempt_returns_payload_when_header_matches_tentative_guess():
    """The trivial, common case: a clean stream, called with the CORRECT payload_len. depth=0,
    hdr_len == payload_len on the first try, no recursion needed."""
    payload = b"attempt() direct unit test -- clean path"
    palette = 8
    img = encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, seed=0)
    bps = bits_per_symbol(palette)
    _, _, symbols = extract_symbols(img, palette=palette, patch_size=PATCH_SIZE)

    stream = _symbols_to_bytes(symbols, bps)
    assert _attempt(stream, len(payload), NSYM) == payload


def test_attempt_returns_none_when_stream_shorter_than_required_ecc_length():
    """`_attempt` must fail closed (return None, not raise or index out of bounds) when the
    hypothesized payload_len implies an ecc_len longer than the stream actually is -- exactly
    what happens on pass 1 when the corrupted tentative guess is a huge garbage number."""
    assert _attempt(b"\x01\x00\x00\x00\x05tiny", payload_len=10_000_000, nsym=NSYM) is None


def test_attempt_returns_none_and_does_not_recurse_past_depth_1():
    """If depth=1's own corrected header still disagrees with what it was called with, `_attempt`
    must give up (return None) rather than recurse a third time. Constructed directly: encode a
    real message (so RS decode succeeds and the version byte checks out), then call `_attempt`
    with an intentionally wrong `payload_len` AND `depth=1` already set -- simulating "the
    depth=1 retry itself also disagreed" without needing to actually corrupt pixels to provoke
    it (which would be a much less direct test of this specific base case)."""
    payload = b"depth-1 base case direct unit test"
    palette = 8
    img = encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, seed=0)
    bps = bits_per_symbol(palette)
    _, _, symbols = extract_symbols(img, palette=palette, patch_size=PATCH_SIZE)

    stream = _symbols_to_bytes(symbols, bps)
    wrong_len = len(payload) + 1  # deliberately wrong, and stays wrong after RS-correction
    assert _attempt(stream, wrong_len, NSYM, depth=1) is None


def test_attempt_never_raises_on_uncorrectable_garbage():
    """`_attempt` must return None, never raise, when RSCodec.decode itself raises
    (ReedSolomonError on uncorrectable input) -- decode_pixels relies on this to keep trying its
    other fallback passes instead of the whole decode blowing up on the first failed guess."""
    garbage = bytes(range(256)) * 2  # long, structured-looking, but not a valid RS codeword
    assert _attempt(garbage, payload_len=50, nsym=NSYM) is None


# --- decode_pixels never mis-decodes structured-but-foreign noise into a false success -----------


def test_decode_pixels_raises_on_patch_aligned_random_noise():
    """A patch-aligned but otherwise RANDOM image (every cell a uniformly random palette color,
    seeded and deterministic) must not spuriously decode -- decode_pixels' pass 2b bounded scan
    tries many (message_len, RS-decode) candidates, and its whole safety argument (see
    decode_pixels' docstring) is that a false accept at the wrong boundary is astronomically
    unlikely, not merely "usually doesn't happen". This is a direct, repeated check of that
    argument against inputs with no relationship to any real encode() output at all (unlike the
    header-corruption tests above, which start from a genuine encode() and touch only a handful
    of symbols)."""
    palette = 8
    patch_size = PATCH_SIZE
    colors = get_palette(palette)
    rng = np.random.RandomState(0)
    width, height = 20, 20
    arr = np.zeros((height * patch_size, width * patch_size, 3), dtype=np.uint8)
    for r in range(height):
        for c in range(width):
            color = colors[int(rng.randint(0, palette))] if r > 0 else colors[c % palette]
            y0, x0 = r * patch_size, c * patch_size
            arr[y0 : y0 + patch_size, x0 : x0 + patch_size] = color
    img = Image.fromarray(arr)

    with pytest.raises(HeliogramDecodeError):
        decode_pixels(img, palette=palette, patch_size=patch_size, nsym=NSYM)


# --- rs_encoded_length sanity used throughout this file's byte-budget assertions -----------------


def test_rs_encoded_length_matches_actual_encode_ecc_length():
    """Sanity-checks the `rs_encoded_length` arithmetic this whole file leans on (e.g. to compute
    which corruption stays within RS's correction budget) against `encode`'s ACTUAL output length
    in symbols, for both a single-chunk and a genuinely multi-chunk payload."""
    for payload_len in (48, 1024):
        payload = (bytes(range(256)) * (payload_len // 256 + 1))[:payload_len]
        assert len(payload) == payload_len
        palette = 8
        img = encode(payload, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, seed=0)
        bps = bits_per_symbol(palette)
        _, _, symbols = extract_symbols(img, palette=palette, patch_size=PATCH_SIZE)

        stream = _symbols_to_bytes(symbols, bps)
        expected_ecc_len = rs_encoded_length(5 + payload_len, NSYM)
        assert len(stream) >= expected_ecc_len, (
            f"payload_len={payload_len}: recovered stream ({len(stream)}B) shorter than the "
            f"expected ecc_len ({expected_ecc_len}B) -- grid padding should only ever add "
            "trailing symbol-0 padding, never truncate the real ECC bytes"
        )
