"""Pytest suite for heliogram codec v0.1: clean roundtrip exactness, determinism, and calibration
recovery under a mild global color shift. Assert-based, no fixtures.

Also covers the capacity-ceiling additions: palettes 32/64/128/256 (exercised automatically since
the palette-driven tests below iterate VALID_PALETTES, which now includes all of them) and
sub-patch packing (`subpatch`/k, explicit cases only -- see the `subpatch` section at the
bottom). All numbers in this file come from `decode_pixels`, the model-free reference decoder
(see heliogram/codec.py's module docstring) -- nothing here is a VLM/Phase-2 result.

Also covers real (not synthetic-pixel-shift) corruption via heliogram.corruption -- see the
"realistic corruption coverage" section near the bottom, which pins measured decode
success/failure for palette 32/64 (subpatch=1) and every SUBPATCH_PALETTES entry (subpatch=2)
under actual resize/JPEG/crop round-trips, mirroring (a subset of) heliogram.harness.CORRUPTIONS.

Palettes 128/256 (added alongside 32/64 to VALID_PALETTES) get the same clean-roundtrip,
determinism, mild-color-shift, and distinctness coverage as every other palette automatically
(again via the VALID_PALETTES-driven tests), plus several dedicated tests near the bottom: an
explicit clean exact-roundtrip check at every `subpatch` value, a DATA HONESTY-mandated pin of
their MEASURED failure under realistic JPEG q70 compression on this reference pixel decoder at
subpatch=1 (256-way nearest-neighbor classification does not survive chroma subsampling) --
mirroring the existing palette=32/64 known-failure test below -- and, closing what was originally
a disclosed gap, a subpatch=2 x real-corruption suite (mild-corruption success at small
payloads, a known jpeg_q95-at-scale failure specific to palette=256, and a known "combined"-
corruption failure at small payloads) in the "realistic corruption coverage" section. See
spec/format-v0.1.md section 2a for the full writeup: P=128/256 clean-decode exactly but are not
claimed to survive corruption on this decoder, and are not a VLM capability claim either way. A
separate test also pins get_palette(P) for P in {2,4,8,16,32,64} as byte-identical to the
pre-128/256-extension implementation.
"""

import hashlib
import io
import json

import numpy as np
from PIL import Image

from heliogram import decode_pixels, encode, get_palette
from heliogram.codec import HeliogramDecodeError, VALID_PALETTES, VALID_SUBPATCHES
from heliogram.corruption import compose, crop_pad, jpeg_compress, resize_roundtrip
from heliogram.dataset import random_payload

PAYLOADS = [
    b"",
    b"a",
    b"hello, heliogram!",
    json.dumps(
        {"id": 42, "name": "heliogram", "tags": ["vlm", "codec"], "active": True}
    ).encode("utf-8"),
    bytes(range(256)) * 2,  # 512 bytes; exercises reedsolo's internal chunking
]

# Palettes explicitly named for the subpatch=2 coverage (unchanged by the 128/256 extension --
# see LARGE_PALETTES below for the dedicated, explicit 128/256 coverage instead of folding them
# in here, since 128/256 are NOT verified against the "combined"/jpeg_q70 corruptions this tuple
# also feeds into further down, only against the REALISTIC_MILD_CORRUPTIONS above).
SUBPATCH_PALETTES = (8, 16, 32, 64)

# Palettes added on top of 32/64 (get_palette's value-tiling scheme extended to 8 and 16 levels
# respectively -- see heliogram/codec.py's get_palette docstring and spec/format-v0.1.md section
# 2a). Covered automatically by every VALID_PALETTES-driven test above (clean roundtrip,
# determinism, mild color shift, distinctness, REALISTIC_MILD_CORRUPTIONS), plus the dedicated
# sections near the bottom of this file for an explicit multi-subpatch clean-roundtrip check and
# the DATA HONESTY-mandated JPEG q70 known-failure pin.
LARGE_PALETTES = (128, 256)

# Real (not synthetic-pixel-shift) corruptions, mirroring the mild end of
# heliogram.harness.CORRUPTIONS' "realistic serving pipeline" envelope (resize +-3-5%, JPEG
# q95/85, slight crop/pad). Empirically confirmed to give decode_success_rate=1.00 for every
# palette in VALID_PALETTES at the (small, <=512B) payload sizes this test file's own PAYLOADS
# list below exercises -- unlike jpeg_q70/"combined", which are NOT reliable at every (palette,
# payload_size) and are covered separately below by tests that pin the measured *failure* rather
# than assume success.
#
# CORRECTION (this comment previously claimed heliogram.harness's own sweep "has not been
# re-run to add 128/256 rows" and that this test file, not results.csv/RESULTS.md, was the
# source of truth for those two palettes -- that was true when first written but is stale as of
# this diff: results.csv/RESULTS.md now DO include a full 512-row sweep covering P=128/256, and
# should be cited for them like any other palette. That sweep also surfaces one exception this
# file's own small payloads do not: `palette=256` under `jpeg_q95` once payload_size >= 1024B --
# decode_success_rate drops to 0.6667 at subpatch=1/4096B and to 0.0000 at subpatch=2/
# {1024,4096,16384}B (every other palette/subpatch/payload_size cell in that same sweep, under
# this same REALISTIC_MILD_CORRUPTIONS set, is 1.0000). That failure is NOT hypothetical -- see
# test_subpatch2_palette256_known_failure_under_jpeg_q95_at_scale below, which pins it directly
# against this reference decoder rather than just citing results.csv, per this project's DATA
# HONESTY rule: a "mild" corruption failing for the largest palette at scale must be shown, not
# smoothed over by an overly broad "every palette, every payload size" claim.
REALISTIC_MILD_CORRUPTIONS = {
    "resize_3pct": lambda img: resize_roundtrip(img, scale=0.97),
    "resize_5pct": lambda img: resize_roundtrip(img, scale=0.95),
    "jpeg_q95": lambda img: jpeg_compress(img, quality=95),
    "crop_pad_2px": lambda img: crop_pad(img, dx=2, dy=2),
}


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_clean_roundtrip_exact():
    for palette in VALID_PALETTES:
        for payload in PAYLOADS:
            img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
            recovered = decode_pixels(img, palette=palette, patch_size=14, nsym=32)
            assert recovered == payload, (
                f"roundtrip mismatch for palette={palette} payload_len={len(payload)}"
            )


def test_determinism_same_args_identical_png_bytes():
    payload = b"determinism check: heliogram v0.1"
    for palette in VALID_PALETTES:
        img1 = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        img2 = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        assert _png_bytes(img1) == _png_bytes(img2)


def test_determinism_different_payload_differs():
    img1 = encode(b"payload one", palette=8)
    img2 = encode(b"payload two", palette=8)
    assert _png_bytes(img1) != _png_bytes(img2)


def test_calibration_recovery_under_mild_global_color_shift():
    shift = np.array([6, -4, 5], dtype=np.int16)
    for palette in VALID_PALETTES:
        payload = f"calibration test palette={palette}".encode("utf-8")
        img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        arr = np.asarray(img, dtype=np.int16)
        shifted = np.clip(arr + shift, 0, 255).astype(np.uint8)
        shifted_img = Image.fromarray(shifted)
        recovered = decode_pixels(shifted_img, palette=palette, patch_size=14, nsym=32)
        assert recovered == payload, f"calibration recovery failed for palette={palette}"


def test_palette_deterministic_and_separable():
    for palette in VALID_PALETTES:
        colors_a = get_palette(palette)
        colors_b = get_palette(palette)
        assert colors_a == colors_b
        assert len(set(colors_a)) == palette  # all colors distinct -- incl. get_palette(256)


def test_get_palette_le64_byte_identical_to_pinned_values():
    """CRITICAL regression guard for the 128/256 extension: get_palette(P) for every P already
    in VALID_PALETTES before 128/256 were added (2, 4, 8, 16, 32, 64) must stay byte-for-byte
    identical. get_palette's P>16 branch reuses the exact same code path for every P>16 value --
    only `levels = palette // 16` grows (2/4 for 32/64, unchanged; 8/16 for the new 128/256) --
    so this test exists to catch a future edit to that shared branch that accidentally changes
    the P<=64 outputs too, not just the new P=128/256 ones.

    These SHA-256 digests were captured directly from `get_palette()` at git HEAD immediately
    before the 128/256 extension (commit 455892b, "feat: raise channel ceiling (palettes 32/64 +
    sub-patch), gate sweep, GPU scaffold") -- extracted via `git show 455892b:heliogram/codec.py`
    into a standalone module and diffed object-for-object against the current implementation's
    output (all six matched exactly) before being hashed -- not re-derived from the current
    implementation alone. The digest is over `json.dumps(get_palette(p))`, an unambiguous
    canonical serialization of the RGB-triple list (order-sensitive, so this also pins color
    order, not just the set of colors).
    """
    expected_sha256 = {
        2: "554980a7a90ce75063a9578526f2a8c895e859c16a221f7d914fe328961cd50a",
        4: "cecdc5540d0779dad9c2d8f8bac11fb96e5770ceb6c2e4139cb8163f4321aac6",
        8: "7522d4c34a53cba0b566648919ef9104bab6bd0594d226b47406afd086e7bcd0",
        16: "f3c5ba052b3f0dd0396f85049338faba0979837bb225ef5b9e1eaf8faae528ae",
        32: "e72627bde60377d53d412bb635363cb0ec30e66c9340f307c2911e26ab554bff",
        64: "ba5ce97d34a323dfd867a7a25e68925aea106f13bf9c1278525f9f345ef77805",
    }
    assert tuple(expected_sha256.keys()) == (2, 4, 8, 16, 32, 64)
    for palette, digest in expected_sha256.items():
        canon = json.dumps(get_palette(palette))
        got = hashlib.sha256(canon.encode("utf-8")).hexdigest()
        assert got == digest, (
            f"get_palette({palette}) changed vs. the pre-128/256-extension pinned value -- "
            "byte-identity guarantee broken by the 128/256 extension"
        )


# --- subpatch (k>1 sub-patch packing) -------------------------------------------------------
#
# These are geometric/decode_pixels-only checks: subpatch>1 raises how many symbols a pixel
# decoder can read out of a patch, not what a real ViT-patch VLM encoder could resolve (see
# heliogram/codec.py's module docstring and spec/format-v0.1.md section 6a). No claim here
# about model capability, only about the reference decoder's exact roundtrip.


def test_subpatch2_clean_roundtrip_exact():
    for palette in SUBPATCH_PALETTES:
        for payload in PAYLOADS:
            img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=2)
            recovered = decode_pixels(img, palette=palette, patch_size=14, nsym=32, subpatch=2)
            assert recovered == payload, (
                f"subpatch=2 roundtrip mismatch for palette={palette} payload_len={len(payload)}"
            )


def test_subpatch2_determinism_same_args_identical_png_bytes():
    payload = b"determinism check: heliogram subpatch=2"
    for palette in SUBPATCH_PALETTES:
        img1 = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=2)
        img2 = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=2)
        assert _png_bytes(img1) == _png_bytes(img2)


def test_subpatch1_output_unchanged_pinned_hash():
    """Guard back-compat: subpatch=1 (the default) must reproduce the exact pre-subpatch v0.1
    byte stream. These SHA-256 digests were captured from `encode()` at the commit immediately
    before sub-patch packing was introduced (git HEAD at the time this test was added) and
    independently re-verified byte-for-byte against that same pre-change implementation -- they
    pin the real historical output, not just whatever the current code happens to produce.
    """
    payload = b"heliogram v0.1 subpatch=1 back-compat guard"
    expected_sha256 = {
        8: "7683a171f5bc500ebd059f13abed3d34c0ab7059c5a5031db04ff7e47606cea2",
        16: "a5420a89d9c35bb2376da2cb81645a5ce6396a4c243a3afa067506bd29e69b34",
    }
    for palette, digest in expected_sha256.items():
        img_default = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        img_explicit = encode(
            payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=1
        )
        png_default = _png_bytes(img_default)
        png_explicit = _png_bytes(img_explicit)
        assert png_default == png_explicit, (
            f"default subpatch differs from explicit subpatch=1 for palette={palette}"
        )
        assert hashlib.sha256(png_default).hexdigest() == digest, (
            f"subpatch=1 PNG bytes changed for palette={palette} -- back-compat break"
        )


def test_subpatch_invalid_values_raise_value_error():
    assert VALID_SUBPATCHES == (1, 2)
    try:
        encode(b"x", palette=8, patch_size=14, subpatch=3)
        assert False, "expected ValueError for subpatch=3"
    except ValueError:
        pass
    try:
        encode(b"x", palette=8, patch_size=15, subpatch=2)  # 15 % 2 != 0
        assert False, "expected ValueError for patch_size not divisible by subpatch"
    except ValueError:
        pass


# --- large palettes (128, 256) --------------------------------------------------------------
#
# 128/256 already get clean-roundtrip/determinism/color-shift/distinctness coverage for free at
# subpatch=1 via the VALID_PALETTES-driven tests above (test_clean_roundtrip_exact etc. all
# iterate VALID_PALETTES, which now includes them). The test below makes that explicit and adds
# subpatch=2 *clean* coverage too (SUBPATCH_PALETTES intentionally does NOT include 128/256 --
# see its definition near the top of this file -- this dedicated section covers them instead).
#
# CORRECTION: this section previously stopped at clean-roundtrip only and said "only the
# clean-roundtrip claim, not the realistic-corruption-survival claim, is being made for these
# two palettes at subpatch=2" -- that was a real, disclosed coverage gap (LARGE_PALETTES x
# subpatch=2 x any corruption was completely untested), not a permanent scoping decision. The
# "realistic corruption coverage" section near the bottom of this file now closes it: see
# test_subpatch2_large_palette_roundtrip_under_realistic_mild_corruption_small_payload,
# test_subpatch2_palette256_known_failure_under_jpeg_q95_at_scale, and
# test_subpatch2_large_palette_known_failure_under_combined_corruption_small_payload.


def test_large_palette_clean_roundtrip_exact_all_subpatches():
    """Explicit clean exact-roundtrip coverage for palette in LARGE_PALETTES (128, 256) at every
    VALID_SUBPATCHES value, mirroring test_subpatch2_clean_roundtrip_exact's shape but scoped to
    the two palettes this extension adds. get_palette(128)/get_palette(256) each still produce
    `palette` pairwise-distinct colors (test_palette_deterministic_and_separable already checks
    this), so on a clean, uncorrupted image nearest-neighbor classification is exact regardless
    of how wide the calibration row (compute_grid enforces width >= palette, so >=128 or >=256
    patches wide) or how many symbols get packed per data patch (subpatch**2) has to be."""
    for palette in LARGE_PALETTES:
        for subpatch in VALID_SUBPATCHES:
            for payload in PAYLOADS:
                img = encode(
                    payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=subpatch
                )
                recovered = decode_pixels(
                    img, palette=palette, patch_size=14, nsym=32, subpatch=subpatch
                )
                assert recovered == payload, (
                    f"clean roundtrip mismatch for palette={palette} subpatch={subpatch} "
                    f"payload_len={len(payload)}"
                )


def test_large_palette_determinism_same_args_identical_png_bytes():
    payload = b"determinism check: heliogram large-palette extension"
    for palette in LARGE_PALETTES:
        for subpatch in VALID_SUBPATCHES:
            img1 = encode(
                payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=subpatch
            )
            img2 = encode(
                payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=subpatch
            )
            assert _png_bytes(img1) == _png_bytes(img2)


# --- realistic corruption coverage (review-fix) --------------------------------------------
#
# Everything above either used a clean image or a synthetic +-6/-4/+5 RGB shift
# (test_calibration_recovery_under_mild_global_color_shift) -- never a real resize/JPEG/crop
# round-trip, and subpatch=2 had NO corruption coverage of any kind. That was an adequate proxy
# for palettes 2/4/8/16 (they survive the full realistic envelope in results.csv/RESULTS.md
# almost everywhere), but it stopped being one once VALID_PALETTES grew to include 32/64 and
# subpatch grew to include 2: the harness's own measured sweep shows both of those genuinely
# fail decode under corruptions it classifies as "realistic" (not just the STRESS_CORRUPTIONS
# diagnostic suite), at specific (payload_size, corruption) combinations. The tests below use
# heliogram.corruption's real primitives (not pixel-shift synthesis) and, per this project's
# DATA HONESTY rule, pin the TRUE measured outcome in each direction: success where the harness
# shows success, and failure where it shows failure -- see each test's docstring for the exact
# results.csv row it reproduces.


def test_realistic_mild_corruption_roundtrip_all_palettes_subpatch1():
    """Closes the gap: no prior test applied a real resize/JPEG/crop corruption to ANY palette
    (only a synthetic global color shift). For palette<=64, empirically verified against
    results.csv: every such palette (including 32/64) decodes successfully under this
    mild-realistic corruption set at every payload size the harness sweep covers -- this is the
    regime where the old clean+shift methodology's conclusion ("robust") actually still holds
    for 32/64, so pin it explicitly instead of leaving it untested. Since this test iterates
    VALID_PALETTES directly, it now runs 128/256 through the same mild-corruption set too, at the
    (small, <=512B) payload sizes in PAYLOADS -- confirmed passing by this test's own execution,
    consistent with results.csv/RESULTS.md's own now-current 512-row sweep (which DOES include
    P=128/256; a stale version of this docstring once claimed otherwise). That harness sweep
    tests larger payloads than PAYLOADS reaches and, at those larger sizes, measures one
    exception this test does NOT exercise: `palette=256` under `jpeg_q95` once payload_size >=
    1024B (decode_success_rate 0.6667 at subpatch=1/4096B, 0.0000 at subpatch=2/{1024,4096,
    16384}B) -- see test_subpatch2_palette256_known_failure_under_jpeg_q95_at_scale below, which
    pins that failure directly rather than leaving it as a citation."""
    for palette in VALID_PALETTES:
        for payload in PAYLOADS:
            img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
            for name, corrupt in REALISTIC_MILD_CORRUPTIONS.items():
                corrupted = corrupt(img)
                recovered = decode_pixels(corrupted, palette=palette, patch_size=14, nsym=32)
                assert recovered == payload, (
                    f"subpatch=1 decode failed under realistic corruption={name} for "
                    f"palette={palette} payload_len={len(payload)}"
                )


def test_palette32_64_known_failure_under_jpeg_q70_at_scale():
    """DATA HONESTY: pins a MEASURED failure, not a hypothetical one -- see results.csv rows
    'palette=32,subpatch=1,payload_size=1024,corruption=jpeg_q70' and
    'palette=64,subpatch=1,payload_size=1024,corruption=jpeg_q70', both decode_success_rate=
    0.0000. This is the exact gap Finding #2 flagged: extending VALID_PALETTES to 32/64 without
    ANY real-corruption test meant a future change to get_palette's P>16 value-tiling scheme
    (v_min floor, hues=16 split) could silently make this better or worse and no assertion would
    move. Uses heliogram.dataset.random_payload -- the same deterministic construction
    heliogram.harness._random_payload uses for trial seed=0 -- so this reproduces the harness's
    own measured cell exactly, at the same payload_size=1024 the finding cites.

    If this test ever starts failing because decode now SUCCEEDS, that is a genuine robustness
    improvement to get_palette/decode_pixels for P>16 -- update this test (and re-run
    heliogram.harness to refresh RESULTS.md/results.csv) rather than just deleting the
    assertion; do not silently paper over a real behavior change either way.
    """
    payload = random_payload(0, 1024)
    for palette in (32, 64):
        img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        corrupted = jpeg_compress(img, quality=70)
        try:
            recovered = decode_pixels(corrupted, palette=palette, patch_size=14, nsym=32)
        except HeliogramDecodeError:
            continue  # expected: RS could not correct -- the measured failure mode
        assert recovered != payload, (
            f"palette={palette} subpatch=1 unexpectedly recovered the exact payload under "
            "jpeg_q70 at payload_size=1024 -- this contradicts the measured "
            "decode_success_rate=0.0000 in results.csv for this exact cell; if genuine, "
            "update RESULTS.md/results.csv (re-run heliogram.harness) and this test together"
        )


def test_palette128_256_known_failure_under_jpeg_q70_at_scale():
    """DATA HONESTY (PROJECT INVARIANT): mirrors test_palette32_64_known_failure_under_jpeg_q70_
    at_scale for the two palettes this extension adds. Confirmed by direct execution against
    this reference decoder, and consistent with results.csv/RESULTS.md's own now-current 512-row
    sweep (which DOES include P=128/256 rows -- a stale version of this docstring once claimed
    the sweep had not been re-run and treated this test as the sole source of truth for these
    cells; it has been, and results.csv confirms the identical result independently: palette=128
    and palette=256, subpatch=1, payload_size=1024, corruption=jpeg_q70 both show
    decode_success_rate=0.0000): at payload_size=1024, palette in {128, 256}, subpatch=1, JPEG
    q70 corruption, decode_pixels does NOT recover the original payload for either palette (both
    raise HeliogramDecodeError in practice -- 256-way nearest-neighbor classification cannot
    separate that many colors once JPEG's chroma subsampling erodes hue differences). This is
    the exact failure the project's benefit claim is conditional on a learned (VLM) reader
    fixing -- it must be demonstrated failing here, not hidden, per this project's DATA HONESTY
    invariant. See spec/format-v0.1.md section 2a.

    If this test ever starts failing because decode now SUCCEEDS, that is a genuine robustness
    improvement to get_palette/decode_pixels for P>64 -- update this test (and re-run
    heliogram.harness to refresh RESULTS.md/results.csv) rather than just deleting the
    assertion; do not silently paper over a real behavior change either way. This test does
    NOT assert 128/256 survive JPEG -- it pins that they currently do not, by design.
    """
    payload = random_payload(0, 1024)
    for palette in LARGE_PALETTES:
        img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0)
        corrupted = jpeg_compress(img, quality=70)
        try:
            recovered = decode_pixels(corrupted, palette=palette, patch_size=14, nsym=32)
        except HeliogramDecodeError:
            continue  # expected: RS could not correct -- the measured failure mode
        assert recovered != payload, (
            f"palette={palette} subpatch=1 unexpectedly recovered the exact payload under "
            "jpeg_q70 at payload_size=1024 -- this contradicts the measured failure for this "
            "config; if genuine, update this test (and consider re-running heliogram.harness to "
            "add 128/256 rows to RESULTS.md/results.csv) rather than silently deleting the check"
        )


def test_subpatch2_roundtrip_under_realistic_mild_corruption():
    """CRITICAL review-fix: prior to this test, subpatch=2 had zero coverage against ANY
    corruption -- test_subpatch2_clean_roundtrip_exact only ever used a clean image. This pins
    the mild end of the realistic corruption envelope (resize +-3-5%, JPEG q95, 2px crop/pad --
    plus jpeg_q85, separately confirmed reliable for subpatch=2 in results.csv/RESULTS.md at
    every tested payload size) as an exact-roundtrip regression guard: a future change to
    extract_symbols' sub-cell sampling geometry (`sub_half`, sub-cell layout order, etc.) that
    broke decoding here would now be caught, where previously it would not have been."""
    corruptions = dict(REALISTIC_MILD_CORRUPTIONS)
    corruptions["jpeg_q85"] = lambda img: jpeg_compress(img, quality=85)
    for palette in SUBPATCH_PALETTES:
        for payload in PAYLOADS:
            img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=2)
            for name, corrupt in corruptions.items():
                corrupted = corrupt(img)
                recovered = decode_pixels(
                    corrupted, palette=palette, patch_size=14, nsym=32, subpatch=2
                )
                assert recovered == payload, (
                    f"subpatch=2 decode failed under realistic corruption={name} for "
                    f"palette={palette} payload_len={len(payload)}"
                )


def test_subpatch2_known_failure_under_combined_corruption_small_payload():
    """DATA HONESTY: pins a MEASURED failure, not a hypothetical one -- see results.csv rows
    'palette={8,16,32,64},subpatch=2,payload_size=48,corruption=combined', ALL
    decode_success_rate=0.0000. 'combined' (resize 5% + JPEG q70 + 2px crop/pad in sequence) is
    part of heliogram.harness.CORRUPTIONS -- the "realistic serving pipeline" suite, not the
    STRESS_CORRUPTIONS diagnostic-only suite -- so this is a genuine, currently-measured
    limitation of subpatch=2 at this payload size, not a manufactured edge case. Uses
    heliogram.dataset.random_payload(0, 48) -- the same construction heliogram.harness uses for
    trial seed=0 at payload_size=48 -- to reproduce the harness's own measured cell exactly.

    This is a regression guard in the OTHER direction from most of this file: if it starts
    failing because decode now SUCCEEDS, that is a genuine improvement -- update this test (and
    re-run heliogram.harness to refresh RESULTS.md/results.csv) rather than silencing it; a
    future change to sub-cell sampling could just as easily make this (or a currently-passing
    subpatch=2 case) silently WORSE, which is exactly the coverage gap this test and
    test_subpatch2_roundtrip_under_realistic_mild_corruption together close.
    """
    payload = random_payload(0, 48)
    for palette in SUBPATCH_PALETTES:
        img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=2)
        corrupted = compose(
            img,
            [
                (resize_roundtrip, {"scale": 0.95}),
                (jpeg_compress, {"quality": 70}),
                (crop_pad, {"dx": 2, "dy": 2}),
            ],
        )
        try:
            recovered = decode_pixels(corrupted, palette=palette, patch_size=14, nsym=32, subpatch=2)
        except HeliogramDecodeError:
            continue  # expected: RS could not correct -- the measured failure mode
        assert recovered != payload, (
            f"palette={palette} subpatch=2 unexpectedly recovered the exact payload under "
            "'combined' corruption at payload_size=48 -- this contradicts the measured "
            "decode_success_rate=0.0000 in results.csv for this exact cell; if genuine, "
            "update RESULTS.md/results.csv (re-run heliogram.harness) and this test together"
        )


# --- large palettes (128, 256) at subpatch=2 under real corruption (review-fix) -------------
#
# Before the tests below, (palette in LARGE_PALETTES) x (subpatch=2) x (any real corruption) was
# completely untested: test_large_palette_clean_roundtrip_exact_all_subpatches only ever used a
# clean image, and SUBPATCH_PALETTES (feeding the two corruption tests above) deliberately
# excludes 128/256 (see its definition near the top of this file). That gap matters specifically
# here because subpatch=2 halves the sub-cell sampling area extract_symbols works with, and
# 128/256 are already the two palettes closest to nearest-neighbor classification's separability
# limit (spec/format-v0.1.md section 2a) -- exactly the intersection where a future change to
# extract_symbols' sub-cell geometry would be most likely to silently regress, and least likely
# to be caught by any other existing test. The three tests below close it, pinning the TRUE
# measured outcome in each direction (success and failure) the same way the rest of this
# "realistic corruption coverage" section does.


def test_subpatch2_large_palette_roundtrip_under_realistic_mild_corruption_small_payload():
    """CRITICAL review-fix: closes the (LARGE_PALETTES x subpatch=2 x any corruption) gap for
    the mild end of the realistic envelope. Uses PAYLOADS minus its largest (512-byte) entry --
    empirically, palette=256/subpatch=2/jpeg_q95 fails to decode that specific combination (see
    test_subpatch2_palette256_known_failure_under_jpeg_q95_at_scale below, which pins the same
    failure mode at a size heliogram.harness actually swept, 1024B, instead of this ad hoc
    512-byte literal); every other (palette, payload, corruption) combination among LARGE_
    PALETTES x PAYLOADS[:-1] x REALISTIC_MILD_CORRUPTIONS at subpatch=2 decodes exactly, verified
    by direct execution. This is a genuine, currently-measured split, not an assumption in either
    direction -- see the module docstring's "realistic corruption coverage" note."""
    for palette in LARGE_PALETTES:
        for payload in PAYLOADS[:-1]:
            img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=2)
            for name, corrupt in REALISTIC_MILD_CORRUPTIONS.items():
                corrupted = corrupt(img)
                recovered = decode_pixels(
                    corrupted, palette=palette, patch_size=14, nsym=32, subpatch=2
                )
                assert recovered == payload, (
                    f"subpatch=2 decode failed under realistic corruption={name} for "
                    f"palette={palette} payload_len={len(payload)}"
                )


def test_subpatch2_palette256_known_failure_under_jpeg_q95_at_scale():
    """DATA HONESTY (PROJECT INVARIANT): pins a MEASURED failure, not a hypothetical one -- see
    results.csv row 'palette=256,subpatch=2,payload_size=1024,corruption=jpeg_q95',
    decode_success_rate=0.0000 (and the same failure recurs at payload_size=4096/16384 in that
    same sweep). jpeg_q95 is the MILDEST corruption in REALISTIC_MILD_CORRUPTIONS above -- the
    tier every other palette/subpatch/payload_size combination in this project survives -- which
    is exactly why this is worth pinning loudly rather than folding quietly into the jpeg_q70
    known-failure tests above: palette=256 at subpatch=2 stops being "robust under mild
    corruption" once payload size (and therefore grid size) grows past 512B, well before jpeg_q70
    or "combined" ever enter the picture. Uses heliogram.dataset.random_payload(0, 1024) -- the
    same construction heliogram.harness._random_payload uses for trial seed=0 -- to reproduce
    the harness's own measured cell exactly.

    palette=128 at this exact cell (subpatch=2, payload_size=1024, jpeg_q95) decodes
    successfully (results.csv: decode_success_rate=1.0000) -- this test is deliberately scoped
    to palette=256 only, not LARGE_PALETTES, so it does not assert a failure that would not
    actually occur for 128.

    If this test ever starts failing because decode now SUCCEEDS, that is a genuine robustness
    improvement to get_palette/decode_pixels/extract_symbols for P=256 at subpatch=2 -- update
    this test (and re-run heliogram.harness to refresh RESULTS.md/results.csv) rather than just
    deleting the assertion; do not silently paper over a real behavior change either way.
    """
    payload = random_payload(0, 1024)
    img = encode(payload, palette=256, patch_size=14, nsym=32, seed=0, subpatch=2)
    corrupted = jpeg_compress(img, quality=95)
    try:
        recovered = decode_pixels(corrupted, palette=256, patch_size=14, nsym=32, subpatch=2)
    except HeliogramDecodeError:
        return  # expected: RS could not correct -- the measured failure mode
    assert recovered != payload, (
        "palette=256 subpatch=2 unexpectedly recovered the exact payload under jpeg_q95 at "
        "payload_size=1024 -- this contradicts the measured decode_success_rate=0.0000 in "
        "results.csv for this exact cell; if genuine, update RESULTS.md/results.csv (re-run "
        "heliogram.harness) and this test together"
    )


def test_subpatch2_large_palette_known_failure_under_combined_corruption_small_payload():
    """DATA HONESTY: pins a MEASURED failure, not a hypothetical one -- see results.csv rows
    'palette={128,256},subpatch=2,payload_size=48,corruption=combined', both
    decode_success_rate=0.0000 -- mirrors
    test_subpatch2_known_failure_under_combined_corruption_small_payload above (which covers
    SUBPATCH_PALETTES) extended to LARGE_PALETTES, closing the same coverage gap for the two
    largest palettes. Uses heliogram.dataset.random_payload(0, 48) -- the same construction
    heliogram.harness uses for trial seed=0 at payload_size=48 -- to reproduce the harness's own
    measured cell exactly.

    This is a regression guard in the OTHER direction from the two tests above: if it starts
    failing because decode now SUCCEEDS, that is a genuine improvement -- update this test (and
    re-run heliogram.harness to refresh RESULTS.md/results.csv) rather than silencing it.
    """
    payload = random_payload(0, 48)
    for palette in LARGE_PALETTES:
        img = encode(payload, palette=palette, patch_size=14, nsym=32, seed=0, subpatch=2)
        corrupted = compose(
            img,
            [
                (resize_roundtrip, {"scale": 0.95}),
                (jpeg_compress, {"quality": 70}),
                (crop_pad, {"dx": 2, "dy": 2}),
            ],
        )
        try:
            recovered = decode_pixels(corrupted, palette=palette, patch_size=14, nsym=32, subpatch=2)
        except HeliogramDecodeError:
            continue  # expected: RS could not correct -- the measured failure mode
        assert recovered != payload, (
            f"palette={palette} subpatch=2 unexpectedly recovered the exact payload under "
            "'combined' corruption at payload_size=48 -- this contradicts the measured "
            "decode_success_rate=0.0000 in results.csv for this exact cell; if genuine, "
            "update RESULTS.md/results.csv (re-run heliogram.harness) and this test together"
        )
