"""heliogram.harness -- CPU-only evaluation harness for the heliogram codec (D3 + Slice B).

For every (palette, subpatch, payload_size) combination in the capacity sweep, and every
corruption in CORRUPTIONS (plus a `clean` no-op), encodes N synthetic random payloads, decodes
them back through decode_pixels (the reference, model-free decoder), and measures:

  symbol_error_rate    fraction of DATA sub-cells whose classified symbol differs from the
                       ground-truth symbol written at encode time (ground truth is read straight
                       off the clean, uncorrupted image, which is exact by construction).
  decode_success_rate  fraction of trials where decode_pixels ran to completion AND returned the
                       exact original payload.
  bits_per_patch       TRUE PAYLOAD DENSITY: payload_bytes*8 / total_patches, i.e. the actual
                       number of PAYLOAD bits (not ecc bits, not raw symbol-channel bits) carried
                       by the WHOLE grid image (calibration row included), counted only on a
                       successful decode (0 contribution otherwise). Earlier versions of this
                       harness computed subpatch^2 * bits_per_symbol * (data_patches/
                       total_patches) * (payload_bytes/ecc_bytes) instead -- that formula credited
                       every DATA patch with carrying a full symbol drawn from the ecc bitstream,
                       which ignores the zero-bit padding compute_grid and _pack_symbols both
                       introduce (grid width is floored at the palette size regardless of how few
                       symbols are needed, and the ecc bitstream is zero-padded to a whole number
                       of symbols), so it overstated true payload density by up to ~3x at small
                       payloads (see _bits_per_patch_on_success's docstring and
                       tests/test_harness.py for hand-computed anchors). This field/function name
                       is unchanged; only the formula is fixed.
  total_patches /      THE BENEFIT METRIC (see _token_crossover): total_patches is the grid's
  base64_token_est /   width*height (~1 self-hosted-VLM token/patch); base64_token_est is
  token_ratio /        ceil(payload/3)*4 base64 chars scaled by BASE64_CHARS_PER_TOKEN (measured
  heliogram_cheaper    chars/token when a tokenizer baseline exists, else the analytic ~1
                       char/token -- see _base64_token_estimate); token_ratio =
                       total_patches/base64_token_est. token_ratio < 1.0 means encoding this
                       payload as a heliogram grid costs FEWER tokens than base64-in-text-context
                       for the SAME bytes -- an accounting fact about token COUNT, independent of
                       (and computed whether or not this cell's decode_success_rate is 1.0 -- see
                       the mandatory caveat below and in RESULTS.md's "Token crossover" section).
  lm_tokens_2x2 /      An ALTERNATIVE token accounting for the pinned Phase-2 target
  lm_token_ratio       (Qwen2.5-VL): that model's vision-language connector merges each 2x2 block
                       of ViT patches into ONE LM-visible token (a "spatial merger"), so the
                       LM-visible token cost of a heliogram grid is ceil(width/2)*ceil(height/2)
                       (lm_tokens_2x2), roughly 4x fewer than total_patches, not total_patches
                       itself. lm_token_ratio = lm_tokens_2x2/base64_token_est is the same
                       accounting comparison as token_ratio but against this merged-token count.
                       MANDATORY caveat: this accounting implicitly assumes the model can read
                       4*k^2*log2(P) bits out of ONE merged embedding (k=subpatch symbols per ViT
                       patch, 4 ViT patches per merged LM token) -- whether it actually can is
                       UNVERIFIED and Phase 2 work, the same epistemic class as the subpatch>1
                       caveat below. The per-patch accounting (total_patches/token_ratio) remains
                       the conservative headline number until Phase 2 measures this.

The sweep covers every palette in heliogram.codec.VALID_PALETTES, currently (2, 4, 8, 16, 32, 64,
128, 256) -- `PALETTES = VALID_PALETTES` below always tracks that tuple, so adding a palette to
the codec adds it to this sweep with no change needed here. `subpatch` (k, geometric
sub-cells/patch) and `payload_size` (amortization of fixed per-message overhead) are swept
dimensions alongside `palette`; a headline "Gate #1" section flags which (palette, subpatch,
payload_size) configs clear a fixed bits/patch bar both on a clean image and under worst-case
tested corruption. RESULTS.md's Headline section also reports two more bars this project
actually cares about more than Gate #1's arbitrary margin: Bar A (beat the ~6 bits/patch base64
density baseline, clean) and the token-crossover verdict above -- see that section for why Gate
#1 is a conservative comfort margin, not the real economic bar. See write_results_md's
"Headline" section for the mandatory subpatch>1 honesty caveat: decode_pixels reads sub-cells
trivially because it samples known exact pixel coordinates -- that a real VLM's vision encoder
could do the same is UNVERIFIED and is Phase 2 work, not a capability claim made here. The same
caveat applies to token_ratio/heliogram_cheaper: token-cheaper is not the same as decodable, and
the largest palettes that actually cross the token bar (128, 256) are measured elsewhere in this
same sweep to FAIL decode under jpeg_q70 -- see the "Token crossover" section, which shows both
facts in the same table on purpose.

Run as `python -m heliogram.harness`. Prints the headline gate table and writes results.csv +
RESULTS.md into the current working directory.
"""

from __future__ import annotations

import csv
import importlib.metadata
import math
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import baselines as _baselines_module
from .baselines import base64_bits_per_token, rendered_text_density
from .codec import (
    HeliogramDecodeError,
    PATCH_SIZE,
    VALID_PALETTES,
    VALID_SUBPATCHES,
    bits_per_symbol,
    compute_grid,
    decode_pixels,
    encode,
    extract_symbols,
    rs_encoded_length,
)
from .corruption import (
    QWEN_DEFAULT_MAX_PIXELS,
    QWEN_GENEROUS_MAX_PIXELS,
    compose,
    crop_pad,
    jpeg_compress,
    qwen_smart_resize,
    resize_roundtrip,
)

PALETTES = VALID_PALETTES  # currently (2,4,8,16,32,64,128,256) -- tracks codec.VALID_PALETTES
NSYM = 32
N_TRIALS = 5
PAYLOAD_SIZE = 48  # bytes per synthetic trial payload -- module default / single-cell fallback

# --- Slice B: capacity + amortization + GATE sweep dimensions --------------------------------
SUBPATCHES = VALID_SUBPATCHES              # sweep dimension: k in {1, 2} sub-cells/data-patch
SWEEP_PAYLOAD_SIZES = (48, 1024, 4096, 16384)  # sweep dimension: bytes/trial payload (amortization)
# The full sweep (palette x subpatch x payload_size x corruption) encodes/corrupts/decodes
# multi-thousand-patch images at the 16384-byte tier; N_TRIALS=5 there is expensive, so the
# sweep specifically (not the module default, and not the diagnostic STRESS suite below, which
# keeps running at N_TRIALS) uses a reduced trial count to bound wall-clock. Documented again in
# RESULTS.md's "Wall-clock note".
SWEEP_N_TRIALS = 3
# Gate #1 bar for the headline sweep section: a config "clears the gate" only if bits/patch is
# at or above this value BOTH clean and under its single worst tested corruption. This is a
# deliberate COMFORT MARGIN above the real economic bar (BASE64_BITS_PER_TOKEN, below) -- not
# the break-even point itself. See RESULTS.md's "Headline" section: Bar A (beat base64 density)
# and the token-crossover verdict are the bars this project's benefit claim actually rests on;
# Gate #1 exists as a stricter, deliberately-padded bar for deciding when to START Phase 2 work.
GATE_BITS_PER_PATCH = 8.0


def _resolve_base64_baseline() -> object:
    """Resolve the base64-token baseline BASE64_BITS_PER_TOKEN (and every Bar A verdict derived
    from it, see below) is computed against: prefer a MEASURED baseline
    (heliogram.baselines.load_measured_baseline(), added to heliogram/baselines.py by a parallel
    work group and NOT present in every checkout -- in particular not in the worktree this
    function was written in) over the analytic base64_bits_per_token() 6.0-bits/token assumption,
    whenever a real measurement is actually available.

    Imported defensively via getattr(...), never `from .baselines import load_measured_baseline`:
    a hard import would raise ImportError at module load time in any checkout that lacks the
    function yet, breaking every caller of heliogram.harness, not just this one baseline lookup.
    Two independent ways this falls through to the analytic baseline: (a) load_measured_baseline
    doesn't exist at all on heliogram.baselines (getattr returns None), or (b) it exists but
    returns None itself (its own docstring: no measurement file found on this machine yet).

    Returns an object exposing at minimum `.bits_per_token` (float) and `.note` (str). The
    measured path additionally exposes `.tokenizer_id` (str) and `.chars_per_token` (float); the
    analytic `Base64Baseline` fallback does not have a `.tokenizer_id` attribute at all, so
    callers that want to report WHICH baseline was actually used should branch on
    `getattr(result, "tokenizer_id", None) is not None`, not on the object's type.
    """
    loader = getattr(_baselines_module, "load_measured_baseline", None)
    if loader is not None:
        measured = loader()
        if measured is not None:
            return measured
    return base64_bits_per_token()


# Bar A -- the REAL economic bar (see module docstring / RESULTS.md "Headline"): does clean
# bits/patch beat plain base64-in-text-context density? Derived from _resolve_base64_baseline()
# (a MEASURED tokenizer baseline when one is available, else the analytic 6.0-bits/token
# assumption -- see that function's docstring) so this number can never silently drift from
# whichever baseline RESULTS.md's "Baselines" section states each verdict was computed against.
# Resolved ONCE here, and BASE64_CHARS_PER_TOKEN below is read off the SAME resolved object, so
# Bar A (bits/token) and Bar C (token count via chars/token) can never rest on two different
# baselines within one run.
_RESOLVED_BASELINE = _resolve_base64_baseline()
BASE64_BITS_PER_TOKEN = _RESOLVED_BASELINE.bits_per_token
# Bar C's token-count side (see _base64_token_estimate): base64 CHARACTERS per token. 1.0 (the
# analytic ~1 char/token assumption) when no measured baseline exists; the measured value when
# one does (e.g. 1.3498 for Qwen2.5-VL's tokenizer, measured 2026-07 -- BPE multi-char merges
# give base64 text MORE than one char/token, which makes base64 CHEAPER than the analytic
# estimate and is therefore the direction ADVERSE to heliogram's Bar C claim). Module-level on
# purpose: tests pin either accounting explicitly by monkeypatching this value.
BASE64_CHARS_PER_TOKEN = getattr(_RESOLVED_BASELINE, "chars_per_token", 1.0)

CORRUPTIONS: Dict[str, Callable] = {
    "clean": lambda img: img,
    "resize_3pct": lambda img: resize_roundtrip(img, scale=0.97),
    "resize_5pct": lambda img: resize_roundtrip(img, scale=0.95),
    "jpeg_q95": lambda img: jpeg_compress(img, quality=95),
    "jpeg_q85": lambda img: jpeg_compress(img, quality=85),
    "jpeg_q70": lambda img: jpeg_compress(img, quality=70),
    "crop_pad_2px": lambda img: crop_pad(img, dx=2, dy=2),
    # The target model's OWN preprocessing (Qwen2/2.5-VL smart_resize) -- the one corruption an
    # in-scope operator cannot opt out of. Two variants, measured separately on purpose:
    #   qwen_smart_resize      operator-controlled pixel bounds (generous max_pixels, as
    #                          scripts/run_probe.py sets): what remains is the UNAVOIDABLE 28px
    #                          (patch*merge) dimension snap. Identity for even-patch-grid images
    #                          under ~16M px; resamples any grid with an odd patch dimension.
    #   qwen_smart_resize_1mp  the STOCK processor defaults (max_pixels=1,003,520): a naive
    #                          operator's pipeline additionally downscales any grid over ~1MP
    #                          wholesale. This is what "just feed it to the processor" does.
    # These are the DOCUMENTED same-size-contract exception (see heliogram.corruption's module
    # docstring): the corrupted image may have a different size -- _run_cell measures it at that
    # size, exactly as the model would see it, rather than resizing back (which would erase the
    # corruption being measured).
    "qwen_smart_resize": lambda img: qwen_smart_resize(
        img, max_pixels=QWEN_GENEROUS_MAX_PIXELS
    ),
    "qwen_smart_resize_1mp": lambda img: qwen_smart_resize(
        img, max_pixels=QWEN_DEFAULT_MAX_PIXELS
    ),
    # "combined": the composed worst-case suite the README's "Corrupted" column describes --
    # resize 5%, JPEG q70, and a 2px crop/pad applied in sequence to the SAME image.
    "combined": lambda img: compose(
        img,
        [
            (resize_roundtrip, {"scale": 0.95}),
            (jpeg_compress, {"quality": 70}),
            (crop_pad, {"dx": 2, "dy": 2}),
        ],
    ),
}

# Diagnostic-only corruptions, well outside the "realistic serving pipeline" envelope the
# CORRUPTIONS suite above targets (resize +-1-5%, JPEG q70-95, slight crop/pad). These exist
# solely to prove decode_pixels' success rate on the realistic suite is a real headroom result
# and not a harness bug that can never observe failure -- see the "Beyond the realistic
# envelope" section of RESULTS.md. Run at a single representative config (subpatch=1,
# payload_size=PAYLOAD_SIZE), not swept across the capacity grid -- see main()/RESULTS.md.
STRESS_CORRUPTIONS: Dict[str, Callable] = {
    "stress_resize_50pct": lambda img: resize_roundtrip(img, scale=0.5),
    "stress_jpeg_q10": lambda img: jpeg_compress(img, quality=10),
    "stress_crop_pad_6px": lambda img: crop_pad(img, dx=6, dy=6),
    "stress_combined": lambda img: compose(
        img,
        [
            (resize_roundtrip, {"scale": 0.5}),
            (jpeg_compress, {"quality": 10}),
            (crop_pad, {"dx": 6, "dy": 6}),
        ],
    ),
}


@dataclass
class CellResult:
    palette: int
    subpatch: int
    payload_size: int
    corruption: str
    bits_per_symbol: int
    symbol_error_rate: float
    decode_success_rate: float
    bits_per_patch: float
    trials: int
    # Token-crossover fields (the benefit metric, see _token_crossover): a property of
    # (palette, subpatch, payload_size) alone, identical across every corruption in the same
    # bucket -- NOT reduced by decode_success_rate the way bits_per_patch is (see the mandatory
    # honesty caveat in _token_crossover's docstring: token-cheaper != decodable). Defaulted so
    # existing call sites/tests that construct CellResult without them keep working unchanged.
    total_patches: int = 0
    base64_token_est: int = 0
    token_ratio: float = 0.0
    heliogram_cheaper: bool = False
    # LM-token (2x2 spatial merger) accounting fields -- see _token_crossover/_grid_stats'
    # docstrings and the module docstring's `lm_tokens_2x2 / lm_token_ratio` entry. Same
    # per-(palette, subpatch, payload_size)-bucket, corruption-independent property as the
    # per-patch fields above; also defaulted for the same backward-compatibility reason.
    lm_tokens_2x2: int = 0
    lm_token_ratio: float = 0.0


def _random_payload(seed: int, size: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


@dataclass
class _GridStats:
    """Grid dimensions encode() would produce for a given (payload_len, palette, nsym, subpatch)
    -- shared by _bits_per_patch_on_success (density) and _token_crossover (patch-vs-token
    accounting) so the two metrics can never desync on the underlying grid math, unlike having
    each recompute it separately.

    `lm_tokens_2x2` is a SECOND token accounting, alongside `total_patches`: the pinned Phase-2
    target (Qwen2.5-VL) merges every 2x2 block of ViT patches into one LM-visible token (a
    "spatial merger"), so the LM-visible cost of a `width x height` patch grid is
    `ceil(width/2) * ceil(height/2)`, not `width*height` -- roughly 4x fewer LM tokens for the
    same image. See the module docstring's `lm_tokens_2x2 / lm_token_ratio` entry and
    `_token_crossover`'s docstring for the MANDATORY caveat this accounting carries."""

    ecc_len: int
    num_symbols: int
    width: int
    height: int
    data_patches: int
    total_patches: int
    lm_tokens_2x2: int


def _grid_stats(payload_len: int, palette: int, nsym: int, subpatch: int = 1) -> _GridStats:
    """Mirrors encode()'s exact grid math (see that function's docstring): `num_symbols`
    ecc-bitstream symbols are packed `subpatch*subpatch` per DATA patch
    (`data_patches_needed = ceil(num_symbols / k**2)`) before `compute_grid` sizes the patch
    grid, so `subpatch=1` reproduces the pre-Slice-B formula exactly (`cells_per_patch=1`
    collapses `data_patches_needed` to `num_symbols`)."""
    bps = bits_per_symbol(palette)
    message_len = 5 + payload_len
    ecc_len = rs_encoded_length(message_len, nsym)
    num_symbols = math.ceil(ecc_len * 8 / bps)
    cells_per_patch = subpatch * subpatch
    data_patches_needed = math.ceil(num_symbols / cells_per_patch)
    width, height = compute_grid(data_patches_needed, palette)
    total_patches = width * height
    data_patches = width * (height - 1)
    lm_tokens_2x2 = math.ceil(width / 2) * math.ceil(height / 2)
    return _GridStats(
        ecc_len=ecc_len,
        num_symbols=num_symbols,
        width=width,
        height=height,
        data_patches=data_patches,
        total_patches=total_patches,
        lm_tokens_2x2=lm_tokens_2x2,
    )


def _bits_per_patch_on_success(
    payload_len: int, palette: int, nsym: int, subpatch: int = 1
) -> float:
    """TRUE PAYLOAD DENSITY (F1 fix): payload_len*8 / total_patches -- the actual number of
    PAYLOAD bits (not ecc bits, not raw symbol-channel bits) carried by the WHOLE grid image
    (calibration row included), for a payload of this size, on a successful decode. A property of
    the format for given (palette, payload_len, nsym, subpatch), independent of which corruption
    (if any) is applied.

    REPLACES an earlier formula: `subpatch^2 * bits_per_symbol * (data_patches/total_patches) *
    (payload_bytes/ecc_bytes)`. That formula treated every DATA patch as if it carried a full
    `subpatch^2`-symbol payload drawn straight from the ecc bitstream, discounted only by (a) the
    calibration-row row (`data_patches/total_patches`) and (b) the RS parity overhead
    (`payload/ecc`). It missed a THIRD source of overhead `compute_grid`/`_pack_symbols`
    introduce: `compute_grid` floors grid `width` at `palette` regardless of how few symbols are
    actually needed (so small payloads at large palettes sit in a much wider grid than their
    symbol count requires), and `_pack_symbols` zero-pads the ecc bitstream up to a whole number
    of `bps`-bit symbols. Both are real, unused capacity that the old formula silently credited as
    if it carried payload bits -- overstating true payload density by up to ~3x at small payloads
    (e.g. P=256/subpatch=1/48B: old formula reports 2.259 bits/patch, but the true density,
    payload_len*8/total_patches = 48*8/512, is 0.75 bits/patch).

    Hand-computed sanity anchors (see tests/test_harness.py for the full derivation), all at
    P=256/subpatch=1/nsym=32: 4096B -> total_patches=5120 -> 4096*8/5120 = 6.4 exactly; 16384B ->
    total_patches=19200 -> 16384*8/19200 = 6.826666... ; 48B -> total_patches=512 ->
    48*8/512 = 0.75 exactly. The `subpatch^2*log2(palette)` ceiling this value must never exceed
    (see write_results_md's "Self-consistency checks") is UNCHANGED by this fix: payload_len*8 <=
    ecc_len*8 <= num_symbols*bps <= data_patches*cells_per_patch*bps always holds (RS only adds
    parity, never drops payload bits, and packing only pads up to whole symbols), and
    total_patches > data_patches (the calibration row), so payload_len*8/total_patches is always
    strictly less than cells_per_patch*bps -- the new, honest formula still respects the same
    geometric ceiling as the old one, it just no longer credits padding as payload."""
    g = _grid_stats(payload_len, palette, nsym, subpatch)
    return payload_len * 8 / g.total_patches


def _base64_token_estimate(payload_len: int) -> int:
    """Estimated TOKEN cost of shipping this payload as base64 text: ceil(n/3)*4 base64
    characters (the standard, exact expansion formula) divided by BASE64_CHARS_PER_TOKEN -- the
    MEASURED chars/token of the resolved tokenizer baseline when one is available (see
    _resolve_base64_baseline; 1.3498 measured for Qwen2.5-VL's tokenizer), else the analytic ~1
    char/token assumption (divisor 1.0, which reproduces the old pure-character count exactly).
    Rounded DOWN (floor): understating base64's token cost is the direction CONSERVATIVE AGAINST
    heliogram's claim (a smaller denominator makes heliogram's token_ratio look worse, never
    better), per this project's honesty rule. This is the TOKEN side of the crossover
    comparison; the PATCH side is _grid_stats(...).total_patches -- see _token_crossover."""
    chars = math.ceil(payload_len / 3) * 4
    if chars == 0:
        return 0  # empty payload costs zero tokens either way (parity with the old char count)
    return max(1, math.floor(chars / BASE64_CHARS_PER_TOKEN))


@dataclass
class TokenCrossover:
    total_patches: int
    base64_token_est: int
    token_ratio: float
    heliogram_cheaper: bool
    # LM-token (2x2 spatial merger) accounting -- see _grid_stats' docstring and this function's
    # docstring for the MANDATORY caveat this pair of fields carries.
    lm_tokens_2x2: int
    lm_token_ratio: float


def _token_crossover(
    payload_len: int, palette: int, nsym: int, subpatch: int = 1
) -> TokenCrossover:
    """THE BENEFIT METRIC (see module docstring): total self-hosted-VLM patch/token cost
    (total_patches, ~1 token/patch) vs. the base64-in-text-context token cost
    (base64_token_est, chars scaled by the resolved chars/token baseline -- see
    _base64_token_estimate) for encoding the SAME payload_len bytes -- an accounting
    comparison of context COST, independent of bits/patch density. A config can win here
    (token_ratio < 1.0) while still being below the base64 BITS/PATCH bar, because RS/framing
    overhead
    amortizes differently across the two encodings as payload grows (heliogram pays a fixed
    calibration-row + per-chunk-RS-parity cost once per image; base64 pays none of that, but
    heliogram's symbols are denser than base64's 6-bit characters once payload is large enough
    to amortize its own fixed cost).

    HONESTY: token_ratio < 1.0 ("heliogram_cheaper") is a fact about TOKEN COUNT ONLY. It says
    nothing about whether any actual reader -- pixel decoder or VLM -- can recover the payload
    from that many patches; see decode_success_rate in the same sweep for that. In particular,
    the P=128/256 rows of this sweep are measured (see RESULTS.md's "Token crossover" section
    and tests/test_roundtrip.py's pinned known-failure tests) to FAIL decode under jpeg_q70 even
    where they are token-cheaper clean -- token-cheaper is not usable on its own; only
    clean-decodable-and-token-cheaper is even a candidate benefit, and realizing it under
    corruption is conditional on Phase 2 (a learned reader), which is not measured here.

    `lm_tokens_2x2`/`lm_token_ratio` are a SECOND accounting of the same comparison, against the
    LM-visible token count Qwen2.5-VL's 2x2 spatial merger would actually present to the language
    model (see `_grid_stats`' docstring) instead of the raw ViT-patch count `total_patches`.
    MANDATORY caveat, same epistemic class as the subpatch>1 pixel-decoder-only caveat elsewhere
    in this module: `lm_token_ratio` implicitly assumes the model can read `4 * subpatch**2 *
    log2(palette)` bits out of ONE merged embedding (4 ViT patches' worth of symbols, each
    carrying `subpatch**2 * log2(palette)` bits, folded into a single LM token by the merger) --
    whether it actually CAN is UNVERIFIED and is Phase 2 work, not measured by this
    (`decode_pixels`-based, pixel-exact) harness. `token_ratio`/`total_patches` remain the
    conservative headline accounting; `lm_token_ratio` is reported alongside it, not instead of
    it.
    """
    g = _grid_stats(payload_len, palette, nsym, subpatch)
    base64_tokens = _base64_token_estimate(payload_len)
    ratio = g.total_patches / base64_tokens
    lm_ratio = g.lm_tokens_2x2 / base64_tokens
    return TokenCrossover(
        total_patches=g.total_patches,
        base64_token_est=base64_tokens,
        token_ratio=ratio,
        heliogram_cheaper=ratio < 1.0,
        lm_tokens_2x2=g.lm_tokens_2x2,
        lm_token_ratio=lm_ratio,
    )


def _run_cell(
    palette: int,
    corruption_name: str,
    corruption_fn: Callable,
    n_trials: int = N_TRIALS,
    payload_size: int = PAYLOAD_SIZE,
    subpatch: int = 1,
) -> CellResult:
    bps = bits_per_symbol(palette)
    symbol_errors = 0
    symbol_total = 0
    successes = 0

    for trial in range(n_trials):
        payload = _random_payload(trial, payload_size)
        clean_img = encode(
            payload, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, seed=0, subpatch=subpatch
        )
        corrupted_img = corruption_fn(clean_img)
        # NO resize-back when a corruption changed the image size: the only size-changing
        # corruptions in this suite are the qwen_smart_resize entries, whose entire measured
        # effect IS the size change (the model sees the resized image at its own patch grid --
        # resizing back would erase the corruption). An earlier version of this loop resized
        # any size-changed output back to clean_img.size as a guard against caller-supplied
        # corruption fns; that guard is gone -- decode_pixels/extract_symbols derive the patch
        # grid from the image's ACTUAL size, so a size-changed image is measured exactly as a
        # model would see it (usually: misaligned grid -> decode failure, reported as such).

        _, _, truth = extract_symbols(
            clean_img, palette=palette, patch_size=PATCH_SIZE, subpatch=subpatch
        )
        _, _, observed = extract_symbols(
            corrupted_img, palette=palette, patch_size=PATCH_SIZE, subpatch=subpatch
        )
        # When a size-changing corruption altered the patch-grid shape, positional comparison
        # below is only meaningful over the overlapping prefix (and near-meaningless even there
        # -- rows have shifted); decode_success_rate is the honest headline for those cells.
        n = min(len(truth), len(observed))
        symbol_errors += sum(1 for i in range(n) if truth[i] != observed[i])
        symbol_total += n

        try:
            decoded = decode_pixels(
                corrupted_img, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, subpatch=subpatch
            )
            if decoded == payload:
                successes += 1
        except HeliogramDecodeError:
            pass

    success_rate = successes / n_trials
    bpp_on_success = _bits_per_patch_on_success(payload_size, palette, NSYM, subpatch)
    crossover = _token_crossover(payload_size, palette, NSYM, subpatch)
    return CellResult(
        palette=palette,
        subpatch=subpatch,
        payload_size=payload_size,
        corruption=corruption_name,
        bits_per_symbol=bps,
        symbol_error_rate=(symbol_errors / symbol_total) if symbol_total else 0.0,
        decode_success_rate=success_rate,
        bits_per_patch=bpp_on_success * success_rate,
        trials=n_trials,
        total_patches=crossover.total_patches,
        base64_token_est=crossover.base64_token_est,
        token_ratio=crossover.token_ratio,
        heliogram_cheaper=crossover.heliogram_cheaper,
        lm_tokens_2x2=crossover.lm_tokens_2x2,
        lm_token_ratio=crossover.lm_token_ratio,
    )


def run(
    palettes: Sequence[int] = PALETTES,
    corruptions: Optional[Dict[str, Callable]] = None,
    n_trials: int = N_TRIALS,
    subpatches: Sequence[int] = (1,),
    payload_sizes: Sequence[int] = (PAYLOAD_SIZE,),
) -> List[CellResult]:
    """Run every (palette, subpatch, payload_size) x corruption cell. Defaults reproduce the
    pre-Slice-B behavior exactly (single subpatch=1, single payload_size=PAYLOAD_SIZE); pass
    `subpatches=SUBPATCHES, payload_sizes=SWEEP_PAYLOAD_SIZES` for the full capacity sweep."""
    if corruptions is None:
        corruptions = CORRUPTIONS
    return [
        _run_cell(palette, name, fn, n_trials, payload_size=payload_size, subpatch=subpatch)
        for palette in palettes
        for subpatch in subpatches
        for payload_size in payload_sizes
        for name, fn in corruptions.items()
    ]


def format_table(results: List[CellResult]) -> str:
    headers = [
        "palette",
        "subpatch",
        "payload_size",
        "bits/sym",
        "corruption",
        "symbol_err_rate",
        "decode_success",
        "bits/patch",
    ]
    rows = [
        [
            str(r.palette),
            str(r.subpatch),
            str(r.payload_size),
            str(r.bits_per_symbol),
            r.corruption,
            f"{r.symbol_error_rate:.4f}",
            f"{r.decode_success_rate:.2f}",
            f"{r.bits_per_patch:.3f}",
        ]
        for r in results
    ]
    widths = [
        max([len(h)] + [len(row[i]) for row in rows]) for i, h in enumerate(headers)
    ]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines += [fmt_row(row) for row in rows]
    return "\n".join(lines)


def format_gate_table(gate_rows: List[Dict[str, object]]) -> str:
    """Render the headline Gate #1 rows (see _gate_rows) as a plain-text table for stdout.
    Includes Bar A (`beats_base64_N`, clean-only -- see BASE64_BITS_PER_TOKEN, whose value N is
    baked into the column name here so the column never silently mislabels itself once
    BASE64_BITS_PER_TOKEN comes from a measured baseline instead of the analytic 6.0 default --
    see _resolve_base64_baseline) alongside Gate #1's clean/worst/both columns, since
    RESULTS.md's Headline section reports both bars and stdout should not silently drop the one
    this project considers the real economic bar."""
    headers = [
        "palette",
        "subpatch",
        "payload",
        "ceiling",
        "clean_bpp",
        f"beats_base64_{BASE64_BITS_PER_TOKEN:.0f}",
        "worst_bpp",
        "worst_corruption",
        "clears_clean_8",
        "clears_worst_8",
        "clears_both_8",
    ]
    rows = [
        [
            str(r["palette"]),
            str(r["subpatch"]),
            str(r["payload_size"]),
            str(r["ceiling"]),
            f"{r['clean']:.3f}",
            "yes" if r["beats_base64_clean"] else "no",
            f"{r['worst']:.3f}",
            str(r["worst_name"]),
            "yes" if r["clears_clean"] else "no",
            "yes" if r["clears_worst"] else "no",
            "YES" if r["clears_both"] else "no",
        ]
        for r in gate_rows
    ]
    widths = [
        max([len(h)] + [len(row[i]) for row in rows]) for i, h in enumerate(headers)
    ]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines += [fmt_row(row) for row in rows]
    return "\n".join(lines)


def _provenance_line() -> str:
    """One-line environment fingerprint (Python version, key dependency versions, OS/arch)
    recorded near the top of RESULTS.md -- see B5 in the fix plan: today's committed
    RESULTS.md/results.csv artifacts are otherwise undiagnosable when a library drifts (numpy,
    Pillow, or reedsolo version bumps can silently change PNG container bytes, JPEG encoder
    behavior/quality curves, or Reed-Solomon chunking/output, any of which would move the
    numbers in this file without changing a single line of heliogram's own code -- see
    tests/test_roundtrip.py's Pillow-12.3-pinned-hash known failure for a concrete example of
    exactly this happening).

    Versions are read via `importlib.metadata.version`, which reads the INSTALLED package's
    dist-info metadata, not the imported module's own `__version__` attribute -- some
    dependencies (e.g. reedsolo) don't reliably expose the latter, and the former is also more
    honest about what's actually pip-installed in this environment (matters if a module were
    ever vendored or monkeypatched). Falls back to "unknown" for any package whose dist-info
    isn't found rather than raising, so this function -- and everything that calls it -- never
    fails an otherwise-successful run just because provenance couldn't be determined.
    """

    def _ver(pkg: str) -> str:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    return (
        f"Python {platform.python_version()}; numpy {_ver('numpy')}; "
        f"Pillow {_ver('Pillow')}; reedsolo {_ver('reedsolo')}; "
        f"platform: {platform.platform()}"
    )


def write_csv(results: List[CellResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "palette",
                "subpatch",
                "payload_size",
                "bits_per_symbol",
                "corruption",
                "symbol_error_rate",
                "decode_success_rate",
                "bits_per_patch",
                "trials",
                "total_patches",
                "base64_token_est",
                "token_ratio",
                "heliogram_cheaper_bool",
                "lm_tokens_2x2",
                "lm_token_ratio",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.palette,
                    r.subpatch,
                    r.payload_size,
                    r.bits_per_symbol,
                    r.corruption,
                    f"{r.symbol_error_rate:.6f}",
                    f"{r.decode_success_rate:.4f}",
                    f"{r.bits_per_patch:.6f}",
                    r.trials,
                    r.total_patches,
                    r.base64_token_est,
                    f"{r.token_ratio:.6f}",
                    r.heliogram_cheaper,
                    r.lm_tokens_2x2,
                    f"{r.lm_token_ratio:.6f}",
                ]
            )


def _summary_rows(
    results: List[CellResult],
) -> Dict[Tuple[int, int, int], Dict[str, object]]:
    """Roll up per-(palette, subpatch, payload_size) bucket: 'clean' bits/patch, the mean
    bits/patch over every non-clean corruption ('corrupted_mean' -- the metric the original
    v0.1 README/RESULTS summary table used, kept for continuity), and the worst (minimum)
    non-clean bits/patch plus which corruption produced it ('corrupted_worst' /
    'corrupted_worst_name') -- the metric the headline Gate #1 table needs, since a config must
    clear the gate even in its worst tested case, not just on average."""
    by_bucket: Dict[Tuple[int, int, int], Dict[str, list]] = {}
    for r in results:
        key = (r.palette, r.subpatch, r.payload_size)
        bucket = by_bucket.setdefault(key, {"clean": [], "corrupted": []})
        if r.corruption == "clean":
            bucket["clean"].append(r.bits_per_patch)
        else:
            bucket["corrupted"].append((r.bits_per_patch, r.corruption))

    summary: Dict[Tuple[int, int, int], Dict[str, object]] = {}
    for key, buckets in by_bucket.items():
        clean_vals = buckets["clean"] or [0.0]
        corrupted_pairs = buckets["corrupted"] or [(0.0, "n/a")]
        corrupted_vals = [v for v, _ in corrupted_pairs]
        worst_val, worst_name = min(corrupted_pairs, key=lambda pair: pair[0])
        summary[key] = {
            "clean": sum(clean_vals) / len(clean_vals),
            "corrupted_mean": sum(corrupted_vals) / len(corrupted_vals),
            "corrupted_worst": worst_val,
            "corrupted_worst_name": worst_name,
        }
    return summary


def _gate_rows(summary: Dict[Tuple[int, int, int], Dict[str, object]]) -> List[Dict[str, object]]:
    """One row per (palette, subpatch, payload_size) bucket for the headline Gate #1 table:
    the ceiling subpatch^2*log2(palette), clean bits/patch, worst-case (min over non-clean
    corruptions) bits/patch and which corruption produced it, and whether each of clean/worst/
    both clears GATE_BITS_PER_PATCH. Also reports `beats_base64_clean`: Bar A, the real economic
    bar (see module docstring) -- does clean bits/patch alone beat BASE64_BITS_PER_TOKEN? Unlike
    Gate #1, Bar A is evaluated clean-only, not clean-and-worst (see RESULTS.md's Headline
    section for why: Bar A is about raw density being worth considering at all; robustness is a
    separate, already-visible column in the same table)."""
    rows: List[Dict[str, object]] = []
    for key in sorted(summary):
        palette, subpatch, payload_size = key
        bps = bits_per_symbol(palette)
        ceiling = subpatch * subpatch * bps
        s = summary[key]
        clean = s["clean"]
        worst = s["corrupted_worst"]
        clears_clean = clean >= GATE_BITS_PER_PATCH - 1e-9
        clears_worst = worst >= GATE_BITS_PER_PATCH - 1e-9
        beats_base64_clean = clean >= BASE64_BITS_PER_TOKEN - 1e-9
        rows.append(
            {
                "palette": palette,
                "subpatch": subpatch,
                "payload_size": payload_size,
                "ceiling": ceiling,
                "clean": clean,
                "worst": worst,
                "worst_name": s["corrupted_worst_name"],
                "clears_clean": clears_clean,
                "clears_worst": clears_worst,
                "clears_both": clears_clean and clears_worst,
                "beats_base64_clean": beats_base64_clean,
            }
        )
    return rows


def _token_crossover_rows(results: List[CellResult]) -> List[Dict[str, object]]:
    """One row per (palette, subpatch, payload_size) bucket for the headline Token-crossover
    table: total_patches/base64_token_est/token_ratio/heliogram_cheaper AND the LM-token (2x2
    merger) accounting lm_tokens_2x2/lm_token_ratio (identical across every corruption in the
    bucket -- see CellResult/_token_crossover -- so read off any one row in the bucket, not
    averaged/reduced), plus the clean and jpeg_q70 decode_success_rate for that SAME bucket so
    the token-count benefit and the pixel-decoder's actual corrupted-decode result sit in the
    same row -- see the Headline section's mandatory caveat: token-cheaper is not the same as
    decodable, and this is where that gets shown, not just asserted."""
    by_bucket: Dict[Tuple[int, int, int], Dict[str, object]] = {}
    for r in results:
        key = (r.palette, r.subpatch, r.payload_size)
        row = by_bucket.setdefault(
            key,
            {
                "palette": r.palette,
                "subpatch": r.subpatch,
                "payload_size": r.payload_size,
                "total_patches": r.total_patches,
                "base64_token_est": r.base64_token_est,
                "token_ratio": r.token_ratio,
                "heliogram_cheaper": r.heliogram_cheaper,
                "lm_tokens_2x2": r.lm_tokens_2x2,
                "lm_token_ratio": r.lm_token_ratio,
                "clean_decode_success": None,
                "jpeg_q70_decode_success": None,
            },
        )
        if r.corruption == "clean":
            row["clean_decode_success"] = r.decode_success_rate
        elif r.corruption == "jpeg_q70":
            row["jpeg_q70_decode_success"] = r.decode_success_rate
    return [by_bucket[k] for k in sorted(by_bucket)]


@dataclass
class ExactCrossover:
    """Result of `exact_crossover_payload_size`'s byte-granular scan. `crossing_bytes` is the
    exact smallest payload size (bytes) where the ratio first drops below 1.0, or None if it
    never does anywhere in `[1, max_bytes_scanned]`. `recrossing_bytes` lists every payload size
    AFTER `crossing_bytes` where the ratio goes back to >= 1.0 -- non-empty here means the
    "crossing" reported is not a stable one-way threshold and callers MUST say so (see the
    function's docstring: this staircase-vs-closed-form ratio is not guaranteed monotonic).
    `lowest_ratio`/`lowest_ratio_bytes` are the minimum ratio observed anywhere in the scan and
    the payload size at which it occurred, reported so "no crossover" can still say how close it
    got."""

    crossing_bytes: Optional[int]
    recrossing_bytes: List[int]
    max_bytes_scanned: int
    lowest_ratio: float
    lowest_ratio_bytes: int


def exact_crossover_payload_size(
    palette: int,
    subpatch: int,
    nsym: int,
    token_estimator: Callable[[int], int],
    lm_tokens: bool = False,
    max_bytes: int = 65536,
) -> ExactCrossover:
    """Exact (byte-granular) replacement for the old `_crossover_payload_size`, which linearly
    interpolated between the handful of payload sizes actually swept by the harness (48, 1024,
    4096, 16384) and reported the result to the nearest byte (e.g. "~3055B") -- spurious
    precision, since interpolating a STAIRCASE function (total_patches/lm_tokens_2x2 only change
    at grid-size boundaries, not smoothly) between two widely-spaced sample points has no reason
    to land anywhere near the function's actual crossing point.

    Both sides of the ratio this function scans are closed-form for ANY payload size, not just
    the handful the harness happens to sweep: the token-count side is `token_estimator(n)`
    (`_base64_token_estimate`, `ceil(n/3)*4` chars scaled by the resolved chars/token baseline
    -- closed-form for every n), and the patch/token side is
    `_grid_stats(n, ...).total_patches` or `.lm_tokens_2x2` (exact for every n, mirroring
    `encode`'s own grid math). So instead of interpolating between samples, this walks payload_len
    = 1..max_bytes ONE BYTE AT A TIME and finds the exact smallest payload_len where
    `numerator(payload_len) / token_estimator(payload_len) < 1.0`, where `numerator` is
    `total_patches` (`lm_tokens=False`, the conservative per-patch accounting) or `lm_tokens_2x2`
    (`lm_tokens=True`, the Qwen2.5-VL 2x2-merger accounting -- see `_grid_stats`'/
    `_token_crossover`'s docstrings for the MANDATORY caveat that accounting carries).

    Because the numerator is a staircase (constant between grid-size boundaries, then jumping)
    while the denominator grows in fixed +4-per-3-bytes steps, the ratio is NOT guaranteed
    monotonic: it can drop below 1.0, tick back up at the next grid-size jump (adding a
    calibration-row patch that isn't there yet at the prior payload size), and cross again. This
    function scans the FULL range (never stops early at the first crossing) so it can report
    every such recrossing in `.recrossing_bytes` -- callers MUST check that list and describe the
    wobble rather than silently stating a single "crosses at NB" threshold when it is non-empty,
    per this project's data-honesty rule (see write_results_md's "Token crossover" section).

    Returns `.crossing_bytes = None` if the ratio never drops below 1.0 anywhere in
    `[1, max_bytes]`. That is NOT the same claim as "never crosses" -- it may cross at a larger,
    unscanned payload size -- callers must report it as "no crossover found in the exact scan up
    to <max_bytes>B", not as a hard negative (same honesty rule the old interpolating version
    carried, just against the wider and exact -- not sampled -- range this version actually
    covers).
    """
    crossing: Optional[int] = None
    recrossing: List[int] = []
    prev_below = False
    lowest_ratio: Optional[float] = None
    lowest_ratio_bytes = 0
    for payload_len in range(1, max_bytes + 1):
        g = _grid_stats(payload_len, palette, nsym, subpatch)
        numerator = g.lm_tokens_2x2 if lm_tokens else g.total_patches
        tokens = token_estimator(payload_len)
        ratio = numerator / tokens if tokens else math.inf
        if lowest_ratio is None or ratio < lowest_ratio:
            lowest_ratio = ratio
            lowest_ratio_bytes = payload_len
        below = ratio < 1.0
        if below and crossing is None:
            crossing = payload_len
        elif crossing is not None and prev_below and not below:
            recrossing.append(payload_len)
        prev_below = below
    return ExactCrossover(
        crossing_bytes=crossing,
        recrossing_bytes=recrossing,
        max_bytes_scanned=max_bytes,
        lowest_ratio=lowest_ratio if lowest_ratio is not None else 0.0,
        lowest_ratio_bytes=lowest_ratio_bytes,
    )


def _describe_exact_crossover(result: ExactCrossover, label: str) -> str:
    """Render one `ExactCrossover` (see `exact_crossover_payload_size`) as a short prose
    fragment for the 'Crossover payload size' subsection -- shared by the per-patch and LM-token
    accountings so the two read identically apart from which numbers they carry. `label` is
    prepended so a line combining both reads unambiguously, e.g. "per-patch crosses at 2989B
    (exact); LM-token crosses at 750B (exact)"."""
    if result.crossing_bytes is None:
        return (
            f"{label} no crossover found in the exact scan up to {result.max_bytes_scanned}B "
            f"(lowest ratio observed: {result.lowest_ratio:.3f} at {result.lowest_ratio_bytes}B)"
        )
    text = f"{label} crosses at {result.crossing_bytes}B (exact)"
    if result.recrossing_bytes:
        text += (
            f" -- WOBBLES: recrosses back to >= 1.0 at {result.recrossing_bytes[0]}B (and "
            f"{len(result.recrossing_bytes) - 1} more time(s) within the scanned range) -- NOT "
            "a stable one-way threshold, see exact_crossover_payload_size's docstring"
        )
    return text


def _token_crossover_section(
    results: List[CellResult],
    payload_sizes_present: List[int],
    subpatches_present: List[int],
    palettes_present: List[int],
) -> List[str]:
    """Builds the '## Token crossover' section of RESULTS.md: THE benefit claim this project
    can currently measure (see module docstring / _token_crossover) -- total patches vs. base64
    token estimate for the SAME payload, at every (palette, subpatch, payload_size) bucket in
    this sweep, with the clean and jpeg_q70 decode_success_rate for that same bucket shown in
    the same row so the token-count benefit and the pixel-decoder's actual corrupted-decode
    result are never shown apart from each other. Also includes a dedicated LM-token (Qwen2.5-VL
    2x2 spatial merger) subsection presenting the SAME comparison against the merged, LM-visible
    token count alongside the conservative per-patch accounting, and an exact (byte-granular, not
    interpolated) crossover-payload-size scan for both accountings -- see
    `exact_crossover_payload_size`."""
    crossover_rows = _token_crossover_rows(results)
    lines = [
        "## Token crossover: the actual measured benefit",
        "",
        "THE benefit claim this project can currently make: does encoding a payload as a "
        "heliogram grid cost fewer total patches (`total_patches`, the grid's width*height -- "
        "~1 token/patch for a self-hosted VLM that tokenizes at the same patch grid) than "
        "base64-ing the same payload bytes into text tokens (`base64_token_est` = "
        "ceil(payload/3)*4 base64 characters divided by "
        f"chars/token = {BASE64_CHARS_PER_TOKEN:.4f} -- the resolved tokenizer baseline, see "
        "Baselines above)? `token_ratio = total_patches / base64_token_est`; "
        "`token_ratio < 1.0` means heliogram is CHEAPER on token count for that payload -- an "
        "accounting fact about total context cost for the WHOLE payload, distinct from the "
        "bits/patch DENSITY bars in the Headline section (a config can win here while losing "
        "on bits/patch, because the two encodings amortize fixed overhead differently as "
        "payload grows: heliogram pays a calibration row + per-RS-chunk parity once per image, "
        "base64 pays none of that but never exceeds 6 bits/char either).",
        "",
        "**HONESTY (mandatory, same rule as everywhere else in this file):** `token_ratio` and "
        "`heliogram_cheaper` are computed from `total_patches` alone -- a property of grid "
        "geometry -- regardless of whether `decode_success_rate` for that same cell is 1.0 or "
        "0.0. Token-cheaper is an accounting fact about COUNT, not a claim that any reader can "
        "actually recover the payload from that many patches. The table below shows both "
        "numbers for every bucket side by side, on purpose: for `palette` in {128, 256}, "
        "`token_ratio` can drop below 1.0 at a payload size where `jpeg_q70 decode success` "
        "is still 0.00 in this same sweep -- so the token-count benefit these two palettes "
        "unlock is currently a CLEAN-CHANNEL-ONLY number. Usability under real corruption is "
        "exactly the Phase-2 reader-robustness bet described in the Headline section above, "
        "not something this table settles. This table's `total_patches` is the CONSERVATIVE, "
        "~1-ViT-patch/token accounting; see the dedicated LM-token subsection below for the "
        "SAME comparison against the Qwen2.5-VL 2x2-merged token count, which carries its own, "
        "separate mandatory caveat.",
        "",
        "| palette | subpatch | payload (B) | total_patches | base64_token_est | token_ratio | "
        "cheaper on tokens? | clean decode success | jpeg_q70 decode success |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in crossover_rows:
        clean_ds = row["clean_decode_success"]
        jpeg_ds = row["jpeg_q70_decode_success"]
        clean_str = f"{clean_ds:.2f}" if clean_ds is not None else "n/a"
        jpeg_str = f"{jpeg_ds:.2f}" if jpeg_ds is not None else "n/a"
        lines.append(
            f"| {row['palette']} | {row['subpatch']} | {row['payload_size']} | "
            f"{row['total_patches']} | {row['base64_token_est']} | "
            f"{row['token_ratio']:.3f} | {'**YES**' if row['heliogram_cheaper'] else 'no'} | "
            f"{clean_str} | {jpeg_str} |"
        )

    lines += [
        "",
        "### LM-token accounting (Qwen2.5-VL 2x2 spatial merger)",
        "",
        "**MANDATORY caveat, same epistemic class as the subpatch>1 pixel-decoder-only caveat "
        "in the Headline section above:** the table below re-accounts total token cost against "
        "`lm_tokens_2x2 = ceil(width/2) * ceil(height/2)` -- the LM-VISIBLE token count after "
        "Qwen2.5-VL's 2x2 spatial merger folds every 2x2 block of ViT patches into ONE token "
        "the language model actually sees -- instead of the raw `total_patches` (ViT-patch) "
        "count the table above uses. Because 4 ViT patches collapse into 1 merged token, this "
        "means each merged LM token must carry `4 * subpatch**2 * log2(palette)` bits of payload "
        "(4 patches' worth of symbols, folded into ONE merged embedding) for a reader to recover "
        "the payload from that many LM tokens. WHETHER the model can actually read that many "
        "symbols back out of a single merged embedding is UNVERIFIED: this harness only ever "
        "samples exact, known pixel coordinates via `decode_pixels`/`extract_symbols` -- it "
        "never asks a real vision encoder (merged or not) to resolve anything. Realizing (or "
        "falsifying) this is exactly the Phase-2 GPU measurement this project has not yet done. "
        "**The per-patch accounting in the table above remains the conservative headline "
        "number; treat every `lm_token_ratio` / 'cheaper (LM)?' value below as an UPPER BOUND "
        "on the possible benefit, not a result.**",
        "",
        "| palette | subpatch | payload (B) | total_patches | token_ratio (per-patch) | "
        "cheaper (per-patch)? | lm_tokens_2x2 | lm_token_ratio (2x2 merger) | cheaper (LM, "
        "UNVERIFIED)? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in crossover_rows:
        lines.append(
            f"| {row['palette']} | {row['subpatch']} | {row['payload_size']} | "
            f"{row['total_patches']} | {row['token_ratio']:.3f} | "
            f"{'yes' if row['heliogram_cheaper'] else 'no'} | {row['lm_tokens_2x2']} | "
            f"{row['lm_token_ratio']:.3f} | "
            f"{'**YES**' if row['lm_token_ratio'] < 1.0 else 'no'} |"
        )

    present_pairs = sorted({(row["palette"], row["subpatch"]) for row in crossover_rows})
    subpatch1_crossings: List[Tuple[int, int]] = []
    exact_max_bytes = 65536

    lines += ["", "### Crossover payload size per (palette, subpatch) -- exact scan", ""]
    lines += [
        "Exact, byte-granular payload size where each accounting's ratio first drops below "
        "1.0 -- NOT the linear interpolation between the handful of swept sample points "
        f"({list(payload_sizes_present)}B) an earlier version of this section used, which could "
        "report spurious precision (e.g. '~3055B') for what is actually a staircase function "
        "(see `exact_crossover_payload_size`'s docstring for why). Both sides of the ratio are "
        "closed-form for ANY payload size -- `_grid_stats(...).total_patches`/`.lm_tokens_2x2` "
        "and `ceil(n/3)*4` -- so this instead walks every payload size from 1B up to "
        f"{exact_max_bytes}B and reports the exact smallest crossing, for BOTH the per-patch and "
        "LM-token (2x2 merger, see the caveat above) accountings. 'no crossover found' means "
        "the ratio never dropped below 1.0 anywhere in the scanned range -- NOT a claim it "
        "never will at a larger, unscanned payload size. Where the ratio recrosses back above "
        "1.0 after its first crossing (the staircase can wobble), that is reported too, rather "
        "than silently stating a single crossing point as if it were a stable threshold.",
        "",
    ]
    for subpatch in subpatches_present:
        label = (
            "VLM-meaningful: one symbol per patch"
            if subpatch == 1
            else "PIXEL-DECODER GEOMETRIC CEILING ONLY -- not a VLM capability claim, see caveat above"
        )
        lines += [f"**subpatch={subpatch} ({label}):**", ""]
        for palette in palettes_present:
            if (palette, subpatch) not in present_pairs:
                continue
            per_patch = exact_crossover_payload_size(
                palette, subpatch, NSYM, _base64_token_estimate,
                lm_tokens=False, max_bytes=exact_max_bytes,
            )
            lm = exact_crossover_payload_size(
                palette, subpatch, NSYM, _base64_token_estimate,
                lm_tokens=True, max_bytes=exact_max_bytes,
            )
            lines.append(
                f"- palette={palette}: "
                f"{_describe_exact_crossover(per_patch, 'per-patch')}; "
                f"{_describe_exact_crossover(lm, 'LM-token (UNVERIFIED, see caveat above)')}"
            )
            if subpatch == 1 and per_patch.crossing_bytes is not None:
                subpatch1_crossings.append((palette, per_patch.crossing_bytes))
        lines.append("")

    lines += ["### Token-crossover verdict", ""]
    if subpatch1_crossings:
        parts = [f"palette={p} at {c}B (exact)" for p, c in sorted(subpatch1_crossings)]
        lines.append(
            "At `subpatch=1` (the only VLM-meaningful regime), the following palette(s) cross "
            "below base64 token count, PER-PATCH accounting, within the exact scanned range: "
            + ", ".join(parts) + ". "
            "This is the project's actual, currently-measured benefit claim: for a large "
            "enough payload, encoding it as a heliogram grid costs fewer total patches than "
            "base64-ing it into text tokens, and (per Bar A in the Headline section) does so "
            "at a bits/patch density that also beats plain base64 text, and is bit-exact on a "
            "successful decode (Reed-Solomon verified). **This is a clean-channel, "
            "token-accounting result only** -- see the mandatory P=128/256 corruption caveat "
            "in the Headline section above: the pixel decoder cannot currently realize this "
            "benefit end to end under `jpeg_q70` at these same palettes. The open question is "
            "purely whether a fine-tuned VLM reader can, which is Phase 2 and is not measured "
            "here. The LM-token (2x2 merger) accounting above crosses at smaller payload sizes "
            "still, but carries the additional, separate, UNVERIFIED assumption that a merged "
            "embedding can be read at 4x the per-patch symbol density -- see that subsection's "
            "caveat; it is not folded into this verdict."
        )
    else:
        lines.append(
            "No `subpatch=1` (VLM-meaningful) palette crosses below base64 token count, "
            f"per-patch accounting, anywhere in the exact scanned range (up to "
            f"{exact_max_bytes}B) in this run."
        )
    lines.append("")
    return lines


def _resolve_text_baselines() -> object:
    """Resolve the OPTIONAL multi-encoding text baseline (heliogram.baselines.
    load_measured_text_baselines) -- the measurement that answers 'is base64 even the right
    bar?'. Same defensive-getattr pattern as _resolve_base64_baseline, same soft-absence
    contract: returns None when no measurement file exists, and callers must then say the bar
    is base64-only (a caveat), never invent a stronger number."""
    loader = getattr(_baselines_module, "load_measured_text_baselines", None)
    if loader is None:
        return None
    return loader()


def write_results_md(
    results: List[CellResult],
    path: Path,
    stress_results: Optional[List[CellResult]] = None,
) -> None:
    baseline = _resolve_base64_baseline()
    text_baselines = _resolve_text_baselines()
    # Is base64 actually the strongest way to ship bytes as text on this tokenizer? This project
    # was burned once by an under-measured baseline (analytic 6.0 -> measured 8.096 flipped every
    # verdict); this block keeps the same error from recurring with base64-vs-better-encodings.
    if text_baselines is not None:
        _strongest = text_baselines.strongest
        if _strongest.bits_per_token > baseline.bits_per_token + 1e-9:
            bar_a_qualifier = (
                f" **MEASURED CAVEAT: the strongest measured text encoding on this tokenizer "
                f"is `{_strongest.encoding}` at {_strongest.bits_per_token:.3f} bits/token, "
                "ABOVE the base64 bar -- every Bar A verdict below UNDERSTATES the true "
                "text-context break-even by that margin; see Baselines.**"
            )
        else:
            bar_a_qualifier = (
                f" (Cross-checked against {len(text_baselines.encodings)} measured text "
                f"encodings: the strongest is `{_strongest.encoding}` at "
                f"{_strongest.bits_per_token:.3f} bits/token, so the base64 bar IS the "
                "operative text-context bar on this tokenizer, not an understatement.)"
            )
    else:
        bar_a_qualifier = (
            " **UNMEASURED CAVEAT: no multi-encoding text-baseline measurement exists in this "
            "checkout (heliogram/data/text_baselines.json missing), so whether base64 is even "
            "the strongest reasonable text encoding on this tokenizer is UNKNOWN -- a stronger "
            "one (ascii85/base85) would raise this bar further. Run `python -m "
            "heliogram.baselines --measure` (needs transformers + HF Hub access) to close "
            "this.**"
        )
    sample_payload = _random_payload(0, PAYLOAD_SIZE)
    rendered = rendered_text_density(sample_payload, patch_size=PATCH_SIZE)
    summary = _summary_rows(results)
    gate_rows = _gate_rows(summary)
    clearing = [r for r in gate_rows if r["clears_both"]]
    any_subpatch1_clears = any(r["clears_both"] for r in gate_rows if r["subpatch"] == 1)
    any_subpatch_gt1_clears = any(r["clears_both"] for r in gate_rows if r["subpatch"] > 1)

    payload_sizes_present = sorted({r.payload_size for r in results})
    subpatches_present = sorted({r.subpatch for r in results})
    palettes_present = sorted({r.palette for r in results})
    sweep_trials = results[0].trials if results else N_TRIALS
    n_corruptions = len({r.corruption for r in results}) or len(CORRUPTIONS)
    # subpatch=1 -> k**2==1, so log2(palette) alone is the ceiling; use the largest palette
    # actually present in `results` (not the module-wide PALETTES constant) so this stays
    # correct if write_results_md is ever called with a partial palette sweep.
    max_subpatch1_ceiling = (
        max(bits_per_symbol(p) for p in palettes_present) if palettes_present else 0
    )
    max_palette_present = max(palettes_present) if palettes_present else 0

    lines = [
        "# heliogram v0.1 -- CPU eval results",
        "",
        f"**Provenance:** {_provenance_line()}",
        "",
        f"Synthetic, seed-deterministic payloads. Capacity sweep: palette in "
        f"{list(palettes_present)}, subpatch (k) in {list(subpatches_present)}, payload_size "
        f"(bytes) in {list(payload_sizes_present)}, x {n_corruptions} corruptions (incl. "
        f"'clean'), {sweep_trials} trials/cell, nsym={NSYM}, patch_size={PATCH_SIZE}px. "
        "Reference decoder = decode_pixels (no model).",
        "",
        "**Scope: this file characterizes the CODE/CHANNEL only.** Every number below comes "
        "from `decode_pixels`, the model-free reference decoder (pixel sampling + "
        "nearest-neighbor classification + Reed-Solomon, no VLM in the loop). Whether a "
        "fine-tuned VLM can realize this same capacity through its own vision encoder is "
        "Phase 2 and is not measured anywhere in this repo -- see the README's "
        "\"Roadmap / Phase-2 boundary\" section.",
        "",
        "**Wall-clock note:** the full sweep below is "
        f"{len(palettes_present)} palettes x {len(subpatches_present)} subpatch values x "
        f"{len(payload_sizes_present)} payload sizes x {n_corruptions} corruptions = "
        f"{len(palettes_present) * len(subpatches_present) * len(payload_sizes_present) * n_corruptions} "
        f"cells; at the largest payload tier ({max(payload_sizes_present)}B) each cell "
        "encodes/corrupts/decodes a multi-thousand-patch image, so trial count for this sweep "
        f"was reduced to {sweep_trials} (module default is {N_TRIALS}) to bound wall-clock. "
        "The diagnostic stress suite below still runs at the module default "
        f"{N_TRIALS} trials, at a single representative config (subpatch=1, "
        f"payload_size={PAYLOAD_SIZE}B) -- see that section.",
        "",
        "## Headline: three bars, and the actual benefit (token crossover)",
        "",
        "This project tracks THREE bars, deliberately kept separate because they answer "
        "different questions -- conflating them is exactly the overclaiming this file exists "
        "to prevent:",
        "",
        f"- **Bar A -- beat base64 density, clean ({BASE64_BITS_PER_TOKEN:.1f} bits/patch):** "
        "the real economic break-even for bits/patch alone (see Baselines below) -- the "
        "minimum for heliogram to be worth considering purely on density. Evaluated CLEAN "
        f"only in the table below (see the 'beats {BASE64_BITS_PER_TOKEN:.0f} clean?' column); "
        "a config beating Bar A clean may or may not survive corruption -- the worst-corruption "
        "columns in the same row show that separately, and it is not folded into this bar."
        + bar_a_qualifier,
        f"- **Bar B -- Gate #1 comfort margin ({GATE_BITS_PER_PATCH:.1f} bits/patch, clean AND "
        "worst-tested-corruption):** originally set as a robustness margin ABOVE Bar A "
        "before this project starts Phase 2 (see the README's Decision Gate). A config \"clears "
        "the gate\" only if its bits/patch is at or above this bar BOTH on a clean image AND in "
        "its single worst-performing tested corruption -- a config that only clears on average "
        "is not a robust win."
        + (
            f" **NOTE: the measured Bar A ({BASE64_BITS_PER_TOKEN:.3f} bits/token) now EXCEEDS "
            f"Gate #1's fixed {GATE_BITS_PER_PATCH:.1f}-bit bar, so Gate #1 no longer functions "
            "as a comfort margin above the economic bar -- clearing Gate #1 is NOT sufficient "
            "to beat measured base64 density. Reported for continuity; Bar A is the bar that "
            "matters.**"
            if BASE64_BITS_PER_TOKEN > GATE_BITS_PER_PATCH
            else " **This is a conservative comfort margin, not the real economic bar** -- see "
            "Bar A and Bar C."
        ),
        "- **Bar C -- token crossover (the actual measured benefit claim):** does encoding a "
        "payload as a heliogram grid cost FEWER total patches (~1 token/patch for a "
        "self-hosted VLM) than base64-ing the same payload into text tokens (at "
        f"chars/token = {BASE64_CHARS_PER_TOKEN:.4f}, the resolved tokenizer baseline)? "
        "This is an ACCOUNTING comparison of token COUNT, not bits/patch density -- a config "
        "can win on Bar C while still failing Bar A, because RS/framing overhead amortizes "
        "differently for the two encodings as payload grows. See the dedicated \"Token "
        "crossover\" section below for the real numbers and the crossover payload size per "
        "palette.",
        "",
        "**MANDATORY honesty caveat:** rows with `subpatch=1` are the VLM-meaningful regime -- "
        "one symbol per DATA patch, i.e. one symbol per (nominal) vision token, the only regime "
        "this project claims any real relevance to a downstream VLM. Rows with `subpatch>1` "
        "are a **PIXEL-DECODER GEOMETRIC CEILING ONLY**: `decode_pixels`/`extract_symbols` can "
        "read sub-patch cells trivially because they sample known, exact pixel coordinates off "
        "a grid whose size they are told in advance -- there is no perception involved. "
        "Whether a real ViT/VLM image encoder can resolve sub-patch structure at all is "
        "**unverified, and doubtful** (a k x k sub-cell grid inside one ViT patch may simply "
        "average out in that patch's embedding). Realizing it is Phase 2 work, gated on GPU "
        "access, and is **not a capability claim** made anywhere in this repo.",
        "",
        "**Also mandatory, and specific to the largest palettes (visible here, in the headline "
        "area, on purpose):** `palette=128` and `palette=256` clean-decode exactly on this "
        "pixel decoder (see `tests/test_roundtrip.py`) but are MEASURED to FAIL decode under "
        "`jpeg_q70` in this very sweep (see the full breakdown below and the \"Token crossover\" "
        "section, which shows the clean-token-cheaper number and the corrupted-decode-failure "
        "number for the SAME cells side by side). The token-count benefit these two palettes "
        "unlock (Bar C) is therefore a property of the CLEAN channel only -- it is **not "
        "currently usable end to end** on this reference decoder, and realizing it under "
        "corruption is conditional on Phase 2 producing a reader that survives corruption at "
        "this palette size, which `decode_pixels` itself does not.",
        "",
        "| palette | subpatch | payload (B) | ceiling k²·log2(P) | clean bits/patch | "
        f"beats {BASE64_BITS_PER_TOKEN:.0f} clean? (Bar A) | clears {GATE_BITS_PER_PATCH:.0f} "
        f"clean? | worst-corruption bits/patch | worst corruption | clears "
        f"{GATE_BITS_PER_PATCH:.0f} corrupted? | clears gate (both, Bar B)? |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in gate_rows:
        lines.append(
            f"| {r['palette']} | {r['subpatch']} | {r['payload_size']} | {r['ceiling']} | "
            f"{r['clean']:.3f} | {'yes' if r['beats_base64_clean'] else 'no'} | "
            f"{'yes' if r['clears_clean'] else 'no'} | {r['worst']:.3f} | "
            f"{r['worst_name']} | {'yes' if r['clears_worst'] else 'no'} | "
            f"{'**YES**' if r['clears_both'] else 'no'} |"
        )

    lines += ["", "**Configs that clear the gate (both clean and worst-case corruption, Bar B):**", ""]
    if clearing:
        for r in clearing:
            lines.append(
                f"- palette={r['palette']}, subpatch={r['subpatch']}, "
                f"payload_size={r['payload_size']}B -- clean {r['clean']:.3f} bits/patch, "
                f"worst {r['worst']:.3f} bits/patch (worst corruption: `{r['worst_name']}`)"
            )
    else:
        lines.append("- none")

    beats_a_clean = [r for r in gate_rows if r["beats_base64_clean"]]
    lines += [
        "",
        "**Configs that beat the base64 density bar clean (Bar A -- may or may not survive "
        "corruption; see the worst-corruption columns in the table above and the \"Token "
        "crossover\" section for whether that matters for tokens too):**",
        "",
    ]
    if beats_a_clean:
        for r in beats_a_clean:
            lines.append(
                f"- palette={r['palette']}, subpatch={r['subpatch']}, "
                f"payload_size={r['payload_size']}B -- clean {r['clean']:.3f} bits/patch "
                f"(worst-corruption: {r['worst']:.3f}, `{r['worst_name']}`, "
                f"{'clears' if r['clears_worst'] else 'does NOT clear'} Bar A under that "
                "corruption)"
            )
    else:
        lines.append("- none")

    lines += ["", "**Verdict (derived from the tables above, not asserted independently):**", ""]
    if clearing and not any_subpatch1_clears and any_subpatch_gt1_clears:
        lines.append(
            "Every Gate #1 (Bar B) clearing config has `subpatch>1` -- the unverified "
            "pixel-decoder geometric ceiling regime. **No `subpatch=1` (VLM-meaningful) config "
            "clears Gate #1 at any tested payload size.** This is not just an unlucky "
            "corruption result: for `subpatch=1` the raw per-symbol ceiling is "
            "`log2(palette)`, which for the largest palette tested "
            f"({max_palette_present}) is only {max_subpatch1_ceiling} bits/patch -- already "
            f"below the {GATE_BITS_PER_PATCH:.0f}-bit Bar B *before* Reed-Solomon/calibration "
            "overhead is even subtracted. No amount of payload-size amortization can close "
            "that gap for `subpatch=1`; only the geometric `subpatch>1` regime can "
            "mathematically reach Bar B, and whether a real VLM can realize that regime is "
            "exactly the open question Phase 2 exists to answer. **Bar A tells a different "
            f"story, though:** {len(beats_a_clean)} config(s) beat the real economic bar "
            f"clean (see the list above){', including subpatch=1 configs' if any(r['subpatch'] == 1 for r in beats_a_clean) else ''} "
            "-- see the \"Token crossover\" section below for what that means in tokens, and "
            "the mandatory P=128/256 corruption caveat above for what it does not yet mean."
        )
    elif any_subpatch1_clears:
        lines.append(
            "At least one `subpatch=1` (VLM-meaningful) config clears Gate #1 (Bar B) both "
            "clean and under worst-case corruption -- see the list above. This is still a "
            "`decode_pixels` (model-free) measurement, not a VLM result; it says the "
            "channel/code can carry the bits, not that any model has been shown to read them."
        )
    else:
        lines.append(
            "No config -- `subpatch=1` or `subpatch>1` -- clears Gate #1 (Bar B) both clean "
            "and under worst-case corruption at the palettes/payload sizes tested here. "
            f"{len(beats_a_clean)} config(s) still beat Bar A (base64 density) clean -- see "
            "the list above and the \"Token crossover\" section below."
        )

    baseline_tokenizer_id = getattr(baseline, "tokenizer_id", None)
    baseline_source = (
        f"MEASURED baseline, tokenizer_id=`{baseline_tokenizer_id}`"
        if baseline_tokenizer_id is not None
        else "ANALYTIC assumption (no measured-tokenizer baseline file found in this run -- "
        "see `heliogram.baselines.load_measured_baseline`)"
    )
    lines += [
        "",
        "## Baselines",
        "",
        f"- **base64 in text context:** ~{baseline.bits_per_token:.1f} bits/token ({baseline.note}). "
        f"Source: {baseline_source} (see `_resolve_base64_baseline`). **Every Bar A "
        "('beats base64 clean?') verdict in the Headline table above and every "
        "`GATE_BITS_PER_PATCH`/`BASE64_BITS_PER_TOKEN` comparison anywhere in this file is "
        "computed directly against THIS number.** Bar C's `token_ratio`/`heliogram_cheaper` in "
        "the \"Token crossover\" section below now derives from the SAME resolved baseline: "
        "`base64_token_est = floor(ceil(payload/3)*4 / chars_per_token)` with `chars_per_token "
        f"= {BASE64_CHARS_PER_TOKEN:.4f}` in this run (1.0 exactly when the source above is "
        "ANALYTIC, reproducing the old pure-character count; the measured value when it is "
        "MEASURED -- floor-rounded because understating base64's token cost is the direction "
        "conservative AGAINST heliogram's claim). The old version of this section documented a "
        "Bar A/Bar C asymmetry here (Bar C stuck on the analytic ~1-char/token estimate even "
        "when a measurement existed); that asymmetry is now closed.",
        (
            "- **Other text encodings (is base64 even the right bar?):** "
            + (
                f"MEASURED ({text_baselines.tokenizer_id}, "
                f"{text_baselines.tokenizer_package}): "
                + ", ".join(
                    f"`{e.encoding}` {e.bits_per_token:.3f} bits/token "
                    f"({e.chars_per_token:.4f} chars/token)"
                    for e in sorted(
                        text_baselines.encodings.values(),
                        key=lambda e: e.bits_per_token,
                        reverse=True,
                    )
                )
                + f". The strongest (`{text_baselines.strongest.encoding}`) is the honest "
                "text-context economic bar; the Bar A verdicts above are computed against "
                "base64 and carry the qualifier stated there."
                if text_baselines is not None
                else "NOT MEASURED in this checkout (heliogram/data/text_baselines.json "
                "missing). base64 is the only measured text encoding, and it is NOT "
                "guaranteed to be the strongest on this tokenizer -- ascii85/base85 pack 8 "
                "bits into 1.25 chars vs base64's 1.33 before BPE effects. Until "
                "`python -m heliogram.baselines --measure` (transformers + HF Hub access "
                "required) is run, every Bar A verdict above should be read as 'beats "
                "base64', not 'beats text context'."
            )
        ),
        f"- **Rendered text (geometric, model-free):** {rendered.chars_per_patch:.2f} "
        f"chars/patch = {rendered.bits_per_patch:.2f} bits/patch typesetting a "
        f"{PAYLOAD_SIZE}-byte payload (base64'd, {rendered.text_len} chars) into "
        f"{rendered.patches_used} patches of the same {PATCH_SIZE}px grid unit. {rendered.note}",
        "",
        "See \"Token crossover\" immediately below for the actual benefit claim (total token "
        "COUNT for a full payload, not bits/patch density) -- beating the bits/patch bar above "
        "is necessary but not sufficient for that; overhead amortization differs between the "
        "two encodings.",
        "",
    ]

    lines += _token_crossover_section(results, payload_sizes_present, subpatches_present, palettes_present)

    lines += [
        "## Summary by sub-patch regime (payload-size amortization)",
        "",
        "Fixed per-message overhead (5-byte frame header + Reed-Solomon parity + the "
        "calibration row) is amortized over more data patches as payload size grows, so "
        "bits/patch should rise toward the `subpatch²·log2(palette)` ceiling as payload "
        "grows -- this is the amortization half of this sweep. 'corr(mean)' is the mean "
        "bits/patch over every non-clean corruption in the table below (resize 3%/5%, JPEG "
        "q95/85/70, crop/pad 2px, the model's own preprocessing qwen_smart_resize / "
        "qwen_smart_resize_1mp, and combined), each counted as 0 on a failed decode.",
        "",
    ]
    for subpatch in subpatches_present:
        label = (
            "VLM-meaningful: one symbol per patch"
            if subpatch == 1
            else "PIXEL-DECODER GEOMETRIC CEILING ONLY -- not a VLM capability claim, see caveat above"
        )
        lines += [f"### subpatch={subpatch} ({label})", ""]
        header_cells = ["Palette", "bits/sym", "ceiling"]
        for p in payload_sizes_present:
            header_cells += [f"{p}B clean", f"{p}B corr(mean)"]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "---|" * len(header_cells))
        for palette in palettes_present:
            bps = bits_per_symbol(palette)
            ceiling = subpatch * subpatch * bps
            row_cells = [str(palette), str(bps), str(ceiling)]
            for p in payload_sizes_present:
                s = summary.get((palette, subpatch, p))
                if s is None:
                    row_cells += ["n/a", "n/a"]
                else:
                    row_cells += [f"{s['clean']:.3f}", f"{s['corrupted_mean']:.3f}"]
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    lines += [
        "## Full breakdown by corruption",
        "",
        "| palette | subpatch | payload | bits/sym | corruption | symbol error rate | "
        "decode success rate | bits/patch |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.palette} | {r.subpatch} | {r.payload_size} | {r.bits_per_symbol} | "
            f"{r.corruption} | {r.symbol_error_rate:.4f} | {r.decode_success_rate:.2f} | "
            f"{r.bits_per_patch:.3f} |"
        )

    lines += [
        "",
        "## Self-consistency checks",
        "",
        "Three invariants must hold if these numbers mean what they claim to mean: (1) "
        "bits/patch (`payload_bytes*8/total_patches`, the TRUE PAYLOAD DENSITY -- see "
        "`_bits_per_patch_on_success`'s docstring) can never exceed `subpatch²·log2(palette)`, "
        "the raw per-DATA-PATCH channel CAPACITY for a subpatch x subpatch grid of symbols per "
        "patch, before calibration-row overhead, Reed-Solomon parity, and grid/bit padding are "
        "all accounted for (this generalizes the pre-Slice-B `<= log2(palette)` check, which is "
        "the `subpatch=1` case where `subpatch²=1`); (2) "
        "mean corrupted bits/patch can never exceed clean bits/patch for the same (palette, "
        "subpatch, payload_size), since corruption only ever removes information relative to "
        "the uncorrupted image; (3) [token crossover] every row's `base64_token_est` must "
        "equal `ceil(payload_size/3)*4` exactly and `token_ratio` must equal "
        "`total_patches/base64_token_est` exactly, independently recomputed here rather than "
        "just re-displaying the harness's own stored values -- if either drifts, the Token "
        "crossover section's numbers are wrong.",
        "",
        "| palette | subpatch | payload | ceiling subpatch²·log2(P) | clean bits/patch | "
        "<= ceiling? | corrupted(mean) bits/patch | <= clean? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    any_bug = False
    for key in sorted(summary):
        palette, subpatch, payload_size = key
        bps = bits_per_symbol(palette)
        ceiling = subpatch * subpatch * bps
        s = summary[key]
        clean_ok = s["clean"] <= ceiling + 1e-9
        corrupted_ok = s["corrupted_mean"] <= s["clean"] + 1e-9
        any_bug = any_bug or not clean_ok or not corrupted_ok
        lines.append(
            f"| {palette} | {subpatch} | {payload_size} | {ceiling} | {s['clean']:.3f} | "
            f"{'yes' if clean_ok else 'NO -- BUG'} | {s['corrupted_mean']:.3f} | "
            f"{'yes' if corrupted_ok else 'NO -- BUG'} |"
        )

    consistency_note = (
        "Invariants (1) and (2) hold for every (palette, subpatch, payload_size) bucket above."
        if not any_bug
        else "**At least one row above is flagged NO -- BUG; see that row -- this would be a "
        "measurement or codec bug, not an expected result.**"
    )

    token_bug_row = None
    for r in results:
        expected_b64 = _base64_token_estimate(r.payload_size)
        expected_ratio = (r.total_patches / expected_b64) if expected_b64 else 0.0
        if r.base64_token_est != expected_b64 or abs(r.token_ratio - expected_ratio) > 1e-6:
            token_bug_row = r
            break
    consistency_note += " " + (
        "Invariant (3) [token crossover] holds for every one of the "
        f"{len(results)} rows in this sweep: base64_token_est and token_ratio were "
        "independently recomputed from payload_size/total_patches for every row and matched "
        "the harness's own stored values exactly."
        if token_bug_row is None
        else "**Invariant (3) [token crossover] is flagged NO -- BUG at "
        f"palette={token_bug_row.palette}, subpatch={token_bug_row.subpatch}, "
        f"payload_size={token_bug_row.payload_size}B, corruption={token_bug_row.corruption} -- "
        f"stored base64_token_est={token_bug_row.base64_token_est}/token_ratio="
        f"{token_bug_row.token_ratio:.6f} did not match the independently recomputed values; "
        "this would be a measurement bug in _token_crossover, not an expected result.**"
    )

    non_clean = [r for r in results if r.corruption != "clean"]
    max_err_cell = max(results, key=lambda r: r.symbol_error_rate) if results else None
    min_success = min((r.decode_success_rate for r in non_clean), default=1.0)
    if min_success >= 1.0 - 1e-9:
        absorption_note = (
            "Within the realistic corruption envelope this harness applies (resize +-1-5%, "
            f"JPEG q70-95, slight crop/pad, the model's own smart_resize preprocessing, and "
            "their composition), decode_success_rate is "
            f"1.00 for every cell in this sweep -- Reed-Solomon (nsym={NSYM}) fully absorbs "
            "the symbol errors this envelope introduces at every tested (palette, subpatch, "
            "payload_size) combination, including the larger palettes/subpatch=2/bigger-"
            "payload cells this sweep adds beyond the original v0.1 4-palette/subpatch=1/"
            "48-byte sweep. That is a real result (this corruption envelope does not stress "
            "the channel's ECC margin for any tested config), not a stuck-at-1.0 measurement "
            "bug."
        )
    else:
        worst_success_cell = min(non_clean, key=lambda r: r.decode_success_rate)
        absorption_note = (
            "Within the realistic corruption envelope this harness applies, decode_success_rate "
            f"drops below 1.00 for at least one cell in this sweep (lowest observed: "
            f"{min_success:.2f}, at palette={worst_success_cell.palette}, "
            f"subpatch={worst_success_cell.subpatch}, "
            f"payload_size={worst_success_cell.payload_size}B, "
            f"corruption={worst_success_cell.corruption}) -- unlike the original v0.1 "
            f"4-palette/subpatch=1/48-byte sweep, where Reed-Solomon (nsym={NSYM}) fully "
            "absorbed every symbol error that same envelope introduced. See the full "
            "breakdown above for every cell where decode_success_rate < 1.00: this is the "
            "realistic corruption envelope actually biting at the larger palette/subpatch/"
            "payload_size combinations this sweep newly covers, not a measurement bug."
        )
    if max_err_cell is not None:
        consistency_note += (
            f" The largest observed symbol_error_rate across the whole sweep is "
            f"{max_err_cell.symbol_error_rate:.4f} (palette={max_err_cell.palette}, "
            f"subpatch={max_err_cell.subpatch}, payload_size={max_err_cell.payload_size}B, "
            f"corruption={max_err_cell.corruption}). {absorption_note}"
        )
    lines += ["", consistency_note]

    if stress_results:
        stress_palettes = sorted({r.palette for r in stress_results})
        lines += [
            "",
            "## Beyond the realistic envelope (diagnostic, single representative config)",
            "",
            "To confirm decode failure is actually reachable by this harness (i.e. that high "
            "success rates above are a real headroom finding and not a bug that can never "
            "observe failure), the same style of trial was re-run under corruption well "
            "outside the 'realistic serving pipeline' envelope: 50% bilinear resize "
            "round-trip, JPEG q10, a 6px crop/pad, and their composition. This diagnostic "
            f"suite runs at a single representative config -- subpatch=1, "
            f"payload_size={PAYLOAD_SIZE}B, {N_TRIALS} trials/cell (the module defaults) -- "
            f"across all {len(stress_palettes)} palettes; it is NOT swept across "
            "subpatch/payload_size the way the headline sweep above is, since its only "
            "purpose is to confirm the harness can observe decode failure at all.",
            "",
            "| palette | corruption | symbol error rate | decode success rate | bits/patch |",
            "|---|---|---|---|---|",
        ]
        for r in stress_results:
            lines.append(
                f"| {r.palette} | {r.corruption} | {r.symbol_error_rate:.4f} | "
                f"{r.decode_success_rate:.2f} | {r.bits_per_patch:.3f} |"
            )
        stress_min_success = min((r.decode_success_rate for r in stress_results), default=1.0)
        if stress_min_success < 1.0 - 1e-9:
            stress_note = (
                "Decode success drops well below 1.00 for at least one palette under this "
                f"diagnostic stress suite (lowest observed: {stress_min_success:.2f}), "
                "confirming the channel does have a real breaking point -- it simply lies "
                "beyond the resize/JPEG/crop ranges a typical serving pipeline applies, "
                "consistent with the realistic-envelope sweep above."
            )
        else:
            stress_note = (
                "Decode success did NOT drop below 1.00 for any palette under this diagnostic "
                "stress suite in this run -- unexpected; treat this as a signal to widen the "
                "stress corruptions rather than assume the channel has no breaking point."
            )
        lines += ["", stress_note]

    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    sweep_results = run(
        palettes=PALETTES,
        subpatches=SUBPATCHES,
        payload_sizes=SWEEP_PAYLOAD_SIZES,
        n_trials=SWEEP_N_TRIALS,
    )
    summary = _summary_rows(sweep_results)
    gate_rows = _gate_rows(summary)
    print("Headline: Gate #1 sweep (palette x subpatch x payload_size)")
    print(
        f"Gate #1 bar: {GATE_BITS_PER_PATCH:.1f} bits/patch, required both clean and under "
        "worst-case tested corruption.\n"
        "MANDATORY honesty caveat: subpatch=1 rows below are the only VLM-meaningful regime "
        "(one symbol per DATA patch, i.e. one symbol per vision token). subpatch>1 rows are a "
        "PIXEL-DECODER GEOMETRIC CEILING ONLY: decode_pixels/extract_symbols sample known, "
        "exact pixel coordinates off a grid they are told in advance, so a subpatch>1 row "
        "'clearing the gate' below is NOT a VLM capability claim -- whether a real ViT/VLM "
        "image encoder can resolve sub-patch structure at all is unverified, and doubtful. "
        "See RESULTS.md's Headline section for the full caveat."
    )
    print(format_gate_table(gate_rows))
    n_clear = sum(1 for r in gate_rows if r["clears_both"])
    n_clear_subpatch1 = sum(
        1 for r in gate_rows if r["clears_both"] and r["subpatch"] == 1
    )
    print(
        f"\n{n_clear}/{len(gate_rows)} configs clear the {GATE_BITS_PER_PATCH:.1f} "
        "bits/patch Gate #1 bar both clean and under worst-case tested corruption "
        f"({n_clear_subpatch1} of those at subpatch=1/VLM-meaningful; the remaining "
        f"{n_clear - n_clear_subpatch1} are subpatch>1 pixel-decoder-only -- see the "
        "MANDATORY caveat above, not a VLM capability claim)."
    )
    n_beats_a = sum(1 for r in gate_rows if r["beats_base64_clean"])
    print(
        f"{n_beats_a}/{len(gate_rows)} configs beat Bar A ({BASE64_BITS_PER_TOKEN:.1f} "
        "bits/patch base64 density, clean-only) -- this, not Gate #1 above, is the real "
        "economic bar; Gate #1 is a deliberate comfort margin above it. See RESULTS.md's "
        "Headline section."
    )

    crossover_rows = _token_crossover_rows(sweep_results)
    palettes_at_subpatch1 = sorted({row["palette"] for row in crossover_rows if row["subpatch"] == 1})
    print(
        "\nToken crossover, subpatch=1 (VLM-meaningful): total_patches vs base64_token_est "
        "(~1 token/patch vs. the resolved chars/token baseline) -- THE benefit claim, exact "
        "byte-granular crossing "
        "(see exact_crossover_payload_size), not interpolated between swept sample points. See "
        "RESULTS.md's \"Token crossover\" section for the full table incl. subpatch>1, the "
        "LM-token (2x2 merger) accounting, and jpeg_q70 decode success:"
    )
    for palette in palettes_at_subpatch1:
        result = exact_crossover_payload_size(palette, 1, NSYM, _base64_token_estimate)
        if result.crossing_bytes is not None:
            wobble = (
                f" -- WOBBLES, recrosses at {result.recrossing_bytes[0]}B, see RESULTS.md"
                if result.recrossing_bytes
                else ""
            )
            print(
                f"  palette={palette}: crosses at {result.crossing_bytes}B (exact, "
                f"token-cheaper than base64 from there on within the scanned range){wobble}"
            )
        else:
            print(
                f"  palette={palette}: no crossover found in the exact scan up to "
                f"{result.max_bytes_scanned}B (lowest token_ratio observed: "
                f"{result.lowest_ratio:.3f} at {result.lowest_ratio_bytes}B)"
            )
    print(
        "MANDATORY: token-cheaper is NOT the same as decodable -- see RESULTS.md's Token "
        "crossover section for the same cells' jpeg_q70 decode_success_rate shown side by "
        "side (palette=128/256 are measured to FAIL there even where they are token-cheaper "
        "clean). The LM-token (2x2 merger) accounting in RESULTS.md crosses at smaller payload "
        "sizes still but carries its own additional UNVERIFIED assumption -- see that "
        "subsection's caveat, not printed here."
    )

    stress_results = run(corruptions=STRESS_CORRUPTIONS)
    stress_table = format_table(stress_results)
    print(
        "\n(diagnostic) beyond the realistic corruption envelope, single representative "
        f"config (subpatch=1, payload_size={PAYLOAD_SIZE}B):"
    )
    print(stress_table)

    out_dir = Path.cwd()
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "RESULTS.md"
    write_csv(sweep_results, csv_path)
    write_results_md(sweep_results, md_path, stress_results=stress_results)
    print(f"\nwrote {csv_path} ({len(sweep_results)} rows) and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
