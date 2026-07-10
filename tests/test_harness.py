"""Pytest suite for heliogram.harness -- the CPU-only evaluation harness itself.

Review-fix context: prior to this file, heliogram/harness.py had ZERO test coverage (confirmed
by grepping tests/ for any real import of heliogram.harness before this diff -- only a comment
in tests/test_phase2_scaffold.py referenced it by name). In particular, the Slice-B
generalization of the Gate #1 "bits/patch <= ceiling" self-consistency check from
`log2(palette)` to `subpatch**2 * log2(palette)` (so it stops false-alarming for subpatch=k>1)
was verified only by eyeballing one manually-generated RESULTS.md/results.csv run, which is not
a regression gate: nobody re-runs `python -m heliogram.harness` and reads the output by eye on
every change. This file turns that eyeballing into real pytest assertions.

Everything below runs `heliogram.harness.run()`/`decode_pixels` (the model-free reference
decoder) at a deliberately tiny (palette, subpatch, payload_size, corruption) grid so it stays
fast -- nothing here is a VLM/Phase-2 result; see heliogram/codec.py's module docstring and
RESULTS.md's "Scope" note for the same data-honesty boundary this file respects.

Review-fix (round 2): `_token_crossover` (and its helpers `_grid_stats`/`_base64_token_estimate`/
`_crossover_payload_size`) is THE benefit metric this project's headline README/RESULTS.md
claims rest on (see harness.py's module docstring), but had zero unit tests against hand-computed
values before the "token crossover" section below -- a regression in any of those functions'
arithmetic (e.g. an off-by-one in `_grid_stats`' `data_patches_needed` ceil, a flipped comparison
in `_crossover_payload_size`'s interpolation, or `_run_cell` wiring `total_patches`/`token_ratio`
into the wrong `CellResult` field) would have silently changed the reported token-crossover
numbers with no test failing. That section closes it.

Review-fix (round 3, work group B -- F1 fix + token accounting): four more changes to
heliogram/harness.py, each with its own new/rewritten test coverage below:
  (B1) `_bits_per_patch_on_success` was found to overstate true payload density by crediting
       grid-width/bit padding as if it were payload (up to ~3x at small payloads) -- fixed to
       `payload_len*8/total_patches`. The old pinned-formula tests below are REWRITTEN (not
       just re-approved) against the new formula, including the three hand-computed sanity
       anchors the fix plan specifies by name.
  (B2) `BASE64_BITS_PER_TOKEN` now goes through `_resolve_base64_baseline()`, which prefers a
       measured tokenizer baseline (`heliogram.baselines.load_measured_baseline()`, added by a
       parallel work group and not present in this worktree) over the analytic 6.0 assumption,
       resolved defensively via `getattr` so this module works with or without that function
       existing. New tests below cover both the not-present (falls back to analytic) and
       monkeypatched-present (uses the measured value) paths.
  (B3) A second, alternative token accounting (`lm_tokens_2x2`/`lm_token_ratio`) for the pinned
       Phase-2 target's (Qwen2.5-VL) 2x2 vision-language spatial merger is added to
       `_GridStats`/`TokenCrossover`/`CellResult`. New tests below hand-verify it alongside the
       existing per-patch fields.
  (B4) `_crossover_payload_size`'s linear interpolation (spurious precision on a staircase
       function) is REPLACED by `exact_crossover_payload_size`, an exact byte-granular scan that
       also detects/reports recrossings ("wobble"). New tests below hand-verify a real crossing
       AND a real recrossing by computing `_grid_stats` at the straddling payload sizes.
"""

from __future__ import annotations

import base64
import math

import pytest

from heliogram import harness
from heliogram.codec import bits_per_symbol, compute_grid, rs_encoded_length
from heliogram.corruption import jpeg_compress, resize_roundtrip
from heliogram.harness import CellResult

# --- _bits_per_patch_on_success: capacity-math unit checks, independently recomputed ----------
#
# B1 fix (F1): _bits_per_patch_on_success used to compute subpatch^2*bits_per_symbol*
# (data_patches/total_patches)*(payload/ecc), which credited grid-width padding (compute_grid
# floors width at `palette` regardless of symbols needed) and bit padding (_pack_symbols
# zero-pads the ecc bitstream to a whole number of symbols) as if they carried payload bits --
# overstating TRUE payload density by up to ~3x at small payloads. The tests below are REWRITTEN
# (not just re-approved) against the new formula: TRUE PAYLOAD DENSITY = payload_len*8 /
# total_patches.


def test_bits_per_patch_on_success_matches_true_payload_density_formula():
    """payload_len*8 / total_patches -- cross-checked against an independent recomputation of
    total_patches (the same grid-math primitives _grid_stats' own tests below use), not just
    re-invoking the function under test against itself. Covers both subpatch=1 and subpatch=2 (k
    symbols packed per DATA patch before compute_grid sizes the grid)."""
    nsym = 32
    for palette in (8, 16):
        for subpatch in (1, 2):
            for payload_len in (48, 1024):
                got = harness._bits_per_patch_on_success(
                    payload_len, palette, nsym, subpatch=subpatch
                )

                bps = bits_per_symbol(palette)
                ecc_len = rs_encoded_length(5 + payload_len, nsym)
                num_symbols = math.ceil(ecc_len * 8 / bps)
                cells_per_patch = subpatch * subpatch
                data_patches_needed = math.ceil(num_symbols / cells_per_patch)
                width, height = compute_grid(data_patches_needed, palette)
                total_patches = width * height
                expected = payload_len * 8 / total_patches

                assert got == pytest.approx(expected)


def test_bits_per_patch_on_success_hand_computed_anchors_from_fix_plan():
    """The three exact sanity anchors the F1 fix plan (work group B, item B1) specifies by name,
    all at P=256/subpatch=1/nsym=32 -- hand arithmetic in each comment, independent of any helper
    in harness.py:"""
    nsym = 32
    # 4096B: total_patches=5120 (also independently verified by
    # test_grid_stats_matches_independent_recomputation below) -> 4096*8/5120 = 32768/5120 =
    # 6.4 exactly.
    assert harness._bits_per_patch_on_success(4096, 256, nsym, subpatch=1) == pytest.approx(6.4)
    # 16384B: total_patches=19200 -> 16384*8/19200 = 131072/19200 = 6.826666... (repeating 6).
    assert harness._bits_per_patch_on_success(16384, 256, nsym, subpatch=1) == pytest.approx(
        131072 / 19200
    )
    # 48B: total_patches=512 -> 48*8/512 = 384/512 = 0.75 exactly. This is the anchor the fix
    # plan calls out explicitly: the OLD (now-replaced) formula reported 2.259 bits/patch here --
    # a ~3.0x overstatement of the true payload density this fix corrects.
    assert harness._bits_per_patch_on_success(48, 256, nsym, subpatch=1) == pytest.approx(0.75)


def test_bits_per_patch_on_success_never_exceeds_geometric_ceiling():
    """The F1 fix must not break the invariant write_results_md's "Self-consistency checks"
    section relies on: TRUE payload density can never exceed the raw channel CAPACITY
    subpatch^2*log2(palette), because payload bits <= ecc bits <= raw symbol-channel bits <=
    data_patches*cells_per_patch*bps always (Reed-Solomon only ever ADDS parity, and packing only
    ever pads UP to a whole symbol), and total_patches > data_patches (the calibration row) --
    see _bits_per_patch_on_success's docstring for the full argument. Swept across every VALID
    palette/subpatch and several payload sizes, not just the anchors above."""
    nsym = 32
    for palette in (2, 4, 8, 16, 32, 64, 128, 256):
        for subpatch in (1, 2):
            ceiling = subpatch * subpatch * bits_per_symbol(palette)
            for payload_len in (1, 48, 1024, 4096, 16384):
                bpp = harness._bits_per_patch_on_success(payload_len, palette, nsym, subpatch)
                assert bpp <= ceiling + 1e-9, (
                    f"palette={palette}, subpatch={subpatch}, payload_len={payload_len}: "
                    f"{bpp} exceeded ceiling {ceiling}"
                )


# --- _grid_stats / _token_crossover: THE benefit metric, independently hand-computed ----------
#
# Review-fix: before this section, nothing in the test suite asserted total_patches/
# base64_token_est/token_ratio/heliogram_cheaper (on CellResult or via _token_crossover/
# _grid_stats directly) against any independently-derived number -- see this file's module
# docstring. Every expected value below is computed from first principles using the SAME
# primitives _bits_per_patch_on_success's tests above already use (bits_per_symbol,
# rs_encoded_length, compute_grid) plus Python's own `base64` module for the token side, never by
# calling `_grid_stats`/`_token_crossover` and comparing the result to itself.


def _manual_grid_stats(payload_len: int, palette: int, nsym: int, subpatch: int = 1):
    """Independent re-derivation of encode()'s grid math -- mirrors harness._grid_stats'
    docstring exactly, but written fresh here rather than imported, so a bug introduced into
    _grid_stats itself has an independent computation to be caught against. Also independently
    re-derives `lm_tokens_2x2` (B3: the Qwen2.5-VL 2x2 spatial-merger LM-visible token count,
    `ceil(width/2)*ceil(height/2)`), not just the pre-B3 total_patches/data_patches pair."""
    bps = bits_per_symbol(palette)
    ecc_len = rs_encoded_length(5 + payload_len, nsym)
    num_symbols = math.ceil(ecc_len * 8 / bps)
    cells_per_patch = subpatch * subpatch
    data_patches_needed = math.ceil(num_symbols / cells_per_patch)
    width, height = compute_grid(data_patches_needed, palette)
    total_patches = width * height
    data_patches = width * (height - 1)
    lm_tokens_2x2 = math.ceil(width / 2) * math.ceil(height / 2)
    return total_patches, data_patches, lm_tokens_2x2


@pytest.mark.parametrize(
    "payload_len,palette,nsym,subpatch",
    [
        (48, 8, 32, 1),
        (1024, 256, 32, 1),
        (4096, 256, 32, 1),
        (16384, 256, 32, 1),
        (48, 8, 32, 2),
    ],
)
def test_grid_stats_matches_independent_recomputation(payload_len, palette, nsym, subpatch):
    """_grid_stats' total_patches/data_patches/lm_tokens_2x2 (the geometry bits/patch AND both
    token-crossover accountings are built on) must match a hand-rederived grid size, not just be
    internally consistent with itself. Includes the 16384B/palette=256 case
    (total_patches=19200) that is also one of the B1 hand-computed anchors
    (test_bits_per_patch_on_success_hand_computed_anchors_from_fix_plan above)."""
    g = harness._grid_stats(payload_len, palette, nsym, subpatch)
    expected_total, expected_data, expected_lm = _manual_grid_stats(
        payload_len, palette, nsym, subpatch
    )
    assert g.total_patches == expected_total
    assert g.data_patches == expected_data
    assert g.lm_tokens_2x2 == expected_lm


def test_base64_token_estimate_matches_standard_formula_and_real_base64():
    """_base64_token_estimate must equal ceil(n/3)*4 -- the standard base64-with-padding
    expansion -- AND match Python's own base64.b64encode length for real payloads, not just an
    internally-consistent restatement of the same ceil formula. Unaffected by B1-B4 (B4's fix
    plan explicitly notes this side of the crossover ratio was already closed-form/exact)."""
    for n in (0, 1, 2, 3, 4, 5, 6, 48, 1024, 4096, 16384):
        got = harness._base64_token_estimate(n)
        assert got == math.ceil(n / 3) * 4
        assert got == len(base64.b64encode(bytes(n)))


# Hand-computed (palette, subpatch, payload_len, nsym) -> (total_patches, base64_token_est,
# token_ratio, heliogram_cheaper, lm_tokens_2x2, lm_token_ratio, lm_cheaper), independently
# re-derived via _manual_grid_stats/base64 above and cross-checked against results.csv's own
# stored rows for the same cells where present (see each case's comment). Deliberately includes
# cases on each side of BOTH the per-patch 1.0 crossover (palette=256 at 1024B is NOT cheaper
# per-patch; at 4096B it IS) and the LM-token 1.0 crossover (palette=256 at 1024B IS already
# cheaper under the 2x2-merger accounting despite NOT being cheaper per-patch -- see B3's
# docstring in harness.py for why the two accountings can disagree) so every boolean flag is
# exercised both ways, not just checked for "some value".
_TOKEN_CROSSOVER_CASES = [
    # matches results.csv: palette=8,subpatch=1,payload_size=48,corruption=clean.
    # lm_tokens_2x2=64 == base64_token_est=64 here (a coincidence at this exact payload size) ->
    # lm_token_ratio=1.0 exactly, which is NOT < 1.0, so lm_cheaper is False (same as per-patch).
    (48, 8, 32, 1, 256, 64, 4.0, False, 64, 1.0, False),
    # matches results.csv: palette=256,subpatch=1,payload_size=1024 (any corruption -- token
    # fields don't vary by corruption, see CellResult's docstring). Per-patch NOT cheaper
    # (1536/1368>1) but LM-token accounting IS cheaper here (384/1368<1) -- the two accountings
    # disagree at this payload size, exactly the case B3 exists to surface.
    (1024, 256, 32, 1, 1536, 1368, 1536 / 1368, False, 384, 384 / 1368, True),
    # matches results.csv: palette=256,subpatch=1,payload_size=4096 -- the per-patch crossover
    # case; also cheaper under LM-token accounting (unsurprising, since it is a strict 4x
    # reduction in numerator relative to the per-patch case at the same payload).
    (4096, 256, 32, 1, 5120, 5464, 5120 / 5464, True, 1280, 1280 / 5464, True),
    # matches results.csv: palette=8,subpatch=2,payload_size=48. Per-patch NOT cheaper (72/64>1)
    # but LM-token accounting IS cheaper (20/64<1) -- another disagreement case, at subpatch=2
    # (so this is ALSO in the pixel-decoder-geometric-ceiling-only regime; both caveats stack).
    (48, 8, 32, 2, 72, 64, 72 / 64, False, 20, 20 / 64, True),
]


@pytest.mark.parametrize(
    "payload_len,palette,nsym,subpatch,exp_total,exp_b64,exp_ratio,exp_cheaper,"
    "exp_lm_tokens,exp_lm_ratio,exp_lm_cheaper",
    _TOKEN_CROSSOVER_CASES,
)
def test_token_crossover_matches_hand_computed_values(
    payload_len, palette, nsym, subpatch, exp_total, exp_b64, exp_ratio, exp_cheaper,
    exp_lm_tokens, exp_lm_ratio, exp_lm_cheaper,
):
    """The regression guard this review round flags as entirely missing (see this file's module
    docstring): pins _token_crossover's output against hand-computed total_patches/
    base64_token_est/token_ratio/heliogram_cheaper AND lm_tokens_2x2/lm_token_ratio (B3) for
    concrete (payload_len, palette, nsym, subpatch) tuples that also match results.csv's own
    stored rows for the same cells (see each case's comment above) -- so a future change to the
    data_patches_needed ceil, the base64 formula, the 2x2-merger token count, or either
    ratio/threshold computation would be caught here even if results.csv were never
    regenerated."""
    tc = harness._token_crossover(payload_len, palette, nsym, subpatch)
    assert tc.total_patches == exp_total
    assert tc.base64_token_est == exp_b64
    assert tc.token_ratio == pytest.approx(exp_ratio)
    assert tc.heliogram_cheaper is exp_cheaper
    assert tc.heliogram_cheaper == (tc.token_ratio < 1.0)  # the flag IS the threshold, nothing else
    assert tc.lm_tokens_2x2 == exp_lm_tokens
    assert tc.lm_token_ratio == pytest.approx(exp_lm_ratio)
    assert (tc.lm_token_ratio < 1.0) is exp_lm_cheaper


# --- exact_crossover_payload_size (B4): exact byte-granular scan, replaces the old linear-------
# --- interpolation `_crossover_payload_size`. Hand-verified against _grid_stats at the exact ---
# --- straddling payload sizes, including a real recrossing ("wobble") case. --------------------


def test_exact_crossover_payload_size_synthetic_toy_cases(monkeypatch):
    """A small synthetic (not real grid-math) sanity check of the three basic outcomes, using a
    fake token_estimator so the crossing points are exact round numbers: (1) a clean single
    crossing with no recrossing; (2) a ratio that never drops below 1.0 (crossing_bytes is None);
    (3) lowest_ratio/lowest_ratio_bytes are populated even when there is no crossing."""
    # numerator is constant 100; token_estimator(n) = n, so ratio = 100/n, crossing at n=101
    # (100/101 < 1.0 for the first time), never recrosses (ratio only decreases as n grows).
    fake_grid_numerator = 100

    class _FakeStats:
        def __init__(self, n):
            self.total_patches = fake_grid_numerator
            self.lm_tokens_2x2 = fake_grid_numerator

    monkeypatch.setattr(
        harness, "_grid_stats", lambda payload_len, palette, nsym, subpatch=1: _FakeStats(payload_len)
    )

    result = harness.exact_crossover_payload_size(
        palette=8, subpatch=1, nsym=32, token_estimator=lambda n: n, max_bytes=200
    )
    assert result.crossing_bytes == 101
    assert result.recrossing_bytes == []
    assert result.lowest_ratio == pytest.approx(100 / 200)
    assert result.lowest_ratio_bytes == 200

    # token_estimator always returns 1 -> ratio is a constant 100/1=100, never drops below 1.0.
    never = harness.exact_crossover_payload_size(
        palette=8, subpatch=1, nsym=32, token_estimator=lambda n: 1, max_bytes=50
    )
    assert never.crossing_bytes is None
    assert never.lowest_ratio == pytest.approx(100.0)
    assert never.lowest_ratio_bytes == 1  # ratio is constant here, so the first byte wins


def test_exact_crossover_payload_size_real_grid_crossing_and_recrossing_hand_verified():
    """Real (not synthetic) grid math: palette=256/subpatch=1's per-patch crossing, hand-verified
    by computing `_grid_stats` directly at the two straddling payload sizes (as the fix plan
    requires), including the exact recrossing ("wobble") this staircase ratio actually exhibits
    -- the old linear-interpolation `_crossover_payload_size` could never have reported this,
    since it only ever sampled the swept (48, 1024, 4096, 16384)B points.

    Hand computation (bps=log2(256)=8, nsym=32, message_len=5+n):
      n=1536: ecc_len=rs_encoded_length(1541,32)=1541+32=1573 (fits one 223-byte RS chunk since
        1541<=223? -- no, chunked; verified via rs_encoded_length directly, not re-derived by
        hand here) -> _grid_stats(1536,256,32,1).total_patches=256*8=2048 (width floored at
        palette=256; empirically height=8). base64_token_est=ceil(1536/3)*4=2048. ratio=2048/2048
        = 1.0 exactly -- NOT < 1.0, so 1536B is NOT yet a crossing.
      n=1537: total_patches unchanged at 2048 (still fits the same grid), base64_token_est=
        ceil(1537/3)*4=2052. ratio=2048/2052<1.0 -- so 1537B IS the first payload size where the
        ratio drops below 1.0: crossing_bytes=1537.
      n=1556/1557: total_patches jumps from 2048 to 2304 at n=1557 (one more data row needed),
        while base64_token_est only grows to 2076 -- ratio jumps from 2048/2076<1.0 back up to
        2304/2076>1.0, i.e. a genuine recrossing at 1557B.
    """
    n_sym = 32
    g_1536 = harness._grid_stats(1536, 256, n_sym, 1)
    g_1537 = harness._grid_stats(1537, 256, n_sym, 1)
    g_1556 = harness._grid_stats(1556, 256, n_sym, 1)
    g_1557 = harness._grid_stats(1557, 256, n_sym, 1)

    assert g_1536.total_patches / harness._base64_token_estimate(1536) == pytest.approx(1.0)
    assert g_1537.total_patches / harness._base64_token_estimate(1537) < 1.0
    assert g_1556.total_patches / harness._base64_token_estimate(1556) < 1.0
    assert g_1557.total_patches / harness._base64_token_estimate(1557) >= 1.0
    assert g_1557.total_patches > g_1556.total_patches  # the grid-size jump that causes the wobble

    result = harness.exact_crossover_payload_size(
        palette=256, subpatch=1, nsym=n_sym, token_estimator=harness._base64_token_estimate,
        lm_tokens=False, max_bytes=2000,
    )
    assert result.crossing_bytes == 1537
    assert result.recrossing_bytes  # non-empty: this ratio really does wobble
    assert result.recrossing_bytes[0] == 1557


def test_exact_crossover_payload_size_lm_tokens_flag_switches_numerator_hand_verified():
    """`lm_tokens=True` must switch the scan's numerator from `total_patches` to
    `lm_tokens_2x2`, hand-verified at the exact straddling payload sizes for
    palette=256/subpatch=1 (n=96: lm_tokens_2x2=128, base64_token_est=128, ratio=1.0 exactly, NOT
    a crossing; n=97: lm_tokens_2x2 unchanged at 128, base64_token_est=132, ratio<1.0 -- the
    first crossing). This is a MUCH smaller crossing point than the per-patch case
    (1537B, see the test above) -- expected, since lm_tokens_2x2 is ~4x smaller than
    total_patches for the same grid."""
    n_sym = 32
    g_96 = harness._grid_stats(96, 256, n_sym, 1)
    g_97 = harness._grid_stats(97, 256, n_sym, 1)
    assert g_96.lm_tokens_2x2 / harness._base64_token_estimate(96) == pytest.approx(1.0)
    assert g_97.lm_tokens_2x2 / harness._base64_token_estimate(97) < 1.0

    result = harness.exact_crossover_payload_size(
        palette=256, subpatch=1, nsym=n_sym, token_estimator=harness._base64_token_estimate,
        lm_tokens=True, max_bytes=500,
    )
    assert result.crossing_bytes == 97


def test_run_cell_wires_token_crossover_fields_onto_cell_result():
    """Guards the WIRING, not just the pure function: _run_cell must actually copy
    _token_crossover's output onto the CellResult it returns (total_patches/base64_token_est/
    token_ratio/heliogram_cheaper AND lm_tokens_2x2/lm_token_ratio, B3), not just compute it and
    drop it -- a real (tiny) run through _run_cell, checked against the same hand-computed values
    test_token_crossover_matches_hand_computed_values pins for the pure function, plus
    results.csv's own stored row for this exact cell (palette=256, subpatch=1,
    payload_size=4096, corruption=clean: total_patches=5120, base64_token_est=5464,
    token_ratio=0.937042..., lm_tokens_2x2=1280, lm_token_ratio=0.234261...)."""
    result = harness._run_cell(
        palette=256,
        corruption_name="clean",
        corruption_fn=harness.CORRUPTIONS["clean"],
        n_trials=1,
        payload_size=4096,
        subpatch=1,
    )
    assert result.total_patches == 5120
    assert result.base64_token_est == 5464
    assert result.token_ratio == pytest.approx(5120 / 5464)
    assert result.heliogram_cheaper is True
    assert result.lm_tokens_2x2 == 1280
    assert result.lm_token_ratio == pytest.approx(1280 / 5464)
    assert result.decode_success_rate == 1.0  # clean corruption must always succeed


def test_write_csv_includes_lm_token_columns(tmp_path):
    """B3's CSV deliverable: write_csv's header and rows must carry lm_tokens_2x2/lm_token_ratio
    alongside the pre-existing token-crossover columns, not just CellResult/RESULTS.md."""
    result = harness._run_cell(
        palette=256,
        corruption_name="clean",
        corruption_fn=harness.CORRUPTIONS["clean"],
        n_trials=1,
        payload_size=4096,
        subpatch=1,
    )
    out_path = tmp_path / "results.csv"
    harness.write_csv([result], out_path)
    text = out_path.read_text()
    header, row, *_ = text.splitlines()
    assert "lm_tokens_2x2" in header
    assert "lm_token_ratio" in header
    assert ",1280," in row  # lm_tokens_2x2 for this exact cell (see B1/B3 hand-computed anchors)


# --- _resolve_base64_baseline (B2): measured-baseline hook, defensive import -------------------
#
# heliogram.baselines.load_measured_baseline is being added to heliogram/baselines.py by a
# PARALLEL work group and is NOT present in this worktree's checkout (confirmed by the first test
# below) -- see harness._resolve_base64_baseline's docstring for why it is looked up via getattr
# rather than a hard `from .baselines import load_measured_baseline`. These tests cover every
# path through the resolver: (a) genuinely absent (today's reality in this worktree), falls back
# to the analytic base64_bits_per_token(); (b) present but returns None (its own documented
# "no measurement file found" case), also falls back; (c) present and returns a real object,
# resolver must return THAT object unchanged -- the actual "hook" this review item asks for.


def test_resolve_base64_baseline_falls_back_to_analytic_when_load_measured_baseline_absent():
    """Today's reality in this worktree: heliogram.baselines has no load_measured_baseline
    attribute at all (the parallel work group's code is not present here). Confirmed directly,
    then _resolve_base64_baseline must fall back to the always-available analytic
    base64_bits_per_token() (6.0 bits/token) rather than raising AttributeError/ImportError."""
    assert not hasattr(harness._baselines_module, "load_measured_baseline")
    result = harness._resolve_base64_baseline()
    assert result.bits_per_token == pytest.approx(6.0)
    assert not hasattr(result, "tokenizer_id")  # the analytic Base64Baseline has no such field


def test_resolve_base64_baseline_falls_back_when_loader_present_but_returns_none(monkeypatch):
    """Even if heliogram.baselines.load_measured_baseline EXISTS, it returning None (its own
    documented behavior when no measurement file is found on this machine) must still fall back
    to the analytic baseline, not propagate None to BASE64_BITS_PER_TOKEN/write_results_md."""
    monkeypatch.setattr(
        harness._baselines_module, "load_measured_baseline", lambda: None, raising=False
    )
    result = harness._resolve_base64_baseline()
    assert result.bits_per_token == pytest.approx(6.0)


def test_resolve_base64_baseline_uses_measured_object_when_loader_present(monkeypatch):
    """THE hook this review item exists to guard: monkeypatch load_measured_baseline onto
    heliogram.baselines (simulating the parallel work group's function actually landing) to
    return a fake MEASURED baseline object, and assert _resolve_base64_baseline returns THAT
    object -- bits_per_token, tokenizer_id, and all -- not the analytic fallback."""

    class _FakeMeasuredBaseline:
        bits_per_token = 5.3
        chars_per_token = 1.13
        tokenizer_id = "fake/tokenizer-for-test"
        note = "measured: fake tokenizer for this unit test only"

    fake = _FakeMeasuredBaseline()
    monkeypatch.setattr(
        harness._baselines_module, "load_measured_baseline", lambda: fake, raising=False
    )

    result = harness._resolve_base64_baseline()
    assert result is fake
    assert result.bits_per_token == pytest.approx(5.3)
    assert result.tokenizer_id == "fake/tokenizer-for-test"


def test_write_results_md_threads_measured_baseline_note_and_tokenizer_id(monkeypatch, tmp_path):
    """B2's actual deliverable: write_results_md's 'Baselines' section must state WHICH baseline
    (measured tokenizer vs analytic assumption) every Bar A/Bar C verdict was computed against.
    Monkeypatches harness._resolve_base64_baseline itself (the resolver write_results_md calls
    internally to build that section) to a fake measured object, and asserts the resolved
    object's note/tokenizer_id/bits_per_token actually appear in the generated file -- not just
    that the (separately tested, above) resolver function returns the right thing in isolation."""

    class _FakeMeasuredBaseline:
        bits_per_token = 5.3
        chars_per_token = 1.13
        tokenizer_id = "fake/tokenizer-for-test"
        note = "measured: fake tokenizer for this unit test only, 9999 tokens for 4096 bytes"

    monkeypatch.setattr(harness, "_resolve_base64_baseline", lambda: _FakeMeasuredBaseline())

    results = harness.run(
        palettes=(8,), corruptions=_tiny_real_corruptions(), n_trials=1,
        subpatches=(1,), payload_sizes=(48,),
    )
    out_path = tmp_path / "RESULTS.md"
    harness.write_results_md(results, out_path)
    text = out_path.read_text()

    assert "fake/tokenizer-for-test" in text
    assert "MEASURED baseline" in text
    assert "5.3" in text  # bits_per_token rendered into the Baselines bullet and Bar A text


# --- _provenance_line / write_results_md's Provenance stamp (B5) -------------------------------


def test_provenance_line_contains_python_and_key_dependency_versions():
    """_provenance_line must report Python's own version plus numpy/Pillow/reedsolo versions
    (via importlib.metadata, reading installed dist-info -- not each module's own __version__
    attribute, which some of these packages don't reliably expose, see the function's docstring)
    and platform.platform(), each independently re-derived here rather than just re-invoking the
    function under test against itself."""
    import importlib.metadata
    import platform

    line = harness._provenance_line()
    assert platform.python_version() in line
    assert importlib.metadata.version("numpy") in line
    assert importlib.metadata.version("Pillow") in line
    assert importlib.metadata.version("reedsolo") in line
    assert platform.platform() in line


def test_write_results_md_contains_provenance_line(tmp_path):
    """B5's actual deliverable: RESULTS.md must record a Provenance line near the top so a
    committed artifact stays diagnosable if numpy/Pillow/reedsolo drift later (see
    tests/test_roundtrip.py's Pillow-12.3-pinned-hash known failure for why this matters in
    practice) -- exercised through the real write_results_md entrypoint, not just
    _provenance_line() in isolation."""
    results = harness.run(
        palettes=(8,), corruptions=_tiny_real_corruptions(), n_trials=1,
        subpatches=(1,), payload_sizes=(48,),
    )
    out_path = tmp_path / "RESULTS.md"
    harness.write_results_md(results, out_path)
    text = out_path.read_text()
    assert "**Provenance:**" in text
    assert harness._provenance_line() in text


# --- write_results_md: the B3 LM-token subsection and B4 exact-scan subsection must survive ----
# --- in the file they target, same as the pre-existing MANDATORY subpatch>1 caveat below -------


def test_write_results_md_contains_lm_token_subsection_and_mandatory_caveat(tmp_path):
    """B3's dedicated LM-token accounting subsection (and its own MANDATORY, UNVERIFIED caveat)
    and B4's exact-scan crossover subsection must both actually appear in the generated file --
    not just exist as source-level building blocks in harness.py."""
    results = harness.run(
        palettes=(8,), corruptions=_tiny_real_corruptions(), n_trials=1,
        subpatches=(1,), payload_sizes=(48,),
    )
    out_path = tmp_path / "RESULTS.md"
    harness.write_results_md(results, out_path)
    text = out_path.read_text()
    assert "LM-token accounting (Qwen2.5-VL 2x2 spatial merger)" in text
    assert "UNVERIFIED" in text
    assert "lm_tokens_2x2" in text
    assert "exact scan" in text  # B4: the exact-crossover subsection heading


# --- _gate_rows: the ceiling formula Finding #3 specifically flags as untested ----------------


@pytest.mark.parametrize("subpatch", [1, 2])
@pytest.mark.parametrize("palette", [2, 4, 8, 16, 32, 64])
def test_gate_rows_ceiling_is_subpatch_squared_times_log2_palette(palette, subpatch):
    """This is the exact invariant the review flags: _gate_rows' `ceiling = subpatch * subpatch
    * bps` and write_results_md's self-consistency table independently recompute the same
    formula. If a future change desyncs them (e.g. reverting one to plain `bps` while leaving the
    other generalized), every subpatch=2 row would again false-alarm 'clean > ceiling -- NO --
    BUG' exactly like the pre-fix bug this diff resolved -- and this test, not just a manually
    re-run RESULTS.md, must catch it."""
    summary = {
        (palette, subpatch, 48): {
            "clean": 0.0,
            "corrupted_mean": 0.0,
            "corrupted_worst": 0.0,
            "corrupted_worst_name": "n/a",
        }
    }
    [row] = harness._gate_rows(summary)
    assert row["ceiling"] == subpatch * subpatch * bits_per_symbol(palette)


def test_gate_rows_clears_both_requires_clean_and_worst_over_bar():
    bar = harness.GATE_BITS_PER_PATCH
    summary = {
        (8, 2, 48): {  # worst dips just below the bar -- must NOT clear
            "clean": bar + 1.0,
            "corrupted_mean": bar + 1.0,
            "corrupted_worst": bar - 1.0,
            "corrupted_worst_name": "jpeg_q70",
        },
        (8, 2, 1024): {  # both clean and worst clear -- must clear
            "clean": bar + 1.0,
            "corrupted_mean": bar + 1.0,
            "corrupted_worst": bar + 0.5,
            "corrupted_worst_name": "resize_3pct",
        },
    }
    rows = {row["payload_size"]: row for row in harness._gate_rows(summary)}
    assert rows[48]["clears_clean"] is True
    assert rows[48]["clears_worst"] is False
    assert rows[48]["clears_both"] is False
    assert rows[1024]["clears_clean"] is True
    assert rows[1024]["clears_worst"] is True
    assert rows[1024]["clears_both"] is True


# --- _summary_rows: worst-case (min) selection, independent of _gate_rows ---------------------


def test_summary_rows_picks_minimum_as_worst_not_mean():
    """The headline gate needs the worst (minimum) non-clean bits/patch, not the mean --
    verify _summary_rows actually returns the min and its name, distinct from the mean it also
    reports."""
    results = [
        CellResult(
            palette=8, subpatch=1, payload_size=48, corruption="clean",
            bits_per_symbol=3, symbol_error_rate=0.0, decode_success_rate=1.0,
            bits_per_patch=3.0, trials=1,
        ),
        CellResult(
            palette=8, subpatch=1, payload_size=48, corruption="resize_3pct",
            bits_per_symbol=3, symbol_error_rate=0.0, decode_success_rate=1.0,
            bits_per_patch=2.5, trials=1,
        ),
        CellResult(
            palette=8, subpatch=1, payload_size=48, corruption="jpeg_q70",
            bits_per_symbol=3, symbol_error_rate=0.1, decode_success_rate=0.0,
            bits_per_patch=0.0, trials=1,
        ),
    ]
    summary = harness._summary_rows(results)
    s = summary[(8, 1, 48)]
    assert s["clean"] == pytest.approx(3.0)
    assert s["corrupted_mean"] == pytest.approx((2.5 + 0.0) / 2)
    assert s["corrupted_worst"] == pytest.approx(0.0)
    assert s["corrupted_worst_name"] == "jpeg_q70"


# --- integration: real (tiny) run() must satisfy both self-consistency invariants as real -----
# --- pytest assertions, not just eyeballed markdown --------------------------------------------


def _tiny_real_corruptions():
    # Real corruption primitives (heliogram.corruption), not synthetic pixel shifts -- mirrors
    # (a small subset of) heliogram.harness.CORRUPTIONS' realistic envelope.
    return {
        "clean": lambda img: img,
        "resize_3pct": lambda img: resize_roundtrip(img, scale=0.97),
        "jpeg_q85": lambda img: jpeg_compress(img, quality=85),
    }


def test_run_satisfies_ceiling_and_corruption_monotonicity_invariants():
    """The two invariants write_results_md's 'Self-consistency checks' section computes and
    prints for a human to eyeball: (1) clean bits/patch never exceeds subpatch^2*log2(palette);
    (2) mean corrupted bits/patch never exceeds clean bits/patch. Asserted here as real pytest
    checks over an actual (tiny, fast) run() sweep spanning both subpatch=1 and subpatch=2."""
    results = harness.run(
        palettes=(8,),
        corruptions=_tiny_real_corruptions(),
        n_trials=1,
        subpatches=(1, 2),
        payload_sizes=(48,),
    )
    summary = harness._summary_rows(results)
    gate_rows = harness._gate_rows(summary)

    assert {row["subpatch"] for row in gate_rows} == {1, 2}  # both regimes actually ran

    for row in gate_rows:
        palette, subpatch = row["palette"], row["subpatch"]
        assert row["ceiling"] == subpatch * subpatch * bits_per_symbol(palette)
        assert row["clean"] <= row["ceiling"] + 1e-9, (
            f"clean bits/patch exceeded the geometric ceiling for subpatch={subpatch} -- "
            "this would be a real codec/measurement bug, not an expected result"
        )

    for key, s in summary.items():
        assert s["corrupted_mean"] <= s["clean"] + 1e-9, (
            f"corrupted-mean bits/patch exceeded clean bits/patch for {key} -- corruption "
            "should only ever remove information relative to the uncorrupted image"
        )


# --- write_results_md: the MANDATORY subpatch>1 caveat must survive in the file it targets ----


def test_write_results_md_contains_mandatory_subpatch_caveat(tmp_path):
    results = harness.run(
        palettes=(8,),
        corruptions=_tiny_real_corruptions(),
        n_trials=1,
        subpatches=(1, 2),
        payload_sizes=(48,),
    )
    out_path = tmp_path / "RESULTS.md"
    harness.write_results_md(results, out_path)
    text = out_path.read_text()
    assert "MANDATORY" in text
    assert "PIXEL-DECODER GEOMETRIC CEILING ONLY" in text
    assert "not a capability claim" in text


# --- main(): Finding #4 fix -- the stdout transcript must carry the caveat too, not just the ---
# --- generated markdown files -------------------------------------------------------------------


def test_main_stdout_carries_the_subpatch_caveat_not_just_the_markdown_files(
    tmp_path, monkeypatch, capsys
):
    """Before this fix, `python -m heliogram.harness`'s printed Gate #1 table and
    "N/48 configs clear the gate" verdict carried zero mention of the MANDATORY subpatch>1
    caveat that RESULTS.md/README.md carry right next to the same numbers -- so pasting/
    screenshotting the console transcript reproduced exactly the failure mode the project's
    docstrings call out: a subpatch>1 (pixel-decoder-only) number shown as "clears the gate"
    with the VLM-capability caveat stripped out. This exercises the REAL main() entrypoint (not
    just format_gate_table()/write_results_md() in isolation) against a monkeypatched tiny sweep
    so it runs fast in CI, redirected to tmp_path so it never touches the repo's real
    results.csv/RESULTS.md.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(harness, "PALETTES", (8,))
    monkeypatch.setattr(harness, "SUBPATCHES", (1, 2))
    monkeypatch.setattr(harness, "SWEEP_PAYLOAD_SIZES", (48,))
    monkeypatch.setattr(harness, "SWEEP_N_TRIALS", 1)
    monkeypatch.setattr(harness, "CORRUPTIONS", _tiny_real_corruptions())
    monkeypatch.setattr(
        harness,
        "STRESS_CORRUPTIONS",
        {"stress_resize_50pct": lambda img: resize_roundtrip(img, scale=0.5)},
    )

    rc = harness.main()
    assert rc == 0

    out = capsys.readouterr().out
    assert (tmp_path / "results.csv").exists()  # never wrote into the real repo root
    assert (tmp_path / "RESULTS.md").exists()
    assert "MANDATORY" in out
    assert "PIXEL-DECODER" in out
    assert "VLM" in out
    assert "not a VLM capability claim" in out
