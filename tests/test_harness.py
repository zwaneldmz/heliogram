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


def test_bits_per_patch_on_success_subpatch1_matches_manual_formula():
    """subpatch=1 must reproduce the pre-Slice-B formula exactly: bits_per_symbol *
    (data_patches/total_patches) * (payload/ecc) -- cross-checked against an independent
    recomputation, not just re-invoking the function under test against itself."""
    nsym = 32
    for palette in (8, 16):
        for payload_len in (48, 1024):
            got = harness._bits_per_patch_on_success(payload_len, palette, nsym, subpatch=1)

            bps = bits_per_symbol(palette)
            ecc_len = rs_encoded_length(5 + payload_len, nsym)
            num_symbols = math.ceil(ecc_len * 8 / bps)
            width, height = compute_grid(num_symbols, palette)
            data_patches = width * (height - 1)
            total_patches = width * height
            expected = bps * (data_patches / total_patches) * (payload_len / ecc_len)

            assert got == pytest.approx(expected)


def test_bits_per_patch_on_success_subpatch2_scales_by_cells_per_patch():
    """subpatch=2 packs k*k=4 symbols/data-patch before compute_grid sizes the grid -- the exact
    generalization this review targets. Cross-checked against an independent recomputation of
    the same subpatch-aware formula (mirrors encode()'s grid math, see that function's
    docstring), not against the function under test itself."""
    palette, payload_len, nsym = 8, 1024, 32
    got = harness._bits_per_patch_on_success(payload_len, palette, nsym, subpatch=2)

    bps = bits_per_symbol(palette)
    ecc_len = rs_encoded_length(5 + payload_len, nsym)
    num_symbols = math.ceil(ecc_len * 8 / bps)
    cells_per_patch = 2 * 2
    data_patches_needed = math.ceil(num_symbols / cells_per_patch)
    width, height = compute_grid(data_patches_needed, palette)
    data_patches = width * (height - 1)
    total_patches = width * height
    expected = cells_per_patch * bps * (data_patches / total_patches) * (payload_len / ecc_len)

    assert got == pytest.approx(expected)


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
    _grid_stats itself has an independent computation to be caught against."""
    bps = bits_per_symbol(palette)
    ecc_len = rs_encoded_length(5 + payload_len, nsym)
    num_symbols = math.ceil(ecc_len * 8 / bps)
    cells_per_patch = subpatch * subpatch
    data_patches_needed = math.ceil(num_symbols / cells_per_patch)
    width, height = compute_grid(data_patches_needed, palette)
    total_patches = width * height
    data_patches = width * (height - 1)
    return total_patches, data_patches


@pytest.mark.parametrize(
    "payload_len,palette,nsym,subpatch",
    [
        (48, 8, 32, 1),
        (1024, 256, 32, 1),
        (4096, 256, 32, 1),
        (48, 8, 32, 2),
    ],
)
def test_grid_stats_matches_independent_recomputation(payload_len, palette, nsym, subpatch):
    """_grid_stats' total_patches/data_patches (the geometry both bits/patch AND token-crossover
    are built on) must match a hand-rederived grid size, not just be internally consistent with
    itself."""
    g = harness._grid_stats(payload_len, palette, nsym, subpatch)
    expected_total, expected_data = _manual_grid_stats(payload_len, palette, nsym, subpatch)
    assert g.total_patches == expected_total
    assert g.data_patches == expected_data


def test_base64_token_estimate_matches_standard_formula_and_real_base64():
    """_base64_token_estimate must equal ceil(n/3)*4 -- the standard base64-with-padding
    expansion -- AND match Python's own base64.b64encode length for real payloads, not just an
    internally-consistent restatement of the same ceil formula."""
    for n in (0, 1, 2, 3, 4, 5, 6, 48, 1024, 4096, 16384):
        got = harness._base64_token_estimate(n)
        assert got == math.ceil(n / 3) * 4
        assert got == len(base64.b64encode(bytes(n)))


# Hand-computed (palette, subpatch, payload_len, nsym) -> (total_patches, base64_token_est,
# token_ratio, heliogram_cheaper), independently re-derived via _manual_grid_stats/base64 above
# and cross-checked against results.csv's own stored rows for the same cells where present (see
# each case's comment). Deliberately includes one case on each side of the 1.0 crossover
# (palette=256 at 1024B is NOT cheaper; at 4096B it IS) so the boolean flag is exercised both
# ways, not just checked for "some value".
_TOKEN_CROSSOVER_CASES = [
    # matches results.csv: palette=8,subpatch=1,payload_size=48,corruption=clean
    (48, 8, 32, 1, 256, 64, 4.0, False),
    # matches results.csv: palette=256,subpatch=1,payload_size=1024 (any corruption -- token
    # fields don't vary by corruption, see CellResult's docstring)
    (1024, 256, 32, 1, 1536, 1368, 1536 / 1368, False),
    # matches results.csv: palette=256,subpatch=1,payload_size=4096 -- the crossover case
    (4096, 256, 32, 1, 5120, 5464, 5120 / 5464, True),
    # matches results.csv: palette=8,subpatch=2,payload_size=48
    (48, 8, 32, 2, 72, 64, 72 / 64, False),
]


@pytest.mark.parametrize(
    "payload_len,palette,nsym,subpatch,exp_total,exp_b64,exp_ratio,exp_cheaper",
    _TOKEN_CROSSOVER_CASES,
)
def test_token_crossover_matches_hand_computed_values(
    payload_len, palette, nsym, subpatch, exp_total, exp_b64, exp_ratio, exp_cheaper
):
    """The regression guard this review round flags as entirely missing (see this file's module
    docstring): pins _token_crossover's output against hand-computed total_patches/
    base64_token_est/token_ratio/heliogram_cheaper for concrete (payload_len, palette, nsym,
    subpatch) tuples that also match results.csv's own stored rows for the same cells (see each
    case's comment above) -- so a future change to the data_patches_needed ceil, the base64
    formula, or the ratio/threshold computation would be caught here even if results.csv were
    never regenerated."""
    tc = harness._token_crossover(payload_len, palette, nsym, subpatch)
    assert tc.total_patches == exp_total
    assert tc.base64_token_est == exp_b64
    assert tc.token_ratio == pytest.approx(exp_ratio)
    assert tc.heliogram_cheaper is exp_cheaper
    assert tc.heliogram_cheaper == (tc.token_ratio < 1.0)  # the flag IS the threshold, nothing else


def test_crossover_payload_size_interpolates_linearly_hand_computed():
    """_crossover_payload_size's three branches, each hand-verified independently of the
    function: (1) linear interpolation between two swept sizes when the ratio crosses 1.0
    between them -- r0=1.2 at 100B, r1=0.8 at 200B, frac=(1.2-1.0)/(1.2-0.8)=0.5, so the crossing
    is 100 + 0.5*(200-100) = 150.0 exactly; (2) already cheaper at the smallest swept size
    returns that size unchanged; (3) a ratio that never drops below 1.0 anywhere in the swept
    range returns None (NOT a claim it never crosses at some larger, untested size -- see the
    function's own docstring)."""
    assert harness._crossover_payload_size([100, 200, 300], [1.2, 0.8, 0.5]) == pytest.approx(150.0)
    assert harness._crossover_payload_size([100, 200, 300], [0.9, 0.8, 0.5]) == 100.0
    assert harness._crossover_payload_size([100, 200, 300], [2.0, 1.5, 1.2]) is None
    assert harness._crossover_payload_size([], []) is None


def test_run_cell_wires_token_crossover_fields_onto_cell_result():
    """Guards the WIRING, not just the pure function: _run_cell must actually copy
    _token_crossover's output onto the CellResult it returns (total_patches/base64_token_est/
    token_ratio/heliogram_cheaper), not just compute it and drop it -- a real (tiny) run through
    _run_cell, checked against the same hand-computed values test_token_crossover_matches_
    hand_computed_values pins for the pure function, plus results.csv's own stored row for this
    exact cell (palette=256, subpatch=1, payload_size=4096, corruption=clean:
    total_patches=5120, base64_token_est=5464, token_ratio=0.937042...)."""
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
    assert result.decode_success_rate == 1.0  # clean corruption must always succeed


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
