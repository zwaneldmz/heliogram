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
"""

from __future__ import annotations

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
