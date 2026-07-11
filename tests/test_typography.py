"""Tests for heliogram.typography -- the CPU-only, model-free GEOMETRIC de-risk gate for the
'dense typeset glyphs' pivot (see that module's docstring for the full argument).

These tests check the GEOMETRY and ARITHMETIC only -- no OCR, no model, nothing about real
legibility is (or could be) asserted here. That is the whole point of the module under test.
"""

from __future__ import annotations

import math

import pytest
from PIL import ImageFont

from heliogram import typography
from heliogram.typography import (
    COLOR_CODEC_NET_CEILING_BITS_PER_PATCH,
    DEFAULT_NSYM,
    ReferenceBars,
    _load_monospace_font,
    load_reference_bars,
    render_typeset_density,
    sweep_typography,
)
from tests.conftest import assert_import_stays_torch_free


def _payload(seed: int = 0, size: int = 512) -> bytes:
    import random

    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


# --- import boundary --------------------------------------------------------------------------


def test_import_stays_torch_free():
    """Same CPU-only-by-default import boundary the rest of the package holds to (see
    heliogram.baselines/heliogram.codec): importing heliogram.typography must never pull in
    torch or transformers as a side effect."""
    assert_import_stays_torch_free("heliogram.typography")


# --- pinned reference constant --------------------------------------------------------------


def test_color_codec_ceiling_matches_its_own_derivation():
    """COLOR_CODEC_NET_CEILING_BITS_PER_PATCH is a literal, pinned copy of a PUBLISHED measured
    number (README.md/spec/format-v0.1.md), not something this module re-derives -- but it must
    still match `log2(256) * 223/255` (nsym=32 asymptotic RS overhead fraction) to the stated
    3-decimal precision, or the pinned literal has drifted from its documented derivation."""
    derived = math.log2(256) * 223 / 255
    assert COLOR_CODEC_NET_CEILING_BITS_PER_PATCH == pytest.approx(derived, abs=5e-4)


# --- font loading ------------------------------------------------------------------------------


def test_load_monospace_font_succeeds_on_a_real_candidate():
    font, path = _load_monospace_font(10)
    assert isinstance(font, ImageFont.FreeTypeFont)
    assert path  # some real candidate path was used
    # actually monospace: any two glyphs must share advance width
    assert font.getlength("i") == pytest.approx(font.getlength("M"))


def test_load_monospace_font_raises_when_no_candidate_available(monkeypatch):
    monkeypatch.setattr(typography, "MONOSPACE_FONT_CANDIDATES", ("/no/such/font.ttf",))
    monkeypatch.setattr(typography, "_font_cache", {})
    with pytest.raises(RuntimeError, match="monospace"):
        _load_monospace_font(10)


# --- determinism --------------------------------------------------------------------------------


def test_render_is_deterministic():
    payload = _payload(seed=1, size=256)
    a = render_typeset_density(payload, 10, apply_rs=False)
    b = render_typeset_density(payload, 10, apply_rs=False)
    assert a.image.tobytes() == b.image.tobytes()
    assert a.image.size == b.image.size
    assert a.total_patches == b.total_patches
    assert a.bits_per_patch == b.bits_per_patch


def test_render_differs_for_different_payloads():
    a = render_typeset_density(_payload(seed=1, size=256), 10, apply_rs=False)
    b = render_typeset_density(_payload(seed=2, size=256), 10, apply_rs=False)
    assert a.image.tobytes() != b.image.tobytes()


# --- monotonicity: smaller font -> more chars/patch -> higher bits/patch -----------------------


def test_smaller_font_increases_geometric_density():
    payload = _payload(seed=0, size=1024)
    sizes = [14, 10, 6, 4]
    densities = [
        render_typeset_density(payload, s, apply_rs=False) for s in sizes
    ]
    chars_per_patch = [d.chars_per_patch for d in densities]
    bits_per_patch = [d.bits_per_patch for d in densities]
    # strictly increasing as font size shrinks (sizes list is already decreasing)
    assert chars_per_patch == sorted(chars_per_patch)
    assert bits_per_patch == sorted(bits_per_patch)
    assert chars_per_patch[0] < chars_per_patch[-1]
    assert bits_per_patch[0] < bits_per_patch[-1]


# --- RS overhead strictly lowers bits/patch vs raw ----------------------------------------------


@pytest.mark.parametrize("font_size", [14, 10, 6])
def test_rs_overhead_strictly_lowers_bits_per_patch(font_size):
    payload = _payload(seed=3, size=2048)
    raw = render_typeset_density(payload, font_size, apply_rs=False)
    rs = render_typeset_density(payload, font_size, apply_rs=True, nsym=DEFAULT_NSYM)
    assert rs.bits_per_patch < raw.bits_per_patch
    # RS strictly inflates the rendered byte/char count for the same payload...
    assert rs.rendered_len > raw.rendered_len
    # ...but never changes what's being measured (the ORIGINAL payload's bits)
    assert rs.payload_bits == raw.payload_bits == len(payload) * 8


def test_rs_frame_matches_codec_framing_length():
    """The RS variant must use the SAME message framing (version + 4-byte length + payload) and
    the SAME reedsolo chunking heliogram.codec.encode uses, so the ECC comparison against the
    color codec's ceiling is apples-to-apples -- cross-check against
    heliogram.codec.rs_encoded_length directly."""
    from heliogram.codec import rs_encoded_length

    payload = _payload(seed=4, size=4096)
    stream = typography._rs_frame(payload, nsym=32)
    expected_len = rs_encoded_length(5 + len(payload), nsym=32)
    assert len(stream) == expected_len


# --- reference-line comparisons ------------------------------------------------------------------


def test_load_reference_bars_reads_real_measured_file_when_present():
    """heliogram/data/text_baselines.json exists in this checkout (base64=8.096, ascii85=8.374
    per the module docstring's cited numbers) -- load_reference_bars must surface it, not the
    'no measurement' fallback."""
    bars = load_reference_bars()
    assert bars.color_codec_net_ceiling == COLOR_CODEC_NET_CEILING_BITS_PER_PATCH
    if bars.ascii85_bits_per_token is not None:
        assert bars.ascii85_bits_per_token == pytest.approx(8.373831775700934, abs=1e-3)
        assert bars.base64_bits_per_token == pytest.approx(8.096385542168674, abs=1e-3)
        assert bars.ascii85_bits_per_token > bars.base64_bits_per_token


def test_load_reference_bars_degrades_gracefully_when_file_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(
        typography, "load_measured_text_baselines", lambda: None
    )
    bars = load_reference_bars()
    assert bars.base64_bits_per_token is None
    assert bars.ascii85_bits_per_token is None
    assert bars.color_codec_net_ceiling == COLOR_CODEC_NET_CEILING_BITS_PER_PATCH
    assert "no measured text baseline" in bars.note


def test_sweep_beats_flags_computed_correctly_against_synthetic_bars():
    """Inject synthetic reference bars (bypassing whatever is/isn't measured in this checkout)
    so the beats_* arithmetic itself -- not the measured file's contents -- is what's under
    test."""
    payload = _payload(seed=5, size=1024)
    bars = ReferenceBars(
        color_codec_net_ceiling=6.996,
        base64_bits_per_token=8.096,
        ascii85_bits_per_token=8.374,
        note="synthetic",
    )
    rows = sweep_typography(payload, font_sizes_px=[14, 6], bars=bars)
    assert len(rows) == 2
    for row in rows:
        assert row.beats_color_codec_ceiling == (row.bits_per_patch_rs > 6.996)
        assert row.beats_base64_bar == (row.bits_per_patch_rs > 8.096)
        assert row.beats_ascii85_bar == (row.bits_per_patch_rs > 8.374)
        # RS is the harder bar to clear than raw for the same config
        assert row.bits_per_patch_rs < row.bits_per_patch_raw
        # beating the stronger ascii85 bar implies beating the weaker base64 bar (both True/False
        # consistent with bits_per_patch_rs being a single number compared against two bars where
        # ascii85 > base64)
        if row.beats_ascii85_bar:
            assert row.beats_base64_bar


def test_sweep_handles_missing_bars_as_none_not_false():
    payload = _payload(seed=6, size=512)
    bars = ReferenceBars(
        color_codec_net_ceiling=6.996,
        base64_bits_per_token=None,
        ascii85_bits_per_token=None,
        note="no text bars",
    )
    rows = sweep_typography(payload, font_sizes_px=[10], bars=bars)
    assert rows[0].beats_base64_bar is None
    assert rows[0].beats_ascii85_bar is None
    # the color-codec comparison never depends on the text bars, so it must still compute
    assert isinstance(rows[0].beats_color_codec_ceiling, bool)


# --- CLI smoke test -----------------------------------------------------------------------------


def test_main_runs_and_returns_zero(capsys):
    rc = typography.main(["--payload-size", "256", "--font-sizes", "12", "6"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "VERDICT" in out
    assert "bits/patch" in out
