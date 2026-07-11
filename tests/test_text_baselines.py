"""Tests for heliogram.baselines' multi-encoding text baselines (module docstring (d)) --
the 'is base64 even the right bar?' measurement -- using injected fake tokenizers, since the
real measurement needs transformers + HuggingFace Hub access this test environment may lack.
"""

from __future__ import annotations

import json

import pytest

from heliogram import baselines
from heliogram.baselines import (
    TEXT_ENCODINGS,
    load_measured_text_baselines,
    measure_text_encoding_baselines,
)


class _CharTokenizer:
    """1 token per character -- makes expected bits/token exactly computable per encoding."""

    def encode(self, text):
        return list(range(len(text)))


class _PairTokenizer:
    """1 token per 2 characters (ceil) -- a crude stand-in for BPE merging."""

    def encode(self, text):
        return list(range((len(text) + 1) // 2))


def test_encodings_cover_expected_set():
    assert set(TEXT_ENCODINGS) == {"base64", "ascii85", "base85", "hex"}


def test_char_tokenizer_ranking_is_exact():
    """With 1 token/char, bits/token == 8 * bytes/chars: ascii85/base85 (1.25 chars/byte)
    must beat base64 (~1.333 chars/byte, plus padding) which must beat hex (2 chars/byte).
    Pool sizes are multiples of 4 so a85/b85 emit exactly ceil(n/4)*5 chars with no padding
    subtleties changing the ranking."""
    result = measure_text_encoding_baselines(
        sizes=(1024,), seeds=(0,), write=False, tokenizer=_CharTokenizer()
    )
    bpt = {name: e.bits_per_token for name, e in result.encodings.items()}
    assert bpt["ascii85"] == pytest.approx(8 / 1.25)  # 6.4
    assert bpt["base85"] == pytest.approx(8 / 1.25)
    assert bpt["base64"] == pytest.approx(1024 * 8 / (((1024 + 2) // 3) * 4))  # ~6.0
    assert bpt["hex"] == pytest.approx(4.0)
    assert result.strongest.encoding in ("ascii85", "base85")
    assert result.strongest.bits_per_token > bpt["base64"]


def test_strongest_reflects_tokenizer_not_alphabet_arithmetic():
    """With a pair-merging tokenizer every encoding's bits/token doubles, but the RANKING logic
    must still read the measured numbers, not per-encoding char arithmetic."""
    result = measure_text_encoding_baselines(
        sizes=(300,), seeds=(0, 1), write=False, tokenizer=_PairTokenizer()
    )
    for e in result.encodings.values():
        assert e.chars_per_token == pytest.approx(2.0, abs=0.02)
    assert result.strongest.bits_per_token == max(
        e.bits_per_token for e in result.encodings.values()
    )


def test_persist_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        baselines, "MEASURED_TEXT_BASELINES_PATH", tmp_path / "text_baselines.json"
    )
    written = measure_text_encoding_baselines(
        sizes=(256,), seeds=(0,), write=True, tokenizer=_CharTokenizer()
    )
    loaded = load_measured_text_baselines()
    assert loaded is not None
    assert set(loaded.encodings) == set(written.encodings)
    assert loaded.strongest.encoding == written.strongest.encoding
    assert loaded.strongest.bits_per_token == pytest.approx(
        written.strongest.bits_per_token
    )
    assert loaded.tokenizer_id == written.tokenizer_id


def test_load_missing_and_malformed_return_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        baselines, "MEASURED_TEXT_BASELINES_PATH", tmp_path / "absent.json"
    )
    assert load_measured_text_baselines() is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(baselines, "MEASURED_TEXT_BASELINES_PATH", bad)
    assert load_measured_text_baselines() is None
    empty = tmp_path / "empty.json"
    empty.write_text(
        json.dumps(
            {
                "tokenizer_id": "x",
                "tokenizer_package": "y",
                "sample_sizes": [],
                "seeds": [],
                "encodings": {},
                "measured_note": "",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(baselines, "MEASURED_TEXT_BASELINES_PATH", empty)
    assert load_measured_text_baselines() is None  # zero encodings == no measurement


def test_results_md_reports_unmeasured_caveat_or_measurement(tmp_path, monkeypatch):
    """RESULTS.md must always say WHERE the text bar stands: either the measured encodings
    table, or the explicit 'base64 may understate' caveat -- never silence."""
    from heliogram import harness

    results = harness.run(
        palettes=(8,),
        corruptions={"clean": lambda im: im, "jpeg_q85": harness.CORRUPTIONS["jpeg_q85"]},
        n_trials=1,
        subpatches=(1,),
        payload_sizes=(48,),
    )
    md_path = tmp_path / "RESULTS.md"

    monkeypatch.setattr(harness, "_resolve_text_baselines", lambda: None)
    harness.write_results_md(results, md_path)
    text = md_path.read_text()
    assert "UNMEASURED CAVEAT" in text
    assert "NOT MEASURED in this checkout" in text

    fake = measure_text_encoding_baselines(
        sizes=(256,), seeds=(0,), write=False, tokenizer=_CharTokenizer()
    )
    monkeypatch.setattr(harness, "_resolve_text_baselines", lambda: fake)
    harness.write_results_md(results, md_path)
    text = md_path.read_text()
    assert "Other text encodings" in text
    assert "`ascii85`" in text
    # with the char tokenizer ascii85 (6.4) is BELOW the measured base64 bar (8.096), so the
    # cross-checked wording (not the ABOVE-the-bar warning) must appear
    assert ("Cross-checked against" in text) or ("ABOVE the base64 bar" in text)
