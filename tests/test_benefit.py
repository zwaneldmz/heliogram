"""Pytest suite for heliogram.benefit (Slice C: the exactness differentiator + the token-savings
demo) -- previously ZERO test coverage despite carrying this project's headline exactness claim
and (via heliogram.baselines' measured tokenizer baseline) the Bar A/C economic claim's honesty.

Covers, per the Group C work-group spec:
  - rs_error_correction_capacity's arithmetic (hand-computed: nsym=32, RS_NSIZE=255 => 16
    correctable bytes/chunk, budget fraction 16/255) and its C2 wording fix (detection beyond
    the correction budget is "overwhelming probability", never an absolute "always detects").
  - token_savings_demo's token-accounting math (base64_token_est/hex_token_est derivations).
  - the new C3 raw-text-tokens row (text-like payloads get a fair raw-text baseline; genuinely
    binary payloads correctly get none) and the printed honesty note.
  - the exactness_argument() path (status tagging, no invented OCR error rate).
  - heliogram.baselines.load_measured_baseline (absence -> None, round-trip when present) --
    tested here per the work-group spec even though the function lives in heliogram.baselines,
    because heliogram.benefit's raw-text row is its main consumer.

CPU-only, no network: every test that could otherwise trigger a real `transformers` import
(heliogram.benefit._estimate_raw_text_tokens's measured path) explicitly monkeypatches
`sys.modules["transformers"]` -- either to `None` (forcing the same ImportError a
transformers-less environment would raise, no real import attempted) or to a small fake module
exposing just the `AutoTokenizer.from_pretrained(...).encode(...)` surface this codebase actually
calls. `monkeypatch.setitem` reverts `sys.modules` at the end of each test, so a REAL transformers
import (if the package happens to be installed in the environment running this suite) is never
triggered and never leaks into later tests -- see test_phase2_scaffold.py's/test_patchsize.py's
identical "torch/transformers not in sys.modules" import-boundary convention, which this file's
first test also asserts and therefore must run before any transformers-touching test pollutes
sys.modules for real (default pytest collection order runs this file, alphabetically first among
tests/test_*.py, before test_phase2_scaffold.py; every transformers-touching test below still
monkeypatches defensively regardless of run order).
"""

from __future__ import annotations

import base64
import json
import sys
import types
from pathlib import Path

import pytest

import heliogram.baselines as baselines_mod
from heliogram.baselines import MeasuredBase64Baseline, load_measured_baseline
from heliogram.benefit import (
    ExactnessPoint,
    RSGuarantee,
    TokenSavingsResult,
    _estimate_raw_text_tokens,
    exactness_argument,
    format_exactness_argument,
    format_token_savings_report,
    rs_error_correction_capacity,
    sample_binary_payload,
    sample_structured_payload,
    token_savings_demo,
)
from heliogram.codec import RS_NSIZE

# --- import-time boundary ------------------------------------------------------------------


def test_import_heliogram_benefit_does_not_pull_in_torch_or_transformers():
    """heliogram.benefit's module-scope import boundary: transformers is only ever imported
    LAZILY, inside _estimate_raw_text_tokens, when the raw-text measured path is actually
    attempted -- never merely by importing this module. Mirrors test_phase2_scaffold.py's/
    test_patchsize.py's equivalent checks for heliogram.vlm/heliogram.patchsize. Must run before
    any test below that installs a fake (or, if misconfigured, a real) transformers module."""
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


# --- rs_error_correction_capacity: hand-computed arithmetic ---------------------------------


def test_rs_error_correction_capacity_hand_computed_default():
    """nsym=32, RS_NSIZE=255 (heliogram.codec.encode's actual defaults) => correctable = 16
    bytes/chunk (floor(32/2)), budget fraction 16/255 -- hand-computed, not re-derived from the
    function under test."""
    rs = rs_error_correction_capacity()
    assert rs.nsize == RS_NSIZE == 255
    assert rs.nsym == 32
    assert rs.max_correctable_byte_errors_per_chunk == 16
    budget_fraction = rs.max_correctable_byte_errors_per_chunk / rs.nsize
    assert budget_fraction == pytest.approx(16 / 255)


def test_rs_error_correction_capacity_odd_nsym_floors_down():
    rs = rs_error_correction_capacity(nsym=7, nsize=255)
    assert rs.max_correctable_byte_errors_per_chunk == 3  # floor(7/2), not round(7/2)


def test_rs_error_correction_capacity_rejects_out_of_range_nsym():
    with pytest.raises(ValueError):
        rs_error_correction_capacity(nsym=0)
    with pytest.raises(ValueError):
        rs_error_correction_capacity(nsym=255, nsize=255)  # nsym must be < nsize
    with pytest.raises(ValueError):
        rs_error_correction_capacity(nsym=-1)


def test_rs_error_correction_capacity_returns_rsguarantee_dataclass():
    rs = rs_error_correction_capacity(nsym=32)
    assert isinstance(rs, RSGuarantee)


# --- C2: detection-beyond-budget wording must be hedged, not absolute -----------------------


def test_rs_error_correction_capacity_note_hedges_detection_not_correction():
    """C2 fix: bounded-distance RS decoding only GUARANTEES correction within the budget; beyond
    it, detection is 'overwhelming probability', not absolute -- reedsolo can (rarely)
    miscorrect into a wrong-but-plausible codeword without raising. The note must say so, and
    must never claim unconditional/absolute detection."""
    rs = rs_error_correction_capacity()
    note = rs.note.lower()
    assert "overwhelming" in note
    assert "not" in note and "absolute" in note
    assert "always detect" not in note
    assert "guarantees detection" not in note
    # the correction half, in contrast, IS an unconditional/exact claim
    assert "exactly" in note


def test_exactness_argument_detection_point_hedges_and_correction_point_does_not():
    points = exactness_argument(nsym=32)
    detection, correction = points[0], points[1]

    assert "detection" in detection.claim.lower()
    assert "overwhelming" in detection.heliogram.lower()
    assert (
        "not an absolute guarantee" in detection.heliogram.lower()
        or "not absolute" in detection.heliogram.lower()
    )
    assert "never returns a silently-wrong" not in detection.heliogram.lower()

    assert "correction" in correction.claim.lower()
    assert "exactly" in correction.heliogram.lower()
    assert "deterministically" in correction.heliogram.lower()


def test_format_exactness_argument_summary_line_hedges_detection():
    text = format_exactness_argument()
    assert "overwhelming" in text.lower()
    # the one-line summary must not reduce back to an unconditional "gives detection" claim
    assert "gives detection + correction" not in text


# --- exactness_argument(): structure, statuses, no invented OCR numbers ----------------------


def test_exactness_argument_structure_and_statuses():
    points = exactness_argument(nsym=32)
    assert len(points) == 4
    assert all(isinstance(p, ExactnessPoint) for p in points)
    statuses = [p.status for p in points]
    assert statuses.count("structural") == 3
    assert statuses.count("open_phase2_measurement") == 1


def test_exactness_argument_open_measurement_point_is_the_error_rate_claim():
    points = exactness_argument(nsym=32)
    open_points = [p for p in points if p.status == "open_phase2_measurement"]
    assert len(open_points) == 1
    assert "error rate" in open_points[0].claim.lower()


def test_exactness_argument_never_invents_an_ocr_error_rate_number():
    """DATA HONESTY: no numeric OCR error/accuracy percentage should ever appear -- every
    OCR-side claim is either structural/logical or explicitly flagged as unmeasured."""
    points = exactness_argument(nsym=32)
    for p in points:
        assert "%" not in p.heliogram
        assert "%" not in p.rendered_text_ocr


def test_exactness_argument_uses_the_given_nsym_in_its_text():
    points = exactness_argument(nsym=10)
    assert "nsym=10" in points[0].heliogram
    assert "5 bytes" in points[0].heliogram  # floor(10/2)


def test_format_exactness_argument_default_uses_exactness_argument():
    assert format_exactness_argument() == format_exactness_argument(exactness_argument())


# --- token_savings_demo: base64/hex token-accounting math ------------------------------------


def test_token_savings_demo_base64_hex_token_accounting(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)  # keep the raw-text row analytic/offline
    payload = sample_structured_payload(seed=1, target_bytes=512)
    result = token_savings_demo(payload, palette=16, nsym=32)

    assert result.payload_len == len(payload)
    assert result.base64_token_est == len(base64.b64encode(payload))
    assert result.hex_token_est == len(payload.hex()) == 2 * len(payload)
    assert result.patches_vs_base64_ratio == pytest.approx(
        result.total_patches / result.base64_token_est
    )
    assert result.patches_vs_hex_ratio == pytest.approx(result.total_patches / result.hex_token_est)
    # hex is always >= base64 tokens for the same payload (2 chars/byte vs ~4/3 chars/byte)
    assert result.hex_token_est > result.base64_token_est


def test_token_savings_demo_total_patches_matches_actual_encoded_image(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    payload = sample_structured_payload(seed=2, target_bytes=256)
    result = token_savings_demo(payload, palette=16, patch_size=14, nsym=32)
    assert result.total_patches > 0
    assert result.clean_roundtrip_ok is True  # a clean-channel decode should always succeed


def test_token_savings_demo_result_is_dataclass_with_expected_fields():
    fields = TokenSavingsResult.__dataclass_fields__
    for name in (
        "raw_text_token_est",
        "patches_vs_raw_text_ratio",
        "raw_text_method_note",
        "is_text_payload",
    ):
        assert name in fields


# --- C3: raw-text row -- text-like payloads get a fair baseline, binary payloads get none ----


def test_sample_binary_payload_is_deterministic_and_not_valid_utf8():
    a = sample_binary_payload(seed=0, target_bytes=256)
    b = sample_binary_payload(seed=0, target_bytes=256)
    c = sample_binary_payload(seed=1, target_bytes=256)
    assert a == b
    assert a != c
    with pytest.raises(UnicodeDecodeError):
        a.decode("utf-8")


def test_sample_binary_payload_differs_from_json_payload_of_same_size():
    binary = sample_binary_payload(seed=0, target_bytes=256)
    # sample_structured_payload doesn't take an exact size, but its output is always valid JSON
    json_payload = sample_structured_payload(seed=0, target_bytes=256)
    json_payload.decode("utf-8")  # must not raise -- JSON is text-like by construction
    assert binary != json_payload


def test_token_savings_demo_text_payload_gets_raw_text_row_analytic_fallback(monkeypatch):
    """transformers unavailable -> analytic ~4 chars/token estimate, clearly labeled as such."""
    monkeypatch.setitem(sys.modules, "transformers", None)
    payload = sample_structured_payload(seed=3, target_bytes=512)
    result = token_savings_demo(payload, palette=16, nsym=32)

    assert result.is_text_payload is True
    text = payload.decode("utf-8")
    expected_tokens = max(1, len(text) // 4)
    assert result.raw_text_token_est == expected_tokens
    assert "analytic" in result.raw_text_method_note.lower()
    assert "not installed" in result.raw_text_method_note.lower()
    assert result.patches_vs_raw_text_ratio == pytest.approx(
        result.total_patches / expected_tokens
    )
    # the whole point of C3: heliogram's patch count is far above the raw-text token count for
    # a text-like (JSON) payload -- this is the "heliogram loses to plain text" fact.
    assert result.total_patches > result.raw_text_token_est


def test_token_savings_demo_binary_payload_has_no_raw_text_baseline(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    payload = sample_binary_payload(seed=4, target_bytes=256)
    result = token_savings_demo(payload, palette=16, nsym=32)

    assert result.is_text_payload is False
    assert result.raw_text_token_est is None
    assert result.patches_vs_raw_text_ratio is None
    assert "N/A" in result.raw_text_method_note
    assert "no fair" in result.raw_text_method_note.lower()


class _FakeTokenizer:
    """Deterministic fake BPE-ish tokenizer: ~1 token per 3 characters."""

    def encode(self, text: str):
        return list(range(max(1, -(-len(text) // 3))))  # ceil(len/3), min 1


class _FakeAutoTokenizer:
    from_pretrained_calls: list = []

    @classmethod
    def from_pretrained(cls, tokenizer_id):
        cls.from_pretrained_calls.append(tokenizer_id)
        return _FakeTokenizer()


def _install_fake_transformers(monkeypatch, auto_tokenizer_cls) -> None:
    fake_module = types.ModuleType("transformers")
    fake_module.AutoTokenizer = auto_tokenizer_cls
    monkeypatch.setitem(sys.modules, "transformers", fake_module)


def test_estimate_raw_text_tokens_measured_path_uses_fake_tokenizer(monkeypatch):
    _FakeAutoTokenizer.from_pretrained_calls = []
    _install_fake_transformers(monkeypatch, _FakeAutoTokenizer)

    text = "hello world, this is some text"
    n_tokens, note = _estimate_raw_text_tokens(text)

    assert n_tokens == len(_FakeTokenizer().encode(text))
    assert "measured" in note.lower()
    assert _FakeAutoTokenizer.from_pretrained_calls == ["Qwen/Qwen2.5-VL-7B-Instruct"]  # default


def test_estimate_raw_text_tokens_uses_measured_tokenizer_id_when_baseline_present(
    monkeypatch, tmp_path
):
    baseline_path = tmp_path / "base64_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "bits_per_token": 5.0,
                "chars_per_token": 1.2,
                "tokens_per_kb": 100.0,
                "tokenizer_id": "fake/tokenizer-id",
                "tokenizer_package": "transformers==0.0.0",
                "sample_sizes": [1024],
                "per_size": {"1024": 5.0},
                "measured_note": "test fixture",
            }
        )
    )
    monkeypatch.setattr(baselines_mod, "MEASURED_BASELINE_PATH", baseline_path)

    _FakeAutoTokenizer.from_pretrained_calls = []
    _install_fake_transformers(monkeypatch, _FakeAutoTokenizer)

    _estimate_raw_text_tokens("some text")
    assert _FakeAutoTokenizer.from_pretrained_calls == ["fake/tokenizer-id"]


def test_estimate_raw_text_tokens_falls_back_when_tokenizer_load_fails(monkeypatch):
    class _FailingAutoTokenizer:
        @staticmethod
        def from_pretrained(tokenizer_id):
            raise OSError("no network in this test")

    _install_fake_transformers(monkeypatch, _FailingAutoTokenizer)

    text = "x" * 40
    n_tokens, note = _estimate_raw_text_tokens(text)

    assert n_tokens == max(1, len(text) // 4)  # falls back to the analytic estimate
    assert "OSError" in note
    assert "analytic" in note.lower()
    # never mislabel a fallback estimate as "measured"
    assert not note.lower().startswith("measured")


def test_estimate_raw_text_tokens_analytic_fallback_when_transformers_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    text = "y" * 100
    n_tokens, note = _estimate_raw_text_tokens(text)
    assert n_tokens == max(1, len(text) // 4)
    assert "not installed" in note.lower()
    assert "analytic" in note.lower()


def test_estimate_raw_text_tokens_empty_text_returns_at_least_one_token(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    n_tokens, _note = _estimate_raw_text_tokens("")
    assert n_tokens == 1


# --- format_token_savings_report: the printed honesty note -----------------------------------


def test_format_token_savings_report_text_payload_prints_wide_margin_honesty_note(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    payload = sample_structured_payload(seed=5, target_bytes=1500)
    result = token_savings_demo(payload, palette=64, nsym=32)
    text = format_token_savings_report(result)

    assert "raw text tokens" in text.lower()
    assert "HONESTY NOTE" in text
    assert "loses to plain text" in text.lower()
    assert "incompressible binary" in text.lower()


def test_format_token_savings_report_binary_payload_prints_no_baseline_honesty_note(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    payload = sample_binary_payload(seed=6, target_bytes=1500)
    result = token_savings_demo(payload, palette=64, nsym=32)
    text = format_token_savings_report(result)

    assert "N/A" in text
    assert "HONESTY NOTE" in text
    assert "candidate niche" in text.lower()
    assert "not valid utf-8" in text.lower()


# --- heliogram.baselines.load_measured_baseline (Group C spec: test it here too) -------------


def test_load_measured_baseline_returns_none_when_file_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(baselines_mod, "MEASURED_BASELINE_PATH", tmp_path / "does_not_exist.json")
    assert load_measured_baseline() is None


def test_load_measured_baseline_returns_none_for_malformed_json(monkeypatch, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json at all")
    monkeypatch.setattr(baselines_mod, "MEASURED_BASELINE_PATH", p)
    assert load_measured_baseline() is None


def test_load_measured_baseline_returns_none_for_missing_keys(monkeypatch, tmp_path):
    p = tmp_path / "partial.json"
    p.write_text(json.dumps({"bits_per_token": 5.0}))  # missing every other required key
    monkeypatch.setattr(baselines_mod, "MEASURED_BASELINE_PATH", p)
    assert load_measured_baseline() is None


def test_load_measured_baseline_round_trips_when_present(monkeypatch, tmp_path):
    p = tmp_path / "base64_baseline.json"
    data = {
        "bits_per_token": 5.234,
        "chars_per_token": 1.147,
        "tokens_per_kb": 175.5,
        "tokenizer_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "tokenizer_package": "transformers==4.51.0",
        "sample_sizes": [1024, 4096, 16384],
        "seeds": [0, 1, 2],
        "per_size": {"1024": 5.4, "4096": 5.2, "16384": 5.1},
        "measured_note": "measured: test fixture",
    }
    p.write_text(json.dumps(data))
    monkeypatch.setattr(baselines_mod, "MEASURED_BASELINE_PATH", p)

    result = load_measured_baseline()
    assert isinstance(result, MeasuredBase64Baseline)
    assert result.bits_per_token == pytest.approx(5.234)
    assert result.chars_per_token == pytest.approx(1.147)
    assert result.tokens_per_kb == pytest.approx(175.5)
    assert result.tokenizer_id == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert result.tokenizer_package == "transformers==4.51.0"
    assert result.sample_sizes == [1024, 4096, 16384]
    assert result.per_size == pytest.approx({1024: 5.4, 4096: 5.2, 16384: 5.1})
    assert result.measured_note == "measured: test fixture"


def test_measured_base64_baseline_note_property_is_the_fixed_cross_group_contract():
    """harness.py reads .note / .bits_per_token / .tokenizer_id off whatever baseline object it
    is handed -- this is a fixed cross-group contract (see baselines.py's MeasuredBase64Baseline
    docstring); this test pins that surface directly, independent of any file on disk."""
    m = MeasuredBase64Baseline(
        bits_per_token=5.0,
        chars_per_token=1.2,
        tokens_per_kb=100.0,
        tokenizer_id="fake/id",
        tokenizer_package="transformers==1.0",
        sample_sizes=[1024],
        per_size={1024: 5.0},
        measured_note="hello world",
    )
    assert m.note == "hello world" == m.measured_note
    assert m.bits_per_token == 5.0
    assert m.tokenizer_id == "fake/id"
