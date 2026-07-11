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
    CostAsymmetryPoint,
    ExactnessPoint,
    RecoveredBitResult,
    RSGuarantee,
    TokenSavingsResult,
    _estimate_raw_text_tokens,
    chance_level_symbol_error,
    cost_asymmetry_points,
    effective_cost_per_recovered_bit,
    exactness_argument,
    format_cost_asymmetry,
    format_effective_cost_per_recovered_bit,
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
    test_patchsize.py's equivalent checks for heliogram.vlm/heliogram.patchsize. Subprocess-
    isolated: see tests/conftest.py."""
    from tests.conftest import assert_import_stays_torch_free

    assert_import_stays_torch_free("heliogram.benefit")


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


# --- Task 4, part 1: cost_asymmetry_points / format_cost_asymmetry --------------------------


def test_cost_asymmetry_points_structure_and_statuses():
    points = cost_asymmetry_points()
    assert len(points) == 3
    assert all(isinstance(p, CostAsymmetryPoint) for p in points)
    # every point in this argument is structural/architectural (true by construction of a
    # ViT-encoder-plus-merger design) -- no point here is an "open measurement" the way
    # exactness_argument's error-rate point is.
    assert all(p.status == "structural" for p in points)


def test_cost_asymmetry_points_covers_the_three_required_claims():
    points = cost_asymmetry_points()
    claims = " ".join(p.claim.lower() for p in points)
    assert "vit" in claims or "vision-tower" in claims or "vision tower" in claims
    assert "kv-cache" in claims or "kv cache" in claims or "activation" in claims
    assert "o(n^2)" in claims or "attention" in claims


def test_cost_asymmetry_points_load_bearing_point_states_count_ne_cost():
    """The third point (prefill attention compute) is explicitly flagged, in the task spec, as
    the LOAD-BEARING honesty point: equal token count does not imply equal compute/memory cost.
    Assert that logical content is actually present in the text, not just claimed."""
    points = cost_asymmetry_points()
    attention_point = points[2]
    assert "o(n^2)" in attention_point.image_tokens.lower()
    assert "on top" in attention_point.image_tokens.lower() or "extra" in attention_point.claim.lower()
    # must not claim the image path is cheaper or equal on compute merely because token counts
    # are comparable -- that is precisely the fallacy this point exists to block.
    combined = (attention_point.claim + attention_point.image_tokens).lower()
    assert "does not" in combined or "not, by itself" in combined or "not equal" in combined.replace(
        "-", " "
    )


def test_cost_asymmetry_points_flags_model_specific_magnitudes_as_assumptions():
    """No FLOP/latency/memory NUMBER may be invented; any model-specific magnitude claim must be
    explicitly flagged as an assumption in the text (per the Task 4 spec)."""
    points = cost_asymmetry_points()
    assumption_points = [p for p in points if "ASSUMPTION" in p.image_tokens]
    assert len(assumption_points) >= 2  # points 1 and 2 both flag model-specific magnitudes
    for p in points:
        # no numeric FLOP/ms/GB figure invented anywhere (only structural/architectural counts
        # like "2x2" or "O(n^2)", which are architecture facts, not invented performance numbers)
        assert "flops" not in p.image_tokens.lower() or "no such number is asserted" in p.image_tokens.lower()


def test_format_cost_asymmetry_default_uses_cost_asymmetry_points():
    assert format_cost_asymmetry() == format_cost_asymmetry(cost_asymmetry_points())


def test_format_cost_asymmetry_prints_all_points_and_load_bearing_summary():
    text = format_cost_asymmetry()
    assert "COST ASYMMETRY" in text
    for i in range(1, 4):
        assert f"{i}." in text
    assert "o(n^2)" in text.lower()
    assert "not, by itself" in text.lower() or "does not" in text.lower()


def test_format_cost_asymmetry_is_deterministic():
    assert format_cost_asymmetry() == format_cost_asymmetry()


# --- Task 4, part 2: chance_level_symbol_error -----------------------------------------------


def test_chance_level_symbol_error_hand_computed():
    """Exact arithmetic (1 - 1/palette), hand-checked against docs/FINDINGS.md's post-merger
    probe table's own "chance" column values at palette=16 and palette=256."""
    assert chance_level_symbol_error(16) == pytest.approx(15 / 16)
    assert chance_level_symbol_error(16) == pytest.approx(0.9375)
    assert chance_level_symbol_error(256) == pytest.approx(255 / 256)
    assert chance_level_symbol_error(256) == pytest.approx(0.99609375)
    assert chance_level_symbol_error(2) == pytest.approx(0.5)


def test_chance_level_symbol_error_rejects_invalid_palette():
    with pytest.raises(ValueError):
        chance_level_symbol_error(0)
    with pytest.raises(ValueError):
        chance_level_symbol_error(-1)


# --- Task 4, part 2: effective_cost_per_recovered_bit -- HONESTY-CRITICAL --------------------


def test_effective_cost_per_recovered_bit_refuses_none_assumption():
    """The function must never silently assume a recovery rate -- assumed_symbol_error=None (or
    simply omitted) must raise, not default to some optimistic number."""
    with pytest.raises(ValueError):
        effective_cost_per_recovered_bit(
            payload_bits=8000,
            tokens=1000,
            assumed_symbol_error=None,
            bits_per_symbol=8,
        )


def test_effective_cost_per_recovered_bit_rejects_out_of_range_error_rate():
    with pytest.raises(ValueError):
        effective_cost_per_recovered_bit(
            payload_bits=8000, tokens=1000, assumed_symbol_error=-0.1, bits_per_symbol=8
        )
    with pytest.raises(ValueError):
        effective_cost_per_recovered_bit(
            payload_bits=8000, tokens=1000, assumed_symbol_error=1.1, bits_per_symbol=8
        )


def test_effective_cost_per_recovered_bit_within_rs_budget_recovers_full_payload():
    """Hand-computed: nsym=32, RS_NSIZE=255 -> rs_budget_fraction = 16/255 (~0.0627, the RS
    budget cited throughout docs/FINDINGS.md/README.md). An assumed_symbol_error exactly AT that
    budget is WITHIN it (<=), so under the code's own (step-function) recovery model, the full
    payload is hypothetically recovered."""
    budget_fraction = 16 / 255
    result = effective_cost_per_recovered_bit(
        payload_bits=48000,
        tokens=7168,
        assumed_symbol_error=budget_fraction,
        bits_per_symbol=8,
        nsym=32,
    )
    assert isinstance(result, RecoveredBitResult)
    assert result.rs_budget_fraction == pytest.approx(16 / 255)
    assert result.within_rs_budget is True
    assert result.recovered_bit_fraction == pytest.approx(1.0)
    assert result.recovered_bits == 48000
    assert result.cost_per_recovered_bit == pytest.approx(7168 / 48000)


def test_effective_cost_per_recovered_bit_chance_level_gives_zero_recovery_and_undefined_cost():
    """HAZARD case (the whole point of this function): an assumed chance-level symbol error
    (the regime the frozen-tower probe actually measured, at/near chance -- see probe_report.md)
    must NOT be assumed to recover anything. recovered_bits must be 0, and cost_per_recovered_bit
    must be undefined (None), never a finite number, never divide-by-zero, never invented."""
    chance = chance_level_symbol_error(256)
    result = effective_cost_per_recovered_bit(
        payload_bits=48000,
        tokens=7168,
        assumed_symbol_error=chance,
        bits_per_symbol=8,
        nsym=32,
    )
    assert result.within_rs_budget is False
    assert result.recovered_bit_fraction == pytest.approx(0.0)
    assert result.recovered_bits == 0
    assert result.cost_per_recovered_bit is None


def test_effective_cost_per_recovered_bit_just_above_budget_also_recovers_nothing():
    budget_fraction = 16 / 255
    result = effective_cost_per_recovered_bit(
        payload_bits=1000,
        tokens=500,
        assumed_symbol_error=budget_fraction + 1e-9,
        bits_per_symbol=8,
        nsym=32,
    )
    assert result.within_rs_budget is False
    assert result.recovered_bits == 0
    assert result.cost_per_recovered_bit is None


def test_effective_cost_per_recovered_bit_rejects_nonpositive_tokens_or_negative_payload():
    with pytest.raises(ValueError):
        effective_cost_per_recovered_bit(
            payload_bits=1000, tokens=0, assumed_symbol_error=0.01, bits_per_symbol=8
        )
    with pytest.raises(ValueError):
        effective_cost_per_recovered_bit(
            payload_bits=-1, tokens=1000, assumed_symbol_error=0.01, bits_per_symbol=8
        )


def test_effective_cost_per_recovered_bit_is_deterministic():
    a = effective_cost_per_recovered_bit(
        payload_bits=48000, tokens=7168, assumed_symbol_error=0.05, bits_per_symbol=8, nsym=32
    )
    b = effective_cost_per_recovered_bit(
        payload_bits=48000, tokens=7168, assumed_symbol_error=0.05, bits_per_symbol=8, nsym=32
    )
    assert a == b


def test_format_effective_cost_per_recovered_bit_contains_mandatory_caveat():
    """The formatter's HEADLINE must state the mandatory conditional-projection caveat, and must
    cross-reference the probe's negative post-merger result -- per the Task 4 spec, verbatim
    enough to be unambiguous to a reader skimming only the top of the report."""
    optimistic = effective_cost_per_recovered_bit(
        payload_bits=48000,
        tokens=7168,
        assumed_symbol_error=16 / 255,
        bits_per_symbol=8,
        assumption_label="RS budget (optimistic anchor)",
    )
    chance = effective_cost_per_recovered_bit(
        payload_bits=48000,
        tokens=7168,
        assumed_symbol_error=chance_level_symbol_error(256),
        bits_per_symbol=8,
        assumption_label="chance-level anchor",
    )
    text = format_effective_cost_per_recovered_bit([optimistic, chance])

    assert "CONDITIONAL projection" in text
    assert "ASSUMED recovery rate" in text
    assert "does NOT achieve" in text
    assert "at/near chance" in text.lower()
    assert "probe_report.md" in text
    assert "docs/FINDINGS.md" in text
    assert "upper bound on a hypothetical" in text.lower()
    assert "not a realized benefit" in text.lower()
    # both scenarios must be labeled explicitly as assumptions, and must be visibly distinct
    assert "RS budget (optimistic anchor)" in text
    assert "chance-level anchor" in text
    assert "undefined (infinite)" in text  # the chance scenario's cost must show as undefined


def test_format_effective_cost_per_recovered_bit_shows_at_least_two_scenarios():
    optimistic = effective_cost_per_recovered_bit(
        payload_bits=8000, tokens=1000, assumed_symbol_error=16 / 255, bits_per_symbol=8
    )
    chance = effective_cost_per_recovered_bit(
        payload_bits=8000,
        tokens=1000,
        assumed_symbol_error=chance_level_symbol_error(256),
        bits_per_symbol=8,
    )
    text = format_effective_cost_per_recovered_bit([optimistic, chance])
    assert text.count("scenario:") == 2


def test_effective_cost_per_recovered_bit_rs_budget_matches_rs_error_correction_capacity():
    """Cross-check: the budget fraction this function derives internally must match
    rs_error_correction_capacity()'s own arithmetic exactly -- no second, drifted definition."""
    rs = rs_error_correction_capacity(nsym=32)
    expected = rs.max_correctable_byte_errors_per_chunk / rs.nsize
    result = effective_cost_per_recovered_bit(
        payload_bits=1000, tokens=500, assumed_symbol_error=0.01, bits_per_symbol=8, nsym=32
    )
    assert result.rs_budget_fraction == pytest.approx(expected)
