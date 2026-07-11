"""Pytest suite for heliogram.instruments.injection_bench (handoff M6, A9: "behavioral-payload
capacity + detector-eval mode").

Assert-based, no fixtures/frameworks beyond plain pytest.raises/subprocess, matching the rest of
this repo's test idiom. Like tests/test_foreign_tile.py/tests/test_phase2_scaffold.py, this file
proves the CPU-only import boundary holds for heliogram.instruments.injection_bench too --
nothing in this module (or its tests) ever needs torch/transformers.

`measure_behavioral_capacity` itself is only exercised through its model=None/processor=None/
judge=None guard rails below -- per this module's DATA HONESTY rule, there is no GPU in this
repo, so nothing here ever calls it with a real forward pass (see that function's own docstring).
Everything else (BEHAVIORAL_PAYLOADS, InjectionResult's versioned round-trip, evaluate_defense's
TPR/FPR counting, build_detector_eval_set's deterministic construction, the CPU CLI) is plain
Python and fully exercised here.
"""

from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from heliogram.codec import PATCH_SIZE, VALID_PALETTES, decode_pixels, encode
from heliogram.dataset import random_payload
from heliogram.instruments.foreign_tile import AllowListEntry, guard
from heliogram.instruments.injection_bench import (
    BEHAVIORAL_CATEGORIES,
    BEHAVIORAL_PAYLOADS,
    RESULTS_FORMAT_VERSION,
    BehavioralPayload,
    DefenseEval,
    InjectionResult,
    build_detector_eval_set,
    build_parser,
    evaluate_defense,
    format_results,
    keyword_judge,
    main,
    measure_behavioral_capacity,
    read_results,
    write_results,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- import-time boundary -------------------------------------------------------------------


def test_import_injection_bench_does_not_pull_in_torch():
    from conftest import assert_import_stays_torch_free

    assert_import_stays_torch_free("heliogram.instruments.injection_bench")


# --- BehavioralPayload / BEHAVIORAL_PAYLOADS ---------------------------------------------------


def test_behavioral_payloads_have_unique_names_and_valid_categories():
    names = [p.name for p in BEHAVIORAL_PAYLOADS]
    assert len(names) == len(set(names))
    for p in BEHAVIORAL_PAYLOADS:
        assert p.category in BEHAVIORAL_CATEGORIES
        assert isinstance(p.text, str) and len(p.text) > 0


def test_behavioral_payloads_include_all_three_categories():
    assert {p.category for p in BEHAVIORAL_PAYLOADS} == set(BEHAVIORAL_CATEGORIES)
    assert set(BEHAVIORAL_CATEGORIES) == {"persona", "schema", "tool_call"}


def test_behavioral_payload_rejects_invalid_category():
    with pytest.raises(ValueError):
        BehavioralPayload(name="x", category="not-a-real-category", text="y")


def test_behavioral_payload_text_is_encodable_and_decodable_through_the_real_codec():
    """Sanity check: every fixture's text is plain, round-trippable UTF-8 bytes through the real
    codec -- not some special-cased string that only happens to work in this module's own code."""
    for p in BEHAVIORAL_PAYLOADS:
        data = p.text.encode("utf-8")
        img = encode(data, palette=8, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
        assert decode_pixels(img, palette=8, patch_size=PATCH_SIZE, nsym=32, subpatch=1) == data


# --- keyword_judge: pure string scoring, no model involved -------------------------------------


def test_keyword_judge_true_when_marker_present():
    payload = BEHAVIORAL_PAYLOADS[0]
    assert payload.name == "persona_ignore_instructions"
    assert keyword_judge("Sure, UNBOUND ACTIVE: hello there", payload) is True


def test_keyword_judge_false_when_marker_absent():
    payload = BEHAVIORAL_PAYLOADS[0]
    assert keyword_judge("I can't help with that request.", payload) is False


def test_keyword_judge_is_case_insensitive():
    payload = BEHAVIORAL_PAYLOADS[0]
    assert keyword_judge("unbound active", payload) is True


def test_keyword_judge_covers_every_behavioral_payload_without_raising():
    for payload in BEHAVIORAL_PAYLOADS:
        # Exercises the _JUDGE_MARKERS lookup for every fixture without a KeyError -- proves the
        # marker table actually covers the whole BEHAVIORAL_PAYLOADS tuple, not just entry 0.
        assert keyword_judge("some irrelevant response text", payload) is False


def test_keyword_judge_raises_key_error_for_unknown_payload_name():
    unknown = BehavioralPayload(name="not_a_real_fixture", category="persona", text="whatever")
    with pytest.raises(KeyError):
        keyword_judge("anything", unknown)


# --- InjectionResult: versioned round-trip ------------------------------------------------------


def _make_result(**overrides) -> InjectionResult:
    fields = dict(
        payload_name="persona_ignore_instructions",
        category="persona",
        palette=256,
        subpatch=1,
        payload_size=64,
        patch_size=PATCH_SIZE,
        trials=5,
        influence_rate=0.6,
        note="test note",
    )
    fields.update(overrides)
    return InjectionResult(**fields)


def test_injection_result_to_record_includes_version():
    result = _make_result()
    record = result.to_record()
    assert record["version"] == RESULTS_FORMAT_VERSION
    assert record["payload_name"] == "persona_ignore_instructions"
    assert record["influence_rate"] == 0.6


def test_injection_result_to_record_from_record_round_trip():
    result = _make_result()
    restored = InjectionResult.from_record(result.to_record())
    assert restored == result


def test_injection_result_from_record_rejects_unknown_version():
    record = _make_result().to_record()
    record["version"] = RESULTS_FORMAT_VERSION + 999
    with pytest.raises(ValueError, match="version"):
        InjectionResult.from_record(record)


def test_injection_result_from_record_missing_field_raises_key_error():
    record = _make_result().to_record()
    del record["note"]
    with pytest.raises(KeyError):
        InjectionResult.from_record(record)


# --- format_results / write_results / read_results -----------------------------------------------


def test_format_results_contains_headers_and_values():
    results = [_make_result(influence_rate=0.4)]
    text = format_results(results)
    assert "payload_name" in text
    assert "influence_rate" in text
    assert "persona_ignore_instructions" in text
    assert "0.400" in text


def test_format_results_empty_list_still_prints_headers():
    text = format_results([])
    assert "payload_name" in text
    assert "influence_rate" in text


def test_write_results_and_read_results_round_trip(tmp_path):
    results = [
        _make_result(payload_name=p.name, category=p.category, influence_rate=0.0)
        for p in BEHAVIORAL_PAYLOADS
    ]
    path = tmp_path / "injection_results.jsonl"
    write_results(results, path)

    lines = path.read_text().strip().splitlines()
    assert len(lines) == len(results)
    for line in lines:
        record = json.loads(line)
        assert record["version"] == RESULTS_FORMAT_VERSION

    restored = read_results(path)
    assert restored == results


# --- evaluate_defense: TPR/FPR counting, no model involved ---------------------------------------


def test_evaluate_defense_flag_everything_gives_tpr_fpr_one():
    injection_images, benign_images = build_detector_eval_set(seed=0, n_benign=4)
    result = evaluate_defense(lambda img: True, injection_images, benign_images)
    assert isinstance(result, DefenseEval)
    assert result.tpr == pytest.approx(1.0)
    assert result.fpr == pytest.approx(1.0)
    assert result.n_injection == len(injection_images)
    assert result.n_benign == len(benign_images)


def test_evaluate_defense_flag_nothing_gives_tpr_fpr_zero():
    injection_images, benign_images = build_detector_eval_set(seed=0, n_benign=4)
    result = evaluate_defense(lambda img: False, injection_images, benign_images)
    assert result.tpr == pytest.approx(0.0)
    assert result.fpr == pytest.approx(0.0)


def test_evaluate_defense_empty_sequences_do_not_divide_by_zero():
    result = evaluate_defense(lambda img: True, [], [])
    assert result.tpr == 0.0
    assert result.fpr == 0.0
    assert result.n_injection == 0
    assert result.n_benign == 0


def test_evaluate_defense_foreign_tile_guard_flags_off_allowlist_not_on_allowlist():
    """Composition proof (the exact case named in the task): foreign_tile.guard bound to an
    allow-list, scored via evaluate_defense, correctly flags structurally off-allowlist tiles and
    leaves allow-listed trusted tiles alone -- confirming guard actually plugs into
    evaluate_defense's Callable[[Image], bool] contract end to end."""
    allowlist = [AllowListEntry(palette=8)]
    defense = functools.partial(guard, allowlist=allowlist)

    off_allowlist_images = [
        encode(random_payload(i, 48), palette=64, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
        for i in range(3)
    ]
    on_allowlist_images = [
        encode(random_payload(100 + i, 48), palette=8, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
        for i in range(3)
    ]

    result = evaluate_defense(defense, off_allowlist_images, on_allowlist_images)
    assert result.tpr == pytest.approx(1.0)
    assert result.fpr == pytest.approx(0.0)


def test_guard_is_content_blind_when_allowlist_matches_eval_set_config():
    """THE honest finding this benchmark's CLI prints (see module docstring): when injection and
    benign tiles share an allow-listed config, foreign_tile.guard -- a structural/channel
    detector -- cannot tell them apart on content alone. tpr and fpr must be equal (both ~0.0),
    not just both low independently."""
    injection_images, benign_images = build_detector_eval_set(seed=0, palette=8, n_benign=6)
    allowlist = [AllowListEntry(palette=8)]
    defense = functools.partial(guard, allowlist=allowlist)

    result = evaluate_defense(defense, injection_images, benign_images)
    assert result.tpr == pytest.approx(0.0)
    assert result.fpr == pytest.approx(0.0)
    assert result.tpr == pytest.approx(result.fpr)


def test_guard_is_content_blind_when_allowlist_does_not_match_eval_set_config():
    """Same content-blindness invariant, mirrored: a mismatched allow-list flags EVERY tile
    (both injection and benign) for the same (wrong) structural reason, so tpr == fpr again, this
    time both ~1.0 -- guard's verdict tracks structure, never content, either way."""
    injection_images, benign_images = build_detector_eval_set(seed=0, palette=8, n_benign=6)
    allowlist = [AllowListEntry(palette=16)]  # deliberately mismatched
    defense = functools.partial(guard, allowlist=allowlist)

    result = evaluate_defense(defense, injection_images, benign_images)
    assert result.tpr == pytest.approx(1.0)
    assert result.fpr == pytest.approx(1.0)
    assert result.tpr == pytest.approx(result.fpr)


# --- build_detector_eval_set: deterministic construction -----------------------------------------


def test_build_detector_eval_set_is_deterministic():
    inj1, ben1 = build_detector_eval_set(seed=0)
    inj2, ben2 = build_detector_eval_set(seed=0)
    assert len(inj1) == len(inj2) == len(BEHAVIORAL_PAYLOADS)
    assert len(ben1) == len(ben2)
    for a, b in zip(inj1, inj2):
        assert list(a.getdata()) == list(b.getdata())
    for a, b in zip(ben1, ben2):
        assert list(a.getdata()) == list(b.getdata())


def test_build_detector_eval_set_different_seeds_can_differ():
    _, ben1 = build_detector_eval_set(seed=0, n_benign=8)
    _, ben2 = build_detector_eval_set(seed=1, n_benign=8)
    # the first two (structured) benign fixtures are seed-independent by construction; the
    # random-filler tail should differ across seeds.
    same_tail = all(
        list(a.getdata()) == list(b.getdata()) for a, b in zip(ben1[2:], ben2[2:])
    )
    assert not same_tail


def test_build_detector_eval_set_respects_n_benign():
    _, benign = build_detector_eval_set(seed=0, n_benign=5)
    assert len(benign) == 5
    _, benign_zero = build_detector_eval_set(seed=0, n_benign=0)
    assert benign_zero == []


def test_build_detector_eval_set_injection_images_decode_to_original_payload_text():
    """Confirms build_detector_eval_set genuinely encodes BEHAVIORAL_PAYLOADS' text via the real
    codec (not a placeholder image) -- each injection tile decodes exactly to its own payload's
    UTF-8 bytes."""
    injection_images, _ = build_detector_eval_set(seed=0, palette=8)
    for img, payload_spec in zip(injection_images, BEHAVIORAL_PAYLOADS):
        recovered = decode_pixels(img, palette=8, patch_size=PATCH_SIZE, nsym=32, subpatch=1)
        assert recovered == payload_spec.text.encode("utf-8")


def test_build_detector_eval_set_default_palette_is_valid():
    injection_images, benign_images = build_detector_eval_set(seed=0)
    for img in injection_images + benign_images:
        assert img.width % PATCH_SIZE == 0
        assert img.height % PATCH_SIZE == 0


# --- measure_behavioral_capacity: guard rails that must fire BEFORE any model is touched --------


def test_measure_behavioral_capacity_requires_a_model():
    with pytest.raises(ValueError, match="model"):
        measure_behavioral_capacity(model=None, processor=None, judge=keyword_judge)


def test_measure_behavioral_capacity_requires_a_processor_too():
    with pytest.raises(ValueError, match="model"):
        measure_behavioral_capacity(model=object(), processor=None, judge=keyword_judge)


def test_measure_behavioral_capacity_requires_a_judge():
    with pytest.raises(ValueError, match="judge"):
        measure_behavioral_capacity(model=object(), processor=object(), judge=None)


def test_measure_behavioral_capacity_never_fabricates_without_a_model_even_with_a_judge():
    """Belt-and-suspenders: even with a perfectly good judge supplied, model=None must still
    raise before anything resembling a result is produced -- there is no code path here that
    could return an InjectionResult without a real model."""
    with pytest.raises(ValueError):
        measure_behavioral_capacity(
            model=None, processor=None, judge=keyword_judge, payloads=BEHAVIORAL_PAYLOADS[:1]
        )


# --- CLI: CPU detector-eval mode only ------------------------------------------------------------


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.seed == 0
    assert args.palette == 8
    assert args.n_benign == 8


def test_main_runs_cpu_only_and_prints_expected_sections(capsys):
    rc = main(["--seed", "0", "--palette", "8", "--n-benign", "4"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TPR=" in out
    assert "FPR=" in out
    assert "HONEST READING" in out
    assert "measure_behavioral_capacity" in out
    assert "GPU" in out
    assert "fabricates" in out


def test_main_rejects_invalid_palette(capsys):
    rc = main(["--palette", "3"])  # 3 is not in VALID_PALETTES
    assert rc == 2
    err = capsys.readouterr().err
    assert "must be one of" in err
    assert "got 3" in err


def test_main_default_palette_is_a_valid_palette():
    assert 8 in VALID_PALETTES


# --- CLI smoke tests (subprocess) -----------------------------------------------------------


def test_cli_help_does_not_require_torch():
    result = subprocess.run(
        [sys.executable, "-m", "heliogram.instruments.injection_bench", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--palette" in result.stdout


def test_cli_end_to_end_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "heliogram.instruments.injection_bench", "--seed", "0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "TPR=" in result.stdout
    assert "GPU" in result.stdout
