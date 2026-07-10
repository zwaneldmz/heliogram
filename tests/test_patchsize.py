"""Pytest suite for heliogram.patchsize (handoff M0 guardrail #3: verify the patch size
empirically for the chosen model, do not hardcode 14 from memory).

Assert-based, no fixtures/frameworks beyond plain pytest.raises/subprocess, matching the rest of
this repo's test idiom. Like tests/test_phase2_scaffold.py, this file's whole point includes
proving the CPU-only import boundary holds: heliogram.patchsize must never require torch/
transformers merely to be imported or to run its documented (no real model) code path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from heliogram.codec import PATCH_SIZE
from heliogram.patchsize import (
    KNOWN_PATCH_SIZES,
    PatchSizeReport,
    build_parser,
    format_report,
    known_patch_size,
    main,
    verify_patch_size,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- import-time boundary ----------------------------------------------------------------------


def test_import_heliogram_patchsize_does_not_pull_in_torch():
    """Central invariant for this module (mirrors test_phase2_scaffold.py's equivalent check for
    heliogram.vlm): `import heliogram.patchsize` must never require torch/transformers, since
    verify_patch_size only ever reads attributes off caller-supplied objects -- it never loads a
    model itself."""
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


# --- known_patch_size -----------------------------------------------------------------------


def test_known_patch_size_qwen25vl_matches_codec_default():
    """Qwen/Qwen2.5-VL-7B-Instruct's documented patch size is 14 -- the same 14 heliogram.codec.
    PATCH_SIZE defaults to. If this ever flips (either KNOWN_PATCH_SIZES is corrected, or
    heliogram.codec.PATCH_SIZE changes), that is a real, deliberate change worth re-deriving --
    update this assertion (and re-check every place PATCH_SIZE=14 is assumed) rather than
    silently deleting it."""
    assert known_patch_size("Qwen/Qwen2.5-VL-7B-Instruct") == 14
    assert known_patch_size("Qwen/Qwen2.5-VL-7B-Instruct") == PATCH_SIZE


def test_known_patch_size_unknown_model_raises_key_error():
    with pytest.raises(KeyError, match="Qwen/Qwen2.5-VL-7B-Instruct"):  # listed in the message
        known_patch_size("not-a-real-model-id")


# --- verify_patch_size: documented path (no processor/config) ---------------------------------


def test_verify_patch_size_documented_path_matches_codec_default():
    report = verify_patch_size("Qwen/Qwen2.5-VL-7B-Instruct")
    assert isinstance(report, PatchSizeReport)
    assert report.source == "documented"
    assert report.patch_size == 14
    assert report.matches_codec_default is True
    assert "no processor" in report.note or "config" in report.note


def test_verify_patch_size_documented_path_unknown_model_raises_key_error():
    with pytest.raises(KeyError):
        verify_patch_size("not-a-real-model-id")


# --- verify_patch_size: measured path (real processor/config attributes) ----------------------


def test_verify_patch_size_measured_from_processor_image_processor_patch_size():
    """A tiny fake object exposing exactly processor.image_processor.patch_size -- no real
    transformers Processor is constructed, just a plain object with the expected attribute
    shape, per verify_patch_size's documented guarded-attribute-chain contract."""

    class _FakeImageProcessor:
        patch_size = 14

    class _FakeProcessor:
        image_processor = _FakeImageProcessor()

    report = verify_patch_size("some-model-not-in-the-table", processor=_FakeProcessor())
    assert report.source == "measured"
    assert report.patch_size == 14
    assert report.matches_codec_default is True


def test_verify_patch_size_measured_from_config_vision_config_patch_size():
    class _FakeVisionConfig:
        patch_size = 16

    class _FakeConfig:
        vision_config = _FakeVisionConfig()

    report = verify_patch_size("another-model", config=_FakeConfig())
    assert report.source == "measured"
    assert report.patch_size == 16
    assert report.matches_codec_default is False  # 16 != codec.PATCH_SIZE (14)


def test_verify_patch_size_measured_from_config_spatial_patch_size_fallback():
    """config.vision_config.patch_size absent -> falls back to spatial_patch_size."""

    class _FakeVisionConfig:
        spatial_patch_size = 32

    class _FakeConfig:
        vision_config = _FakeVisionConfig()

    report = verify_patch_size("yet-another-model", config=_FakeConfig())
    assert report.source == "measured"
    assert report.patch_size == 32


def test_verify_patch_size_processor_takes_priority_over_config():
    class _FakeImageProcessor:
        patch_size = 14

    class _FakeProcessor:
        image_processor = _FakeImageProcessor()

    class _FakeVisionConfig:
        patch_size = 99  # deliberately different, to prove processor wins

    class _FakeConfig:
        vision_config = _FakeVisionConfig()

    report = verify_patch_size("m", processor=_FakeProcessor(), config=_FakeConfig())
    assert report.patch_size == 14


def test_verify_patch_size_measured_path_never_fabricates_when_attribute_missing():
    """A processor/config WAS supplied (empirical verification was explicitly requested), but it
    has none of the expected attribute paths -- this must raise, never silently fall back to the
    documented table (that would misrepresent a failed measurement as a real one)."""
    with pytest.raises(ValueError, match="patch_size"):
        verify_patch_size("m", processor=object())
    with pytest.raises(ValueError):
        verify_patch_size("m", config=object())


# --- format_report / CLI ------------------------------------------------------------------------


def test_format_report_mentions_model_and_match_status():
    report = verify_patch_size("Qwen/Qwen2.5-VL-7B-Instruct")
    text = format_report(report)
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in text
    assert "14" in text
    assert "matches" in text.lower()


def test_build_parser_requires_model_flag():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # --model is required


def test_main_prints_report_for_known_model(capsys):
    rc = main(["--model", "Qwen/Qwen2.5-VL-7B-Instruct"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen2.5-VL-7B-Instruct" in out
    assert "documented" in out


def test_main_returns_error_code_for_unknown_model(capsys):
    rc = main(["--model", "not-a-real-model-id"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not-a-real-model-id" in err


def test_cli_help_does_not_require_torch():
    result = subprocess.run(
        [sys.executable, "-m", "heliogram.patchsize", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--model" in result.stdout


def test_cli_end_to_end_matches_codec_default():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "heliogram.patchsize",
            "--model",
            "Qwen/Qwen2.5-VL-7B-Instruct",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert f"{PATCH_SIZE}px" in result.stdout
    assert "matches: yes" in result.stdout


# --- KNOWN_PATCH_SIZES table sanity -------------------------------------------------------------


def test_known_patch_sizes_table_has_the_two_documented_qwen_models():
    assert KNOWN_PATCH_SIZES["Qwen/Qwen2.5-VL-7B-Instruct"] == 14
    assert KNOWN_PATCH_SIZES["Qwen/Qwen2-VL-7B-Instruct"] == 14
