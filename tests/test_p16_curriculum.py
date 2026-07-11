"""Tests for the P=16 merger-focused curriculum retarget in scripts/train_qlora.py.

Same CPU-only, no-torch contract as tests/test_phase2_scaffold.py: everything here loads
scripts/train_qlora.py as a plain module and exercises only its pure-Python curriculum-building
and CLI-parsing code, never anything that needs torch/transformers/peft/bitsandbytes. See that
file's module docstring and _load_train_qlora_module for why the module is loaded this way
(scripts/ has no __init__.py, by design).

This experiment is the ONE curriculum the session-2 probe results (RUNBOOK-GPU.md section 2.5,
docs/FINDINGS.md section 3, probe_report_premerger.md) leave standing after the original
large-palette bet (build_curriculum, DEFAULT_PALETTES={64,128,256}) was measured dead: palette=16
pre-merger ViT-block output preserves color identity (13.4% linear-probe symbol error) but the
2x2 merger MLP then destroys it (back to ~66-74% post-merger) -- so the only live fine-tune
target is the merger, at palette=16 only. These tests check the SCAFFOLD is wired correctly
(palette/subpatch pinning, non-decreasing corruption schedule, CLI selection, back-compat with
the old curriculum) -- they say nothing about whether the fine-tune itself would work, which
requires a GPU this environment does not have.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_train_qlora_module():
    """Identical technique to tests/test_phase2_scaffold.py's helper of the same name -- kept as
    its own copy here (rather than importing that test module) so this file can run standalone
    and stays independent of the other test file's internals."""
    spec = importlib.util.spec_from_file_location(
        "train_qlora_p16", REPO_ROOT / "scripts" / "train_qlora.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train_qlora = _load_train_qlora_module()


# --- build_p16_merger_curriculum(): palette/subpatch pinning + corruption schedule -------------


def test_p16_curriculum_is_palette_16_subpatch_1_throughout():
    stages = train_qlora.build_p16_merger_curriculum()
    assert len(stages) > 0
    for stage in stages:
        assert list(stage.palettes) == [16], (
            f"stage {stage.name} must be palette=16 only -- palette=256 is measured at/near "
            "chance even pre-merger (probe_report_premerger.md), so any other palette in this "
            "curriculum would train toward a regime the probes already showed is dead"
        )
        assert list(stage.subpatches) == [1], (
            f"stage {stage.name} must be subpatch=1 -- subpatch>1 is a documented, separate, "
            "pixel-decoder-only geometric ceiling (codec.py's DATA HONESTY note), not part of "
            "this experiment"
        )


def test_p16_curriculum_corruption_prob_non_decreasing():
    stages = train_qlora.build_p16_merger_curriculum()
    probs = [stage.corruption_prob for stage in stages]
    assert probs == sorted(probs), (
        f"corruption_prob must rise (or stay flat) stage over stage, got {probs} -- a curriculum "
        "that gets EASIER partway through would undo the point of a curriculum"
    )
    # and it should actually rise somewhere, not just be flat all the way through, so the
    # "clean+corrupted stages, corruption_prob rising" requirement is exercised for real
    assert probs[0] < probs[-1]
    assert probs[0] == 0.0, "curriculum must open with a clean warm-up stage"


def test_p16_curriculum_corruptions_limited_to_measured_survivors():
    """Every stage's corruption suite must be a subset of {clean, jpeg_q70} -- the two
    corruptions probe_report_premerger.md actually measured a surviving palette=16 pre-merger
    signal under (13.4% clean, 19.0% jpeg_q70). Any other corruption (resize/jpeg_q85/combined)
    is untested at this tap point and palette, so spending this curriculum's budget on it would
    be an unmeasured claim, not the measured-target experiment this curriculum is supposed to be."""
    stages = train_qlora.build_p16_merger_curriculum()
    for stage in stages:
        assert stage.corruptions is not None, f"stage {stage.name} must set a narrowed corruptions dict"
        names = set(stage.corruptions)
        assert names <= {"clean", "jpeg_q70"}, f"stage {stage.name} has unexpected corruptions {names}"
        if stage.corruption_prob < 1.0:
            assert "clean" in names, (
                f"stage {stage.name} has corruption_prob < 1.0 but no 'clean' entry -- "
                "generate_examples looks up 'clean' unconditionally for the no-corruption branch"
            )


def test_p16_curriculum_payload_sizes_span_amortization_range():
    """At least one stage must cover a wide payload range (small/medium/large) so the curriculum
    actually exercises how the fixed per-grid calibration/RS-parity overhead amortizes, not just
    a single tiny payload size throughout."""
    stages = train_qlora.build_p16_merger_curriculum()
    all_payload_sizes = {size for stage in stages for size in stage.payload_sizes}
    assert min(all_payload_sizes) <= 128
    assert max(all_payload_sizes) >= 4096


def test_p16_curriculum_respects_n_examples_per_stage_arg():
    stages = train_qlora.build_p16_merger_curriculum(n_examples_per_stage=123)
    for stage in stages:
        assert stage.n_examples == 123


# --- back-compat: the OLD curriculum builder must still exist, unchanged in shape -------------


def test_build_curriculum_still_present_and_targets_default_palettes():
    """build_curriculum (the original Slice C, large-palette bet) must NOT be deleted or
    repurposed -- other code/tests and --curriculum large_palette (the CLI default) depend on
    it still existing and still targeting DEFAULT_PALETTES, even though the session-2 probe
    verdict measured that regime dead. Retargeting this project onto P=16 must not silently
    break the old default."""
    from heliogram.dataset import DEFAULT_PALETTES

    stages = train_qlora.build_curriculum()
    assert len(stages) == 4  # unchanged stage count from before this retarget
    large_palette_stages = [s for s in stages if set(s.palettes) == set(DEFAULT_PALETTES)]
    assert len(large_palette_stages) >= 3, (
        "build_curriculum must still concentrate its main stages on DEFAULT_PALETTES -- this "
        "retarget adds a new curriculum, it must not touch the old one's behavior"
    )


# --- CLI: --curriculum flag ---------------------------------------------------------------------


def test_curriculum_builders_registry_has_both_options():
    assert set(train_qlora.CURRICULUM_BUILDERS) == {"large_palette", "p16_merger"}
    assert train_qlora.CURRICULUM_BUILDERS["large_palette"] is train_qlora.build_curriculum
    assert (
        train_qlora.CURRICULUM_BUILDERS["p16_merger"] is train_qlora.build_p16_merger_curriculum
    )


def test_cli_curriculum_flag_defaults_to_large_palette():
    parser = train_qlora.build_parser()
    args = parser.parse_args(["--output-dir", "/tmp/whatever"])
    assert args.curriculum == "large_palette"


def test_cli_curriculum_flag_selects_p16_merger():
    parser = train_qlora.build_parser()
    args = parser.parse_args(["--curriculum", "p16_merger"])
    assert args.curriculum == "p16_merger"


def test_cli_curriculum_flag_rejects_unknown_value():
    parser = train_qlora.build_parser()
    try:
        parser.parse_args(["--curriculum", "not_a_real_curriculum"])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("argparse should reject an unknown --curriculum choice")


def test_train_qlora_cli_help_documents_curriculum_flag():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_qlora.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--curriculum" in result.stdout
    assert "p16_merger" in result.stdout
    assert "large_palette" in result.stdout
