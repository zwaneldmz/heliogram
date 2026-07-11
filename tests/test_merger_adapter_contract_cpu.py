"""CPU contract tests for scripts/train_merger_adapter.py -- Task 2's cheap merger-adapter go/no-go
scaffold.

Same CPU-only, no-torch contract as tests/test_phase2_scaffold.py and
tests/test_p16_curriculum.py: this file loads scripts/train_merger_adapter.py as a plain module
(scripts/ has no __init__.py, by design -- see tests/test_phase2_scaffold.py's
_load_train_qlora_module for the identical technique, including registering the module in
sys.modules BEFORE exec_module runs so its @dataclass-decorated classes' type-resolution
machinery finds it) and exercises ONLY the model-free half: argparse/config construction, the
pure-Python cost-estimate + budget-abort path, and the model=None refuse guards on the real-run
entry points. Nothing here ever loads torch/transformers/peft -- that half of the module is,
by its own module docstring, never run against real weights in this repository.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_train_merger_adapter_module():
    """Identical technique to tests/test_phase2_scaffold.py's _load_train_qlora_module (see that
    function's docstring for why sys.modules must be populated before exec_module runs)."""
    spec = importlib.util.spec_from_file_location(
        "train_merger_adapter", REPO_ROOT / "scripts" / "train_merger_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tma = _load_train_merger_adapter_module()


# --- import boundary: torch/transformers-free -------------------------------------------------


def test_import_stays_torch_free():
    """The central invariant (mirrors tests/test_phase2_scaffold.py's
    test_import_heliogram_does_not_pull_in_torch and tests/test_p16_curriculum.py's identical
    concern for scripts/train_qlora.py): merely importing/exec'ing this module must never pull
    in torch/transformers, checked in a FRESH SUBPROCESS so this test's own result does not
    depend on whether some OTHER, already-collected test file legitimately imported torch first."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts'); import train_merger_adapter; "
            "assert 'torch' not in sys.modules, 'importing the module pulled in torch'; "
            "assert 'transformers' not in sys.modules, 'importing the module pulled in transformers'",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_cli_help_does_not_require_torch_and_documents_key_flags():
    """--help must work even with zero GPU packages installed: argparse setup happens before any
    lazy torch/transformers/peft import (mirrors tests/test_phase2_scaffold.py's
    test_train_qlora_cli_help_does_not_require_torch)."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "train_merger_adapter.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    for flag in (
        "--design", "--palette", "--corruptions", "--n-train-images", "--n-test-images",
        "--epochs", "--lora-variant", "--gpu-hourly-rate-usd", "--budget-cap-usd",
    ):
        assert flag in result.stdout, f"{flag} missing from --help output"


def test_cli_default_invocation_hits_the_refuse_guard():
    """The whole point of this script (see its module docstring): running it from the command
    line, in ANY environment including this one, always reaches run_design_a's/run_design_b's
    model=None guard and raises -- there is no flag that hands main() a real model. A cheap,
    within-budget config must fail with OUR ValueError message, not a torch ImportError or a
    budget-abort RuntimeError (both of which would also be "a refusal", but not the specific
    guard this test pins)."""
    result = subprocess.run(
        [
            sys.executable, str(REPO_ROOT / "scripts" / "train_merger_adapter.py"),
            "--design", "a", "--n-train-images", "2", "--n-test-images", "1", "--epochs", "1",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "requires a real, already-loaded model" in result.stderr
    assert "run_design_a" in result.stderr
    # the cost-estimate/config report must have been printed to stdout BEFORE the guard fired
    assert "projected cost" in result.stdout
    assert "HONEST-REPORTING CAVEAT" in result.stdout


def test_driver_cli_help_does_not_require_torch():
    """scripts/drive_merger_adapter.py is the REAL-RUN driver (it loads a model and calls
    run_design_a/b), but importing it and running --help must still work with zero GPU packages:
    the only torch import is function-local in run_probe._load_tower, reached exclusively during
    an actual run. Guards the same torch-free-import invariant the scaffold itself keeps, so the
    driver can't silently grow a top-level `import torch` that breaks `--help` on a CPU box."""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "drive_merger_adapter.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "torch" not in result.stderr.lower()
    for flag in ("--design", "--palette", "--corruptions", "--budget-cap-usd"):
        assert flag in result.stdout, f"{flag} missing from driver --help output"


# --- refuse-without-model guards (in-process, no subprocess needed) ---------------------------


def test_run_design_a_requires_a_model():
    with pytest.raises(ValueError, match="model"):
        tma.run_design_a(model=None, processor=None, dtype=None)


def test_run_design_a_requires_a_processor_and_dtype_too():
    with pytest.raises(ValueError, match="model"):
        tma.run_design_a(model=object(), processor=None, dtype=None)
    with pytest.raises(ValueError, match="model"):
        tma.run_design_a(model=object(), processor=object(), dtype=None)


def test_run_design_b_requires_a_model():
    with pytest.raises(ValueError, match="model"):
        tma.run_design_b(model=None, processor=None, dtype=None)


def test_run_design_b_requires_a_processor_and_dtype_too():
    with pytest.raises(ValueError, match="model"):
        tma.run_design_b(model=object(), processor=None, dtype=None)
    with pytest.raises(ValueError, match="model"):
        tma.run_design_b(model=object(), processor=object(), dtype=None)


def test_run_design_a_and_b_reject_any_palette_other_than_16():
    """Scope guard: probe_report_premerger.md measured palette=256 already at/near chance
    pre-merger, so no merger-side adapter can recover it -- both entry points must refuse before
    doing any (fake) work, even with a "model" object present."""
    with pytest.raises(ValueError, match="palette=16"):
        tma.run_design_a(model=object(), processor=object(), dtype=object(), palette=256)
    with pytest.raises(ValueError, match="palette=16"):
        tma.run_design_b(model=object(), processor=object(), dtype=object(), palette=256)


def test_run_design_b_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant"):
        tma.run_design_b(model=object(), processor=object(), dtype=object(), variant="B3")


def test_run_design_a_and_b_reject_unknown_corruption():
    with pytest.raises(ValueError, match="unknown corruption"):
        tma.run_design_a(
            model=object(), processor=object(), dtype=object(), corruptions=["not_a_real_one"]
        )
    with pytest.raises(ValueError, match="unknown corruption"):
        tma.run_design_b(
            model=object(), processor=object(), dtype=object(), corruptions=["not_a_real_one"]
        )


# --- cost estimate: pure Python, deterministic, no torch ---------------------------------------


def test_estimate_runtime_seconds_deterministic_and_monotonic():
    a1 = tma.estimate_runtime_seconds("a", n_train_images=6, n_test_images=3, epochs=60)
    a2 = tma.estimate_runtime_seconds("a", n_train_images=6, n_test_images=3, epochs=60)
    assert a1 == a2  # deterministic for fixed inputs
    assert a1 > 0

    b_small = tma.estimate_runtime_seconds("b", n_train_images=6, n_test_images=3, epochs=5)
    b_big = tma.estimate_runtime_seconds("b", n_train_images=6, n_test_images=3, epochs=50)
    assert b_big > b_small  # more epochs -> more projected time

    # design b (real training) must project MORE time than design a (frozen forward pass only)
    # for the same image counts and a realistic epoch count
    b_default_epochs = tma.estimate_runtime_seconds("b", n_train_images=6, n_test_images=3, epochs=20)
    assert b_default_epochs > a1


def test_estimate_runtime_scales_with_corruptions_and_includes_alignment():
    """The estimator must (a) count the one-time clean alignment sanity cell and (b) multiply the
    per-corruption work by n_corruptions -- the fix for the earlier undercount that ignored both,
    understating a pre-spend budget gate's projected cost by ~len(corruptions)."""
    per_corr_a = (6 + 3) * tma.DESIGN_A_SECONDS_PER_IMAGE
    a1 = tma.estimate_runtime_seconds("a", 6, 3, 60, n_corruptions=1)
    a2 = tma.estimate_runtime_seconds("a", 6, 3, 60, n_corruptions=2)
    # alignment cell is present even at n_corruptions=1 (alignment + one per-corruption block)
    assert a1 == (6 + 3) * tma.DESIGN_A_SECONDS_PER_IMAGE + per_corr_a
    # each extra corruption adds exactly one more per-corruption block
    assert a2 - a1 == per_corr_a
    # Design B scales its training block by corruption count the same way
    per_corr_b = 6 * 60 * tma.DESIGN_B_SECONDS_PER_TRAIN_STEP + 3 * tma.DESIGN_B_SECONDS_PER_EVAL_IMAGE
    b1 = tma.estimate_runtime_seconds("b", 6, 3, 60, n_corruptions=1)
    b2 = tma.estimate_runtime_seconds("b", 6, 3, 60, n_corruptions=2)
    assert b2 - b1 == per_corr_b


def test_estimate_runtime_seconds_rejects_unknown_design():
    with pytest.raises(ValueError, match="design"):
        tma.estimate_runtime_seconds("c", n_train_images=1, n_test_images=1, epochs=1)


def test_estimate_runtime_seconds_rejects_negative_inputs():
    with pytest.raises(ValueError):
        tma.estimate_runtime_seconds("a", n_train_images=-1, n_test_images=1, epochs=1)


def test_estimate_cost_usd_scales_with_hourly_rate():
    cheap = tma.estimate_cost_usd("a", 6, 3, 60, gpu_hourly_rate_usd=1.0)
    expensive = tma.estimate_cost_usd("a", 6, 3, 60, gpu_hourly_rate_usd=10.0)
    assert expensive == pytest.approx(cheap * 10.0)
    assert cheap == tma.estimate_cost_usd("a", 6, 3, 60, gpu_hourly_rate_usd=1.0)  # deterministic


def test_estimate_cost_usd_rejects_negative_rate():
    with pytest.raises(ValueError):
        tma.estimate_cost_usd("a", 1, 1, 1, gpu_hourly_rate_usd=-1.0)


def test_check_budget_passes_under_cap():
    tma.check_budget(1.0, cap_usd=40.0)  # must not raise


def test_check_budget_aborts_over_cap():
    """The abort-over-budget path: a config with enough train images/epochs to blow past the
    default tens-of-dollars cap must raise RuntimeError, never silently proceed."""
    huge_cost = tma.estimate_cost_usd("b", n_train_images=10_000, n_test_images=10_000, epochs=10_000)
    assert huge_cost > tma.BUDGET_CAP_USD_DEFAULT
    with pytest.raises(RuntimeError, match="exceeds the tens-of-dollars"):
        tma.check_budget(huge_cost)


def test_check_budget_respects_an_explicit_higher_cap():
    huge_cost = tma.estimate_cost_usd("b", n_train_images=10_000, n_test_images=10_000, epochs=10_000)
    tma.check_budget(huge_cost, cap_usd=huge_cost + 1.0)  # must not raise


# --- alignment sanity assert: pure numeric comparison ------------------------------------------


def test_check_alignment_sanity_passes_near_the_committed_number():
    # PREMERGER_CLEAN_SYMBOL_ERROR is cited from probe_report_premerger.md (0.1344) -- a measured
    # value within tolerance must pass silently.
    tma.check_alignment_sanity(0.135)
    tma.check_alignment_sanity(tma.PREMERGER_CLEAN_SYMBOL_ERROR)  # exact match


def test_check_alignment_sanity_fails_loud_when_far_off():
    with pytest.raises(RuntimeError, match="ALIGNMENT SANITY CHECK FAILED"):
        tma.check_alignment_sanity(0.90)  # near-chance -- nowhere near the committed 13.4%
    with pytest.raises(RuntimeError, match="probe_report_premerger.md"):
        tma.check_alignment_sanity(0.90)


def test_check_alignment_sanity_respects_custom_target_and_tolerance():
    tma.check_alignment_sanity(0.50, target=0.50, tolerance=0.01)  # must not raise
    with pytest.raises(RuntimeError):
        tma.check_alignment_sanity(0.50, target=0.10, tolerance=0.01)


# --- pure-Python config/arg building -------------------------------------------------------------


def _args(**overrides):
    defaults = dict(
        design="a", model_id="Qwen/Qwen2.5-VL-3B-Instruct", palette=16, corruptions="clean,jpeg_q70",
        n_train_images=6, n_test_images=3, payload_size=1024, seed=0, epochs=60, hidden_dim=64,
        lora_rank=8, lora_alpha=16, lora_variant="B1", nsym=32,
        gpu_hourly_rate_usd=tma.GPU_HOURLY_RATE_USD_DEFAULT, budget_cap_usd=tma.BUDGET_CAP_USD_DEFAULT,
    )
    defaults.update(overrides)

    class _Ns:
        pass

    ns = _Ns()
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


def test_build_run_config_parses_corruptions_and_computes_cost():
    config = tma.build_run_config(_args())
    assert config.corruptions == ("clean", "jpeg_q70")
    assert config.design == "a"
    assert config.palette == 16
    # __post_init__ passes n_corruptions=len(corruptions) (2 here: clean,jpeg_q70), because a
    # run extracts/trains once per requested corruption -- so the consistency check must too.
    assert config.estimated_cost_usd == tma.estimate_cost_usd(
        "a", 6, 3, 60, tma.GPU_HOURLY_RATE_USD_DEFAULT, n_corruptions=2
    )


def test_build_run_config_rejects_non_16_palette():
    with pytest.raises(ValueError, match="palette=16"):
        tma.build_run_config(_args(palette=8))


def test_build_run_config_rejects_bad_design():
    with pytest.raises(ValueError, match="design"):
        tma.build_run_config(_args(design="z"))


def test_build_run_config_rejects_bad_lora_variant():
    with pytest.raises(ValueError, match="lora_variant"):
        tma.build_run_config(_args(lora_variant="B9"))


def test_build_run_config_rejects_unknown_corruption():
    with pytest.raises(ValueError, match="unknown corruption"):
        tma.build_run_config(_args(corruptions="clean,not_a_real_corruption"))


def test_format_config_report_includes_seed_config_and_baseline_sources():
    """Requirement: the run report must log seed, config, and the frozen-merger baseline
    source (probe_report.md / probe_report_premerger.md) -- and the mandatory HONEST-REPORTING
    caveat."""
    config = tma.build_run_config(_args(seed=7))
    report = tma.format_config_report(config)
    assert "seed: 7" in report
    assert "probe_report_premerger.md" in report
    assert "probe_report.md" in report
    assert "HONEST-REPORTING CAVEAT" in report
    assert tma.HONEST_CAVEAT_TEXT in report
    assert "readout head is NOT the LM" in report


def test_honest_caveat_text_documents_necessary_not_sufficient_and_stronger_negative():
    """Pin the two specific claims the task's HONEST-REPORTING caveat must make (see module
    docstring): a readout-head pass does not prove the LM uses the symbols (necessary, not
    sufficient), and a Design-B failure is a STRONGER negative than Step 0's own merged-probe
    fail."""
    assert "NOT the LM" in tma.HONEST_CAVEAT_TEXT
    assert "Necessary, not sufficient" in tma.HONEST_CAVEAT_TEXT
    assert "REOPENS" in tma.HONEST_CAVEAT_TEXT
    assert "STRONGER" in tma.HONEST_CAVEAT_TEXT
    assert "Qwen2.5-VL" in tma.HONEST_CAVEAT_TEXT


# --- Design A's numpy readout head: pure numpy, no model, no torch -----------------------------


def test_group_quads_reshapes_patch_rows_into_concatenated_quads():
    import numpy as np

    patch_embeddings = np.arange(8 * 3).reshape(8, 3).astype(float)  # 8 patches (2 merge units), hidden=3
    grouped = tma._group_quads(patch_embeddings)
    assert grouped.shape == (2, 12)
    assert (grouped[0] == patch_embeddings[0:4].reshape(-1)).all()
    assert (grouped[1] == patch_embeddings[4:8].reshape(-1)).all()


def test_group_quads_rejects_non_multiple_of_four():
    import numpy as np

    with pytest.raises(ValueError, match="multiple of 4"):
        tma._group_quads(np.zeros((5, 3)))


def test_fit_quad_readout_mlp_deterministic_and_learns_a_separable_toy_problem():
    """No model, no torch: a small synthetic problem where the 4 quad-position labels are a
    deterministic (noiseless) linear function of the input features. The MLP readout should
    drive symbol error to (near) zero -- proving the training loop actually fits something,
    not just that it runs without crashing -- and repeated fits with the same seed must agree
    exactly (determinism requirement)."""
    import numpy as np

    rng = np.random.default_rng(0)
    palette = 4
    n = 64
    hidden = 2  # vision_hidden per patch
    dim = 4 * hidden
    X = rng.normal(size=(n, dim))
    # deterministic labels: each of the 4 positions' symbol = argmax of its own hidden slice,
    # modulo palette -- a simple, learnable, noiseless function of X.
    y = np.zeros((n, 4), dtype=np.int64)
    for k in range(4):
        slice_ = X[:, k * hidden : (k + 1) * hidden]
        y[:, k] = np.argmax(slice_, axis=1) % palette

    fit1 = tma.fit_quad_readout_mlp(X, y, X, y, palette=palette, hidden_dim=16, seed=0, epochs=200)
    fit2 = tma.fit_quad_readout_mlp(X, y, X, y, palette=palette, hidden_dim=16, seed=0, epochs=200)
    assert fit1.symbol_error == fit2.symbol_error  # deterministic for a fixed seed
    assert fit1.symbol_error < 0.5  # meaningfully better than chance (1 - 1/4 = 0.75)


def test_fit_quad_readout_mlp_rejects_no_valid_labels():
    import numpy as np

    X = np.zeros((4, 8))
    y = np.full((4, 4), -1, dtype=np.int64)
    with pytest.raises(ValueError, match="no valid"):
        tma.fit_quad_readout_mlp(X, y, X, y, palette=16)


# --- MERGER_LORA_TARGET_MODULES: the same names train_qlora.py's LORA_MERGER_TARGET_MODULES uses ---


def test_merger_lora_target_modules_matches_train_qlora_convention():
    assert tma.MERGER_LORA_TARGET_MODULES == ["mlp.0", "mlp.2"]
