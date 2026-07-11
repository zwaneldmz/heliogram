#!/usr/bin/env python
"""scripts/train_merger_adapter.py -- Task 2: the cheap merger-adapter go/no-go (GPU-only;
REFUSING scaffold, never run against real weights in this repo).

*** GPU REQUIRED. UNTESTED IN THIS REPO. *** There is no GPU, torch, transformers, peft, or
bitsandbytes in the environment this script was written in -- it has never been run or imported
end to end against real weights. Every function that needs torch/transformers/peft imports those
packages LOCALLY, inside itself; nothing above `if __name__ == "__main__":` requires them, so
`python scripts/train_merger_adapter.py --help` works with zero GPU packages installed. This
file's CLI entry point (`main`) goes further than that minimum, by construction: it NEVER loads a
real model itself (there is no flag that hands it one -- a live Python model object cannot be
passed over argparse), so every invocation of this script from the command line in ANY
environment -- this one included -- reaches `run_design_a`'s or `run_design_b`'s explicit
model=None guard and raises before touching torch at all. The only way either design's real logic
ever executes is by importing this module from a GPU-side driver that has ALREADY loaded a real
`model`/`processor`/`dtype` (e.g. via `scripts/run_probe.py`'s `_load_tower`, the exact same
loader this file reuses) and calling `run_design_a(model, processor, dtype=dtype, ...)` or
`run_design_b(...)` directly -- see the "REAL INVOCATION" note near the bottom of this docstring.

THE QUESTION THIS FILE SCAFFOLDS (Task 2, following on from Phase-2 Step 0): `probe_report_premerger.md`
measured that a frozen Qwen2.5-VL's pre-merger ViT-block output linearly carries per-patch
palette=16 symbol identity at 13.4% error (clean) / 19.0% (jpeg_q70) -- above the 6.27%
Reed-Solomon budget but far below chance (93.75%) -- while the SAME 16-color code is measured
back at 65.5-73.6% error (at/near chance) once read POST the 2x2 spatial merger
(`probe_report.md` / `probe_report_7b.md`, cited in `docs/FINDINGS.md` section 3). That is the
textbook signature of a bottleneck LAYER, not a bottleneck further upstream: the vision blocks
preserve the signal, the merger MLP destroys it. The frozen-encoder probe (Step 0) cannot answer
whether TRAINING the merger (ViT + LM otherwise frozen) can be made to preserve it -- that is
what this file exists to scaffold, cheaply, as a go/no-go BEFORE any tens-of-GPU-hour QLoRA spend
(`scripts/train_qlora.py`'s `build_p16_merger_curriculum`).

TWO DESIGNS, both implemented below as DESIGNED, REFUSING scaffolds (see `run_design_a`/
`run_design_b`'s own docstrings for the full argument):

  Design A (`--design a`, the cheap diagnostic): frozen pre-merger ViT features, grouped into
    the merger's own 2x2 quads (`_group_quads`), fed through a small trained readout head
    (`fit_quad_readout_mlp` -- concat(4 patch embeddings) -> one hidden ReLU layer -> 4 x
    palette-way logits, plain numpy, no torch), scored per-patch symbol error against the RS
    budget. Cheapest possible test of "can a NONLINEAR readout (not just the linear probe Step 0
    already ran) recover more of the signal from the SAME frozen features" -- no GPU backprop
    through the tower at all, just one frozen forward pass per image plus a CPU-cheap numpy fit.

  Design B (`--design b`, the actual gate): freeze ViT + LM, train ONLY the merger --
    variant B1 (preferred): LoRA on `model.visual.merger.mlp`'s two Linear layers (`mlp.0`/
    `mlp.2`, the exact target-module pair `scripts/train_qlora.py`'s `LORA_MERGER_TARGET_MODULES`
    already names for the same reason); variant B2: a small parallel residual adapter around the
    (frozen) merger when peft is unavailable or B1 underperforms. A `_QuadReadoutHead` (torch,
    trained jointly, gradients flowing back into the merger LoRA/adapter parameters -- this is
    the one thing Design A's frozen-feature numpy head structurally CANNOT do) sits on the
    POST-merger token and is trained (cross-entropy summed over the 4 constituent symbol
    positions) to recover them. Scored against the RS budget AND against the stock frozen-merger
    baseline (`POSTMERGER_SYMBOL_ERROR_RANGE`, cited from `probe_report.md`/`probe_report_7b.md`,
    never re-measured here).

REFUSE-WITHOUT-MODEL (mirrors `heliogram.vlm.zero_shot_symbol_error` and
`heliogram.instruments.injection_bench.measure_behavioral_capacity`'s guard EXACTLY): every
real-run entry point below (`run_design_a`, `run_design_b`, and the shared
`_measure_premerger_clean_symbol_error`/alignment-check path they both call first) requires a
real, already-loaded `model`/`processor`/`dtype` and raises ValueError IMMEDIATELY, before any
other work, if any of those is `None`. There is no fallback path, cached result, or heuristic
default that could return an invented symbol-error number. Nothing in this file ever fabricates a
training result or a metric.

NEVER INVENT A NUMBER: the committed pre-merger palette=16 linear-probe error this file's
alignment-sanity check targets (`PREMERGER_CLEAN_SYMBOL_ERROR` = 0.1344 clean, and 0.1902 for
jpeg_q70) is CITED from `probe_report_premerger.md` (see `docs/FINDINGS.md` section 3 for the
same numbers presented alongside the post-merger comparison) -- it is never recomputed or
hardcoded as if this file measured it. Likewise `POSTMERGER_SYMBOL_ERROR_RANGE` (0.6551-0.7358)
is cited from `probe_report.md`/`probe_report_7b.md`, not remeasured.

DETERMINISM: every random draw this file's real-run paths would make (payload bytes via
`heliogram.dataset.generate_examples`/`heliogram.dataset.random_payload`, the numpy readout
head's minibatch order and weight init) is seeded from the single `--seed` argument, exactly
like `heliogram/probe.py`'s `fit_linear_probe` and `scripts/run_probe.py`'s `_cell_arrays`
seed-range convention (train seed_base = seed + 1_000, test seed_base = seed + 2_000_000, so
train/test payloads never collide).

SCOPE: pinned to Qwen2.5-VL (3B/7B, `transformers==5.13.0` -- the same version the model-
interface contracts in `tests/test_probe_contract_cpu.py` and `tests/test_train_qlora_lora_targets.py`
verify against), and to `palette=16` ONLY (`probe_report_premerger.md` already measured
`palette=256` at/near chance even pre-merger -- 80.8% clean -- so no merger-side adapter, however
trained, can recover a palette that coarse a probe already showed the vision BLOCKS themselves
discard; both `run_design_a`/`run_design_b` raise ValueError on any other palette).

COST/TIME BUDGET (the Step-0 ethos -- see `RUNBOOK-GPU.md`'s own "~$1-2" / "~$15-40" framing for
the probe and curriculum respectively): `estimate_cost_usd` is a pure-Python, deterministic
function of `--design`/`--n-train-images`/`--n-test-images`/`--epochs`/`--gpu-hourly-rate-usd`
(no torch, fully unit-testable); `check_budget` raises RuntimeError -- ABORTING before any run
would start -- whenever the projected cost exceeds `--budget-cap-usd` (default a few dollars,
comfortably inside the tens-of-dollars ceiling this whole go/no-go is supposed to stay under).
`main()` always prints the projected cost/time estimate and runs this check before reaching
either design's model=None guard.

ALIGNMENT SANITY ASSERT: before either design trusts a single new number it computes, BOTH
`run_design_a` and `run_design_b` first re-run the EXACT SAME (palette=16, clean, pre_merger tap
point) probe cell `probe_report_premerger.md` was generated from
(`_measure_premerger_clean_symbol_error`, built entirely from functions REUSED from
`scripts/run_probe.py` -- see below) and check the result against `PREMERGER_CLEAN_SYMBOL_ERROR`
within `ALIGNMENT_TOLERANCE` (`check_alignment_sanity`). A GPU run whose installed
transformers/peft version, model id, or window-shuffle handling has drifted since that report was
generated fails LOUDLY here (RuntimeError) instead of silently reporting a new, uncomparable
number as if it meant something.

ALIGNMENT CODE REUSE, NOT REINVENTION: this file imports (via `_bind_run_probe_alignment_helpers`,
never duplicates) `_extract_pre_merger_embeddings`, `_match_reverse_indices`, `_load_tower`,
`_extract_embeddings`, `MERGE`, `_resolve_visual_tower`, `_merged_embeddings_tensor`, and
`_cell_arrays` from `scripts/run_probe.py` -- the ALREADY CPU-verified (`tests/
test_probe_contract_cpu.py`) window-shuffle unshuffle/merge-unit alignment code -- plus
`heliogram.probe.merged_token_labels`/`evaluate_cell`/`rs_symbol_error_budget`. Design B's
gradient-enabled forward pass (`_forward_post_merger_quad_logits`) is new code (training needs
gradients; `run_probe.py`'s own `_extract_embeddings` wraps its forward pass in
`torch.no_grad()`, which would silently block backprop into the merger LoRA/adapter parameters),
but it still calls `_resolve_visual_tower`/`_merged_embeddings_tensor` for tower resolution and
merged-token-field selection rather than re-deriving those contracts independently.

HONEST-REPORTING CAVEAT (baked into every returned report -- `HONEST_CAVEAT_TEXT` -- read this
before trusting any number either design would produce): A trained readout head is NOT the LM.
Design-B success means the info CAN be made recoverable at the LM boundary by a cheaply-trained
merger -- the go/no-go flips to "go" and the fine-tune question REOPENS -- it does NOT prove the
LM zero-shot uses those symbols or that the economics win. Necessary, not sufficient. If Design B
stays well above budget (RS budget and/or the stock frozen-merger baseline), the negative result
is STRONGER than Step 0's own merged-probe fail: even a cheaply-trained, code-aware merger cannot
carry the signal, not just a frozen, generically-pretrained one -- scoped to Qwen2.5-VL, the one
tower family tested here.

DATA HONESTY (mirrors `scripts/run_probe.py`'s own module docstring): this file has never been
run against real WEIGHTS in this repository -- there is no GPU and no HF Hub access here. The
model-FREE half -- `estimate_cost_usd`/`check_budget`, `RunConfig`/`build_run_config`,
`check_alignment_sanity`'s numeric comparison, `fit_quad_readout_mlp`/`_group_quads`, argparse
setup, and every model=None guard -- is plain Python/numpy and IS fully exercised on CPU (see
tests/test_merger_adapter_contract_cpu.py). The model-DEPENDENT half (everything past each
design's alignment check: real embedding extraction, LoRA/adapter attachment, the training loop)
follows the documented Hugging Face Qwen2.5-VL + peft pattern `scripts/train_qlora.py`/
`scripts/run_probe.py` already use, but -- like those two scripts -- has never executed against
real weights; treat it as a documented, reasonable starting point, not a verified integration.

REAL INVOCATION (GPU box only, never here):
    from scripts.run_probe import _load_tower  # or: exec via importlib, see _load_run_probe_module
    model, processor, dtype = _load_tower("Qwen/Qwen2.5-VL-3B-Instruct", "cuda", "bfloat16")
    import scripts.train_merger_adapter as tma
    report = tma.run_design_a(model, processor, dtype=dtype, device="cuda")
    # or, for the actual gate:
    report = tma.run_design_b(model, processor, dtype=dtype, device="cuda", variant="B1")
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Allow running as `python scripts/train_merger_adapter.py` from anywhere without an editable
# install -- see scripts/gen_dataset.py's identical comment for why this is needed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from heliogram.codec import PATCH_SIZE  # noqa: E402 -- no heavy deps
from heliogram.dataset import generate_examples, target_to_symbols  # noqa: E402 -- no heavy deps
from heliogram.harness import CORRUPTIONS  # noqa: E402 -- no heavy deps
from heliogram.probe import (  # noqa: E402 -- no heavy deps (pure numpy)
    evaluate_cell,
    merged_token_labels,
    rs_symbol_error_budget,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

__all__ = [
    "PREMERGER_BASELINE_SOURCE",
    "PREMERGER_CLEAN_SYMBOL_ERROR",
    "PREMERGER_JPEG_Q70_SYMBOL_ERROR",
    "POSTMERGER_BASELINE_SOURCE",
    "POSTMERGER_SYMBOL_ERROR_RANGE",
    "HONEST_CAVEAT_TEXT",
    "estimate_runtime_seconds",
    "estimate_cost_usd",
    "check_budget",
    "check_alignment_sanity",
    "RunConfig",
    "build_run_config",
    "format_config_report",
    "ReadoutFitResult",
    "fit_quad_readout_mlp",
    "run_design_a",
    "run_design_b",
    "build_parser",
    "main",
]

# --------------------------------------------------------------------------------------------
# CITED baselines (DATA HONESTY: read from committed reports, never recomputed by this file --
# see module docstring's "NEVER INVENT A NUMBER" paragraph).
# --------------------------------------------------------------------------------------------

PREMERGER_BASELINE_SOURCE = "probe_report_premerger.md (see also docs/FINDINGS.md section 3)"
PREMERGER_CLEAN_SYMBOL_ERROR = 0.1344  # palette=16, clean, pre-merger tap point
PREMERGER_JPEG_Q70_SYMBOL_ERROR = 0.1902  # palette=16, jpeg_q70, pre-merger tap point

POSTMERGER_BASELINE_SOURCE = (
    "probe_report.md (3B) / probe_report_7b.md (7B), see also docs/FINDINGS.md section 3"
)
POSTMERGER_SYMBOL_ERROR_RANGE = (0.6551, 0.7358)  # palette=16, clean, post-merger, stock frozen

RS_NSYM_DEFAULT = 32

ALIGNMENT_TOLERANCE = 0.02  # deterministic tolerance band around PREMERGER_CLEAN_SYMBOL_ERROR

HONEST_CAVEAT_TEXT = (
    "A trained readout head is NOT the LM. Design-B success means the info CAN be made "
    "recoverable at the LM boundary by a cheap merger -- the go/no-go flips to 'go' and the "
    "fine-tune question REOPENS -- it does NOT prove the LM zero-shot uses those symbols or that "
    "the economics win. Necessary, not sufficient. If Design B stays well above budget (RS "
    "budget and/or the stock frozen-merger baseline), the negative result is STRONGER than Step "
    "0's own merged-probe fail: even a cheaply-trained, code-aware merger can't carry the "
    "signal -- scoped to Qwen2.5-VL."
)


# --------------------------------------------------------------------------------------------
# Cost/time budget estimate -- pure Python, no torch, fully unit-testable (see module docstring's
# "COST/TIME BUDGET" paragraph).
# --------------------------------------------------------------------------------------------

GPU_HOURLY_RATE_USD_DEFAULT = 2.0  # a rented-GPU rate assumption; matches RUNBOOK-GPU.md's own
# "~$1-2" framing for a single Step-0-scale GPU session, not a claim about any specific vendor's
# current pricing.

BUDGET_CAP_USD_DEFAULT = 40.0  # tens-of-dollars Step-0 ethos -- matches RUNBOOK-GPU.md section
# 2.5's own "~$15-40" framing for the (much larger) full p16 merger curriculum; this file's
# per-cell diagnostic run should cost a small fraction of that ceiling by design.

# Rough, deliberately conservative per-unit-of-work wall-clock assumptions (seconds), used only to
# produce a projected estimate BEFORE any GPU is touched -- not a measured benchmark (there is no
# GPU here to benchmark against). Design A does ONE frozen forward pass per image (no backprop
# through the tower) plus a CPU-cheap numpy head fit (negligible GPU time, left out of the
# estimate); Design B does `epochs` forward+backward passes per TRAINING image through (part of)
# the vision tower plus the head, which is the whole reason it needs a budget gate Design A does
# not.
DESIGN_A_SECONDS_PER_IMAGE = 0.5
DESIGN_B_SECONDS_PER_TRAIN_STEP = 2.5
DESIGN_B_SECONDS_PER_EVAL_IMAGE = 0.5


def estimate_runtime_seconds(
    design: str, n_train_images: int, n_test_images: int, epochs: int
) -> float:
    """Deterministic, pure-Python projected wall-clock time (seconds) for one `run_design_a`/
    `run_design_b` call, given the same knobs that actually determine its real workload size.
    Never touches torch, never measures anything -- see the module-level constants' comments for
    the (conservative, unverified -- no GPU here to verify against) per-unit-of-work assumptions
    this multiplies out. Raises ValueError for any `design` other than 'a'/'b'."""
    if n_train_images < 0 or n_test_images < 0 or epochs < 0:
        raise ValueError(
            f"n_train_images ({n_train_images}), n_test_images ({n_test_images}), and epochs "
            f"({epochs}) must all be >= 0"
        )
    if design == "a":
        return (n_train_images + n_test_images) * DESIGN_A_SECONDS_PER_IMAGE
    if design == "b":
        return (
            n_train_images * epochs * DESIGN_B_SECONDS_PER_TRAIN_STEP
            + n_test_images * DESIGN_B_SECONDS_PER_EVAL_IMAGE
        )
    raise ValueError(f"design must be 'a' or 'b', got {design!r}")


def estimate_cost_usd(
    design: str,
    n_train_images: int,
    n_test_images: int,
    epochs: int,
    gpu_hourly_rate_usd: float = GPU_HOURLY_RATE_USD_DEFAULT,
) -> float:
    """Projected USD cost = projected hours * `gpu_hourly_rate_usd`. Pure Python, deterministic
    for fixed inputs -- see `estimate_runtime_seconds` for the time half of this."""
    if gpu_hourly_rate_usd < 0:
        raise ValueError(f"gpu_hourly_rate_usd must be >= 0, got {gpu_hourly_rate_usd!r}")
    seconds = estimate_runtime_seconds(design, n_train_images, n_test_images, epochs)
    hours = seconds / 3600.0
    return hours * gpu_hourly_rate_usd


def check_budget(
    estimated_cost_usd: float, cap_usd: float = BUDGET_CAP_USD_DEFAULT
) -> None:
    """ABORT (raise RuntimeError) before any run would start whenever the projected cost exceeds
    `cap_usd` -- the Step-0 ethos (see module docstring): a clean-image FAIL/budget-blowout here
    is meant to be caught for a few dollars, not discovered after tens of GPU-hours. Never
    silently proceeds over budget; there is no override short of raising `cap_usd` explicitly."""
    if estimated_cost_usd > cap_usd:
        raise RuntimeError(
            f"projected cost ${estimated_cost_usd:.2f} exceeds the tens-of-dollars Step-0 budget "
            f"cap (${cap_usd:.2f}) -- ABORTING before any GPU spend. Lower --n-train-images/"
            "--n-test-images/--epochs, or pass a higher --budget-cap-usd if you have reviewed "
            "and explicitly accept the higher projected spend."
        )


# --------------------------------------------------------------------------------------------
# Alignment sanity assert -- pure numeric comparison is testable without a model; the real
# measurement it checks (`_measure_premerger_clean_symbol_error`) needs one (see below).
# --------------------------------------------------------------------------------------------


def check_alignment_sanity(
    measured_clean_symbol_error: float,
    target: float = PREMERGER_CLEAN_SYMBOL_ERROR,
    tolerance: float = ALIGNMENT_TOLERANCE,
) -> None:
    """Pure-Python guard: raises RuntimeError unless `measured_clean_symbol_error` reproduces the
    committed `probe_report_premerger.md` palette=16/clean/pre_merger number (`target`) within
    `tolerance`. Called by both `run_design_a` and `run_design_b` (via
    `_measure_premerger_clean_symbol_error`) BEFORE either trusts a single new number it computes
    -- see module docstring's ALIGNMENT SANITY ASSERT paragraph for why: a fresh GPU session
    whose transformers/peft version, model id, or window-shuffle handling has drifted since that
    report was generated should fail LOUDLY here, not silently report an incomparable number."""
    if abs(measured_clean_symbol_error - target) > tolerance:
        raise RuntimeError(
            "ALIGNMENT SANITY CHECK FAILED: this run's palette=16, clean, pre-merger linear-probe "
            f"symbol error ({measured_clean_symbol_error:.4f}) does not reproduce the committed "
            f"{PREMERGER_BASELINE_SOURCE} number (target {target:.4f} +/- {tolerance:.4f}). "
            "Something about the alignment (window-shuffle unshuffling, merge-unit grouping, "
            "model/processor/transformers version, or palette) has drifted since that report was "
            "generated -- DO NOT TRUST any new number this run produces until this reproduces. "
            "See scripts/run_probe.py's _extract_pre_merger_embeddings/_match_reverse_indices "
            "docstrings for what could have changed."
        )


# --------------------------------------------------------------------------------------------
# Config -- pure Python, no torch.
# --------------------------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Pure-Python, fully-validated run configuration -- everything `main()` needs to print a
    cost estimate and a run report BEFORE (and regardless of whether) a real model is ever
    available. `estimated_cost_usd` is computed in `__post_init__` (never left for a caller to
    forget), so a `RunConfig` is always internally consistent with `estimate_cost_usd`."""

    design: str
    model_id: str
    palette: int
    seed: int
    n_train_images: int
    n_test_images: int
    payload_size: int
    corruptions: Tuple[str, ...]
    epochs: int
    hidden_dim: int
    lora_rank: int
    lora_alpha: int
    lora_variant: str
    nsym: int
    gpu_hourly_rate_usd: float
    budget_cap_usd: float
    estimated_cost_usd: float = field(init=False)

    def __post_init__(self) -> None:
        if self.design not in ("a", "b"):
            raise ValueError(f"design must be 'a' or 'b', got {self.design!r}")
        if self.palette != 16:
            raise ValueError(
                f"palette={self.palette}: this experiment is scoped to palette=16 ONLY -- "
                f"{PREMERGER_BASELINE_SOURCE} measured palette=256 already at/near chance "
                "pre-merger (80.8% clean symbol error), so no merger-side adapter can recover a "
                "palette this coarse-grained a probe already showed the vision BLOCKS themselves "
                "discard. See heliogram/probe.py and RUNBOOK-GPU.md section 2.5 for the full "
                "localization argument."
            )
        if self.lora_variant not in ("B1", "B2"):
            raise ValueError(
                f"lora_variant must be 'B1' (merger-LoRA, preferred) or 'B2' (parallel adapter), "
                f"got {self.lora_variant!r}"
            )
        unknown = [c for c in self.corruptions if c not in CORRUPTIONS]
        if unknown:
            raise ValueError(f"unknown corruption(s) {unknown}; choose from {list(CORRUPTIONS)}")
        if not set(self.corruptions) <= {"clean", "jpeg_q70"}:
            # Not a hard error -- only "clean"/"jpeg_q70" are the two corruptions
            # probe_report_premerger.md actually measured a surviving pre-merger signal under at
            # palette=16 (see PREMERGER_JPEG_Q70_SYMBOL_ERROR); anything else is an unmeasured
            # regime at this tap point, same caveat scripts/train_qlora.py's
            # _P16_MEASURED_CORRUPTION_NAMES documents for the same reason.
            pass
        self.estimated_cost_usd = estimate_cost_usd(
            self.design,
            self.n_train_images,
            self.n_test_images,
            self.epochs,
            self.gpu_hourly_rate_usd,
        )


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Pure-Python: argparse Namespace -> validated RunConfig. No torch, no model, no I/O."""
    corruptions = tuple(c.strip() for c in args.corruptions.split(",") if c.strip())
    return RunConfig(
        design=args.design,
        model_id=args.model_id,
        palette=args.palette,
        seed=args.seed,
        n_train_images=args.n_train_images,
        n_test_images=args.n_test_images,
        payload_size=args.payload_size,
        corruptions=corruptions,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_variant=args.lora_variant,
        nsym=args.nsym,
        gpu_hourly_rate_usd=args.gpu_hourly_rate_usd,
        budget_cap_usd=args.budget_cap_usd,
    )


def format_config_report(config: RunConfig) -> str:
    """Human-readable run-report header: seed, full config, projected cost, and the frozen-
    merger baseline sources this run's alignment check and (for Design B) final verdict are
    measured against -- printed by `main()` BEFORE any model-loading attempt, and returned inside
    every `run_design_a`/`run_design_b` report dict (see module docstring's "Log/emit seed,
    config, and the frozen-merger baseline source" requirement)."""
    lines = [
        "=== scripts/train_merger_adapter.py: run config ===",
        f"design: {config.design} (variant {config.lora_variant} if design=='b')",
        f"model_id: {config.model_id}",
        f"palette: {config.palette} (pinned; see RunConfig.__post_init__)",
        f"seed: {config.seed}",
        f"corruptions: {list(config.corruptions)}",
        f"n_train_images: {config.n_train_images}, n_test_images: {config.n_test_images}, "
        f"payload_size: {config.payload_size}",
        f"epochs: {config.epochs}, hidden_dim: {config.hidden_dim}, "
        f"lora_rank: {config.lora_rank}, lora_alpha: {config.lora_alpha}, nsym: {config.nsym}",
        f"pre-merger alignment baseline (clean): {PREMERGER_CLEAN_SYMBOL_ERROR:.4f} "
        f"(source: {PREMERGER_BASELINE_SOURCE})",
        f"post-merger stock frozen-merger baseline (clean): "
        f"{POSTMERGER_SYMBOL_ERROR_RANGE[0]:.4f}-{POSTMERGER_SYMBOL_ERROR_RANGE[1]:.4f} "
        f"(source: {POSTMERGER_BASELINE_SOURCE})",
        f"RS symbol-error budget (nsym={config.nsym}): {rs_symbol_error_budget(config.nsym):.4f}",
        f"projected cost: ${config.estimated_cost_usd:.4f} "
        f"(cap ${config.budget_cap_usd:.2f}, @ ${config.gpu_hourly_rate_usd:.2f}/GPU-hour)",
        "",
        "HONEST-REPORTING CAVEAT: " + HONEST_CAVEAT_TEXT,
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# scripts/run_probe.py alignment-code reuse (never reinvented -- see module docstring).
# --------------------------------------------------------------------------------------------


def _load_run_probe_module():
    """Load scripts/run_probe.py as a plain module -- the SAME technique
    tests/test_probe_contract_cpu.py's `_load_run_probe` helper uses (scripts/ has no
    __init__.py, by design, so this is a file-path load, not a package import). run_probe.py's
    own top-level imports are torch-free (argparse/json/numpy/heliogram only -- see its module
    docstring), so loading it here never pulls in torch; only actually CALLING its
    `_load_tower`/`_extract_embeddings`/etc. does (their torch imports are local to those
    functions, same convention as everywhere else in this repo)."""
    spec = importlib.util.spec_from_file_location("run_probe", _SCRIPTS_DIR / "run_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_probe", module)
    spec.loader.exec_module(module)
    return module


def _bind_run_probe_alignment_helpers() -> Dict[str, Any]:
    """Binds the SPECIFIC names this file reuses from scripts/run_probe.py -- the window-shuffle
    alignment logic (CPU-verified in tests/test_probe_contract_cpu.py) is IMPORTED here, never
    reimplemented: `_extract_pre_merger_embeddings`, `_match_reverse_indices`, `_load_tower`,
    `_extract_embeddings`, `MERGE`, plus `_resolve_visual_tower`/`_merged_embeddings_tensor`/
    `_cell_arrays` (used by Design B's gradient-enabled forward pass and by the shared alignment
    check, respectively -- see module docstring's "ALIGNMENT CODE REUSE" paragraph)."""
    rp = _load_run_probe_module()
    return {
        "_extract_pre_merger_embeddings": rp._extract_pre_merger_embeddings,
        "_match_reverse_indices": rp._match_reverse_indices,
        "_load_tower": rp._load_tower,
        "_extract_embeddings": rp._extract_embeddings,
        "MERGE": rp.MERGE,
        "_resolve_visual_tower": rp._resolve_visual_tower,
        "_merged_embeddings_tensor": rp._merged_embeddings_tensor,
        "_cell_arrays": rp._cell_arrays,
    }


def _measure_premerger_clean_symbol_error(
    helpers: Dict[str, Any],
    model: Any,
    processor: Any,
    dtype: Any,
    device: str,
    palette: int,
    seed: int,
    n_train_images: int,
    n_test_images: int,
    payload_size: int,
) -> float:
    """Re-runs the EXACT (palette=16, clean, pre_merger) probe cell probe_report_premerger.md was
    generated from, via `_cell_arrays`/`evaluate_cell` (both reused, not reinvented -- see module
    docstring), and returns its test-split symbol error for `check_alignment_sanity` to judge.
    Same disjoint train/test seed-range convention as scripts/run_probe.py's `main()`."""
    clean_fn = CORRUPTIONS["clean"]
    cell_arrays = helpers["_cell_arrays"]
    X_tr, y_tr = cell_arrays(
        model, processor, dtype, device, palette, "clean", clean_fn,
        n_train_images, payload_size, seed_base=seed + 1_000, stage="pre_merger",
    )
    X_te, y_te = cell_arrays(
        model, processor, dtype, device, palette, "clean", clean_fn,
        n_test_images, payload_size, seed_base=seed + 2_000_000, stage="pre_merger",
    )
    cell = evaluate_cell(palette, "clean", X_tr, y_tr, X_te, y_te, seed=seed)
    return cell.fit.symbol_error


def _group_quads(pre_merger_embeddings: np.ndarray) -> np.ndarray:
    """Group per-PATCH pre-merger embeddings (raster order, row `m * 4 + p` -- see
    scripts/run_probe.py's `_extract_pre_merger_embeddings` docstring for why that ordering
    holds; REUSED here, not reinvented) into Design-A's readout-head input: one row per merged
    token, its 4 within-unit patch embeddings concatenated (TL, TR, BL, BR, that fixed order) into
    a single `4 * hidden`-wide row. `pre_merger_embeddings.shape[0]` must be a multiple of 4."""
    n_patches, hidden = pre_merger_embeddings.shape
    if n_patches % 4 != 0:
        raise ValueError(
            f"expected a multiple of 4 patch rows (one 2x2 merge unit each), got {n_patches}"
        )
    return pre_merger_embeddings.reshape(n_patches // 4, 4 * hidden)


# --------------------------------------------------------------------------------------------
# Design A: frozen-feature readout head (pure numpy fit; extraction needs a real model).
# --------------------------------------------------------------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _masked_symbol_error(logits: np.ndarray, y: np.ndarray) -> float:
    pred = logits.argmax(axis=-1)
    valid = y >= 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("nan")
    return float((pred[valid] != y[valid]).sum() / n_valid)


@dataclass
class ReadoutFitResult:
    """Outcome of one `fit_quad_readout_mlp` fit -- same field shape as
    `heliogram.probe.ProbeFitResult` (symbol_error/train_symbol_error headline pair) plus the
    hidden_dim this head used, so an underfit head is distinguishable from an underpowered one."""

    symbol_error: float
    train_symbol_error: float
    n_train_positions: int
    n_test_positions: int
    hidden_dim: int
    epochs: int
    seed: int


def fit_quad_readout_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    palette: int,
    hidden_dim: int = 64,
    seed: int = 0,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 0.05,
    momentum: float = 0.9,
    l2: float = 1e-4,
) -> ReadoutFitResult:
    """Design-A's readout head: concat(4 patch embeddings) -> one hidden ReLU layer -> 4 x
    `palette`-way logits, trained with plain-numpy minibatch SGD -- mirrors
    `heliogram.probe.fit_linear_probe`'s exact style/hyperparameter conventions (boring,
    deterministic SGD; train-statistics-only standardization; no schedule tuning, no test-set
    peeking) with one hidden layer inserted, since Design A's whole point is testing whether a
    NONLINEAR readout recovers more signal than Step 0's linear probe already measured.

    `X_*`: `(n_merged_tokens, 4 * vision_hidden)` -- already-concatenated quad features (see
    `_group_quads`). `y_*`: `(n_merged_tokens, 4)` symbols, with `-1` marking calibration-row
    positions to EXCLUDE from loss and metrics (same convention as
    `heliogram.probe.merged_token_labels`).

    THIS IS NOT THE LM (see module docstring's HONEST-REPORTING CAVEAT): a head trained on
    FROZEN embeddings only tests whether the information is linearly-plus-one-hidden-layer
    recoverable at this tap point -- it says nothing about whether an LM ever reads it this way.
    """
    if X_train.ndim != 2 or y_train.ndim != 2 or X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train (N,D) and y_train (N,K) must align on N")
    if X_test.shape[1] != X_train.shape[1] or y_test.shape[1] != y_train.shape[1]:
        raise ValueError("train/test feature dims and position counts must match")
    if (y_train >= 0).sum() == 0:
        raise ValueError("y_train has no valid (>=0) labels -- nothing to fit")

    rng = np.random.default_rng(seed)
    n, dim = X_train.shape
    k = y_train.shape[1]

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-6
    Xtr = ((X_train - mu) / sd).astype(np.float64)
    Xte = ((X_test - mu) / sd).astype(np.float64)

    w1 = rng.normal(0.0, 0.05, size=(dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    w2 = np.zeros((hidden_dim, k * palette))
    b2 = np.zeros(k * palette)
    vw1, vb1 = np.zeros_like(w1), np.zeros_like(b1)
    vw2, vb2 = np.zeros_like(w2), np.zeros_like(b2)

    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = Xtr[idx]
            yb = y_train[idx]

            h_pre = xb @ w1 + b1
            h = np.maximum(h_pre, 0.0)
            logits = (h @ w2 + b2).reshape(len(idx), k, palette)
            probs = _softmax(logits)

            grad_logits = probs.copy()
            valid = yb >= 0
            rows, cols = np.nonzero(valid)
            grad_logits[rows, cols, yb[rows, cols]] -= 1.0
            grad_logits[~valid] = 0.0
            n_valid = max(int(valid.sum()), 1)
            grad_logits = grad_logits.reshape(len(idx), k * palette) / n_valid

            gw2 = h.T @ grad_logits + l2 * w2
            gb2 = grad_logits.sum(axis=0)
            grad_h = (grad_logits @ w2.T) * (h_pre > 0)
            gw1 = xb.T @ grad_h + l2 * w1
            gb1 = grad_h.sum(axis=0)

            vw2 = momentum * vw2 - lr * gw2
            vb2 = momentum * vb2 - lr * gb2
            vw1 = momentum * vw1 - lr * gw1
            vb1 = momentum * vb1 - lr * gb1
            w2 += vw2
            b2 += vb2
            w1 += vw1
            b1 += vb1

    def _forward(X: np.ndarray) -> np.ndarray:
        h = np.maximum(X @ w1 + b1, 0.0)
        return (h @ w2 + b2).reshape(X.shape[0], k, palette)

    train_err = _masked_symbol_error(_forward(Xtr), y_train)
    test_err = _masked_symbol_error(_forward(Xte), y_test)

    return ReadoutFitResult(
        symbol_error=test_err,
        train_symbol_error=train_err,
        n_train_positions=int((y_train >= 0).sum()),
        n_test_positions=int((y_test >= 0).sum()),
        hidden_dim=hidden_dim,
        epochs=epochs,
        seed=seed,
    )


def run_design_a(
    model: Any = None,
    processor: Any = None,
    dtype: Any = None,
    device: str = "cuda",
    palette: int = 16,
    corruptions: Sequence[str] = ("clean", "jpeg_q70"),
    n_train_images: int = 6,
    n_test_images: int = 3,
    payload_size: int = 1024,
    hidden_dim: int = 64,
    epochs: int = 60,
    seed: int = 0,
    nsym: int = RS_NSYM_DEFAULT,
) -> Dict[str, Any]:
    """Design A real-run entry point (see module docstring). REFUSE-WITHOUT-MODEL: `model`,
    `processor`, and `dtype` must be real, already-loaded objects (e.g. from
    `scripts/run_probe.py`'s `_load_tower`) -- passing any of them as `None` (the default; this
    function is never given a real model by this file's own CLI, see module docstring's "REAL
    INVOCATION" note) raises ValueError immediately, mirroring `heliogram.vlm.
    zero_shot_symbol_error`'s and `heliogram.instruments.injection_bench.
    measure_behavioral_capacity`'s guard exactly. Never fabricates a symbol-error number.

    On a real GPU box this would: (1) re-run and check the palette=16/clean/pre_merger alignment
    sanity assert (`check_alignment_sanity`) against `probe_report_premerger.md`; (2) for each
    requested corruption, extract PRE-merger per-patch embeddings (`_extract_embeddings(stage=
    "pre_merger")`, reused from scripts/run_probe.py), group them into merge-unit quads
    (`_group_quads`), and fit `fit_quad_readout_mlp` against the RS budget. Returns a report dict
    including the config, the alignment-check result, per-corruption symbol errors, and
    `HONEST_CAVEAT_TEXT`.
    """
    if model is None or processor is None or dtype is None:
        raise ValueError(
            "run_design_a requires a real, already-loaded model, processor, and torch dtype "
            "(got model=None/processor=None/dtype=None) -- it never fabricates a readout-head "
            "result. Load a frozen Qwen2.5-VL tower the same way scripts/run_probe.py's main() "
            "does (see that script's _load_tower) and pass the resulting (model, processor, "
            "dtype) here. This mirrors heliogram.vlm.zero_shot_symbol_error's and heliogram."
            "instruments.injection_bench.measure_behavioral_capacity's guard exactly -- see this "
            "module's docstring's REFUSE-WITHOUT-MODEL paragraph."
        )
    if palette != 16:
        raise ValueError(
            f"palette={palette}: this experiment is scoped to palette=16 ONLY -- see "
            f"RunConfig.__post_init__'s identical guard for the full argument "
            f"({PREMERGER_BASELINE_SOURCE})."
        )
    unknown = [c for c in corruptions if c not in CORRUPTIONS]
    if unknown:
        raise ValueError(f"unknown corruption(s) {unknown}; choose from {list(CORRUPTIONS)}")

    helpers = _bind_run_probe_alignment_helpers()

    clean_measured = _measure_premerger_clean_symbol_error(
        helpers, model, processor, dtype, device, palette, seed,
        n_train_images, n_test_images, payload_size,
    )
    check_alignment_sanity(clean_measured)

    rs_budget = rs_symbol_error_budget(nsym)
    cell_arrays = helpers["_cell_arrays"]
    per_corruption = []
    for cname in corruptions:
        cfn = CORRUPTIONS[cname]
        X_tr_patch, y_tr_patch = cell_arrays(
            model, processor, dtype, device, palette, cname, cfn,
            n_train_images, payload_size, seed_base=seed + 1_000, stage="pre_merger",
        )
        X_te_patch, y_te_patch = cell_arrays(
            model, processor, dtype, device, palette, cname, cfn,
            n_test_images, payload_size, seed_base=seed + 2_000_000, stage="pre_merger",
        )
        X_tr, y_tr = _group_quads(X_tr_patch), y_tr_patch.reshape(-1, 4)
        X_te, y_te = _group_quads(X_te_patch), y_te_patch.reshape(-1, 4)

        fit = fit_quad_readout_mlp(
            X_tr, y_tr, X_te, y_te, palette=palette, hidden_dim=hidden_dim,
            seed=seed, epochs=epochs,
        )
        per_corruption.append(
            {
                "corruption": cname,
                "symbol_error": fit.symbol_error,
                "train_symbol_error": fit.train_symbol_error,
                "rs_budget": rs_budget,
                "below_rs_budget": fit.symbol_error <= rs_budget,
            }
        )

    return {
        "design": "A",
        "palette": palette,
        "seed": seed,
        "alignment_check_clean_pre_merger_symbol_error": clean_measured,
        "alignment_check_target": PREMERGER_CLEAN_SYMBOL_ERROR,
        "alignment_check_source": PREMERGER_BASELINE_SOURCE,
        "per_corruption": per_corruption,
        "caveat": HONEST_CAVEAT_TEXT,
    }


# --------------------------------------------------------------------------------------------
# Design B: train the merger (LoRA or parallel adapter) + a jointly-trained torch readout head.
# --------------------------------------------------------------------------------------------

# The exact same two Linear-layer names scripts/train_qlora.py's LORA_MERGER_TARGET_MODULES
# targets (mlp.0/mlp.2, around the merger's GELU) -- duplicated here rather than cross-script-
# imported so this file's torch-dependent path stays independent of scripts/train_qlora.py's
# regex-building machinery, the same "duplicate rather than cross-import a torch-adjacent detail"
# choice heliogram/instruments/injection_bench.py makes for its own prompt constant (see that
# module's docstring). Relative to `visual.merger` itself (not the full dotted model path
# train_qlora.py's regex is anchored against), so a bare LoraConfig(target_modules=...) suffix
# match is correct here without needing that regex machinery at all.
MERGER_LORA_TARGET_MODULES = ["mlp.0", "mlp.2"]


def run_design_b(
    model: Any = None,
    processor: Any = None,
    dtype: Any = None,
    device: str = "cuda",
    palette: int = 16,
    variant: str = "B1",
    corruptions: Sequence[str] = ("clean", "jpeg_q70"),
    n_train_images: int = 6,
    n_test_images: int = 3,
    payload_size: int = 1024,
    hidden_dim: int = 64,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 0,
    nsym: int = RS_NSYM_DEFAULT,
) -> Dict[str, Any]:
    """Design B real-run entry point -- THE ACTUAL GATE (see module docstring). REFUSE-WITHOUT-
    MODEL: same guard as `run_design_a`, raises ValueError immediately on `model`/`processor`/
    `dtype`=None. `variant`: `"B1"` (default, preferred) LoRA-tunes `model.visual.merger.mlp`'s
    two Linear layers (`MERGER_LORA_TARGET_MODULES`) via peft; `"B2"` attaches a small parallel
    residual adapter (`_ParallelMergerAdapter`) around the frozen merger instead (for when peft
    is unavailable, or B1 underperforms -- see that class's docstring).

    On a real GPU box this would: (1) re-run the same alignment sanity assert `run_design_a` does
    (`check_alignment_sanity`); (2) freeze the whole model, then unfreeze only the LoRA/adapter
    parameters plus a jointly-trained `_QuadReadoutHead` sitting on the POST-merger token; (3)
    for each requested corruption, train (cross-entropy summed over the 4 constituent symbol
    positions per merged token, masking calibration-row positions) for `epochs` passes over
    `n_train_images` freshly generated examples, then evaluate held-out symbol error against the
    RS budget AND the stock frozen-merger baseline (`POSTMERGER_SYMBOL_ERROR_RANGE`, cited, never
    remeasured). Returns a report dict, same shape as `run_design_a`'s, plus
    `stock_post_merger_baseline_*` fields and `HONEST_CAVEAT_TEXT`.

    UNTESTED (same caveat as scripts/train_qlora.py's own module docstring): the exact
    peft/LoraConfig call sequence and the gradient-enabled forward pass
    (`_forward_post_merger_quad_logits`) follow the documented Hugging Face + peft pattern, but
    have never executed against real weights here."""
    if model is None or processor is None or dtype is None:
        raise ValueError(
            "run_design_b requires a real, already-loaded model, processor, and torch dtype "
            "(got model=None/processor=None/dtype=None) -- it never fabricates a training "
            "result. Load a frozen Qwen2.5-VL tower the same way scripts/run_probe.py's main() "
            "does (see that script's _load_tower) and pass the resulting (model, processor, "
            "dtype) here. This mirrors heliogram.vlm.zero_shot_symbol_error's and heliogram."
            "instruments.injection_bench.measure_behavioral_capacity's guard exactly -- see this "
            "module's docstring's REFUSE-WITHOUT-MODEL paragraph."
        )
    if palette != 16:
        raise ValueError(
            f"palette={palette}: this experiment is scoped to palette=16 ONLY -- see "
            f"RunConfig.__post_init__'s identical guard for the full argument "
            f"({PREMERGER_BASELINE_SOURCE})."
        )
    if variant not in ("B1", "B2"):
        raise ValueError(
            f"variant must be 'B1' (merger-LoRA, preferred) or 'B2' (parallel adapter), got "
            f"{variant!r}"
        )
    unknown = [c for c in corruptions if c not in CORRUPTIONS]
    if unknown:
        raise ValueError(f"unknown corruption(s) {unknown}; choose from {list(CORRUPTIONS)}")

    helpers = _bind_run_probe_alignment_helpers()

    clean_measured = _measure_premerger_clean_symbol_error(
        helpers, model, processor, dtype, device, palette, seed,
        n_train_images, n_test_images, payload_size,
    )
    check_alignment_sanity(clean_measured)

    import torch

    visual = helpers["_resolve_visual_tower"](model)
    for p in model.parameters():
        p.requires_grad_(False)

    if variant == "B1":
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            target_modules=MERGER_LORA_TARGET_MODULES,
        )
        trainable_module = get_peft_model(visual.merger, lora_config)
        out_features = visual.merger.mlp[-1].out_features
    else:
        trainable_module = _ParallelMergerAdapter(visual.merger, rank=lora_rank)
        out_features = trainable_module.out_features
    for p in trainable_module.parameters():
        p.requires_grad_(True)

    head = _QuadReadoutHead(in_features=out_features, hidden_dim=hidden_dim, palette=palette)
    optimizer = torch.optim.AdamW(
        [p for p in trainable_module.parameters() if p.requires_grad] + list(head.parameters()),
        lr=lr,
    )

    rs_budget = rs_symbol_error_budget(nsym)
    per_corruption = []
    for cname in corruptions:
        cfn = CORRUPTIONS[cname]
        train_examples = list(
            generate_examples(
                n_train_images, palettes=[palette], subpatches=[1], payload_sizes=[payload_size],
                seed=seed + 1_000, corruptions={"clean": lambda im: im, cname: cfn},
                corruption_prob=0.0 if cname == "clean" else 1.0,
            )
        )
        test_examples = list(
            generate_examples(
                n_test_images, palettes=[palette], subpatches=[1], payload_sizes=[payload_size],
                seed=seed + 2_000_000, corruptions={"clean": lambda im: im, cname: cfn},
                corruption_prob=0.0 if cname == "clean" else 1.0,
            )
        )

        for _ in range(epochs):
            for ex in train_examples:
                optimizer.zero_grad()
                logits = _forward_post_merger_quad_logits(
                    helpers, model, processor, dtype, device, ex, palette, head
                )
                labels_t = _example_quad_labels(ex, palette, helpers["MERGE"], logits.device)
                loss = _quad_cross_entropy(logits, labels_t)
                loss.backward()
                optimizer.step()

        test_errors = []
        with torch.no_grad():
            for ex in test_examples:
                logits = _forward_post_merger_quad_logits(
                    helpers, model, processor, dtype, device, ex, palette, head
                )
                labels_t = _example_quad_labels(ex, palette, helpers["MERGE"], logits.device)
                test_errors.append(_quad_symbol_error(logits, labels_t))
        symbol_error = float(sum(test_errors) / len(test_errors)) if test_errors else float("nan")
        per_corruption.append(
            {
                "corruption": cname,
                "symbol_error": symbol_error,
                "rs_budget": rs_budget,
                "below_rs_budget": symbol_error <= rs_budget,
                "below_stock_post_merger_baseline": symbol_error < POSTMERGER_SYMBOL_ERROR_RANGE[0],
            }
        )

    return {
        "design": "B",
        "variant": variant,
        "palette": palette,
        "seed": seed,
        "alignment_check_clean_pre_merger_symbol_error": clean_measured,
        "alignment_check_target": PREMERGER_CLEAN_SYMBOL_ERROR,
        "alignment_check_source": PREMERGER_BASELINE_SOURCE,
        "stock_post_merger_baseline_range": POSTMERGER_SYMBOL_ERROR_RANGE,
        "stock_post_merger_baseline_source": POSTMERGER_BASELINE_SOURCE,
        "per_corruption": per_corruption,
        "caveat": HONEST_CAVEAT_TEXT,
    }


def _example_quad_labels(example: Any, palette: int, merge: int, torch_device: Any):
    """`heliogram.dataset.Example` -> `(n_merged_tokens, 4)` torch label tensor, via
    `heliogram.probe.merged_token_labels` (reused, not reinvented)."""
    import torch

    width = example.image.width // PATCH_SIZE
    height = example.image.height // PATCH_SIZE
    symbols = target_to_symbols(example.target, palette)
    labels = merged_token_labels(width, height, symbols, merge=merge)
    return torch.tensor(labels, dtype=torch.long, device=torch_device)


def _forward_post_merger_quad_logits(
    helpers: Dict[str, Any], model: Any, processor: Any, dtype: Any, device: str,
    example: Any, palette: int, head: "torch.nn.Module",
):
    """Gradient-ENABLED forward pass: image -> POST-merger tokens (with LoRA/adapter active) ->
    `head` -> `(n_merged_tokens, 4, palette)` logits. NEW code (not a call to scripts/
    run_probe.py's `_extract_embeddings`, which wraps its forward pass in `torch.no_grad()` --
    exactly right for a frozen-tower PROBE, exactly wrong for training, since it would silently
    block gradients from ever reaching the merger LoRA/adapter parameters) -- but still reuses
    `_resolve_visual_tower`/`_merged_embeddings_tensor` for tower resolution and merged-token
    output-field selection, the same alignment-sensitive contracts `_extract_embeddings` itself
    relies on, rather than re-deriving them independently (see module docstring's "ALIGNMENT CODE
    REUSE" paragraph)."""
    visual = helpers["_resolve_visual_tower"](model)
    out = processor.image_processor(images=[example.image.convert("RGB")], return_tensors="pt")
    grid_thw = out["image_grid_thw"].to(device)
    pixel_values = out["pixel_values"].to(device=device, dtype=dtype)

    width = example.image.width // PATCH_SIZE
    height = example.image.height // PATCH_SIZE
    n_units = (height // helpers["MERGE"]) * (width // helpers["MERGE"])

    visual_out = visual(pixel_values, grid_thw=grid_thw)
    merged = helpers["_merged_embeddings_tensor"](visual_out, n_units)
    return head(merged.to(next(head.parameters()).dtype))


def _quad_cross_entropy(logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
    """Cross-entropy summed over the 4 constituent symbol positions per merged token, masking
    calibration-row (`label < 0`) positions -- the Design-B training objective the module
    docstring/HANDOVER context specifies."""
    import torch.nn.functional as F

    valid = labels >= 0
    if not bool(valid.any()):
        raise RuntimeError(
            "no valid (non-calibration-row) labels in this batch -- cannot compute a loss"
        )
    return F.cross_entropy(logits[valid], labels[valid], reduction="sum")


def _quad_symbol_error(logits: "torch.Tensor", labels: "torch.Tensor") -> float:
    valid = labels >= 0
    if not bool(valid.any()):
        return float("nan")
    pred = logits.argmax(dim=-1)
    return float((pred[valid] != labels[valid]).float().mean().item())


class _QuadReadoutHead:
    """Design-B's jointly-trained torch readout head: one POST-merger token
    (`out_hidden_size`-wide) -> one hidden ReLU layer -> 4 x `palette`-way logits. Unlike Design
    A's frozen-feature `fit_quad_readout_mlp` (plain numpy, no gradients into the tower), this is
    a real `torch.nn.Module` specifically so gradients flow back through it INTO the merger LoRA/
    adapter parameters during joint training -- the one thing Design A structurally cannot do.
    Defined with a lazy torch import inside `__new__`-style construction (see `__init__`) so this
    class is DEFINABLE without torch installed (referencing it in a docstring/annotation, or in
    type-only contexts, never imports torch) but only ever INSTANTIATED from inside
    `run_design_b`, which has already required torch to be present."""

    def __new__(cls, in_features: int, hidden_dim: int, palette: int):
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 4 * palette),
                )
                self.palette = palette

            def forward(self, merged_tokens):
                n = merged_tokens.shape[0]
                return self.net(merged_tokens).reshape(n, 4, self.palette)

        return _Impl()


class _ParallelMergerAdapter:
    """Design-B variant B2: a small trainable residual adapter run in PARALLEL with the (frozen)
    merger MLP, added to its output -- an alternative to B1's LoRA-on-merger-Linear-layers for
    when peft is unavailable, or B1 underperforms. Only this adapter's own down/up-projection
    parameters are trained; `merger` itself is called under `torch.no_grad()` internally (the
    caller is still responsible for freezing its parameters, same as B1's base model). Zero-
    initialized `up` projection: the adapter starts as an exact identity (zero residual
    contribution) at step 0, so training begins from the untouched frozen-merger baseline rather
    than an arbitrary random perturbation of it. Same lazy-torch-import-in-`__new__` construction
    as `_QuadReadoutHead`, for the same "definable without torch, only ever instantiated once
    torch is already required" reason."""

    def __new__(cls, merger: Any, rank: int = 8):
        import torch
        import torch.nn as nn

        class _Impl(nn.Module):
            def __init__(self):
                super().__init__()
                in_features = merger.mlp[0].in_features
                self.out_features = merger.mlp[-1].out_features
                self.down = nn.Linear(in_features, rank, bias=False)
                self.up = nn.Linear(rank, self.out_features, bias=False)
                nn.init.zeros_(self.up.weight)
                self._merger = merger

            def forward(self, merger_input):
                with torch.no_grad():
                    frozen_out = self._merger(merger_input)
                return frozen_out + self.up(self.down(merger_input))

        return _Impl()


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--design", choices=["a", "b"], default="a",
        help="'a': cheap frozen-feature readout-head diagnostic. 'b': the actual gate -- train "
        "the merger (LoRA or parallel adapter) + a jointly-trained head (default: a)",
    )
    parser.add_argument(
        "--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF model id (default matches scripts/run_probe.py's own default)",
    )
    parser.add_argument(
        "--palette", type=int, default=16,
        help="pinned to 16 (see module docstring's SCOPE paragraph); any other value raises",
    )
    parser.add_argument(
        "--corruptions", default="clean,jpeg_q70",
        help="comma-separated; default is the two corruptions probe_report_premerger.md "
        "actually measured a surviving palette=16 pre-merger signal under",
    )
    parser.add_argument("--n-train-images", type=int, default=6)
    parser.add_argument("--n-test-images", type=int, default=3)
    parser.add_argument("--payload-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Design A: numpy readout-head epochs (cheap, CPU-side). Design B: real training "
        "epochs over the vision tower + head (default lower for design b via the entry-point "
        "default -- override explicitly if you pick --design b)",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="readout-head hidden width")
    parser.add_argument("--lora-rank", type=int, default=8, help="design b only")
    parser.add_argument("--lora-alpha", type=int, default=16, help="design b only")
    parser.add_argument(
        "--lora-variant", choices=["B1", "B2"], default="B1",
        help="design b only: B1 (merger-LoRA via peft, preferred) or B2 (parallel adapter)",
    )
    parser.add_argument("--nsym", type=int, default=RS_NSYM_DEFAULT, help="Reed-Solomon parity bytes")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--gpu-hourly-rate-usd", type=float, default=GPU_HOURLY_RATE_USD_DEFAULT,
        help=f"assumed rented-GPU rate for the cost estimate (default: "
        f"{GPU_HOURLY_RATE_USD_DEFAULT})",
    )
    parser.add_argument(
        "--budget-cap-usd", type=float, default=BUDGET_CAP_USD_DEFAULT,
        help=f"ABORT before any run if the projected cost exceeds this (default: "
        f"{BUDGET_CAP_USD_DEFAULT}, the tens-of-dollars Step-0 ethos)",
    )
    parser.add_argument("--out", default=None, help="optional path to write the run-config report to")
    parser.add_argument("--json", dest="json_out", default=None, help="optional path to write JSON config to")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    print(__doc__)
    print(
        "\n*** GPU REQUIRED. This CLI entry point never loads a model itself -- there is no flag "
        "that hands it one (a live Python model object cannot be passed over argparse). Every "
        "invocation below reaches run_design_a's/run_design_b's explicit model=None guard and "
        "raises before touching torch at all. See this module's docstring's 'REAL INVOCATION' "
        "note for how to actually run either design, on a GPU box, from a driver that has already "
        "loaded a real model. ***\n"
    )

    config = build_run_config(args)
    report_text = format_config_report(config)
    print(report_text)
    if args.out:
        Path(args.out).write_text(report_text + "\n")
        print(f"\nwrote {args.out}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(asdict(config), indent=2))
        print(f"wrote {args.json_out}")

    # ABORT before any run would start if the projected cost is over budget -- BEFORE the
    # (always-firing, in this CLI) model=None guard below, so an over-budget config is caught
    # even though this CLI never gets far enough to spend anything either way.
    check_budget(config.estimated_cost_usd, config.budget_cap_usd)

    entry = run_design_a if config.design == "a" else run_design_b
    kwargs: Dict[str, Any] = dict(
        model=None, processor=None, dtype=None, device=args.device,
        palette=config.palette, corruptions=config.corruptions,
        n_train_images=config.n_train_images, n_test_images=config.n_test_images,
        payload_size=config.payload_size, hidden_dim=config.hidden_dim,
        epochs=config.epochs, seed=config.seed, nsym=config.nsym,
    )
    if config.design == "b":
        kwargs.update(variant=config.lora_variant, lora_rank=config.lora_rank, lora_alpha=config.lora_alpha)
    entry(**kwargs)  # always raises ValueError -- see module docstring's REAL INVOCATION note
    return 0  # unreachable; kept for the same signature convention as every other script's main()


if __name__ == "__main__":
    sys.exit(main())
