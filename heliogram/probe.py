"""heliogram.probe -- the Step-0 frozen-encoder linear probe: CPU-testable core.

THE QUESTION THIS MODULE DECIDES (the cheapest decisive Phase-2 experiment, single-digit
GPU-hours): after heliogram grids pass through a real, FROZEN VLM vision tower (pinned target:
Qwen2.5-VL, whose 2x2 spatial merger folds four 14px ViT patches into one LM-visible token), do
the merged-token embeddings still LINEARLY separate the four patch symbols each token carries?

- If a linear probe with oracle labels can read the symbols out of the frozen embeddings (symbol
  error at or below the Reed-Solomon budget, `rs_symbol_error_budget`), the information survives
  to the LM boundary and a fine-tune only has to teach the language model to READ what is
  already there -- the QLoRA bet is de-risked before a single training-hour is spent.
- If the probe cannot separate the symbols ON CLEAN IMAGES, no LINEARLY-DECODABLE per-patch
  signal survives to this tap point for a LoRA on top of the same frozen tower to exploit
  linearly -- the LM-token accounting branch of the project (see RESULTS.md's "LM-token
  accounting" caveat) is unsupported by this (linear) probe, cheaply. A higher-capacity,
  NONLINEAR probe run at the same tap point could still recover something a linear readout
  misses (see docs/FINDINGS.md Section 5, "Honest limitations") -- that experiment is not run
  here (see scripts/run_probe.py's `--probe-head mlp`, a designed refusing stub), so treat a
  clean-image FAIL as strong negative evidence, not proof that no computation downstream could
  ever extract the symbols.

DATA HONESTY (same rules as everywhere in this repo):
- This module is the model-FREE half: label bookkeeping, the linear probe itself (plain numpy,
  no torch), and the report. It never touches a model and can be fully tested on CPU -- see
  tests/test_probe.py. The model-DEPENDENT half (loading the frozen tower, extracting
  embeddings) lives in scripts/run_probe.py and is UNTESTED against a real model in this repo
  (no GPU here), with the same status as heliogram/vlm.py's model paths.
- A probe PASS is evidence about the ENCODER's embeddings, not a demonstration that the full VLM
  (LM decoding included) can transcribe grids -- that is what the subsequent QLoRA run measures.
  A probe FAIL on clean images, however, is close to conclusive in the negative direction for
  this tower: linear non-separability with oracle labels means no LINEARLY-DECODABLE per-patch
  signal survives to this tap point for a LINEAR fine-tune head of the (frozen) tower's
  consumers to work with -- a higher-capacity/nonlinear probe could still differ (untested here;
  see docs/FINDINGS.md Section 5).
- Token-order assumption: `merged_token_labels` lays labels out in raster order over the merged
  grid (row-major, top-left to bottom-right), matching Qwen2.5-VL's documented merger output
  order. If a tower ordered tokens differently, the probe would score at CHANCE level even on
  clean images -- an unmistakable, fail-loud signature (documented in `merged_token_labels`'s
  docstring) rather than a silently wrong number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .codec import RS_NSIZE

__all__ = [
    "rs_symbol_error_budget",
    "merged_token_labels",
    "ProbeFitResult",
    "fit_linear_probe",
    "ProbeCellReport",
    "evaluate_cell",
    "format_report",
]


def rs_symbol_error_budget(nsym: int = 32) -> float:
    """The sustained symbol-error rate Reed-Solomon can absorb: floor(nsym/2) correctable bytes
    per RS_NSIZE-byte chunk (16/255 ~= 6.27% for the default nsym=32). Symbol errors above this
    rate make decode failure near-certain; below it, decode succeeds (substitution errors only
    -- RS does not correct insertions/deletions, which a per-patch probe cannot produce anyway).
    Exact only for palette=256 (1 symbol == 1 byte); for smaller palettes multiple symbols share
    a byte, making the byte-error rate at a given symbol-error rate slightly HIGHER, so treating
    this as the bar is mildly generous to the code at P<256 -- the same approximation
    heliogram.instruments.bayes_bound documents."""
    return (nsym // 2) / RS_NSIZE


def merged_token_labels(
    width: int, height: int, symbols: Sequence[int], merge: int = 2
) -> np.ndarray:
    """Map per-patch ground-truth symbols onto merged-token positions.

    `width`/`height` are PATCH-grid dimensions of the (even-padded, see
    heliogram.dataset.pad_to_even_patch_grid) image; both must be divisible by `merge`.
    `symbols` is the subpatch=1 data-patch symbol list exactly as heliogram.codec.extract_symbols
    returns it: row-major over data rows 1..height-1 (row 0 is the calibration row and carries no
    data symbol).

    Returns an int64 array of shape (n_merged_tokens, merge*merge): entry [m, p] is the symbol
    under sub-position p (row-major within the merge block: top-left, top-right, bottom-left,
    bottom-right for merge=2) of merged token m, or -1 where that patch is a calibration-row
    patch (position must be EXCLUDED from probe loss and metrics, not predicted).

    Merged tokens are laid out in raster order over the merged grid -- m = mr * (width//merge)
    + mc -- matching Qwen2.5-VL's merger output order. ORDER-ASSUMPTION HONESTY: if a real tower
    emitted tokens in any other order, a probe trained against these labels would sit at chance
    level even on clean images -- a loud, unambiguous failure mode, not a quietly wrong result.
    """
    if merge < 1:
        raise ValueError(f"merge must be >= 1, got {merge!r}")
    if width % merge or height % merge:
        raise ValueError(
            f"patch grid ({width}x{height}) must be divisible by merge={merge} -- pad the image "
            "first (heliogram.dataset.pad_to_even_patch_grid)"
        )
    expected = width * (height - 1)
    if len(symbols) != expected:
        raise ValueError(
            f"symbols has length {len(symbols)}, expected width*(height-1)={expected} for a "
            f"{width}x{height} patch grid at subpatch=1"
        )

    wm = width // merge
    hm = height // merge
    labels = np.full((wm * hm, merge * merge), -1, dtype=np.int64)
    for mr in range(hm):
        for mc in range(wm):
            m = mr * wm + mc
            for dr in range(merge):
                for dc in range(merge):
                    row = mr * merge + dr
                    col = mc * merge + dc
                    if row == 0:
                        continue  # calibration row: no data symbol, stays -1
                    labels[m, dr * merge + dc] = symbols[(row - 1) * width + col]
    return labels


@dataclass
class ProbeFitResult:
    """Outcome of one linear-probe fit. `symbol_error` is the headline: fraction of VALID
    (label >= 0) test positions predicted wrong. `per_position_error` breaks that down by
    sub-position within the merge block (length merge*merge; NaN where a position had no valid
    test labels). `train_symbol_error` is reported so an underfit (train error itself high) is
    distinguishable from a generalization gap."""

    symbol_error: float
    per_position_error: List[float]
    train_symbol_error: float
    n_train_positions: int
    n_test_positions: int
    n_classes: int
    epochs: int
    seed: int


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _masked_error(logits: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    """(overall_error, per_position_error) over valid (y >= 0) entries. logits: (N, K, P);
    y: (N, K). Positions with zero valid entries get NaN per-position error."""
    pred = logits.argmax(axis=-1)
    valid = y >= 0
    n_valid = int(valid.sum())
    overall = float((pred[valid] != y[valid]).sum() / n_valid) if n_valid else float("nan")
    per_pos = []
    for k in range(y.shape[1]):
        vk = valid[:, k]
        if vk.any():
            per_pos.append(float((pred[vk, k] != y[vk, k]).mean()))
        else:
            per_pos.append(float("nan"))
    return overall, np.array(per_pos)


def fit_linear_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    seed: int = 0,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 0.05,
    momentum: float = 0.9,
    l2: float = 1e-4,
) -> ProbeFitResult:
    """Multi-position softmax regression, plain numpy (no torch -- CPU-testable, and on a real
    run the compute is dominated by embedding extraction, not this fit).

    X_*: float embeddings, shape (n_tokens, dim). y_*: int labels, shape (n_tokens, K) with K
    positions per token (K = merge**2 = 4 for the Qwen 2x2 merger) and -1 marking positions to
    EXCLUDE from loss and metrics (calibration-row patches). One weight matrix (dim, K*P) fits
    all K positions jointly -- position-specific readout, shared input standardization (train
    statistics only, applied to both splits; a probe must not peek at test statistics).

    Deterministic for fixed inputs + seed (seeded shuffles, no other randomness). Minibatch SGD
    with momentum and L2; deliberately boring -- the probe's job is to measure LINEAR
    separability, not to be a clever classifier, so no schedule tuning, no early stopping on
    test (that would leak), no nonlinearity.
    """
    if X_train.ndim != 2 or y_train.ndim != 2 or X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train (N,D) and y_train (N,K) must align on N")
    if X_test.shape[1] != X_train.shape[1] or y_test.shape[1] != y_train.shape[1]:
        raise ValueError("train/test feature dims and position counts must match")
    if (y_train >= 0).sum() == 0:
        raise ValueError("y_train has no valid (>= 0) labels -- nothing to fit")

    rng = np.random.default_rng(seed)
    n, dim = X_train.shape
    k = y_train.shape[1]

    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-6
    Xtr = ((X_train - mu) / sd).astype(np.float64)
    Xte = ((X_test - mu) / sd).astype(np.float64)

    w = np.zeros((dim, k * n_classes))
    b = np.zeros(k * n_classes)
    vw = np.zeros_like(w)
    vb = np.zeros_like(b)

    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = Xtr[idx]
            yb = y_train[idx]
            logits = (xb @ w + b).reshape(len(idx), k, n_classes)
            probs = _softmax(logits)
            grad = probs.copy()
            valid = yb >= 0
            rows, cols = np.nonzero(valid)
            grad[rows, cols, yb[rows, cols]] -= 1.0
            grad[~valid] = 0.0
            n_valid = max(int(valid.sum()), 1)
            grad = grad.reshape(len(idx), k * n_classes) / n_valid
            gw = xb.T @ grad + l2 * w
            gb = grad.sum(axis=0)
            vw = momentum * vw - lr * gw
            vb = momentum * vb - lr * gb
            w += vw
            b += vb

    train_logits = (Xtr @ w + b).reshape(n, k, n_classes)
    train_err, _ = _masked_error(train_logits, y_train)
    test_logits = (Xte @ w + b).reshape(len(Xte), k, n_classes)
    test_err, per_pos = _masked_error(test_logits, y_test)

    return ProbeFitResult(
        symbol_error=test_err,
        per_position_error=per_pos.tolist(),
        train_symbol_error=train_err,
        n_train_positions=int((y_train >= 0).sum()),
        n_test_positions=int((y_test >= 0).sum()),
        n_classes=n_classes,
        epochs=epochs,
        seed=seed,
    )


@dataclass
class ProbeCellReport:
    """One (palette, corruption) probe cell, with the verdict spelled out against the two
    reference lines that matter: chance (1 - 1/P: the probe learned nothing) and the RS budget
    (`rs_symbol_error_budget`: the error rate below which end-to-end decode succeeds)."""

    palette: int
    corruption: str
    fit: ProbeFitResult
    rs_budget: float
    chance_error: float = field(init=False)
    verdict: str = field(init=False)

    def __post_init__(self) -> None:
        self.chance_error = 1.0 - 1.0 / self.palette
        e = self.fit.symbol_error
        if math.isnan(e):
            self.verdict = "NO VALID TEST LABELS"
        elif e <= self.rs_budget:
            self.verdict = "BELOW RS BUDGET -- information present and linearly readable"
        elif e < 0.5 * self.chance_error:
            self.verdict = (
                "above RS budget but far below chance -- partial signal; not decodable "
                "end-to-end at this operating point via a linear readout"
            )
        else:
            self.verdict = (
                "at/near chance -- no LINEARLY-DECODABLE per-patch signal survives to this tap "
                "point (if this happens on CLEAN images, check the token-order assumption "
                "first, then treat the LM-token branch as unsupported by a LINEAR probe for "
                "this tower; a higher-capacity/nonlinear probe is untested and could differ)"
            )


def evaluate_cell(
    palette: int,
    corruption: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    nsym: int = 32,
    seed: int = 0,
    epochs: int = 60,
) -> ProbeCellReport:
    """Fit + verdict for one (palette, corruption) cell. Thin composition of fit_linear_probe
    and ProbeCellReport so callers (scripts/run_probe.py, tests) share one code path."""
    fit = fit_linear_probe(
        X_train, y_train, X_test, y_test, n_classes=palette, seed=seed, epochs=epochs
    )
    return ProbeCellReport(
        palette=palette, corruption=corruption, fit=fit, rs_budget=rs_symbol_error_budget(nsym)
    )


def format_report(cells: Sequence[ProbeCellReport], model_id: str = "(unspecified)") -> str:
    """Markdown report over probe cells -- the artifact a RunPod session should paste back into
    the repo. States the model, the bars, and the honest scope (linear probe over frozen
    embeddings with oracle labels; NOT an end-to-end VLM transcription result)."""
    lines = [
        "# Frozen-encoder linear-probe report (Phase-2 Step 0)",
        "",
        f"Model (frozen vision tower): `{model_id}`. Probe: linear softmax readout over "
        "merged-token embeddings, oracle labels, train-statistics standardization only.",
        "",
        "**Scope honesty:** a PASS here means the frozen embeddings linearly carry the "
        "symbols -- it de-risks, but does not replace, the fine-tune (the LM still has to "
        "learn to read them). A FAIL on clean images means no LINEARLY-DECODABLE per-patch "
        "signal survives to this tap point; a higher-capacity/nonlinear probe run at the same "
        "tap point could still differ (untested here -- see docs/FINDINGS.md Section 5), so "
        "treat a FAIL as strong, not absolute, negative evidence.",
        "",
        "| palette | corruption | probe symbol error | train error | RS budget | chance | "
        "verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        lines.append(
            f"| {c.palette} | {c.corruption} | {c.fit.symbol_error:.4f} | "
            f"{c.fit.train_symbol_error:.4f} | {c.rs_budget:.4f} | {c.chance_error:.4f} | "
            f"{c.verdict} |"
        )
    lines.append("")
    return "\n".join(lines)
