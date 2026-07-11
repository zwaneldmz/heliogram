#!/usr/bin/env python3
"""scripts/drive_merger_adapter.py -- REAL-RUN driver for the Task-2 merger-adapter go/no-go.

*** GPU REQUIRED. *** scripts/train_merger_adapter.py's OWN CLI refuses by design: a live model
object cannot be passed over argparse, so every invocation there reaches run_design_a/b's
model=None guard and raises. This thin driver is the intended "REAL INVOCATION" (see that
module's docstring): it loads a frozen Qwen2.5-VL tower via scripts/run_probe._load_tower, prints
the projected cost and enforces the same budget cap, then calls run_design_a / run_design_b with
the REAL model.

Every honesty guarantee lives in the functions this driver calls, unchanged: the
refuse-without-model guard (now satisfied by a real model), the alignment sanity assert that must
reproduce the committed ~13.4% pre-merger palette=16 clean symbol error before any new number is
trusted, the "a trained readout head is NOT the LM" caveat, and scoring against BOTH the RS budget
and the stock frozen-merger baseline. This driver adds no measurement logic of its own.

WORKFLOW: run Design A first (cheap, ~cents -- one frozen forward pass per image + a numpy head).
If A is at/near chance, the information is gone at the merger INPUT -> strong negative, stop. Only
if A recovers the four symbols run Design B (the actual trainable-merger gate, ~$1-tens).

This module imports torch-free (the torch import is function-local in _load_tower), so
`python scripts/drive_merger_adapter.py --help` works with no GPU; an actual run needs the GPU
environment (pip install -r requirements-gpu.lock.txt)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _load_script_module(name: str):
    """Load scripts/<name>.py by file path (scripts/ is not an importable package), the same way
    tests/test_probe.py loads run_probe -- so this driver works from a bare `git clone` without an
    editable install or a scripts/__init__.py."""
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


tma = _load_script_module("train_merger_adapter")
run_probe = _load_script_module("run_probe")


def main(argv=None) -> int:
    # Reuse train_merger_adapter's own parser/config/cost machinery verbatim -- the driver only
    # adds the one thing that CLI structurally cannot: a real, already-loaded model.
    args = tma.build_parser().parse_args(argv)
    config = tma.build_run_config(args)
    print(tma.format_config_report(config))

    # Enforce the tens-of-dollars budget cap BEFORE loading any weights, exactly as the scaffold's
    # own CLI does -- an over-budget config aborts here, having spent nothing.
    tma.check_budget(config.estimated_cost_usd, config.budget_cap_usd)

    print(f"\nloading {config.model_id} ({args.dtype}, device={args.device}) ...", flush=True)
    model, processor, dtype = run_probe._load_tower(config.model_id, args.device, args.dtype)

    common = dict(
        model=model, processor=processor, dtype=dtype, device=args.device,
        palette=config.palette, corruptions=config.corruptions,
        n_train_images=config.n_train_images, n_test_images=config.n_test_images,
        payload_size=config.payload_size, hidden_dim=config.hidden_dim,
        epochs=config.epochs, seed=config.seed, nsym=config.nsym,
    )
    if config.design == "a":
        report = tma.run_design_a(**common)
    else:
        report = tma.run_design_b(
            variant=config.lora_variant, lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha, **common,
        )

    rendered = json.dumps(report, indent=2, default=str)
    print("\n=== RESULT (Design " + config.design.upper() + ") ===")
    print(rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n")
        print(f"\nwrote {args.out}")
    if args.json_out:
        Path(args.json_out).write_text(rendered + "\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
