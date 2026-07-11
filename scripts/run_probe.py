#!/usr/bin/env python
"""Phase-2 Step 0 runner: frozen Qwen2.5-VL vision tower + linear probe (GPU, ~minutes).

Decides the go/no-go question documented in heliogram/probe.py: do the frozen tower's
merged-token embeddings linearly carry the 4 patch symbols each LM token covers? Run this
BEFORE any QLoRA spend -- a clean-image FAIL here kills the fine-tune for tens of dollars.

Usage (on a GPU box with `pip install -e . -r requirements-gpu.txt` done):

    python scripts/run_probe.py                          # defaults: 3B model, P in {16,128,256}
    python scripts/run_probe.py --model-id Qwen/Qwen2.5-VL-7B-Instruct --palettes 16,256 \\
        --corruptions clean,jpeg_q70 --n-train-images 6 --n-test-images 3 \\
        --out probe_report.md --json probe_report.json

TWO PROBE STAGES (--probe-stage):
  merged (default)  POST-merger embeddings, 4 symbols per LM-visible token -- the LM-token
                    branch's go/no-go. A fail here (measured 2026-07: at/near chance on the 3B
                    tower, every palette, clean included) localizes the loss to AT OR BEFORE
                    the merger output, but not which side.
  pre_merger        per-PATCH states at the merger's INPUT (post vision blocks), one symbol
                    per row -- the LOCALIZATION run that splits the ambiguity: readable here
                    but not at `merged` means the merger MLP destroys the symbols, and
                    scripts/train_qlora.py's default merger-LoRA has a concrete, targeted job;
                    unreadable even here means the vision BLOCKS already discarded flat-color
                    identity and no merger/LM tuning can recover it -- stop.
                    Window-order unshuffling is recovered from the tower's own outputs by
                    exact row-matching (no private transformers imports); every ordering link
                    is CPU-verified in tests/test_probe_contract_cpu.py, including that
                    re-running the tower's merger on the unshuffled states reproduces
                    pooler_output exactly.

DATA HONESTY (mirrors heliogram/vlm.py's module docstring): this file has never been run
against real WEIGHTS in this repository -- there is no GPU and no HF Hub access here. But the
model-INTERFACE contract it relies on (visual-tower attribute path, image-processor call and
grid_thw output, visual() call signature, merged-token count and which output field carries the
RASTER-ORDERED merged embeddings) is CPU-VERIFIED against transformers 5.13.0 using a tiny
RANDOM-WEIGHT Qwen2.5-VL instantiated from a local config -- see
tests/test_probe_contract_cpu.py, which runs this file's own _extract_embeddings/_cell_arrays
end to end (random weights => probe at chance, contracts exercised for real). That test caught
two run-blocking defects the first time it ran: `model.visual` does not exist in transformers
5.x (`model.model.visual` does), and the tower returns BaseModelOutputWithPooling whose
`pooler_output` -- NOT `last_hidden_state`, which is pre-merger and WINDOW-SHUFFLED -- holds
the merged tokens. The model-free half (labels, probe, verdicts, report) lives in
heliogram/probe.py and IS tested (tests/test_probe.py), including an end-to-end
synthetic-encoder run that exercises the exact label/token ordering this script relies on. Two
loud guard rails protect the real run:
 (1) an identity-preprocessing assertion -- if the processor's smart_resize touches the image
     (moving pixels off the symbol lattice), this script raises instead of measuring garbage;
 (2) the chance-level signature -- if the tower's token order differs from the raster-order
     assumption, the probe scores at chance ON CLEAN IMAGES, which the report calls out
     explicitly rather than reporting it as a quiet negative.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, before any pip'd copy

from heliogram.codec import PATCH_SIZE
from heliogram.dataset import generate_examples
from heliogram.harness import CORRUPTIONS
from heliogram.probe import evaluate_cell, format_report, merged_token_labels

MERGE = 2  # Qwen2.5-VL spatial merger: 2x2 ViT patches -> 1 LM-visible token


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--palettes", default="16,128,256",
                    help="comma-separated; default runs the README's Phase-2 order")
    ap.add_argument("--corruptions", default="clean,jpeg_q85,jpeg_q70",
                    help="comma-separated names from heliogram.harness.CORRUPTIONS")
    ap.add_argument("--n-train-images", type=int, default=6)
    ap.add_argument("--n-test-images", type=int, default=3)
    ap.add_argument("--payload-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--probe-stage", default="merged", choices=["merged", "pre_merger"],
                    help="merged (default): probe POST-merger embeddings, 4 symbols per LM "
                    "token -- the LM-token-branch go/no-go. pre_merger: probe the per-patch "
                    "states at the merger's INPUT (1 symbol per row) -- the LOCALIZATION run: "
                    "readable here but not at 'merged' means the merger MLP destroys the "
                    "symbols (and merger-LoRA has a concrete target); unreadable even here "
                    "means the vision blocks already discarded them and no merger/LM tuning "
                    "can recover it.")
    ap.add_argument("--out", default="probe_report.md")
    ap.add_argument("--json", dest="json_out", default=None)
    return ap.parse_args(argv)


def _resolve_visual_tower(model):
    """The vision tower's attribute path moved across transformers versions: `model.visual` in
    the 4.4x-era releases, `model.model.visual` in transformers 5.x (VERIFIED against
    transformers 5.13.0 with a CPU-instantiated random-weight Qwen2.5-VL -- the top-level
    `.visual` attribute simply does not exist there and the original `model.visual(...)` call
    raised AttributeError; see tests/test_probe_contract_cpu.py). Resolve both, fail loudly on
    neither."""
    visual = getattr(model, "visual", None)
    if visual is None:
        inner = getattr(model, "model", None)
        visual = getattr(inner, "visual", None) if inner is not None else None
    if visual is None:
        raise RuntimeError(
            f"could not find the vision tower on {type(model).__name__}: neither `.visual` "
            "(transformers 4.x layout) nor `.model.visual` (transformers 5.x layout) exists -- "
            "check the installed transformers version's Qwen2.5-VL module layout."
        )
    return visual


def _merged_embeddings_tensor(visual_out, expected_tokens: int):
    """Extract the MERGED-token embedding matrix from whatever the visual tower returned.

    transformers 5.x returns BaseModelOutputWithPooling where `pooler_output` holds the merged
    tokens RESTORED TO RASTER ORDER (reverse_indices applied after the merger) and
    `last_hidden_state` holds the pre-merger, WINDOW-SHUFFLED per-patch sequence -- reading
    last_hidden_state would be silent garbage twice over (wrong granularity AND shuffled order).
    Verified against transformers 5.13.0 source + a CPU tiny-model run (pooler_output shape ==
    merged count, last_hidden_state shape == patch count). Older transformers returned the
    merged tensor directly. Token count is checked in every branch; a mismatch raises rather
    than measuring the wrong thing."""
    import torch

    if isinstance(visual_out, torch.Tensor):
        emb = visual_out
    else:
        pooled = getattr(visual_out, "pooler_output", None)
        last = getattr(visual_out, "last_hidden_state", None)
        if pooled is not None and pooled.shape[0] == expected_tokens:
            emb = pooled
        elif last is not None and last.shape[0] == expected_tokens:
            # A tower that returns merged states in last_hidden_state (no pooling field match).
            emb = last
        else:
            shapes = {
                "pooler_output": tuple(pooled.shape) if pooled is not None else None,
                "last_hidden_state": tuple(last.shape) if last is not None else None,
            }
            raise RuntimeError(
                f"visual tower returned {type(visual_out).__name__} with shapes {shapes}, "
                f"but no field has the expected merged-token count {expected_tokens} -- "
                "token-count contract violated; do not trust this run."
            )
    if emb.shape[0] != expected_tokens:
        raise RuntimeError(
            f"visual tower returned {emb.shape[0]} tokens, expected {expected_tokens} -- "
            "token-count contract violated; do not trust this run."
        )
    return emb


def _match_reverse_indices(pooled, merger_out):
    """Recover the window-unshuffle permutation WITHOUT importing transformers' private index
    helpers: the 5.x vision tower computes `pooler_output = merger_out[reverse_indices]` (its
    merger runs in window-shuffled order, then rows are restored to raster order), so matching
    each pooled row to its byte-identical row in the raw merger output recovers
    `reverse_indices` exactly. Version-robust (any tower where the returned merged rows are a
    permutation of the merger's raw output rows -- including the identity permutation -- works)
    and fail-loud: raises on any duplicate, unmatched, or non-permutation row rather than
    returning a plausible-but-wrong ordering. Duplicate merged rows are effectively impossible
    for real inputs (rotary position embeddings make every token position-distinct), so hitting
    the duplicate guard means something is deeply wrong -- do not trust that run."""
    import torch

    if pooled.shape != merger_out.shape:
        raise RuntimeError(
            f"pooled/merger-output shape mismatch: {tuple(pooled.shape)} vs "
            f"{tuple(merger_out.shape)} -- the merger hook captured something unexpected"
        )
    out_np = merger_out.detach().float().cpu().numpy()
    pooled_np = pooled.detach().float().cpu().numpy()
    lookup = {}
    for j in range(out_np.shape[0]):
        key = out_np[j].tobytes()
        if key in lookup:
            raise RuntimeError(
                f"merger output rows {lookup[key]} and {j} are byte-identical -- cannot "
                "recover the unshuffle permutation unambiguously; do not trust this run"
            )
        lookup[key] = j
    reverse = []
    for i in range(pooled_np.shape[0]):
        j = lookup.get(pooled_np[i].tobytes())
        if j is None:
            raise RuntimeError(
                f"pooled row {i} has no byte-identical row in the raw merger output -- the "
                "pooled-rows-are-a-permutation-of-merger-rows contract does not hold for this "
                "transformers version; do not trust this run"
            )
        reverse.append(j)
    if len(set(reverse)) != len(reverse):
        raise RuntimeError("recovered index list is not a permutation; do not trust this run")
    return torch.tensor(reverse, dtype=torch.long, device=merger_out.device)


def _extract_pre_merger_embeddings(visual, pixel_values, grid_thw, n_units: int):
    """Per-PATCH hidden states at the merger's INPUT (post vision blocks, pre 2x2 merge),
    restored to raster order: row `m * 4 + p` is merge-unit m (raster over the merged grid),
    within-unit position p (row-major within the 2x2 block: TL, TR, BL, BR) -- exactly the
    label layout heliogram.probe.merged_token_labels uses, so `labels.reshape(-1, 1)` aligns
    1:1 with these rows.

    WHY THE ORDERING HOLDS (each piece CPU-verified in tests/test_probe_contract_cpu.py):
      - the processor's `pixel_values` rows are already grouped 4-consecutive-per-merge-unit,
        units in raster order, TL/TR/BL/BR within each unit (verified directly by decoding the
        patch COLORS back out of pixel_values for a known image);
      - the tower's blocks preserve row order except one shuffle at merge-unit granularity
        (`window_index`), applied before the blocks and undone (post-merger) via
        `reverse_indices` -- recovered here by `_match_reverse_indices`, no private imports;
      - re-running the tower's own merger on the unshuffled states returned here reproduces
        `pooler_output` byte-for-byte (the end-to-end assertion in the test file).
    """
    import torch

    captured = {}

    def _pre_hook(module, args):
        captured["in"] = args[0]

    def _post_hook(module, args, output):
        captured["out"] = output

    h1 = visual.merger.register_forward_pre_hook(_pre_hook)
    h2 = visual.merger.register_forward_hook(_post_hook)
    try:
        with torch.no_grad():
            visual_out = visual(pixel_values, grid_thw=grid_thw)
    finally:
        h1.remove()
        h2.remove()
    if "in" not in captured or "out" not in captured:
        raise RuntimeError(
            "the merger hooks never fired -- this tower's merger module path differs from "
            "visual.merger; do not trust this run"
        )

    pooled = _merged_embeddings_tensor(visual_out, n_units)
    reverse = _match_reverse_indices(pooled, captured["out"])

    merger_in = captured["in"]
    seq_len, hidden = merger_in.shape
    if seq_len != n_units * 4:
        raise RuntimeError(
            f"merger input has {seq_len} rows, expected {n_units * 4} (4 patches per merged "
            "token) -- token-count contract violated; do not trust this run"
        )
    units = merger_in.reshape(n_units, 4, hidden)
    return units[reverse].reshape(seq_len, hidden), pooled, reverse
    """Load the full model once, keep only what the probe needs (the visual tower + processor).
    The load call itself is untested against real WEIGHTS in this repo (no HF Hub access/GPU
    here), but the attribute layout, call signature, and output contract it relies on are
    CPU-verified against transformers 5.13.0 -- see tests/test_probe_contract_cpu.py."""
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = getattr(torch, dtype_name)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, dtype=dtype, device_map=device
        )
    except TypeError:  # older transformers: the kwarg was torch_dtype
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device
        )
    model.eval()
    # Wide identity bounds: images from heliogram.dataset are already even-patch-grid (pixel
    # dims are exact multiples of 28 = patch*merge), so smart_resize with generous
    # min/max_pixels is the identity. The per-image assertion below is the real guard.
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=28 * 28, max_pixels=16_000_000
    )
    return model, processor, dtype


def _extract_embeddings(model, processor, dtype, device: str, img, stage: str = "merged") -> np.ndarray:
    """One image -> float32 numpy embeddings for the requested probe stage. Asserts identity
    preprocessing: the processor's reported patch grid must equal the image's own 14px grid.

    stage="merged" (default): (n_merged_tokens, out_hidden) POST-merger embeddings, raster
    order -- what the LM actually sees; one row per 2x2 merge unit, 4 symbols per row.
    stage="pre_merger": (n_patches, vision_hidden) PRE-merger per-patch states, raster order
    (see _extract_pre_merger_embeddings) -- one row per patch, ONE symbol per row. This is the
    LOCALIZATION stage: if symbols are linearly readable here but not at "merged", the merger
    MLP (which scripts/train_qlora.py LoRA-tunes by default) is what destroys them and a
    targeted fine-tune has a concrete target; if they are unreadable even here, the vision
    blocks themselves discarded the information and no merger/LM tuning can recover it."""
    import torch

    out = processor.image_processor(images=[img.convert("RGB")], return_tensors="pt")
    grid_thw = out["image_grid_thw"]  # (1, 3): t, h_patches, w_patches
    t, h, w = (int(x) for x in grid_thw[0])
    exp_w, exp_h = img.width // PATCH_SIZE, img.height // PATCH_SIZE
    if (t, h, w) != (1, exp_h, exp_w):
        raise RuntimeError(
            f"processor resized the grid: image is {exp_w}x{exp_h} patches but the processor "
            f"reports t={t}, h={h}, w={w}. smart_resize moved pixels off the symbol lattice -- "
            "fix min_pixels/max_pixels (see heliogram/dataset.py's PROCESSOR RESIZE HAZARD "
            "note) instead of measuring a corrupted channel."
        )
    visual = _resolve_visual_tower(model)
    n_units = (exp_h // MERGE) * (exp_w // MERGE)
    pixel_values = out["pixel_values"].to(device=device, dtype=dtype)
    grid_thw = grid_thw.to(device)

    if stage == "pre_merger":
        emb, _, _ = _extract_pre_merger_embeddings(visual, pixel_values, grid_thw, n_units)
        return emb.float().cpu().numpy()
    if stage != "merged":
        raise ValueError(f"unknown probe stage {stage!r}; use 'merged' or 'pre_merger'")
    with torch.no_grad():
        visual_out = visual(pixel_values, grid_thw=grid_thw)
    emb = _merged_embeddings_tensor(visual_out, n_units)
    return emb.float().cpu().numpy()


def _cell_arrays(model, processor, dtype, device, palette, corruption_name, corruption_fn,
                 n_images, payload_size, seed_base, stage="merged"):
    """Generate n_images examples for one (palette, corruption) and return stacked
    (embeddings, labels). Labels come from the CLEAN image (dataset contract); embeddings from
    the corrupted one -- exactly the read-through-corruption task.

    stage="merged": one row per merged token, labels shape (n, 4) -- 4 symbols per token.
    stage="pre_merger": one row per PATCH, labels shape (n, 1) -- merged_token_labels
    row-major-flattened, which aligns 1:1 with _extract_pre_merger_embeddings' row order
    (unit-raster, then TL/TR/BL/BR within the unit; see that function's docstring)."""
    xs, ys = [], []
    examples = generate_examples(
        n_images,
        palettes=[palette],
        subpatches=[1],
        payload_sizes=[payload_size],
        seed=seed_base,
        corruptions={"clean": lambda im: im, corruption_name: corruption_fn},
        corruption_prob=0.0 if corruption_name == "clean" else 1.0,
    )
    from heliogram.dataset import target_to_symbols

    for ex in examples:
        w, h = ex.image.width // PATCH_SIZE, ex.image.height // PATCH_SIZE
        symbols = target_to_symbols(ex.target, palette)
        labels = merged_token_labels(w, h, symbols, merge=MERGE)
        if stage == "pre_merger":
            labels = labels.reshape(-1, 1)
        emb = _extract_embeddings(model, processor, dtype, device, ex.image, stage=stage)
        if emb.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"embedding/label count mismatch: {emb.shape[0]} vs {labels.shape[0]}"
            )
        xs.append(emb)
        ys.append(labels)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def main(argv=None) -> int:
    args = _parse_args(argv)
    palettes = [int(p) for p in args.palettes.split(",") if p]
    corruption_names = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    unknown = [c for c in corruption_names if c not in CORRUPTIONS]
    if unknown:
        raise SystemExit(f"unknown corruption(s) {unknown}; choose from {list(CORRUPTIONS)}")

    print(f"loading {args.model_id} ({args.dtype}, device={args.device}) ...", flush=True)
    model, processor, dtype = _load_tower(args.model_id, args.device, args.dtype)

    cells = []
    for palette in palettes:
        for cname in corruption_names:
            cfn = CORRUPTIONS[cname]
            print(f"[palette={palette} corruption={cname} stage={args.probe_stage}] "
                  "extracting embeddings ...", flush=True)
            # Disjoint seed ranges -> disjoint payloads between train and test.
            X_tr, y_tr = _cell_arrays(model, processor, dtype, args.device, palette, cname, cfn,
                                      args.n_train_images, args.payload_size,
                                      seed_base=args.seed + 1_000, stage=args.probe_stage)
            X_te, y_te = _cell_arrays(model, processor, dtype, args.device, palette, cname, cfn,
                                      args.n_test_images, args.payload_size,
                                      seed_base=args.seed + 2_000_000, stage=args.probe_stage)
            cell = evaluate_cell(palette, cname, X_tr, y_tr, X_te, y_te,
                                 seed=args.seed, epochs=args.epochs)
            print(f"  symbol_error={cell.fit.symbol_error:.4f} "
                  f"(train {cell.fit.train_symbol_error:.4f}, RS budget {cell.rs_budget:.4f}, "
                  f"chance {cell.chance_error:.4f}) -> {cell.verdict}", flush=True)
            cells.append(cell)

    report = format_report(cells, model_id=f"{args.model_id} [probe-stage={args.probe_stage}]")
    Path(args.out).write_text(report)
    print(f"\nwrote {args.out}")
    if args.json_out:
        payload = [
            {"palette": c.palette, "corruption": c.corruption, "rs_budget": c.rs_budget,
             "chance_error": c.chance_error, "verdict": c.verdict, "fit": asdict(c.fit)}
            for c in cells
        ]
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
