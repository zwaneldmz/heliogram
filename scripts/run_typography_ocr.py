#!/usr/bin/env python
"""scripts/run_typography_ocr.py -- Phase-2 typography READABILITY runner (GPU, zero-shot, ~$1-2).

Decides the go/no-go question documented in heliogram/ocr_eval.py: heliogram.typography already
measured that RS-framed ascii85 typeset text clears the color codec's 6.996 bits/patch ceiling at
every font size, and the harder measured ascii85 text-token bar (8.374 bits/token,
heliogram/data/text_baselines.json) from 12px down -- but that is a GEOMETRIC, model-free upper
bound that assumes perfect legibility. This script runs a STOCK (not fine-tuned) Qwen2.5-VL
against the SAME rendered images and measures whether it can actually read them well enough to
recover the payload -- i.e. whether the typography pivot's density economics are backed by real
readability, or only by an assumption. Run this BEFORE any typography-focused fine-tune spend:
a clean, generous-font-size FAIL here kills the pivot for a few dollars, exactly the same
"cheap experiment before expensive one" role scripts/run_probe.py plays for the color-codec
Phase-2 branch.

Usage (on a GPU box with `pip install -e ".[gpu]"` or `pip install -e . -r requirements-gpu.txt`
done):

    python scripts/run_typography_ocr.py                                   # defaults below
    python scripts/run_typography_ocr.py --model-id Qwen/Qwen2.5-VL-7B-Instruct \\
        --font-sizes 14,12,10,8 --payload-size 256 --n-trials 5 \\
        --out typography_ocr_report.md --json typography_ocr_report.json

DATA HONESTY (mirrors heliogram/vlm.py's and scripts/run_probe.py's module docstrings): this
file has never been run against real WEIGHTS in this repository -- there is no GPU and no HF Hub
access here. The model-INTERFACE contract it relies on (chat-template + processor + generate +
batch_decode) is the SAME one `heliogram.vlm.QwenVLDecoder._generate` documents and
`heliogram.ocr_eval._generate` mirrors byte-for-byte, and the `_load_model`/image-processor
identity-preprocessing pattern below is copied from `scripts/run_probe.py`'s `_load_tower`/
`_extract_embeddings`, which that script's own docstring documents as CPU-contract-verified
against transformers 5.13.0 (see tests/test_probe_contract_cpu.py) even though the underlying
weights have never been exercised. Treat this script as a documented, reasonable starting point,
not a verified integration -- `python scripts/run_typography_ocr.py --help` works without a GPU
(argparse only); actually running it requires one.

WHAT THIS MEASURES, PRECISELY: for each font size in `--font-sizes`, this renders `--n-trials`
random payloads (raw ascii85, no ECC, AND RS-framed ascii85, nsym=`--nsym` -- the SAME two
variants heliogram.typography measures) via heliogram.ocr_eval.render_ocr_example, asks the
STOCK model to transcribe each one, and measures character error rate (CER), exact-match rate,
and (crucially) decode_success_rate -- whether the transcription, fed through
recover_payload_from_transcription, recovers the EXACT original payload bytes. The report
cross-references these against heliogram.typography's own geometric bars (sweep_typography over
a larger reference payload, `--geometry-payload-size`, matching heliogram.typography's own
default) so a reader sees readability and density side by side in one table, not two separate
documents that have to be manually reconciled.

VERDICT RULE (see this file's `_verdict` function and RUNBOOK-GPU.md section 3.5): the pivot is
**REAL** if some font size BOTH beats the 8.374 bits/token ascii85 bar geometrically AND reads
well enough that `decode_success_rate` clears `DECISIVE_DECODE_SUCCESS_THRESHOLD` (RS variant);
it needs **FINE-TUNING** if readability only holds at font sizes too big to beat that bar (dense
enough to matter economically only where the stock tower cannot read); it is **DEAD** if even the
largest (most generous) swept font size does not transcribe reliably at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, before any pip'd copy

from heliogram.codec import PATCH_SIZE  # noqa: E402 -- no heavy deps
from heliogram.dataset import random_payload  # noqa: E402 -- no heavy deps
from heliogram.ocr_eval import (  # noqa: E402 -- no heavy deps (lazy torch inside evaluate_ocr)
    DEFAULT_OCR_FONT_SIZES_PX,
    DEFAULT_OCR_PAYLOAD_SIZE,
    OcrConfig,
    OcrResult,
    evaluate_ocr,
)
from heliogram.typography import (  # noqa: E402 -- no heavy deps
    DEFAULT_NSYM,
    DEFAULT_PAYLOAD_SIZE as TYPOGRAPHY_DEFAULT_PAYLOAD_SIZE,
    load_reference_bars,
    sweep_typography,
)

# Threshold for "reads well enough to matter" in the verdict rule below: the FRACTION of trials
# (at the RS-framed variant, the ECC-honest one) whose transcription must fully recover the
# original payload bytes via recover_payload_from_transcription. An explicit, documented,
# somewhat-arbitrary choice (0.5 -- a majority of trials) rather than a hidden magic number:
# lower would call "occasionally decodes" a pass, higher would demand near-perfect reliability
# from a STOCK, zero-shot model that has never seen this task -- 0.5 is a reasonable bar for
# "this is a real, repeatable signal worth a fine-tune's investment", not for "ready to ship".
DECISIVE_DECODE_SUCCESS_THRESHOLD = 0.5


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument(
        "--font-sizes",
        default=",".join(str(s) for s in DEFAULT_OCR_FONT_SIZES_PX),
        help=f"comma-separated font sizes (px) to sweep (default: "
        f"{','.join(str(s) for s in DEFAULT_OCR_FONT_SIZES_PX)})",
    )
    ap.add_argument(
        "--payload-size",
        type=int,
        default=DEFAULT_OCR_PAYLOAD_SIZE,
        help=f"payload size (bytes) actually rendered/transcribed per OCR trial (default: "
        f"{DEFAULT_OCR_PAYLOAD_SIZE} -- kept small on purpose, see heliogram.ocr_eval's "
        "DEFAULT_OCR_PAYLOAD_SIZE docstring: this measures READABILITY, not capacity, and a "
        "larger payload only inflates generation cost before the readability question is "
        "answered)",
    )
    ap.add_argument(
        "--geometry-payload-size",
        type=int,
        default=TYPOGRAPHY_DEFAULT_PAYLOAD_SIZE,
        help="payload size (bytes) used ONLY for the geometric reference row (sweep_typography) "
        f"-- decoupled from --payload-size so the geometric column matches "
        "heliogram.typography's own default sweep (default: "
        f"{TYPOGRAPHY_DEFAULT_PAYLOAD_SIZE}), while the actual OCR trials stay cheap",
    )
    ap.add_argument("--nsym", type=int, default=DEFAULT_NSYM, help="RS parity bytes/chunk")
    ap.add_argument("--n-trials", type=int, default=3, help="payload trials per (font size, ecc variant) cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--max-pixels",
        type=int,
        default=16_000_000,
        help="processor max_pixels -- wide on purpose so smart_resize does not shrink the "
        "rendered typography image (mirrors scripts/run_probe.py's _load_tower)",
    )
    ap.add_argument(
        "--strict-identity",
        action="store_true",
        help="raise instead of warning when the processor's smart_resize does not reproduce the "
        "rendered image's own patch grid identically for a given font size (see "
        "_assert_identity_preprocessing below) -- off by default because heliogram.typography's "
        "canvas layout rounds only to whole patch_size (14px) multiples, not the stricter "
        "patch_size*merge_size (28px) multiples smart_resize snaps to, so some font sizes are "
        "expected to trip this; the report flags it either way",
    )
    ap.add_argument("--out", default="typography_ocr_report.md")
    ap.add_argument("--json", dest="json_out", default=None)
    return ap.parse_args(argv)


def _load_model(model_id: str, device: str, dtype_name: str, max_pixels: int):
    """Load the stock (not fine-tuned) Qwen2.5-VL model + processor. UNTESTED against real
    weights in this repo (no GPU here) -- mirrors scripts/run_probe.py's `_load_tower` (same
    dtype/torch_dtype fallback, same device_map, same wide min/max_pixels rationale)."""
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
    # Wide identity bounds, mirroring scripts/run_probe.py's _load_tower: rendered typography
    # images are typically much smaller than 16M px, so this budget should never force a
    # downscale -- _assert_identity_preprocessing below is the per-image guard that actually
    # verifies this rather than trusting the budget alone.
    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=28 * 28, max_pixels=max_pixels
    )
    return model, processor


def _assert_identity_preprocessing(processor, image, *, strict: bool) -> bool:
    """Check whether the processor's mandatory `smart_resize` step reproduces `image`'s own
    14px-patch grid identically -- i.e. whether the model is actually seeing the pristine
    rendering, not a resized version of it. Mirrors scripts/run_probe.py's `_extract_embeddings`
    identity-preprocessing guard (same `image_grid_thw` check), applied here to a rendered
    typography image rather than a heliogram color-codec grid.

    Returns True if identity held, False otherwise. Raises RuntimeError instead of returning
    False when `strict=True` -- off by default (see `--strict-identity`'s help text) because
    `heliogram.typography._layout_canvas` rounds the rendered canvas only to whole `patch_size`
    (14px) multiples, not the stricter `patch_size * merge_size` (28px) multiples Qwen's
    `smart_resize` snaps to, so some font sizes are EXPECTED to trip this check -- a false trip
    is not a bug and should not silently abort the whole sweep by default; the report calls out
    which font sizes it affected instead, since an OCR read through an extra uncontrolled resize
    is a real, if different, measurement (worse-case-realistic, not clean-case).
    """
    out = processor.image_processor(images=[image.convert("RGB")], return_tensors="pt")
    grid_thw = out["image_grid_thw"]
    t, h, w = (int(x) for x in grid_thw[0])
    exp_w, exp_h = image.width // PATCH_SIZE, image.height // PATCH_SIZE
    ok = (t, h, w) == (1, exp_h, exp_w)
    if not ok and strict:
        raise RuntimeError(
            f"processor resized the rendered typography image: it is {exp_w}x{exp_h} patches "
            f"but the processor reports t={t}, h={h}, w={w} -- smart_resize moved pixels off "
            "the typeset lattice. Re-run without --strict-identity to measure through this "
            "resize instead, or pick font sizes whose rendered canvas happens to land on a "
            "28px-aligned grid."
        )
    return ok


def _geometry_by_font_size(payload_size: int, font_sizes, nsym: int, seed: int):
    """Geometric (model-free, perfect-legibility) reference row per font size, via
    heliogram.typography.sweep_typography/load_reference_bars -- REUSED, not recomputed here."""
    bars = load_reference_bars()
    payload = random_payload(seed, payload_size)
    # align=2 to match heliogram.ocr_eval.render_ocr_example: the geometry row must describe the
    # SAME 28px-aligned canvas the model is actually shown (identity under smart_resize), so the
    # "beats 8.374?" verdict reflects the image the OCR call graded, not a tighter unaligned
    # canvas the processor would have resampled before the model ever saw it.
    rows = sweep_typography(
        payload, font_sizes_px=list(font_sizes), nsym=nsym, bars=bars, align=2
    )
    return bars, {row.font_size_px: row for row in rows}


def _verdict(font_sizes, geometry, ocr_by_font_and_variant, bars) -> str:
    """The three-way call this whole experiment exists to make -- see this file's module
    docstring for the full rule. Reads off `ocr_by_font_and_variant[(font_size, apply_rs=True)]
    .decode_success_rate` (the ECC-honest variant -- see heliogram.typography's own module
    docstring for why RS, not raw, is the fair headline figure) and `geometry[font_size]
    .beats_ascii85_bar`."""
    if bars.ascii85_bits_per_token is None:
        return (
            "VERDICT: UNDETERMINED -- no measured ascii85 text-token bar is available in this "
            "checkout (heliogram/data/text_baselines.json missing; run `python -m "
            "heliogram.baselines --measure`). Readability numbers below are still real "
            "measurements, just not yet compared against the economic bar."
        )

    readable = [
        fs
        for fs in font_sizes
        if ocr_by_font_and_variant[(fs, True)].decode_success_rate
        >= DECISIVE_DECODE_SUCCESS_THRESHOLD
    ]
    beats_bar = [fs for fs in font_sizes if geometry[fs].beats_ascii85_bar]
    overlap = sorted(set(readable) & set(beats_bar))
    largest = max(font_sizes)
    largest_ok = largest in readable

    if overlap:
        return (
            f"VERDICT: REAL. Font size(s) {overlap}px BOTH clear the measured ascii85 bar "
            f"(8.374 bits/token) geometrically AND read at decode_success_rate >= "
            f"{DECISIVE_DECODE_SUCCESS_THRESHOLD:.0%} zero-shot (RS-framed variant) -- the "
            "typography pivot's density economics are backed by real stock-model readability, "
            "not merely a geometric assumption. This is still a ZERO-SHOT floor, not a "
            "production result -- fine-tuning should only improve on this."
        )
    if not largest_ok:
        return (
            f"VERDICT: DEAD. Even the largest swept font size ({largest}px) does not reliably "
            f"transcribe (decode_success_rate < {DECISIVE_DECODE_SUCCESS_THRESHOLD:.0%}, RS "
            "variant) -- the stock tower cannot read this dense a typeset rendering at all, at "
            "any size tested here. Larger font sizes (lower density, easier geometry) would need "
            "to be swept to fully rule this out, but the typography pivot has no support from "
            "this run."
        )
    return (
        f"VERDICT: NEEDS FINE-TUNING. Readable font size(s) {sorted(readable)}px do not overlap "
        f"with the font size(s) {sorted(beats_bar)}px that clear the 8.374 bits/token bar "
        "geometrically -- the stock tower can read this rendering, just not yet small/dense "
        "enough for the economics to work. A fine-tune targeting exactly the gap between these "
        "two font-size sets is the next decisive, bounded experiment."
    )


def _fmt_bool(b) -> str:
    if b is None:
        return "n/a"
    return "YES" if b else "no"


def _write_report(
    out_path: Path,
    args,
    bars,
    geometry,
    ocr_by_font_and_variant,
    identity_ok,
    verdict: str,
) -> str:
    lines = []
    lines.append("# heliogram typography OCR readability report")
    lines.append("")
    lines.append(
        "Zero-shot (no fine-tuning) readability of RS-framed/raw ascii85 typeset text by a "
        f"STOCK `{args.model_id}` -- see heliogram/ocr_eval.py and this script's module "
        "docstring for the full argument. DATA HONESTY: every number below comes from a real "
        "`model.generate(...)` call; nothing here is estimated or geometric except the "
        "'geom bits/patch (RS)' and 'beats 8.374?' columns, which are the model-free upper "
        f"bound from heliogram.typography at a {args.geometry_payload_size}-byte reference "
        "payload (decoupled from the smaller OCR trial payload for cost reasons -- see "
        "--geometry-payload-size)."
    )
    lines.append("")
    lines.append(
        f"model: {args.model_id} ({args.dtype}, device={args.device}) | OCR payload_size: "
        f"{args.payload_size}B | geometry reference payload_size: {args.geometry_payload_size}B "
        f"| n_trials: {args.n_trials} | seed: {args.seed} | nsym: {args.nsym}"
    )
    lines.append("")
    lines.append(f"reference: color codec net ceiling = {bars.color_codec_net_ceiling:.3f} bits/patch")
    if bars.ascii85_bits_per_token is not None:
        lines.append(
            f"reference: measured ascii85 text-token bar = {bars.ascii85_bits_per_token:.3f} "
            "bits/token (strongest measured text encoding, heliogram/data/text_baselines.json)"
        )
    if bars.base64_bits_per_token is not None:
        lines.append(f"reference: measured base64 text-token bar = {bars.base64_bits_per_token:.3f} bits/token")
    if bars.ascii85_bits_per_token is None:
        lines.append(f"NOTE: {bars.note}")
    lines.append("")

    header = (
        "| font(px) | geom bits/patch(RS) | beats 8.374? | identity? | "
        "CER(raw) | exact(raw) | decode-ok(raw) | CER(RS) | exact(RS) | decode-ok(RS) |"
    )
    sep = "|" + "---|" * 10
    lines.append(header)
    lines.append(sep)
    for fs in args.font_sizes_list:
        geo = geometry[fs]
        raw = ocr_by_font_and_variant[(fs, False)]
        rs = ocr_by_font_and_variant[(fs, True)]
        lines.append(
            f"| {fs} | {geo.bits_per_patch_rs:.3f} | {_fmt_bool(geo.beats_ascii85_bar)} | "
            f"{_fmt_bool(identity_ok[fs])} | {raw.mean_cer:.3f} | {raw.exact_match_rate:.0%} | "
            f"{raw.decode_success_rate:.0%} | {rs.mean_cer:.3f} | {rs.exact_match_rate:.0%} | "
            f"{rs.decode_success_rate:.0%} |"
        )
    lines.append("")
    lines.append(
        "`identity?` = did the processor's mandatory smart_resize step reproduce this font "
        "size's rendered canvas exactly (no); `no` means this row's OCR call went through an "
        "extra uncontrolled resize on top of the rendering -- see "
        "_assert_identity_preprocessing's docstring."
    )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(verdict)
    lines.append("")
    lines.append(
        "This is a ZERO-SHOT (no training) measurement -- cheap (~$1-2 of GPU time for the "
        "default sweep) and decisive for whether the typography pivot is worth pursuing further "
        "at all, exactly the same role scripts/run_probe.py's frozen-encoder linear probe plays "
        "for the color-codec branch. See RUNBOOK-GPU.md section 3.5."
    )
    report = "\n".join(lines) + "\n"
    out_path.write_text(report)
    return report


def main(argv=None) -> int:
    args = _parse_args(argv)
    font_sizes = [int(s) for s in args.font_sizes.split(",") if s]
    args.font_sizes_list = font_sizes  # stashed for _write_report's convenience

    print(f"loading {args.model_id} ({args.dtype}, device={args.device}) ...", flush=True)
    model, processor = _load_model(args.model_id, args.device, args.dtype, args.max_pixels)

    print("computing geometric reference rows (heliogram.typography, model-free) ...", flush=True)
    bars, geometry = _geometry_by_font_size(
        args.geometry_payload_size, font_sizes, args.nsym, args.seed
    )

    configs: List[OcrConfig] = []
    for fs in font_sizes:
        for apply_rs in (False, True):
            configs.append(
                OcrConfig(
                    font_size_px=fs,
                    payload_size=args.payload_size,
                    apply_rs=apply_rs,
                    nsym=args.nsym,
                )
            )

    print(
        f"running zero-shot OCR over {len(font_sizes)} font size(s) x 2 ecc variants x "
        f"{args.n_trials} trials ...",
        flush=True,
    )
    results: List[OcrResult] = evaluate_ocr(
        model, processor, configs, n_trials=args.n_trials, seed=args.seed
    )
    ocr_by_font_and_variant = {(r.config.font_size_px, r.config.apply_rs): r for r in results}

    print("checking identity preprocessing per font size ...", flush=True)
    identity_ok = {}
    for fs in font_sizes:
        from heliogram.ocr_eval import render_ocr_example

        probe_payload = random_payload(args.seed + 9_000, args.payload_size)
        probe_image = render_ocr_example(
            probe_payload, fs, apply_rs=True, nsym=args.nsym
        ).image
        identity_ok[fs] = _assert_identity_preprocessing(
            processor, probe_image, strict=args.strict_identity
        )
        if not identity_ok[fs]:
            print(f"  WARNING: font_size={fs}px did not preprocess as identity", flush=True)

    verdict = _verdict(font_sizes, geometry, ocr_by_font_and_variant, bars)

    report = _write_report(
        Path(args.out), args, bars, geometry, ocr_by_font_and_variant, identity_ok, verdict
    )
    print(f"\nwrote {args.out}")

    if args.json_out:
        payload = {
            "model_id": args.model_id,
            "dtype": args.dtype,
            "payload_size": args.payload_size,
            "geometry_payload_size": args.geometry_payload_size,
            "n_trials": args.n_trials,
            "seed": args.seed,
            "nsym": args.nsym,
            "font_sizes": font_sizes,
            "identity_preprocessing_ok": identity_ok,
            "results": [
                {"font_size_px": r.config.font_size_px, "apply_rs": r.config.apply_rs, **asdict(r)}
                for r in results
            ],
            "verdict": verdict,
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.json_out}")

    print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
