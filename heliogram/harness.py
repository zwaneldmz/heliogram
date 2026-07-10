"""heliogram.harness -- CPU-only evaluation harness for the heliogram codec (D3 + Slice B).

For every (palette, subpatch, payload_size) combination in the capacity sweep, and every
corruption in CORRUPTIONS (plus a `clean` no-op), encodes N synthetic random payloads, decodes
them back through decode_pixels (the reference, model-free decoder), and measures:

  symbol_error_rate    fraction of DATA sub-cells whose classified symbol differs from the
                       ground-truth symbol written at encode time (ground truth is read straight
                       off the clean, uncorrupted image, which is exact by construction).
  decode_success_rate  fraction of trials where decode_pixels ran to completion AND returned the
                       exact original payload.
  bits_per_patch       subpatch^2 * bits_per_symbol * (data_patches/total_patches) *
                       (payload_bytes/ecc_bytes), i.e. the codec's raw per-DATA-PATCH density
                       (subpatch^2 symbols/patch) discounted by (a) the calibration row overhead
                       and (b) the Reed-Solomon parity overhead, counted only on a successful
                       decode (0 contribution otherwise).

Slice B adds the capacity + amortization + GATE sweep: `subpatch` (k, geometric sub-cells/patch)
and `payload_size` (amortization of fixed per-message overhead) as swept dimensions alongside
`palette`, plus a headline "Gate #1" section that flags which (palette, subpatch, payload_size)
configs clear a fixed bits/patch bar both on a clean image and under worst-case tested
corruption. See write_results_md's "Headline" section for the mandatory subpatch>1 honesty
caveat: decode_pixels reads sub-cells trivially because it samples known exact pixel
coordinates -- that a real VLM's vision encoder could do the same is UNVERIFIED and is Phase 2
work, not a capability claim made here.

Run as `python -m heliogram.harness`. Prints the headline gate table and writes results.csv +
RESULTS.md into the current working directory.
"""

from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .baselines import base64_bits_per_token, rendered_text_density
from .codec import (
    HeliogramDecodeError,
    PATCH_SIZE,
    VALID_PALETTES,
    VALID_SUBPATCHES,
    bits_per_symbol,
    compute_grid,
    decode_pixels,
    encode,
    extract_symbols,
    rs_encoded_length,
)
from .corruption import compose, crop_pad, jpeg_compress, resize_roundtrip

PALETTES = VALID_PALETTES
NSYM = 32
N_TRIALS = 5
PAYLOAD_SIZE = 48  # bytes per synthetic trial payload -- module default / single-cell fallback

# --- Slice B: capacity + amortization + GATE sweep dimensions --------------------------------
SUBPATCHES = VALID_SUBPATCHES              # sweep dimension: k in {1, 2} sub-cells/data-patch
SWEEP_PAYLOAD_SIZES = (48, 1024, 4096, 16384)  # sweep dimension: bytes/trial payload (amortization)
# The full sweep (palette x subpatch x payload_size x corruption) encodes/corrupts/decodes
# multi-thousand-patch images at the 16384-byte tier; N_TRIALS=5 there is expensive, so the
# sweep specifically (not the module default, and not the diagnostic STRESS suite below, which
# keeps running at N_TRIALS) uses a reduced trial count to bound wall-clock. Documented again in
# RESULTS.md's "Wall-clock note".
SWEEP_N_TRIALS = 3
# Gate #1 bar for the headline sweep section: a config "clears the gate" only if bits/patch is
# at or above this value BOTH clean and under its single worst tested corruption.
GATE_BITS_PER_PATCH = 8.0

CORRUPTIONS: Dict[str, Callable] = {
    "clean": lambda img: img,
    "resize_3pct": lambda img: resize_roundtrip(img, scale=0.97),
    "resize_5pct": lambda img: resize_roundtrip(img, scale=0.95),
    "jpeg_q95": lambda img: jpeg_compress(img, quality=95),
    "jpeg_q85": lambda img: jpeg_compress(img, quality=85),
    "jpeg_q70": lambda img: jpeg_compress(img, quality=70),
    "crop_pad_2px": lambda img: crop_pad(img, dx=2, dy=2),
    # "combined": the composed worst-case suite the README's "Corrupted" column describes --
    # resize 5%, JPEG q70, and a 2px crop/pad applied in sequence to the SAME image.
    "combined": lambda img: compose(
        img,
        [
            (resize_roundtrip, {"scale": 0.95}),
            (jpeg_compress, {"quality": 70}),
            (crop_pad, {"dx": 2, "dy": 2}),
        ],
    ),
}

# Diagnostic-only corruptions, well outside the "realistic serving pipeline" envelope the
# CORRUPTIONS suite above targets (resize +-1-5%, JPEG q70-95, slight crop/pad). These exist
# solely to prove decode_pixels' success rate on the realistic suite is a real headroom result
# and not a harness bug that can never observe failure -- see the "Beyond the realistic
# envelope" section of RESULTS.md. Run at a single representative config (subpatch=1,
# payload_size=PAYLOAD_SIZE), not swept across the capacity grid -- see main()/RESULTS.md.
STRESS_CORRUPTIONS: Dict[str, Callable] = {
    "stress_resize_50pct": lambda img: resize_roundtrip(img, scale=0.5),
    "stress_jpeg_q10": lambda img: jpeg_compress(img, quality=10),
    "stress_crop_pad_6px": lambda img: crop_pad(img, dx=6, dy=6),
    "stress_combined": lambda img: compose(
        img,
        [
            (resize_roundtrip, {"scale": 0.5}),
            (jpeg_compress, {"quality": 10}),
            (crop_pad, {"dx": 6, "dy": 6}),
        ],
    ),
}


@dataclass
class CellResult:
    palette: int
    subpatch: int
    payload_size: int
    corruption: str
    bits_per_symbol: int
    symbol_error_rate: float
    decode_success_rate: float
    bits_per_patch: float
    trials: int


def _random_payload(seed: int, size: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(size))


def _bits_per_patch_on_success(
    payload_len: int, palette: int, nsym: int, subpatch: int = 1
) -> float:
    """subpatch^2 * bits_per_symbol * (data_patches/total_patches) * (payload_bytes/ecc_bytes)
    for a payload of this size -- a property of the format for given (palette, payload_len,
    nsym, subpatch), independent of which corruption (if any) is applied.

    Mirrors encode()'s exact grid math: `num_symbols` ecc-bitstream symbols are packed
    `subpatch*subpatch` per DATA patch (`data_patches_needed = ceil(num_symbols / k**2)`) before
    `compute_grid` sizes the patch grid, so `subpatch=1` reproduces the pre-Slice-B formula
    exactly (`cells_per_patch=1` collapses `data_patches_needed` to `num_symbols`).
    """
    bps = bits_per_symbol(palette)
    message_len = 5 + payload_len
    ecc_len = rs_encoded_length(message_len, nsym)
    num_symbols = math.ceil(ecc_len * 8 / bps)
    cells_per_patch = subpatch * subpatch
    data_patches_needed = math.ceil(num_symbols / cells_per_patch)
    width, height = compute_grid(data_patches_needed, palette)
    total_patches = width * height
    data_patches = width * (height - 1)
    return cells_per_patch * bps * (data_patches / total_patches) * (payload_len / ecc_len)


def _run_cell(
    palette: int,
    corruption_name: str,
    corruption_fn: Callable,
    n_trials: int = N_TRIALS,
    payload_size: int = PAYLOAD_SIZE,
    subpatch: int = 1,
) -> CellResult:
    bps = bits_per_symbol(palette)
    symbol_errors = 0
    symbol_total = 0
    successes = 0

    for trial in range(n_trials):
        payload = _random_payload(trial, payload_size)
        clean_img = encode(
            payload, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, seed=0, subpatch=subpatch
        )
        corrupted_img = corruption_fn(clean_img)
        if corrupted_img.size != clean_img.size:
            corrupted_img = corrupted_img.resize(clean_img.size)

        _, _, truth = extract_symbols(
            clean_img, palette=palette, patch_size=PATCH_SIZE, subpatch=subpatch
        )
        _, _, observed = extract_symbols(
            corrupted_img, palette=palette, patch_size=PATCH_SIZE, subpatch=subpatch
        )
        n = min(len(truth), len(observed))
        symbol_errors += sum(1 for i in range(n) if truth[i] != observed[i])
        symbol_total += n

        try:
            decoded = decode_pixels(
                corrupted_img, palette=palette, patch_size=PATCH_SIZE, nsym=NSYM, subpatch=subpatch
            )
            if decoded == payload:
                successes += 1
        except HeliogramDecodeError:
            pass

    success_rate = successes / n_trials
    bpp_on_success = _bits_per_patch_on_success(payload_size, palette, NSYM, subpatch)
    return CellResult(
        palette=palette,
        subpatch=subpatch,
        payload_size=payload_size,
        corruption=corruption_name,
        bits_per_symbol=bps,
        symbol_error_rate=(symbol_errors / symbol_total) if symbol_total else 0.0,
        decode_success_rate=success_rate,
        bits_per_patch=bpp_on_success * success_rate,
        trials=n_trials,
    )


def run(
    palettes: Sequence[int] = PALETTES,
    corruptions: Optional[Dict[str, Callable]] = None,
    n_trials: int = N_TRIALS,
    subpatches: Sequence[int] = (1,),
    payload_sizes: Sequence[int] = (PAYLOAD_SIZE,),
) -> List[CellResult]:
    """Run every (palette, subpatch, payload_size) x corruption cell. Defaults reproduce the
    pre-Slice-B behavior exactly (single subpatch=1, single payload_size=PAYLOAD_SIZE); pass
    `subpatches=SUBPATCHES, payload_sizes=SWEEP_PAYLOAD_SIZES` for the full capacity sweep."""
    if corruptions is None:
        corruptions = CORRUPTIONS
    return [
        _run_cell(palette, name, fn, n_trials, payload_size=payload_size, subpatch=subpatch)
        for palette in palettes
        for subpatch in subpatches
        for payload_size in payload_sizes
        for name, fn in corruptions.items()
    ]


def format_table(results: List[CellResult]) -> str:
    headers = [
        "palette",
        "subpatch",
        "payload_size",
        "bits/sym",
        "corruption",
        "symbol_err_rate",
        "decode_success",
        "bits/patch",
    ]
    rows = [
        [
            str(r.palette),
            str(r.subpatch),
            str(r.payload_size),
            str(r.bits_per_symbol),
            r.corruption,
            f"{r.symbol_error_rate:.4f}",
            f"{r.decode_success_rate:.2f}",
            f"{r.bits_per_patch:.3f}",
        ]
        for r in results
    ]
    widths = [
        max([len(h)] + [len(row[i]) for row in rows]) for i, h in enumerate(headers)
    ]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines += [fmt_row(row) for row in rows]
    return "\n".join(lines)


def format_gate_table(gate_rows: List[Dict[str, object]]) -> str:
    """Render the headline Gate #1 rows (see _gate_rows) as a plain-text table for stdout."""
    headers = [
        "palette",
        "subpatch",
        "payload",
        "ceiling",
        "clean_bpp",
        "worst_bpp",
        "worst_corruption",
        "clears_clean",
        "clears_worst",
        "clears_both",
    ]
    rows = [
        [
            str(r["palette"]),
            str(r["subpatch"]),
            str(r["payload_size"]),
            str(r["ceiling"]),
            f"{r['clean']:.3f}",
            f"{r['worst']:.3f}",
            str(r["worst_name"]),
            "yes" if r["clears_clean"] else "no",
            "yes" if r["clears_worst"] else "no",
            "YES" if r["clears_both"] else "no",
        ]
        for r in gate_rows
    ]
    widths = [
        max([len(h)] + [len(row[i]) for row in rows]) for i, h in enumerate(headers)
    ]

    def fmt_row(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines += [fmt_row(row) for row in rows]
    return "\n".join(lines)


def write_csv(results: List[CellResult], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "palette",
                "subpatch",
                "payload_size",
                "bits_per_symbol",
                "corruption",
                "symbol_error_rate",
                "decode_success_rate",
                "bits_per_patch",
                "trials",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.palette,
                    r.subpatch,
                    r.payload_size,
                    r.bits_per_symbol,
                    r.corruption,
                    f"{r.symbol_error_rate:.6f}",
                    f"{r.decode_success_rate:.4f}",
                    f"{r.bits_per_patch:.6f}",
                    r.trials,
                ]
            )


def _summary_rows(
    results: List[CellResult],
) -> Dict[Tuple[int, int, int], Dict[str, object]]:
    """Roll up per-(palette, subpatch, payload_size) bucket: 'clean' bits/patch, the mean
    bits/patch over every non-clean corruption ('corrupted_mean' -- the metric the original
    v0.1 README/RESULTS summary table used, kept for continuity), and the worst (minimum)
    non-clean bits/patch plus which corruption produced it ('corrupted_worst' /
    'corrupted_worst_name') -- the metric the headline Gate #1 table needs, since a config must
    clear the gate even in its worst tested case, not just on average."""
    by_bucket: Dict[Tuple[int, int, int], Dict[str, list]] = {}
    for r in results:
        key = (r.palette, r.subpatch, r.payload_size)
        bucket = by_bucket.setdefault(key, {"clean": [], "corrupted": []})
        if r.corruption == "clean":
            bucket["clean"].append(r.bits_per_patch)
        else:
            bucket["corrupted"].append((r.bits_per_patch, r.corruption))

    summary: Dict[Tuple[int, int, int], Dict[str, object]] = {}
    for key, buckets in by_bucket.items():
        clean_vals = buckets["clean"] or [0.0]
        corrupted_pairs = buckets["corrupted"] or [(0.0, "n/a")]
        corrupted_vals = [v for v, _ in corrupted_pairs]
        worst_val, worst_name = min(corrupted_pairs, key=lambda pair: pair[0])
        summary[key] = {
            "clean": sum(clean_vals) / len(clean_vals),
            "corrupted_mean": sum(corrupted_vals) / len(corrupted_vals),
            "corrupted_worst": worst_val,
            "corrupted_worst_name": worst_name,
        }
    return summary


def _gate_rows(summary: Dict[Tuple[int, int, int], Dict[str, object]]) -> List[Dict[str, object]]:
    """One row per (palette, subpatch, payload_size) bucket for the headline Gate #1 table:
    the ceiling subpatch^2*log2(palette), clean bits/patch, worst-case (min over non-clean
    corruptions) bits/patch and which corruption produced it, and whether each of clean/worst/
    both clears GATE_BITS_PER_PATCH."""
    rows: List[Dict[str, object]] = []
    for key in sorted(summary):
        palette, subpatch, payload_size = key
        bps = bits_per_symbol(palette)
        ceiling = subpatch * subpatch * bps
        s = summary[key]
        clean = s["clean"]
        worst = s["corrupted_worst"]
        clears_clean = clean >= GATE_BITS_PER_PATCH - 1e-9
        clears_worst = worst >= GATE_BITS_PER_PATCH - 1e-9
        rows.append(
            {
                "palette": palette,
                "subpatch": subpatch,
                "payload_size": payload_size,
                "ceiling": ceiling,
                "clean": clean,
                "worst": worst,
                "worst_name": s["corrupted_worst_name"],
                "clears_clean": clears_clean,
                "clears_worst": clears_worst,
                "clears_both": clears_clean and clears_worst,
            }
        )
    return rows


def write_results_md(
    results: List[CellResult],
    path: Path,
    stress_results: Optional[List[CellResult]] = None,
) -> None:
    baseline = base64_bits_per_token()
    sample_payload = _random_payload(0, PAYLOAD_SIZE)
    rendered = rendered_text_density(sample_payload, patch_size=PATCH_SIZE)
    summary = _summary_rows(results)
    gate_rows = _gate_rows(summary)
    clearing = [r for r in gate_rows if r["clears_both"]]
    any_subpatch1_clears = any(r["clears_both"] for r in gate_rows if r["subpatch"] == 1)
    any_subpatch_gt1_clears = any(r["clears_both"] for r in gate_rows if r["subpatch"] > 1)

    payload_sizes_present = sorted({r.payload_size for r in results})
    subpatches_present = sorted({r.subpatch for r in results})
    palettes_present = sorted({r.palette for r in results})
    sweep_trials = results[0].trials if results else N_TRIALS
    n_corruptions = len({r.corruption for r in results}) or len(CORRUPTIONS)
    # subpatch=1 -> k**2==1, so log2(palette) alone is the ceiling; use the largest palette
    # actually present in `results` (not the module-wide PALETTES constant) so this stays
    # correct if write_results_md is ever called with a partial palette sweep.
    max_subpatch1_ceiling = (
        max(bits_per_symbol(p) for p in palettes_present) if palettes_present else 0
    )
    max_palette_present = max(palettes_present) if palettes_present else 0

    lines = [
        "# heliogram v0.1 -- CPU eval results",
        "",
        f"Synthetic, seed-deterministic payloads. Capacity sweep: palette in "
        f"{list(palettes_present)}, subpatch (k) in {list(subpatches_present)}, payload_size "
        f"(bytes) in {list(payload_sizes_present)}, x {n_corruptions} corruptions (incl. "
        f"'clean'), {sweep_trials} trials/cell, nsym={NSYM}, patch_size={PATCH_SIZE}px. "
        "Reference decoder = decode_pixels (no model).",
        "",
        "**Scope: this file characterizes the CODE/CHANNEL only.** Every number below comes "
        "from `decode_pixels`, the model-free reference decoder (pixel sampling + "
        "nearest-neighbor classification + Reed-Solomon, no VLM in the loop). Whether a "
        "fine-tuned VLM can realize this same capacity through its own vision encoder is "
        "Phase 2 and is not measured anywhere in this repo -- see the README's "
        "\"Roadmap / Phase-2 boundary\" section.",
        "",
        "**Wall-clock note:** the full sweep below is "
        f"{len(palettes_present)} palettes x {len(subpatches_present)} subpatch values x "
        f"{len(payload_sizes_present)} payload sizes x {n_corruptions} corruptions = "
        f"{len(palettes_present) * len(subpatches_present) * len(payload_sizes_present) * n_corruptions} "
        f"cells; at the largest payload tier ({max(payload_sizes_present)}B) each cell "
        "encodes/corrupts/decodes a multi-thousand-patch image, so trial count for this sweep "
        f"was reduced to {sweep_trials} (module default is {N_TRIALS}) to bound wall-clock. "
        "The diagnostic stress suite below still runs at the module default "
        f"{N_TRIALS} trials, at a single representative config (subpatch=1, "
        f"payload_size={PAYLOAD_SIZE}B) -- see that section.",
        "",
        "## Headline: does any config clear the Gate #1 bar?",
        "",
        f"**Gate #1 bar: {GATE_BITS_PER_PATCH:.1f} bits/patch.** A config \"clears the gate\" "
        "only if its bits/patch is at or above this bar BOTH on a clean image AND in its "
        "single worst-performing tested corruption (not the mean of all corruptions) -- a "
        "config that only clears on average is not a robust win.",
        "",
        "**MANDATORY honesty caveat:** rows with `subpatch=1` are the VLM-meaningful regime -- "
        "one symbol per DATA patch, i.e. one symbol per (nominal) vision token, the only regime "
        "this project claims any real relevance to a downstream VLM. Rows with `subpatch>1` "
        "are a **PIXEL-DECODER GEOMETRIC CEILING ONLY**: `decode_pixels`/`extract_symbols` can "
        "read sub-patch cells trivially because they sample known, exact pixel coordinates off "
        "a grid whose size they are told in advance -- there is no perception involved. "
        "Whether a real ViT/VLM image encoder can resolve sub-patch structure at all is "
        "**unverified, and doubtful** (a k x k sub-cell grid inside one ViT patch may simply "
        "average out in that patch's embedding). Realizing it is Phase 2 work, gated on GPU "
        "access, and is **not a capability claim** made anywhere in this repo.",
        "",
        "| palette | subpatch | payload (B) | ceiling k²·log2(P) | clean bits/patch | "
        f"clears {GATE_BITS_PER_PATCH:.0f} clean? | worst-corruption bits/patch | "
        f"worst corruption | clears {GATE_BITS_PER_PATCH:.0f} corrupted? | clears gate (both)? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in gate_rows:
        lines.append(
            f"| {r['palette']} | {r['subpatch']} | {r['payload_size']} | {r['ceiling']} | "
            f"{r['clean']:.3f} | {'yes' if r['clears_clean'] else 'no'} | {r['worst']:.3f} | "
            f"{r['worst_name']} | {'yes' if r['clears_worst'] else 'no'} | "
            f"{'**YES**' if r['clears_both'] else 'no'} |"
        )

    lines += ["", "**Configs that clear the gate (both clean and worst-case corruption):**", ""]
    if clearing:
        for r in clearing:
            lines.append(
                f"- palette={r['palette']}, subpatch={r['subpatch']}, "
                f"payload_size={r['payload_size']}B -- clean {r['clean']:.3f} bits/patch, "
                f"worst {r['worst']:.3f} bits/patch (worst corruption: `{r['worst_name']}`)"
            )
    else:
        lines.append("- none")

    lines += ["", "**Verdict (derived from the table above, not asserted independently):**", ""]
    if clearing and not any_subpatch1_clears and any_subpatch_gt1_clears:
        lines.append(
            "Every clearing config has `subpatch>1` -- the unverified pixel-decoder geometric "
            "ceiling regime. **No `subpatch=1` (VLM-meaningful) config clears the gate at any "
            "tested payload size.** This is not just an unlucky corruption result: for "
            "`subpatch=1` the raw per-symbol ceiling is `log2(palette)`, which for the largest "
            f"palette tested ({max_palette_present}) is only {max_subpatch1_ceiling} "
            "bits/patch -- already below "
            f"the {GATE_BITS_PER_PATCH:.0f}-bit bar *before* Reed-Solomon/calibration overhead "
            "is even subtracted. No amount of payload-size amortization can close that gap for "
            "`subpatch=1`; only the geometric `subpatch>1` regime can mathematically reach the "
            "bar, and whether a real VLM can realize that regime is exactly the open question "
            "Phase 2 exists to answer."
        )
    elif any_subpatch1_clears:
        lines.append(
            "At least one `subpatch=1` (VLM-meaningful) config clears the gate both clean and "
            "under worst-case corruption -- see the list above. This is still a `decode_pixels` "
            "(model-free) measurement, not a VLM result; it says the channel/code can carry "
            "the bits, not that any model has been shown to read them."
        )
    else:
        lines.append(
            "No config -- `subpatch=1` or `subpatch>1` -- clears the gate both clean and under "
            "worst-case corruption at the palettes/payload sizes tested here."
        )

    lines += [
        "",
        "## Baselines",
        "",
        f"- **base64 in text context:** ~{baseline.bits_per_token:.1f} bits/token ({baseline.note})",
        f"- **Rendered text (geometric, model-free):** {rendered.chars_per_patch:.2f} "
        f"chars/patch = {rendered.bits_per_patch:.2f} bits/patch typesetting a "
        f"{PAYLOAD_SIZE}-byte payload (base64'd, {rendered.text_len} chars) into "
        f"{rendered.patches_used} patches of the same {PATCH_SIZE}px grid unit. {rendered.note}",
        "",
        "## Summary by sub-patch regime (payload-size amortization)",
        "",
        "Fixed per-message overhead (5-byte frame header + Reed-Solomon parity + the "
        "calibration row) is amortized over more data patches as payload size grows, so "
        "bits/patch should rise toward the `subpatch²·log2(palette)` ceiling as payload "
        "grows -- this is the amortization half of this sweep. 'corr(mean)' is the mean "
        "bits/patch over every non-clean corruption in the table below (resize 3%/5%, JPEG "
        "q95/85/70, crop/pad 2px, combined), each counted as 0 on a failed decode.",
        "",
    ]
    for subpatch in subpatches_present:
        label = (
            "VLM-meaningful: one symbol per patch"
            if subpatch == 1
            else "PIXEL-DECODER GEOMETRIC CEILING ONLY -- not a VLM capability claim, see caveat above"
        )
        lines += [f"### subpatch={subpatch} ({label})", ""]
        header_cells = ["Palette", "bits/sym", "ceiling"]
        for p in payload_sizes_present:
            header_cells += [f"{p}B clean", f"{p}B corr(mean)"]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("|" + "---|" * len(header_cells))
        for palette in palettes_present:
            bps = bits_per_symbol(palette)
            ceiling = subpatch * subpatch * bps
            row_cells = [str(palette), str(bps), str(ceiling)]
            for p in payload_sizes_present:
                s = summary.get((palette, subpatch, p))
                if s is None:
                    row_cells += ["n/a", "n/a"]
                else:
                    row_cells += [f"{s['clean']:.3f}", f"{s['corrupted_mean']:.3f}"]
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    lines += [
        "## Full breakdown by corruption",
        "",
        "| palette | subpatch | payload | bits/sym | corruption | symbol error rate | "
        "decode success rate | bits/patch |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.palette} | {r.subpatch} | {r.payload_size} | {r.bits_per_symbol} | "
            f"{r.corruption} | {r.symbol_error_rate:.4f} | {r.decode_success_rate:.2f} | "
            f"{r.bits_per_patch:.3f} |"
        )

    lines += [
        "",
        "## Self-consistency checks",
        "",
        "Two invariants must hold if these numbers mean what they claim to mean: (1) "
        "bits/patch can never exceed `subpatch²·log2(palette)` -- the raw per-DATA-PATCH "
        "density for a subpatch x subpatch grid of symbols per patch, before calibration-row "
        "and Reed-Solomon overhead are subtracted (this generalizes the pre-Slice-B "
        "`<= log2(palette)` check, which is the `subpatch=1` case where `subpatch²=1`); (2) "
        "mean corrupted bits/patch can never exceed clean bits/patch for the same (palette, "
        "subpatch, payload_size), since corruption only ever removes information relative to "
        "the uncorrupted image.",
        "",
        "| palette | subpatch | payload | ceiling subpatch²·log2(P) | clean bits/patch | "
        "<= ceiling? | corrupted(mean) bits/patch | <= clean? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    any_bug = False
    for key in sorted(summary):
        palette, subpatch, payload_size = key
        bps = bits_per_symbol(palette)
        ceiling = subpatch * subpatch * bps
        s = summary[key]
        clean_ok = s["clean"] <= ceiling + 1e-9
        corrupted_ok = s["corrupted_mean"] <= s["clean"] + 1e-9
        any_bug = any_bug or not clean_ok or not corrupted_ok
        lines.append(
            f"| {palette} | {subpatch} | {payload_size} | {ceiling} | {s['clean']:.3f} | "
            f"{'yes' if clean_ok else 'NO -- BUG'} | {s['corrupted_mean']:.3f} | "
            f"{'yes' if corrupted_ok else 'NO -- BUG'} |"
        )

    consistency_note = (
        "Both invariants hold for every (palette, subpatch, payload_size) bucket above."
        if not any_bug
        else "**At least one row above is flagged NO -- BUG; see that row -- this would be a "
        "measurement or codec bug, not an expected result.**"
    )

    non_clean = [r for r in results if r.corruption != "clean"]
    max_err_cell = max(results, key=lambda r: r.symbol_error_rate) if results else None
    min_success = min((r.decode_success_rate for r in non_clean), default=1.0)
    if min_success >= 1.0 - 1e-9:
        absorption_note = (
            "Within the realistic corruption envelope this harness applies (resize +-1-5%, "
            f"JPEG q70-95, slight crop/pad, and their composition), decode_success_rate is "
            f"1.00 for every cell in this sweep -- Reed-Solomon (nsym={NSYM}) fully absorbs "
            "the symbol errors this envelope introduces at every tested (palette, subpatch, "
            "payload_size) combination, including the larger palettes/subpatch=2/bigger-"
            "payload cells this sweep adds beyond the original v0.1 4-palette/subpatch=1/"
            "48-byte sweep. That is a real result (this corruption envelope does not stress "
            "the channel's ECC margin for any tested config), not a stuck-at-1.0 measurement "
            "bug."
        )
    else:
        worst_success_cell = min(non_clean, key=lambda r: r.decode_success_rate)
        absorption_note = (
            "Within the realistic corruption envelope this harness applies, decode_success_rate "
            f"drops below 1.00 for at least one cell in this sweep (lowest observed: "
            f"{min_success:.2f}, at palette={worst_success_cell.palette}, "
            f"subpatch={worst_success_cell.subpatch}, "
            f"payload_size={worst_success_cell.payload_size}B, "
            f"corruption={worst_success_cell.corruption}) -- unlike the original v0.1 "
            f"4-palette/subpatch=1/48-byte sweep, where Reed-Solomon (nsym={NSYM}) fully "
            "absorbed every symbol error that same envelope introduced. See the full "
            "breakdown above for every cell where decode_success_rate < 1.00: this is the "
            "realistic corruption envelope actually biting at the larger palette/subpatch/"
            "payload_size combinations this sweep newly covers, not a measurement bug."
        )
    if max_err_cell is not None:
        consistency_note += (
            f" The largest observed symbol_error_rate across the whole sweep is "
            f"{max_err_cell.symbol_error_rate:.4f} (palette={max_err_cell.palette}, "
            f"subpatch={max_err_cell.subpatch}, payload_size={max_err_cell.payload_size}B, "
            f"corruption={max_err_cell.corruption}). {absorption_note}"
        )
    lines += ["", consistency_note]

    if stress_results:
        stress_palettes = sorted({r.palette for r in stress_results})
        lines += [
            "",
            "## Beyond the realistic envelope (diagnostic, single representative config)",
            "",
            "To confirm decode failure is actually reachable by this harness (i.e. that high "
            "success rates above are a real headroom finding and not a bug that can never "
            "observe failure), the same style of trial was re-run under corruption well "
            "outside the 'realistic serving pipeline' envelope: 50% bilinear resize "
            "round-trip, JPEG q10, a 6px crop/pad, and their composition. This diagnostic "
            f"suite runs at a single representative config -- subpatch=1, "
            f"payload_size={PAYLOAD_SIZE}B, {N_TRIALS} trials/cell (the module defaults) -- "
            f"across all {len(stress_palettes)} palettes; it is NOT swept across "
            "subpatch/payload_size the way the headline sweep above is, since its only "
            "purpose is to confirm the harness can observe decode failure at all.",
            "",
            "| palette | corruption | symbol error rate | decode success rate | bits/patch |",
            "|---|---|---|---|---|",
        ]
        for r in stress_results:
            lines.append(
                f"| {r.palette} | {r.corruption} | {r.symbol_error_rate:.4f} | "
                f"{r.decode_success_rate:.2f} | {r.bits_per_patch:.3f} |"
            )
        stress_min_success = min((r.decode_success_rate for r in stress_results), default=1.0)
        if stress_min_success < 1.0 - 1e-9:
            stress_note = (
                "Decode success drops well below 1.00 for at least one palette under this "
                f"diagnostic stress suite (lowest observed: {stress_min_success:.2f}), "
                "confirming the channel does have a real breaking point -- it simply lies "
                "beyond the resize/JPEG/crop ranges a typical serving pipeline applies, "
                "consistent with the realistic-envelope sweep above."
            )
        else:
            stress_note = (
                "Decode success did NOT drop below 1.00 for any palette under this diagnostic "
                "stress suite in this run -- unexpected; treat this as a signal to widen the "
                "stress corruptions rather than assume the channel has no breaking point."
            )
        lines += ["", stress_note]

    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> int:
    sweep_results = run(
        palettes=PALETTES,
        subpatches=SUBPATCHES,
        payload_sizes=SWEEP_PAYLOAD_SIZES,
        n_trials=SWEEP_N_TRIALS,
    )
    summary = _summary_rows(sweep_results)
    gate_rows = _gate_rows(summary)
    print("Headline: Gate #1 sweep (palette x subpatch x payload_size)")
    print(
        f"Gate #1 bar: {GATE_BITS_PER_PATCH:.1f} bits/patch, required both clean and under "
        "worst-case tested corruption.\n"
        "MANDATORY honesty caveat: subpatch=1 rows below are the only VLM-meaningful regime "
        "(one symbol per DATA patch, i.e. one symbol per vision token). subpatch>1 rows are a "
        "PIXEL-DECODER GEOMETRIC CEILING ONLY: decode_pixels/extract_symbols sample known, "
        "exact pixel coordinates off a grid they are told in advance, so a subpatch>1 row "
        "'clearing the gate' below is NOT a VLM capability claim -- whether a real ViT/VLM "
        "image encoder can resolve sub-patch structure at all is unverified, and doubtful. "
        "See RESULTS.md's Headline section for the full caveat."
    )
    print(format_gate_table(gate_rows))
    n_clear = sum(1 for r in gate_rows if r["clears_both"])
    n_clear_subpatch1 = sum(
        1 for r in gate_rows if r["clears_both"] and r["subpatch"] == 1
    )
    print(
        f"\n{n_clear}/{len(gate_rows)} configs clear the {GATE_BITS_PER_PATCH:.1f} "
        "bits/patch Gate #1 bar both clean and under worst-case tested corruption "
        f"({n_clear_subpatch1} of those at subpatch=1/VLM-meaningful; the remaining "
        f"{n_clear - n_clear_subpatch1} are subpatch>1 pixel-decoder-only -- see the "
        "MANDATORY caveat above, not a VLM capability claim)."
    )

    stress_results = run(corruptions=STRESS_CORRUPTIONS)
    stress_table = format_table(stress_results)
    print(
        "\n(diagnostic) beyond the realistic corruption envelope, single representative "
        f"config (subpatch=1, payload_size={PAYLOAD_SIZE}B):"
    )
    print(stress_table)

    out_dir = Path.cwd()
    csv_path = out_dir / "results.csv"
    md_path = out_dir / "RESULTS.md"
    write_csv(sweep_results, csv_path)
    write_results_md(sweep_results, md_path, stress_results=stress_results)
    print(f"\nwrote {csv_path} ({len(sweep_results)} rows) and {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
