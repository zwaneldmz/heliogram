# heliogram: findings

This document summarizes what heliogram actually measured, in one place, for
readers who want the verdict without the sweep tables. It invents no numbers:
every quantitative claim below cites the file it comes from. It is a defensive-
security and measurement writeup of completed work — it does not propose,
design, or improve any offensive capability, and none of the instruments it
describes are modified here.

## 1. Summary

heliogram tested one economic hypothesis: that a patch-aligned optical codec —
solid-color 14px blocks, one symbol per ViT patch, Reed–Solomon error
correction — could pack more payload bits per vision token than base64/ascii85
text costs per text token, making encoded images a cheaper context medium for
self-hosted VLMs (target: Qwen2.5-VL). The hypothesis was measured, not
assumed, on two independent tracks: a model-free channel/harness sweep (Phase
1, this repo, `RESULTS.md`), and a frozen-vision-tower linear probe run on real
Qwen2.5-VL weights (Phase-2 Step 0, `probe_report*.md`). **Both measurements
say no.** The codec's own physical ceiling for the only VLM-meaningful regime
(one symbol per patch) sits below the measured text baseline, and — more
fundamentally — the probes show the model's own vision pipeline actively
discards this kind of structure before an LM ever sees it. The project also
shipped two defensive instruments (`heliogram.instruments.foreign_tile`, a
structural pre-ingest detector, and `heliogram.instruments.injection_bench`, a
behavioral-payload benchmark harness) before doing any capability-facing work,
and wrote a release gate in `README.md` conditioning any future fine-tuned
reader on measuring both a behavioral-capacity number and a detector TPR/FPR
number together. The behavioral-attack side of that gate has not been measured
here — no tuned reader exists in this repo — so what follows is a threat model
and a detector, honestly labeled as such, not a demonstrated exploit.

## 2. The economic result (negative)

The harness (`python -m heliogram.harness`, output in `RESULTS.md` and
`results.csv`) sweeps palette (2–256 colors), `subpatch` (1 or 2), payload size
(48B–16KB), and a realistic corruption suite (resize, JPEG q70/85/95,
crop/pad, and the target model's own `smart_resize` preprocessing) and reports
**effective bits/patch** — payload bits divided by total grid patches, on
successful Reed–Solomon-verified decode only.

The only VLM-meaningful regime is `subpatch=1` (one symbol per one nominal
vision token); `subpatch>1` is a pixel-decoder-only geometric ceiling with no
evidence a real ViT can resolve sub-patch structure (`README.md`, "Capacity
sweep"). For `subpatch=1`, the hard architectural ceiling is `log2(palette)`,
capped at exactly 8 bits at `palette=256`; Reed–Solomon and calibration-row
overhead push the achievable **net** ceiling, as payload size grows without
bound, to `log2(256) × 223/255 ≈ 6.996 bits/patch` (`README.md`, "Capacity
sweep"; `RESULTS.md`, Headline table — the measured max observed in the sweep
is 6.827 bits/patch at palette=256, 16KB). That ceiling is a mathematical
consequence of the code's own framing/ECC overhead, not a fitting artifact.

The text-context bar it needs to beat was measured, not assumed:
`heliogram.baselines.measure_text_encoding_baselines` ran the real Qwen2.5-VL
tokenizer (`transformers==5.13.0`) against base64/ascii85/base85/hex samples
and persisted the result to `heliogram/data/text_baselines.json`. The
strongest text encoding measured is **ascii85 at 8.374 bits/token**
(base64 measures 8.096 bits/token; base85 8.178; hex 4.534 —
`heliogram/data/text_baselines.json`). Since 6.996 < 8.096 < 8.374, no
`subpatch=1` configuration can beat the text bar on a per-patch density
accounting, against either baseline — and `RESULTS.md` confirms 0 of 64 swept
`subpatch=1` configs beat even the weaker base64 bar clean, and none can by
construction at any payload size (`RESULTS.md`, "Headline"; `README.md`,
"Capacity sweep" and "Roadmap / Phase-2 boundary"). An exact byte-granular
token-count scan up to 64KB likewise found no `subpatch=1` palette crossing
below base64 token count on per-patch accounting anywhere in range — the best
ratio observed is 1.16, i.e. 16% *more* tokens than base64 (`RESULTS.md`,
"Token crossover: the actual measured benefit"). The only route under which
heliogram's token accounting crosses below base64 at all requires assuming a
model can read four patch-symbols out of one 2×2-merged LM token embedding —
an explicitly **unverified** assumption the README flags as "the same
epistemic class as the sub-patch caveat" and "the entire load-bearing wall of
the project's economic case" (`README.md`, headline paragraph; `RESULTS.md`,
"LM-token accounting" section) — and that is exactly the question Section 3
below reports a direct measurement against.

**The `smart_resize` robustness finding.** Because the operator-controlled
in-scope deployment cannot opt out of the target model's own image
preprocessing, the harness added Qwen2.5-VL's `smart_resize` (which snaps
input resolution to 28px multiples before the ViT ever sees a pixel) as a
measured corruption row. Under the *default* encoding, several palettes that
had cleared the density gate in an earlier revision (`palette=8`/`subpatch=2`)
were newly measured to fail this corruption at every payload size, because
their default grids land on an odd patch dimension and the 28px snap resamples
data rows off the symbol lattice — decode success drops to 0.00 for exactly
those rows (`README.md`, "Capacity sweep", Bar B verdict). The fix,
`encode(..., align=2)`, rounds the grid to even patch dimensions before
layout, making the `smart_resize` snap a no-op with zero wire-format change;
this is implemented and pinned by test (`tests/test_smart_resize.py`,
`spec/format-v0.1.md` §6), and re-encoding the affected configs with
`align=2` was measured to survive the full corruption suite at 1KB (9.752
clean bits/patch) and 4KB (10.089), clearing the 8.0-bit Gate #1 bar again,
while the 16KB grid separately exceeds the stock processor's ~1-megapixel
budget and needs the operator to widen `max_pixels` (`README.md`, "Capacity
sweep", Bar B verdict). This is a real, measured robustness finding about the
codec's interaction with the target model's mandatory preprocessing — not
merely a caveat, since it changed which configs clear the gate and shipped
with a working fix.

## 3. The mechanism, from the probes

The harness above only measures a model-free reference pixel decoder — it
cannot say whether a real Qwen2.5-VL vision tower even carries this
information as far as the language model. Phase-2 Step 0
(`scripts/run_probe.py`, `heliogram/probe.py`) closes that gap directly: it
pushes heliogram grids through the stock, frozen Qwen2.5-VL vision tower and
trains a linear probe from token embeddings to the ground-truth patch symbols.
A probe at/below the Reed–Solomon symbol-error budget means the information
survives to the point the probe reads from; a probe at chance means it did
not, and no fine-tune of the layers downstream of that point can recover it
(module docstring intent, echoed in every probe report's header).

Four probe runs are in the repo, at two tap points in the model (post the 2×2
merger — the default report; and, in `probe_report_premerger.md`, before the
merger, directly on ViT block output) and two model sizes:

| Report | Tower | Tap point | Palette | Corruption | Probe symbol error | RS budget | Chance | Verdict |
|---|---|---|---:|---|---:|---:|---:|---|
| `probe_report.md` | Qwen2.5-VL-3B | post-merger | 16 | clean | 0.7358 | 0.0627 | 0.9375 | at/near chance |
| `probe_report.md` | Qwen2.5-VL-3B | post-merger | 128 | clean | 0.9245 | 0.0627 | 0.9922 | at/near chance |
| `probe_report.md` | Qwen2.5-VL-3B | post-merger | 256 | clean | 0.9070 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_7b.md` | Qwen2.5-VL-7B | post-merger | 16 | clean | 0.6551 | 0.0627 | 0.9375 | at/near chance |
| `probe_report_7b.md` | Qwen2.5-VL-7B | post-merger | 256 | clean | 0.8979 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_easy.md` | Qwen2.5-VL-3B | post-merger | 2 | clean | 0.1831 | 0.0627 | 0.5000 | above RS budget, far below chance — partial signal |
| `probe_report_easy.md` | Qwen2.5-VL-3B | post-merger | 4 | clean | 0.3329 | 0.0627 | 0.7500 | above RS budget, far below chance — partial signal |
| `probe_report_premerger.md` | Qwen2.5-VL-3B | **pre-merger** | 16 | clean | **0.1344** | 0.0627 | 0.9375 | above RS budget, far below chance — partial signal |
| `probe_report_premerger.md` | Qwen2.5-VL-3B | pre-merger | 16 | jpeg_q70 | 0.1902 | 0.0627 | 0.9375 | above RS budget, far below chance — partial signal |
| `probe_report_premerger.md` | Qwen2.5-VL-3B | pre-merger | 256 | clean | 0.8081 | 0.0627 | 0.9961 | at/near chance |
| `probe_report_premerger.md` | Qwen2.5-VL-3B | pre-merger | 256 | jpeg_q70 | 0.8193 | 0.0627 | 0.9961 | at/near chance |

Reading this table with the two tap points side by side is the whole finding:

- **Post-merger, every palette on both 3B and 7B is at/near chance** —
  `palette=16` gets closest (0.6551–0.7358 symbol error against 0.9375
  chance), but is still nowhere near the 0.0627 RS budget a real decode would
  need. The merged-token embedding a language model actually consumes does not
  linearly carry these symbols, at either model size tested.
- **Pre-merger tells a sharper story: it depends on palette.** At `palette=16`
  (4 bits/symbol), the pre-merger ViT-block output gives a probe symbol error
  of **13.4%** clean — still above the 6.27% RS budget (so not directly
  decodable end-to-end via a linear readout) but "far below chance," i.e. a
  real partial signal the vision blocks themselves preserve, that survives
  degrading only modestly under `jpeg_q70` (19.0%). At `palette=256` (8
  bits/symbol), pre-merger is already at/near chance (80.8% clean, 81.9%
  jpeg_q70) — the vision blocks destroy this much finer color structure
  themselves, before the merger ever runs.
- **The 2×2 merger MLP is what finishes off the palette=16 signal.** The same
  16-color code that survives the vision blocks at 13.4% error is back at
  65.5–73.6% error (still at/near chance) once read post-merger on 3B/7B. The
  merger's job is to fold 4 ViT-patch embeddings into 1 language-visible
  token; that fold is measured here to erase most of the linearly-recoverable,
  fine-grained color-symbol information the blocks upstream still held.
- **`palette=2`/`4` (the easiest, lowest-bit-depth codes) leave a comparable
  partial-but-undecodable signal even post-merger** (`probe_report_easy.md`:
  18.3%/33.3% error against 50%/75% chance) — consistent with the same
  mechanism: coarser symbol codes survive further into the pipeline, finer
  ones don't, and none of the tested codes reach the RS budget at the point a
  language model actually reads from.

The unifying read: a VLM's own perception is a strong learned prior over
natural images, and it visibly discards machine-dense, out-of-distribution
color structure at multiple stages — first partially in the vision blocks
(more so as color depth rises), then decisively in the patch-merger MLP. This
single mechanism is simultaneously **why the compression scheme fails**
(there is no bit-exact channel left for a linear reader to exploit once the LM
actually sees the tokens) **and a measured, favorable fact for anyone worried
about human-opaque payloads slipping through image inputs**: on this
architecture and this hand-designed code, the representation a downstream
language model consumes does not linearly preserve the fine symbolic
structure a covert channel would need.

## 4. Defensive contribution

Independent of the economic result, the project shipped two defensive
instruments deliberately *before* any capability-facing (fine-tuning) work —
"detector before capability" (`README.md`, "Phase-2 safety release gate").

- **`heliogram.instruments.foreign_tile`** is a lightweight, model-free
  pre-ingest guard: `guard(img, allowlist)` flags an image carrying a
  heliogram-like payload that is not on a trusted allow-list, before that
  image ever reaches a decoder or model. Its design rests on a structural
  signature (module docstring): a genuine heliogram tile tiles evenly into
  patch-size cells that are each near-solid color, which natural photos
  essentially never exhibit by accident; `patch_structure_score` is the cheap
  first filter, and only images that pass it face the decisive second check —
  whether they decode under any of the caller's trusted `(palette,
  patch_size, subpatch)` configs. Anything patch-structured that decodes under
  none of them is flagged as foreign. `evaluate_detector` reports true-positive
  rate at a bounded false-positive rate, using "learned-alphabet" tiles (data
  cells repainted through a permuted palette, specifically constructed to
  defeat the reference decoder's calibration-from-row-0 recovery) as the hard
  positive case — the module is explicit that this is a structural/decode-
  based classifier, not a learned one, and contains no model import at any
  scope (`heliogram/instruments/foreign_tile.py` module docstring).
- **`heliogram.instruments.injection_bench`** is the harness pointed at
  *behavioral* payloads (persona hijack, forced output schema, triggered tool
  call), split explicitly into two halves per the project's data-honesty
  rule: `measure_behavioral_capacity` requires a real model+processor+judge
  and raises immediately without one, rather than ever fabricating an
  influence number — it is not run anywhere in this repo (no GPU here); and
  `evaluate_defense`/`build_detector_eval_set`, the detector-evaluation mode,
  is pure CPU image-counting that scores a candidate defense (demonstrated
  against `foreign_tile.guard`) and is fully buildable and tested here. The
  module is explicit about the resulting coverage gap: `guard` is blind to
  decoded payload *content* by design, so it cannot by itself catch a
  behavioral payload encoded under a trusted, allow-listed config — a real
  limitation of that particular defense, not a bug (`heliogram/instruments/
  injection_bench.py` module docstring). All `BEHAVIORAL_PAYLOADS` fixtures
  are labeled synthetic test strings — fictional personas, invented tool
  names, JSON actions no real system implements — used only to probe channel
  capacity, not working exploits.
- **The written release gate** (`README.md`, "Phase-2 safety release gate")
  commits, in advance of any tuned reader existing, to publishing three things
  together before releasing any Phase-2 adapter or fine-tuned weights: (1)
  measured behavioral capacity against the actual tuned model via
  `measure_behavioral_capacity`; (2) `foreign_tile.guard`'s TPR at a bounded
  FPR measured against tiles the *tuned* model decodes, not only tiles the
  reference pixel decoder decodes; and (3) a decision rule, stated before
  there is any incentive to soften it, that no adapter is released if
  behavioral payloads survive corruption at rates the guard cannot catch at
  bounded FPR. The README is explicit that this is "a commitment about what
  gets published and what release is conditioned on, not a claim that we have
  already run it" — nothing in this repo has a GPU, so neither number has
  been measured yet.

**Honestly:** the behavioral-attack capacity itself has **not** been
measured — there is no tuned reader in this repo, and
`measure_behavioral_capacity` is designed to refuse to run without one. What
exists today is a threat model, a structural detector with a measured
detection methodology, and a behavioral-benchmark harness ready to run against
a future model — not a demonstrated exploit.

## 5. Honest limitations

- The probe measures the **linear** readability of embeddings for **one**
  hand-designed code (solid 14px-block symbols, a fixed deterministic
  palette, one calibration row). It is a de-risking signal, not a proven
  capacity bound: a nonlinear probe, or a probe with more capacity or more
  training data, could recover more than a linear readout does at the same
  tap point (probe report headers: "a PASS here means the frozen embeddings
  linearly carry the symbols — it de-risks, but does not replace, the
  fine-tune").
- A **fine-tuned** adversary — one that trains the vision tower or merger
  itself to preserve this structure, rather than reading frozen embeddings —
  is a different, unmeasured question; the probes only characterize what the
  stock, frozen weights already preserve.
- A **different code** (different palette design, different patch geometry,
  a learned encoding rather than a hand-designed one) could interact with the
  vision blocks and merger differently; nothing here bounds codes not tested.
- Every number is specific to **Qwen2.5-VL** (3B and 7B variants tested,
  `transformers==5.13.0`). Nothing here transfers to other model families or
  to closed API models, which the project explicitly places out of scope
  because their preprocessing is opaque and changeable (`README.md`,
  "Scope").

## 6. What would make this a stronger result

The single measurement that would resolve the open question on both sides —
the economic one (can a fine-tuned reader realize the LM-token accounting
route despite the frozen-tower probe's negative signal?) and the safety one
(is the detector still adequate once there is something real to detect?) — is
one measured behavioral run against a tuned reader, executed exactly as
`README.md`'s Phase-2 safety release gate already specifies: run
`measure_behavioral_capacity` against the actual fine-tuned model and publish
whatever reliable behavioral influence survives the corruption suite,
alongside `foreign_tile.guard`'s TPR/FPR measured against tiles that same
tuned model decodes (not only tiles the reference pixel decoder decodes), and
apply the pre-committed decision rule to the result. That run requires a GPU
this environment does not have and a fine-tuned reader that does not exist
yet; it is out of scope for this document and is not performed here.
