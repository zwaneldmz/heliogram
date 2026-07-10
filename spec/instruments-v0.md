# heliogram instruments — data formats v0

Companion to [`format-v0.1.md`](format-v0.1.md) (the codec wire format). This file pins the two
*data* formats the gate-independent instruments expose to the outside world. Everything else the
instruments compute (saliency maps, fingerprints, detector TPR/FPR) is a plain in-memory
dataclass with no persistence contract, so it is not pinned here — only the formats a *stored*
or *submitted* artifact must obey are normative.

Scope note (same as the rest of Phase 1): every number these instruments produce comes from the
model-free reference decoder / channel measurements, **not** a VLM, unless a real model is passed
to a model-requiring entry point (which raises without one — it never fabricates). See the
README's "Instruments" section.

## 1. Injection-benchmark submission record (`heliogram.instruments.injection_bench`)

The behavioral-injection benchmark's **submission format** is one JSON object per result,
versioned by the module-level `RESULTS_FORMAT_VERSION` (currently `1`). `write_results()` emits
JSONL (one record per line); `read_results()` parses it back; `InjectionResult.to_record()` /
`from_record()` are the exact inverse pair.

Each record is:

```
{
  "version":        int,     # == RESULTS_FORMAT_VERSION at write time; from_record() raises on an unknown version
  "payload_name":   str,     # which BEHAVIORAL_PAYLOADS entry
  "category":       str,     # one of BEHAVIORAL_CATEGORIES (persona / schema / tool_call)
  "palette":        int,     # codec palette the payload was encoded at
  "subpatch":       int,
  "payload_size":   int,     # bytes
  "patch_size":     int,     # px
  "trials":         int,
  "influence_rate": float,   # fraction of trials the behavior fired -- ONLY ever produced by a real model run
  "note":           str
}
```

`version` is the compatibility gate: bump `RESULTS_FORMAT_VERSION` and `to_record()`/
`from_record()` together whenever the field set changes, so an old reader rejects a newer record
loudly (`from_record()` raises `ValueError`) instead of silently mis-parsing it. `influence_rate`
is never present in a record produced without a real model in the loop — `measure_behavioral_
capacity()` raises `ValueError` on `model=None`/`processor=None` rather than emitting a fabricated
rate.

## 2. Foreign-tile allow-list entry (`heliogram.instruments.foreign_tile`)

The pre-ingest guard trusts a sequence of `AllowListEntry`, each naming one codec configuration
the guard should treat as legitimate:

```
AllowListEntry(palette=8, patch_size=14, subpatch=1)
```

An image is **not** flagged foreign iff it either fails the cheap patch-structure filter (a
natural image — high within-patch color variance) or decodes cleanly under at least one
allow-list entry via `decode_pixels`. `nsym` is not an allow-list field (it is assumed shared,
defaulting to the codec's `nsym=32`, since `decode_pixels` needs it but `AllowListEntry` mirrors
only the fields that change the *pixels*). A patch-structured tile that no entry decodes is
foreign. This is a structural heuristic, not a learned classifier — see the module's
`DEFAULT_VAR_THRESHOLD` comment for the tunable ceiling and its upgrade path.

## 3. Bayes-bound instrument (`heliogram.instruments.bayes_bound`)

**Purpose:** `heliogram.codec.decode_pixels` samples ONE center pixel per patch and
nearest-neighbor classifies it — a real, cheap, model-free reference decoder, but by
construction a LOWER BOUND on what the pixel channel can carry, not a measurement of the
channel itself. The measured `palette` ∈ {128, 256} failures under `jpeg_q70` (symbol error
0.28–0.49, see `heliogram.codec`'s DATA HONESTY note) do not, by themselves, prove that no
decoder — a fine-tuned VLM included — could read those palettes under JPEG; they only prove
this *particular* decoder cannot. `bayes_bound_cell`/`run` answer the decision-relevant
question cheaply, on CPU: does the color information survive this corruption at all for
(near-)optimal classifiers over whole-patch statistics, or is it physically destroyed — closing
the Phase-2 "fine-tune a VLM to read P=256 under JPEG" bet a priori where it is?

**Method:** for each (palette, corruption) cell, `n_images` images carrying deterministic
pseudorandom payloads are `encode()`d at `subpatch=1` and corrupted via
`heliogram.harness.CORRUPTIONS[name]` (reused, never redefined — no JPEG quality/composition is
re-specified in this module). Ground truth is `extract_symbols` on the CLEAN image, exact by
construction. Two classifiers run on WHOLE-PATCH mean-RGB features (mean over the *full*
`patch_size` × `patch_size` patch, not the reference decoder's single sampled pixel): a
calibration-NN oracle (`calibration_nn_whole_patch` — `decode_pixels`' own row-0-calibration/
nearest-neighbor strategy, upgraded to whole-patch statistics at both steps) and an approx-Bayes
Gaussian oracle (`fit_gaussian_oracle`/`predict_gaussian_oracle` — a per-class-mean,
shared-diagonal-covariance, uniform-prior discriminant fit on a TRAINING split of the images and
evaluated on a disjoint HELD-OUT split). `decode_pixels`' own center-pixel NN error (via
`extract_symbols` directly) is reported alongside both, measured on the SAME held-out images so
all three numbers are apples-to-apples comparable. `RS_BUDGET = floor(nsym/2) / RS_NSIZE` (16/255
≈ 6.27% for the codec's own default `nsym=32`) is the Reed-Solomon correction-budget line; a
cell's verdict is `"information present -- a better reader could work"` if the BEST of all three
measured error rates beats the budget, `"information likely destroyed at this operating point"`
otherwise.

**Honest limits:**

- **No persisted/versioned submission format.** Unlike sections 1–2 above, `BayesBoundCell` is a
  plain in-memory dataclass with no `to_record()`/`from_record()`/write/read pair — per this
  file's own scope note, only formats a *stored* or *submitted* artifact must obey are normative
  here, and nothing in this module writes one. `format_table`'s markdown output (optionally to
  `--out FILE`) is a human-readable report, not a machine-parseable wire format — do not depend
  on its exact column layout across runs.
- **Whole-patch mean is not uniformly better — this is measured, not assumed.** Whole-patch
  features dominate a center-pixel sample under PURE JPEG quantization/chroma-subsampling noise
  (measured; see the module's own tests), but are measured to be dramatically WORSE than a
  center-pixel sample under corruption that includes even a small translational shift
  (`combined`'s 2px `crop_pad`), because averaging the whole patch pulls in real
  neighboring-patch content the instant the grid is misaligned. This is exactly why
  `best_error`/`verdict` are computed over all THREE classifiers, `decode_pixels`' own
  center-pixel NN included, not just the two whole-patch ones — see the module's own "MEASURED,
  HONEST REVERSAL" docstring section for the mechanism and the measured numbers.
- **Not a proof of impossibility.** A real VLM's vision encoder sees the full spatial pixel grid
  (cross-patch context, learned features far richer than a per-patch RGB mean), not the
  3-number-per-patch whole-patch-mean summary these classifiers use. An above-budget
  Gaussian-oracle result is strong evidence that this corruption destroys color information at
  this operating point for any per-patch-summary-statistic reader — it is **not** an absolute
  proof that no reader whatsoever could do better.
- **Byte/symbol unit mismatch.** `RS_BUDGET` is stated in RS byte-errors per 255-byte codeword;
  the error rates this module reports are per-SYMBOL (a `log2(palette)`-bit unit, not necessarily
  one byte). Comparing the two treats a symbol error and a byte error as roughly interchangeable
  — the same approximation this project's own headline motivation numbers already make (see
  `heliogram.codec`'s P=128/256-under-`jpeg_q70` DATA HONESTY note), not an exact equivalence.
- **Small default sample size.** `n_images=6` (3 train / 3 test by default) trades statistical
  precision for CPU wall-clock (the full default sweep — 4 palettes × 3 corruptions — runs in a
  few seconds in this repo's CPU environment); treat a single run's numbers as a fast signal, not
  a tightly-bounded estimate. Missing-class fallback in the Gaussian oracle (a symbol value never
  seen in the training split falls back to `get_palette`'s clean reference color) can only ever
  make that class's measured error WORSE, never better — see `fit_gaussian_oracle`'s docstring.

**How to run:** `python3 -m heliogram.instruments.bayes_bound` (defaults: palette ∈
{32, 64, 128, 256} × corruption ∈ {jpeg_q85, jpeg_q70, combined}, `n_images=6`,
`payload_size=1024`B) prints a markdown table; `--out FILE` also writes it to disk.
`--palettes`/`--corruptions`/`--n-images`/`--payload-size`/`--seed` override the sweep — see
`build_parser()`. See `tests/test_bayes_bound.py` for the fast CPU subset this repo's own test
suite runs (`palette=32`, one corruption, `n_images=2`, a small payload).
