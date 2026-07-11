# heliogram

**If one ViT patch (~1 vision token) can reliably carry more than one text token's worth of data (8.096 bits for base64, measured on Qwen2.5-VL's tokenizer), then encoded images are a cheaper context medium for self-hosted VLMs — heliogram measures whether that's true. Measured answer so far: not per-patch (the codec's per-patch ceiling is 6.996 bits); the surviving route is multiple patch-symbols per merged LM token, which is Phase-2 work.**

heliogram is a patch-aligned optical codec plus an evaluation harness. It encodes
arbitrary bytes as a grid of solid-color 14x14 blocks (one symbol per ViT patch),
protects them with Reed–Solomon ECC, runs them through the corruptions a real
serving pipeline applies (resize, JPEG, crop/pad, and the target model's own
`smart_resize` preprocessing), and reports **effective
bits/patch** — the channel capacity of the patch grid, measured with a model-free
reference decoder. Format details: [`spec/format-v0.1.md`](spec/format-v0.1.md).

Apache-2.0. All data is synthetic and seed-deterministic; same inputs produce
pixel-identical PNGs (the PNG *container* bytes can differ across Pillow
versions — e.g. a chunk-encoding change between Pillow releases — but the
decoded pixel grid, and therefore every symbol the codec reads back, is
identical).

## Scope

- **In scope:** self-hosted / open-weight VLMs where the operator controls image
  preprocessing end to end (resolution, patching, no server-side re-encoding they
  can't inspect).
- **Out of scope:** closed API models. Their preprocessing (resizing, tiling,
  re-compression, token accounting) is opaque and changeable, so no capacity claim
  made here transfers to them. We do not test against them and won't.

**A note on scope vs. the JPEG numbers above.** The in-scope operator (previous
bullet) controls preprocessing end to end, which means they can choose to serve
lossless PNG straight through to the model — under that reading, the JPEG
q70/q85/q95 corruption rows measured throughout this README and `RESULTS.md`
are a robustness *margin* for pipelines that fall outside strict control (and a
deliberate stress test), not the operative in-scope number; the clean-channel
bits/patch and token-crossover figures are. That does not weaken the JPEG
failure caveats above — `palette=128`/`256` really are measured to fail decode
under JPEG q70 at every tested payload size, and that stays reported exactly as
measured — it only names which number this project's scope claim actually rests
on. The honest flip side: the one corruption an in-scope operator *cannot* opt
out of is the target model's own image preprocessing — e.g. Qwen2.5-VL's
`smart_resize`, which snaps input resolution to 28px (patch-size × merge-size)
multiples before the ViT ever sees a pixel. **That resize is now IN the
corruption suite and measured** (two rows: `qwen_smart_resize`, the mandatory
28px snap under operator-widened pixel bounds; and `qwen_smart_resize_1mp`,
the stock processor's ~1-megapixel budget, which additionally downscales any
larger grid wholesale — see `RESULTS.md`). The encode-side fix is also
implemented and pinned by test: `encode(..., align=2)` rounds the grid up to
even patch dimensions *before* layout, making `smart_resize` the identity on
the emitted image with zero wire-format change (`decode_pixels` round-trips it
with no flag; see `spec/format-v0.1.md` §6 and `tests/test_smart_resize.py`) —
a deployment targeting a 2×2-merge VLM should always encode with `align=2`,
and `heliogram.dataset`'s Phase-2 training generators now do exactly that.

This is a measurement project. If the numbers come out below the base64 line, that
is a result, not a failure — it bounds what optical-context schemes can gain from
denser-than-text symbol coding, and we'll report it as such.

## Results

> Measured by the eval harness (`python -m heliogram.harness`), which writes
> `RESULTS.md` and `results.csv`. Full breakdown, self-consistency checks, the
> capacity/amortization/Gate sweep, the token-crossover benefit, and a
> diagnostic stress test beyond this envelope are in
> [`RESULTS.md`](RESULTS.md).

**The honest headline, from the actual sweep — and it changed when we
measured the tokenizer:** against the MEASURED base64 baseline
(Qwen2.5-VL's tokenizer: **8.096 bits/token**, 1.3498 base64 chars/token —
BPE merges make base64 substantially cheaper than the naive 6.0-bits/token
estimate this project originally compared against), **no `subpatch=1`
config (one symbol per patch — the only VLM-meaningful regime) beats
base64 on density or on token count, at any payload size, and none ever
can on per-patch accounting:** the per-patch net ceiling is `log2(256) ×
223/255 ≈ 6.996` bits/patch, strictly below the measured 8.096 bar, and
the exact token-crossover scan finds no per-patch crossing up to 64KB
(best ratio observed: 1.16 — 16% *more* tokens than base64). The earlier
"beats base64 at 4KB" headline was an artifact of the unmeasured analytic
baseline, and this paragraph is its correction, not its defense. **The
surviving candidate benefit is the LM-token accounting:** Qwen2.5-VL's
2×2 merger presents ~4× fewer LM tokens than ViT patches, under which
heliogram crosses below base64 from tiny payloads (~50–130B, all
palettes ≥8) — but that accounting carries an extra, UNVERIFIED
assumption (that a model can read 4 symbols' worth of bits out of one
merged embedding), the same epistemic class as the sub-patch caveat. So
the project's open question has sharpened: not "can a VLM read 256
colours under JPEG" but **"can a VLM read multiple patch-symbols through
its own merger?"** — plus the measured fact (see the `bayes_bound`
instrument) that at `palette=128` under pure JPEG q70 the information
demonstrably survives in whole-patch statistics (oracle error 0.5%, 13×
below the RS budget) even though the reference decoder fails there. On
the reference decoder itself: `palette=256` fails JPEG q70 at every
payload size (and q85 from 1KB up); `palette=128` fails q70 from 1KB up.
Bit-exactness on successful decode (Reed–Solomon-verified) still holds
everywhere. See "Capacity sweep" and "Token crossover" below for the
numbers.

Effective bits/patch = payload bits / total grid patches — TRUE payload
density, with the calibration row, Reed–Solomon parity, framing, and grid
padding all counted against it — on successful RS decode only (0 on a failed
decode). (Corrected in v0.2: the earlier formula credited grid-padding
patches as if they carried payload and overstated density by up to 3x at
small payloads; every number in this README and RESULTS.md uses the
corrected definition.) "Corrupted" is the mean bits/patch over every non-clean
corruption tested: bilinear resize ±3–5%, JPEG q70–95, a slight crop/pad,
Qwen2.5-VL's own `smart_resize` preprocessing (two variants — see the Scope
note above), and the resize+JPEG+crop composition ("combined"). Reference
pixel decoder (`decode_pixels`), no model. Table below: 48-byte synthetic payloads, subpatch=1 (one symbol per
patch — see "Capacity sweep" below for the fuller grid), nsym=32,
patch_size=14px, 3 trials/cell.

| Palette | bits/symbol | Clean bits/patch | Corrupted bits/patch |
|--------:|------------:|-----------------:|----------------------:|
| 2       | 1           | 0.527            | 0.410                 |
| 4       | 2           | 1.064            | 0.827                 |
| 8       | 3           | 1.500            | 1.500                 |
| 16      | 4           | 2.000            | 2.000                 |
| 32      | 5           | 2.000            | 2.000                 |
| 64      | 6           | 2.000            | 1.556                 |
| 128     | 7           | 1.500            | 1.500                 |
| 256     | 8           | 0.750            | 0.583                 |

The break-even line to beat is **8.096 bits/patch** (the MEASURED base64
baseline — see Baselines); at this small (48-byte) payload every palette
here is far below it,
including the two largest — this is a real result, not a failure to report:
fixed overhead (32-byte RS parity + a calibration row that widens with the
palette) is the bottleneck at 48 bytes, not palette size. For `palette` in
`{8,16,32,128}`, clean and corrupted are identical at this payload size
because `decode_success_rate` is 1.00 across the whole
realistic-serving-pipeline envelope tested here (resize ±3–5%, JPEG q70–95,
slight crop/pad, and — new in this revision — the target model's own
`smart_resize` preprocessing) — Reed–Solomon (nsym=32) fully absorbs the
symbol errors that envelope introduces for those palettes at this size, and
those palettes' 48-byte grids happen to land on even patch dimensions, so
`smart_resize` passes them through untouched. (`palette=128` failed JPEG q70
here in earlier releases; that failure was the decoder's unprotected length
header — fixed in v0.2 by recovering the header through RS — not the
channel.) **`palette` 2, 4, and 64 are new casualties of measuring
`smart_resize`:** their 48-byte grids have an odd patch dimension in the
default encoding, and the mandatory 28px snap resamples every data row off
the symbol lattice — decode drops to 0.00 for exactly those rows of the
suite (that, not JPEG, is what pulls their corrupted means down above). The
fix is `encode(..., align=2)`, which makes the snap a no-op — see the Scope
note near the top of this file. **`palette=256` fails the old way:** it
fails JPEG q70 and the fully-composed `combined` even at this smallest
tested payload, which is why its corrupted column (0.750→0.583) is below its
own clean column — see `RESULTS.md` for the exact per-corruption rates. A diagnostic stress
test well outside the realistic envelope (50% resize, JPEG q10, 6px
crop/pad, composed) finds the expected fall-off at larger palettes — decode
success drops to 0.00 at palette≥4 under the composed stress case,
including 128/256 — confirming the channel has a real breaking point, just
further out than a typical serving pipeline pushes it for the smaller
palettes and well *inside* it for the two largest. See `RESULTS.md` for
that table. bits/patch (and whether corruption is survived at all) both
change with payload size — see the capacity sweep below, which measures
that directly instead of via a single one-off comparison.

### Capacity sweep: sub-patch and payload size (three bars, not one)

> Full sweep (64 configs), self-consistency checks, the token-crossover
> table, and the complete 512-row per-corruption breakdown are in the
> "Headline" and "Token crossover" sections of [`RESULTS.md`](RESULTS.md).

Beyond the single-payload table above, the harness sweeps `subpatch` (k ∈
{1, 2}: k×k solid-color sub-cells per data patch instead of 1 — a purely
geometric density knob, `spec/format-v0.1.md` §6a), `payload_size` (48B –
16KB, to amortize the fixed 5-byte header + RS parity + calibration-row
overhead), and now all 8 palettes in `VALID_PALETTES` (2–256) against the
same realistic corruption suite, at 3 trials/cell (reduced from 5 to bound
wall-clock at the 16KB tier — 64 configs × 8 corruptions = 512 cells total).
This project tracks three separate bars over that sweep, on purpose — see
`RESULTS.md`'s Headline section for why conflating them is exactly the
overclaiming this file exists to avoid:

- **Bar A — beat base64 density, clean (8.096 bits/patch, MEASURED
  Qwen2.5-VL tokenizer; was 6.0 analytic until the measurement).** The real
  economic break-even for density alone.
- **Bar B — Gate #1 comfort margin (8.0 bits/patch, clean *and* worst-tested
  corruption).** Originally padded above Bar A (see "Decision Gate" below) —
  **note the measured Bar A (8.096) now sits ABOVE this fixed 8.0 bar, so
  Gate #1 no longer functions as a comfort margin; clearing it is not
  sufficient to beat measured base64 density.** Kept for continuity.
- **Bar C — token crossover.** Does encoding a payload cost fewer total
  patches than base64-ing it into text tokens? An accounting comparison of
  token *count*, not bits/patch density — see below.

**Bar B verdict, from the actual run: 0 of 64 configs clear the gate.**
This is a change from the previous revision (3 of 64: palette=8, subpatch=2,
at 1KB/4KB/16KB) and the cause is the newly-measured corruption, not a
regression in the codec: all three former clearing configs produce grids
with an odd patch dimension in the default encoding, and the target model's
own `qwen_smart_resize` preprocessing (now in the suite) resamples them off
the symbol lattice — decode 0.00, worst-corruption bits/patch 0. **The
encode-side fix restores them, measured:** re-encoding those exact configs
with `align=2` survives the full 9-corruption suite at 1KB (9.752 clean
bits/patch) and 4KB (10.089) — clearing the 8.0 gate again — while the 16KB
grid (112×114 patches ≈ 2.5M px) additionally exceeds the STOCK processor's
~1MP pixel budget (`qwen_smart_resize_1mp`) and needs the operator to widen
`max_pixels` (a processor constructor argument, in-scope for this project's
operator) to survive. The harness sweep itself keeps measuring the default
(`align=1`) encoding, so these numbers stay the honest
what-you-get-by-default story; the `align=2` remeasurement above is a
spot-check, not a full sweep.

**Bar A verdict against the MEASURED baseline: 18 of 64 configs beat base64
density clean — every one of them `subpatch=2` (the pixel-decoder-only
geometric regime). NO `subpatch=1` config clears it, and none ever can:**
the per-patch net ceiling is `log2(256) × 223/255 ≈ 6.996` bits/patch,
strictly below the measured 8.096 bar. (Under the earlier unmeasured 6.0
analytic bar, `palette=256` at 4KB/16KB appeared to clear Bar A at
6.400/6.827 — the measurement erased that margin; this is exactly the
correction the measured baseline existed to force, reported as such.)

**Bar C verdict against the MEASURED baseline: on per-patch accounting (~1
token per ViT patch), NO `subpatch=1` palette crosses below base64 token
count anywhere in the exact scan up to 64KB** — the best ratio observed is
1.16 (`palette=256`, i.e. 16% MORE tokens than base64; under the old
analytic ~1-char/token estimate it appeared to cross at 1537B — measuring
the tokenizer erased that crossing too). Concretely: `palette=256` needs
5,120 patches at a 4KB payload versus base64's measured ~4,048 tokens
(1.26×). **The only accounting under which heliogram crosses is the
LM-token one (Qwen2.5-VL's 2×2 merger: ~4× fewer LM tokens than patches),
where all palettes ≥8 cross from ~50–130B payloads (e.g. `palette=256` at
4KB: 1,280 LM tokens vs ~4,048 — 0.32×). That accounting carries an extra,
UNVERIFIED assumption — that a model can read 4 patch-symbols' worth of
bits out of ONE merged embedding — the same epistemic class as the
sub-patch caveat, and it is now the entire load-bearing wall of the
project's economic case.**

**Honesty caveat (mandatory, same as `RESULTS.md`, and the whole reason Bar
C is a bet and not a result): `palette=128`/`256` clean-decode exactly on
this pixel decoder but are MEASURED to fail decode under JPEG q70 at every
payload size in this sweep** (0.00 decode success at 48B/1KB/4KB/16KB, both
palettes), and, at larger payloads, even JPEG q85. **The token-count and
density benefit above is therefore a clean-channel-only number — it is not
usable end to end on this reference decoder.** Whether a fine-tuned VLM
reader can survive JPEG q70 at a 256-colour palette where `decode_pixels`
cannot is exactly the Phase-2 question this project exists to answer, not
something claimed here.

**A second, independent honesty caveat, this one about `subpatch>1`: every
Bar B (Gate #1)-clearing config, and most Bar A/C-clearing configs at
`subpatch=2`, use `subpatch=2`, which is a PIXEL-DECODER GEOMETRIC CEILING,
not a VLM capability claim.** `decode_pixels` reads k×k sub-cells trivially
— it samples known, exact pixel coordinates off a grid it's told the size of
in advance — but there is no evidence a real ViT/VLM image encoder can
resolve structure below its own patch. `subpatch=1` is the only
VLM-meaningful regime (one symbol per patch = one nominal vision token). The
hard architectural ceiling for `subpatch=1` is `log2(P)`, which reaches
exactly 8 at `P=256` — but Reed–Solomon/calibration overhead caps the
*net* ceiling at `log2(P) × 223/255 ≈ 6.996` bits/patch as payload size
grows without bound, strictly below 8 for every finite payload; the measured
max in this sweep is 6.827 (palette=256, 16KB payload). **No `subpatch=1`
config clears Bar B (Gate #1) anywhere in this sweep, or ever can.**

Two more things this sweep found, both measured (not cherry-picked):

- **Higher palette ≠ better once corruption is in the loop, and it gets
  monotonically worse.** At `subpatch=2`/16KB, each palette from 8 upward has
  a *higher clean* ceiling than the last (8: 10.357, 16: 13.788, 32: 17.120,
  64: 20.480, 128: 23.814, 256: 25.600 bits/patch) but fails against a
  strictly larger set of the JPEG/resize/crop corruptions: palette=16 fails
  only the fully-composed `combined`; 32 and 64 add `JPEG q70`; 128 adds
  `JPEG q85` too; and 256 fails `JPEG q95` as well — the *mildest* JPEG
  setting this harness tests. (The two `qwen_smart_resize` rows added in
  this revision fail for every palette at this 16KB tier — those grids have
  odd patch dimensions in the default encoding — which is a grid-alignment
  effect orthogonal to the palette-vs-JPEG story this bullet is about; see
  the Bar B verdict above and `encode(..., align=2)`.) Of palette 8 and up,
  only palette=8/subpatch=2 survives every JPEG/resize/crop corruption
  (palette=2/4 also do, but their clean ceiling is too low for the
  comparison to be interesting here).
- **Amortization raises the floor but not the ceiling.** For `subpatch=1`,
  bits/patch rises with payload size (palette=64: 2.000 → 4.923 → 5.120 →
  5.185; palette=256: 0.750 → 5.333 → 6.400 → 6.827 bits/patch, at
  48B/1KB/4KB/16KB) but asymptotically approaches `log2(palette) × 223/255`
  — it pays down the fixed per-message overhead, it cannot raise the
  per-symbol ceiling itself. Against the measured 8.096 baseline no
  `subpatch=1` amortization curve ever crosses (the `palette=256` asymptote
  is 6.996); under the old analytic 6.0 bar it appeared to cross between
  1KB and 4KB — that appearance did not survive measuring the tokenizer.

Every number above is `decode_pixels` (pixel decoder, CPU, no VLM) — see the
Scope note at the top of `RESULTS.md`.

## Baselines

- **base64 in text context: 8.096 bits/token, MEASURED** (Qwen2.5-VL's
  tokenizer, `transformers==5.13.0`, 9 samples across 1KB/4KB/16KB payloads;
  committed as `heliogram/data/base64_baseline.json`, which the harness
  prefers over the analytic figure and reports as the source of every Bar
  A/Bar C verdict). The old ~6.0 analytic estimate (log2(64) per char at ~1
  char/token) understated base64 by 35%: BPE merges give base64 1.3498
  chars/token. **This single measurement is what killed the `subpatch=1`
  per-patch economic claim above — exactly the adverse-direction error the
  earlier honesty note here predicted.** `python -m heliogram.baselines
  --measure` reproduces it (or measures a different tokenizer).
- **Is base64 even the right bar? (UNMEASURED — the same class of baseline
  error, still open.)** base64 is not optimal for BPE vocabularies: ascii85/
  base85 pack 8 bits into 1.25 chars vs base64's 1.33 before BPE effects, so
  the honest economic bar is the *strongest* reasonable text encoding on the
  target tokenizer, which may sit above 8.096 bits/token. The measurement code
  ships in this repo (`heliogram.baselines.measure_text_encoding_baselines`,
  covering base64/ascii85/base85/hex on identical samples; persisted to
  `heliogram/data/text_baselines.json`, which `RESULTS.md`'s Bar A qualifier
  reads), but the environment that prepared this branch had no HuggingFace Hub
  access, so it has not been run — `python -m heliogram.baselines --measure`
  now measures both this and the base64 baseline in one command (~1 min, CPU;
  see `RUNBOOK-GPU.md` step 1). Until then, every "beats base64" verdict reads
  as exactly that — *base64*, not *text context*.
- **Rendered text (honesty guardrail).** The obvious competitor is just
  typesetting the payload small and letting the VLM read it — that's what
  DeepSeek-OCR and Glyph exploit. `heliogram.baselines` typesets the payload onto
  the same patch grid and reports chars/patch as a geometric, model-free estimate.
  The *true* rendered-text capacity depends on the un-fine-tuned VLM's OCR and is
  a Phase-2 measurement. If rendered text matches the codec's bits/patch, the
  codec's only remaining advantages are exactness (ECC-verified bytes) and
  robustness — and we'll say so.
- **Token crossover (the actual benefit claim — see "Capacity sweep" above
  and `RESULTS.md`).** Beating bits/patch density is necessary but not
  sufficient for beating base64 end to end: base64 pays a fixed 4/3-per-byte
  expansion with no per-message overhead, while heliogram pays a calibration
  row + Reed–Solomon parity once per *image*, amortized differently as
  payload grows. `heliogram.harness._token_crossover` compares
  `total_patches` (the grid's width×height, ~1 token/patch for a self-hosted
  VLM) directly against `base64_token_est` (`ceil(payload/3)*4` chars scaled
  by the measured 1.3498 chars/token) for the SAME payload. Measured (exact
  byte-granular scan, measured baseline): NO `subpatch=1` palette crosses on
  per-patch accounting anywhere up to 64KB (best ratio 1.16 at
  `palette=256`); the LM-token (2×2 merger) accounting crosses from
  ~50–130B for all palettes ≥8, under its separately-flagged unverified
  read-4-symbols-per-merged-token assumption.

## Quickstart

```bash
pip install -e .
python demo.py            # encode a small JSON doc, decode it back,
                          # print patch count vs. base64 token count
python -m heliogram.harness   # full palette x subpatch x payload x corruption sweep -> RESULTS.md, CSV
python -m heliogram.benefit   # token-savings demo (P=256, a real ~6KB JSON doc) vs. base64/hex
                               # tokens, PLUS the exactness-vs-OCR argument -- CPU-only, no model
pytest                    # roundtrip, determinism, calibration-recovery tests
```

Requires Python with `pillow`, `numpy`, `reedsolo` (installed by the command above).

## How it works (v0.1, one paragraph)

Payload bytes are framed (version byte + 4-byte length), Reed–Solomon coded
(default 32 parity bytes), split into log2(P)-bit symbols for a palette of
P ∈ {2, 4, 8, 16, 32, 64, 128, 256} deterministic, maximally separable colors, and
painted one symbol per 14x14 patch (or, with `subpatch`>1, a k×k grid of
sub-cells per patch — see the capacity sweep above) on a square-ish grid.
Row 0 is a calibration row cycling
through the palette so the decoder can recover the palette's post-corruption RGB
values and nearest-neighbor classify every data patch. The reference decoder
(`decode_pixels`) samples patch centers only — it is deliberately dumb, so that
what it measures is the channel, not decoder cleverness. `encode(..., align=2)`
rounds the grid up to even patch dimensions before layout (28px-multiple pixel
dims at the default patch size) so the pinned target model's own `smart_resize`
preprocessing passes the image through untouched — wire-compatible, decoder
needs no flag. Full spec: [`spec/format-v0.1.md`](spec/format-v0.1.md).

## Instruments (gate-independent)

The codec's headline number is conditional on Decision Gate #1; these instruments are not.
They are useful whether or not heliogram beats base64, and (except where a real VLM is
required, which is flagged) they run on CPU here. Each measures **the channel / the reference
pixel decoder, not a VLM** — the same scope caveat as the rest of Phase 1. Model-requiring
paths **raise** without a real model rather than fabricate a number (mirroring
`heliogram.vlm`); nothing below invents a model result.

- **`heliogram.patchsize`** — makes `codec.PATCH_SIZE` (14) *auditable* against a real model
  instead of a remembered constant. `known_patch_size("Qwen/Qwen2.5-VL-7B-Instruct")` returns
  the documented ViT patch size; `verify_patch_size(processor=...)` reads it empirically off a
  real processor/config when you have one (source `"measured"`), else reports the documented
  value (`"documented"`) and says so. Never fabricates a "measured" number without a real
  object. CLI: `python3 -m heliogram.patchsize --model Qwen/Qwen2.5-VL-7B-Instruct`.
- **`heliogram.instruments.foreign_tile`** — a lightweight, model-free **pre-ingest guard**
  (build the detector *before* any capability work). `guard(img, allowlist)` flags an image
  carrying a heliogram-like payload that is **not** on a trusted allow-list: natural images
  (high within-patch variance) pass; patch-structured tiles that no allow-listed
  `(palette, subpatch)` decodes are flagged. `evaluate_detector(...)` reports TPR at a bounded
  FPR, with learned-alphabet-style tiles (data cells repainted through a permuted palette, so
  they defeat `decode_pixels`' calibration-from-row-0 recovery) as the hard positives.
- **`heliogram.instruments.saliency`** — recoverable bits by patch **position**: a per-grid-
  position symbol-error map over the corruption suite (`position_error_map(...)`). Model-free
  byproduct of the sweep. (Honest finding baked into the tests: `crop_pad` on this codec is a
  *cliff*, not a gradient, at `subpatch=1` — the measurable position effect lives at
  `subpatch=2`, where calibration and data cross their misalignment thresholds at different
  shifts. See the module docstring.)
- **`heliogram.instruments.fingerprint`** — a per-corruption symbol-error **signature** of an
  encode/decode config; `detect_swap(reference, observed)` flags a silently-swapped
  encoder/decoder in a blind test (demonstrated against `swapped_palette_encode`). Model-free.
- **`heliogram.instruments.injection_bench`** — the harness pointed at **behavioral** payloads
  (persona / schema / tool-call), with a **versioned submission format**
  (`RESULTS_FORMAT_VERSION`, `write_results`/`read_results`). The behavioral-capacity
  measurement (`measure_behavioral_capacity`) **requires a real model and raises without one**.
  The **detector-evaluation mode** (`evaluate_defense`) is CPU and runs here — e.g. scoring
  `foreign_tile.guard` as a candidate defense (TPR/FPR over injection vs. benign tiles).
- **`heliogram.instruments.learned_alphabet`** + **`heliogram.encoder`** — a CPU, model-free
  palette optimizer (`optimize_palette` / `compare_to_handcrafted`) that searches for colors
  minimizing symbol error under corruption and reports the learned code's error **beside** the
  handcrafted `get_palette(P)` baseline. **This optimizes against the pixel decoder, not a VLM
  encoder** — the same "channel measurement, not a capability claim" caveat as `subpatch>1`.
  The true frozen-encoder-gradient version (optimize *pixels* against a real ViT with input-
  pixel gradients) is the lazy-GPU scaffold `heliogram.encoder.FrozenEncoderHandle`, which
  raises without a GPU model and has not been run here.
- **`heliogram.instruments.bayes_bound`** — is large-palette-under-JPEG information
  **physically destroyed, or just unread?** `decode_pixels` samples ONE center pixel per patch,
  so its failures are a *lower* bound on the channel, not the channel. This instrument measures
  (near-)optimal classifiers over whole-patch statistics (a calibration-NN oracle and a
  labeled-split Gaussian oracle) against the RS budget (~6.3% symbol error). First run's
  decision-relevant readings: at `palette=128`/`jpeg_q70` the Gaussian oracle reads the channel
  at 0.5% error — **13× below budget: the information survives pure JPEG; the reference decoder
  was simply too weak, and a learned reader has real headroom there.** `palette=256`/`jpeg_q70`
  is borderline-destroyed (oracle 8.2% vs 6.3% budget); anything including the 2px shift
  (`combined`) destroys whole-patch statistics outright while center-pixel sampling shrugs it
  off — a measured, unanticipated trade both future decoders must navigate. CPU, ~4s.

## Roadmap / Phase-2 boundary

Phase 1 (this repo) is entirely model-free: codec, corruption suite, harness,
baselines. It answers one question — *how many bits does a patch carry through
realistic preprocessing?*

**Decision Gate #1:** Phase 2 starts only if the corrupted (worst-tested-case,
not mean) bits/patch number clears a working bar of **~8 bits/patch** —
originally a deliberate margin above the then-analytic ~6 bit/token base64
break-even. **The measured baseline inverted that relationship: base64 is
actually 8.096 bits/token (Baselines above), so the fixed 8.0 gate now sits
BELOW the real economic bar and no longer functions as a comfort margin.**
Bar A (beat measured base64 density, clean) and Bar C (token crossover) are
the two bars this project's actual benefit claim rests on (see "Capacity
sweep" above); Gate #1 is kept and reported for continuity, not as a
sufficient condition for anything.

The capacity/amortization/Gate sweep (see "Capacity sweep" above; full detail
in `RESULTS.md`) found that **0 of the 64 tested (palette, subpatch,
payload_size) configs clear the Gate #1 (8-bit) bar under the
default encoding** — the 3 that cleared it in the previous revision (all
`palette=8`/`subpatch=2`) are killed by the newly-measured
`qwen_smart_resize` corruption (the model's own preprocessing) and come back
only with the `align=2` encode-side fix (measured for 1KB/4KB; the 16KB grid
additionally needs the operator to widen the processor's pixel budget — see
the Bar B verdict in "Capacity sweep" above). Even setting `smart_resize`
aside, every near-clearing config uses `subpatch=2`, the unverified,
pixel-decoder-only geometric regime. **No `subpatch=1` (the actually
VLM-meaningful regime) config clears Gate #1, and none can by construction
at any payload size:** the hard per-symbol ceiling
for `subpatch=1` is `log2(P)`, which reaches exactly 8 only at the largest
palette, `P=256` — but Reed–Solomon/calibration overhead caps the achievable
*net* ceiling at `log2(P) × 223/255 ≈ 6.996` bits/patch as payload size grows
without bound, strictly below 8 for any finite payload (measured max in this
sweep: 6.827, palette=256, 16KB).

**Against the measured tokenizer baseline, the per-patch economic case for
`subpatch=1` is closed — no density win is mathematically possible (ceiling
6.996 < 8.096) and no token-count crossing exists in the scanned range. What
still motivates Phase 2 is a different, sharper pair of measured facts:**
(a) under **LM-token accounting** (Qwen2.5-VL's 2×2 merger, ~4× fewer LM
tokens than ViT patches), every palette ≥8 crosses below base64 from
~50–130B payloads — IF a model can read 4 patch-symbols through one merged
embedding, which is unverified and is now the project's central question;
and (b) the **`bayes_bound` instrument** measured that at `palette=128`
under pure JPEG q70 the symbol information demonstrably survives in
whole-patch statistics (Gaussian-oracle error 0.5% vs the ~6.3% RS budget)
even though `decode_pixels` fails there — real headroom for a learned
reader on the corruption axis too. Both benefit routes remain explicitly
conditional on a learned reader; nothing model-based is claimed here.
Phase 2 is GPU work and is **not** in this repo yet:

- **Step 0 — the frozen-encoder linear probe (`scripts/run_probe.py`, single-digit GPU-hours,
  runs BEFORE any training spend):** push heliogram grids through the stock, frozen Qwen2.5-VL
  vision tower and train a linear probe from merged-token embeddings to the 4 patch symbols
  each token covers (`heliogram/probe.py` — the model-free half is CPU-tested, including an
  end-to-end synthetic-encoder rehearsal of the exact token-ordering contract). Probe error
  at/below the RS budget on clean images ⇒ the information survives to the LM boundary and the
  fine-tune only has to teach the LM to read it; probe at chance on clean images ⇒ the tower
  discarded it and the LM-token branch dies for tens of dollars instead of a training run.
- **Fine-tune an open VLM to decode heliogram images natively, retargeted at exactly the
  measured gaps above — in this order:** first the LM-token readout question at a
  corruption-proof palette (`palette=16`, `subpatch=1`: can the model read 4 easy symbols per
  merged token at all — the cheapest decisive experiment, isolating readout from color
  robustness), then the corruption axis where `bayes_bound` measured real headroom
  (`palette=128` under `jpeg_q70`/`jpeg_q85`), and only then `palette=256` (whose q70 cell the
  same instrument measured as borderline-destroyed). Not sub-patch geometry (`subpatch>1` stays
  a documented, secondary, pixel-decoder-only geometric ceiling, unchanged from above). See
  `heliogram/dataset.py`'s retargeted defaults (`DEFAULT_PALETTES`) and
  `scripts/train_qlora.py`'s retargeted curriculum (`build_curriculum()`) for where this plays
  out; `heliogram.vlm.QwenVLDecoder` (see "Phase 2 (GPU)" below) is the decoder plug point, and
  the `VLMDecoder` stub in `heliogram/codec.py` remains the minimal hand-rolled alternative (it
  currently raises `NotImplementedError`).
- Measure the un-fine-tuned VLM's OCR on the rendered-text baseline for a fair comparison — not
  implemented anywhere in this repo; needs a real model run.
- **Report the exactness niche regardless of how that comparison lands** (see
  `heliogram/benefit.py`): Reed–Solomon gives detection *and* correction on every decode; a raw
  VLM transcription of rendered text gives neither, unless a checksum/ECC layer is added on top
  of it — at which point it has reinvented what heliogram already has natively. This holds even
  if a future measurement shows rendered-text OCR matching heliogram's bits/patch density; no
  OCR error rate is invented to make this argument, and the real number is itself Phase-2 work.
- Semantic use of optically-injected context (retrieval, tool outputs, logs).

No model-based numbers appear anywhere in this repo until Phase 2 produces them.

## Phase-2 safety release gate

Phase 2, if it happens, deliberately trains a model to do one specific new thing:
reliably read machine-dense, human-opaque images through realistic corruption.
That is exactly the capability that makes image-borne prompt injection
un-reviewable by a human in the loop — a payload a person can't visually tell
apart from noise, that the model reads anyway, is a channel nobody watching the
screen can catch. We built `heliogram.instruments.foreign_tile` (a structural
pre-ingest guard) and `heliogram.instruments.injection_bench` (a behavioral-
payload benchmark) in Phase 1, before any fine-tuning, on purpose — see
"Instruments (gate-independent)" above: the detector ships before the
capability does, not after. We are not going to reverse that ordering by
shipping a tuned reader without also shipping the numbers that say whether the
detector still catches it.

Before we release any Phase-2 adapter or fine-tuned weights, we commit to
publishing all three of the following, together, in the same measured,
caveat-attached style this repo's other numbers already use — not a blog post,
not a vibe:

1. **Behavioral capacity, on the tuned model.** Run
   `heliogram.instruments.injection_bench.measure_behavioral_capacity` against
   the actual fine-tuned reader — not the pixel decoder — and publish however
   much reliable behavioral influence (persona hijack, forced output schema,
   triggered tool call) survives the corruption suite. This is exactly the
   number `measure_behavioral_capacity` currently refuses to fabricate,
   because there is no model to run it against yet (see "Instruments"
   above); Phase 2 removes that excuse, and we intend to use it.
2. **Detector TPR/FPR, measured against what the tuned model actually reads.**
   Publish `heliogram.instruments.foreign_tile.guard`'s TPR at a bounded FPR,
   measured against tiles the TUNED model decodes — not only tiles
   `decode_pixels` decodes. A guard evaluated solely against the reference
   pixel decoder says nothing about whether it also catches what a learned
   reader, with its own generalization quirks, ends up picking up; this
   project does not get to grade its own detector on the easy version of the
   test.
3. **A decision rule, stated in advance, before there is any incentive to
   soften it:** no adapter or fine-tuned weights are released if behavioral
   payloads survive corruption at rates `foreign_tile.guard` cannot catch at a
   bounded false-positive rate. "Survive" and "catch" here mean exactly what
   (1) and (2) above measure — a published number compared against another
   published number, not a judgment call made after the fact.

This gate is a commitment about what gets published and what release is
conditioned on, not a claim that we have already run it: nothing in this repo
has a GPU, so neither (1) nor (2) has been measured yet (see "Phase 2 (GPU)"
below). We are writing the gate down before the fine-tuned model exists
specifically so there is no room to relax it once there is a working model
someone is proud of.

## Phase 2 (GPU) — how to run when you have a GPU

> **Start with [`RUNBOOK-GPU.md`](RUNBOOK-GPU.md)** — the ordered, copy-paste
> runbook for a rented-GPU session (setup → baselines → Step-0 probe → optional
> zero-shot floor → QLoRA), with per-step decision rules and the list of
> artifacts to commit back. The model-facing code below has been CPU-verified
> against `transformers==5.13.0` with a random-weight Qwen2.5-VL
> (`tests/test_probe_contract_cpu.py`, `tests/test_train_qlora_lora_targets.py`)
> — that verification caught three run-blocking defects (visual-tower attribute
> path, merged-tokens-in-`pooler_output`, and a PEFT suffix-matching leak that
> silently LoRA-tuned the vision blocks), all fixed on this branch. What remains
> genuinely unrun is the same code against real weights.

Everything above this section is Phase 1: model-free. This repo also ships a **Phase-2
scaffold** — code to generate training data, fine-tune a VLM, and plug it into the same
`decode()` call Phase 1 uses — that only *runs* once you have a GPU. Nothing in it has been
executed here: there is no GPU in this environment, so every number anywhere else in this
README and in `RESULTS.md` still comes from `decode_pixels` (see the Scope note at the top of
`RESULTS.md`). Importing `heliogram` never requires `torch`/`transformers`/`peft`/
`bitsandbytes` — those are imported lazily, only inside the functions that actually need them
(`heliogram/vlm.py`, `scripts/train_qlora.py`); `python -c "import heliogram"` succeeds with
only `pillow`/`numpy`/`reedsolo` installed, same as Phase 1. One exception to "nothing in this
section has been executed": `heliogram/benefit.py` (step 6 below) is model-free like Phase 1
and its numbers ARE real, run on demand on CPU — it is scaffold in the sense of anticipating
Phase 2's outcome (the exactness argument, the token-count accounting), not in the sense of
being unrun.

1. **Install GPU extras** (on a machine with a GPU):

   ```bash
   pip install -e ".[gpu]"          # or: pip install -r requirements-gpu.txt
   ```

2. **Generate a synthetic training set** — `heliogram/dataset.py`, CPU-only, needs no GPU and
   runs fine in this same environment. It reuses `encode`/`extract_symbols` from
   `heliogram.codec`, so every training target is the *exact* ground-truth symbol sequence the
   codec wrote for that image — no hand labeling, no drift between what was encoded and what
   the model is asked to transcribe. Its defaults are retargeted at the large-palette bet above:
   `--palettes` defaults to `{64, 128, 256}` and `--corruption-prob` defaults to `0.5` (see
   `heliogram/dataset.py`'s module docstring), so a bare invocation already generates data for
   exactly the gap this project needs closed:

   ```bash
   python scripts/gen_dataset.py --out data/phase2_train --n 2000 --seed 0
   python scripts/gen_dataset.py --out data/phase2_val   --n 200  --seed 1
   ```

   This writes `data/phase2_train/images/*.png` plus a `manifest.jsonl` with one
   `{"image_path", "palette", "subpatch", "target", ...}` record per line — `target` is the
   symbol sequence rendered as a compact string (see `heliogram.dataset.symbols_to_target`).

3. **(Optional) Measure the stock model's zero-shot symbol error first.**
   `heliogram.vlm.zero_shot_symbol_error(model, processor, configs)` runs an *unmodified*
   Qwen2.5-VL (or whatever you load) directly over encoded images and reports real
   symbol-error/decode-success numbers against ground truth. It never returns a number without
   a real model behind it — pass `model=None` and it raises immediately instead of guessing.

4. **QLoRA fine-tune** — `scripts/train_qlora.py` loads Qwen2.5-VL-7B-Instruct in 4-bit
   (bitsandbytes NF4) via Transformers, attaches LoRA adapters (PEFT) on the attention/
   projection layers, and trains across a curriculum: a cheap small-palette warm-up, then three
   stages that concentrate specifically on `palette` in `{64, 128, 256}` at `subpatch=1` —
   clean first, then corrupted, then a final stage that up-weights the specific corruptions
   (`jpeg_q70`/`jpeg_q85`/`combined`) this palette range is measured to fail under — see
   `build_curriculum()`:

   ```bash
   python scripts/train_qlora.py --output-dir checkpoints/qwen25vl-heliogram-lora
   ```

   This script is a documented **starting point, not a verified recipe** — see its top-of-file
   docstring for VRAM expectations and its explicit "GPU required, untested here" caveat.

5. **Plug the fine-tuned model into `decode()`**, exactly like the model-free reference decoder:

   ```python
   from heliogram import decode
   from heliogram.vlm import QwenVLDecoder

   decoder = QwenVLDecoder(model=my_loaded_model, processor=my_loaded_processor,
                            palette=256, subpatch=1)   # the large-palette bet -- see above
   payload = decode(img, palette=256, subpatch=1, decoder=decoder)
   ```

   `QwenVLDecoder` prompts the model to transcribe the symbol grid, parses the response back
   into symbols, and feeds them through the *same* Reed–Solomon/framing layer `decode_pixels`
   uses — there is no separate RS implementation for the model path.

6. **See the token savings and the exactness niche, right now, without a GPU** —
   `python -m heliogram.benefit` encodes a real ~6KB synthetic JSON payload at `palette=256`,
   reports patches vs. base64 tokens vs. a raw-byte (hex) text-encoding baseline, and then
   *demonstrates* the mandatory caveat live: it decodes the clean image (passes, bit-exact) and
   then a real JPEG q70 re-encode of that same image (fails, exactly as `codec.py`/`RESULTS.md`
   measure) — so the "this needs the Phase-2 reader" caveat is a fresh measurement on every run,
   not a citation. The same module also prints the exactness argument: Reed–Solomon gives
   detection *and* correction on every decode; free-form OCR gives neither — a durable niche
   even if a future measurement shows rendered-text OCR matching heliogram's bits/patch density
   (see "Roadmap / Phase-2 boundary" above and `heliogram/benefit.py`'s module docstring).

**The boundary this doesn't cross.** `heliogram/dataset.py`, `scripts/gen_dataset.py`, and
`heliogram/benefit.py` are all CPU-only and *have* been exercised in this repo (the first two
via `tests/`; `heliogram/benefit.py` by direct runs of `python -m heliogram.benefit`, not yet
its own pytest file). Everything past that — loading a real VLM, `zero_shot_symbol_error` with
a real model, `scripts/train_qlora.py`, `QwenVLDecoder.__call__` actually reaching
`.generate()` — requires a GPU this environment doesn't have and is untested here. **Decision Gate #1 (see "Roadmap / Phase-2 boundary" above)
is decided by measured numbers from an actual fine-tuned model, run through this same
`decode()` path, on held-out data — not by this scaffold's existence.** Until that run happens
and its numbers are written up the same way `RESULTS.md` was, treat everything in this section
as "code that should let someone with a GPU go measure this," not as a result.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 heliogram contributors.
