# heliogram

**If one ViT patch (~1 vision token) can reliably carry more than one text token's worth of data (~6 bits for base64), then encoded images are a cheaper context medium for self-hosted VLMs — heliogram measures whether that's true.**

heliogram is a patch-aligned optical codec plus an evaluation harness. It encodes
arbitrary bytes as a grid of solid-color 14x14 blocks (one symbol per ViT patch),
protects them with Reed–Solomon ECC, runs them through the corruptions a real
serving pipeline applies (resize, JPEG, crop/pad), and reports **effective
bits/patch** — the channel capacity of the patch grid, measured with a model-free
reference decoder. Format details: [`spec/format-v0.1.md`](spec/format-v0.1.md).

Apache-2.0. All data is synthetic and seed-deterministic; same inputs produce
byte-identical PNGs.

## Scope

- **In scope:** self-hosted / open-weight VLMs where the operator controls image
  preprocessing end to end (resolution, patching, no server-side re-encoding they
  can't inspect).
- **Out of scope:** closed API models. Their preprocessing (resizing, tiling,
  re-compression, token accounting) is opaque and changeable, so no capacity claim
  made here transfers to them. We do not test against them and won't.

This is a measurement project. If the numbers come out below the base64 line, that
is a result, not a failure — it bounds what optical-context schemes can gain from
denser-than-text symbol coding, and we'll report it as such.

## Results

> Measured by the eval harness (`python -m heliogram.harness`), which writes
> `RESULTS.md` and `results.csv`. Full breakdown, self-consistency checks, and
> a diagnostic stress test beyond this envelope are in [`RESULTS.md`](RESULTS.md).

Effective bits/patch = bits_per_symbol x (data patches / total patches) x
(payload bytes / ECC-coded bytes), counted only on successful RS decode (0 on
a failed decode). "Corrupted" is the mean bits/patch over every non-clean
corruption tested: bilinear resize ±3–5%, JPEG q70–95, a slight crop/pad, and
their composition ("combined"). Reference pixel decoder (`decode_pixels`), no
model. 48-byte synthetic payloads, nsym=32, patch_size=14px, 5 trials/cell.

| Palette | bits/symbol | Clean bits/patch | Corrupted bits/patch |
|--------:|------------:|-----------------:|----------------------:|
| 2       | 1           | 0.544            | 0.544                 |
| 4       | 2           | 1.070            | 1.070                 |
| 8       | 3           | 1.588            | 1.588                 |
| 16      | 4           | 2.071            | 2.071                 |

The break-even line to beat is **~6 bits/patch**; the codec is currently well
below it at every palette (see Baselines) — this is a real result, not a
failure to report. Clean and corrupted are identical at every palette because
`decode_success_rate` is 1.00 across the whole realistic-serving-pipeline
corruption envelope tested here (resize ±3–5%, JPEG q70–95, slight crop/pad):
Reed–Solomon (nsym=32) fully absorbs the symbol errors that envelope
introduces (largest observed symbol error rate: 0.0011, at palette=16 /
JPEG q70). A diagnostic stress test well outside that envelope (50% resize,
JPEG q10, 6px crop/pad, composed) does find the expected fall-off at larger
palettes — decode success drops to 0.00 at palette≥4 under the composed
stress case — confirming the channel has a real breaking point, just further
out than a typical serving pipeline pushes it. See `RESULTS.md` for that
table. The current bottleneck at these small (48B) payload sizes is fixed
overhead (32-byte RS parity + calibration row), not corruption; bits/patch
grows with payload size (see `demo.py`, which measures 2.37 bits/patch for a
178-byte payload at the same palette=8 the harness measures at 1.59 bits/patch
for 48 bytes).

## Baselines

- **base64 in text context: ~6 bits/token.** base64 encodes 6 payload bits per
  character, and common tokenizers emit roughly one token per base64 character
  (often slightly better via multi-char merges). `heliogram.baselines.
  base64_bits_per_token()` returns the 6.0 analytic figure by default and accepts
  a HuggingFace tokenizer to measure the real ratio for your model.
- **Rendered text (honesty guardrail).** The obvious competitor is just
  typesetting the payload small and letting the VLM read it — that's what
  DeepSeek-OCR and Glyph exploit. `heliogram.baselines` typesets the payload onto
  the same patch grid and reports chars/patch as a geometric, model-free estimate.
  The *true* rendered-text capacity depends on the un-fine-tuned VLM's OCR and is
  a Phase-2 measurement. If rendered text matches the codec's bits/patch, the
  codec's only remaining advantages are exactness (ECC-verified bytes) and
  robustness — and we'll say so.

## Quickstart

```bash
pip install -e .
python demo.py            # encode a small JSON doc, decode it back,
                          # print patch count vs. base64 token count
python -m heliogram.harness   # full palette x corruption sweep -> RESULTS.md, CSV
pytest                    # roundtrip, determinism, calibration-recovery tests
```

Requires Python with `pillow`, `numpy`, `reedsolo` (installed by the command above).

## How it works (v0.1, one paragraph)

Payload bytes are framed (version byte + 4-byte length), Reed–Solomon coded
(default 32 parity bytes), split into log2(P)-bit symbols for a palette of
P ∈ {2, 4, 8, 16} deterministic, maximally separable colors, and painted one
symbol per 14x14 patch on a square-ish grid. Row 0 is a calibration row cycling
through the palette so the decoder can recover the palette's post-corruption RGB
values and nearest-neighbor classify every data patch. The reference decoder
(`decode_pixels`) samples patch centers only — it is deliberately dumb, so that
what it measures is the channel, not decoder cleverness. Full spec:
[`spec/format-v0.1.md`](spec/format-v0.1.md).

## Roadmap / Phase-2 boundary

Phase 1 (this repo) is entirely model-free: codec, corruption suite, harness,
baselines. It answers one question — *how many bits does a patch carry through
realistic preprocessing?*

**Decision Gate #1:** Phase 2 starts only if the corrupted bits/patch number
meaningfully beats ~6 bits/patch. Phase 2 is GPU work and is **not** in this
repo yet:

- Fine-tune an open VLM to decode heliogram images natively (the
  `VLMDecoder` stub in `heliogram/codec.py` is the plug point; it currently
  raises `NotImplementedError`).
- Measure the un-fine-tuned VLM's OCR on the rendered-text baseline for a fair
  comparison.
- Semantic use of optically-injected context (retrieval, tool outputs, logs).

No model-based numbers appear anywhere in this repo until Phase 2 produces them.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright 2026 heliogram contributors.
