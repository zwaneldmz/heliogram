# heliogram v0.1 -- CPU eval results

Synthetic, seed-deterministic payloads (48 random bytes/trial, 5 trials/cell), nsym=32, patch_size=14px. Reference decoder = decode_pixels (no model).

**Scope: this file characterizes the CODE/CHANNEL only.** Every number below comes from `decode_pixels`, the model-free reference decoder (pixel sampling + nearest-neighbor classification + Reed-Solomon, no VLM in the loop). Whether a fine-tuned VLM can realize this same capacity through its own vision encoder is Phase 2 and is not measured anywhere in this repo -- see the README's "Roadmap / Phase-2 boundary" section.

## Baselines

- **base64 in text context:** ~6.0 bits/token (analytic: base64 alphabet size 64 -> log2(64)=6 bits/char; ~1 char/token for typical BPE tokenizers on base64 streams. Pass a real tokenizer for a measured value.)
- **Rendered text (geometric, model-free):** 2.13 chars/patch = 12.80 bits/patch typesetting this trial's 48-byte payload (base64'd, 64 chars) into 30 patches of the same 14px grid unit. geometric/model-free: measures typeset packing density only, assumes perfect legibility. Real bits/patch for rendered text needs OCR accuracy from an un-fine-tuned VLM (Phase 2, out of scope here).

## Summary (matches README results table)

| Palette | bits/symbol | Clean bits/patch | Corrupted bits/patch |
|--------:|------------:|------------------:|----------------------:|
| 2 | 1 | 0.544 | 0.544 |
| 4 | 2 | 1.070 | 1.070 |
| 8 | 3 | 1.588 | 1.588 |
| 16 | 4 | 2.071 | 2.071 |

'Corrupted' above is the mean bits/patch over every non-clean row in the breakdown table below (resize 3%/5%, JPEG q95/85/70, crop/pad 2px, and their composition 'combined'), each counted as 0 on a failed decode.

## Full breakdown by corruption

| palette | bits/sym | corruption | symbol error rate | decode success rate | bits/patch |
|---|---|---|---|---|---|
| 2 | 1 | clean | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | resize_3pct | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | resize_5pct | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | combined | 0.0000 | 1.00 | 0.544 |
| 4 | 2 | clean | 0.0000 | 1.00 | 1.070 |
| 4 | 2 | resize_3pct | 0.0000 | 1.00 | 1.070 |
| 4 | 2 | resize_5pct | 0.0000 | 1.00 | 1.070 |
| 4 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.070 |
| 4 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.070 |
| 4 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.070 |
| 4 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.070 |
| 4 | 2 | combined | 0.0000 | 1.00 | 1.070 |
| 8 | 3 | clean | 0.0000 | 1.00 | 1.588 |
| 8 | 3 | resize_3pct | 0.0000 | 1.00 | 1.588 |
| 8 | 3 | resize_5pct | 0.0000 | 1.00 | 1.588 |
| 8 | 3 | jpeg_q95 | 0.0000 | 1.00 | 1.588 |
| 8 | 3 | jpeg_q85 | 0.0000 | 1.00 | 1.588 |
| 8 | 3 | jpeg_q70 | 0.0000 | 1.00 | 1.588 |
| 8 | 3 | crop_pad_2px | 0.0000 | 1.00 | 1.588 |
| 8 | 3 | combined | 0.0000 | 1.00 | 1.588 |
| 16 | 4 | clean | 0.0000 | 1.00 | 2.071 |
| 16 | 4 | resize_3pct | 0.0000 | 1.00 | 2.071 |
| 16 | 4 | resize_5pct | 0.0000 | 1.00 | 2.071 |
| 16 | 4 | jpeg_q95 | 0.0000 | 1.00 | 2.071 |
| 16 | 4 | jpeg_q85 | 0.0000 | 1.00 | 2.071 |
| 16 | 4 | jpeg_q70 | 0.0011 | 1.00 | 2.071 |
| 16 | 4 | crop_pad_2px | 0.0000 | 1.00 | 2.071 |
| 16 | 4 | combined | 0.0000 | 1.00 | 2.071 |

## Self-consistency checks

Two invariants must hold if these numbers mean what they claim to mean: (1) bits/patch can never exceed log2(palette) -- that is the raw per-symbol density before calibration-row and Reed-Solomon overhead are subtracted; (2) corrupted bits/patch can never exceed clean bits/patch for the same palette, since corruption only ever removes information relative to the uncorrupted image.

| palette | log2(palette) | clean bits/patch | <= log2(palette)? | corrupted bits/patch | <= clean? |
|---|---|---|---|---|---|
| 2 | 1 | 0.544 | yes | 0.544 | yes |
| 4 | 2 | 1.070 | yes | 1.070 | yes |
| 8 | 3 | 1.588 | yes | 1.588 | yes |
| 16 | 4 | 2.071 | yes | 2.071 | yes |

Both hold for every palette above. Note that clean == corrupted in the summary table: within the realistic corruption envelope this harness applies (resize +-1-5%, JPEG q70-95, slight crop/pad, and their composition), `decode_success_rate` is 1.00 for every cell (see the breakdown table), so Reed-Solomon (nsym=32, correcting up to 16 byte errors per 255-byte chunk) fully absorbs the symbol errors this envelope introduces -- the largest observed symbol_error_rate is 0.0011 (palette=16, jpeg_q70), far under the ~6% byte-error budget nsym=32 buys for this payload size. That is a real result (this corruption envelope does not stress the channel's ECC margin for any tested palette), not a stuck-at-1.0 measurement bug.

## Beyond the realistic envelope (diagnostic, not part of the headline table)

To confirm decode failure is actually reachable by this harness (i.e. that 100% success above is a real headroom finding and not a bug that can never observe failure), the same trials were re-run under corruption well outside the 'realistic serving pipeline' envelope: 50% bilinear resize round-trip, JPEG q10, a 6px crop/pad, and their composition.

| palette | corruption | symbol error rate | decode success rate | bits/patch |
|---|---|---|---|---|
| 2 | stress_resize_50pct | 0.0000 | 1.00 | 0.544 |
| 2 | stress_jpeg_q10 | 0.0000 | 1.00 | 0.544 |
| 2 | stress_crop_pad_6px | 0.0000 | 1.00 | 0.544 |
| 2 | stress_combined | 0.0205 | 0.80 | 0.435 |
| 4 | stress_resize_50pct | 0.0000 | 1.00 | 1.070 |
| 4 | stress_jpeg_q10 | 0.0000 | 1.00 | 1.070 |
| 4 | stress_crop_pad_6px | 0.0000 | 1.00 | 1.070 |
| 4 | stress_combined | 0.1158 | 0.00 | 0.000 |
| 8 | stress_resize_50pct | 0.0000 | 1.00 | 1.588 |
| 8 | stress_jpeg_q10 | 0.0200 | 1.00 | 1.588 |
| 8 | stress_crop_pad_6px | 0.0000 | 1.00 | 1.588 |
| 8 | stress_combined | 0.2825 | 0.00 | 0.000 |
| 16 | stress_resize_50pct | 0.0000 | 1.00 | 2.071 |
| 16 | stress_jpeg_q10 | 0.1216 | 0.20 | 0.414 |
| 16 | stress_crop_pad_6px | 0.0000 | 1.00 | 2.071 |
| 16 | stress_combined | 0.6114 | 0.00 | 0.000 |

Decode success drops well below 1.00 at higher palettes under this diagnostic stress suite, confirming the channel does have a real breaking point -- it simply lies beyond the resize/JPEG/crop ranges a typical serving pipeline applies, which is why the headline table above shows no degradation.
