# heliogram v0.1 -- CPU eval results

Synthetic, seed-deterministic payloads. Capacity sweep: palette in [2, 4, 8, 16, 32, 64], subpatch (k) in [1, 2], payload_size (bytes) in [48, 1024, 4096, 16384], x 8 corruptions (incl. 'clean'), 3 trials/cell, nsym=32, patch_size=14px. Reference decoder = decode_pixels (no model).

**Scope: this file characterizes the CODE/CHANNEL only.** Every number below comes from `decode_pixels`, the model-free reference decoder (pixel sampling + nearest-neighbor classification + Reed-Solomon, no VLM in the loop). Whether a fine-tuned VLM can realize this same capacity through its own vision encoder is Phase 2 and is not measured anywhere in this repo -- see the README's "Roadmap / Phase-2 boundary" section.

**Wall-clock note:** the full sweep below is 6 palettes x 2 subpatch values x 4 payload sizes x 8 corruptions = 384 cells; at the largest payload tier (16384B) each cell encodes/corrupts/decodes a multi-thousand-patch image, so trial count for this sweep was reduced to 3 (module default is 5) to bound wall-clock. The diagnostic stress suite below still runs at the module default 5 trials, at a single representative config (subpatch=1, payload_size=48B) -- see that section.

## Headline: does any config clear the Gate #1 bar?

**Gate #1 bar: 8.0 bits/patch.** A config "clears the gate" only if its bits/patch is at or above this bar BOTH on a clean image AND in its single worst-performing tested corruption (not the mean of all corruptions) -- a config that only clears on average is not a robust win.

**MANDATORY honesty caveat:** rows with `subpatch=1` are the VLM-meaningful regime -- one symbol per DATA patch, i.e. one symbol per (nominal) vision token, the only regime this project claims any real relevance to a downstream VLM. Rows with `subpatch>1` are a **PIXEL-DECODER GEOMETRIC CEILING ONLY**: `decode_pixels`/`extract_symbols` can read sub-patch cells trivially because they sample known, exact pixel coordinates off a grid whose size they are told in advance -- there is no perception involved. Whether a real ViT/VLM image encoder can resolve sub-patch structure at all is **unverified, and doubtful** (a k x k sub-cell grid inside one ViT patch may simply average out in that patch's embedding). Realizing it is Phase 2 work, gated on GPU access, and is **not a capability claim** made anywhere in this repo.

| palette | subpatch | payload (B) | ceiling k²·log2(P) | clean bits/patch | clears 8 clean? | worst-corruption bits/patch | worst corruption | clears 8 corrupted? | clears gate (both)? |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 1 | 0.544 | no | 0.544 | resize_3pct | no | no |
| 2 | 1 | 1024 | 1 | 0.853 | no | 0.853 | resize_3pct | no | no |
| 2 | 1 | 4096 | 1 | 0.865 | no | 0.865 | resize_3pct | no | no |
| 2 | 1 | 16384 | 1 | 0.871 | no | 0.871 | resize_3pct | no | no |
| 2 | 2 | 48 | 4 | 2.097 | no | 2.097 | resize_3pct | no | no |
| 2 | 2 | 1024 | 4 | 3.376 | no | 3.376 | resize_3pct | no | no |
| 2 | 2 | 4096 | 4 | 3.444 | no | 3.444 | resize_3pct | no | no |
| 2 | 2 | 16384 | 4 | 3.476 | no | 3.476 | resize_3pct | no | no |
| 4 | 1 | 48 | 2 | 1.070 | no | 1.070 | resize_3pct | no | no |
| 4 | 1 | 1024 | 2 | 1.698 | no | 1.698 | resize_3pct | no | no |
| 4 | 1 | 4096 | 2 | 1.727 | no | 1.727 | resize_3pct | no | no |
| 4 | 1 | 16384 | 2 | 1.741 | no | 1.741 | resize_3pct | no | no |
| 4 | 2 | 48 | 8 | 4.066 | no | 4.066 | resize_3pct | no | no |
| 4 | 2 | 1024 | 8 | 6.693 | no | 6.693 | resize_3pct | no | no |
| 4 | 2 | 4096 | 8 | 6.859 | no | 6.859 | resize_3pct | no | no |
| 4 | 2 | 16384 | 8 | 6.937 | no | 6.937 | resize_3pct | no | no |
| 8 | 1 | 48 | 3 | 1.588 | no | 1.588 | resize_3pct | no | no |
| 8 | 1 | 1024 | 3 | 2.538 | no | 2.538 | resize_3pct | no | no |
| 8 | 1 | 4096 | 3 | 2.586 | no | 2.586 | resize_3pct | no | no |
| 8 | 1 | 16384 | 3 | 2.609 | no | 2.609 | resize_3pct | no | no |
| 8 | 2 | 48 | 12 | 6.024 | no | 0.000 | combined | no | no |
| 8 | 2 | 1024 | 12 | 9.978 | yes | 9.978 | resize_3pct | yes | **YES** |
| 8 | 2 | 4096 | 12 | 10.255 | yes | 10.255 | resize_3pct | yes | **YES** |
| 8 | 2 | 16384 | 12 | 10.389 | yes | 10.389 | resize_3pct | yes | **YES** |
| 16 | 1 | 48 | 4 | 2.071 | no | 2.071 | resize_3pct | no | no |
| 16 | 1 | 1024 | 4 | 3.376 | no | 3.376 | resize_3pct | no | no |
| 16 | 1 | 4096 | 4 | 3.444 | no | 3.444 | resize_3pct | no | no |
| 16 | 1 | 16384 | 4 | 3.476 | no | 3.476 | resize_3pct | no | no |
| 16 | 2 | 48 | 16 | 6.776 | no | 0.000 | combined | no | no |
| 16 | 2 | 1024 | 16 | 13.228 | yes | 0.000 | combined | no | no |
| 16 | 2 | 4096 | 16 | 13.639 | yes | 0.000 | combined | no | no |
| 16 | 2 | 16384 | 16 | 13.833 | yes | 0.000 | combined | no | no |
| 32 | 1 | 48 | 5 | 2.353 | no | 2.353 | resize_3pct | no | no |
| 32 | 1 | 1024 | 5 | 4.210 | no | 0.000 | jpeg_q70 | no | no |
| 32 | 1 | 4096 | 5 | 4.300 | no | 0.000 | jpeg_q70 | no | no |
| 32 | 1 | 16384 | 5 | 4.342 | no | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 48 | 20 | 7.529 | no | 0.000 | combined | no | no |
| 32 | 2 | 1024 | 20 | 16.148 | yes | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 4096 | 20 | 17.001 | yes | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 16384 | 20 | 17.271 | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 1 | 48 | 6 | 2.259 | no | 2.259 | resize_3pct | no | no |
| 64 | 1 | 1024 | 6 | 4.969 | no | 0.000 | jpeg_q70 | no | no |
| 64 | 1 | 4096 | 6 | 5.154 | no | 0.000 | combined | no | no |
| 64 | 1 | 16384 | 6 | 5.208 | no | 1.736 | jpeg_q70 | no | no |
| 64 | 2 | 48 | 24 | 6.776 | no | 0.000 | combined | no | no |
| 64 | 2 | 1024 | 24 | 18.086 | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 2 | 4096 | 24 | 20.073 | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 2 | 16384 | 24 | 20.702 | yes | 0.000 | jpeg_q70 | no | no |

**Configs that clear the gate (both clean and worst-case corruption):**

- palette=8, subpatch=2, payload_size=1024B -- clean 9.978 bits/patch, worst 9.978 bits/patch (worst corruption: `resize_3pct`)
- palette=8, subpatch=2, payload_size=4096B -- clean 10.255 bits/patch, worst 10.255 bits/patch (worst corruption: `resize_3pct`)
- palette=8, subpatch=2, payload_size=16384B -- clean 10.389 bits/patch, worst 10.389 bits/patch (worst corruption: `resize_3pct`)

**Verdict (derived from the table above, not asserted independently):**

Every clearing config has `subpatch>1` -- the unverified pixel-decoder geometric ceiling regime. **No `subpatch=1` (VLM-meaningful) config clears the gate at any tested payload size.** This is not just an unlucky corruption result: for `subpatch=1` the raw per-symbol ceiling is `log2(palette)`, which for the largest palette tested (64) is only 6 bits/patch -- already below the 8-bit bar *before* Reed-Solomon/calibration overhead is even subtracted. No amount of payload-size amortization can close that gap for `subpatch=1`; only the geometric `subpatch>1` regime can mathematically reach the bar, and whether a real VLM can realize that regime is exactly the open question Phase 2 exists to answer.

## Baselines

- **base64 in text context:** ~6.0 bits/token (analytic: base64 alphabet size 64 -> log2(64)=6 bits/char; ~1 char/token for typical BPE tokenizers on base64 streams. Pass a real tokenizer for a measured value.)
- **Rendered text (geometric, model-free):** 2.13 chars/patch = 12.80 bits/patch typesetting a 48-byte payload (base64'd, 64 chars) into 30 patches of the same 14px grid unit. geometric/model-free: measures typeset packing density only, assumes perfect legibility. Real bits/patch for rendered text needs OCR accuracy from an un-fine-tuned VLM (Phase 2, out of scope here).

## Summary by sub-patch regime (payload-size amortization)

Fixed per-message overhead (5-byte frame header + Reed-Solomon parity + the calibration row) is amortized over more data patches as payload size grows, so bits/patch should rise toward the `subpatch²·log2(palette)` ceiling as payload grows -- this is the amortization half of this sweep. 'corr(mean)' is the mean bits/patch over every non-clean corruption in the table below (resize 3%/5%, JPEG q95/85/70, crop/pad 2px, combined), each counted as 0 on a failed decode.

### subpatch=1 (VLM-meaningful: one symbol per patch)

| Palette | bits/sym | ceiling | 48B clean | 48B corr(mean) | 1024B clean | 1024B corr(mean) | 4096B clean | 4096B corr(mean) | 16384B clean | 16384B corr(mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 0.544 | 0.544 | 0.853 | 0.853 | 0.865 | 0.865 | 0.871 | 0.871 |
| 4 | 2 | 2 | 1.070 | 1.070 | 1.698 | 1.698 | 1.727 | 1.727 | 1.741 | 1.741 |
| 8 | 3 | 3 | 1.588 | 1.588 | 2.538 | 2.538 | 2.586 | 2.586 | 2.609 | 2.609 |
| 16 | 4 | 4 | 2.071 | 2.071 | 3.376 | 3.376 | 3.444 | 3.444 | 3.476 | 3.476 |
| 32 | 5 | 5 | 2.353 | 2.353 | 4.210 | 3.408 | 4.300 | 3.071 | 4.342 | 3.102 |
| 64 | 6 | 6 | 2.259 | 2.259 | 4.969 | 4.022 | 5.154 | 3.927 | 5.208 | 4.712 |

### subpatch=2 (PIXEL-DECODER GEOMETRIC CEILING ONLY -- not a VLM capability claim, see caveat above)

| Palette | bits/sym | ceiling | 48B clean | 48B corr(mean) | 1024B clean | 1024B corr(mean) | 4096B clean | 4096B corr(mean) | 16384B clean | 16384B corr(mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 4 | 2.097 | 2.097 | 3.376 | 3.376 | 3.444 | 3.444 | 3.476 | 3.476 |
| 4 | 2 | 8 | 4.066 | 4.066 | 6.693 | 6.693 | 6.859 | 6.859 | 6.937 | 6.937 |
| 8 | 3 | 12 | 6.024 | 5.163 | 9.978 | 9.978 | 10.255 | 10.255 | 10.389 | 10.389 |
| 16 | 4 | 16 | 6.776 | 5.808 | 13.228 | 11.339 | 13.639 | 11.690 | 13.833 | 11.857 |
| 32 | 5 | 20 | 7.529 | 6.454 | 16.148 | 11.534 | 17.001 | 12.144 | 17.271 | 12.337 |
| 64 | 6 | 24 | 6.776 | 5.808 | 18.086 | 12.918 | 20.073 | 14.338 | 20.702 | 14.787 |

## Full breakdown by corruption

| palette | subpatch | payload | bits/sym | corruption | symbol error rate | decode success rate | bits/patch |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 1 | clean | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | 48 | 1 | resize_3pct | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | 48 | 1 | resize_5pct | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | 48 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | 48 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | 48 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | 48 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | 48 | 1 | combined | 0.0000 | 1.00 | 0.544 |
| 2 | 1 | 1024 | 1 | clean | 0.0000 | 1.00 | 0.853 |
| 2 | 1 | 1024 | 1 | resize_3pct | 0.0000 | 1.00 | 0.853 |
| 2 | 1 | 1024 | 1 | resize_5pct | 0.0000 | 1.00 | 0.853 |
| 2 | 1 | 1024 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.853 |
| 2 | 1 | 1024 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.853 |
| 2 | 1 | 1024 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.853 |
| 2 | 1 | 1024 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.853 |
| 2 | 1 | 1024 | 1 | combined | 0.0000 | 1.00 | 0.853 |
| 2 | 1 | 4096 | 1 | clean | 0.0000 | 1.00 | 0.865 |
| 2 | 1 | 4096 | 1 | resize_3pct | 0.0000 | 1.00 | 0.865 |
| 2 | 1 | 4096 | 1 | resize_5pct | 0.0000 | 1.00 | 0.865 |
| 2 | 1 | 4096 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.865 |
| 2 | 1 | 4096 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.865 |
| 2 | 1 | 4096 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.865 |
| 2 | 1 | 4096 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.865 |
| 2 | 1 | 4096 | 1 | combined | 0.0000 | 1.00 | 0.865 |
| 2 | 1 | 16384 | 1 | clean | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | resize_3pct | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | resize_5pct | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | combined | 0.0000 | 1.00 | 0.871 |
| 2 | 2 | 48 | 1 | clean | 0.0000 | 1.00 | 2.097 |
| 2 | 2 | 48 | 1 | resize_3pct | 0.0000 | 1.00 | 2.097 |
| 2 | 2 | 48 | 1 | resize_5pct | 0.0000 | 1.00 | 2.097 |
| 2 | 2 | 48 | 1 | jpeg_q95 | 0.0000 | 1.00 | 2.097 |
| 2 | 2 | 48 | 1 | jpeg_q85 | 0.0000 | 1.00 | 2.097 |
| 2 | 2 | 48 | 1 | jpeg_q70 | 0.0000 | 1.00 | 2.097 |
| 2 | 2 | 48 | 1 | crop_pad_2px | 0.0000 | 1.00 | 2.097 |
| 2 | 2 | 48 | 1 | combined | 0.0000 | 1.00 | 2.097 |
| 2 | 2 | 1024 | 1 | clean | 0.0000 | 1.00 | 3.376 |
| 2 | 2 | 1024 | 1 | resize_3pct | 0.0000 | 1.00 | 3.376 |
| 2 | 2 | 1024 | 1 | resize_5pct | 0.0000 | 1.00 | 3.376 |
| 2 | 2 | 1024 | 1 | jpeg_q95 | 0.0000 | 1.00 | 3.376 |
| 2 | 2 | 1024 | 1 | jpeg_q85 | 0.0000 | 1.00 | 3.376 |
| 2 | 2 | 1024 | 1 | jpeg_q70 | 0.0000 | 1.00 | 3.376 |
| 2 | 2 | 1024 | 1 | crop_pad_2px | 0.0000 | 1.00 | 3.376 |
| 2 | 2 | 1024 | 1 | combined | 0.0000 | 1.00 | 3.376 |
| 2 | 2 | 4096 | 1 | clean | 0.0000 | 1.00 | 3.444 |
| 2 | 2 | 4096 | 1 | resize_3pct | 0.0000 | 1.00 | 3.444 |
| 2 | 2 | 4096 | 1 | resize_5pct | 0.0000 | 1.00 | 3.444 |
| 2 | 2 | 4096 | 1 | jpeg_q95 | 0.0000 | 1.00 | 3.444 |
| 2 | 2 | 4096 | 1 | jpeg_q85 | 0.0000 | 1.00 | 3.444 |
| 2 | 2 | 4096 | 1 | jpeg_q70 | 0.0000 | 1.00 | 3.444 |
| 2 | 2 | 4096 | 1 | crop_pad_2px | 0.0000 | 1.00 | 3.444 |
| 2 | 2 | 4096 | 1 | combined | 0.0000 | 1.00 | 3.444 |
| 2 | 2 | 16384 | 1 | clean | 0.0000 | 1.00 | 3.476 |
| 2 | 2 | 16384 | 1 | resize_3pct | 0.0000 | 1.00 | 3.476 |
| 2 | 2 | 16384 | 1 | resize_5pct | 0.0000 | 1.00 | 3.476 |
| 2 | 2 | 16384 | 1 | jpeg_q95 | 0.0000 | 1.00 | 3.476 |
| 2 | 2 | 16384 | 1 | jpeg_q85 | 0.0000 | 1.00 | 3.476 |
| 2 | 2 | 16384 | 1 | jpeg_q70 | 0.0000 | 1.00 | 3.476 |
| 2 | 2 | 16384 | 1 | crop_pad_2px | 0.0000 | 1.00 | 3.476 |
| 2 | 2 | 16384 | 1 | combined | 0.0000 | 1.00 | 3.476 |
| 4 | 1 | 48 | 2 | clean | 0.0000 | 1.00 | 1.070 |
| 4 | 1 | 48 | 2 | resize_3pct | 0.0000 | 1.00 | 1.070 |
| 4 | 1 | 48 | 2 | resize_5pct | 0.0000 | 1.00 | 1.070 |
| 4 | 1 | 48 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.070 |
| 4 | 1 | 48 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.070 |
| 4 | 1 | 48 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.070 |
| 4 | 1 | 48 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.070 |
| 4 | 1 | 48 | 2 | combined | 0.0000 | 1.00 | 1.070 |
| 4 | 1 | 1024 | 2 | clean | 0.0000 | 1.00 | 1.698 |
| 4 | 1 | 1024 | 2 | resize_3pct | 0.0000 | 1.00 | 1.698 |
| 4 | 1 | 1024 | 2 | resize_5pct | 0.0000 | 1.00 | 1.698 |
| 4 | 1 | 1024 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.698 |
| 4 | 1 | 1024 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.698 |
| 4 | 1 | 1024 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.698 |
| 4 | 1 | 1024 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.698 |
| 4 | 1 | 1024 | 2 | combined | 0.0000 | 1.00 | 1.698 |
| 4 | 1 | 4096 | 2 | clean | 0.0000 | 1.00 | 1.727 |
| 4 | 1 | 4096 | 2 | resize_3pct | 0.0000 | 1.00 | 1.727 |
| 4 | 1 | 4096 | 2 | resize_5pct | 0.0000 | 1.00 | 1.727 |
| 4 | 1 | 4096 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.727 |
| 4 | 1 | 4096 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.727 |
| 4 | 1 | 4096 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.727 |
| 4 | 1 | 4096 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.727 |
| 4 | 1 | 4096 | 2 | combined | 0.0000 | 1.00 | 1.727 |
| 4 | 1 | 16384 | 2 | clean | 0.0000 | 1.00 | 1.741 |
| 4 | 1 | 16384 | 2 | resize_3pct | 0.0000 | 1.00 | 1.741 |
| 4 | 1 | 16384 | 2 | resize_5pct | 0.0000 | 1.00 | 1.741 |
| 4 | 1 | 16384 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.741 |
| 4 | 1 | 16384 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.741 |
| 4 | 1 | 16384 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.741 |
| 4 | 1 | 16384 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.741 |
| 4 | 1 | 16384 | 2 | combined | 0.0000 | 1.00 | 1.741 |
| 4 | 2 | 48 | 2 | clean | 0.0000 | 1.00 | 4.066 |
| 4 | 2 | 48 | 2 | resize_3pct | 0.0000 | 1.00 | 4.066 |
| 4 | 2 | 48 | 2 | resize_5pct | 0.0000 | 1.00 | 4.066 |
| 4 | 2 | 48 | 2 | jpeg_q95 | 0.0000 | 1.00 | 4.066 |
| 4 | 2 | 48 | 2 | jpeg_q85 | 0.0000 | 1.00 | 4.066 |
| 4 | 2 | 48 | 2 | jpeg_q70 | 0.0000 | 1.00 | 4.066 |
| 4 | 2 | 48 | 2 | crop_pad_2px | 0.0000 | 1.00 | 4.066 |
| 4 | 2 | 48 | 2 | combined | 0.0000 | 1.00 | 4.066 |
| 4 | 2 | 1024 | 2 | clean | 0.0000 | 1.00 | 6.693 |
| 4 | 2 | 1024 | 2 | resize_3pct | 0.0000 | 1.00 | 6.693 |
| 4 | 2 | 1024 | 2 | resize_5pct | 0.0000 | 1.00 | 6.693 |
| 4 | 2 | 1024 | 2 | jpeg_q95 | 0.0000 | 1.00 | 6.693 |
| 4 | 2 | 1024 | 2 | jpeg_q85 | 0.0000 | 1.00 | 6.693 |
| 4 | 2 | 1024 | 2 | jpeg_q70 | 0.0000 | 1.00 | 6.693 |
| 4 | 2 | 1024 | 2 | crop_pad_2px | 0.0000 | 1.00 | 6.693 |
| 4 | 2 | 1024 | 2 | combined | 0.0000 | 1.00 | 6.693 |
| 4 | 2 | 4096 | 2 | clean | 0.0000 | 1.00 | 6.859 |
| 4 | 2 | 4096 | 2 | resize_3pct | 0.0000 | 1.00 | 6.859 |
| 4 | 2 | 4096 | 2 | resize_5pct | 0.0000 | 1.00 | 6.859 |
| 4 | 2 | 4096 | 2 | jpeg_q95 | 0.0000 | 1.00 | 6.859 |
| 4 | 2 | 4096 | 2 | jpeg_q85 | 0.0000 | 1.00 | 6.859 |
| 4 | 2 | 4096 | 2 | jpeg_q70 | 0.0000 | 1.00 | 6.859 |
| 4 | 2 | 4096 | 2 | crop_pad_2px | 0.0000 | 1.00 | 6.859 |
| 4 | 2 | 4096 | 2 | combined | 0.0000 | 1.00 | 6.859 |
| 4 | 2 | 16384 | 2 | clean | 0.0000 | 1.00 | 6.937 |
| 4 | 2 | 16384 | 2 | resize_3pct | 0.0000 | 1.00 | 6.937 |
| 4 | 2 | 16384 | 2 | resize_5pct | 0.0000 | 1.00 | 6.937 |
| 4 | 2 | 16384 | 2 | jpeg_q95 | 0.0000 | 1.00 | 6.937 |
| 4 | 2 | 16384 | 2 | jpeg_q85 | 0.0000 | 1.00 | 6.937 |
| 4 | 2 | 16384 | 2 | jpeg_q70 | 0.0000 | 1.00 | 6.937 |
| 4 | 2 | 16384 | 2 | crop_pad_2px | 0.0000 | 1.00 | 6.937 |
| 4 | 2 | 16384 | 2 | combined | 0.0000 | 1.00 | 6.937 |
| 8 | 1 | 48 | 3 | clean | 0.0000 | 1.00 | 1.588 |
| 8 | 1 | 48 | 3 | resize_3pct | 0.0000 | 1.00 | 1.588 |
| 8 | 1 | 48 | 3 | resize_5pct | 0.0000 | 1.00 | 1.588 |
| 8 | 1 | 48 | 3 | jpeg_q95 | 0.0000 | 1.00 | 1.588 |
| 8 | 1 | 48 | 3 | jpeg_q85 | 0.0000 | 1.00 | 1.588 |
| 8 | 1 | 48 | 3 | jpeg_q70 | 0.0000 | 1.00 | 1.588 |
| 8 | 1 | 48 | 3 | crop_pad_2px | 0.0000 | 1.00 | 1.588 |
| 8 | 1 | 48 | 3 | combined | 0.0000 | 1.00 | 1.588 |
| 8 | 1 | 1024 | 3 | clean | 0.0000 | 1.00 | 2.538 |
| 8 | 1 | 1024 | 3 | resize_3pct | 0.0000 | 1.00 | 2.538 |
| 8 | 1 | 1024 | 3 | resize_5pct | 0.0000 | 1.00 | 2.538 |
| 8 | 1 | 1024 | 3 | jpeg_q95 | 0.0000 | 1.00 | 2.538 |
| 8 | 1 | 1024 | 3 | jpeg_q85 | 0.0000 | 1.00 | 2.538 |
| 8 | 1 | 1024 | 3 | jpeg_q70 | 0.0000 | 1.00 | 2.538 |
| 8 | 1 | 1024 | 3 | crop_pad_2px | 0.0000 | 1.00 | 2.538 |
| 8 | 1 | 1024 | 3 | combined | 0.0000 | 1.00 | 2.538 |
| 8 | 1 | 4096 | 3 | clean | 0.0000 | 1.00 | 2.586 |
| 8 | 1 | 4096 | 3 | resize_3pct | 0.0000 | 1.00 | 2.586 |
| 8 | 1 | 4096 | 3 | resize_5pct | 0.0000 | 1.00 | 2.586 |
| 8 | 1 | 4096 | 3 | jpeg_q95 | 0.0000 | 1.00 | 2.586 |
| 8 | 1 | 4096 | 3 | jpeg_q85 | 0.0000 | 1.00 | 2.586 |
| 8 | 1 | 4096 | 3 | jpeg_q70 | 0.0000 | 1.00 | 2.586 |
| 8 | 1 | 4096 | 3 | crop_pad_2px | 0.0000 | 1.00 | 2.586 |
| 8 | 1 | 4096 | 3 | combined | 0.0000 | 1.00 | 2.586 |
| 8 | 1 | 16384 | 3 | clean | 0.0000 | 1.00 | 2.609 |
| 8 | 1 | 16384 | 3 | resize_3pct | 0.0000 | 1.00 | 2.609 |
| 8 | 1 | 16384 | 3 | resize_5pct | 0.0000 | 1.00 | 2.609 |
| 8 | 1 | 16384 | 3 | jpeg_q95 | 0.0000 | 1.00 | 2.609 |
| 8 | 1 | 16384 | 3 | jpeg_q85 | 0.0000 | 1.00 | 2.609 |
| 8 | 1 | 16384 | 3 | jpeg_q70 | 0.0000 | 1.00 | 2.609 |
| 8 | 1 | 16384 | 3 | crop_pad_2px | 0.0000 | 1.00 | 2.609 |
| 8 | 1 | 16384 | 3 | combined | 0.0000 | 1.00 | 2.609 |
| 8 | 2 | 48 | 3 | clean | 0.0000 | 1.00 | 6.024 |
| 8 | 2 | 48 | 3 | resize_3pct | 0.0000 | 1.00 | 6.024 |
| 8 | 2 | 48 | 3 | resize_5pct | 0.0000 | 1.00 | 6.024 |
| 8 | 2 | 48 | 3 | jpeg_q95 | 0.0000 | 1.00 | 6.024 |
| 8 | 2 | 48 | 3 | jpeg_q85 | 0.0000 | 1.00 | 6.024 |
| 8 | 2 | 48 | 3 | jpeg_q70 | 0.0000 | 1.00 | 6.024 |
| 8 | 2 | 48 | 3 | crop_pad_2px | 0.0000 | 1.00 | 6.024 |
| 8 | 2 | 48 | 3 | combined | 0.0130 | 0.00 | 0.000 |
| 8 | 2 | 1024 | 3 | clean | 0.0000 | 1.00 | 9.978 |
| 8 | 2 | 1024 | 3 | resize_3pct | 0.0000 | 1.00 | 9.978 |
| 8 | 2 | 1024 | 3 | resize_5pct | 0.0000 | 1.00 | 9.978 |
| 8 | 2 | 1024 | 3 | jpeg_q95 | 0.0000 | 1.00 | 9.978 |
| 8 | 2 | 1024 | 3 | jpeg_q85 | 0.0000 | 1.00 | 9.978 |
| 8 | 2 | 1024 | 3 | jpeg_q70 | 0.0000 | 1.00 | 9.978 |
| 8 | 2 | 1024 | 3 | crop_pad_2px | 0.0000 | 1.00 | 9.978 |
| 8 | 2 | 1024 | 3 | combined | 0.0055 | 1.00 | 9.978 |
| 8 | 2 | 4096 | 3 | clean | 0.0000 | 1.00 | 10.255 |
| 8 | 2 | 4096 | 3 | resize_3pct | 0.0000 | 1.00 | 10.255 |
| 8 | 2 | 4096 | 3 | resize_5pct | 0.0000 | 1.00 | 10.255 |
| 8 | 2 | 4096 | 3 | jpeg_q95 | 0.0000 | 1.00 | 10.255 |
| 8 | 2 | 4096 | 3 | jpeg_q85 | 0.0000 | 1.00 | 10.255 |
| 8 | 2 | 4096 | 3 | jpeg_q70 | 0.0000 | 1.00 | 10.255 |
| 8 | 2 | 4096 | 3 | crop_pad_2px | 0.0000 | 1.00 | 10.255 |
| 8 | 2 | 4096 | 3 | combined | 0.0071 | 1.00 | 10.255 |
| 8 | 2 | 16384 | 3 | clean | 0.0000 | 1.00 | 10.389 |
| 8 | 2 | 16384 | 3 | resize_3pct | 0.0000 | 1.00 | 10.389 |
| 8 | 2 | 16384 | 3 | resize_5pct | 0.0000 | 1.00 | 10.389 |
| 8 | 2 | 16384 | 3 | jpeg_q95 | 0.0000 | 1.00 | 10.389 |
| 8 | 2 | 16384 | 3 | jpeg_q85 | 0.0000 | 1.00 | 10.389 |
| 8 | 2 | 16384 | 3 | jpeg_q70 | 0.0000 | 1.00 | 10.389 |
| 8 | 2 | 16384 | 3 | crop_pad_2px | 0.0000 | 1.00 | 10.389 |
| 8 | 2 | 16384 | 3 | combined | 0.0062 | 1.00 | 10.389 |
| 16 | 1 | 48 | 4 | clean | 0.0000 | 1.00 | 2.071 |
| 16 | 1 | 48 | 4 | resize_3pct | 0.0000 | 1.00 | 2.071 |
| 16 | 1 | 48 | 4 | resize_5pct | 0.0000 | 1.00 | 2.071 |
| 16 | 1 | 48 | 4 | jpeg_q95 | 0.0000 | 1.00 | 2.071 |
| 16 | 1 | 48 | 4 | jpeg_q85 | 0.0000 | 1.00 | 2.071 |
| 16 | 1 | 48 | 4 | jpeg_q70 | 0.0000 | 1.00 | 2.071 |
| 16 | 1 | 48 | 4 | crop_pad_2px | 0.0000 | 1.00 | 2.071 |
| 16 | 1 | 48 | 4 | combined | 0.0000 | 1.00 | 2.071 |
| 16 | 1 | 1024 | 4 | clean | 0.0000 | 1.00 | 3.376 |
| 16 | 1 | 1024 | 4 | resize_3pct | 0.0000 | 1.00 | 3.376 |
| 16 | 1 | 1024 | 4 | resize_5pct | 0.0000 | 1.00 | 3.376 |
| 16 | 1 | 1024 | 4 | jpeg_q95 | 0.0000 | 1.00 | 3.376 |
| 16 | 1 | 1024 | 4 | jpeg_q85 | 0.0000 | 1.00 | 3.376 |
| 16 | 1 | 1024 | 4 | jpeg_q70 | 0.0003 | 1.00 | 3.376 |
| 16 | 1 | 1024 | 4 | crop_pad_2px | 0.0000 | 1.00 | 3.376 |
| 16 | 1 | 1024 | 4 | combined | 0.0001 | 1.00 | 3.376 |
| 16 | 1 | 4096 | 4 | clean | 0.0000 | 1.00 | 3.444 |
| 16 | 1 | 4096 | 4 | resize_3pct | 0.0000 | 1.00 | 3.444 |
| 16 | 1 | 4096 | 4 | resize_5pct | 0.0000 | 1.00 | 3.444 |
| 16 | 1 | 4096 | 4 | jpeg_q95 | 0.0000 | 1.00 | 3.444 |
| 16 | 1 | 4096 | 4 | jpeg_q85 | 0.0000 | 1.00 | 3.444 |
| 16 | 1 | 4096 | 4 | jpeg_q70 | 0.0007 | 1.00 | 3.444 |
| 16 | 1 | 4096 | 4 | crop_pad_2px | 0.0000 | 1.00 | 3.444 |
| 16 | 1 | 4096 | 4 | combined | 0.0002 | 1.00 | 3.444 |
| 16 | 1 | 16384 | 4 | clean | 0.0000 | 1.00 | 3.476 |
| 16 | 1 | 16384 | 4 | resize_3pct | 0.0000 | 1.00 | 3.476 |
| 16 | 1 | 16384 | 4 | resize_5pct | 0.0000 | 1.00 | 3.476 |
| 16 | 1 | 16384 | 4 | jpeg_q95 | 0.0000 | 1.00 | 3.476 |
| 16 | 1 | 16384 | 4 | jpeg_q85 | 0.0000 | 1.00 | 3.476 |
| 16 | 1 | 16384 | 4 | jpeg_q70 | 0.0007 | 1.00 | 3.476 |
| 16 | 1 | 16384 | 4 | crop_pad_2px | 0.0000 | 1.00 | 3.476 |
| 16 | 1 | 16384 | 4 | combined | 0.0001 | 1.00 | 3.476 |
| 16 | 2 | 48 | 4 | clean | 0.0000 | 1.00 | 6.776 |
| 16 | 2 | 48 | 4 | resize_3pct | 0.0000 | 1.00 | 6.776 |
| 16 | 2 | 48 | 4 | resize_5pct | 0.0000 | 1.00 | 6.776 |
| 16 | 2 | 48 | 4 | jpeg_q95 | 0.0000 | 1.00 | 6.776 |
| 16 | 2 | 48 | 4 | jpeg_q85 | 0.0000 | 1.00 | 6.776 |
| 16 | 2 | 48 | 4 | jpeg_q70 | 0.0035 | 1.00 | 6.776 |
| 16 | 2 | 48 | 4 | crop_pad_2px | 0.0000 | 1.00 | 6.776 |
| 16 | 2 | 48 | 4 | combined | 0.1354 | 0.00 | 0.000 |
| 16 | 2 | 1024 | 4 | clean | 0.0000 | 1.00 | 13.228 |
| 16 | 2 | 1024 | 4 | resize_3pct | 0.0000 | 1.00 | 13.228 |
| 16 | 2 | 1024 | 4 | resize_5pct | 0.0000 | 1.00 | 13.228 |
| 16 | 2 | 1024 | 4 | jpeg_q95 | 0.0000 | 1.00 | 13.228 |
| 16 | 2 | 1024 | 4 | jpeg_q85 | 0.0000 | 1.00 | 13.228 |
| 16 | 2 | 1024 | 4 | jpeg_q70 | 0.0037 | 1.00 | 13.228 |
| 16 | 2 | 1024 | 4 | crop_pad_2px | 0.0000 | 1.00 | 13.228 |
| 16 | 2 | 1024 | 4 | combined | 0.1260 | 0.00 | 0.000 |
| 16 | 2 | 4096 | 4 | clean | 0.0000 | 1.00 | 13.639 |
| 16 | 2 | 4096 | 4 | resize_3pct | 0.0000 | 1.00 | 13.639 |
| 16 | 2 | 4096 | 4 | resize_5pct | 0.0000 | 1.00 | 13.639 |
| 16 | 2 | 4096 | 4 | jpeg_q95 | 0.0000 | 1.00 | 13.639 |
| 16 | 2 | 4096 | 4 | jpeg_q85 | 0.0000 | 1.00 | 13.639 |
| 16 | 2 | 4096 | 4 | jpeg_q70 | 0.0018 | 1.00 | 13.639 |
| 16 | 2 | 4096 | 4 | crop_pad_2px | 0.0000 | 1.00 | 13.639 |
| 16 | 2 | 4096 | 4 | combined | 0.1179 | 0.00 | 0.000 |
| 16 | 2 | 16384 | 4 | clean | 0.0000 | 1.00 | 13.833 |
| 16 | 2 | 16384 | 4 | resize_3pct | 0.0000 | 1.00 | 13.833 |
| 16 | 2 | 16384 | 4 | resize_5pct | 0.0000 | 1.00 | 13.833 |
| 16 | 2 | 16384 | 4 | jpeg_q95 | 0.0000 | 1.00 | 13.833 |
| 16 | 2 | 16384 | 4 | jpeg_q85 | 0.0000 | 1.00 | 13.833 |
| 16 | 2 | 16384 | 4 | jpeg_q70 | 0.0019 | 1.00 | 13.833 |
| 16 | 2 | 16384 | 4 | crop_pad_2px | 0.0000 | 1.00 | 13.833 |
| 16 | 2 | 16384 | 4 | combined | 0.1223 | 0.00 | 0.000 |
| 32 | 1 | 48 | 5 | clean | 0.0000 | 1.00 | 2.353 |
| 32 | 1 | 48 | 5 | resize_3pct | 0.0000 | 1.00 | 2.353 |
| 32 | 1 | 48 | 5 | resize_5pct | 0.0000 | 1.00 | 2.353 |
| 32 | 1 | 48 | 5 | jpeg_q95 | 0.0000 | 1.00 | 2.353 |
| 32 | 1 | 48 | 5 | jpeg_q85 | 0.0021 | 1.00 | 2.353 |
| 32 | 1 | 48 | 5 | jpeg_q70 | 0.0417 | 1.00 | 2.353 |
| 32 | 1 | 48 | 5 | crop_pad_2px | 0.0000 | 1.00 | 2.353 |
| 32 | 1 | 48 | 5 | combined | 0.0354 | 1.00 | 2.353 |
| 32 | 1 | 1024 | 5 | clean | 0.0000 | 1.00 | 4.210 |
| 32 | 1 | 1024 | 5 | resize_3pct | 0.0000 | 1.00 | 4.210 |
| 32 | 1 | 1024 | 5 | resize_5pct | 0.0000 | 1.00 | 4.210 |
| 32 | 1 | 1024 | 5 | jpeg_q95 | 0.0000 | 1.00 | 4.210 |
| 32 | 1 | 1024 | 5 | jpeg_q85 | 0.0083 | 1.00 | 4.210 |
| 32 | 1 | 1024 | 5 | jpeg_q70 | 0.0491 | 0.00 | 0.000 |
| 32 | 1 | 1024 | 5 | crop_pad_2px | 0.0000 | 1.00 | 4.210 |
| 32 | 1 | 1024 | 5 | combined | 0.0344 | 0.67 | 2.807 |
| 32 | 1 | 4096 | 5 | clean | 0.0000 | 1.00 | 4.300 |
| 32 | 1 | 4096 | 5 | resize_3pct | 0.0000 | 1.00 | 4.300 |
| 32 | 1 | 4096 | 5 | resize_5pct | 0.0000 | 1.00 | 4.300 |
| 32 | 1 | 4096 | 5 | jpeg_q95 | 0.0000 | 1.00 | 4.300 |
| 32 | 1 | 4096 | 5 | jpeg_q85 | 0.0041 | 1.00 | 4.300 |
| 32 | 1 | 4096 | 5 | jpeg_q70 | 0.0320 | 0.00 | 0.000 |
| 32 | 1 | 4096 | 5 | crop_pad_2px | 0.0000 | 1.00 | 4.300 |
| 32 | 1 | 4096 | 5 | combined | 0.0243 | 0.00 | 0.000 |
| 32 | 1 | 16384 | 5 | clean | 0.0000 | 1.00 | 4.342 |
| 32 | 1 | 16384 | 5 | resize_3pct | 0.0000 | 1.00 | 4.342 |
| 32 | 1 | 16384 | 5 | resize_5pct | 0.0000 | 1.00 | 4.342 |
| 32 | 1 | 16384 | 5 | jpeg_q95 | 0.0000 | 1.00 | 4.342 |
| 32 | 1 | 16384 | 5 | jpeg_q85 | 0.0032 | 1.00 | 4.342 |
| 32 | 1 | 16384 | 5 | jpeg_q70 | 0.0278 | 0.00 | 0.000 |
| 32 | 1 | 16384 | 5 | crop_pad_2px | 0.0000 | 1.00 | 4.342 |
| 32 | 1 | 16384 | 5 | combined | 0.0221 | 0.00 | 0.000 |
| 32 | 2 | 48 | 5 | clean | 0.0000 | 1.00 | 7.529 |
| 32 | 2 | 48 | 5 | resize_3pct | 0.0000 | 1.00 | 7.529 |
| 32 | 2 | 48 | 5 | resize_5pct | 0.0000 | 1.00 | 7.529 |
| 32 | 2 | 48 | 5 | jpeg_q95 | 0.0000 | 1.00 | 7.529 |
| 32 | 2 | 48 | 5 | jpeg_q85 | 0.0065 | 1.00 | 7.529 |
| 32 | 2 | 48 | 5 | jpeg_q70 | 0.0404 | 1.00 | 7.529 |
| 32 | 2 | 48 | 5 | crop_pad_2px | 0.0000 | 1.00 | 7.529 |
| 32 | 2 | 48 | 5 | combined | 0.1719 | 0.00 | 0.000 |
| 32 | 2 | 1024 | 5 | clean | 0.0000 | 1.00 | 16.148 |
| 32 | 2 | 1024 | 5 | resize_3pct | 0.0000 | 1.00 | 16.148 |
| 32 | 2 | 1024 | 5 | resize_5pct | 0.0000 | 1.00 | 16.148 |
| 32 | 2 | 1024 | 5 | jpeg_q95 | 0.0000 | 1.00 | 16.148 |
| 32 | 2 | 1024 | 5 | jpeg_q85 | 0.0146 | 1.00 | 16.148 |
| 32 | 2 | 1024 | 5 | jpeg_q70 | 0.0988 | 0.00 | 0.000 |
| 32 | 2 | 1024 | 5 | crop_pad_2px | 0.0000 | 1.00 | 16.148 |
| 32 | 2 | 1024 | 5 | combined | 0.2698 | 0.00 | 0.000 |
| 32 | 2 | 4096 | 5 | clean | 0.0000 | 1.00 | 17.001 |
| 32 | 2 | 4096 | 5 | resize_3pct | 0.0000 | 1.00 | 17.001 |
| 32 | 2 | 4096 | 5 | resize_5pct | 0.0000 | 1.00 | 17.001 |
| 32 | 2 | 4096 | 5 | jpeg_q95 | 0.0000 | 1.00 | 17.001 |
| 32 | 2 | 4096 | 5 | jpeg_q85 | 0.0187 | 1.00 | 17.001 |
| 32 | 2 | 4096 | 5 | jpeg_q70 | 0.0930 | 0.00 | 0.000 |
| 32 | 2 | 4096 | 5 | crop_pad_2px | 0.0000 | 1.00 | 17.001 |
| 32 | 2 | 4096 | 5 | combined | 0.2750 | 0.00 | 0.000 |
| 32 | 2 | 16384 | 5 | clean | 0.0000 | 1.00 | 17.271 |
| 32 | 2 | 16384 | 5 | resize_3pct | 0.0000 | 1.00 | 17.271 |
| 32 | 2 | 16384 | 5 | resize_5pct | 0.0000 | 1.00 | 17.271 |
| 32 | 2 | 16384 | 5 | jpeg_q95 | 0.0000 | 1.00 | 17.271 |
| 32 | 2 | 16384 | 5 | jpeg_q85 | 0.0103 | 1.00 | 17.271 |
| 32 | 2 | 16384 | 5 | jpeg_q70 | 0.0719 | 0.00 | 0.000 |
| 32 | 2 | 16384 | 5 | crop_pad_2px | 0.0000 | 1.00 | 17.271 |
| 32 | 2 | 16384 | 5 | combined | 0.2670 | 0.00 | 0.000 |
| 64 | 1 | 48 | 6 | clean | 0.0000 | 1.00 | 2.259 |
| 64 | 1 | 48 | 6 | resize_3pct | 0.0000 | 1.00 | 2.259 |
| 64 | 1 | 48 | 6 | resize_5pct | 0.0000 | 1.00 | 2.259 |
| 64 | 1 | 48 | 6 | jpeg_q95 | 0.0000 | 1.00 | 2.259 |
| 64 | 1 | 48 | 6 | jpeg_q85 | 0.0052 | 1.00 | 2.259 |
| 64 | 1 | 48 | 6 | jpeg_q70 | 0.0182 | 1.00 | 2.259 |
| 64 | 1 | 48 | 6 | crop_pad_2px | 0.0000 | 1.00 | 2.259 |
| 64 | 1 | 48 | 6 | combined | 0.0026 | 1.00 | 2.259 |
| 64 | 1 | 1024 | 6 | clean | 0.0000 | 1.00 | 4.969 |
| 64 | 1 | 1024 | 6 | resize_3pct | 0.0000 | 1.00 | 4.969 |
| 64 | 1 | 1024 | 6 | resize_5pct | 0.0000 | 1.00 | 4.969 |
| 64 | 1 | 1024 | 6 | jpeg_q95 | 0.0000 | 1.00 | 4.969 |
| 64 | 1 | 1024 | 6 | jpeg_q85 | 0.0037 | 1.00 | 4.969 |
| 64 | 1 | 1024 | 6 | jpeg_q70 | 0.0419 | 0.00 | 0.000 |
| 64 | 1 | 1024 | 6 | crop_pad_2px | 0.0000 | 1.00 | 4.969 |
| 64 | 1 | 1024 | 6 | combined | 0.0306 | 0.67 | 3.312 |
| 64 | 1 | 4096 | 6 | clean | 0.0000 | 1.00 | 5.154 |
| 64 | 1 | 4096 | 6 | resize_3pct | 0.0000 | 1.00 | 5.154 |
| 64 | 1 | 4096 | 6 | resize_5pct | 0.0000 | 1.00 | 5.154 |
| 64 | 1 | 4096 | 6 | jpeg_q95 | 0.0000 | 1.00 | 5.154 |
| 64 | 1 | 4096 | 6 | jpeg_q85 | 0.0032 | 1.00 | 5.154 |
| 64 | 1 | 4096 | 6 | jpeg_q70 | 0.0387 | 0.33 | 1.718 |
| 64 | 1 | 4096 | 6 | crop_pad_2px | 0.0000 | 1.00 | 5.154 |
| 64 | 1 | 4096 | 6 | combined | 0.0318 | 0.00 | 0.000 |
| 64 | 1 | 16384 | 6 | clean | 0.0000 | 1.00 | 5.208 |
| 64 | 1 | 16384 | 6 | resize_3pct | 0.0000 | 1.00 | 5.208 |
| 64 | 1 | 16384 | 6 | resize_5pct | 0.0000 | 1.00 | 5.208 |
| 64 | 1 | 16384 | 6 | jpeg_q95 | 0.0000 | 1.00 | 5.208 |
| 64 | 1 | 16384 | 6 | jpeg_q85 | 0.0017 | 1.00 | 5.208 |
| 64 | 1 | 16384 | 6 | jpeg_q70 | 0.0252 | 0.33 | 1.736 |
| 64 | 1 | 16384 | 6 | crop_pad_2px | 0.0000 | 1.00 | 5.208 |
| 64 | 1 | 16384 | 6 | combined | 0.0191 | 1.00 | 5.208 |
| 64 | 2 | 48 | 6 | clean | 0.0000 | 1.00 | 6.776 |
| 64 | 2 | 48 | 6 | resize_3pct | 0.0000 | 1.00 | 6.776 |
| 64 | 2 | 48 | 6 | resize_5pct | 0.0000 | 1.00 | 6.776 |
| 64 | 2 | 48 | 6 | jpeg_q95 | 0.0000 | 1.00 | 6.776 |
| 64 | 2 | 48 | 6 | jpeg_q85 | 0.0000 | 1.00 | 6.776 |
| 64 | 2 | 48 | 6 | jpeg_q70 | 0.0169 | 1.00 | 6.776 |
| 64 | 2 | 48 | 6 | crop_pad_2px | 0.0000 | 1.00 | 6.776 |
| 64 | 2 | 48 | 6 | combined | 0.3008 | 0.00 | 0.000 |
| 64 | 2 | 1024 | 6 | clean | 0.0000 | 1.00 | 18.086 |
| 64 | 2 | 1024 | 6 | resize_3pct | 0.0000 | 1.00 | 18.086 |
| 64 | 2 | 1024 | 6 | resize_5pct | 0.0000 | 1.00 | 18.086 |
| 64 | 2 | 1024 | 6 | jpeg_q95 | 0.0000 | 1.00 | 18.086 |
| 64 | 2 | 1024 | 6 | jpeg_q85 | 0.0082 | 1.00 | 18.086 |
| 64 | 2 | 1024 | 6 | jpeg_q70 | 0.0761 | 0.00 | 0.000 |
| 64 | 2 | 1024 | 6 | crop_pad_2px | 0.0000 | 1.00 | 18.086 |
| 64 | 2 | 1024 | 6 | combined | 0.4230 | 0.00 | 0.000 |
| 64 | 2 | 4096 | 6 | clean | 0.0000 | 1.00 | 20.073 |
| 64 | 2 | 4096 | 6 | resize_3pct | 0.0000 | 1.00 | 20.073 |
| 64 | 2 | 4096 | 6 | resize_5pct | 0.0000 | 1.00 | 20.073 |
| 64 | 2 | 4096 | 6 | jpeg_q95 | 0.0000 | 1.00 | 20.073 |
| 64 | 2 | 4096 | 6 | jpeg_q85 | 0.0085 | 1.00 | 20.073 |
| 64 | 2 | 4096 | 6 | jpeg_q70 | 0.0877 | 0.00 | 0.000 |
| 64 | 2 | 4096 | 6 | crop_pad_2px | 0.0000 | 1.00 | 20.073 |
| 64 | 2 | 4096 | 6 | combined | 0.4342 | 0.00 | 0.000 |
| 64 | 2 | 16384 | 6 | clean | 0.0000 | 1.00 | 20.702 |
| 64 | 2 | 16384 | 6 | resize_3pct | 0.0000 | 1.00 | 20.702 |
| 64 | 2 | 16384 | 6 | resize_5pct | 0.0000 | 1.00 | 20.702 |
| 64 | 2 | 16384 | 6 | jpeg_q95 | 0.0000 | 1.00 | 20.702 |
| 64 | 2 | 16384 | 6 | jpeg_q85 | 0.0082 | 1.00 | 20.702 |
| 64 | 2 | 16384 | 6 | jpeg_q70 | 0.0832 | 0.00 | 0.000 |
| 64 | 2 | 16384 | 6 | crop_pad_2px | 0.0000 | 1.00 | 20.702 |
| 64 | 2 | 16384 | 6 | combined | 0.4348 | 0.00 | 0.000 |

## Self-consistency checks

Two invariants must hold if these numbers mean what they claim to mean: (1) bits/patch can never exceed `subpatch²·log2(palette)` -- the raw per-DATA-PATCH density for a subpatch x subpatch grid of symbols per patch, before calibration-row and Reed-Solomon overhead are subtracted (this generalizes the pre-Slice-B `<= log2(palette)` check, which is the `subpatch=1` case where `subpatch²=1`); (2) mean corrupted bits/patch can never exceed clean bits/patch for the same (palette, subpatch, payload_size), since corruption only ever removes information relative to the uncorrupted image.

| palette | subpatch | payload | ceiling subpatch²·log2(P) | clean bits/patch | <= ceiling? | corrupted(mean) bits/patch | <= clean? |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 1 | 0.544 | yes | 0.544 | yes |
| 2 | 1 | 1024 | 1 | 0.853 | yes | 0.853 | yes |
| 2 | 1 | 4096 | 1 | 0.865 | yes | 0.865 | yes |
| 2 | 1 | 16384 | 1 | 0.871 | yes | 0.871 | yes |
| 2 | 2 | 48 | 4 | 2.097 | yes | 2.097 | yes |
| 2 | 2 | 1024 | 4 | 3.376 | yes | 3.376 | yes |
| 2 | 2 | 4096 | 4 | 3.444 | yes | 3.444 | yes |
| 2 | 2 | 16384 | 4 | 3.476 | yes | 3.476 | yes |
| 4 | 1 | 48 | 2 | 1.070 | yes | 1.070 | yes |
| 4 | 1 | 1024 | 2 | 1.698 | yes | 1.698 | yes |
| 4 | 1 | 4096 | 2 | 1.727 | yes | 1.727 | yes |
| 4 | 1 | 16384 | 2 | 1.741 | yes | 1.741 | yes |
| 4 | 2 | 48 | 8 | 4.066 | yes | 4.066 | yes |
| 4 | 2 | 1024 | 8 | 6.693 | yes | 6.693 | yes |
| 4 | 2 | 4096 | 8 | 6.859 | yes | 6.859 | yes |
| 4 | 2 | 16384 | 8 | 6.937 | yes | 6.937 | yes |
| 8 | 1 | 48 | 3 | 1.588 | yes | 1.588 | yes |
| 8 | 1 | 1024 | 3 | 2.538 | yes | 2.538 | yes |
| 8 | 1 | 4096 | 3 | 2.586 | yes | 2.586 | yes |
| 8 | 1 | 16384 | 3 | 2.609 | yes | 2.609 | yes |
| 8 | 2 | 48 | 12 | 6.024 | yes | 5.163 | yes |
| 8 | 2 | 1024 | 12 | 9.978 | yes | 9.978 | yes |
| 8 | 2 | 4096 | 12 | 10.255 | yes | 10.255 | yes |
| 8 | 2 | 16384 | 12 | 10.389 | yes | 10.389 | yes |
| 16 | 1 | 48 | 4 | 2.071 | yes | 2.071 | yes |
| 16 | 1 | 1024 | 4 | 3.376 | yes | 3.376 | yes |
| 16 | 1 | 4096 | 4 | 3.444 | yes | 3.444 | yes |
| 16 | 1 | 16384 | 4 | 3.476 | yes | 3.476 | yes |
| 16 | 2 | 48 | 16 | 6.776 | yes | 5.808 | yes |
| 16 | 2 | 1024 | 16 | 13.228 | yes | 11.339 | yes |
| 16 | 2 | 4096 | 16 | 13.639 | yes | 11.690 | yes |
| 16 | 2 | 16384 | 16 | 13.833 | yes | 11.857 | yes |
| 32 | 1 | 48 | 5 | 2.353 | yes | 2.353 | yes |
| 32 | 1 | 1024 | 5 | 4.210 | yes | 3.408 | yes |
| 32 | 1 | 4096 | 5 | 4.300 | yes | 3.071 | yes |
| 32 | 1 | 16384 | 5 | 4.342 | yes | 3.102 | yes |
| 32 | 2 | 48 | 20 | 7.529 | yes | 6.454 | yes |
| 32 | 2 | 1024 | 20 | 16.148 | yes | 11.534 | yes |
| 32 | 2 | 4096 | 20 | 17.001 | yes | 12.144 | yes |
| 32 | 2 | 16384 | 20 | 17.271 | yes | 12.337 | yes |
| 64 | 1 | 48 | 6 | 2.259 | yes | 2.259 | yes |
| 64 | 1 | 1024 | 6 | 4.969 | yes | 4.022 | yes |
| 64 | 1 | 4096 | 6 | 5.154 | yes | 3.927 | yes |
| 64 | 1 | 16384 | 6 | 5.208 | yes | 4.712 | yes |
| 64 | 2 | 48 | 24 | 6.776 | yes | 5.808 | yes |
| 64 | 2 | 1024 | 24 | 18.086 | yes | 12.918 | yes |
| 64 | 2 | 4096 | 24 | 20.073 | yes | 14.338 | yes |
| 64 | 2 | 16384 | 24 | 20.702 | yes | 14.787 | yes |

Both invariants hold for every (palette, subpatch, payload_size) bucket above. The largest observed symbol_error_rate across the whole sweep is 0.4348 (palette=64, subpatch=2, payload_size=16384B, corruption=combined). Within the realistic corruption envelope this harness applies, decode_success_rate drops below 1.00 for at least one cell in this sweep (lowest observed: 0.00, at palette=8, subpatch=2, payload_size=48B, corruption=combined) -- unlike the original v0.1 4-palette/subpatch=1/48-byte sweep, where Reed-Solomon (nsym=32) fully absorbed every symbol error that same envelope introduced. See the full breakdown above for every cell where decode_success_rate < 1.00: this is the realistic corruption envelope actually biting at the larger palette/subpatch/payload_size combinations this sweep newly covers, not a measurement bug.

## Beyond the realistic envelope (diagnostic, single representative config)

To confirm decode failure is actually reachable by this harness (i.e. that high success rates above are a real headroom finding and not a bug that can never observe failure), the same style of trial was re-run under corruption well outside the 'realistic serving pipeline' envelope: 50% bilinear resize round-trip, JPEG q10, a 6px crop/pad, and their composition. This diagnostic suite runs at a single representative config -- subpatch=1, payload_size=48B, 5 trials/cell (the module defaults) -- across all 6 palettes; it is NOT swept across subpatch/payload_size the way the headline sweep above is, since its only purpose is to confirm the harness can observe decode failure at all.

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
| 32 | stress_resize_50pct | 0.0000 | 1.00 | 2.353 |
| 32 | stress_jpeg_q10 | 0.2550 | 0.00 | 0.000 |
| 32 | stress_crop_pad_6px | 0.0000 | 1.00 | 2.353 |
| 32 | stress_combined | 0.6687 | 0.00 | 0.000 |
| 64 | stress_resize_50pct | 0.0000 | 1.00 | 2.259 |
| 64 | stress_jpeg_q10 | 0.3063 | 0.00 | 0.000 |
| 64 | stress_crop_pad_6px | 0.0000 | 1.00 | 2.259 |
| 64 | stress_combined | 0.8859 | 0.00 | 0.000 |

Decode success drops well below 1.00 for at least one palette under this diagnostic stress suite (lowest observed: 0.00), confirming the channel does have a real breaking point -- it simply lies beyond the resize/JPEG/crop ranges a typical serving pipeline applies, consistent with the realistic-envelope sweep above.
