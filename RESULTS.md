# heliogram v0.1 -- CPU eval results

Synthetic, seed-deterministic payloads. Capacity sweep: palette in [2, 4, 8, 16, 32, 64, 128, 256], subpatch (k) in [1, 2], payload_size (bytes) in [48, 1024, 4096, 16384], x 8 corruptions (incl. 'clean'), 3 trials/cell, nsym=32, patch_size=14px. Reference decoder = decode_pixels (no model).

**Scope: this file characterizes the CODE/CHANNEL only.** Every number below comes from `decode_pixels`, the model-free reference decoder (pixel sampling + nearest-neighbor classification + Reed-Solomon, no VLM in the loop). Whether a fine-tuned VLM can realize this same capacity through its own vision encoder is Phase 2 and is not measured anywhere in this repo -- see the README's "Roadmap / Phase-2 boundary" section.

**Wall-clock note:** the full sweep below is 8 palettes x 2 subpatch values x 4 payload sizes x 8 corruptions = 512 cells; at the largest payload tier (16384B) each cell encodes/corrupts/decodes a multi-thousand-patch image, so trial count for this sweep was reduced to 3 (module default is 5) to bound wall-clock. The diagnostic stress suite below still runs at the module default 5 trials, at a single representative config (subpatch=1, payload_size=48B) -- see that section.

## Headline: three bars, and the actual benefit (token crossover)

This project tracks THREE bars, deliberately kept separate because they answer different questions -- conflating them is exactly the overclaiming this file exists to prevent:

- **Bar A -- beat base64 density, clean (6.0 bits/patch):** the real economic break-even for bits/patch alone (see Baselines below) -- the minimum for heliogram to be worth considering purely on density. Evaluated CLEAN only in the table below (see the 'beats 6 clean?' column); a config beating Bar A clean may or may not survive corruption -- the worst-corruption columns in the same row show that separately, and it is not folded into this bar.
- **Bar B -- Gate #1 comfort margin (8.0 bits/patch, clean AND worst-tested-corruption):** deliberately set above Bar A as a robustness margin before this project starts Phase 2 (see the README's Decision Gate). A config "clears the gate" only if its bits/patch is at or above this bar BOTH on a clean image AND in its single worst-performing tested corruption -- a config that only clears on average is not a robust win. **This is a conservative comfort margin, not the real economic bar** -- see Bar A and Bar C.
- **Bar C -- token crossover (the actual measured benefit claim):** does encoding a payload as a heliogram grid cost FEWER total patches (~1 token/patch for a self-hosted VLM) than base64-ing the same payload into text tokens (~1 token/char)? This is an ACCOUNTING comparison of token COUNT, not bits/patch density -- a config can win on Bar C while still failing Bar A, because RS/framing overhead amortizes differently for the two encodings as payload grows. See the dedicated "Token crossover" section below for the real numbers and the crossover payload size per palette.

**MANDATORY honesty caveat:** rows with `subpatch=1` are the VLM-meaningful regime -- one symbol per DATA patch, i.e. one symbol per (nominal) vision token, the only regime this project claims any real relevance to a downstream VLM. Rows with `subpatch>1` are a **PIXEL-DECODER GEOMETRIC CEILING ONLY**: `decode_pixels`/`extract_symbols` can read sub-patch cells trivially because they sample known, exact pixel coordinates off a grid whose size they are told in advance -- there is no perception involved. Whether a real ViT/VLM image encoder can resolve sub-patch structure at all is **unverified, and doubtful** (a k x k sub-cell grid inside one ViT patch may simply average out in that patch's embedding). Realizing it is Phase 2 work, gated on GPU access, and is **not a capability claim** made anywhere in this repo.

**Also mandatory, and specific to the largest palettes (visible here, in the headline area, on purpose):** `palette=128` and `palette=256` clean-decode exactly on this pixel decoder (see `tests/test_roundtrip.py`) but are MEASURED to FAIL decode under `jpeg_q70` in this very sweep (see the full breakdown below and the "Token crossover" section, which shows the clean-token-cheaper number and the corrupted-decode-failure number for the SAME cells side by side). The token-count benefit these two palettes unlock (Bar C) is therefore a property of the CLEAN channel only -- it is **not currently usable end to end** on this reference decoder, and realizing it under corruption is conditional on Phase 2 producing a reader that survives corruption at this palette size, which `decode_pixels` itself does not.

| palette | subpatch | payload (B) | ceiling k²·log2(P) | clean bits/patch | beats 6 clean? (Bar A) | clears 8 clean? | worst-corruption bits/patch | worst corruption | clears 8 corrupted? | clears gate (both, Bar B)? |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 1 | 0.544 | no | no | 0.544 | resize_3pct | no | no |
| 2 | 1 | 1024 | 1 | 0.853 | no | no | 0.853 | resize_3pct | no | no |
| 2 | 1 | 4096 | 1 | 0.865 | no | no | 0.865 | resize_3pct | no | no |
| 2 | 1 | 16384 | 1 | 0.871 | no | no | 0.871 | resize_3pct | no | no |
| 2 | 2 | 48 | 4 | 2.097 | no | no | 2.097 | resize_3pct | no | no |
| 2 | 2 | 1024 | 4 | 3.376 | no | no | 3.376 | resize_3pct | no | no |
| 2 | 2 | 4096 | 4 | 3.444 | no | no | 3.444 | resize_3pct | no | no |
| 2 | 2 | 16384 | 4 | 3.476 | no | no | 3.476 | resize_3pct | no | no |
| 4 | 1 | 48 | 2 | 1.070 | no | no | 1.070 | resize_3pct | no | no |
| 4 | 1 | 1024 | 2 | 1.698 | no | no | 1.698 | resize_3pct | no | no |
| 4 | 1 | 4096 | 2 | 1.727 | no | no | 1.727 | resize_3pct | no | no |
| 4 | 1 | 16384 | 2 | 1.741 | no | no | 1.741 | resize_3pct | no | no |
| 4 | 2 | 48 | 8 | 4.066 | no | no | 4.066 | resize_3pct | no | no |
| 4 | 2 | 1024 | 8 | 6.693 | yes | no | 6.693 | resize_3pct | no | no |
| 4 | 2 | 4096 | 8 | 6.859 | yes | no | 6.859 | resize_3pct | no | no |
| 4 | 2 | 16384 | 8 | 6.937 | yes | no | 6.937 | resize_3pct | no | no |
| 8 | 1 | 48 | 3 | 1.588 | no | no | 1.588 | resize_3pct | no | no |
| 8 | 1 | 1024 | 3 | 2.538 | no | no | 2.538 | resize_3pct | no | no |
| 8 | 1 | 4096 | 3 | 2.586 | no | no | 2.586 | resize_3pct | no | no |
| 8 | 1 | 16384 | 3 | 2.609 | no | no | 2.609 | resize_3pct | no | no |
| 8 | 2 | 48 | 12 | 6.024 | yes | no | 0.000 | combined | no | no |
| 8 | 2 | 1024 | 12 | 9.978 | yes | yes | 9.978 | resize_3pct | yes | **YES** |
| 8 | 2 | 4096 | 12 | 10.255 | yes | yes | 10.255 | resize_3pct | yes | **YES** |
| 8 | 2 | 16384 | 12 | 10.389 | yes | yes | 10.389 | resize_3pct | yes | **YES** |
| 16 | 1 | 48 | 4 | 2.071 | no | no | 2.071 | resize_3pct | no | no |
| 16 | 1 | 1024 | 4 | 3.376 | no | no | 3.376 | resize_3pct | no | no |
| 16 | 1 | 4096 | 4 | 3.444 | no | no | 3.444 | resize_3pct | no | no |
| 16 | 1 | 16384 | 4 | 3.476 | no | no | 3.476 | resize_3pct | no | no |
| 16 | 2 | 48 | 16 | 6.776 | yes | no | 0.000 | combined | no | no |
| 16 | 2 | 1024 | 16 | 13.228 | yes | yes | 0.000 | combined | no | no |
| 16 | 2 | 4096 | 16 | 13.639 | yes | yes | 0.000 | combined | no | no |
| 16 | 2 | 16384 | 16 | 13.833 | yes | yes | 0.000 | combined | no | no |
| 32 | 1 | 48 | 5 | 2.353 | no | no | 2.353 | resize_3pct | no | no |
| 32 | 1 | 1024 | 5 | 4.210 | no | no | 0.000 | jpeg_q70 | no | no |
| 32 | 1 | 4096 | 5 | 4.300 | no | no | 0.000 | jpeg_q70 | no | no |
| 32 | 1 | 16384 | 5 | 4.342 | no | no | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 48 | 20 | 7.529 | yes | no | 0.000 | combined | no | no |
| 32 | 2 | 1024 | 20 | 16.148 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 4096 | 20 | 17.001 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 16384 | 20 | 17.271 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 1 | 48 | 6 | 2.259 | no | no | 2.259 | resize_3pct | no | no |
| 64 | 1 | 1024 | 6 | 4.969 | no | no | 0.000 | jpeg_q70 | no | no |
| 64 | 1 | 4096 | 6 | 5.154 | no | no | 0.000 | combined | no | no |
| 64 | 1 | 16384 | 6 | 5.208 | no | no | 1.736 | jpeg_q70 | no | no |
| 64 | 2 | 48 | 24 | 6.776 | yes | no | 0.000 | combined | no | no |
| 64 | 2 | 1024 | 24 | 18.086 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 2 | 4096 | 24 | 20.073 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 2 | 16384 | 24 | 20.702 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 128 | 1 | 48 | 7 | 1.976 | no | no | 0.000 | jpeg_q70 | no | no |
| 128 | 1 | 1024 | 7 | 5.526 | no | no | 0.000 | jpeg_q70 | no | no |
| 128 | 1 | 4096 | 7 | 5.950 | no | no | 0.000 | jpeg_q85 | no | no |
| 128 | 1 | 16384 | 7 | 6.073 | yes | no | 0.000 | jpeg_q85 | no | no |
| 128 | 2 | 48 | 28 | 7.906 | yes | no | 0.000 | combined | no | no |
| 128 | 2 | 1024 | 28 | 18.086 | yes | yes | 0.000 | jpeg_q85 | no | no |
| 128 | 2 | 4096 | 28 | 22.325 | yes | yes | 0.000 | jpeg_q85 | no | no |
| 128 | 2 | 16384 | 28 | 23.889 | yes | yes | 0.000 | jpeg_q85 | no | no |
| 256 | 1 | 48 | 8 | 2.259 | no | no | 0.000 | jpeg_q70 | no | no |
| 256 | 1 | 1024 | 8 | 5.742 | no | no | 0.000 | jpeg_q85 | no | no |
| 256 | 1 | 4096 | 8 | 6.611 | yes | no | 0.000 | jpeg_q85 | no | no |
| 256 | 1 | 16384 | 8 | 6.895 | yes | no | 0.000 | jpeg_q85 | no | no |
| 256 | 2 | 48 | 32 | 9.035 | yes | yes | 0.000 | jpeg_q85 | no | no |
| 256 | 2 | 1024 | 32 | 18.373 | yes | yes | 0.000 | jpeg_q95 | no | no |
| 256 | 2 | 4096 | 32 | 23.195 | yes | yes | 0.000 | jpeg_q95 | no | no |
| 256 | 2 | 16384 | 32 | 26.554 | yes | yes | 0.000 | jpeg_q95 | no | no |

**Configs that clear the gate (both clean and worst-case corruption, Bar B):**

- palette=8, subpatch=2, payload_size=1024B -- clean 9.978 bits/patch, worst 9.978 bits/patch (worst corruption: `resize_3pct`)
- palette=8, subpatch=2, payload_size=4096B -- clean 10.255 bits/patch, worst 10.255 bits/patch (worst corruption: `resize_3pct`)
- palette=8, subpatch=2, payload_size=16384B -- clean 10.389 bits/patch, worst 10.389 bits/patch (worst corruption: `resize_3pct`)

**Configs that beat the base64 density bar clean (Bar A -- may or may not survive corruption; see the worst-corruption columns in the table above and the "Token crossover" section for whether that matters for tokens too):**

- palette=4, subpatch=2, payload_size=1024B -- clean 6.693 bits/patch (worst-corruption: 6.693, `resize_3pct`, does NOT clear Bar A under that corruption)
- palette=4, subpatch=2, payload_size=4096B -- clean 6.859 bits/patch (worst-corruption: 6.859, `resize_3pct`, does NOT clear Bar A under that corruption)
- palette=4, subpatch=2, payload_size=16384B -- clean 6.937 bits/patch (worst-corruption: 6.937, `resize_3pct`, does NOT clear Bar A under that corruption)
- palette=8, subpatch=2, payload_size=48B -- clean 6.024 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=8, subpatch=2, payload_size=1024B -- clean 9.978 bits/patch (worst-corruption: 9.978, `resize_3pct`, clears Bar A under that corruption)
- palette=8, subpatch=2, payload_size=4096B -- clean 10.255 bits/patch (worst-corruption: 10.255, `resize_3pct`, clears Bar A under that corruption)
- palette=8, subpatch=2, payload_size=16384B -- clean 10.389 bits/patch (worst-corruption: 10.389, `resize_3pct`, clears Bar A under that corruption)
- palette=16, subpatch=2, payload_size=48B -- clean 6.776 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=16, subpatch=2, payload_size=1024B -- clean 13.228 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=16, subpatch=2, payload_size=4096B -- clean 13.639 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=16, subpatch=2, payload_size=16384B -- clean 13.833 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=32, subpatch=2, payload_size=48B -- clean 7.529 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=32, subpatch=2, payload_size=1024B -- clean 16.148 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=32, subpatch=2, payload_size=4096B -- clean 17.001 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=32, subpatch=2, payload_size=16384B -- clean 17.271 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=64, subpatch=2, payload_size=48B -- clean 6.776 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=64, subpatch=2, payload_size=1024B -- clean 18.086 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=64, subpatch=2, payload_size=4096B -- clean 20.073 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=64, subpatch=2, payload_size=16384B -- clean 20.702 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=128, subpatch=1, payload_size=16384B -- clean 6.073 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=128, subpatch=2, payload_size=48B -- clean 7.906 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=128, subpatch=2, payload_size=1024B -- clean 18.086 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=128, subpatch=2, payload_size=4096B -- clean 22.325 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=128, subpatch=2, payload_size=16384B -- clean 23.889 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=1, payload_size=4096B -- clean 6.611 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=1, payload_size=16384B -- clean 6.895 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=2, payload_size=48B -- clean 9.035 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=2, payload_size=1024B -- clean 18.373 bits/patch (worst-corruption: 0.000, `jpeg_q95`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=2, payload_size=4096B -- clean 23.195 bits/patch (worst-corruption: 0.000, `jpeg_q95`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=2, payload_size=16384B -- clean 26.554 bits/patch (worst-corruption: 0.000, `jpeg_q95`, does NOT clear Bar A under that corruption)

**Verdict (derived from the tables above, not asserted independently):**

Every Gate #1 (Bar B) clearing config has `subpatch>1` -- the unverified pixel-decoder geometric ceiling regime. **No `subpatch=1` (VLM-meaningful) config clears Gate #1 at any tested payload size.** This is not just an unlucky corruption result: for `subpatch=1` the raw per-symbol ceiling is `log2(palette)`, which for the largest palette tested (256) is only 8 bits/patch -- already below the 8-bit Bar B *before* Reed-Solomon/calibration overhead is even subtracted. No amount of payload-size amortization can close that gap for `subpatch=1`; only the geometric `subpatch>1` regime can mathematically reach Bar B, and whether a real VLM can realize that regime is exactly the open question Phase 2 exists to answer. **Bar A tells a different story, though:** 30 config(s) beat the real economic bar clean (see the list above), including subpatch=1 configs -- see the "Token crossover" section below for what that means in tokens, and the mandatory P=128/256 corruption caveat above for what it does not yet mean.

## Baselines

- **base64 in text context:** ~6.0 bits/token (analytic: base64 alphabet size 64 -> log2(64)=6 bits/char; ~1 char/token for typical BPE tokenizers on base64 streams. Pass a real tokenizer for a measured value.)
- **Rendered text (geometric, model-free):** 2.13 chars/patch = 12.80 bits/patch typesetting a 48-byte payload (base64'd, 64 chars) into 30 patches of the same 14px grid unit. geometric/model-free: measures typeset packing density only, assumes perfect legibility. Real bits/patch for rendered text needs OCR accuracy from an un-fine-tuned VLM (Phase 2, out of scope here).

See "Token crossover" immediately below for the actual benefit claim (total token COUNT for a full payload, not bits/patch density) -- beating the bits/patch bar above is necessary but not sufficient for that; overhead amortization differs between the two encodings.

## Token crossover: the actual measured benefit

THE benefit claim this project can currently make: does encoding a payload as a heliogram grid cost fewer total patches (`total_patches`, the grid's width*height -- ~1 token/patch for a self-hosted VLM that tokenizes at the same patch grid) than base64-ing the same payload bytes into text tokens (`base64_token_est` = ceil(payload/3)*4 base64 characters, ~1 token/char for typical BPE tokenizers -- see Baselines above)? `token_ratio = total_patches / base64_token_est`; `token_ratio < 1.0` means heliogram is CHEAPER on token count for that payload -- an accounting fact about total context cost for the WHOLE payload, distinct from the bits/patch DENSITY bars in the Headline section (a config can win here while losing on bits/patch, because the two encodings amortize fixed overhead differently as payload grows: heliogram pays a calibration row + per-RS-chunk parity once per image, base64 pays none of that but never exceeds 6 bits/char either).

**HONESTY (mandatory, same rule as everywhere else in this file):** `token_ratio` and `heliogram_cheaper` are computed from `total_patches` alone -- a property of grid geometry -- regardless of whether `decode_success_rate` for that same cell is 1.0 or 0.0. Token-cheaper is an accounting fact about COUNT, not a claim that any reader can actually recover the payload from that many patches. The table below shows both numbers for every bucket side by side, on purpose: for `palette` in {128, 256}, `token_ratio` can drop below 1.0 at a payload size where `jpeg_q70 decode success` is still 0.00 in this same sweep -- so the token-count benefit these two palettes unlock is currently a CLEAN-CHANNEL-ONLY number. Usability under real corruption is exactly the Phase-2 reader-robustness bet described in the Headline section above, not something this table settles.

| palette | subpatch | payload (B) | total_patches | base64_token_est | token_ratio | cheaper on tokens? | clean decode success | jpeg_q70 decode success |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 729 | 64 | 11.391 | no | 1.00 | 1.00 |
| 2 | 1 | 1024 | 9702 | 1368 | 7.092 | no | 1.00 | 1.00 |
| 2 | 1 | 4096 | 38025 | 5464 | 6.959 | no | 1.00 | 1.00 |
| 2 | 1 | 16384 | 150544 | 21848 | 6.891 | no | 1.00 | 1.00 |
| 2 | 2 | 48 | 196 | 64 | 3.062 | no | 1.00 | 1.00 |
| 2 | 2 | 1024 | 2450 | 1368 | 1.791 | no | 1.00 | 1.00 |
| 2 | 2 | 4096 | 9604 | 5464 | 1.758 | no | 1.00 | 1.00 |
| 2 | 2 | 16384 | 37830 | 21848 | 1.732 | no | 1.00 | 1.00 |
| 4 | 1 | 48 | 361 | 64 | 5.641 | no | 1.00 | 1.00 |
| 4 | 1 | 1024 | 4830 | 1368 | 3.531 | no | 1.00 | 1.00 |
| 4 | 1 | 4096 | 19044 | 5464 | 3.485 | no | 1.00 | 1.00 |
| 4 | 1 | 16384 | 75350 | 21848 | 3.449 | no | 1.00 | 1.00 |
| 4 | 2 | 48 | 100 | 64 | 1.562 | no | 1.00 | 1.00 |
| 4 | 2 | 1024 | 1225 | 1368 | 0.895 | **YES** | 1.00 | 1.00 |
| 4 | 2 | 4096 | 4830 | 5464 | 0.884 | **YES** | 1.00 | 1.00 |
| 4 | 2 | 16384 | 18906 | 21848 | 0.865 | **YES** | 1.00 | 1.00 |
| 8 | 1 | 48 | 256 | 64 | 4.000 | no | 1.00 | 1.00 |
| 8 | 1 | 1024 | 3249 | 1368 | 2.375 | no | 1.00 | 1.00 |
| 8 | 1 | 4096 | 12769 | 5464 | 2.337 | no | 1.00 | 1.00 |
| 8 | 1 | 16384 | 50400 | 21848 | 2.307 | no | 1.00 | 1.00 |
| 8 | 2 | 48 | 72 | 64 | 1.125 | no | 1.00 | 1.00 |
| 8 | 2 | 1024 | 841 | 1368 | 0.615 | **YES** | 1.00 | 1.00 |
| 8 | 2 | 4096 | 3249 | 5464 | 0.595 | **YES** | 1.00 | 1.00 |
| 8 | 2 | 16384 | 12656 | 21848 | 0.579 | **YES** | 1.00 | 1.00 |
| 16 | 1 | 48 | 192 | 64 | 3.000 | no | 1.00 | 1.00 |
| 16 | 1 | 1024 | 2450 | 1368 | 1.791 | no | 1.00 | 1.00 |
| 16 | 1 | 4096 | 9604 | 5464 | 1.758 | no | 1.00 | 1.00 |
| 16 | 1 | 16384 | 37830 | 21848 | 1.732 | no | 1.00 | 1.00 |
| 16 | 2 | 48 | 64 | 64 | 1.000 | no | 1.00 | 1.00 |
| 16 | 2 | 1024 | 625 | 1368 | 0.457 | **YES** | 1.00 | 1.00 |
| 16 | 2 | 4096 | 2450 | 5464 | 0.448 | **YES** | 1.00 | 1.00 |
| 16 | 2 | 16384 | 9506 | 21848 | 0.435 | **YES** | 1.00 | 1.00 |
| 32 | 1 | 48 | 192 | 64 | 3.000 | no | 1.00 | 1.00 |
| 32 | 1 | 1024 | 1980 | 1368 | 1.447 | no | 1.00 | 0.00 |
| 32 | 1 | 4096 | 7656 | 5464 | 1.401 | no | 1.00 | 0.00 |
| 32 | 1 | 16384 | 30276 | 21848 | 1.386 | no | 1.00 | 0.00 |
| 32 | 2 | 48 | 96 | 64 | 1.500 | no | 1.00 | 1.00 |
| 32 | 2 | 1024 | 512 | 1368 | 0.374 | **YES** | 1.00 | 0.00 |
| 32 | 2 | 4096 | 1936 | 5464 | 0.354 | **YES** | 1.00 | 0.00 |
| 32 | 2 | 16384 | 7656 | 21848 | 0.350 | **YES** | 1.00 | 0.00 |
| 64 | 1 | 48 | 192 | 64 | 3.000 | no | 1.00 | 1.00 |
| 64 | 1 | 1024 | 1664 | 1368 | 1.216 | no | 1.00 | 0.00 |
| 64 | 1 | 4096 | 6400 | 5464 | 1.171 | no | 1.00 | 0.33 |
| 64 | 1 | 16384 | 25281 | 21848 | 1.157 | no | 1.00 | 0.33 |
| 64 | 2 | 48 | 128 | 64 | 2.000 | no | 1.00 | 1.00 |
| 64 | 2 | 1024 | 512 | 1368 | 0.374 | **YES** | 1.00 | 0.00 |
| 64 | 2 | 4096 | 1664 | 5464 | 0.305 | **YES** | 1.00 | 0.00 |
| 64 | 2 | 16384 | 6400 | 21848 | 0.293 | **YES** | 1.00 | 0.00 |
| 128 | 1 | 48 | 256 | 64 | 4.000 | no | 1.00 | 0.00 |
| 128 | 1 | 1024 | 1536 | 1368 | 1.123 | no | 1.00 | 0.00 |
| 128 | 1 | 4096 | 5632 | 5464 | 1.031 | no | 1.00 | 0.00 |
| 128 | 1 | 16384 | 21609 | 21848 | 0.989 | **YES** | 1.00 | 0.00 |
| 128 | 2 | 48 | 256 | 64 | 4.000 | no | 1.00 | 0.33 |
| 128 | 2 | 1024 | 512 | 1368 | 0.374 | **YES** | 1.00 | 0.00 |
| 128 | 2 | 4096 | 1536 | 5464 | 0.281 | **YES** | 1.00 | 0.00 |
| 128 | 2 | 16384 | 5504 | 21848 | 0.252 | **YES** | 1.00 | 0.00 |
| 256 | 1 | 48 | 512 | 64 | 8.000 | no | 1.00 | 0.00 |
| 256 | 1 | 1024 | 1536 | 1368 | 1.123 | no | 1.00 | 0.00 |
| 256 | 1 | 4096 | 5120 | 5464 | 0.937 | **YES** | 1.00 | 0.00 |
| 256 | 1 | 16384 | 19200 | 21848 | 0.879 | **YES** | 1.00 | 0.00 |
| 256 | 2 | 48 | 512 | 64 | 8.000 | no | 1.00 | 0.00 |
| 256 | 2 | 1024 | 768 | 1368 | 0.561 | **YES** | 1.00 | 0.00 |
| 256 | 2 | 4096 | 1536 | 5464 | 0.281 | **YES** | 1.00 | 0.00 |
| 256 | 2 | 16384 | 5120 | 21848 | 0.234 | **YES** | 1.00 | 0.00 |

### Crossover payload size per (palette, subpatch)

For each palette, the payload size (within the swept range [48, 1024, 4096, 16384]B) where `token_ratio` first drops below 1.0, linearly interpolated between the two nearest swept sizes when the crossing happens between them (see `_crossover_payload_size`). 'no crossover in tested range' means the ratio never dropped below 1.0 at any size swept here -- NOT a claim that it never will at a larger, untested payload size.

**subpatch=1 (VLM-meaningful: one symbol per patch):**

- palette=2: no crossover in tested range up to 16384B (lowest token_ratio observed: 6.891 at 16384B)
- palette=4: no crossover in tested range up to 16384B (lowest token_ratio observed: 3.449 at 16384B)
- palette=8: no crossover in tested range up to 16384B (lowest token_ratio observed: 2.307 at 16384B)
- palette=16: no crossover in tested range up to 16384B (lowest token_ratio observed: 1.732 at 16384B)
- palette=32: no crossover in tested range up to 16384B (lowest token_ratio observed: 1.386 at 16384B)
- palette=64: no crossover in tested range up to 16384B (lowest token_ratio observed: 1.157 at 16384B)
- palette=128: crosses ~13159B (lowest token_ratio observed in this sweep: 0.989 at 16384B)
- palette=256: crosses ~3055B (lowest token_ratio observed in this sweep: 0.879 at 16384B)

**subpatch=2 (PIXEL-DECODER GEOMETRIC CEILING ONLY -- not a VLM capability claim, see caveat above):**

- palette=2: no crossover in tested range up to 16384B (lowest token_ratio observed: 1.732 at 16384B)
- palette=4: crosses ~871B (lowest token_ratio observed in this sweep: 0.865 at 16384B)
- palette=8: crosses ~287B (lowest token_ratio observed in this sweep: 0.579 at 16384B)
- palette=16: crosses ~48B (lowest token_ratio observed in this sweep: 0.435 at 16384B)
- palette=32: crosses ~481B (lowest token_ratio observed in this sweep: 0.350 at 16384B)
- palette=64: crosses ~648B (lowest token_ratio observed in this sweep: 0.293 at 16384B)
- palette=128: crosses ~856B (lowest token_ratio observed in this sweep: 0.252 at 16384B)
- palette=256: crosses ~966B (lowest token_ratio observed in this sweep: 0.234 at 16384B)

### Token-crossover verdict

At `subpatch=1` (the only VLM-meaningful regime), the following palette(s) cross below base64 token count within the swept payload range: palette=128 at ~13159B, palette=256 at ~3055B. This is the project's actual, currently-measured benefit claim: for a large enough payload, encoding it as a heliogram grid costs fewer total patches than base64-ing it into text tokens, and (per Bar A in the Headline section) does so at a bits/patch density that also beats plain base64 text, and is bit-exact on a successful decode (Reed-Solomon verified). **This is a clean-channel, token-accounting result only** -- see the mandatory P=128/256 corruption caveat in the Headline section above: the pixel decoder cannot currently realize this benefit end to end under `jpeg_q70` at these same palettes. The open question is purely whether a fine-tuned VLM reader can, which is Phase 2 and is not measured here.

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
| 128 | 7 | 7 | 1.976 | 1.412 | 5.526 | 3.684 | 5.950 | 3.400 | 6.073 | 3.470 |
| 256 | 8 | 8 | 2.259 | 1.506 | 5.742 | 3.281 | 6.611 | 3.463 | 6.895 | 3.940 |

### subpatch=2 (PIXEL-DECODER GEOMETRIC CEILING ONLY -- not a VLM capability claim, see caveat above)

| Palette | bits/sym | ceiling | 48B clean | 48B corr(mean) | 1024B clean | 1024B corr(mean) | 4096B clean | 4096B corr(mean) | 16384B clean | 16384B corr(mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 4 | 2.097 | 2.097 | 3.376 | 3.376 | 3.444 | 3.444 | 3.476 | 3.476 |
| 4 | 2 | 8 | 4.066 | 4.066 | 6.693 | 6.693 | 6.859 | 6.859 | 6.937 | 6.937 |
| 8 | 3 | 12 | 6.024 | 5.163 | 9.978 | 9.978 | 10.255 | 10.255 | 10.389 | 10.389 |
| 16 | 4 | 16 | 6.776 | 5.808 | 13.228 | 11.339 | 13.639 | 11.690 | 13.833 | 11.857 |
| 32 | 5 | 20 | 7.529 | 6.454 | 16.148 | 11.534 | 17.001 | 12.144 | 17.271 | 12.337 |
| 64 | 6 | 24 | 6.776 | 5.808 | 18.086 | 12.918 | 20.073 | 14.338 | 20.702 | 14.787 |
| 128 | 7 | 28 | 7.906 | 6.024 | 18.086 | 10.335 | 22.325 | 12.757 | 23.889 | 13.651 |
| 256 | 8 | 32 | 9.035 | 5.163 | 18.373 | 7.874 | 23.195 | 9.941 | 26.554 | 11.380 |

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
| 128 | 1 | 48 | 7 | clean | 0.0000 | 1.00 | 1.976 |
| 128 | 1 | 48 | 7 | resize_3pct | 0.0000 | 1.00 | 1.976 |
| 128 | 1 | 48 | 7 | resize_5pct | 0.0000 | 1.00 | 1.976 |
| 128 | 1 | 48 | 7 | jpeg_q95 | 0.0000 | 1.00 | 1.976 |
| 128 | 1 | 48 | 7 | jpeg_q85 | 0.0234 | 0.67 | 1.318 |
| 128 | 1 | 48 | 7 | jpeg_q70 | 0.0469 | 0.00 | 0.000 |
| 128 | 1 | 48 | 7 | crop_pad_2px | 0.0000 | 1.00 | 1.976 |
| 128 | 1 | 48 | 7 | combined | 0.0651 | 0.33 | 0.659 |
| 128 | 1 | 1024 | 7 | clean | 0.0000 | 1.00 | 5.526 |
| 128 | 1 | 1024 | 7 | resize_3pct | 0.0000 | 1.00 | 5.526 |
| 128 | 1 | 1024 | 7 | resize_5pct | 0.0000 | 1.00 | 5.526 |
| 128 | 1 | 1024 | 7 | jpeg_q95 | 0.0000 | 1.00 | 5.526 |
| 128 | 1 | 1024 | 7 | jpeg_q85 | 0.0365 | 0.67 | 3.684 |
| 128 | 1 | 1024 | 7 | jpeg_q70 | 0.1823 | 0.00 | 0.000 |
| 128 | 1 | 1024 | 7 | crop_pad_2px | 0.0000 | 1.00 | 5.526 |
| 128 | 1 | 1024 | 7 | combined | 0.1574 | 0.00 | 0.000 |
| 128 | 1 | 4096 | 7 | clean | 0.0000 | 1.00 | 5.950 |
| 128 | 1 | 4096 | 7 | resize_3pct | 0.0000 | 1.00 | 5.950 |
| 128 | 1 | 4096 | 7 | resize_5pct | 0.0000 | 1.00 | 5.950 |
| 128 | 1 | 4096 | 7 | jpeg_q95 | 0.0001 | 1.00 | 5.950 |
| 128 | 1 | 4096 | 7 | jpeg_q85 | 0.0380 | 0.00 | 0.000 |
| 128 | 1 | 4096 | 7 | jpeg_q70 | 0.1840 | 0.00 | 0.000 |
| 128 | 1 | 4096 | 7 | crop_pad_2px | 0.0000 | 1.00 | 5.950 |
| 128 | 1 | 4096 | 7 | combined | 0.1648 | 0.00 | 0.000 |
| 128 | 1 | 16384 | 7 | clean | 0.0000 | 1.00 | 6.073 |
| 128 | 1 | 16384 | 7 | resize_3pct | 0.0000 | 1.00 | 6.073 |
| 128 | 1 | 16384 | 7 | resize_5pct | 0.0000 | 1.00 | 6.073 |
| 128 | 1 | 16384 | 7 | jpeg_q95 | 0.0000 | 1.00 | 6.073 |
| 128 | 1 | 16384 | 7 | jpeg_q85 | 0.0391 | 0.00 | 0.000 |
| 128 | 1 | 16384 | 7 | jpeg_q70 | 0.1822 | 0.00 | 0.000 |
| 128 | 1 | 16384 | 7 | crop_pad_2px | 0.0000 | 1.00 | 6.073 |
| 128 | 1 | 16384 | 7 | combined | 0.1551 | 0.00 | 0.000 |
| 128 | 2 | 48 | 7 | clean | 0.0000 | 1.00 | 7.906 |
| 128 | 2 | 48 | 7 | resize_3pct | 0.0000 | 1.00 | 7.906 |
| 128 | 2 | 48 | 7 | resize_5pct | 0.0000 | 1.00 | 7.906 |
| 128 | 2 | 48 | 7 | jpeg_q95 | 0.0000 | 1.00 | 7.906 |
| 128 | 2 | 48 | 7 | jpeg_q85 | 0.0059 | 1.00 | 7.906 |
| 128 | 2 | 48 | 7 | jpeg_q70 | 0.0345 | 0.33 | 2.635 |
| 128 | 2 | 48 | 7 | crop_pad_2px | 0.0000 | 1.00 | 7.906 |
| 128 | 2 | 48 | 7 | combined | 0.3783 | 0.00 | 0.000 |
| 128 | 2 | 1024 | 7 | clean | 0.0000 | 1.00 | 18.086 |
| 128 | 2 | 1024 | 7 | resize_3pct | 0.0000 | 1.00 | 18.086 |
| 128 | 2 | 1024 | 7 | resize_5pct | 0.0000 | 1.00 | 18.086 |
| 128 | 2 | 1024 | 7 | jpeg_q95 | 0.0002 | 1.00 | 18.086 |
| 128 | 2 | 1024 | 7 | jpeg_q85 | 0.0940 | 0.00 | 0.000 |
| 128 | 2 | 1024 | 7 | jpeg_q70 | 0.2971 | 0.00 | 0.000 |
| 128 | 2 | 1024 | 7 | crop_pad_2px | 0.0000 | 1.00 | 18.086 |
| 128 | 2 | 1024 | 7 | combined | 0.6447 | 0.00 | 0.000 |
| 128 | 2 | 4096 | 7 | clean | 0.0000 | 1.00 | 22.325 |
| 128 | 2 | 4096 | 7 | resize_3pct | 0.0000 | 1.00 | 22.325 |
| 128 | 2 | 4096 | 7 | resize_5pct | 0.0000 | 1.00 | 22.325 |
| 128 | 2 | 4096 | 7 | jpeg_q95 | 0.0002 | 1.00 | 22.325 |
| 128 | 2 | 4096 | 7 | jpeg_q85 | 0.1021 | 0.00 | 0.000 |
| 128 | 2 | 4096 | 7 | jpeg_q70 | 0.3166 | 0.00 | 0.000 |
| 128 | 2 | 4096 | 7 | crop_pad_2px | 0.0000 | 1.00 | 22.325 |
| 128 | 2 | 4096 | 7 | combined | 0.6636 | 0.00 | 0.000 |
| 128 | 2 | 16384 | 7 | clean | 0.0000 | 1.00 | 23.889 |
| 128 | 2 | 16384 | 7 | resize_3pct | 0.0000 | 1.00 | 23.889 |
| 128 | 2 | 16384 | 7 | resize_5pct | 0.0000 | 1.00 | 23.889 |
| 128 | 2 | 16384 | 7 | jpeg_q95 | 0.0002 | 1.00 | 23.889 |
| 128 | 2 | 16384 | 7 | jpeg_q85 | 0.1084 | 0.00 | 0.000 |
| 128 | 2 | 16384 | 7 | jpeg_q70 | 0.3318 | 0.00 | 0.000 |
| 128 | 2 | 16384 | 7 | crop_pad_2px | 0.0000 | 1.00 | 23.889 |
| 128 | 2 | 16384 | 7 | combined | 0.6718 | 0.00 | 0.000 |
| 256 | 1 | 48 | 8 | clean | 0.0000 | 1.00 | 2.259 |
| 256 | 1 | 48 | 8 | resize_3pct | 0.0000 | 1.00 | 2.259 |
| 256 | 1 | 48 | 8 | resize_5pct | 0.0000 | 1.00 | 2.259 |
| 256 | 1 | 48 | 8 | jpeg_q95 | 0.0000 | 1.00 | 2.259 |
| 256 | 1 | 48 | 8 | jpeg_q85 | 0.0443 | 0.67 | 1.506 |
| 256 | 1 | 48 | 8 | jpeg_q70 | 0.1081 | 0.00 | 0.000 |
| 256 | 1 | 48 | 8 | crop_pad_2px | 0.0000 | 1.00 | 2.259 |
| 256 | 1 | 48 | 8 | combined | 0.0964 | 0.00 | 0.000 |
| 256 | 1 | 1024 | 8 | clean | 0.0000 | 1.00 | 5.742 |
| 256 | 1 | 1024 | 8 | resize_3pct | 0.0000 | 1.00 | 5.742 |
| 256 | 1 | 1024 | 8 | resize_5pct | 0.0000 | 1.00 | 5.742 |
| 256 | 1 | 1024 | 8 | jpeg_q95 | 0.0260 | 1.00 | 5.742 |
| 256 | 1 | 1024 | 8 | jpeg_q85 | 0.2896 | 0.00 | 0.000 |
| 256 | 1 | 1024 | 8 | jpeg_q70 | 0.4833 | 0.00 | 0.000 |
| 256 | 1 | 1024 | 8 | crop_pad_2px | 0.0000 | 1.00 | 5.742 |
| 256 | 1 | 1024 | 8 | combined | 0.4409 | 0.00 | 0.000 |
| 256 | 1 | 4096 | 8 | clean | 0.0000 | 1.00 | 6.611 |
| 256 | 1 | 4096 | 8 | resize_3pct | 0.0000 | 1.00 | 6.611 |
| 256 | 1 | 4096 | 8 | resize_5pct | 0.0000 | 1.00 | 6.611 |
| 256 | 1 | 4096 | 8 | jpeg_q95 | 0.0232 | 0.67 | 4.407 |
| 256 | 1 | 4096 | 8 | jpeg_q85 | 0.2844 | 0.00 | 0.000 |
| 256 | 1 | 4096 | 8 | jpeg_q70 | 0.4824 | 0.00 | 0.000 |
| 256 | 1 | 4096 | 8 | crop_pad_2px | 0.0000 | 1.00 | 6.611 |
| 256 | 1 | 4096 | 8 | combined | 0.4624 | 0.00 | 0.000 |
| 256 | 1 | 16384 | 8 | clean | 0.0000 | 1.00 | 6.895 |
| 256 | 1 | 16384 | 8 | resize_3pct | 0.0000 | 1.00 | 6.895 |
| 256 | 1 | 16384 | 8 | resize_5pct | 0.0000 | 1.00 | 6.895 |
| 256 | 1 | 16384 | 8 | jpeg_q95 | 0.0240 | 1.00 | 6.895 |
| 256 | 1 | 16384 | 8 | jpeg_q85 | 0.2943 | 0.00 | 0.000 |
| 256 | 1 | 16384 | 8 | jpeg_q70 | 0.4922 | 0.00 | 0.000 |
| 256 | 1 | 16384 | 8 | crop_pad_2px | 0.0000 | 1.00 | 6.895 |
| 256 | 1 | 16384 | 8 | combined | 0.4661 | 0.00 | 0.000 |
| 256 | 2 | 48 | 8 | clean | 0.0000 | 1.00 | 9.035 |
| 256 | 2 | 48 | 8 | resize_3pct | 0.0000 | 1.00 | 9.035 |
| 256 | 2 | 48 | 8 | resize_5pct | 0.0000 | 1.00 | 9.035 |
| 256 | 2 | 48 | 8 | jpeg_q95 | 0.0033 | 1.00 | 9.035 |
| 256 | 2 | 48 | 8 | jpeg_q85 | 0.0234 | 0.00 | 0.000 |
| 256 | 2 | 48 | 8 | jpeg_q70 | 0.0407 | 0.00 | 0.000 |
| 256 | 2 | 48 | 8 | crop_pad_2px | 0.0000 | 1.00 | 9.035 |
| 256 | 2 | 48 | 8 | combined | 0.5166 | 0.00 | 0.000 |
| 256 | 2 | 1024 | 8 | clean | 0.0000 | 1.00 | 18.373 |
| 256 | 2 | 1024 | 8 | resize_3pct | 0.0000 | 1.00 | 18.373 |
| 256 | 2 | 1024 | 8 | resize_5pct | 0.0000 | 1.00 | 18.373 |
| 256 | 2 | 1024 | 8 | jpeg_q95 | 0.0623 | 0.00 | 0.000 |
| 256 | 2 | 1024 | 8 | jpeg_q85 | 0.2607 | 0.00 | 0.000 |
| 256 | 2 | 1024 | 8 | jpeg_q70 | 0.5980 | 0.00 | 0.000 |
| 256 | 2 | 1024 | 8 | crop_pad_2px | 0.0000 | 1.00 | 18.373 |
| 256 | 2 | 1024 | 8 | combined | 0.6678 | 0.00 | 0.000 |
| 256 | 2 | 4096 | 8 | clean | 0.0000 | 1.00 | 23.195 |
| 256 | 2 | 4096 | 8 | resize_3pct | 0.0000 | 1.00 | 23.195 |
| 256 | 2 | 4096 | 8 | resize_5pct | 0.0000 | 1.00 | 23.195 |
| 256 | 2 | 4096 | 8 | jpeg_q95 | 0.0675 | 0.00 | 0.000 |
| 256 | 2 | 4096 | 8 | jpeg_q85 | 0.3824 | 0.00 | 0.000 |
| 256 | 2 | 4096 | 8 | jpeg_q70 | 0.6100 | 0.00 | 0.000 |
| 256 | 2 | 4096 | 8 | crop_pad_2px | 0.0000 | 1.00 | 23.195 |
| 256 | 2 | 4096 | 8 | combined | 0.8104 | 0.00 | 0.000 |
| 256 | 2 | 16384 | 8 | clean | 0.0000 | 1.00 | 26.554 |
| 256 | 2 | 16384 | 8 | resize_3pct | 0.0000 | 1.00 | 26.554 |
| 256 | 2 | 16384 | 8 | resize_5pct | 0.0000 | 1.00 | 26.554 |
| 256 | 2 | 16384 | 8 | jpeg_q95 | 0.0730 | 0.00 | 0.000 |
| 256 | 2 | 16384 | 8 | jpeg_q85 | 0.4157 | 0.00 | 0.000 |
| 256 | 2 | 16384 | 8 | jpeg_q70 | 0.6255 | 0.00 | 0.000 |
| 256 | 2 | 16384 | 8 | crop_pad_2px | 0.0000 | 1.00 | 26.554 |
| 256 | 2 | 16384 | 8 | combined | 0.8090 | 0.00 | 0.000 |

## Self-consistency checks

Three invariants must hold if these numbers mean what they claim to mean: (1) bits/patch can never exceed `subpatch²·log2(palette)` -- the raw per-DATA-PATCH density for a subpatch x subpatch grid of symbols per patch, before calibration-row and Reed-Solomon overhead are subtracted (this generalizes the pre-Slice-B `<= log2(palette)` check, which is the `subpatch=1` case where `subpatch²=1`); (2) mean corrupted bits/patch can never exceed clean bits/patch for the same (palette, subpatch, payload_size), since corruption only ever removes information relative to the uncorrupted image; (3) [token crossover] every row's `base64_token_est` must equal `ceil(payload_size/3)*4` exactly and `token_ratio` must equal `total_patches/base64_token_est` exactly, independently recomputed here rather than just re-displaying the harness's own stored values -- if either drifts, the Token crossover section's numbers are wrong.

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
| 128 | 1 | 48 | 7 | 1.976 | yes | 1.412 | yes |
| 128 | 1 | 1024 | 7 | 5.526 | yes | 3.684 | yes |
| 128 | 1 | 4096 | 7 | 5.950 | yes | 3.400 | yes |
| 128 | 1 | 16384 | 7 | 6.073 | yes | 3.470 | yes |
| 128 | 2 | 48 | 28 | 7.906 | yes | 6.024 | yes |
| 128 | 2 | 1024 | 28 | 18.086 | yes | 10.335 | yes |
| 128 | 2 | 4096 | 28 | 22.325 | yes | 12.757 | yes |
| 128 | 2 | 16384 | 28 | 23.889 | yes | 13.651 | yes |
| 256 | 1 | 48 | 8 | 2.259 | yes | 1.506 | yes |
| 256 | 1 | 1024 | 8 | 5.742 | yes | 3.281 | yes |
| 256 | 1 | 4096 | 8 | 6.611 | yes | 3.463 | yes |
| 256 | 1 | 16384 | 8 | 6.895 | yes | 3.940 | yes |
| 256 | 2 | 48 | 32 | 9.035 | yes | 5.163 | yes |
| 256 | 2 | 1024 | 32 | 18.373 | yes | 7.874 | yes |
| 256 | 2 | 4096 | 32 | 23.195 | yes | 9.941 | yes |
| 256 | 2 | 16384 | 32 | 26.554 | yes | 11.380 | yes |

Invariants (1) and (2) hold for every (palette, subpatch, payload_size) bucket above. Invariant (3) [token crossover] holds for every one of the 512 rows in this sweep: base64_token_est and token_ratio were independently recomputed from payload_size/total_patches for every row and matched the harness's own stored values exactly. The largest observed symbol_error_rate across the whole sweep is 0.8104 (palette=256, subpatch=2, payload_size=4096B, corruption=combined). Within the realistic corruption envelope this harness applies, decode_success_rate drops below 1.00 for at least one cell in this sweep (lowest observed: 0.00, at palette=8, subpatch=2, payload_size=48B, corruption=combined) -- unlike the original v0.1 4-palette/subpatch=1/48-byte sweep, where Reed-Solomon (nsym=32) fully absorbed every symbol error that same envelope introduced. See the full breakdown above for every cell where decode_success_rate < 1.00: this is the realistic corruption envelope actually biting at the larger palette/subpatch/payload_size combinations this sweep newly covers, not a measurement bug.

## Beyond the realistic envelope (diagnostic, single representative config)

To confirm decode failure is actually reachable by this harness (i.e. that high success rates above are a real headroom finding and not a bug that can never observe failure), the same style of trial was re-run under corruption well outside the 'realistic serving pipeline' envelope: 50% bilinear resize round-trip, JPEG q10, a 6px crop/pad, and their composition. This diagnostic suite runs at a single representative config -- subpatch=1, payload_size=48B, 5 trials/cell (the module defaults) -- across all 8 palettes; it is NOT swept across subpatch/payload_size the way the headline sweep above is, since its only purpose is to confirm the harness can observe decode failure at all.

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
| 128 | stress_resize_50pct | 0.0000 | 1.00 | 1.976 |
| 128 | stress_jpeg_q10 | 0.5734 | 0.00 | 0.000 |
| 128 | stress_crop_pad_6px | 0.0000 | 1.00 | 1.976 |
| 128 | stress_combined | 0.9484 | 0.00 | 0.000 |
| 256 | stress_resize_50pct | 0.0000 | 1.00 | 2.259 |
| 256 | stress_jpeg_q10 | 0.6539 | 0.00 | 0.000 |
| 256 | stress_crop_pad_6px | 0.0000 | 1.00 | 2.259 |
| 256 | stress_combined | 0.9609 | 0.00 | 0.000 |

Decode success drops well below 1.00 for at least one palette under this diagnostic stress suite (lowest observed: 0.00), confirming the channel does have a real breaking point -- it simply lies beyond the resize/JPEG/crop ranges a typical serving pipeline applies, consistent with the realistic-envelope sweep above.
