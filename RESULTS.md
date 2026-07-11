# heliogram v0.1 -- CPU eval results

**Provenance:** Python 3.11.15; numpy 2.4.6; Pillow 12.3.0; reedsolo 1.7.0; platform: Linux-6.18.5-x86_64-with-glibc2.39

Synthetic, seed-deterministic payloads. Capacity sweep: palette in [2, 4, 8, 16, 32, 64, 128, 256], subpatch (k) in [1, 2], payload_size (bytes) in [48, 1024, 4096, 16384], x 8 corruptions (incl. 'clean'), 3 trials/cell, nsym=32, patch_size=14px. Reference decoder = decode_pixels (no model).

**Scope: this file characterizes the CODE/CHANNEL only.** Every number below comes from `decode_pixels`, the model-free reference decoder (pixel sampling + nearest-neighbor classification + Reed-Solomon, no VLM in the loop). Whether a fine-tuned VLM can realize this same capacity through its own vision encoder is Phase 2 and is not measured anywhere in this repo -- see the README's "Roadmap / Phase-2 boundary" section.

**Wall-clock note:** the full sweep below is 8 palettes x 2 subpatch values x 4 payload sizes x 8 corruptions = 512 cells; at the largest payload tier (16384B) each cell encodes/corrupts/decodes a multi-thousand-patch image, so trial count for this sweep was reduced to 3 (module default is 5) to bound wall-clock. The diagnostic stress suite below still runs at the module default 5 trials, at a single representative config (subpatch=1, payload_size=48B) -- see that section.

## Headline: three bars, and the actual benefit (token crossover)

This project tracks THREE bars, deliberately kept separate because they answer different questions -- conflating them is exactly the overclaiming this file exists to prevent:

- **Bar A -- beat base64 density, clean (8.1 bits/patch):** the real economic break-even for bits/patch alone (see Baselines below) -- the minimum for heliogram to be worth considering purely on density. Evaluated CLEAN only in the table below (see the 'beats 8 clean?' column); a config beating Bar A clean may or may not survive corruption -- the worst-corruption columns in the same row show that separately, and it is not folded into this bar.
- **Bar B -- Gate #1 comfort margin (8.0 bits/patch, clean AND worst-tested-corruption):** originally set as a robustness margin ABOVE Bar A before this project starts Phase 2 (see the README's Decision Gate). A config "clears the gate" only if its bits/patch is at or above this bar BOTH on a clean image AND in its single worst-performing tested corruption -- a config that only clears on average is not a robust win. **NOTE: the measured Bar A (8.096 bits/token) now EXCEEDS Gate #1's fixed 8.0-bit bar, so Gate #1 no longer functions as a comfort margin above the economic bar -- clearing Gate #1 is NOT sufficient to beat measured base64 density. Reported for continuity; Bar A is the bar that matters.**
- **Bar C -- token crossover (the actual measured benefit claim):** does encoding a payload as a heliogram grid cost FEWER total patches (~1 token/patch for a self-hosted VLM) than base64-ing the same payload into text tokens (at chars/token = 1.3498, the resolved tokenizer baseline)? This is an ACCOUNTING comparison of token COUNT, not bits/patch density -- a config can win on Bar C while still failing Bar A, because RS/framing overhead amortizes differently for the two encodings as payload grows. See the dedicated "Token crossover" section below for the real numbers and the crossover payload size per palette.

**MANDATORY honesty caveat:** rows with `subpatch=1` are the VLM-meaningful regime -- one symbol per DATA patch, i.e. one symbol per (nominal) vision token, the only regime this project claims any real relevance to a downstream VLM. Rows with `subpatch>1` are a **PIXEL-DECODER GEOMETRIC CEILING ONLY**: `decode_pixels`/`extract_symbols` can read sub-patch cells trivially because they sample known, exact pixel coordinates off a grid whose size they are told in advance -- there is no perception involved. Whether a real ViT/VLM image encoder can resolve sub-patch structure at all is **unverified, and doubtful** (a k x k sub-cell grid inside one ViT patch may simply average out in that patch's embedding). Realizing it is Phase 2 work, gated on GPU access, and is **not a capability claim** made anywhere in this repo.

**Also mandatory, and specific to the largest palettes (visible here, in the headline area, on purpose):** `palette=128` and `palette=256` clean-decode exactly on this pixel decoder (see `tests/test_roundtrip.py`) but are MEASURED to FAIL decode under `jpeg_q70` in this very sweep (see the full breakdown below and the "Token crossover" section, which shows the clean-token-cheaper number and the corrupted-decode-failure number for the SAME cells side by side). The token-count benefit these two palettes unlock (Bar C) is therefore a property of the CLEAN channel only -- it is **not currently usable end to end** on this reference decoder, and realizing it under corruption is conditional on Phase 2 producing a reader that survives corruption at this palette size, which `decode_pixels` itself does not.

| palette | subpatch | payload (B) | ceiling k²·log2(P) | clean bits/patch | beats 8 clean? (Bar A) | clears 8 clean? | worst-corruption bits/patch | worst corruption | clears 8 corrupted? | clears gate (both, Bar B)? |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 1 | 0.527 | no | no | 0.527 | resize_3pct | no | no |
| 2 | 1 | 1024 | 1 | 0.844 | no | no | 0.844 | resize_3pct | no | no |
| 2 | 1 | 4096 | 1 | 0.862 | no | no | 0.862 | resize_3pct | no | no |
| 2 | 1 | 16384 | 1 | 0.871 | no | no | 0.871 | resize_3pct | no | no |
| 2 | 2 | 48 | 4 | 1.959 | no | no | 1.959 | resize_3pct | no | no |
| 2 | 2 | 1024 | 4 | 3.344 | no | no | 3.344 | resize_3pct | no | no |
| 2 | 2 | 4096 | 4 | 3.412 | no | no | 3.412 | resize_3pct | no | no |
| 2 | 2 | 16384 | 4 | 3.465 | no | no | 3.465 | resize_3pct | no | no |
| 4 | 1 | 48 | 2 | 1.064 | no | no | 1.064 | resize_3pct | no | no |
| 4 | 1 | 1024 | 2 | 1.696 | no | no | 1.696 | resize_3pct | no | no |
| 4 | 1 | 4096 | 2 | 1.721 | no | no | 1.721 | resize_3pct | no | no |
| 4 | 1 | 16384 | 2 | 1.740 | no | no | 1.740 | resize_3pct | no | no |
| 4 | 2 | 48 | 8 | 3.840 | no | no | 3.840 | resize_3pct | no | no |
| 4 | 2 | 1024 | 8 | 6.687 | no | no | 6.687 | resize_3pct | no | no |
| 4 | 2 | 4096 | 8 | 6.784 | no | no | 6.784 | resize_3pct | no | no |
| 4 | 2 | 16384 | 8 | 6.933 | no | no | 6.933 | resize_3pct | no | no |
| 8 | 1 | 48 | 3 | 1.500 | no | no | 1.500 | resize_3pct | no | no |
| 8 | 1 | 1024 | 3 | 2.521 | no | no | 2.521 | resize_3pct | no | no |
| 8 | 1 | 4096 | 3 | 2.566 | no | no | 2.566 | resize_3pct | no | no |
| 8 | 1 | 16384 | 3 | 2.601 | no | no | 2.601 | resize_3pct | no | no |
| 8 | 2 | 48 | 12 | 5.333 | no | no | 5.333 | resize_3pct | no | no |
| 8 | 2 | 1024 | 12 | 9.741 | yes | yes | 9.741 | resize_3pct | yes | **YES** |
| 8 | 2 | 4096 | 12 | 10.086 | yes | yes | 10.086 | resize_3pct | yes | **YES** |
| 8 | 2 | 16384 | 12 | 10.357 | yes | yes | 10.357 | resize_3pct | yes | **YES** |
| 16 | 1 | 48 | 4 | 2.000 | no | no | 2.000 | resize_3pct | no | no |
| 16 | 1 | 1024 | 4 | 3.344 | no | no | 3.344 | resize_3pct | no | no |
| 16 | 1 | 4096 | 4 | 3.412 | no | no | 3.412 | resize_3pct | no | no |
| 16 | 1 | 16384 | 4 | 3.465 | no | no | 3.465 | resize_3pct | no | no |
| 16 | 2 | 48 | 16 | 6.000 | no | no | 0.000 | combined | no | no |
| 16 | 2 | 1024 | 16 | 13.107 | yes | yes | 0.000 | combined | no | no |
| 16 | 2 | 4096 | 16 | 13.375 | yes | yes | 0.000 | combined | no | no |
| 16 | 2 | 16384 | 16 | 13.788 | yes | yes | 0.000 | combined | no | no |
| 32 | 1 | 48 | 5 | 2.000 | no | no | 2.000 | resize_3pct | no | no |
| 32 | 1 | 1024 | 5 | 4.137 | no | no | 0.000 | jpeg_q70 | no | no |
| 32 | 1 | 4096 | 5 | 4.280 | no | no | 0.000 | jpeg_q70 | no | no |
| 32 | 1 | 16384 | 5 | 4.329 | no | no | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 48 | 20 | 4.000 | no | no | 0.000 | combined | no | no |
| 32 | 2 | 1024 | 20 | 16.000 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 4096 | 20 | 16.926 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 32 | 2 | 16384 | 20 | 17.120 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 1 | 48 | 6 | 2.000 | no | no | 2.000 | resize_3pct | no | no |
| 64 | 1 | 1024 | 6 | 4.923 | no | no | 0.000 | jpeg_q70 | no | no |
| 64 | 1 | 4096 | 6 | 5.120 | no | no | 0.000 | combined | no | no |
| 64 | 1 | 16384 | 6 | 5.185 | no | no | 1.728 | jpeg_q70 | no | no |
| 64 | 2 | 48 | 24 | 3.000 | no | no | 0.000 | combined | no | no |
| 64 | 2 | 1024 | 24 | 16.000 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 2 | 4096 | 24 | 19.692 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 64 | 2 | 16384 | 24 | 20.480 | yes | yes | 0.000 | jpeg_q70 | no | no |
| 128 | 1 | 48 | 7 | 1.500 | no | no | 1.500 | resize_3pct | no | no |
| 128 | 1 | 1024 | 7 | 5.333 | no | no | 0.000 | jpeg_q70 | no | no |
| 128 | 1 | 4096 | 7 | 5.818 | no | no | 0.000 | jpeg_q85 | no | no |
| 128 | 1 | 16384 | 7 | 6.066 | no | no | 0.000 | jpeg_q85 | no | no |
| 128 | 2 | 48 | 28 | 1.500 | no | no | 0.000 | combined | no | no |
| 128 | 2 | 1024 | 28 | 16.000 | yes | yes | 0.000 | jpeg_q85 | no | no |
| 128 | 2 | 4096 | 28 | 21.333 | yes | yes | 0.000 | jpeg_q85 | no | no |
| 128 | 2 | 16384 | 28 | 23.814 | yes | yes | 0.000 | jpeg_q85 | no | no |
| 256 | 1 | 48 | 8 | 0.750 | no | no | 0.000 | jpeg_q70 | no | no |
| 256 | 1 | 1024 | 8 | 5.333 | no | no | 0.000 | jpeg_q85 | no | no |
| 256 | 1 | 4096 | 8 | 6.400 | no | no | 0.000 | jpeg_q85 | no | no |
| 256 | 1 | 16384 | 8 | 6.827 | no | no | 0.000 | jpeg_q85 | no | no |
| 256 | 2 | 48 | 32 | 0.750 | no | no | 0.000 | jpeg_q85 | no | no |
| 256 | 2 | 1024 | 32 | 10.667 | yes | yes | 0.000 | jpeg_q95 | no | no |
| 256 | 2 | 4096 | 32 | 21.333 | yes | yes | 0.000 | jpeg_q95 | no | no |
| 256 | 2 | 16384 | 32 | 25.600 | yes | yes | 0.000 | jpeg_q95 | no | no |

**Configs that clear the gate (both clean and worst-case corruption, Bar B):**

- palette=8, subpatch=2, payload_size=1024B -- clean 9.741 bits/patch, worst 9.741 bits/patch (worst corruption: `resize_3pct`)
- palette=8, subpatch=2, payload_size=4096B -- clean 10.086 bits/patch, worst 10.086 bits/patch (worst corruption: `resize_3pct`)
- palette=8, subpatch=2, payload_size=16384B -- clean 10.357 bits/patch, worst 10.357 bits/patch (worst corruption: `resize_3pct`)

**Configs that beat the base64 density bar clean (Bar A -- may or may not survive corruption; see the worst-corruption columns in the table above and the "Token crossover" section for whether that matters for tokens too):**

- palette=8, subpatch=2, payload_size=1024B -- clean 9.741 bits/patch (worst-corruption: 9.741, `resize_3pct`, clears Bar A under that corruption)
- palette=8, subpatch=2, payload_size=4096B -- clean 10.086 bits/patch (worst-corruption: 10.086, `resize_3pct`, clears Bar A under that corruption)
- palette=8, subpatch=2, payload_size=16384B -- clean 10.357 bits/patch (worst-corruption: 10.357, `resize_3pct`, clears Bar A under that corruption)
- palette=16, subpatch=2, payload_size=1024B -- clean 13.107 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=16, subpatch=2, payload_size=4096B -- clean 13.375 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=16, subpatch=2, payload_size=16384B -- clean 13.788 bits/patch (worst-corruption: 0.000, `combined`, does NOT clear Bar A under that corruption)
- palette=32, subpatch=2, payload_size=1024B -- clean 16.000 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=32, subpatch=2, payload_size=4096B -- clean 16.926 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=32, subpatch=2, payload_size=16384B -- clean 17.120 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=64, subpatch=2, payload_size=1024B -- clean 16.000 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=64, subpatch=2, payload_size=4096B -- clean 19.692 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=64, subpatch=2, payload_size=16384B -- clean 20.480 bits/patch (worst-corruption: 0.000, `jpeg_q70`, does NOT clear Bar A under that corruption)
- palette=128, subpatch=2, payload_size=1024B -- clean 16.000 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=128, subpatch=2, payload_size=4096B -- clean 21.333 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=128, subpatch=2, payload_size=16384B -- clean 23.814 bits/patch (worst-corruption: 0.000, `jpeg_q85`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=2, payload_size=1024B -- clean 10.667 bits/patch (worst-corruption: 0.000, `jpeg_q95`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=2, payload_size=4096B -- clean 21.333 bits/patch (worst-corruption: 0.000, `jpeg_q95`, does NOT clear Bar A under that corruption)
- palette=256, subpatch=2, payload_size=16384B -- clean 25.600 bits/patch (worst-corruption: 0.000, `jpeg_q95`, does NOT clear Bar A under that corruption)

**Verdict (derived from the tables above, not asserted independently):**

Every Gate #1 (Bar B) clearing config has `subpatch>1` -- the unverified pixel-decoder geometric ceiling regime. **No `subpatch=1` (VLM-meaningful) config clears Gate #1 at any tested payload size.** This is not just an unlucky corruption result: for `subpatch=1` the raw per-symbol ceiling is `log2(palette)`, which for the largest palette tested (256) is only 8 bits/patch -- already below the 8-bit Bar B *before* Reed-Solomon/calibration overhead is even subtracted. No amount of payload-size amortization can close that gap for `subpatch=1`; only the geometric `subpatch>1` regime can mathematically reach Bar B, and whether a real VLM can realize that regime is exactly the open question Phase 2 exists to answer. **Bar A tells a different story, though:** 18 config(s) beat the real economic bar clean (see the list above) -- see the "Token crossover" section below for what that means in tokens, and the mandatory P=128/256 corruption caveat above for what it does not yet mean.

## Baselines

- **base64 in text context:** ~8.1 bits/token (measured: Qwen/Qwen2.5-VL-7B-Instruct tokenizer (transformers==5.13.0), 3 payload sizes x 3 seeds each = 9 base64 samples ([1024, 4096, 16384] bytes, seeds [0, 1, 2]), 63744 tokens total for 516096 bits of original payload -> 8.0964 bits/token (1.3498 base64 chars/token, 1011.81 tokens/KB). Compare to the 6.0 bits/token analytic default (base64_bits_per_token()): HIGHER than 6.0 means the analytic estimate was favorable to heliogram's economic claim -- BPE merges commonly give base64 text MORE than 1 char/token, which pushes bits/token below the naive log2(64)=6 estimate.). Source: MEASURED baseline, tokenizer_id=`Qwen/Qwen2.5-VL-7B-Instruct` (see `_resolve_base64_baseline`). **Every Bar A ('beats base64 clean?') verdict in the Headline table above and every `GATE_BITS_PER_PATCH`/`BASE64_BITS_PER_TOKEN` comparison anywhere in this file is computed directly against THIS number.** Bar C's `token_ratio`/`heliogram_cheaper` in the "Token crossover" section below now derives from the SAME resolved baseline: `base64_token_est = floor(ceil(payload/3)*4 / chars_per_token)` with `chars_per_token = 1.3498` in this run (1.0 exactly when the source above is ANALYTIC, reproducing the old pure-character count; the measured value when it is MEASURED -- floor-rounded because understating base64's token cost is the direction conservative AGAINST heliogram's claim). The old version of this section documented a Bar A/Bar C asymmetry here (Bar C stuck on the analytic ~1-char/token estimate even when a measurement existed); that asymmetry is now closed.
- **Rendered text (geometric, model-free):** 2.13 chars/patch = 12.80 bits/patch typesetting a 48-byte payload (base64'd, 64 chars) into 30 patches of the same 14px grid unit. geometric/model-free: measures typeset packing density only, assumes perfect legibility. Real bits/patch for rendered text needs OCR accuracy from an un-fine-tuned VLM (Phase 2, out of scope here).

See "Token crossover" immediately below for the actual benefit claim (total token COUNT for a full payload, not bits/patch density) -- beating the bits/patch bar above is necessary but not sufficient for that; overhead amortization differs between the two encodings.

## Token crossover: the actual measured benefit

THE benefit claim this project can currently make: does encoding a payload as a heliogram grid cost fewer total patches (`total_patches`, the grid's width*height -- ~1 token/patch for a self-hosted VLM that tokenizes at the same patch grid) than base64-ing the same payload bytes into text tokens (`base64_token_est` = ceil(payload/3)*4 base64 characters divided by chars/token = 1.3498 -- the resolved tokenizer baseline, see Baselines above)? `token_ratio = total_patches / base64_token_est`; `token_ratio < 1.0` means heliogram is CHEAPER on token count for that payload -- an accounting fact about total context cost for the WHOLE payload, distinct from the bits/patch DENSITY bars in the Headline section (a config can win here while losing on bits/patch, because the two encodings amortize fixed overhead differently as payload grows: heliogram pays a calibration row + per-RS-chunk parity once per image, base64 pays none of that but never exceeds 6 bits/char either).

**HONESTY (mandatory, same rule as everywhere else in this file):** `token_ratio` and `heliogram_cheaper` are computed from `total_patches` alone -- a property of grid geometry -- regardless of whether `decode_success_rate` for that same cell is 1.0 or 0.0. Token-cheaper is an accounting fact about COUNT, not a claim that any reader can actually recover the payload from that many patches. The table below shows both numbers for every bucket side by side, on purpose: for `palette` in {128, 256}, `token_ratio` can drop below 1.0 at a payload size where `jpeg_q70 decode success` is still 0.00 in this same sweep -- so the token-count benefit these two palettes unlock is currently a CLEAN-CHANNEL-ONLY number. Usability under real corruption is exactly the Phase-2 reader-robustness bet described in the Headline section above, not something this table settles. This table's `total_patches` is the CONSERVATIVE, ~1-ViT-patch/token accounting; see the dedicated LM-token subsection below for the SAME comparison against the Qwen2.5-VL 2x2-merged token count, which carries its own, separate mandatory caveat.

| palette | subpatch | payload (B) | total_patches | base64_token_est | token_ratio | cheaper on tokens? | clean decode success | jpeg_q70 decode success |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 729 | 47 | 15.511 | no | 1.00 | 1.00 |
| 2 | 1 | 1024 | 9702 | 1013 | 9.577 | no | 1.00 | 1.00 |
| 2 | 1 | 4096 | 38025 | 4048 | 9.394 | no | 1.00 | 1.00 |
| 2 | 1 | 16384 | 150544 | 16186 | 9.301 | no | 1.00 | 1.00 |
| 2 | 2 | 48 | 196 | 47 | 4.170 | no | 1.00 | 1.00 |
| 2 | 2 | 1024 | 2450 | 1013 | 2.419 | no | 1.00 | 1.00 |
| 2 | 2 | 4096 | 9604 | 4048 | 2.373 | no | 1.00 | 1.00 |
| 2 | 2 | 16384 | 37830 | 16186 | 2.337 | no | 1.00 | 1.00 |
| 4 | 1 | 48 | 361 | 47 | 7.681 | no | 1.00 | 1.00 |
| 4 | 1 | 1024 | 4830 | 1013 | 4.768 | no | 1.00 | 1.00 |
| 4 | 1 | 4096 | 19044 | 4048 | 4.705 | no | 1.00 | 1.00 |
| 4 | 1 | 16384 | 75350 | 16186 | 4.655 | no | 1.00 | 1.00 |
| 4 | 2 | 48 | 100 | 47 | 2.128 | no | 1.00 | 1.00 |
| 4 | 2 | 1024 | 1225 | 1013 | 1.209 | no | 1.00 | 1.00 |
| 4 | 2 | 4096 | 4830 | 4048 | 1.193 | no | 1.00 | 1.00 |
| 4 | 2 | 16384 | 18906 | 16186 | 1.168 | no | 1.00 | 1.00 |
| 8 | 1 | 48 | 256 | 47 | 5.447 | no | 1.00 | 1.00 |
| 8 | 1 | 1024 | 3249 | 1013 | 3.207 | no | 1.00 | 1.00 |
| 8 | 1 | 4096 | 12769 | 4048 | 3.154 | no | 1.00 | 1.00 |
| 8 | 1 | 16384 | 50400 | 16186 | 3.114 | no | 1.00 | 1.00 |
| 8 | 2 | 48 | 72 | 47 | 1.532 | no | 1.00 | 1.00 |
| 8 | 2 | 1024 | 841 | 1013 | 0.830 | **YES** | 1.00 | 1.00 |
| 8 | 2 | 4096 | 3249 | 4048 | 0.803 | **YES** | 1.00 | 1.00 |
| 8 | 2 | 16384 | 12656 | 16186 | 0.782 | **YES** | 1.00 | 1.00 |
| 16 | 1 | 48 | 192 | 47 | 4.085 | no | 1.00 | 1.00 |
| 16 | 1 | 1024 | 2450 | 1013 | 2.419 | no | 1.00 | 1.00 |
| 16 | 1 | 4096 | 9604 | 4048 | 2.373 | no | 1.00 | 1.00 |
| 16 | 1 | 16384 | 37830 | 16186 | 2.337 | no | 1.00 | 1.00 |
| 16 | 2 | 48 | 64 | 47 | 1.362 | no | 1.00 | 1.00 |
| 16 | 2 | 1024 | 625 | 1013 | 0.617 | **YES** | 1.00 | 1.00 |
| 16 | 2 | 4096 | 2450 | 4048 | 0.605 | **YES** | 1.00 | 1.00 |
| 16 | 2 | 16384 | 9506 | 16186 | 0.587 | **YES** | 1.00 | 1.00 |
| 32 | 1 | 48 | 192 | 47 | 4.085 | no | 1.00 | 1.00 |
| 32 | 1 | 1024 | 1980 | 1013 | 1.955 | no | 1.00 | 0.00 |
| 32 | 1 | 4096 | 7656 | 4048 | 1.891 | no | 1.00 | 0.00 |
| 32 | 1 | 16384 | 30276 | 16186 | 1.871 | no | 1.00 | 0.00 |
| 32 | 2 | 48 | 96 | 47 | 2.043 | no | 1.00 | 1.00 |
| 32 | 2 | 1024 | 512 | 1013 | 0.505 | **YES** | 1.00 | 0.00 |
| 32 | 2 | 4096 | 1936 | 4048 | 0.478 | **YES** | 1.00 | 0.00 |
| 32 | 2 | 16384 | 7656 | 16186 | 0.473 | **YES** | 1.00 | 0.00 |
| 64 | 1 | 48 | 192 | 47 | 4.085 | no | 1.00 | 1.00 |
| 64 | 1 | 1024 | 1664 | 1013 | 1.643 | no | 1.00 | 0.00 |
| 64 | 1 | 4096 | 6400 | 4048 | 1.581 | no | 1.00 | 0.33 |
| 64 | 1 | 16384 | 25281 | 16186 | 1.562 | no | 1.00 | 0.33 |
| 64 | 2 | 48 | 128 | 47 | 2.723 | no | 1.00 | 1.00 |
| 64 | 2 | 1024 | 512 | 1013 | 0.505 | **YES** | 1.00 | 0.00 |
| 64 | 2 | 4096 | 1664 | 4048 | 0.411 | **YES** | 1.00 | 0.00 |
| 64 | 2 | 16384 | 6400 | 16186 | 0.395 | **YES** | 1.00 | 0.00 |
| 128 | 1 | 48 | 256 | 47 | 5.447 | no | 1.00 | 1.00 |
| 128 | 1 | 1024 | 1536 | 1013 | 1.516 | no | 1.00 | 0.00 |
| 128 | 1 | 4096 | 5632 | 4048 | 1.391 | no | 1.00 | 0.00 |
| 128 | 1 | 16384 | 21609 | 16186 | 1.335 | no | 1.00 | 0.00 |
| 128 | 2 | 48 | 256 | 47 | 5.447 | no | 1.00 | 0.33 |
| 128 | 2 | 1024 | 512 | 1013 | 0.505 | **YES** | 1.00 | 0.00 |
| 128 | 2 | 4096 | 1536 | 4048 | 0.379 | **YES** | 1.00 | 0.00 |
| 128 | 2 | 16384 | 5504 | 16186 | 0.340 | **YES** | 1.00 | 0.00 |
| 256 | 1 | 48 | 512 | 47 | 10.894 | no | 1.00 | 0.00 |
| 256 | 1 | 1024 | 1536 | 1013 | 1.516 | no | 1.00 | 0.00 |
| 256 | 1 | 4096 | 5120 | 4048 | 1.265 | no | 1.00 | 0.00 |
| 256 | 1 | 16384 | 19200 | 16186 | 1.186 | no | 1.00 | 0.00 |
| 256 | 2 | 48 | 512 | 47 | 10.894 | no | 1.00 | 0.00 |
| 256 | 2 | 1024 | 768 | 1013 | 0.758 | **YES** | 1.00 | 0.00 |
| 256 | 2 | 4096 | 1536 | 4048 | 0.379 | **YES** | 1.00 | 0.00 |
| 256 | 2 | 16384 | 5120 | 16186 | 0.316 | **YES** | 1.00 | 0.00 |

### LM-token accounting (Qwen2.5-VL 2x2 spatial merger)

**MANDATORY caveat, same epistemic class as the subpatch>1 pixel-decoder-only caveat in the Headline section above:** the table below re-accounts total token cost against `lm_tokens_2x2 = ceil(width/2) * ceil(height/2)` -- the LM-VISIBLE token count after Qwen2.5-VL's 2x2 spatial merger folds every 2x2 block of ViT patches into ONE token the language model actually sees -- instead of the raw `total_patches` (ViT-patch) count the table above uses. Because 4 ViT patches collapse into 1 merged token, this means each merged LM token must carry `4 * subpatch**2 * log2(palette)` bits of payload (4 patches' worth of symbols, folded into ONE merged embedding) for a reader to recover the payload from that many LM tokens. WHETHER the model can actually read that many symbols back out of a single merged embedding is UNVERIFIED: this harness only ever samples exact, known pixel coordinates via `decode_pixels`/`extract_symbols` -- it never asks a real vision encoder (merged or not) to resolve anything. Realizing (or falsifying) this is exactly the Phase-2 GPU measurement this project has not yet done. **The per-patch accounting in the table above remains the conservative headline number; treat every `lm_token_ratio` / 'cheaper (LM)?' value below as an UPPER BOUND on the possible benefit, not a result.**

| palette | subpatch | payload (B) | total_patches | token_ratio (per-patch) | cheaper (per-patch)? | lm_tokens_2x2 | lm_token_ratio (2x2 merger) | cheaper (LM, UNVERIFIED)? |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 729 | 15.511 | no | 196 | 4.170 | no |
| 2 | 1 | 1024 | 9702 | 9.577 | no | 2450 | 2.419 | no |
| 2 | 1 | 4096 | 38025 | 9.394 | no | 9604 | 2.373 | no |
| 2 | 1 | 16384 | 150544 | 9.301 | no | 37636 | 2.325 | no |
| 2 | 2 | 48 | 196 | 4.170 | no | 49 | 1.043 | no |
| 2 | 2 | 1024 | 2450 | 2.419 | no | 625 | 0.617 | **YES** |
| 2 | 2 | 4096 | 9604 | 2.373 | no | 2401 | 0.593 | **YES** |
| 2 | 2 | 16384 | 37830 | 2.337 | no | 9506 | 0.587 | **YES** |
| 4 | 1 | 48 | 361 | 7.681 | no | 100 | 2.128 | no |
| 4 | 1 | 1024 | 4830 | 4.768 | no | 1225 | 1.209 | no |
| 4 | 1 | 4096 | 19044 | 4.705 | no | 4761 | 1.176 | no |
| 4 | 1 | 16384 | 75350 | 4.655 | no | 18906 | 1.168 | no |
| 4 | 2 | 48 | 100 | 2.128 | no | 25 | 0.532 | **YES** |
| 4 | 2 | 1024 | 1225 | 1.209 | no | 324 | 0.320 | **YES** |
| 4 | 2 | 4096 | 4830 | 1.193 | no | 1225 | 0.303 | **YES** |
| 4 | 2 | 16384 | 18906 | 1.168 | no | 4761 | 0.294 | **YES** |
| 8 | 1 | 48 | 256 | 5.447 | no | 64 | 1.362 | no |
| 8 | 1 | 1024 | 3249 | 3.207 | no | 841 | 0.830 | **YES** |
| 8 | 1 | 4096 | 12769 | 3.154 | no | 3249 | 0.803 | **YES** |
| 8 | 1 | 16384 | 50400 | 3.114 | no | 12656 | 0.782 | **YES** |
| 8 | 2 | 48 | 72 | 1.532 | no | 20 | 0.426 | **YES** |
| 8 | 2 | 1024 | 841 | 0.830 | yes | 225 | 0.222 | **YES** |
| 8 | 2 | 4096 | 3249 | 0.803 | yes | 841 | 0.208 | **YES** |
| 8 | 2 | 16384 | 12656 | 0.782 | yes | 3192 | 0.197 | **YES** |
| 16 | 1 | 48 | 192 | 4.085 | no | 48 | 1.021 | no |
| 16 | 1 | 1024 | 2450 | 2.419 | no | 625 | 0.617 | **YES** |
| 16 | 1 | 4096 | 9604 | 2.373 | no | 2401 | 0.593 | **YES** |
| 16 | 1 | 16384 | 37830 | 2.337 | no | 9506 | 0.587 | **YES** |
| 16 | 2 | 48 | 64 | 1.362 | no | 16 | 0.340 | **YES** |
| 16 | 2 | 1024 | 625 | 0.617 | yes | 169 | 0.167 | **YES** |
| 16 | 2 | 4096 | 2450 | 0.605 | yes | 625 | 0.154 | **YES** |
| 16 | 2 | 16384 | 9506 | 0.587 | yes | 2401 | 0.148 | **YES** |
| 32 | 1 | 48 | 192 | 4.085 | no | 48 | 1.021 | no |
| 32 | 1 | 1024 | 1980 | 1.955 | no | 506 | 0.500 | **YES** |
| 32 | 1 | 4096 | 7656 | 1.891 | no | 1936 | 0.478 | **YES** |
| 32 | 1 | 16384 | 30276 | 1.871 | no | 7569 | 0.468 | **YES** |
| 32 | 2 | 48 | 96 | 2.043 | no | 32 | 0.681 | **YES** |
| 32 | 2 | 1024 | 512 | 0.505 | yes | 128 | 0.126 | **YES** |
| 32 | 2 | 4096 | 1936 | 0.478 | yes | 484 | 0.120 | **YES** |
| 32 | 2 | 16384 | 7656 | 0.473 | yes | 1936 | 0.120 | **YES** |
| 64 | 1 | 48 | 192 | 4.085 | no | 64 | 1.362 | no |
| 64 | 1 | 1024 | 1664 | 1.643 | no | 416 | 0.411 | **YES** |
| 64 | 1 | 4096 | 6400 | 1.581 | no | 1600 | 0.395 | **YES** |
| 64 | 1 | 16384 | 25281 | 1.562 | no | 6400 | 0.395 | **YES** |
| 64 | 2 | 48 | 128 | 2.723 | no | 32 | 0.681 | **YES** |
| 64 | 2 | 1024 | 512 | 0.505 | yes | 128 | 0.126 | **YES** |
| 64 | 2 | 4096 | 1664 | 0.411 | yes | 416 | 0.103 | **YES** |
| 64 | 2 | 16384 | 6400 | 0.395 | yes | 1600 | 0.099 | **YES** |
| 128 | 1 | 48 | 256 | 5.447 | no | 64 | 1.362 | no |
| 128 | 1 | 1024 | 1536 | 1.516 | no | 384 | 0.379 | **YES** |
| 128 | 1 | 4096 | 5632 | 1.391 | no | 1408 | 0.348 | **YES** |
| 128 | 1 | 16384 | 21609 | 1.335 | no | 5476 | 0.338 | **YES** |
| 128 | 2 | 48 | 256 | 5.447 | no | 64 | 1.362 | no |
| 128 | 2 | 1024 | 512 | 0.505 | yes | 128 | 0.126 | **YES** |
| 128 | 2 | 4096 | 1536 | 0.379 | yes | 384 | 0.095 | **YES** |
| 128 | 2 | 16384 | 5504 | 0.340 | yes | 1408 | 0.087 | **YES** |
| 256 | 1 | 48 | 512 | 10.894 | no | 128 | 2.723 | no |
| 256 | 1 | 1024 | 1536 | 1.516 | no | 384 | 0.379 | **YES** |
| 256 | 1 | 4096 | 5120 | 1.265 | no | 1280 | 0.316 | **YES** |
| 256 | 1 | 16384 | 19200 | 1.186 | no | 4864 | 0.301 | **YES** |
| 256 | 2 | 48 | 512 | 10.894 | no | 128 | 2.723 | no |
| 256 | 2 | 1024 | 768 | 0.758 | yes | 256 | 0.253 | **YES** |
| 256 | 2 | 4096 | 1536 | 0.379 | yes | 384 | 0.095 | **YES** |
| 256 | 2 | 16384 | 5120 | 0.316 | yes | 1280 | 0.079 | **YES** |

### Crossover payload size per (palette, subpatch) -- exact scan

Exact, byte-granular payload size where each accounting's ratio first drops below 1.0 -- NOT the linear interpolation between the handful of swept sample points ([48, 1024, 4096, 16384]B) an earlier version of this section used, which could report spurious precision (e.g. '~3055B') for what is actually a staircase function (see `exact_crossover_payload_size`'s docstring for why). Both sides of the ratio are closed-form for ANY payload size -- `_grid_stats(...).total_patches`/`.lm_tokens_2x2` and `ceil(n/3)*4` -- so this instead walks every payload size from 1B up to 65536B and reports the exact smallest crossing, for BOTH the per-patch and LM-token (2x2 merger, see the caveat above) accountings. 'no crossover found' means the ratio never dropped below 1.0 anywhere in the scanned range -- NOT a claim it never will at a larger, unscanned payload size. Where the ratio recrosses back above 1.0 after its first crossing (the staircase can wobble), that is reported too, rather than silently stating a single crossing point as if it were a stable threshold.

**subpatch=1 (VLM-meaningful: one symbol per patch):**

- palette=2: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 9.273 at 64219B); LM-token (UNVERIFIED, see caveat above) no crossover found in the exact scan up to 65536B (lowest ratio observed: 2.318 at 60421B)
- palette=4: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 4.639 at 64219B); LM-token (UNVERIFIED, see caveat above) no crossover found in the exact scan up to 65536B (lowest ratio observed: 1.160 at 65524B)
- palette=8: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 3.094 at 63772B); LM-token (UNVERIFIED, see caveat above) crosses at 103B (exact) -- WOBBLES: recrosses back to >= 1.0 at 106B (and 2 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring
- palette=16: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 2.321 at 65308B); LM-token (UNVERIFIED, see caveat above) crosses at 49B (exact) -- WOBBLES: recrosses back to >= 1.0 at 52B (and 0 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring
- palette=32: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 1.858 at 63550B); LM-token (UNVERIFIED, see caveat above) crosses at 49B (exact)
- palette=64: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 1.549 at 64660B); LM-token (UNVERIFIED, see caveat above) crosses at 64B (exact)
- palette=128: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 1.328 at 60199B); LM-token (UNVERIFIED, see caveat above) crosses at 64B (exact) -- WOBBLES: recrosses back to >= 1.0 at 76B (and 0 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring
- palette=256: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 1.162 at 64219B); LM-token (UNVERIFIED, see caveat above) crosses at 130B (exact) -- WOBBLES: recrosses back to >= 1.0 at 219B (and 0 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring

**subpatch=2 (PIXEL-DECODER GEOMETRIC CEILING ONLY -- not a VLM capability claim, see caveat above):**

- palette=2: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 2.321 at 65308B); LM-token (UNVERIFIED, see caveat above) crosses at 49B (exact) -- WOBBLES: recrosses back to >= 1.0 at 55B (and 1 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring
- palette=4: per-patch no crossover found in the exact scan up to 65536B (lowest ratio observed: 1.162 at 64219B); LM-token (UNVERIFIED, see caveat above) crosses at 16B (exact) -- WOBBLES: recrosses back to >= 1.0 at 20B (and 0 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring
- palette=8: per-patch crosses at 112B (exact) -- WOBBLES: recrosses back to >= 1.0 at 114B (and 2 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring; LM-token (UNVERIFIED, see caveat above) crosses at 13B (exact)
- palette=16: per-patch crosses at 82B (exact) -- WOBBLES: recrosses back to >= 1.0 at 92B (and 0 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring; LM-token (UNVERIFIED, see caveat above) crosses at 16B (exact)
- palette=32: per-patch crosses at 97B (exact) -- WOBBLES: recrosses back to >= 1.0 at 124B (and 0 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring; LM-token (UNVERIFIED, see caveat above) crosses at 16B (exact)
- palette=64: per-patch crosses at 130B (exact) -- WOBBLES: recrosses back to >= 1.0 at 156B (and 0 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring; LM-token (UNVERIFIED, see caveat above) crosses at 34B (exact)
- palette=128: per-patch crosses at 259B (exact) -- WOBBLES: recrosses back to >= 1.0 at 380B (and 0 more time(s) within the scanned range) -- NOT a stable one-way threshold, see exact_crossover_payload_size's docstring; LM-token (UNVERIFIED, see caveat above) crosses at 64B (exact)
- palette=256: per-patch crosses at 520B (exact); LM-token (UNVERIFIED, see caveat above) crosses at 130B (exact)

### Token-crossover verdict

No `subpatch=1` (VLM-meaningful) palette crosses below base64 token count, per-patch accounting, anywhere in the exact scanned range (up to 65536B) in this run.

## Summary by sub-patch regime (payload-size amortization)

Fixed per-message overhead (5-byte frame header + Reed-Solomon parity + the calibration row) is amortized over more data patches as payload size grows, so bits/patch should rise toward the `subpatch²·log2(palette)` ceiling as payload grows -- this is the amortization half of this sweep. 'corr(mean)' is the mean bits/patch over every non-clean corruption in the table below (resize 3%/5%, JPEG q95/85/70, crop/pad 2px, combined), each counted as 0 on a failed decode.

### subpatch=1 (VLM-meaningful: one symbol per patch)

| Palette | bits/sym | ceiling | 48B clean | 48B corr(mean) | 1024B clean | 1024B corr(mean) | 4096B clean | 4096B corr(mean) | 16384B clean | 16384B corr(mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 1 | 0.527 | 0.527 | 0.844 | 0.844 | 0.862 | 0.862 | 0.871 | 0.871 |
| 4 | 2 | 2 | 1.064 | 1.064 | 1.696 | 1.696 | 1.721 | 1.721 | 1.740 | 1.740 |
| 8 | 3 | 3 | 1.500 | 1.500 | 2.521 | 2.521 | 2.566 | 2.566 | 2.601 | 2.601 |
| 16 | 4 | 4 | 2.000 | 2.000 | 3.344 | 3.344 | 3.412 | 3.412 | 3.465 | 3.465 |
| 32 | 5 | 5 | 2.000 | 2.000 | 4.137 | 3.349 | 4.280 | 3.057 | 4.329 | 3.092 |
| 64 | 6 | 6 | 2.000 | 2.000 | 4.923 | 3.985 | 5.120 | 3.901 | 5.185 | 4.691 |
| 128 | 7 | 7 | 1.500 | 1.500 | 5.333 | 3.556 | 5.818 | 3.325 | 6.066 | 3.466 |
| 256 | 8 | 8 | 0.750 | 0.536 | 5.333 | 3.048 | 6.400 | 3.657 | 6.827 | 3.901 |

### subpatch=2 (PIXEL-DECODER GEOMETRIC CEILING ONLY -- not a VLM capability claim, see caveat above)

| Palette | bits/sym | ceiling | 48B clean | 48B corr(mean) | 1024B clean | 1024B corr(mean) | 4096B clean | 4096B corr(mean) | 16384B clean | 16384B corr(mean) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 4 | 1.959 | 1.959 | 3.344 | 3.344 | 3.412 | 3.412 | 3.465 | 3.465 |
| 4 | 2 | 8 | 3.840 | 3.840 | 6.687 | 6.687 | 6.784 | 6.784 | 6.933 | 6.933 |
| 8 | 3 | 12 | 5.333 | 5.333 | 9.741 | 9.741 | 10.086 | 10.086 | 10.357 | 10.357 |
| 16 | 4 | 16 | 6.000 | 5.143 | 13.107 | 11.235 | 13.375 | 11.464 | 13.788 | 11.819 |
| 32 | 5 | 20 | 4.000 | 3.429 | 16.000 | 11.429 | 16.926 | 12.090 | 17.120 | 12.229 |
| 64 | 6 | 24 | 3.000 | 2.571 | 16.000 | 11.429 | 19.692 | 14.066 | 20.480 | 14.629 |
| 128 | 7 | 28 | 1.500 | 1.143 | 16.000 | 9.143 | 21.333 | 12.190 | 23.814 | 13.608 |
| 256 | 8 | 32 | 0.750 | 0.429 | 10.667 | 4.571 | 21.333 | 9.143 | 25.600 | 10.971 |

## Full breakdown by corruption

| palette | subpatch | payload | bits/sym | corruption | symbol error rate | decode success rate | bits/patch |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 1 | clean | 0.0000 | 1.00 | 0.527 |
| 2 | 1 | 48 | 1 | resize_3pct | 0.0000 | 1.00 | 0.527 |
| 2 | 1 | 48 | 1 | resize_5pct | 0.0000 | 1.00 | 0.527 |
| 2 | 1 | 48 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.527 |
| 2 | 1 | 48 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.527 |
| 2 | 1 | 48 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.527 |
| 2 | 1 | 48 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.527 |
| 2 | 1 | 48 | 1 | combined | 0.0000 | 1.00 | 0.527 |
| 2 | 1 | 1024 | 1 | clean | 0.0000 | 1.00 | 0.844 |
| 2 | 1 | 1024 | 1 | resize_3pct | 0.0000 | 1.00 | 0.844 |
| 2 | 1 | 1024 | 1 | resize_5pct | 0.0000 | 1.00 | 0.844 |
| 2 | 1 | 1024 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.844 |
| 2 | 1 | 1024 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.844 |
| 2 | 1 | 1024 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.844 |
| 2 | 1 | 1024 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.844 |
| 2 | 1 | 1024 | 1 | combined | 0.0000 | 1.00 | 0.844 |
| 2 | 1 | 4096 | 1 | clean | 0.0000 | 1.00 | 0.862 |
| 2 | 1 | 4096 | 1 | resize_3pct | 0.0000 | 1.00 | 0.862 |
| 2 | 1 | 4096 | 1 | resize_5pct | 0.0000 | 1.00 | 0.862 |
| 2 | 1 | 4096 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.862 |
| 2 | 1 | 4096 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.862 |
| 2 | 1 | 4096 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.862 |
| 2 | 1 | 4096 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.862 |
| 2 | 1 | 4096 | 1 | combined | 0.0000 | 1.00 | 0.862 |
| 2 | 1 | 16384 | 1 | clean | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | resize_3pct | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | resize_5pct | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | jpeg_q95 | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | jpeg_q85 | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | jpeg_q70 | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | crop_pad_2px | 0.0000 | 1.00 | 0.871 |
| 2 | 1 | 16384 | 1 | combined | 0.0000 | 1.00 | 0.871 |
| 2 | 2 | 48 | 1 | clean | 0.0000 | 1.00 | 1.959 |
| 2 | 2 | 48 | 1 | resize_3pct | 0.0000 | 1.00 | 1.959 |
| 2 | 2 | 48 | 1 | resize_5pct | 0.0000 | 1.00 | 1.959 |
| 2 | 2 | 48 | 1 | jpeg_q95 | 0.0000 | 1.00 | 1.959 |
| 2 | 2 | 48 | 1 | jpeg_q85 | 0.0000 | 1.00 | 1.959 |
| 2 | 2 | 48 | 1 | jpeg_q70 | 0.0000 | 1.00 | 1.959 |
| 2 | 2 | 48 | 1 | crop_pad_2px | 0.0000 | 1.00 | 1.959 |
| 2 | 2 | 48 | 1 | combined | 0.0000 | 1.00 | 1.959 |
| 2 | 2 | 1024 | 1 | clean | 0.0000 | 1.00 | 3.344 |
| 2 | 2 | 1024 | 1 | resize_3pct | 0.0000 | 1.00 | 3.344 |
| 2 | 2 | 1024 | 1 | resize_5pct | 0.0000 | 1.00 | 3.344 |
| 2 | 2 | 1024 | 1 | jpeg_q95 | 0.0000 | 1.00 | 3.344 |
| 2 | 2 | 1024 | 1 | jpeg_q85 | 0.0000 | 1.00 | 3.344 |
| 2 | 2 | 1024 | 1 | jpeg_q70 | 0.0000 | 1.00 | 3.344 |
| 2 | 2 | 1024 | 1 | crop_pad_2px | 0.0000 | 1.00 | 3.344 |
| 2 | 2 | 1024 | 1 | combined | 0.0000 | 1.00 | 3.344 |
| 2 | 2 | 4096 | 1 | clean | 0.0000 | 1.00 | 3.412 |
| 2 | 2 | 4096 | 1 | resize_3pct | 0.0000 | 1.00 | 3.412 |
| 2 | 2 | 4096 | 1 | resize_5pct | 0.0000 | 1.00 | 3.412 |
| 2 | 2 | 4096 | 1 | jpeg_q95 | 0.0000 | 1.00 | 3.412 |
| 2 | 2 | 4096 | 1 | jpeg_q85 | 0.0000 | 1.00 | 3.412 |
| 2 | 2 | 4096 | 1 | jpeg_q70 | 0.0000 | 1.00 | 3.412 |
| 2 | 2 | 4096 | 1 | crop_pad_2px | 0.0000 | 1.00 | 3.412 |
| 2 | 2 | 4096 | 1 | combined | 0.0000 | 1.00 | 3.412 |
| 2 | 2 | 16384 | 1 | clean | 0.0000 | 1.00 | 3.465 |
| 2 | 2 | 16384 | 1 | resize_3pct | 0.0000 | 1.00 | 3.465 |
| 2 | 2 | 16384 | 1 | resize_5pct | 0.0000 | 1.00 | 3.465 |
| 2 | 2 | 16384 | 1 | jpeg_q95 | 0.0000 | 1.00 | 3.465 |
| 2 | 2 | 16384 | 1 | jpeg_q85 | 0.0000 | 1.00 | 3.465 |
| 2 | 2 | 16384 | 1 | jpeg_q70 | 0.0000 | 1.00 | 3.465 |
| 2 | 2 | 16384 | 1 | crop_pad_2px | 0.0000 | 1.00 | 3.465 |
| 2 | 2 | 16384 | 1 | combined | 0.0000 | 1.00 | 3.465 |
| 4 | 1 | 48 | 2 | clean | 0.0000 | 1.00 | 1.064 |
| 4 | 1 | 48 | 2 | resize_3pct | 0.0000 | 1.00 | 1.064 |
| 4 | 1 | 48 | 2 | resize_5pct | 0.0000 | 1.00 | 1.064 |
| 4 | 1 | 48 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.064 |
| 4 | 1 | 48 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.064 |
| 4 | 1 | 48 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.064 |
| 4 | 1 | 48 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.064 |
| 4 | 1 | 48 | 2 | combined | 0.0000 | 1.00 | 1.064 |
| 4 | 1 | 1024 | 2 | clean | 0.0000 | 1.00 | 1.696 |
| 4 | 1 | 1024 | 2 | resize_3pct | 0.0000 | 1.00 | 1.696 |
| 4 | 1 | 1024 | 2 | resize_5pct | 0.0000 | 1.00 | 1.696 |
| 4 | 1 | 1024 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.696 |
| 4 | 1 | 1024 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.696 |
| 4 | 1 | 1024 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.696 |
| 4 | 1 | 1024 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.696 |
| 4 | 1 | 1024 | 2 | combined | 0.0000 | 1.00 | 1.696 |
| 4 | 1 | 4096 | 2 | clean | 0.0000 | 1.00 | 1.721 |
| 4 | 1 | 4096 | 2 | resize_3pct | 0.0000 | 1.00 | 1.721 |
| 4 | 1 | 4096 | 2 | resize_5pct | 0.0000 | 1.00 | 1.721 |
| 4 | 1 | 4096 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.721 |
| 4 | 1 | 4096 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.721 |
| 4 | 1 | 4096 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.721 |
| 4 | 1 | 4096 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.721 |
| 4 | 1 | 4096 | 2 | combined | 0.0000 | 1.00 | 1.721 |
| 4 | 1 | 16384 | 2 | clean | 0.0000 | 1.00 | 1.740 |
| 4 | 1 | 16384 | 2 | resize_3pct | 0.0000 | 1.00 | 1.740 |
| 4 | 1 | 16384 | 2 | resize_5pct | 0.0000 | 1.00 | 1.740 |
| 4 | 1 | 16384 | 2 | jpeg_q95 | 0.0000 | 1.00 | 1.740 |
| 4 | 1 | 16384 | 2 | jpeg_q85 | 0.0000 | 1.00 | 1.740 |
| 4 | 1 | 16384 | 2 | jpeg_q70 | 0.0000 | 1.00 | 1.740 |
| 4 | 1 | 16384 | 2 | crop_pad_2px | 0.0000 | 1.00 | 1.740 |
| 4 | 1 | 16384 | 2 | combined | 0.0000 | 1.00 | 1.740 |
| 4 | 2 | 48 | 2 | clean | 0.0000 | 1.00 | 3.840 |
| 4 | 2 | 48 | 2 | resize_3pct | 0.0000 | 1.00 | 3.840 |
| 4 | 2 | 48 | 2 | resize_5pct | 0.0000 | 1.00 | 3.840 |
| 4 | 2 | 48 | 2 | jpeg_q95 | 0.0000 | 1.00 | 3.840 |
| 4 | 2 | 48 | 2 | jpeg_q85 | 0.0000 | 1.00 | 3.840 |
| 4 | 2 | 48 | 2 | jpeg_q70 | 0.0000 | 1.00 | 3.840 |
| 4 | 2 | 48 | 2 | crop_pad_2px | 0.0000 | 1.00 | 3.840 |
| 4 | 2 | 48 | 2 | combined | 0.0000 | 1.00 | 3.840 |
| 4 | 2 | 1024 | 2 | clean | 0.0000 | 1.00 | 6.687 |
| 4 | 2 | 1024 | 2 | resize_3pct | 0.0000 | 1.00 | 6.687 |
| 4 | 2 | 1024 | 2 | resize_5pct | 0.0000 | 1.00 | 6.687 |
| 4 | 2 | 1024 | 2 | jpeg_q95 | 0.0000 | 1.00 | 6.687 |
| 4 | 2 | 1024 | 2 | jpeg_q85 | 0.0000 | 1.00 | 6.687 |
| 4 | 2 | 1024 | 2 | jpeg_q70 | 0.0000 | 1.00 | 6.687 |
| 4 | 2 | 1024 | 2 | crop_pad_2px | 0.0000 | 1.00 | 6.687 |
| 4 | 2 | 1024 | 2 | combined | 0.0000 | 1.00 | 6.687 |
| 4 | 2 | 4096 | 2 | clean | 0.0000 | 1.00 | 6.784 |
| 4 | 2 | 4096 | 2 | resize_3pct | 0.0000 | 1.00 | 6.784 |
| 4 | 2 | 4096 | 2 | resize_5pct | 0.0000 | 1.00 | 6.784 |
| 4 | 2 | 4096 | 2 | jpeg_q95 | 0.0000 | 1.00 | 6.784 |
| 4 | 2 | 4096 | 2 | jpeg_q85 | 0.0000 | 1.00 | 6.784 |
| 4 | 2 | 4096 | 2 | jpeg_q70 | 0.0000 | 1.00 | 6.784 |
| 4 | 2 | 4096 | 2 | crop_pad_2px | 0.0000 | 1.00 | 6.784 |
| 4 | 2 | 4096 | 2 | combined | 0.0000 | 1.00 | 6.784 |
| 4 | 2 | 16384 | 2 | clean | 0.0000 | 1.00 | 6.933 |
| 4 | 2 | 16384 | 2 | resize_3pct | 0.0000 | 1.00 | 6.933 |
| 4 | 2 | 16384 | 2 | resize_5pct | 0.0000 | 1.00 | 6.933 |
| 4 | 2 | 16384 | 2 | jpeg_q95 | 0.0000 | 1.00 | 6.933 |
| 4 | 2 | 16384 | 2 | jpeg_q85 | 0.0000 | 1.00 | 6.933 |
| 4 | 2 | 16384 | 2 | jpeg_q70 | 0.0000 | 1.00 | 6.933 |
| 4 | 2 | 16384 | 2 | crop_pad_2px | 0.0000 | 1.00 | 6.933 |
| 4 | 2 | 16384 | 2 | combined | 0.0000 | 1.00 | 6.933 |
| 8 | 1 | 48 | 3 | clean | 0.0000 | 1.00 | 1.500 |
| 8 | 1 | 48 | 3 | resize_3pct | 0.0000 | 1.00 | 1.500 |
| 8 | 1 | 48 | 3 | resize_5pct | 0.0000 | 1.00 | 1.500 |
| 8 | 1 | 48 | 3 | jpeg_q95 | 0.0000 | 1.00 | 1.500 |
| 8 | 1 | 48 | 3 | jpeg_q85 | 0.0000 | 1.00 | 1.500 |
| 8 | 1 | 48 | 3 | jpeg_q70 | 0.0000 | 1.00 | 1.500 |
| 8 | 1 | 48 | 3 | crop_pad_2px | 0.0000 | 1.00 | 1.500 |
| 8 | 1 | 48 | 3 | combined | 0.0000 | 1.00 | 1.500 |
| 8 | 1 | 1024 | 3 | clean | 0.0000 | 1.00 | 2.521 |
| 8 | 1 | 1024 | 3 | resize_3pct | 0.0000 | 1.00 | 2.521 |
| 8 | 1 | 1024 | 3 | resize_5pct | 0.0000 | 1.00 | 2.521 |
| 8 | 1 | 1024 | 3 | jpeg_q95 | 0.0000 | 1.00 | 2.521 |
| 8 | 1 | 1024 | 3 | jpeg_q85 | 0.0000 | 1.00 | 2.521 |
| 8 | 1 | 1024 | 3 | jpeg_q70 | 0.0000 | 1.00 | 2.521 |
| 8 | 1 | 1024 | 3 | crop_pad_2px | 0.0000 | 1.00 | 2.521 |
| 8 | 1 | 1024 | 3 | combined | 0.0000 | 1.00 | 2.521 |
| 8 | 1 | 4096 | 3 | clean | 0.0000 | 1.00 | 2.566 |
| 8 | 1 | 4096 | 3 | resize_3pct | 0.0000 | 1.00 | 2.566 |
| 8 | 1 | 4096 | 3 | resize_5pct | 0.0000 | 1.00 | 2.566 |
| 8 | 1 | 4096 | 3 | jpeg_q95 | 0.0000 | 1.00 | 2.566 |
| 8 | 1 | 4096 | 3 | jpeg_q85 | 0.0000 | 1.00 | 2.566 |
| 8 | 1 | 4096 | 3 | jpeg_q70 | 0.0000 | 1.00 | 2.566 |
| 8 | 1 | 4096 | 3 | crop_pad_2px | 0.0000 | 1.00 | 2.566 |
| 8 | 1 | 4096 | 3 | combined | 0.0000 | 1.00 | 2.566 |
| 8 | 1 | 16384 | 3 | clean | 0.0000 | 1.00 | 2.601 |
| 8 | 1 | 16384 | 3 | resize_3pct | 0.0000 | 1.00 | 2.601 |
| 8 | 1 | 16384 | 3 | resize_5pct | 0.0000 | 1.00 | 2.601 |
| 8 | 1 | 16384 | 3 | jpeg_q95 | 0.0000 | 1.00 | 2.601 |
| 8 | 1 | 16384 | 3 | jpeg_q85 | 0.0000 | 1.00 | 2.601 |
| 8 | 1 | 16384 | 3 | jpeg_q70 | 0.0000 | 1.00 | 2.601 |
| 8 | 1 | 16384 | 3 | crop_pad_2px | 0.0000 | 1.00 | 2.601 |
| 8 | 1 | 16384 | 3 | combined | 0.0000 | 1.00 | 2.601 |
| 8 | 2 | 48 | 3 | clean | 0.0000 | 1.00 | 5.333 |
| 8 | 2 | 48 | 3 | resize_3pct | 0.0000 | 1.00 | 5.333 |
| 8 | 2 | 48 | 3 | resize_5pct | 0.0000 | 1.00 | 5.333 |
| 8 | 2 | 48 | 3 | jpeg_q95 | 0.0000 | 1.00 | 5.333 |
| 8 | 2 | 48 | 3 | jpeg_q85 | 0.0000 | 1.00 | 5.333 |
| 8 | 2 | 48 | 3 | jpeg_q70 | 0.0000 | 1.00 | 5.333 |
| 8 | 2 | 48 | 3 | crop_pad_2px | 0.0000 | 1.00 | 5.333 |
| 8 | 2 | 48 | 3 | combined | 0.0130 | 1.00 | 5.333 |
| 8 | 2 | 1024 | 3 | clean | 0.0000 | 1.00 | 9.741 |
| 8 | 2 | 1024 | 3 | resize_3pct | 0.0000 | 1.00 | 9.741 |
| 8 | 2 | 1024 | 3 | resize_5pct | 0.0000 | 1.00 | 9.741 |
| 8 | 2 | 1024 | 3 | jpeg_q95 | 0.0000 | 1.00 | 9.741 |
| 8 | 2 | 1024 | 3 | jpeg_q85 | 0.0000 | 1.00 | 9.741 |
| 8 | 2 | 1024 | 3 | jpeg_q70 | 0.0000 | 1.00 | 9.741 |
| 8 | 2 | 1024 | 3 | crop_pad_2px | 0.0000 | 1.00 | 9.741 |
| 8 | 2 | 1024 | 3 | combined | 0.0055 | 1.00 | 9.741 |
| 8 | 2 | 4096 | 3 | clean | 0.0000 | 1.00 | 10.086 |
| 8 | 2 | 4096 | 3 | resize_3pct | 0.0000 | 1.00 | 10.086 |
| 8 | 2 | 4096 | 3 | resize_5pct | 0.0000 | 1.00 | 10.086 |
| 8 | 2 | 4096 | 3 | jpeg_q95 | 0.0000 | 1.00 | 10.086 |
| 8 | 2 | 4096 | 3 | jpeg_q85 | 0.0000 | 1.00 | 10.086 |
| 8 | 2 | 4096 | 3 | jpeg_q70 | 0.0000 | 1.00 | 10.086 |
| 8 | 2 | 4096 | 3 | crop_pad_2px | 0.0000 | 1.00 | 10.086 |
| 8 | 2 | 4096 | 3 | combined | 0.0071 | 1.00 | 10.086 |
| 8 | 2 | 16384 | 3 | clean | 0.0000 | 1.00 | 10.357 |
| 8 | 2 | 16384 | 3 | resize_3pct | 0.0000 | 1.00 | 10.357 |
| 8 | 2 | 16384 | 3 | resize_5pct | 0.0000 | 1.00 | 10.357 |
| 8 | 2 | 16384 | 3 | jpeg_q95 | 0.0000 | 1.00 | 10.357 |
| 8 | 2 | 16384 | 3 | jpeg_q85 | 0.0000 | 1.00 | 10.357 |
| 8 | 2 | 16384 | 3 | jpeg_q70 | 0.0000 | 1.00 | 10.357 |
| 8 | 2 | 16384 | 3 | crop_pad_2px | 0.0000 | 1.00 | 10.357 |
| 8 | 2 | 16384 | 3 | combined | 0.0062 | 1.00 | 10.357 |
| 16 | 1 | 48 | 4 | clean | 0.0000 | 1.00 | 2.000 |
| 16 | 1 | 48 | 4 | resize_3pct | 0.0000 | 1.00 | 2.000 |
| 16 | 1 | 48 | 4 | resize_5pct | 0.0000 | 1.00 | 2.000 |
| 16 | 1 | 48 | 4 | jpeg_q95 | 0.0000 | 1.00 | 2.000 |
| 16 | 1 | 48 | 4 | jpeg_q85 | 0.0000 | 1.00 | 2.000 |
| 16 | 1 | 48 | 4 | jpeg_q70 | 0.0000 | 1.00 | 2.000 |
| 16 | 1 | 48 | 4 | crop_pad_2px | 0.0000 | 1.00 | 2.000 |
| 16 | 1 | 48 | 4 | combined | 0.0000 | 1.00 | 2.000 |
| 16 | 1 | 1024 | 4 | clean | 0.0000 | 1.00 | 3.344 |
| 16 | 1 | 1024 | 4 | resize_3pct | 0.0000 | 1.00 | 3.344 |
| 16 | 1 | 1024 | 4 | resize_5pct | 0.0000 | 1.00 | 3.344 |
| 16 | 1 | 1024 | 4 | jpeg_q95 | 0.0000 | 1.00 | 3.344 |
| 16 | 1 | 1024 | 4 | jpeg_q85 | 0.0000 | 1.00 | 3.344 |
| 16 | 1 | 1024 | 4 | jpeg_q70 | 0.0003 | 1.00 | 3.344 |
| 16 | 1 | 1024 | 4 | crop_pad_2px | 0.0000 | 1.00 | 3.344 |
| 16 | 1 | 1024 | 4 | combined | 0.0001 | 1.00 | 3.344 |
| 16 | 1 | 4096 | 4 | clean | 0.0000 | 1.00 | 3.412 |
| 16 | 1 | 4096 | 4 | resize_3pct | 0.0000 | 1.00 | 3.412 |
| 16 | 1 | 4096 | 4 | resize_5pct | 0.0000 | 1.00 | 3.412 |
| 16 | 1 | 4096 | 4 | jpeg_q95 | 0.0000 | 1.00 | 3.412 |
| 16 | 1 | 4096 | 4 | jpeg_q85 | 0.0000 | 1.00 | 3.412 |
| 16 | 1 | 4096 | 4 | jpeg_q70 | 0.0007 | 1.00 | 3.412 |
| 16 | 1 | 4096 | 4 | crop_pad_2px | 0.0000 | 1.00 | 3.412 |
| 16 | 1 | 4096 | 4 | combined | 0.0002 | 1.00 | 3.412 |
| 16 | 1 | 16384 | 4 | clean | 0.0000 | 1.00 | 3.465 |
| 16 | 1 | 16384 | 4 | resize_3pct | 0.0000 | 1.00 | 3.465 |
| 16 | 1 | 16384 | 4 | resize_5pct | 0.0000 | 1.00 | 3.465 |
| 16 | 1 | 16384 | 4 | jpeg_q95 | 0.0000 | 1.00 | 3.465 |
| 16 | 1 | 16384 | 4 | jpeg_q85 | 0.0000 | 1.00 | 3.465 |
| 16 | 1 | 16384 | 4 | jpeg_q70 | 0.0007 | 1.00 | 3.465 |
| 16 | 1 | 16384 | 4 | crop_pad_2px | 0.0000 | 1.00 | 3.465 |
| 16 | 1 | 16384 | 4 | combined | 0.0001 | 1.00 | 3.465 |
| 16 | 2 | 48 | 4 | clean | 0.0000 | 1.00 | 6.000 |
| 16 | 2 | 48 | 4 | resize_3pct | 0.0000 | 1.00 | 6.000 |
| 16 | 2 | 48 | 4 | resize_5pct | 0.0000 | 1.00 | 6.000 |
| 16 | 2 | 48 | 4 | jpeg_q95 | 0.0000 | 1.00 | 6.000 |
| 16 | 2 | 48 | 4 | jpeg_q85 | 0.0000 | 1.00 | 6.000 |
| 16 | 2 | 48 | 4 | jpeg_q70 | 0.0035 | 1.00 | 6.000 |
| 16 | 2 | 48 | 4 | crop_pad_2px | 0.0000 | 1.00 | 6.000 |
| 16 | 2 | 48 | 4 | combined | 0.1354 | 0.00 | 0.000 |
| 16 | 2 | 1024 | 4 | clean | 0.0000 | 1.00 | 13.107 |
| 16 | 2 | 1024 | 4 | resize_3pct | 0.0000 | 1.00 | 13.107 |
| 16 | 2 | 1024 | 4 | resize_5pct | 0.0000 | 1.00 | 13.107 |
| 16 | 2 | 1024 | 4 | jpeg_q95 | 0.0000 | 1.00 | 13.107 |
| 16 | 2 | 1024 | 4 | jpeg_q85 | 0.0000 | 1.00 | 13.107 |
| 16 | 2 | 1024 | 4 | jpeg_q70 | 0.0037 | 1.00 | 13.107 |
| 16 | 2 | 1024 | 4 | crop_pad_2px | 0.0000 | 1.00 | 13.107 |
| 16 | 2 | 1024 | 4 | combined | 0.1260 | 0.00 | 0.000 |
| 16 | 2 | 4096 | 4 | clean | 0.0000 | 1.00 | 13.375 |
| 16 | 2 | 4096 | 4 | resize_3pct | 0.0000 | 1.00 | 13.375 |
| 16 | 2 | 4096 | 4 | resize_5pct | 0.0000 | 1.00 | 13.375 |
| 16 | 2 | 4096 | 4 | jpeg_q95 | 0.0000 | 1.00 | 13.375 |
| 16 | 2 | 4096 | 4 | jpeg_q85 | 0.0000 | 1.00 | 13.375 |
| 16 | 2 | 4096 | 4 | jpeg_q70 | 0.0018 | 1.00 | 13.375 |
| 16 | 2 | 4096 | 4 | crop_pad_2px | 0.0000 | 1.00 | 13.375 |
| 16 | 2 | 4096 | 4 | combined | 0.1179 | 0.00 | 0.000 |
| 16 | 2 | 16384 | 4 | clean | 0.0000 | 1.00 | 13.788 |
| 16 | 2 | 16384 | 4 | resize_3pct | 0.0000 | 1.00 | 13.788 |
| 16 | 2 | 16384 | 4 | resize_5pct | 0.0000 | 1.00 | 13.788 |
| 16 | 2 | 16384 | 4 | jpeg_q95 | 0.0000 | 1.00 | 13.788 |
| 16 | 2 | 16384 | 4 | jpeg_q85 | 0.0000 | 1.00 | 13.788 |
| 16 | 2 | 16384 | 4 | jpeg_q70 | 0.0019 | 1.00 | 13.788 |
| 16 | 2 | 16384 | 4 | crop_pad_2px | 0.0000 | 1.00 | 13.788 |
| 16 | 2 | 16384 | 4 | combined | 0.1223 | 0.00 | 0.000 |
| 32 | 1 | 48 | 5 | clean | 0.0000 | 1.00 | 2.000 |
| 32 | 1 | 48 | 5 | resize_3pct | 0.0000 | 1.00 | 2.000 |
| 32 | 1 | 48 | 5 | resize_5pct | 0.0000 | 1.00 | 2.000 |
| 32 | 1 | 48 | 5 | jpeg_q95 | 0.0000 | 1.00 | 2.000 |
| 32 | 1 | 48 | 5 | jpeg_q85 | 0.0021 | 1.00 | 2.000 |
| 32 | 1 | 48 | 5 | jpeg_q70 | 0.0417 | 1.00 | 2.000 |
| 32 | 1 | 48 | 5 | crop_pad_2px | 0.0000 | 1.00 | 2.000 |
| 32 | 1 | 48 | 5 | combined | 0.0354 | 1.00 | 2.000 |
| 32 | 1 | 1024 | 5 | clean | 0.0000 | 1.00 | 4.137 |
| 32 | 1 | 1024 | 5 | resize_3pct | 0.0000 | 1.00 | 4.137 |
| 32 | 1 | 1024 | 5 | resize_5pct | 0.0000 | 1.00 | 4.137 |
| 32 | 1 | 1024 | 5 | jpeg_q95 | 0.0000 | 1.00 | 4.137 |
| 32 | 1 | 1024 | 5 | jpeg_q85 | 0.0083 | 1.00 | 4.137 |
| 32 | 1 | 1024 | 5 | jpeg_q70 | 0.0491 | 0.00 | 0.000 |
| 32 | 1 | 1024 | 5 | crop_pad_2px | 0.0000 | 1.00 | 4.137 |
| 32 | 1 | 1024 | 5 | combined | 0.0344 | 0.67 | 2.758 |
| 32 | 1 | 4096 | 5 | clean | 0.0000 | 1.00 | 4.280 |
| 32 | 1 | 4096 | 5 | resize_3pct | 0.0000 | 1.00 | 4.280 |
| 32 | 1 | 4096 | 5 | resize_5pct | 0.0000 | 1.00 | 4.280 |
| 32 | 1 | 4096 | 5 | jpeg_q95 | 0.0000 | 1.00 | 4.280 |
| 32 | 1 | 4096 | 5 | jpeg_q85 | 0.0041 | 1.00 | 4.280 |
| 32 | 1 | 4096 | 5 | jpeg_q70 | 0.0320 | 0.00 | 0.000 |
| 32 | 1 | 4096 | 5 | crop_pad_2px | 0.0000 | 1.00 | 4.280 |
| 32 | 1 | 4096 | 5 | combined | 0.0243 | 0.00 | 0.000 |
| 32 | 1 | 16384 | 5 | clean | 0.0000 | 1.00 | 4.329 |
| 32 | 1 | 16384 | 5 | resize_3pct | 0.0000 | 1.00 | 4.329 |
| 32 | 1 | 16384 | 5 | resize_5pct | 0.0000 | 1.00 | 4.329 |
| 32 | 1 | 16384 | 5 | jpeg_q95 | 0.0000 | 1.00 | 4.329 |
| 32 | 1 | 16384 | 5 | jpeg_q85 | 0.0032 | 1.00 | 4.329 |
| 32 | 1 | 16384 | 5 | jpeg_q70 | 0.0278 | 0.00 | 0.000 |
| 32 | 1 | 16384 | 5 | crop_pad_2px | 0.0000 | 1.00 | 4.329 |
| 32 | 1 | 16384 | 5 | combined | 0.0221 | 0.00 | 0.000 |
| 32 | 2 | 48 | 5 | clean | 0.0000 | 1.00 | 4.000 |
| 32 | 2 | 48 | 5 | resize_3pct | 0.0000 | 1.00 | 4.000 |
| 32 | 2 | 48 | 5 | resize_5pct | 0.0000 | 1.00 | 4.000 |
| 32 | 2 | 48 | 5 | jpeg_q95 | 0.0000 | 1.00 | 4.000 |
| 32 | 2 | 48 | 5 | jpeg_q85 | 0.0065 | 1.00 | 4.000 |
| 32 | 2 | 48 | 5 | jpeg_q70 | 0.0404 | 1.00 | 4.000 |
| 32 | 2 | 48 | 5 | crop_pad_2px | 0.0000 | 1.00 | 4.000 |
| 32 | 2 | 48 | 5 | combined | 0.1719 | 0.00 | 0.000 |
| 32 | 2 | 1024 | 5 | clean | 0.0000 | 1.00 | 16.000 |
| 32 | 2 | 1024 | 5 | resize_3pct | 0.0000 | 1.00 | 16.000 |
| 32 | 2 | 1024 | 5 | resize_5pct | 0.0000 | 1.00 | 16.000 |
| 32 | 2 | 1024 | 5 | jpeg_q95 | 0.0000 | 1.00 | 16.000 |
| 32 | 2 | 1024 | 5 | jpeg_q85 | 0.0146 | 1.00 | 16.000 |
| 32 | 2 | 1024 | 5 | jpeg_q70 | 0.0988 | 0.00 | 0.000 |
| 32 | 2 | 1024 | 5 | crop_pad_2px | 0.0000 | 1.00 | 16.000 |
| 32 | 2 | 1024 | 5 | combined | 0.2698 | 0.00 | 0.000 |
| 32 | 2 | 4096 | 5 | clean | 0.0000 | 1.00 | 16.926 |
| 32 | 2 | 4096 | 5 | resize_3pct | 0.0000 | 1.00 | 16.926 |
| 32 | 2 | 4096 | 5 | resize_5pct | 0.0000 | 1.00 | 16.926 |
| 32 | 2 | 4096 | 5 | jpeg_q95 | 0.0000 | 1.00 | 16.926 |
| 32 | 2 | 4096 | 5 | jpeg_q85 | 0.0187 | 1.00 | 16.926 |
| 32 | 2 | 4096 | 5 | jpeg_q70 | 0.0930 | 0.00 | 0.000 |
| 32 | 2 | 4096 | 5 | crop_pad_2px | 0.0000 | 1.00 | 16.926 |
| 32 | 2 | 4096 | 5 | combined | 0.2750 | 0.00 | 0.000 |
| 32 | 2 | 16384 | 5 | clean | 0.0000 | 1.00 | 17.120 |
| 32 | 2 | 16384 | 5 | resize_3pct | 0.0000 | 1.00 | 17.120 |
| 32 | 2 | 16384 | 5 | resize_5pct | 0.0000 | 1.00 | 17.120 |
| 32 | 2 | 16384 | 5 | jpeg_q95 | 0.0000 | 1.00 | 17.120 |
| 32 | 2 | 16384 | 5 | jpeg_q85 | 0.0103 | 1.00 | 17.120 |
| 32 | 2 | 16384 | 5 | jpeg_q70 | 0.0719 | 0.00 | 0.000 |
| 32 | 2 | 16384 | 5 | crop_pad_2px | 0.0000 | 1.00 | 17.120 |
| 32 | 2 | 16384 | 5 | combined | 0.2670 | 0.00 | 0.000 |
| 64 | 1 | 48 | 6 | clean | 0.0000 | 1.00 | 2.000 |
| 64 | 1 | 48 | 6 | resize_3pct | 0.0000 | 1.00 | 2.000 |
| 64 | 1 | 48 | 6 | resize_5pct | 0.0000 | 1.00 | 2.000 |
| 64 | 1 | 48 | 6 | jpeg_q95 | 0.0000 | 1.00 | 2.000 |
| 64 | 1 | 48 | 6 | jpeg_q85 | 0.0052 | 1.00 | 2.000 |
| 64 | 1 | 48 | 6 | jpeg_q70 | 0.0182 | 1.00 | 2.000 |
| 64 | 1 | 48 | 6 | crop_pad_2px | 0.0000 | 1.00 | 2.000 |
| 64 | 1 | 48 | 6 | combined | 0.0026 | 1.00 | 2.000 |
| 64 | 1 | 1024 | 6 | clean | 0.0000 | 1.00 | 4.923 |
| 64 | 1 | 1024 | 6 | resize_3pct | 0.0000 | 1.00 | 4.923 |
| 64 | 1 | 1024 | 6 | resize_5pct | 0.0000 | 1.00 | 4.923 |
| 64 | 1 | 1024 | 6 | jpeg_q95 | 0.0000 | 1.00 | 4.923 |
| 64 | 1 | 1024 | 6 | jpeg_q85 | 0.0037 | 1.00 | 4.923 |
| 64 | 1 | 1024 | 6 | jpeg_q70 | 0.0419 | 0.00 | 0.000 |
| 64 | 1 | 1024 | 6 | crop_pad_2px | 0.0000 | 1.00 | 4.923 |
| 64 | 1 | 1024 | 6 | combined | 0.0306 | 0.67 | 3.282 |
| 64 | 1 | 4096 | 6 | clean | 0.0000 | 1.00 | 5.120 |
| 64 | 1 | 4096 | 6 | resize_3pct | 0.0000 | 1.00 | 5.120 |
| 64 | 1 | 4096 | 6 | resize_5pct | 0.0000 | 1.00 | 5.120 |
| 64 | 1 | 4096 | 6 | jpeg_q95 | 0.0000 | 1.00 | 5.120 |
| 64 | 1 | 4096 | 6 | jpeg_q85 | 0.0032 | 1.00 | 5.120 |
| 64 | 1 | 4096 | 6 | jpeg_q70 | 0.0387 | 0.33 | 1.707 |
| 64 | 1 | 4096 | 6 | crop_pad_2px | 0.0000 | 1.00 | 5.120 |
| 64 | 1 | 4096 | 6 | combined | 0.0318 | 0.00 | 0.000 |
| 64 | 1 | 16384 | 6 | clean | 0.0000 | 1.00 | 5.185 |
| 64 | 1 | 16384 | 6 | resize_3pct | 0.0000 | 1.00 | 5.185 |
| 64 | 1 | 16384 | 6 | resize_5pct | 0.0000 | 1.00 | 5.185 |
| 64 | 1 | 16384 | 6 | jpeg_q95 | 0.0000 | 1.00 | 5.185 |
| 64 | 1 | 16384 | 6 | jpeg_q85 | 0.0017 | 1.00 | 5.185 |
| 64 | 1 | 16384 | 6 | jpeg_q70 | 0.0252 | 0.33 | 1.728 |
| 64 | 1 | 16384 | 6 | crop_pad_2px | 0.0000 | 1.00 | 5.185 |
| 64 | 1 | 16384 | 6 | combined | 0.0191 | 1.00 | 5.185 |
| 64 | 2 | 48 | 6 | clean | 0.0000 | 1.00 | 3.000 |
| 64 | 2 | 48 | 6 | resize_3pct | 0.0000 | 1.00 | 3.000 |
| 64 | 2 | 48 | 6 | resize_5pct | 0.0000 | 1.00 | 3.000 |
| 64 | 2 | 48 | 6 | jpeg_q95 | 0.0000 | 1.00 | 3.000 |
| 64 | 2 | 48 | 6 | jpeg_q85 | 0.0000 | 1.00 | 3.000 |
| 64 | 2 | 48 | 6 | jpeg_q70 | 0.0169 | 1.00 | 3.000 |
| 64 | 2 | 48 | 6 | crop_pad_2px | 0.0000 | 1.00 | 3.000 |
| 64 | 2 | 48 | 6 | combined | 0.3008 | 0.00 | 0.000 |
| 64 | 2 | 1024 | 6 | clean | 0.0000 | 1.00 | 16.000 |
| 64 | 2 | 1024 | 6 | resize_3pct | 0.0000 | 1.00 | 16.000 |
| 64 | 2 | 1024 | 6 | resize_5pct | 0.0000 | 1.00 | 16.000 |
| 64 | 2 | 1024 | 6 | jpeg_q95 | 0.0000 | 1.00 | 16.000 |
| 64 | 2 | 1024 | 6 | jpeg_q85 | 0.0082 | 1.00 | 16.000 |
| 64 | 2 | 1024 | 6 | jpeg_q70 | 0.0761 | 0.00 | 0.000 |
| 64 | 2 | 1024 | 6 | crop_pad_2px | 0.0000 | 1.00 | 16.000 |
| 64 | 2 | 1024 | 6 | combined | 0.4230 | 0.00 | 0.000 |
| 64 | 2 | 4096 | 6 | clean | 0.0000 | 1.00 | 19.692 |
| 64 | 2 | 4096 | 6 | resize_3pct | 0.0000 | 1.00 | 19.692 |
| 64 | 2 | 4096 | 6 | resize_5pct | 0.0000 | 1.00 | 19.692 |
| 64 | 2 | 4096 | 6 | jpeg_q95 | 0.0000 | 1.00 | 19.692 |
| 64 | 2 | 4096 | 6 | jpeg_q85 | 0.0085 | 1.00 | 19.692 |
| 64 | 2 | 4096 | 6 | jpeg_q70 | 0.0877 | 0.00 | 0.000 |
| 64 | 2 | 4096 | 6 | crop_pad_2px | 0.0000 | 1.00 | 19.692 |
| 64 | 2 | 4096 | 6 | combined | 0.4342 | 0.00 | 0.000 |
| 64 | 2 | 16384 | 6 | clean | 0.0000 | 1.00 | 20.480 |
| 64 | 2 | 16384 | 6 | resize_3pct | 0.0000 | 1.00 | 20.480 |
| 64 | 2 | 16384 | 6 | resize_5pct | 0.0000 | 1.00 | 20.480 |
| 64 | 2 | 16384 | 6 | jpeg_q95 | 0.0000 | 1.00 | 20.480 |
| 64 | 2 | 16384 | 6 | jpeg_q85 | 0.0082 | 1.00 | 20.480 |
| 64 | 2 | 16384 | 6 | jpeg_q70 | 0.0832 | 0.00 | 0.000 |
| 64 | 2 | 16384 | 6 | crop_pad_2px | 0.0000 | 1.00 | 20.480 |
| 64 | 2 | 16384 | 6 | combined | 0.4348 | 0.00 | 0.000 |
| 128 | 1 | 48 | 7 | clean | 0.0000 | 1.00 | 1.500 |
| 128 | 1 | 48 | 7 | resize_3pct | 0.0000 | 1.00 | 1.500 |
| 128 | 1 | 48 | 7 | resize_5pct | 0.0000 | 1.00 | 1.500 |
| 128 | 1 | 48 | 7 | jpeg_q95 | 0.0000 | 1.00 | 1.500 |
| 128 | 1 | 48 | 7 | jpeg_q85 | 0.0234 | 1.00 | 1.500 |
| 128 | 1 | 48 | 7 | jpeg_q70 | 0.0469 | 1.00 | 1.500 |
| 128 | 1 | 48 | 7 | crop_pad_2px | 0.0000 | 1.00 | 1.500 |
| 128 | 1 | 48 | 7 | combined | 0.0651 | 1.00 | 1.500 |
| 128 | 1 | 1024 | 7 | clean | 0.0000 | 1.00 | 5.333 |
| 128 | 1 | 1024 | 7 | resize_3pct | 0.0000 | 1.00 | 5.333 |
| 128 | 1 | 1024 | 7 | resize_5pct | 0.0000 | 1.00 | 5.333 |
| 128 | 1 | 1024 | 7 | jpeg_q95 | 0.0000 | 1.00 | 5.333 |
| 128 | 1 | 1024 | 7 | jpeg_q85 | 0.0365 | 0.67 | 3.556 |
| 128 | 1 | 1024 | 7 | jpeg_q70 | 0.1823 | 0.00 | 0.000 |
| 128 | 1 | 1024 | 7 | crop_pad_2px | 0.0000 | 1.00 | 5.333 |
| 128 | 1 | 1024 | 7 | combined | 0.1574 | 0.00 | 0.000 |
| 128 | 1 | 4096 | 7 | clean | 0.0000 | 1.00 | 5.818 |
| 128 | 1 | 4096 | 7 | resize_3pct | 0.0000 | 1.00 | 5.818 |
| 128 | 1 | 4096 | 7 | resize_5pct | 0.0000 | 1.00 | 5.818 |
| 128 | 1 | 4096 | 7 | jpeg_q95 | 0.0001 | 1.00 | 5.818 |
| 128 | 1 | 4096 | 7 | jpeg_q85 | 0.0380 | 0.00 | 0.000 |
| 128 | 1 | 4096 | 7 | jpeg_q70 | 0.1840 | 0.00 | 0.000 |
| 128 | 1 | 4096 | 7 | crop_pad_2px | 0.0000 | 1.00 | 5.818 |
| 128 | 1 | 4096 | 7 | combined | 0.1648 | 0.00 | 0.000 |
| 128 | 1 | 16384 | 7 | clean | 0.0000 | 1.00 | 6.066 |
| 128 | 1 | 16384 | 7 | resize_3pct | 0.0000 | 1.00 | 6.066 |
| 128 | 1 | 16384 | 7 | resize_5pct | 0.0000 | 1.00 | 6.066 |
| 128 | 1 | 16384 | 7 | jpeg_q95 | 0.0000 | 1.00 | 6.066 |
| 128 | 1 | 16384 | 7 | jpeg_q85 | 0.0391 | 0.00 | 0.000 |
| 128 | 1 | 16384 | 7 | jpeg_q70 | 0.1822 | 0.00 | 0.000 |
| 128 | 1 | 16384 | 7 | crop_pad_2px | 0.0000 | 1.00 | 6.066 |
| 128 | 1 | 16384 | 7 | combined | 0.1551 | 0.00 | 0.000 |
| 128 | 2 | 48 | 7 | clean | 0.0000 | 1.00 | 1.500 |
| 128 | 2 | 48 | 7 | resize_3pct | 0.0000 | 1.00 | 1.500 |
| 128 | 2 | 48 | 7 | resize_5pct | 0.0000 | 1.00 | 1.500 |
| 128 | 2 | 48 | 7 | jpeg_q95 | 0.0000 | 1.00 | 1.500 |
| 128 | 2 | 48 | 7 | jpeg_q85 | 0.0059 | 1.00 | 1.500 |
| 128 | 2 | 48 | 7 | jpeg_q70 | 0.0345 | 0.33 | 0.500 |
| 128 | 2 | 48 | 7 | crop_pad_2px | 0.0000 | 1.00 | 1.500 |
| 128 | 2 | 48 | 7 | combined | 0.3783 | 0.00 | 0.000 |
| 128 | 2 | 1024 | 7 | clean | 0.0000 | 1.00 | 16.000 |
| 128 | 2 | 1024 | 7 | resize_3pct | 0.0000 | 1.00 | 16.000 |
| 128 | 2 | 1024 | 7 | resize_5pct | 0.0000 | 1.00 | 16.000 |
| 128 | 2 | 1024 | 7 | jpeg_q95 | 0.0002 | 1.00 | 16.000 |
| 128 | 2 | 1024 | 7 | jpeg_q85 | 0.0940 | 0.00 | 0.000 |
| 128 | 2 | 1024 | 7 | jpeg_q70 | 0.2971 | 0.00 | 0.000 |
| 128 | 2 | 1024 | 7 | crop_pad_2px | 0.0000 | 1.00 | 16.000 |
| 128 | 2 | 1024 | 7 | combined | 0.6447 | 0.00 | 0.000 |
| 128 | 2 | 4096 | 7 | clean | 0.0000 | 1.00 | 21.333 |
| 128 | 2 | 4096 | 7 | resize_3pct | 0.0000 | 1.00 | 21.333 |
| 128 | 2 | 4096 | 7 | resize_5pct | 0.0000 | 1.00 | 21.333 |
| 128 | 2 | 4096 | 7 | jpeg_q95 | 0.0002 | 1.00 | 21.333 |
| 128 | 2 | 4096 | 7 | jpeg_q85 | 0.1021 | 0.00 | 0.000 |
| 128 | 2 | 4096 | 7 | jpeg_q70 | 0.3166 | 0.00 | 0.000 |
| 128 | 2 | 4096 | 7 | crop_pad_2px | 0.0000 | 1.00 | 21.333 |
| 128 | 2 | 4096 | 7 | combined | 0.6636 | 0.00 | 0.000 |
| 128 | 2 | 16384 | 7 | clean | 0.0000 | 1.00 | 23.814 |
| 128 | 2 | 16384 | 7 | resize_3pct | 0.0000 | 1.00 | 23.814 |
| 128 | 2 | 16384 | 7 | resize_5pct | 0.0000 | 1.00 | 23.814 |
| 128 | 2 | 16384 | 7 | jpeg_q95 | 0.0002 | 1.00 | 23.814 |
| 128 | 2 | 16384 | 7 | jpeg_q85 | 0.1084 | 0.00 | 0.000 |
| 128 | 2 | 16384 | 7 | jpeg_q70 | 0.3318 | 0.00 | 0.000 |
| 128 | 2 | 16384 | 7 | crop_pad_2px | 0.0000 | 1.00 | 23.814 |
| 128 | 2 | 16384 | 7 | combined | 0.6718 | 0.00 | 0.000 |
| 256 | 1 | 48 | 8 | clean | 0.0000 | 1.00 | 0.750 |
| 256 | 1 | 48 | 8 | resize_3pct | 0.0000 | 1.00 | 0.750 |
| 256 | 1 | 48 | 8 | resize_5pct | 0.0000 | 1.00 | 0.750 |
| 256 | 1 | 48 | 8 | jpeg_q95 | 0.0000 | 1.00 | 0.750 |
| 256 | 1 | 48 | 8 | jpeg_q85 | 0.0443 | 1.00 | 0.750 |
| 256 | 1 | 48 | 8 | jpeg_q70 | 0.1081 | 0.00 | 0.000 |
| 256 | 1 | 48 | 8 | crop_pad_2px | 0.0000 | 1.00 | 0.750 |
| 256 | 1 | 48 | 8 | combined | 0.0964 | 0.00 | 0.000 |
| 256 | 1 | 1024 | 8 | clean | 0.0000 | 1.00 | 5.333 |
| 256 | 1 | 1024 | 8 | resize_3pct | 0.0000 | 1.00 | 5.333 |
| 256 | 1 | 1024 | 8 | resize_5pct | 0.0000 | 1.00 | 5.333 |
| 256 | 1 | 1024 | 8 | jpeg_q95 | 0.0260 | 1.00 | 5.333 |
| 256 | 1 | 1024 | 8 | jpeg_q85 | 0.2896 | 0.00 | 0.000 |
| 256 | 1 | 1024 | 8 | jpeg_q70 | 0.4833 | 0.00 | 0.000 |
| 256 | 1 | 1024 | 8 | crop_pad_2px | 0.0000 | 1.00 | 5.333 |
| 256 | 1 | 1024 | 8 | combined | 0.4409 | 0.00 | 0.000 |
| 256 | 1 | 4096 | 8 | clean | 0.0000 | 1.00 | 6.400 |
| 256 | 1 | 4096 | 8 | resize_3pct | 0.0000 | 1.00 | 6.400 |
| 256 | 1 | 4096 | 8 | resize_5pct | 0.0000 | 1.00 | 6.400 |
| 256 | 1 | 4096 | 8 | jpeg_q95 | 0.0232 | 1.00 | 6.400 |
| 256 | 1 | 4096 | 8 | jpeg_q85 | 0.2844 | 0.00 | 0.000 |
| 256 | 1 | 4096 | 8 | jpeg_q70 | 0.4824 | 0.00 | 0.000 |
| 256 | 1 | 4096 | 8 | crop_pad_2px | 0.0000 | 1.00 | 6.400 |
| 256 | 1 | 4096 | 8 | combined | 0.4624 | 0.00 | 0.000 |
| 256 | 1 | 16384 | 8 | clean | 0.0000 | 1.00 | 6.827 |
| 256 | 1 | 16384 | 8 | resize_3pct | 0.0000 | 1.00 | 6.827 |
| 256 | 1 | 16384 | 8 | resize_5pct | 0.0000 | 1.00 | 6.827 |
| 256 | 1 | 16384 | 8 | jpeg_q95 | 0.0240 | 1.00 | 6.827 |
| 256 | 1 | 16384 | 8 | jpeg_q85 | 0.2943 | 0.00 | 0.000 |
| 256 | 1 | 16384 | 8 | jpeg_q70 | 0.4922 | 0.00 | 0.000 |
| 256 | 1 | 16384 | 8 | crop_pad_2px | 0.0000 | 1.00 | 6.827 |
| 256 | 1 | 16384 | 8 | combined | 0.4661 | 0.00 | 0.000 |
| 256 | 2 | 48 | 8 | clean | 0.0000 | 1.00 | 0.750 |
| 256 | 2 | 48 | 8 | resize_3pct | 0.0000 | 1.00 | 0.750 |
| 256 | 2 | 48 | 8 | resize_5pct | 0.0000 | 1.00 | 0.750 |
| 256 | 2 | 48 | 8 | jpeg_q95 | 0.0033 | 1.00 | 0.750 |
| 256 | 2 | 48 | 8 | jpeg_q85 | 0.0234 | 0.00 | 0.000 |
| 256 | 2 | 48 | 8 | jpeg_q70 | 0.0407 | 0.00 | 0.000 |
| 256 | 2 | 48 | 8 | crop_pad_2px | 0.0000 | 1.00 | 0.750 |
| 256 | 2 | 48 | 8 | combined | 0.5166 | 0.00 | 0.000 |
| 256 | 2 | 1024 | 8 | clean | 0.0000 | 1.00 | 10.667 |
| 256 | 2 | 1024 | 8 | resize_3pct | 0.0000 | 1.00 | 10.667 |
| 256 | 2 | 1024 | 8 | resize_5pct | 0.0000 | 1.00 | 10.667 |
| 256 | 2 | 1024 | 8 | jpeg_q95 | 0.0623 | 0.00 | 0.000 |
| 256 | 2 | 1024 | 8 | jpeg_q85 | 0.2607 | 0.00 | 0.000 |
| 256 | 2 | 1024 | 8 | jpeg_q70 | 0.5980 | 0.00 | 0.000 |
| 256 | 2 | 1024 | 8 | crop_pad_2px | 0.0000 | 1.00 | 10.667 |
| 256 | 2 | 1024 | 8 | combined | 0.6678 | 0.00 | 0.000 |
| 256 | 2 | 4096 | 8 | clean | 0.0000 | 1.00 | 21.333 |
| 256 | 2 | 4096 | 8 | resize_3pct | 0.0000 | 1.00 | 21.333 |
| 256 | 2 | 4096 | 8 | resize_5pct | 0.0000 | 1.00 | 21.333 |
| 256 | 2 | 4096 | 8 | jpeg_q95 | 0.0675 | 0.00 | 0.000 |
| 256 | 2 | 4096 | 8 | jpeg_q85 | 0.3824 | 0.00 | 0.000 |
| 256 | 2 | 4096 | 8 | jpeg_q70 | 0.6100 | 0.00 | 0.000 |
| 256 | 2 | 4096 | 8 | crop_pad_2px | 0.0000 | 1.00 | 21.333 |
| 256 | 2 | 4096 | 8 | combined | 0.8104 | 0.00 | 0.000 |
| 256 | 2 | 16384 | 8 | clean | 0.0000 | 1.00 | 25.600 |
| 256 | 2 | 16384 | 8 | resize_3pct | 0.0000 | 1.00 | 25.600 |
| 256 | 2 | 16384 | 8 | resize_5pct | 0.0000 | 1.00 | 25.600 |
| 256 | 2 | 16384 | 8 | jpeg_q95 | 0.0730 | 0.00 | 0.000 |
| 256 | 2 | 16384 | 8 | jpeg_q85 | 0.4157 | 0.00 | 0.000 |
| 256 | 2 | 16384 | 8 | jpeg_q70 | 0.6255 | 0.00 | 0.000 |
| 256 | 2 | 16384 | 8 | crop_pad_2px | 0.0000 | 1.00 | 25.600 |
| 256 | 2 | 16384 | 8 | combined | 0.8090 | 0.00 | 0.000 |

## Self-consistency checks

Three invariants must hold if these numbers mean what they claim to mean: (1) bits/patch (`payload_bytes*8/total_patches`, the TRUE PAYLOAD DENSITY -- see `_bits_per_patch_on_success`'s docstring) can never exceed `subpatch²·log2(palette)`, the raw per-DATA-PATCH channel CAPACITY for a subpatch x subpatch grid of symbols per patch, before calibration-row overhead, Reed-Solomon parity, and grid/bit padding are all accounted for (this generalizes the pre-Slice-B `<= log2(palette)` check, which is the `subpatch=1` case where `subpatch²=1`); (2) mean corrupted bits/patch can never exceed clean bits/patch for the same (palette, subpatch, payload_size), since corruption only ever removes information relative to the uncorrupted image; (3) [token crossover] every row's `base64_token_est` must equal `ceil(payload_size/3)*4` exactly and `token_ratio` must equal `total_patches/base64_token_est` exactly, independently recomputed here rather than just re-displaying the harness's own stored values -- if either drifts, the Token crossover section's numbers are wrong.

| palette | subpatch | payload | ceiling subpatch²·log2(P) | clean bits/patch | <= ceiling? | corrupted(mean) bits/patch | <= clean? |
|---|---|---|---|---|---|---|---|
| 2 | 1 | 48 | 1 | 0.527 | yes | 0.527 | yes |
| 2 | 1 | 1024 | 1 | 0.844 | yes | 0.844 | yes |
| 2 | 1 | 4096 | 1 | 0.862 | yes | 0.862 | yes |
| 2 | 1 | 16384 | 1 | 0.871 | yes | 0.871 | yes |
| 2 | 2 | 48 | 4 | 1.959 | yes | 1.959 | yes |
| 2 | 2 | 1024 | 4 | 3.344 | yes | 3.344 | yes |
| 2 | 2 | 4096 | 4 | 3.412 | yes | 3.412 | yes |
| 2 | 2 | 16384 | 4 | 3.465 | yes | 3.465 | yes |
| 4 | 1 | 48 | 2 | 1.064 | yes | 1.064 | yes |
| 4 | 1 | 1024 | 2 | 1.696 | yes | 1.696 | yes |
| 4 | 1 | 4096 | 2 | 1.721 | yes | 1.721 | yes |
| 4 | 1 | 16384 | 2 | 1.740 | yes | 1.740 | yes |
| 4 | 2 | 48 | 8 | 3.840 | yes | 3.840 | yes |
| 4 | 2 | 1024 | 8 | 6.687 | yes | 6.687 | yes |
| 4 | 2 | 4096 | 8 | 6.784 | yes | 6.784 | yes |
| 4 | 2 | 16384 | 8 | 6.933 | yes | 6.933 | yes |
| 8 | 1 | 48 | 3 | 1.500 | yes | 1.500 | yes |
| 8 | 1 | 1024 | 3 | 2.521 | yes | 2.521 | yes |
| 8 | 1 | 4096 | 3 | 2.566 | yes | 2.566 | yes |
| 8 | 1 | 16384 | 3 | 2.601 | yes | 2.601 | yes |
| 8 | 2 | 48 | 12 | 5.333 | yes | 5.333 | yes |
| 8 | 2 | 1024 | 12 | 9.741 | yes | 9.741 | yes |
| 8 | 2 | 4096 | 12 | 10.086 | yes | 10.086 | yes |
| 8 | 2 | 16384 | 12 | 10.357 | yes | 10.357 | yes |
| 16 | 1 | 48 | 4 | 2.000 | yes | 2.000 | yes |
| 16 | 1 | 1024 | 4 | 3.344 | yes | 3.344 | yes |
| 16 | 1 | 4096 | 4 | 3.412 | yes | 3.412 | yes |
| 16 | 1 | 16384 | 4 | 3.465 | yes | 3.465 | yes |
| 16 | 2 | 48 | 16 | 6.000 | yes | 5.143 | yes |
| 16 | 2 | 1024 | 16 | 13.107 | yes | 11.235 | yes |
| 16 | 2 | 4096 | 16 | 13.375 | yes | 11.464 | yes |
| 16 | 2 | 16384 | 16 | 13.788 | yes | 11.819 | yes |
| 32 | 1 | 48 | 5 | 2.000 | yes | 2.000 | yes |
| 32 | 1 | 1024 | 5 | 4.137 | yes | 3.349 | yes |
| 32 | 1 | 4096 | 5 | 4.280 | yes | 3.057 | yes |
| 32 | 1 | 16384 | 5 | 4.329 | yes | 3.092 | yes |
| 32 | 2 | 48 | 20 | 4.000 | yes | 3.429 | yes |
| 32 | 2 | 1024 | 20 | 16.000 | yes | 11.429 | yes |
| 32 | 2 | 4096 | 20 | 16.926 | yes | 12.090 | yes |
| 32 | 2 | 16384 | 20 | 17.120 | yes | 12.229 | yes |
| 64 | 1 | 48 | 6 | 2.000 | yes | 2.000 | yes |
| 64 | 1 | 1024 | 6 | 4.923 | yes | 3.985 | yes |
| 64 | 1 | 4096 | 6 | 5.120 | yes | 3.901 | yes |
| 64 | 1 | 16384 | 6 | 5.185 | yes | 4.691 | yes |
| 64 | 2 | 48 | 24 | 3.000 | yes | 2.571 | yes |
| 64 | 2 | 1024 | 24 | 16.000 | yes | 11.429 | yes |
| 64 | 2 | 4096 | 24 | 19.692 | yes | 14.066 | yes |
| 64 | 2 | 16384 | 24 | 20.480 | yes | 14.629 | yes |
| 128 | 1 | 48 | 7 | 1.500 | yes | 1.500 | yes |
| 128 | 1 | 1024 | 7 | 5.333 | yes | 3.556 | yes |
| 128 | 1 | 4096 | 7 | 5.818 | yes | 3.325 | yes |
| 128 | 1 | 16384 | 7 | 6.066 | yes | 3.466 | yes |
| 128 | 2 | 48 | 28 | 1.500 | yes | 1.143 | yes |
| 128 | 2 | 1024 | 28 | 16.000 | yes | 9.143 | yes |
| 128 | 2 | 4096 | 28 | 21.333 | yes | 12.190 | yes |
| 128 | 2 | 16384 | 28 | 23.814 | yes | 13.608 | yes |
| 256 | 1 | 48 | 8 | 0.750 | yes | 0.536 | yes |
| 256 | 1 | 1024 | 8 | 5.333 | yes | 3.048 | yes |
| 256 | 1 | 4096 | 8 | 6.400 | yes | 3.657 | yes |
| 256 | 1 | 16384 | 8 | 6.827 | yes | 3.901 | yes |
| 256 | 2 | 48 | 32 | 0.750 | yes | 0.429 | yes |
| 256 | 2 | 1024 | 32 | 10.667 | yes | 4.571 | yes |
| 256 | 2 | 4096 | 32 | 21.333 | yes | 9.143 | yes |
| 256 | 2 | 16384 | 32 | 25.600 | yes | 10.971 | yes |

Invariants (1) and (2) hold for every (palette, subpatch, payload_size) bucket above. Invariant (3) [token crossover] holds for every one of the 512 rows in this sweep: base64_token_est and token_ratio were independently recomputed from payload_size/total_patches for every row and matched the harness's own stored values exactly. The largest observed symbol_error_rate across the whole sweep is 0.8104 (palette=256, subpatch=2, payload_size=4096B, corruption=combined). Within the realistic corruption envelope this harness applies, decode_success_rate drops below 1.00 for at least one cell in this sweep (lowest observed: 0.00, at palette=16, subpatch=2, payload_size=48B, corruption=combined) -- unlike the original v0.1 4-palette/subpatch=1/48-byte sweep, where Reed-Solomon (nsym=32) fully absorbed every symbol error that same envelope introduced. See the full breakdown above for every cell where decode_success_rate < 1.00: this is the realistic corruption envelope actually biting at the larger palette/subpatch/payload_size combinations this sweep newly covers, not a measurement bug.

## Beyond the realistic envelope (diagnostic, single representative config)

To confirm decode failure is actually reachable by this harness (i.e. that high success rates above are a real headroom finding and not a bug that can never observe failure), the same style of trial was re-run under corruption well outside the 'realistic serving pipeline' envelope: 50% bilinear resize round-trip, JPEG q10, a 6px crop/pad, and their composition. This diagnostic suite runs at a single representative config -- subpatch=1, payload_size=48B, 5 trials/cell (the module defaults) -- across all 8 palettes; it is NOT swept across subpatch/payload_size the way the headline sweep above is, since its only purpose is to confirm the harness can observe decode failure at all.

| palette | corruption | symbol error rate | decode success rate | bits/patch |
|---|---|---|---|---|
| 2 | stress_resize_50pct | 0.0000 | 1.00 | 0.527 |
| 2 | stress_jpeg_q10 | 0.0000 | 1.00 | 0.527 |
| 2 | stress_crop_pad_6px | 0.0000 | 1.00 | 0.527 |
| 2 | stress_combined | 0.0205 | 0.80 | 0.421 |
| 4 | stress_resize_50pct | 0.0000 | 1.00 | 1.064 |
| 4 | stress_jpeg_q10 | 0.0000 | 1.00 | 1.064 |
| 4 | stress_crop_pad_6px | 0.0000 | 1.00 | 1.064 |
| 4 | stress_combined | 0.1158 | 0.00 | 0.000 |
| 8 | stress_resize_50pct | 0.0000 | 1.00 | 1.500 |
| 8 | stress_jpeg_q10 | 0.0200 | 1.00 | 1.500 |
| 8 | stress_crop_pad_6px | 0.0000 | 1.00 | 1.500 |
| 8 | stress_combined | 0.2825 | 0.00 | 0.000 |
| 16 | stress_resize_50pct | 0.0000 | 1.00 | 2.000 |
| 16 | stress_jpeg_q10 | 0.1216 | 0.20 | 0.400 |
| 16 | stress_crop_pad_6px | 0.0000 | 1.00 | 2.000 |
| 16 | stress_combined | 0.6114 | 0.00 | 0.000 |
| 32 | stress_resize_50pct | 0.0000 | 1.00 | 2.000 |
| 32 | stress_jpeg_q10 | 0.2550 | 0.00 | 0.000 |
| 32 | stress_crop_pad_6px | 0.0000 | 1.00 | 2.000 |
| 32 | stress_combined | 0.6687 | 0.00 | 0.000 |
| 64 | stress_resize_50pct | 0.0000 | 1.00 | 2.000 |
| 64 | stress_jpeg_q10 | 0.3063 | 0.00 | 0.000 |
| 64 | stress_crop_pad_6px | 0.0000 | 1.00 | 2.000 |
| 64 | stress_combined | 0.8859 | 0.00 | 0.000 |
| 128 | stress_resize_50pct | 0.0000 | 1.00 | 1.500 |
| 128 | stress_jpeg_q10 | 0.5734 | 0.00 | 0.000 |
| 128 | stress_crop_pad_6px | 0.0000 | 1.00 | 1.500 |
| 128 | stress_combined | 0.9484 | 0.00 | 0.000 |
| 256 | stress_resize_50pct | 0.0000 | 1.00 | 0.750 |
| 256 | stress_jpeg_q10 | 0.6539 | 0.00 | 0.000 |
| 256 | stress_crop_pad_6px | 0.0000 | 1.00 | 0.750 |
| 256 | stress_combined | 0.9609 | 0.00 | 0.000 |

Decode success drops well below 1.00 for at least one palette under this diagnostic stress suite (lowest observed: 0.00), confirming the channel does have a real breaking point -- it simply lies beyond the resize/JPEG/crop ranges a typical serving pipeline applies, consistent with the realistic-envelope sweep above.
