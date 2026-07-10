# heliogram codec format v0.1

This is a mini-RFC for the wire format implemented by `heliogram/codec.py`. It is normative:
if this document and the code ever disagree, that is a bug in one of them, not a choice point.

## 1. Overview

A heliogram image is a grid of `PATCH_SIZE x PATCH_SIZE` px solid-color blocks ("patches").
Each patch encodes one symbol from a palette of `P` deterministic, separable colors, where `P`
in `{2, 4, 8, 16, 32, 64, 128, 256}` and `bits_per_symbol = log2(P)`. One patch is intended to
correspond to roughly one ViT/vision-token patch in a downstream VLM's image encoder.

Optionally (`subpatch`/`k` > 1, section 6a), each DATA patch is itself subdivided into a `k x k`
grid of solid-color sub-cells, each carrying its own symbol -- `k*k` symbols per data patch
instead of 1. `k=1` (the default) is exactly the one-symbol-per-patch layout described in this
section; everything through section 6 describes that `k=1` case unless a `subpatch` sub-section
says otherwise.

```
+----------------------------------------+
| row 0: CALIBRATION ROW                 |  patch i = palette[i % P], i = 0..width-1
+----------------------------------------+
| row 1: data patch, data patch, ...     |
| row 2: data patch, data patch, ...     |  row-major, all rows after row 0
| ...                                     |
+----------------------------------------+
```

Image pixel size = `(width * PATCH_SIZE) x (height * PATCH_SIZE)`, regardless of `subpatch` --
sub-patch packing only changes what happens *inside* each data patch, never the outer patch grid
dimensions.

## 2. Palette (P <= 16)

`get_palette(P)` returns `P` RGB triples, deterministic in `P` alone (no seed, no randomness).
For `P <= 16` (the only sizes in the original v0.1 release, and byte-for-byte UNCHANGED here):

```
for i in range(P):
    hue = i / P
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)   # full saturation, full value
    color_i = (round(r*255), round(g*255), round(b*255))
```

Evenly spaced hues at full S/V maximize pairwise RGB separation for a given `P`, which is what
the nearest-neighbor classifier in the decoder relies on. Same `P` always yields the same colors;
different `P` values are independent palettes (the `P=4` palette is not a subset of `P=8`, etc).

Reference values (as produced by the current implementation):

| P  | colors (R,G,B) |
|---:|---|
| 2  | (255,0,0), (0,255,255) |
| 4  | (255,0,0), (128,255,0), (0,255,255), (128,0,255) |
| 8  | (255,0,0), (255,191,0), (128,255,0), (0,255,64), (0,255,255), (0,64,255), (128,0,255), (255,0,191) |
| 16 | (255,0,0), (255,96,0), (255,191,0), (223,255,0), (128,255,0), (32,255,0), (0,255,64), (0,255,159), (0,255,255), (0,159,255), (0,64,255), (32,0,255), (128,0,255), (223,0,255), (255,0,191), (255,0,96) |

## 2a. Palette (P > 16: 32, 64, 128, 256)

Hue-only separation at full S/V (section 2) runs out once `P` gets past 16 colors packed around
one hue wheel -- adjacent hues get close in RGB, and this is made worse under realistic
corruption because JPEG's chroma subsampling erodes hue (chroma) differences faster than
brightness (luma) ones. So for `P > 16` (32, 64, 128, and 256 are in `VALID_PALETTES`),
`get_palette` additionally varies VALUE: the palette is `levels = P // 16` value-levels of the
same 16 hues used by the `P=16` case, `S=1.0`, value stepped evenly from `1.0` down to a floor
of `0.4`:

```
hues = 16
levels = P // hues            # 2 for P=32, 4 for P=64, 8 for P=128, 16 for P=256
for lvl in range(levels):
    value = 1.0 if levels == 1 else 1.0 - lvl * (1.0 - 0.4) / (levels - 1)
    for h in range(hues):
        hue = h / hues
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, value)
        color_{lvl*hues + h} = (round(r*255), round(g*255), round(b*255))
```

The brightest level (`lvl=0`, `value=1.0`) is therefore identical to the `P=16` hue set from
section 2. Still fully deterministic (no RNG); all `P` colors are distinct for every `P` in
`VALID_PALETTES` (`tests/test_roundtrip.py::test_palette_deterministic_and_separable` checks
this, including `get_palette(256)`). `P=128` (`levels=8`) and `P=256` (`levels=16`) reuse this
exact same branch unchanged -- only `levels` grows -- so `get_palette(P)` for every `P` already
in `VALID_PALETTES` before 128/256 were added (2, 4, 8, 16, 32, 64) is byte-for-byte identical to
before this extension (`tests/test_roundtrip.py::test_get_palette_le64_byte_identical_to_pinned_values`
pins this against SHA-256 digests captured directly from git history, not just re-derived from
the current code).

**Honesty note:** varying value in addition to hue is a *design choice* aimed at improving
separability under corruption (a second axis riding on luma, which survives JPEG's chroma
subsampling better than hue does), not a result derived from a corruption sweep in advance of
shipping it. It has since been checked against real (not synthetic-pixel-shift) corruption by
`tests/test_roundtrip.py`, with a clear split between what holds and what doesn't:

- **Mild realistic corruption (resize ±3-5%, JPEG q95, slight crop/pad):** every palette in
  `VALID_PALETTES` -- including 32, 64, 128, and 256 -- decodes successfully at the (small,
  <=512B) payload sizes `tests/test_roundtrip.py` exercises for this suite
  (`test_realistic_mild_corruption_roundtrip_all_palettes_subpatch1`,
  `test_large_palette_clean_roundtrip_exact_all_subpatches` for 128/256's own determinism/clean
  coverage). This is also confirmed by `heliogram.harness`'s own published sweep
  (`results.csv`/`RESULTS.md`), which now covers P=128/256 too (a previous version of this note
  said that sweep had not been re-run for those two palettes and treated the pytest file as the
  sole source of truth for them; it has been re-run since, and results.csv/RESULTS.md should be
  cited for P=128/256 like any other palette). **With one measured exception at larger payload
  sizes:** that same sweep shows `palette=256` failing `JPEG q95` specifically once
  payload_size >= 1024B -- `decode_success_rate` drops to 0.6667 at `subpatch=1`/4096B and to
  0.0000 at `subpatch=2`/{1024, 4096, 16384}B; every other palette/subpatch/payload_size cell
  under this same mild-corruption set in that 512-row sweep is 1.0000.
  `tests/test_roundtrip.py::test_subpatch2_palette256_known_failure_under_jpeg_q95_at_scale`
  pins that exact failure directly against the reference decoder, the same way the JPEG q70
  known-failure tests below do for their own cells -- not just a citation of `results.csv`.
- **JPEG q70 at payload_size=1024 (subpatch=1):** palette 32, 64, 128, and **256 all
  MEASURABLY FAIL** to decode on this reference pixel decoder
  (`test_palette32_64_known_failure_under_jpeg_q70_at_scale`,
  `test_palette128_256_known_failure_under_jpeg_q70_at_scale`) -- nearest-neighbor classification
  cannot separate that many colors once chroma subsampling has eroded hue differences, and the
  value-tiering axis (varying luma, which JPEG preserves better) does not rescue it at q70. This
  is a **measured failure, pinned by a test that asserts the failure**, not a hypothetical one
  glossed over as "corruption is out of scope."

Net: the value-tiering design choice measurably helps at the mild end of a realistic serving
envelope and measurably does not help against JPEG q70 specifically, for every `P > 16` in
`VALID_PALETTES` tested this way, including the newly added 128 and 256.

Reference values (as produced by the current implementation):

| P  | colors (R,G,B) |
|---:|---|
| 32 | (255,0,0), (255,96,0), (255,191,0), (223,255,0), (128,255,0), (32,255,0), (0,255,64), (0,255,159), (0,255,255), (0,159,255), (0,64,255), (32,0,255), (128,0,255), (223,0,255), (255,0,191), (255,0,96), (102,0,0), (102,38,0), (102,77,0), (89,102,0), (51,102,0), (13,102,0), (0,102,26), (0,102,64), (0,102,102), (0,64,102), (0,26,102), (13,0,102), (51,0,102), (89,0,102), (102,0,77), (102,0,38) |
| 64 | (255,0,0), (255,96,0), (255,191,0), (223,255,0), (128,255,0), (32,255,0), (0,255,64), (0,255,159), (0,255,255), (0,159,255), (0,64,255), (32,0,255), (128,0,255), (223,0,255), (255,0,191), (255,0,96), (204,0,0), (204,77,0), (204,153,0), (179,204,0), (102,204,0), (26,204,0), (0,204,51), (0,204,128), (0,204,204), (0,128,204), (0,51,204), (26,0,204), (102,0,204), (179,0,204), (204,0,153), (204,0,77), (153,0,0), (153,57,0), (153,115,0), (134,153,0), (77,153,0), (19,153,0), (0,153,38), (0,153,96), (0,153,153), (0,96,153), (0,38,153), (19,0,153), (77,0,153), (134,0,153), (153,0,115), (153,0,57), (102,0,0), (102,38,0), (102,77,0), (89,102,0), (51,102,0), (13,102,0), (0,102,26), (0,102,64), (0,102,102), (0,64,102), (0,26,102), (13,0,102), (51,0,102), (89,0,102), (102,0,77), (102,0,38) |
| 128 | (255,0,0), (255,96,0), (255,191,0), (223,255,0), (128,255,0), (32,255,0), (0,255,64), (0,255,159), (0,255,255), (0,159,255), (0,64,255), (32,0,255), (128,0,255), (223,0,255), (255,0,191), (255,0,96), (233,0,0), (233,87,0), (233,175,0), (204,233,0), (117,233,0), (29,233,0), (0,233,58), (0,233,146), (0,233,233), (0,146,233), (0,58,233), (29,0,233), (117,0,233), (204,0,233), (233,0,175), (233,0,87), (211,0,0), (211,79,0), (211,158,0), (185,211,0), (106,211,0), (26,211,0), (0,211,53), (0,211,132), (0,211,211), (0,132,211), (0,53,211), (26,0,211), (106,0,211), (185,0,211), (211,0,158), (211,0,79), (189,0,0), (189,71,0), (189,142,0), (166,189,0), (95,189,0), (24,189,0), (0,189,47), (0,189,118), (0,189,189), (0,118,189), (0,47,189), (24,0,189), (95,0,189), (166,0,189), (189,0,142), (189,0,71), (168,0,0), (168,63,0), (168,126,0), (147,168,0), (84,168,0), (21,168,0), (0,168,42), (0,168,105), (0,168,168), (0,105,168), (0,42,168), (21,0,168), (84,0,168), (147,0,168), (168,0,126), (168,0,63), (146,0,0), (146,55,0), (146,109,0), (128,146,0), (73,146,0), (18,146,0), (0,146,36), (0,146,91), (0,146,146), (0,91,146), (0,36,146), (18,0,146), (73,0,146), (128,0,146), (146,0,109), (146,0,55), (124,0,0), (124,46,0), (124,93,0), (108,124,0), (62,124,0), (15,124,0), (0,124,31), (0,124,77), (0,124,124), (0,77,124), (0,31,124), (15,0,124), (62,0,124), (108,0,124), (124,0,93), (124,0,46), (102,0,0), (102,38,0), (102,77,0), (89,102,0), (51,102,0), (13,102,0), (0,102,26), (0,102,64), (0,102,102), (0,64,102), (0,26,102), (13,0,102), (51,0,102), (89,0,102), (102,0,77), (102,0,38) |
| 256 | (255,0,0), (255,96,0), (255,191,0), (223,255,0), (128,255,0), (32,255,0), (0,255,64), (0,255,159), (0,255,255), (0,159,255), (0,64,255), (32,0,255), (128,0,255), (223,0,255), (255,0,191), (255,0,96), (245,0,0), (245,92,0), (245,184,0), (214,245,0), (122,245,0), (31,245,0), (0,245,61), (0,245,153), (0,245,245), (0,153,245), (0,61,245), (31,0,245), (122,0,245), (214,0,245), (245,0,184), (245,0,92), (235,0,0), (235,88,0), (235,176,0), (205,235,0), (117,235,0), (29,235,0), (0,235,59), (0,235,147), (0,235,235), (0,147,235), (0,59,235), (29,0,235), (117,0,235), (205,0,235), (235,0,176), (235,0,88), (224,0,0), (224,84,0), (224,168,0), (196,224,0), (112,224,0), (28,224,0), (0,224,56), (0,224,140), (0,224,224), (0,140,224), (0,56,224), (28,0,224), (112,0,224), (196,0,224), (224,0,168), (224,0,84), (214,0,0), (214,80,0), (214,161,0), (187,214,0), (107,214,0), (27,214,0), (0,214,54), (0,214,134), (0,214,214), (0,134,214), (0,54,214), (27,0,214), (107,0,214), (187,0,214), (214,0,161), (214,0,80), (204,0,0), (204,77,0), (204,153,0), (179,204,0), (102,204,0), (26,204,0), (0,204,51), (0,204,128), (0,204,204), (0,128,204), (0,51,204), (26,0,204), (102,0,204), (179,0,204), (204,0,153), (204,0,77), (194,0,0), (194,73,0), (194,145,0), (170,194,0), (97,194,0), (24,194,0), (0,194,48), (0,194,121), (0,194,194), (0,121,194), (0,48,194), (24,0,194), (97,0,194), (170,0,194), (194,0,145), (194,0,73), (184,0,0), (184,69,0), (184,138,0), (161,184,0), (92,184,0), (23,184,0), (0,184,46), (0,184,115), (0,184,184), (0,115,184), (0,46,184), (23,0,184), (92,0,184), (161,0,184), (184,0,138), (184,0,69), (173,0,0), (173,65,0), (173,130,0), (152,173,0), (87,173,0), (22,173,0), (0,173,43), (0,173,108), (0,173,173), (0,108,173), (0,43,173), (22,0,173), (87,0,173), (152,0,173), (173,0,130), (173,0,65), (163,0,0), (163,61,0), (163,122,0), (143,163,0), (82,163,0), (20,163,0), (0,163,41), (0,163,102), (0,163,163), (0,102,163), (0,41,163), (20,0,163), (82,0,163), (143,0,163), (163,0,122), (163,0,61), (153,0,0), (153,57,0), (153,115,0), (134,153,0), (76,153,0), (19,153,0), (0,153,38), (0,153,96), (0,153,153), (0,96,153), (0,38,153), (19,0,153), (76,0,153), (134,0,153), (153,0,115), (153,0,57), (143,0,0), (143,54,0), (143,107,0), (125,143,0), (71,143,0), (18,143,0), (0,143,36), (0,143,89), (0,143,143), (0,89,143), (0,36,143), (18,0,143), (71,0,143), (125,0,143), (143,0,107), (143,0,54), (133,0,0), (133,50,0), (133,99,0), (116,133,0), (66,133,0), (17,133,0), (0,133,33), (0,133,83), (0,133,133), (0,83,133), (0,33,133), (17,0,133), (66,0,133), (116,0,133), (133,0,99), (133,0,50), (122,0,0), (122,46,0), (122,92,0), (107,122,0), (61,122,0), (15,122,0), (0,122,31), (0,122,76), (0,122,122), (0,76,122), (0,31,122), (15,0,122), (61,0,122), (107,0,122), (122,0,92), (122,0,46), (112,0,0), (112,42,0), (112,84,0), (98,112,0), (56,112,0), (14,112,0), (0,112,28), (0,112,70), (0,112,112), (0,70,112), (0,28,112), (14,0,112), (56,0,112), (98,0,112), (112,0,84), (112,0,42), (102,0,0), (102,38,0), (102,77,0), (89,102,0), (51,102,0), (13,102,0), (0,102,26), (0,102,64), (0,102,102), (0,64,102), (0,26,102), (13,0,102), (51,0,102), (89,0,102), (102,0,77), (102,0,38) |

## 2b. Net bits/patch ceiling at `subpatch=1` (the VLM-meaningful regime)

This is the number that actually matters for the project's benefit claim, so it gets its own
callout rather than being left implicit in section 6's general packing math. At `subpatch=1`
(one symbol per DATA patch -- the only regime with any claimed relevance to a real vision
token; see section 6a's honesty note for why `subpatch>1` doesn't count here), the **net**
bits/patch a successful decode delivers -- after paying for the calibration row and Reed-Solomon
parity, not the raw `log2(P)` per-symbol ceiling -- is:

```
net_bits_per_patch = log2(P) * (data_patches / total_patches) * (payload_bytes / ecc_bytes)
```

`data_patches / total_patches` approaches 1 as the grid grows taller relative to its one
calibration row (amortizes with payload size). `payload_bytes / ecc_bytes` is capped by
Reed-Solomon: `heliogram.codec.rs_encoded_length` chunks messages into `nsize=255`-byte blocks
and spends `nsym` bytes of each on parity, so for the default `nsym=32` this factor approaches
(never exceeds) `(255 - 32) / 255 = 223/255 ≈ 0.875` as payload size grows and per-message
framing overhead (the 5-byte header, the sub-223-byte final chunk) shrinks in relative terms.
Both factors are independently confirmed against `heliogram.codec`'s own grid/RS-length
functions, not asserted from the formula alone.

Recomputing this directly from `heliogram.codec.compute_grid`/`rs_encoded_length` (not copied
from any prior sweep) at `nsym=32`, `patch_size=14`, gives, at a 4KB payload:

| P (subpatch=1) | net bits/patch @ 4KB | fraction of `log2(P)` ceiling | net bits/patch @ 16KB |
|---:|---:|---:|---:|
| 64  | 5.154 | 0.859 | 5.208 |
| 128 | 5.950 | 0.850 | 6.073 |
| 256 | 6.611 | 0.826 | 6.895 |

As payload size keeps growing, both factors above keep climbing toward their respective
asymptotes (the fraction-of-ceiling column approaches `223/255 = 0.87451`), so `P=256` climbs
toward `log2(256) * 223/255 ≈ 6.996` bits/patch (6.970 already at 64KB, 6.983 at 256KB) without
ever reaching that asymptote exactly at any finite payload size.

**What this means, stated plainly and in both directions:**

- `P=256` at `subpatch=1` **beats** the ~6 bits/token base64 baseline (section "Baselines" in the
  README) once payload size is large enough to amortize the fixed overhead: at 1KB it is still
  below the baseline (5.742 bits/patch), but it crosses 6.0 by roughly 1.5KB and is comfortably
  past it by 4KB (6.611 bits/patch, already in the table above) -- computed the same way as the
  table above, not asserted. This is the channel's real, measured headroom over plain base64
  text tokens once payload size is large enough, on this reference decoder.
- `P=256` at `subpatch=1` **does not clear** the project's own conservative `Gate #1` bar of
  **8 bits/patch** (see the README's "Roadmap / Phase-2 boundary" and `heliogram.harness`'s
  `GATE_BITS_PER_PATCH`) at any payload size -- `log2(256) * 223/255 ≈ 6.996` is the hard
  asymptotic ceiling for this palette at one symbol per patch, strictly below both 7 and 8
  regardless of payload-size amortization. No palette in `VALID_PALETTES` can clear that bar at
  `subpatch=1`; it is an architectural fact of `log2(P) * 223/255 < 8` for every `P <= 256` (the
  largest, `P=256`, tops out at just under 7), not a corruption result.
- Both bullets above are about a **clean, uncorrupted image**. Section 2a already establishes
  that at payload_size=1024, `P=256` (like `P=32/64/128`) fails to decode at all under JPEG q70
  on this reference pixel decoder -- so at that tested payload size, the realistic (corrupted)
  net bits/patch for `P=256` at `subpatch=1` measures to **0** by `heliogram.harness`'s own
  convention (0 contribution on a failed decode), not 5.7 (its clean value at that same payload
  size). The 4KB/16KB clean figures in the table above have not themselves been re-tested under
  JPEG q70 for `P=128/256` in this slice (only payload_size=1024 has a dedicated pinned test);
  what IS established is the general mechanism: this palette's clean ceiling (5.95-7.0
  bits/patch across the payload sizes above) is only ever realized if a decoder survives the
  corruption a serving pipeline actually applies -- which `decode_pixels` is measured not to,
  for `P=128/256`, at the one payload size tested. Supplying that survival is exactly the
  capability a fine-tuned VLM reader (Phase 2, not run in this repo) would need for this "beats
  base64" number to hold up outside a clean lab condition. That is this extension's entire
  benefit claim, and it is explicitly conditional on Phase 2, not demonstrated by anything in
  this repo.

## 3. Calibration row

Row 0 has `width` patches. Patch `i` (`i = 0..width-1`) is painted `palette[i % P]`. The grid is
always sized so that `width >= P` (see section 6), guaranteeing every one of the `P` colors
appears at least once in row 0. Row 0 is always full `PATCH_SIZE x PATCH_SIZE` patches --
`subpatch` (section 6a) only subdivides DATA patches (rows `1..height-1`), never row 0.

The decoder averages the sampled center-pixel color of every row-0 patch with the same
`i % P` index to recover that color's actual on-image RGB value after any corruption
(resize/JPEG/color-shift/etc). This is the codec's only redundancy mechanism for color drift;
everything else (bit errors) is handled by Reed-Solomon (section 5).

## 4. Payload framing

```
message = version (1 byte) || payload_len (4 bytes, big-endian uint32) || payload (payload_len bytes)
```

- `version` is the format version byte, currently `CODEC_VERSION = 1`.
- `payload_len` is the length of `payload` in bytes (NOT the length of `message` or of the
  ECC-coded bytes).
- `payload` is caller-supplied bytes, unconstrained.

## 5. Reed-Solomon ECC

`message` is protected with `reedsolo.RSCodec(nsym)`, `nsym` configurable (default 32 parity
bytes). reedsolo is systematic and self-chunking: for a message of length `L`, it emits
`ceil(L / (255 - nsym))` chunks, each of which is `chunk_message_bytes || nsym parity bytes`
(the message bytes appear byte-for-byte, unmodified, followed by their parity). Concatenating
chunks gives `ecc_bytes`, of total length:

```
rs_encoded_length(L, nsym) = L + nsym * ceil(L / (255 - nsym))
```

For the default `nsym=32` and small messages (`L <= 223` bytes), this is one chunk:
`ecc_len = L + 32`.

`ecc_bytes` is exactly what gets bit-packed into patches (section 6) -- there is no separate
padding step applied to the message before RS coding. Because reedsolo is systematic, the first
`L` bytes of `ecc_bytes` are always `message` unchanged, and any padding needed to fill out the
patch grid (section 6) comes strictly *after* `ecc_bytes`, as all-zero symbols. This is what
lets the decoder recover `payload_len` (and hence the exact length of the real `ecc_bytes`
region) directly from the start of the recovered stream, before it knows anything about how much
of the grid's capacity is genuine data versus trailing padding.

## 6. Symbol packing and grid sizing (`subpatch=1`)

This section describes the `subpatch=1` (default) case, i.e. one symbol per data patch; section
6a generalizes it for `subpatch=k>1`.

1. `ecc_bytes` -> bitstream, MSB-first within each byte, bytes in order.
2. The bitstream is split into groups of `bits_per_symbol = log2(P)` bits, MSB-first per group,
   each group interpreted as an unsigned integer symbol value in `[0, P)`. If the bitstream
   length is not a multiple of `bits_per_symbol`, it is zero-padded at the end to the next
   symbol boundary. This can happen whenever `bits_per_symbol` does not evenly divide 8: true for
   `P=8` (3 bits), `P=32` (5 bits), `P=64` (6 bits), and `P=128` (7 bits); it never happens for
   `P=2,4,16,256` (1, 2, 4, 8 bits, all divisors of 8 -- `P=256`'s 8 bits/symbol is exactly one
   byte, the trivial case -- so a byte-aligned bitstream is already symbol-aligned too).
   Let `num_symbols = ceil(len(ecc_bytes) * 8 / bits_per_symbol)`.
3. Grid size: `width = max(P, ceil(sqrt(num_symbols)))`, `data_rows = ceil(num_symbols / width)`,
   `height = data_rows + 1` (the `+1` is the calibration row). This is the smallest roughly-square
   grid, subject to `width >= P`, that can hold `num_symbols` data-patch symbols.
4. The `num_symbols` symbols are written into data patches (all patches in rows `1..height-1`)
   row-major. Any remaining data patches (`width * (height - 1) - num_symbols`, which is 0 when
   `num_symbols` happens to fill the grid exactly) are set to symbol value `0`.

## 6a. Sub-patch packing (`subpatch`/`k` > 1)

`encode`/`extract_symbols`/`decode_pixels`/`decode` all take a `subpatch` parameter (`k`,
default `1`, must be a member of `VALID_SUBPATCHES = (1, 2)`, and `PATCH_SIZE` must be evenly
divisible by `k` -- both violations raise `ValueError`; the default `PATCH_SIZE=14` satisfies
`14 % 2 == 0`). It is a decoder-side configuration parameter like `palette`/`patch_size`/`nsym`
(section 7): not self-describing in the byte stream, so a decoder must be called with the same
`k` the image was encoded with.

**Layout.** The calibration row (row 0) is completely unaffected -- see section 3. Each DATA
patch (every patch in rows `1..height-1`) is subdivided into a `k x k` grid of
`PATCH_SIZE/k`-px solid-color sub-cells, each carrying one symbol: `k*k` symbols per data patch
instead of 1. Within a patch, sub-cells are row-major (for `k=2`: top-left, top-right,
bottom-left, bottom-right).

**Symbol order.** The overall symbol sequence -- what gets bit-packed from/into `ecc_bytes`,
section 6 steps 1-2/4 -- is ordered *data-patch row-major, then sub-cell row-major*: for each
data patch in the same row-major order as section 6, emit that patch's `k*k` sub-cell symbols in
their own row-major order before moving to the next data patch. `k=1` collapses this to exactly
section 6's one-symbol-per-patch order (a single trivial "sub-cell" per patch), which is why
`subpatch=1` reproduces `encode`'s pre-`subpatch` output byte-for-byte
(`tests/test_roundtrip.py::test_subpatch1_output_unchanged_pinned_hash` pins this with a SHA-256
of the actual PNG bytes).

**Grid sizing.** Section 6 step 3's grid-sizing formula is reused unchanged, but on the
DATA-PATCH count rather than the raw symbol count: `data_patches_needed =
ceil(num_symbols / k**2)`, then `width, height = compute_grid(data_patches_needed, P)` (the same
`compute_grid` from section 6/8 -- it has no notion of `subpatch` itself; callers convert). Data
patches beyond what's needed are still zero-padded, now at the sub-cell level: total sub-cell
capacity is `width * (height - 1) * k**2`, and `capacity - num_symbols` trailing sub-cells get
symbol `0`. For `k=1`, `data_patches_needed == num_symbols` exactly, so this is bit-for-bit
section 6's original grid math.

**DATA HONESTY note (important):** `subpatch` is a purely *geometric* density knob. The
reference decoder (`decode_pixels`/`extract_symbols`) can read sub-cells because it samples exact
pixel centers off a grid it knows the size of in advance -- there is no perception involved. A
real ViT/VLM image encoder tokenizes the image at its *own* fixed patch grid (typically matched
to `PATCH_SIZE`, per this format's design intent) and very likely **cannot resolve structure
smaller than one of its own patches** -- a `2x2` sub-cell grid inside one ViT patch may just
average out to a blur in that patch's embedding, or may not; nobody has tested it here. So
`subpatch>1` capacity/bits-per-patch numbers produced by this module (e.g. via
`heliogram.harness`, itself out of scope for this change) are a geometric **upper bound** on
what the channel could carry if perfectly read, not a demonstrated property of any model,
including the eventual `VLMDecoder` (section 9). This is precisely the axis Phase 2 must test,
not a capability claim Phase 1 gets to make.

## 7. Decoding (`decode_pixels`)

1. Read image dimensions; `width = img.width // PATCH_SIZE`, `height = img.height // PATCH_SIZE`
   (unaffected by `subpatch` -- section 6a's sub-cells live inside these same patch dimensions).
2. Sample the center pixel of every row-0 patch; average by `i % P` to recover per-color
   calibration RGB (`extract_symbols`). Any color index with zero row-0 samples (impossible in
   an image produced by `encode`, since `width >= P`) falls back to the canonical palette color.
3. For `subpatch=1`: sample the center pixel of every data patch (rows `1..height-1`, row-major)
   and classify it to the nearest recovered calibration color (squared Euclidean distance in
   RGB) -> a symbol value per data patch. For `subpatch=k>1` (section 6a): sample the center
   pixel of every sub-cell instead (still nearest-recovered-color classification), in data-patch
   row-major then sub-cell row-major order.
4. Concatenate all symbols from step 3 in that order, unpack each into `bits_per_symbol`
   MSB-first bits, concatenate all bits, and repack into bytes (MSB-first, 8 bits/byte, trailing
   partial byte dropped). Call this `stream`.
5. Read `version = stream[0]` and `payload_len = big_endian_uint32(stream[1:5])`. Compute
   `message_len = 5 + payload_len` and `ecc_len = rs_encoded_length(message_len, nsym)`.
6. Take `ecc_bytes = stream[:ecc_len]` (this is well-defined because `stream`'s length is always
   >= the grid capacity's byte-aligned prefix, which is always >= `ecc_len` by construction).
7. RS-decode `ecc_bytes` with `reedsolo.RSCodec(nsym).decode(...)` to recover (and correct, up to
   `nsym/2` symbol errors per 255-byte chunk) `message`. Verify `message[0] == CODEC_VERSION`.
8. Return `message[5 : 5 + payload_len]` as the payload.

Any failure along this path (stream too short, RS decode raising, version byte mismatch after
correction) raises `heliogram.codec.HeliogramDecodeError` rather than returning garbage silently.

Note that `palette`, `patch_size`, `nsym`, and `subpatch` are decoder-side configuration
parameters, not self-describing fields in the byte stream (only `version` and `payload_len` are
in-band). A decoder must be called with the same `palette`/`patch_size`/`nsym`/`subpatch` the
image was encoded with.

## 8. Public API surface (`heliogram/codec.py`)

```
encode(data: bytes, palette=8, patch_size=14, nsym=32, seed=0, subpatch=1) -> PIL.Image.Image
decode_pixels(img: PIL.Image.Image, palette=8, patch_size=14, nsym=32, subpatch=1) -> bytes
decode(img, palette=8, patch_size=14, nsym=32, subpatch=1, decoder=None) -> bytes   # plug point, defaults to decode_pixels
get_palette(palette: int) -> list[tuple[int, int, int]]
extract_symbols(img, palette=8, patch_size=14, subpatch=1) -> (width, height, symbols)   # classify only, no RS/framing
rs_encoded_length(message_len, nsym, nsize=255) -> int
compute_grid(num_symbols, palette) -> (width, height)   # no subpatch param -- see section 6a
bits_per_symbol(palette) -> int
VALID_PALETTES == (2, 4, 8, 16, 32, 64, 128, 256)
VALID_SUBPATCHES == (1, 2)
HeliogramDecodeError(Exception)
VLMDecoder  # Phase-2 plug point; __call__ accepts (and ignores) `subpatch` so that
            # decode(img, ..., decoder=VLMDecoder()) forwards subpatch cleanly and reaches
            # its NotImplementedError body. A real VLM decoder must honor `subpatch`.
```

`encode()` is fully deterministic: identical arguments always produce a byte-identical PNG.
`seed` is accepted for forward API compatibility (e.g. a future dithering/anti-aliasing pass)
but has no effect in v0.1, since nothing in this format is randomized. `subpatch=1` for every
function above reproduces the exact pre-`subpatch` v0.1 output and behavior (see section 6a).

## 9. Phase-2 boundary

`decode_pixels` is a reference, model-free decoder: it exists to measure the channel (how many
bits per patch survive realistic corruption), not to be the intended production decoder.
`VLMDecoder` in `heliogram/codec.py` is the marked plug point for a GPU-fine-tuned VLM decoder
that would replace `decode_pixels`' nearest-neighbor classifier; it currently raises
`NotImplementedError`. `decode(img, ..., decoder=SomeDecoder())` is the call site that would
switch decoders without touching the framing/RS layer. No Phase-2 work (fine-tuning, model-based
numbers) is implemented in this repo -- see the README's "Roadmap / Phase-2 boundary" section.

This boundary matters more, not less, after this change: palette sizes 32/64/128/256
(section 2a/2b) and sub-patch packing (section 6a) both *raise the geometric ceiling*
`decode_pixels` can measure, but neither has been shown to be readable by an actual VLM image
encoder -- if anything, sub-patch packing is designed to exceed what a patch-grid ViT encoder
can plausibly resolve, by construction. Every number these features can produce through
`decode_pixels` or `heliogram.harness` is, and must be reported as, a model-free pixel-decoder
measurement or an analytic upper bound -- never attributed to a VLM until Phase 2 actually
measures one.

`P=128/256` sharpen this boundary specifically: section 2b's "beats base64" net-bits/patch
number is a **clean-image-only** ceiling, and section 2a pins that these two palettes are
already measured to fail outright (not just degrade) under JPEG q70 on `decode_pixels` at the
one payload size tested. So the entire quantitative case for adding 128/256 -- cheaper-than-text
context for large binary payloads, bit-exact -- is conditional on a learned (VLM) reader
recovering robustness that the reference pixel decoder demonstrably does not have at these
palette sizes. That is not a caveat to mention in passing; it is the open question this
extension exists to eventually answer, and Phase 1 (this repo) answers "no, not with this
decoder" for the corrupted case while establishing that the clean-case number is real and worth
the bet.
