# heliogram codec format v0.1

This is a mini-RFC for the wire format implemented by `heliogram/codec.py`. It is normative:
if this document and the code ever disagree, that is a bug in one of them, not a choice point.

## 1. Overview

A heliogram image is a grid of `PATCH_SIZE x PATCH_SIZE` px solid-color blocks ("patches").
Each patch encodes one symbol from a palette of `P` deterministic, separable colors, where `P`
in `{2, 4, 8, 16, 32, 64}` and `bits_per_symbol = log2(P)`. One patch is intended to correspond
to roughly one ViT/vision-token patch in a downstream VLM's image encoder.

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

## 2a. Palette (P > 16: 32, 64)

Hue-only separation at full S/V (section 2) runs out once `P` gets past 16 colors packed around
one hue wheel -- adjacent hues get close in RGB, and this is made worse under realistic
corruption because JPEG's chroma subsampling erodes hue (chroma) differences faster than
brightness (luma) ones. So for `P > 16` (only 32 and 64 are in `VALID_PALETTES`), `get_palette`
additionally varies VALUE: the palette is `levels = P // 16` value-levels of the same 16 hues
used by the `P=16` case, `S=1.0`, value stepped evenly from `1.0` down to a floor of `0.4`:

```
hues = 16
levels = P // hues            # 2 for P=32, 4 for P=64
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
this, including `get_palette(64)`).

**Honesty note:** varying value in addition to hue is a *design choice* aimed at improving
separability under corruption (a second axis riding on luma, which survives JPEG's chroma
subsampling better than hue does) -- it is **not an empirically measured result**. No
corruption/JPEG sweep at P=32/64 has been run as part of this change; `tests/test_roundtrip.py`
only checks clean-image exactness, determinism, and a mild uncompressed color shift for these
palettes (the same checks the P<=16 palettes get). Measuring P=32/64 under the realistic
corruption suite is `heliogram.harness` work, out of scope here.

Reference values (as produced by the current implementation):

| P  | colors (R,G,B) |
|---:|---|
| 32 | (255,0,0), (255,96,0), (255,191,0), (223,255,0), (128,255,0), (32,255,0), (0,255,64), (0,255,159), (0,255,255), (0,159,255), (0,64,255), (32,0,255), (128,0,255), (223,0,255), (255,0,191), (255,0,96), (102,0,0), (102,38,0), (102,77,0), (89,102,0), (51,102,0), (13,102,0), (0,102,26), (0,102,64), (0,102,102), (0,64,102), (0,26,102), (13,0,102), (51,0,102), (89,0,102), (102,0,77), (102,0,38) |
| 64 | (255,0,0), (255,96,0), (255,191,0), (223,255,0), (128,255,0), (32,255,0), (0,255,64), (0,255,159), (0,255,255), (0,159,255), (0,64,255), (32,0,255), (128,0,255), (223,0,255), (255,0,191), (255,0,96), (204,0,0), (204,77,0), (204,153,0), (179,204,0), (102,204,0), (26,204,0), (0,204,51), (0,204,128), (0,204,204), (0,128,204), (0,51,204), (26,0,204), (102,0,204), (179,0,204), (204,0,153), (204,0,77), (153,0,0), (153,57,0), (153,115,0), (134,153,0), (77,153,0), (19,153,0), (0,153,38), (0,153,96), (0,153,153), (0,96,153), (0,38,153), (19,0,153), (77,0,153), (134,0,153), (153,0,115), (153,0,57), (102,0,0), (102,38,0), (102,77,0), (89,102,0), (51,102,0), (13,102,0), (0,102,26), (0,102,64), (0,102,102), (0,64,102), (0,26,102), (13,0,102), (51,0,102), (89,0,102), (102,0,77), (102,0,38) |

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
   `P=8` (3 bits), `P=32` (5 bits), and `P=64` (6 bits); it never happens for `P=2,4,16` (1, 2, 4
   bits, all divisors of 8, so a byte-aligned bitstream is already symbol-aligned too).
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
VALID_PALETTES == (2, 4, 8, 16, 32, 64)
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

This boundary matters more, not less, after this change: palette sizes 32/64 (section 2a) and
sub-patch packing (section 6a) both *raise the geometric ceiling* `decode_pixels` can measure,
but neither has been shown to be readable by an actual VLM image encoder -- if anything,
sub-patch packing is designed to exceed what a patch-grid ViT encoder can plausibly resolve, by
construction. Every number these features can produce through `decode_pixels` or
`heliogram.harness` is, and must be reported as, a model-free pixel-decoder measurement or an
analytic upper bound -- never attributed to a VLM until Phase 2 actually measures one.
