# heliogram codec format v0.1

This is a mini-RFC for the wire format implemented by `heliogram/codec.py`. It is normative:
if this document and the code ever disagree, that is a bug in one of them, not a choice point.

## 1. Overview

A heliogram image is a grid of `PATCH_SIZE x PATCH_SIZE` px solid-color blocks ("patches").
Each patch encodes exactly one symbol from a palette of `P` deterministic, separable colors,
where `P` in `{2, 4, 8, 16}` and `bits_per_symbol = log2(P)`. One patch is intended to correspond
to roughly one ViT/vision-token patch in a downstream VLM's image encoder.

```
+----------------------------------------+
| row 0: CALIBRATION ROW                 |  patch i = palette[i % P], i = 0..width-1
+----------------------------------------+
| row 1: data patch, data patch, ...     |
| row 2: data patch, data patch, ...     |  row-major, all rows after row 0
| ...                                     |
+----------------------------------------+
```

Image pixel size = `(width * PATCH_SIZE) x (height * PATCH_SIZE)`.

## 2. Palette

`get_palette(P)` returns `P` RGB triples, deterministic in `P` alone (no seed, no randomness):

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

## 3. Calibration row

Row 0 has `width` patches. Patch `i` (`i = 0..width-1`) is painted `palette[i % P]`. The grid is
always sized so that `width >= P` (see section 6), guaranteeing every one of the `P` colors
appears at least once in row 0.

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

## 6. Symbol packing and grid sizing

1. `ecc_bytes` -> bitstream, MSB-first within each byte, bytes in order.
2. The bitstream is split into groups of `bits_per_symbol = log2(P)` bits, MSB-first per group,
   each group interpreted as an unsigned integer symbol value in `[0, P)`. If the bitstream
   length is not a multiple of `bits_per_symbol`, it is zero-padded at the end to the next
   symbol boundary (this can only happen for `P=8`, since 8 is a multiple of 1, 2, and 4).
   Let `num_symbols = ceil(len(ecc_bytes) * 8 / bits_per_symbol)`.
3. Grid size: `width = max(P, ceil(sqrt(num_symbols)))`, `data_rows = ceil(num_symbols / width)`,
   `height = data_rows + 1` (the `+1` is the calibration row). This is the smallest roughly-square
   grid, subject to `width >= P`, that can hold `num_symbols` data-patch symbols.
4. The `num_symbols` symbols are written into data patches (all patches in rows `1..height-1`)
   row-major. Any remaining data patches (`width * (height - 1) - num_symbols`, which is 0 when
   `num_symbols` happens to fill the grid exactly) are set to symbol value `0`.

## 7. Decoding (`decode_pixels`)

1. Read image dimensions; `width = img.width // PATCH_SIZE`, `height = img.height // PATCH_SIZE`.
2. Sample the center pixel of every row-0 patch; average by `i % P` to recover per-color
   calibration RGB (`extract_symbols`). Any color index with zero row-0 samples (impossible in
   an image produced by `encode`, since `width >= P`) falls back to the canonical palette color.
3. Sample the center pixel of every data patch (rows `1..height-1`, row-major) and classify it
   to the nearest recovered calibration color (squared Euclidean distance in RGB) -> a symbol
   value per data patch.
4. Concatenate all data-patch symbols in row-major order, unpack each into `bits_per_symbol`
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

Note that `palette`, `patch_size`, and `nsym` are decoder-side configuration parameters, not
self-describing fields in the byte stream (only `version` and `payload_len` are in-band). A
decoder must be called with the same `palette`/`patch_size`/`nsym` the image was encoded with.

## 8. Public API surface (`heliogram/codec.py`)

```
encode(data: bytes, palette=8, patch_size=14, nsym=32, seed=0) -> PIL.Image.Image
decode_pixels(img: PIL.Image.Image, palette=8, patch_size=14, nsym=32) -> bytes
decode(img, palette=8, patch_size=14, nsym=32, decoder=None) -> bytes   # plug point, defaults to decode_pixels
get_palette(palette: int) -> list[tuple[int, int, int]]
extract_symbols(img, palette=8, patch_size=14) -> (width, height, symbols)   # classify only, no RS/framing
rs_encoded_length(message_len, nsym, nsize=255) -> int
compute_grid(num_symbols, palette) -> (width, height)
bits_per_symbol(palette) -> int
HeliogramDecodeError(Exception)
VLMDecoder  # Phase-2 plug point, __call__ raises NotImplementedError
```

`encode()` is fully deterministic: identical arguments always produce a byte-identical PNG.
`seed` is accepted for forward API compatibility (e.g. a future dithering/anti-aliasing pass)
but has no effect in v0.1, since nothing in this format is randomized.

## 9. Phase-2 boundary

`decode_pixels` is a reference, model-free decoder: it exists to measure the channel (how many
bits per patch survive realistic corruption), not to be the intended production decoder.
`VLMDecoder` in `heliogram/codec.py` is the marked plug point for a GPU-fine-tuned VLM decoder
that would replace `decode_pixels`' nearest-neighbor classifier; it currently raises
`NotImplementedError`. `decode(img, ..., decoder=SomeDecoder())` is the call site that would
switch decoders without touching the framing/RS layer. No Phase-2 work (fine-tuning, model-based
numbers) is implemented in this repo -- see the README's "Roadmap / Phase-2 boundary" section.
