"""heliogram.benefit -- Slice C: the exactness differentiator, and a live token-savings demo.

Two small, honest analyses that stand on their own even in the world where a future Phase-2
measurement shows a fine-tuned VLM's rendered-text OCR matches heliogram's bits/patch density.
The README's Baselines section already names this possibility explicitly: "If rendered text
matches the codec's bits/patch, the codec's only remaining advantages are exactness (ECC-
verified bytes) and robustness -- and we'll say so." This module is that "and we'll say so":

  1. `exactness_argument()` / `rs_error_correction_capacity()` -- a STRUCTURAL argument for why
     heliogram's Reed-Solomon-verified decode is a durable niche even where density merely ties
     with rendered-text-via-VLM-OCR: RS gives detection *and* correction; free-form OCR gives
     neither. No OCR error rate is invented anywhere below -- every point about OCR is either a
     structural/logical fact (an un-instrumented free-text transcription has no built-in way to
     know it is wrong) or is explicitly flagged as an open Phase-2 measurement.
  2. `token_savings_demo()` -- encodes a real ~4-8KB structured (JSON) payload at `palette=256`
     and reports patches vs. base64 tokens vs. a "raw-byte" (hex) text-tokenization baseline,
     using the SAME "~1 token/char" convention `heliogram.baselines`/`heliogram.harness` already
     use elsewhere in this project (not a second, drifted definition of "a token"). It then
     immediately (and loudly) demonstrates -- by actually running `decode_pixels` twice, once on
     the clean image and once on a real JPEG q70 re-encode of that SAME image -- exactly why this
     is a clean-channel-only number today: decoding this demo's own output in a real serving
     pipeline needs the Phase-2 reader (`heliogram.vlm.QwenVLDecoder` with a fine-tuned model,
     not `decode_pixels`), because `palette=256` is measured, right here and now, to survive the
     clean decode and fail the corrupted one -- see RESULTS.md for the same fact at other payload
     sizes and codec.py's DATA HONESTY note for why.

DATA HONESTY: everything below is either (a) exact arithmetic/deterministic code -- RS
error-correction capacity, grid sizing via `heliogram.codec.encode` itself, base64/hex token
counts -- recomputed fresh every time this module runs, never copied from a prior sweep, or (b)
a structural, logical argument about what Reed-Solomon vs. free-form OCR can and cannot
guarantee. No invented OCR accuracy/error-rate number appears anywhere in this file.

No torch/transformers anywhere in this module -- only `heliogram.codec`/`heliogram.corruption`
(pillow/numpy/reedsolo), the same base dependencies as the rest of Phase 1. `import
heliogram.benefit` is always safe in a CPU-only, no-torch environment.

Run directly: `python -m heliogram.benefit` (module-relative imports, like `heliogram.dataset`/
`heliogram.vlm`, mean this must be run as `-m`, not as a standalone script path).
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
from dataclasses import dataclass
from typing import List, Optional

from .codec import HeliogramDecodeError, PATCH_SIZE, RS_NSIZE, decode_pixels, encode
from .corruption import jpeg_compress

__all__ = [
    "RSGuarantee",
    "rs_error_correction_capacity",
    "ExactnessPoint",
    "exactness_argument",
    "format_exactness_argument",
    "TokenSavingsResult",
    "sample_structured_payload",
    "token_savings_demo",
    "format_token_savings_report",
]


# --------------------------------------------------------------------------------------------
# 1. The exactness differentiator
# --------------------------------------------------------------------------------------------


@dataclass
class RSGuarantee:
    """The real, deterministic Reed-Solomon guarantee behind heliogram's "exact/verifiable"
    claim -- see `rs_error_correction_capacity`."""

    nsize: int
    nsym: int
    max_correctable_byte_errors_per_chunk: int
    note: str


def rs_error_correction_capacity(nsym: int = 32, nsize: int = RS_NSIZE) -> RSGuarantee:
    """The concrete, computable guarantee heliogram's Reed-Solomon layer provides per
    `nsize`-byte codeword (255 by default -- `heliogram.codec.RS_NSIZE`, reedsolo's GF(256)
    block size): a code with `nsym` parity bytes CORRECTS up to `floor(nsym/2)` byte errors per
    chunk, and DETECTS -- raises, rather than silently returning wrong bytes -- whenever more
    than that are corrupted. `heliogram.codec.decode_pixels` and
    `heliogram.vlm._payload_from_symbols` both rely on exactly this behavior (via
    `reedsolo.RSCodec.decode`, unmodified) for the entire "bit-exact or explicitly failed, never
    silently wrong" property this module's argument rests on.

    Pure arithmetic (`nsym // 2`) -- not a measurement, not a model, and not specific to this
    project: it is a property of Reed-Solomon codes in general, restated here in terms of the
    exact parameters `heliogram.codec.encode`'s default (`nsym=32`) uses.
    """
    if not (0 < nsym < nsize):
        raise ValueError(f"nsym must be in (0, {nsize}), got {nsym!r}")
    correctable = nsym // 2
    return RSGuarantee(
        nsize=nsize,
        nsym=nsym,
        max_correctable_byte_errors_per_chunk=correctable,
        note=(
            f"RS(n={nsize}, parity={nsym}) corrects up to {correctable} corrupted byte(s) per "
            f"{nsize}-byte chunk, and DETECTS (raises reedsolo.ReedSolomonError, which "
            "heliogram.codec.decode_pixels re-raises as HeliogramDecodeError) rather than "
            "silently returning wrong bytes once more than that are corrupted in a chunk. This "
            "is a property of the Reed-Solomon code itself, not a measurement this repo made."
        ),
    )


@dataclass
class ExactnessPoint:
    """One point of contrast between heliogram's RS-verified decode and reading rendered text
    off an image via a VLM's free-form OCR. `status` is `"structural"` for a claim that is true
    by construction/math (no model run needed to know it), or `"open_phase2_measurement"` for a
    claim whose actual number can only come from running a real VLM -- never invented here."""

    claim: str
    heliogram: str
    rendered_text_ocr: str
    status: str


def exactness_argument(nsym: int = 32) -> List[ExactnessPoint]:
    """THE exactness differentiator (decision 2): heliogram's durable niche even in a world
    where a fine-tuned VLM's rendered-text OCR someday matches heliogram's bits/patch density.
    Framed, deliberately, as a structural comparison -- what each scheme's OWN mechanism can and
    cannot guarantee -- not as a race between two accuracy percentages, because only ONE side of
    that race (heliogram's, via Reed-Solomon) has a number computable without a GPU.

    The one-line version, stated plainly: Reed-Solomon gives detection + correction; free-form
    OCR gives neither.
    """
    rs = rs_error_correction_capacity(nsym)
    return [
        ExactnessPoint(
            claim="Error detection (does the reader know when it's wrong?)",
            heliogram=(
                f"Reed-Solomon (nsym={nsym}) detects corruption beyond its correction budget "
                f"({rs.max_correctable_byte_errors_per_chunk} bytes per {rs.nsize}-byte chunk) "
                "and raises HeliogramDecodeError instead of returning a payload. A failed "
                "decode is visibly a failure -- decode_pixels never returns a silently-wrong "
                "answer, and neither would a VLM decoder plugged into the same RS/framing layer "
                "(heliogram.vlm._payload_from_symbols reuses it unmodified)."
            ),
            rendered_text_ocr=(
                "Free-form OCR has no analogous check built in: a VLM transcribing rendered "
                "text returns SOME string whether or not every character was read correctly -- "
                "there is no signal, from the OCR output alone, distinguishing a correct "
                "transcription from a confidently-wrong (hallucinated or misread) one."
            ),
            status="structural",
        ),
        ExactnessPoint(
            claim="Error correction (can the reader fix small mistakes on its own?)",
            heliogram=(
                f"Up to {rs.max_correctable_byte_errors_per_chunk} corrupted bytes per "
                f"{rs.nsize}-byte chunk are corrected exactly, deterministically, with no model "
                "or heuristic in the loop -- see rs_error_correction_capacity() above."
            ),
            rendered_text_ocr=(
                "No correction mechanism exists unless one is added ON TOP of the rendering "
                "(e.g. a checksum typeset alongside the text, or an RS-coded payload rendered "
                "as characters instead of colors) -- at which point the rendered-text scheme "
                "has simply reinvented an ECC layer, which heliogram already has natively."
            ),
            status="structural",
        ),
        ExactnessPoint(
            claim="Ground-truth / training-label quality",
            heliogram=(
                "heliogram.dataset's training targets are read directly off the pixels the "
                "codec itself just wrote (extract_symbols on the clean image) -- exact by "
                "construction, zero hand-labeling, zero OCR-grading-OCR circularity."
            ),
            rendered_text_ocr=(
                "Building a labeled set for OCR correctness needs an independent source of "
                "truth for what the rendered text says; grading a VLM's OCR against another "
                "OCR pass (or against the rendering pipeline's own font-metrics assumptions) "
                "risks circularity heliogram's codec-verified targets never face."
            ),
            status="structural",
        ),
        ExactnessPoint(
            claim="Actual error rate delivered, under real corruption",
            heliogram=(
                "Measured on the pixel decoder (see RESULTS.md): a decode either succeeds "
                "exactly or is detected as a failure -- there is no partial-credit 'mostly "
                "right' payload state. Whether a fine-tuned VLM reader matches this at "
                "palette in {64, 128, 256} is Phase 2's open question (see "
                "heliogram/dataset.py's retargeted curriculum), not settled here."
            ),
            rendered_text_ocr=(
                "NOT MEASURED HERE, and not invented: the real character/word error rate of a "
                "VLM's OCR on heliogram-scale rendered text needs an actual model run (see "
                "heliogram.baselines.rendered_text_density's docstring and the README's "
                "Baselines section, both of which already flag this as Phase-2, model-required "
                "work) -- no figure is asserted in this module or anywhere else in this repo."
            ),
            status="open_phase2_measurement",
        ),
    ]


def format_exactness_argument(points: Optional[List[ExactnessPoint]] = None) -> str:
    """Pretty-print `exactness_argument()`'s output as plain text."""
    if points is None:
        points = exactness_argument()
    lines = [
        "EXACTNESS DIFFERENTIATOR: heliogram (RS-verified) vs. rendered-text-via-VLM-OCR",
        "=" * 78,
        "",
    ]
    for i, p in enumerate(points, 1):
        tag = "[structural]" if p.status == "structural" else "[OPEN -- Phase 2 measurement]"
        lines += [
            f"{i}. {p.claim}  {tag}",
            f"   heliogram:        {p.heliogram}",
            f"   rendered-text OCR: {p.rendered_text_ocr}",
            "",
        ]
    lines.append(
        "One line: Reed-Solomon gives detection + correction; free-form OCR gives neither. "
        "That holds even if a future measurement shows OCR matching heliogram's bits/patch."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# 2. Token-savings demo (CPU, no model)
# --------------------------------------------------------------------------------------------


def sample_structured_payload(seed: int = 0, target_bytes: int = 6000) -> bytes:
    """A deterministic, synthetic ~`target_bytes` JSON document -- a batch of structured sensor-
    reading records, in the spirit of `demo.py`'s small JSON example but scaled up into the
    4-8KB range where `heliogram.harness` measures `palette=256` to cross below base64 token
    count (see RESULTS.md's "Token crossover" section). Synthetic and seed-deterministic, like
    every other payload in this project (see the README's opening line): same `seed`/
    `target_bytes` always produce byte-identical output.
    """
    rng = random.Random(seed)
    records: List[dict] = []
    i = 0
    encoded = b""
    while len(encoded) < target_bytes:
        records.append(
            {
                "id": i,
                "ts": 1_700_000_000 + i * 60,
                "sensor": f"sensor-{i % 24:02d}",
                "reading_c": round(rng.uniform(-10.0, 45.0), 2),
                "humidity_pct": round(rng.uniform(0.0, 100.0), 1),
                "battery_v": round(rng.uniform(3.0, 4.2), 3),
                "ok": rng.random() > 0.05,
            }
        )
        encoded = json.dumps(records, separators=(",", ":")).encode("utf-8")
        i += 1
    return encoded


@dataclass
class TokenSavingsResult:
    payload_len: int
    palette: int
    patch_size: int
    nsym: int
    total_patches: int
    base64_token_est: int
    hex_token_est: int
    patches_vs_base64_ratio: float
    patches_vs_hex_ratio: float
    clean_roundtrip_ok: bool
    jpeg_q70_roundtrip_ok: bool
    note: str


def token_savings_demo(
    payload: bytes, palette: int = 256, patch_size: int = PATCH_SIZE, nsym: int = 32
) -> TokenSavingsResult:
    """THE token-savings demo (decision 3): encode `payload` for real at `palette` (default
    256 -- the config README/RESULTS.md pin as this project's actual measured token-crossover
    win once payload size is large enough, see "Token crossover" in RESULTS.md), and report:

      - `total_patches`: the REAL patch-grid size `encode()` just produced for this payload --
        not an estimate; an actual image was built and its width/height measured.
      - `base64_token_est`: the length of `base64.b64encode(payload)`, ~1 token/char -- the SAME
        convention `heliogram.baselines.base64_bits_per_token`/`heliogram.harness`'s
        `_base64_token_estimate` use, so this demo's numbers are never a second, drifted
        definition of "a base64 token".
      - `hex_token_est`: the length of `payload.hex()`, ~1 token/char -- one reasonable "raw
        bytes as text" baseline: two ASCII hex digits per byte, no bit-packing (unlike base64's
        3-bytes-into-4-chars packing). This is a real, if naive, encoding people reach for when
        they don't want to bother with base64 at all (e.g. Postgres's `bytea` hex format,
        `xxd`-style dumps) -- one reasonable definition of "raw-byte tokenization", not the only
        possible one; the same ~1-token/char assumption is just applied to a different alphabet.

    Then, to make the mandatory caveat a LIVE measurement rather than a citation: decodes the
    CLEAN image (expected to succeed -- bit-exact, RS-verified) and separately corrupts that
    SAME image with a real JPEG q70 re-encode (`heliogram.corruption.jpeg_compress`, the exact
    corruption `heliogram.codec`/RESULTS.md already measure this palette to fail under) and
    attempts `decode_pixels` on THAT, recording whether it succeeded. Both booleans are real
    outcomes of two decode attempts run just now, not looked up from a table -- see
    `TokenSavingsResult.note`.
    """
    img = encode(payload, palette=palette, patch_size=patch_size, nsym=nsym, seed=0, subpatch=1)
    total_patches = (img.width // patch_size) * (img.height // patch_size)

    base64_token_est = len(base64.b64encode(payload))
    hex_token_est = len(payload.hex())

    try:
        clean_ok = decode_pixels(img, palette=palette, patch_size=patch_size, nsym=nsym) == payload
    except HeliogramDecodeError:
        clean_ok = False

    corrupted_img = jpeg_compress(img, quality=70)
    try:
        jpeg_ok = (
            decode_pixels(corrupted_img, palette=palette, patch_size=patch_size, nsym=nsym)
            == payload
        )
    except HeliogramDecodeError:
        jpeg_ok = False

    return TokenSavingsResult(
        payload_len=len(payload),
        palette=palette,
        patch_size=patch_size,
        nsym=nsym,
        total_patches=total_patches,
        base64_token_est=base64_token_est,
        hex_token_est=hex_token_est,
        patches_vs_base64_ratio=(total_patches / base64_token_est) if base64_token_est else 0.0,
        patches_vs_hex_ratio=(total_patches / hex_token_est) if hex_token_est else 0.0,
        clean_roundtrip_ok=clean_ok,
        jpeg_q70_roundtrip_ok=jpeg_ok,
        note=(
            "total_patches/base64_token_est/hex_token_est and both roundtrip booleans are all "
            "real for THIS run (an actual image was built; both decode attempts actually ran) "
            "-- not looked up from RESULTS.md. clean_roundtrip_ok=False or "
            "jpeg_q70_roundtrip_ok=True for a large palette would both be genuine news worth "
            "investigating, not an outcome this function assumes."
        ),
    )


def format_token_savings_report(result: TokenSavingsResult) -> str:
    """Pretty-print a `TokenSavingsResult` as plain text, including the mandatory Phase-2
    caveat -- loud, not buried, per decision 3."""
    lines = [
        "TOKEN-SAVINGS DEMO (CPU, no model) -- heliogram vs. text-context encodings",
        "=" * 78,
        "",
        f"payload:              {result.payload_len} bytes",
        f"palette / patch_size / nsym: {result.palette} / {result.patch_size}px / {result.nsym}",
        "",
        f"{'heliogram patches:':<23}{result.total_patches:>8}   (~1 self-hosted-VLM token/patch)",
        f"{'base64 tokens:':<23}{result.base64_token_est:>8}   (~1 token/char, base64.b64encode)",
        f"{'hex (raw-byte) tokens:':<23}{result.hex_token_est:>8}   (~1 token/char, payload.hex())",
        "",
        f"patches / base64 tokens: {result.patches_vs_base64_ratio:.3f}x  "
        f"({'heliogram CHEAPER' if result.patches_vs_base64_ratio < 1.0 else 'base64 cheaper'})",
        f"patches / hex tokens:    {result.patches_vs_hex_ratio:.3f}x  "
        f"({'heliogram CHEAPER' if result.patches_vs_hex_ratio < 1.0 else 'hex cheaper'})",
        "",
        f"LIVE decode check, clean image:      "
        f"{'PASS (bit-exact)' if result.clean_roundtrip_ok else 'FAIL'}",
        f"LIVE decode check, same image @ JPEG q70: "
        f"{'PASS' if result.jpeg_q70_roundtrip_ok else 'FAIL (measured, right now, on this image)'}",
        "",
    ]
    if result.clean_roundtrip_ok and not result.jpeg_q70_roundtrip_ok:
        lines.append(
            "CAVEAT (loud, not buried): the token savings above are real, but this demo's own "
            "JPEG q70 check just failed on decode_pixels for this exact image -- see codec.py's "
            "DATA HONESTY note and RESULTS.md. Decoding heliogram output like this in a real "
            "serving pipeline needs the Phase-2 reader (heliogram.vlm.QwenVLDecoder with a "
            "fine-tuned model), NOT decode_pixels. The token-count benefit is a clean-channel-"
            "only number until that reader exists and is measured to survive this corruption."
        )
    elif not result.clean_roundtrip_ok:
        lines.append(
            "NOTE: the clean-image decode check FAILED for this run -- that would itself be "
            "unexpected (see tests/test_roundtrip.py's clean-roundtrip coverage) and worth "
            "investigating; the token-count numbers above do not depend on it, but the "
            "'bit-exact' half of the exactness argument does."
        )
    elif result.palette >= 128:
        lines.append(
            "NOTE: the JPEG q70 decode check PASSED for this run at palette="
            f"{result.palette} -- that would contradict RESULTS.md's pinned measurement (this "
            "palette range is measured to FAIL jpeg_q70 at every tested payload size) and is "
            "worth investigating, not simply trusting; see tests/test_roundtrip.py's "
            "known-failure tests for the pinned case."
        )
    else:
        lines.append(
            f"NOTE: the JPEG q70 decode check PASSED for this run -- expected at palette="
            f"{result.palette}: decode_pixels is measured (RESULTS.md) to handle this smaller "
            "palette fine across the realistic corruption envelope. The caveat above is "
            "specific to the large palettes ({64, 128, 256}) this project's benefit claim and "
            "Phase-2 retarget are actually about."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=0, help="payload RNG seed (default: 0)")
    parser.add_argument(
        "--target-bytes",
        type=int,
        default=6000,
        help="approximate size of the synthetic JSON payload, in bytes (default: 6000)",
    )
    parser.add_argument(
        "--palette", type=int, default=256, help="heliogram palette size (default: 256)"
    )
    parser.add_argument("--nsym", type=int, default=32, help="Reed-Solomon parity bytes (default: 32)")
    parser.add_argument(
        "--patch-size", type=int, default=PATCH_SIZE, help=f"patch size in px (default: {PATCH_SIZE})"
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    payload = sample_structured_payload(seed=args.seed, target_bytes=args.target_bytes)
    result = token_savings_demo(
        payload, palette=args.palette, patch_size=args.patch_size, nsym=args.nsym
    )
    print(format_token_savings_report(result))
    print()
    print(format_exactness_argument())
    return 0


if __name__ == "__main__":
    sys.exit(main())
