"""heliogram.benefit -- Slice C: the exactness differentiator, and a live token-savings demo.

Two small, honest analyses that stand on their own even in the world where a future Phase-2
measurement shows a fine-tuned VLM's rendered-text OCR matches heliogram's bits/patch density.
The README's Baselines section already names this possibility explicitly: "If rendered text
matches the codec's bits/patch, the codec's only remaining advantages are exactness (ECC-
verified bytes) and robustness -- and we'll say so." This module is that "and we'll say so":

  1. `exactness_argument()` / `rs_error_correction_capacity()` -- a STRUCTURAL argument for why
     heliogram's Reed-Solomon-verified decode is a durable niche even where density merely ties
     with rendered-text-via-VLM-OCR: RS gives EXACT correction within its budget, and detects-
     or-fails with OVERWHELMING but NOT ABSOLUTE probability beyond it (bounded-distance RS
     decoding can, rarely, miscorrect into a wrong-but-plausible codeword once the correction
     budget is exceeded -- see rs_error_correction_capacity()'s docstring); free-form OCR gives
     neither the correction nor any detection signal at all. No OCR error rate is invented
     anywhere below -- every point about OCR is either a structural/logical fact (an
     un-instrumented free-text transcription has no built-in way to know it is wrong) or is
     explicitly flagged as an open Phase-2 measurement.
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
from typing import List, Optional, Tuple

from .baselines import load_measured_baseline
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
    "sample_binary_payload",
    "token_savings_demo",
    "format_token_savings_report",
    "CostAsymmetryPoint",
    "cost_asymmetry_points",
    "format_cost_asymmetry",
    "chance_level_symbol_error",
    "RecoveredBitResult",
    "effective_cost_per_recovered_bit",
    "format_effective_cost_per_recovered_bit",
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
    chunk EXACTLY, deterministically -- that half of the guarantee is absolute, no probability
    involved. Beyond that budget, bounded-distance RS decoding (`reedsolo.RSCodec.decode`,
    unmodified) DETECTS OR FAILS WITH OVERWHELMING PROBABILITY -- this half is NOT an absolute
    guarantee: once corruption in a chunk exceeds `floor(nsym/2)` errors, there is a small but
    nonzero chance the decoder locks onto a different, still-internally-consistent codeword and
    returns wrong bytes without raising at all (a "miscorrection"), rather than raising
    `reedsolo.ReedSolomonError` (which `heliogram.codec.decode_pixels` re-raises as
    `HeliogramDecodeError`). `heliogram.codec.decode_pixels` and
    `heliogram.vlm._payload_from_symbols` both rely on exactly this behavior for heliogram's
    real property: "bit-exact, or detects/fails with overwhelming (not absolute) probability --
    never silently wrong with certainty, but not PROVABLY never silently wrong either." See
    `exactness_argument()` for the full comparison against free-form OCR, which has no
    correction mechanism and no detection guarantee at all, absolute or probabilistic.

    Pure arithmetic (`nsym // 2`) -- not a measurement, not a model, and not specific to this
    project: it is a property of Reed-Solomon codes in general, restated here in terms of the
    exact parameters `heliogram.codec.encode`'s default (`nsym=32`) uses. The miscorrection-risk
    caveat above is likewise a standard, textbook property of bounded-distance RS decoding, not
    something this repo measured -- no miscorrection RATE is asserted or measured anywhere in
    this module; only that it is not exactly zero once the correction budget is exceeded.
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
            f"{nsize}-byte chunk EXACTLY (absolute guarantee, no probability involved). Beyond "
            f"that budget, it detects or fails with OVERWHELMING PROBABILITY -- raising "
            "reedsolo.ReedSolomonError (which heliogram.codec.decode_pixels re-raises as "
            "HeliogramDecodeError) rather than silently returning wrong bytes -- but this is "
            "NOT an absolute guarantee: bounded-distance RS decoding has a small, nonzero "
            "chance of miscorrection (returning a different, internally-consistent-looking but "
            f"wrong codeword without raising) once more than {correctable} bytes per chunk are "
            "corrupted. This is a property of the Reed-Solomon code itself, not a measurement "
            "this repo made, and no miscorrection rate is asserted or measured here."
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
                f"Reed-Solomon (nsym={nsym}) detects or fails with OVERWHELMING PROBABILITY -- "
                f"not an absolute guarantee -- once corruption exceeds its correction budget "
                f"({rs.max_correctable_byte_errors_per_chunk} bytes per {rs.nsize}-byte chunk), "
                "raising HeliogramDecodeError instead of returning a payload. Bounded-distance "
                "RS decoding has a small, nonzero chance of 'miscorrection' (locking onto a "
                "different, internally-consistent-looking codeword and returning wrong bytes "
                "WITHOUT raising) once that budget is exceeded -- see "
                "rs_error_correction_capacity()'s note for why. What IS absolute: WITHIN the "
                f"correction budget ({rs.max_correctable_byte_errors_per_chunk} bytes/chunk), "
                "decode_pixels corrects exactly, every time, with no probability involved at "
                "all -- only the beyond-budget DETECTION half of the claim is probabilistic."
            ),
            rendered_text_ocr=(
                "Free-form OCR has no analogous check built in at all -- not even a "
                "probabilistic one: a VLM transcribing rendered text returns SOME string "
                "whether or not every character was read correctly, with no failure mode, "
                "confidence signal, or miscorrection-vs-correct distinction available from the "
                "OCR output alone. Reed-Solomon's detection is 'overwhelmingly likely, not "
                "absolute'; free-form OCR's detection is simply absent."
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
                "Measured on the pixel decoder (see RESULTS.md): across every corruption trial "
                "actually run there, a decode either succeeded exactly or was detected as a "
                "failure -- no 'mostly right' partial-credit payload was observed. That is an "
                "empirical observation over the trials actually run, NOT a proof that "
                "miscorrection is impossible: rs_error_correction_capacity() explains why "
                "bounded-distance RS decoding detects or fails with overwhelming, not absolute, "
                "probability once corruption exceeds the correction budget -- no miscorrection "
                "RATE has been measured here. Whether a fine-tuned VLM reader matches even the "
                "'exact or detected' half of this at palette in {64, 128, 256} is Phase 2's "
                "open question (see heliogram/dataset.py's retargeted curriculum), not settled "
                "here."
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
        "One line: Reed-Solomon gives exact correction within its budget, and detects-or-fails "
        "with overwhelming (not absolute) probability beyond it; free-form OCR gives neither. "
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


def sample_binary_payload(seed: int = 0, target_bytes: int = 6000) -> bytes:
    """A deterministic, synthetic ~`target_bytes` payload of GENUINELY BINARY bytes -- the
    payload shape decision 3's baseline-honesty fix (C3) exists for: uniformly random bytes have
    no fair "raw text in context" baseline at all, because they are (overwhelmingly likely) not
    even valid UTF-8, let alone readable text -- the ONLY ways to put them in a text-only LLM
    context are exactly the binary-to-text encodings (base64, hex) `token_savings_demo` already
    compares against. This is the payload SHAPE this project's candidate niche actually targets
    (see this module's docstring and README's Baselines section): incompressible binary data
    that must sit verbatim in context, as opposed to `sample_structured_payload`'s JSON, which is
    text-like and (per `token_savings_demo`'s new raw-text row) heliogram loses to badly.

    Uses the SAME sampling convention as `heliogram.baselines.base64_bits_per_token`'s default
    sample (`random.Random(seed).getrandbits(8)`), just at `target_bytes` length instead of a
    fixed 4096 -- so this is not a third, drifted definition of "random bytes" in this project.
    Synthetic and seed-deterministic: same `seed`/`target_bytes` always produce byte-identical
    output.
    """
    rng = random.Random(seed)
    return bytes(rng.getrandbits(8) for _ in range(target_bytes))


def _estimate_raw_text_tokens(text: str) -> Tuple[int, str]:
    """Best-effort token count for `text` AS PLAIN TEXT (not base64/hex) -- the fair text-context
    baseline for a text-like payload (decision C3): if the payload IS text (e.g. JSON), the
    honest comparison for "does heliogram save context tokens" is against the raw text sitting
    in context, not against a base64/hex re-encoding of it that no one is forced to use for text.

    Tries, in order:
      1. If `transformers` is importable, load a real tokenizer -- the SAME `tokenizer_id`
         `heliogram.baselines.measure_base64_baseline` last measured (via
         `heliogram.baselines.load_measured_baseline`), or the same
         "Qwen/Qwen2.5-VL-7B-Instruct" default if no measurement has been persisted yet -- and
         tokenize `text` FOR REAL. Any failure (no network/no cached files/etc) falls through to
         (2) rather than raising, but the returned method note says so honestly (never mislabels
         a fallback estimate as "measured").
      2. Otherwise, the standard analytic rule of thumb for English/JSON-ish text with common BPE
         tokenizers: ~4 characters/token (`len(text) // 4`, minimum 1 token for nonempty text).
         This is deliberately a ROUND, CONSERVATIVE number -- the commonly cited range for
         English/code/JSON text is closer to 3.5-4 chars/token, and 4 is the token-count-
         FAVORABLE-to-heliogram end of that range (fewer chars/token would mean MORE raw-text
         tokens, i.e. an even wider heliogram loss) -- so this fallback cannot be accused of
         padding the honesty caveat below by underestimating plain text's token cost.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        AutoTokenizer = None  # noqa: N806

    if AutoTokenizer is not None:
        measured = load_measured_baseline()
        tokenizer_id = measured.tokenizer_id if measured is not None else "Qwen/Qwen2.5-VL-7B-Instruct"
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            n_tokens = len(tokenizer.encode(text))
            return n_tokens, (
                f"measured: {tokenizer_id} tokenizer, tokenized directly on the actual payload "
                "text (not a synthetic sample)"
            )
        except Exception as exc:  # network/cache-miss/etc -- fall back honestly, don't raise
            fallback_prefix = (
                f"transformers is installed but loading/using {tokenizer_id!r} failed "
                f"({exc.__class__.__name__}); falling back to the analytic estimate -- "
            )
    else:
        fallback_prefix = "transformers not installed; "

    n_tokens = max(1, len(text) // 4)
    return n_tokens, (
        fallback_prefix
        + "analytic estimate: ~4 characters/token, the standard conservative rule-of-thumb "
        "ratio for English/JSON-ish text with common BPE tokenizers. Install "
        "heliogram[baseline] (transformers) and run "
        "`python -m heliogram.baselines --measure` for a real measured tokenizer this function "
        "will then use instead."
    )


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
    is_text_payload: bool
    raw_text_token_est: Optional[int]
    patches_vs_raw_text_ratio: Optional[float]
    raw_text_method_note: str
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
      - `raw_text_token_est` / `is_text_payload` (BASELINE-HONESTY FIX, decision C3): base64/hex
        are the fair baseline ONLY for payloads that must be binary-to-text encoded to sit in a
        text context in the first place. If `payload` decodes as valid UTF-8 (checked here, not
        assumed), it is TEXT-LIKE, and the fair text-context comparison is the RAW TEXT ITSELF
        sitting in context -- nobody base64-encodes JSON before pasting it into a prompt. In that
        case `raw_text_token_est` is `_estimate_raw_text_tokens`'s token count for the actual
        decoded text (measured with a real tokenizer if available, else the standard ~4
        chars/token analytic estimate -- see that function's docstring), `is_text_payload=True`,
        and `patches_vs_raw_text_ratio` is set. For a genuinely binary payload (does not decode
        as UTF-8 -- e.g. `sample_binary_payload`'s output), there IS no fair raw-text baseline at
        all (you cannot paste arbitrary bytes into a text-only context without SOME binary-to-
        text encoding), so `is_text_payload=False` and `raw_text_token_est`/
        `patches_vs_raw_text_ratio` are both `None` -- `raw_text_method_note` explains why. This
        is precisely the payload shape (incompressible binary) this project's candidate niche
        targets; see `format_token_savings_report`'s printed honesty note.

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
        payload_text = payload.decode("utf-8")
        is_text_payload = True
    except UnicodeDecodeError:
        payload_text = None
        is_text_payload = False

    if is_text_payload:
        raw_text_token_est, raw_text_method_note = _estimate_raw_text_tokens(payload_text)
        patches_vs_raw_text_ratio: Optional[float] = (
            (total_patches / raw_text_token_est) if raw_text_token_est else None
        )
    else:
        raw_text_token_est = None
        patches_vs_raw_text_ratio = None
        raw_text_method_note = (
            "N/A: payload is not valid UTF-8, so it has no fair 'raw text in context' baseline "
            "at all -- you cannot paste arbitrary bytes into a text-only context without SOME "
            "binary-to-text encoding (base64/hex are exactly that encoding, and are already "
            "compared above). This IS the payload shape (incompressible binary) this project's "
            "candidate niche targets -- see the printed honesty note below."
        )

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
        is_text_payload=is_text_payload,
        raw_text_token_est=raw_text_token_est,
        patches_vs_raw_text_ratio=patches_vs_raw_text_ratio,
        raw_text_method_note=raw_text_method_note,
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
    ]
    if result.is_text_payload and result.raw_text_token_est is not None:
        lines.append(
            f"{'raw text tokens:':<23}{result.raw_text_token_est:>8}   "
            f"({result.raw_text_method_note})"
        )
    else:
        lines.append(f"{'raw text tokens:':<23}{'N/A':>8}   ({result.raw_text_method_note})")
    lines += [
        "",
        f"patches / base64 tokens: {result.patches_vs_base64_ratio:.3f}x  "
        f"({'heliogram CHEAPER' if result.patches_vs_base64_ratio < 1.0 else 'base64 cheaper'})",
        f"patches / hex tokens:    {result.patches_vs_hex_ratio:.3f}x  "
        f"({'heliogram CHEAPER' if result.patches_vs_hex_ratio < 1.0 else 'hex cheaper'})",
    ]
    if result.is_text_payload and result.patches_vs_raw_text_ratio is not None:
        lines.append(
            f"patches / raw text tokens: {result.patches_vs_raw_text_ratio:.3f}x  "
            f"({'heliogram CHEAPER' if result.patches_vs_raw_text_ratio < 1.0 else 'raw text cheaper'})"
        )
    lines += [
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

    lines.append("")
    if result.is_text_payload and result.patches_vs_raw_text_ratio is not None:
        if result.patches_vs_raw_text_ratio >= 1.0:
            lines.append(
                "HONESTY NOTE (decision C3): this payload is TEXT-LIKE (valid UTF-8 -- e.g. "
                "JSON), and for text-like payloads heliogram LOSES to plain text sitting "
                f"directly in context, by a wide margin ({result.patches_vs_raw_text_ratio:.1f}x "
                "more heliogram patches than raw-text tokens for this payload). Nobody is "
                "forced to base64/hex-encode JSON before pasting it into a prompt -- that "
                "comparison above is real, but it is not the fair one for THIS payload shape. "
                "heliogram's candidate niche is INCOMPRESSIBLE BINARY payloads that must sit "
                "verbatim in context (no text representation exists at all) -- see "
                "sample_binary_payload()/token_savings_demo() run on a binary payload, where "
                "this raw-text row is N/A by construction because no fair text baseline exists."
            )
        else:
            lines.append(
                "HONESTY NOTE (decision C3): this payload is TEXT-LIKE (valid UTF-8), and even "
                "so heliogram came out cheaper than raw text in context for this run "
                f"({result.patches_vs_raw_text_ratio:.3f}x) -- that would be a genuinely "
                "interesting result worth double-checking (the general expectation documented "
                "here is that heliogram loses to plain text for text-like payloads by a wide "
                "margin), not simply trusted at face value."
            )
    elif not result.is_text_payload:
        lines.append(
            "HONESTY NOTE (decision C3): this payload is NOT valid UTF-8 (genuinely binary), so "
            "there is no fair 'raw text in context' baseline to compare against at all -- the "
            "base64/hex comparisons above ARE the fair ones for this payload shape. This is "
            "heliogram's candidate niche: incompressible binary data that must sit verbatim in "
            "context. For text-like payloads (e.g. JSON), see the raw-text row: heliogram loses "
            "to plain text in context by a wide margin there, and this project says so plainly."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# 3. Cost asymmetry: equal token COUNT is not equal token COST
# --------------------------------------------------------------------------------------------


@dataclass
class CostAsymmetryPoint:
    """One point in the cost-asymmetry argument (Task 4, part 1): even where an image/merged
    vision token and a text token cost the SAME number of LM-context tokens, they are not
    compute/memory-equivalent -- the image path pays for a full vision-tower forward pass a text
    token never incurs. Same shape as `ExactnessPoint`: a claim, plus each side's story, plus a
    `status` tag. `status` is always `"structural"` here -- every claim below is true by
    construction of a ViT-encoder-plus-2x2-merger architecture (which Qwen2.5-VL, this project's
    pinned scope, documented-ly is) -- but anywhere a claim's MAGNITUDE (a FLOP count, a
    millisecond figure, a specific encoder depth, a serving stack's caching policy) would depend
    on Qwen2.5-VL's specific numbers or deployment, the text flags it inline as an explicit
    "(ASSUMPTION: ...)" aside rather than asserting a number. No FLOP/latency/memory figure is
    invented anywhere in this dataclass or `cost_asymmetry_points()`."""

    claim: str
    image_tokens: str
    text_tokens: str
    status: str


def cost_asymmetry_points() -> List[CostAsymmetryPoint]:
    """THE cost-asymmetry note (Task 4, part 1): a structural, model-free argument that
    RESULTS.md's per-2x2-merged-token accounting (lm_tokens_2x2 / lm_token_ratio -- NOT
    recomputed here; see that file for the actual counts) answers only "how many tokens", never
    "how much compute/memory". Even at EQUAL token COUNT, a heliogram image token is not
    free-equivalent to a text token, because reaching that merged token at all required routing
    pixels through Qwen2.5-VL's own frozen vision tower -- patch embed, N ViT transformer blocks,
    and the 2x2 patch merger -- BEFORE the LM ever sees it, while a text token is a direct
    embedding-table lookup. The three points below make that precise without inventing a single
    FLOP or millisecond number anywhere.
    """
    return [
        CostAsymmetryPoint(
            claim=(
                "ViT encoder overhead: image tokens pay for a full vision-tower forward pass "
                "that text tokens never incur"
            ),
            image_tokens=(
                "Every heliogram patch must be routed through Qwen2.5-VL's frozen vision tower "
                "(patch embed -> N ViT transformer blocks -> the 2x2 patch merger that folds "
                "four raw patches into one LM-visible token) before the merged token exists at "
                "all -- extra prefill compute and latency with no text-side equivalent step. "
                "(ASSUMPTION, model-specific: the actual FLOP count or millisecond cost of that "
                "pass depends on Qwen2.5-VL's specific encoder depth/width and the serving "
                "stack's batching/caching behavior -- no such number is asserted here, only that "
                "the step exists and a text token never pays it.)"
            ),
            text_tokens=(
                "A text token is an embedding-table lookup: O(1), no separate network forward "
                "pass, no analogous 'text tower' stage of any kind."
            ),
            status="structural",
        ),
        CostAsymmetryPoint(
            claim=(
                "KV-cache / activation footprint: the encoder's own activations are extra "
                "memory a text-only prompt never allocates"
            ),
            image_tokens=(
                "Producing each merged image token requires materializing the vision tower's "
                "own intermediate activations (patch embeddings, every ViT block's activations, "
                "the merger's inputs) -- memory a text-only forward pass never allocates at "
                "all. Once a merged token IS in the LM's context, its own per-token KV-cache "
                "entry is comparable in size to a text token's (same hidden dimension) -- the "
                "asymmetry is not in the LM's KV-cache line, it is in the encoder's OWN "
                "footprint sitting in front of it. (ASSUMPTION, model-specific: whether/how long "
                "the encoder's activations are retained -- freed immediately after the merger "
                "runs, or kept for some multi-turn image-caching optimization -- is a "
                "serving-stack decision this module does not assume a specific answer for.)"
            ),
            text_tokens=(
                "A text token's KV-cache entry is created directly from its embedding-table "
                "lookup -- no upstream encoder activations exist to allocate or free at all."
            ),
            status="structural",
        ),
        CostAsymmetryPoint(
            claim=(
                "Prefill attention compute: O(n^2) applies equally to both token kinds, so equal "
                "token COUNT is NOT equal token COST -- the load-bearing point"
            ),
            image_tokens=(
                "Once merged image tokens sit in the LM's context, self-attention over them "
                "costs the same O(n^2) in sequence length as attention over the same COUNT of "
                "text tokens -- LM-side attention compute per token is comparable, not "
                "penalized. But the image path additionally pays for the vision-tower forward "
                "pass (point 1 above) and its own activation footprint (point 2 above) BEFORE "
                "any of those tokens exist for the LM to attend over at all -- cost a text-only "
                "prompt of the identical final token count never incurs. So RESULTS.md's "
                "'fewer/comparable tokens than base64/hex' result is a token-COUNT claim; it "
                "does NOT by itself imply an equal or lower compute/memory COST -- the two are "
                "equal ONLY inside the LM's own attention layers, and the image path pays extra "
                "outside them, on top."
            ),
            text_tokens=(
                "A same-length text prompt pays exactly that O(n^2) LM-attention cost and "
                "nothing else -- there is no upstream stage to add anything on top of it."
            ),
            status="structural",
        ),
    ]


def format_cost_asymmetry(points: Optional[List[CostAsymmetryPoint]] = None) -> str:
    """Pretty-print `cost_asymmetry_points()`'s output as plain text."""
    if points is None:
        points = cost_asymmetry_points()
    lines = [
        "COST ASYMMETRY: equal token COUNT is not equal token COST (image vs. text tokens)",
        "=" * 78,
        "",
    ]
    for i, p in enumerate(points, 1):
        lines += [
            f"{i}. {p.claim}  [{p.status}]",
            f"   heliogram image tokens: {p.image_tokens}",
            f"   text tokens:            {p.text_tokens}",
            "",
        ]
    lines.append(
        "One line: self-attention is O(n^2) in sequence length for BOTH token kinds, so equal "
        "token count buys equal LM-attention cost -- but the image path pays its vision-tower "
        "encoder pass and activation footprint ON TOP of that, with no text-side analogue. "
        "'Cheaper on token count' (RESULTS.md) does not, by itself, mean 'cheaper on compute or "
        "memory'."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# 4. Effective cost per recovered bit -- HYPOTHETICAL, assumption-gated (Task 4, part 2)
# --------------------------------------------------------------------------------------------


def chance_level_symbol_error(palette: int) -> float:
    """Exact arithmetic, not a measurement: if a reader carries literally zero information about
    a patch's symbol and guesses uniformly at random among the `palette` possible values, it is
    wrong `1 - 1/palette` of the time. This is the SAME convention `heliogram.probe`'s
    `ProbeCellReport.chance_error` already uses (`1.0 - 1.0 / self.palette`) -- restated here,
    not imported, because this module is deliberately restricted to `heliogram.codec` /
    `heliogram.corruption` / `heliogram.baselines` imports only (see module docstring) and must
    not pull in `heliogram.probe`. Not a drifted second definition: same formula, same numbers
    (e.g. palette=256 -> 255/256 = 0.99609375, palette=16 -> 15/16 = 0.9375 -- matching
    docs/FINDINGS.md's post-merger probe table's "chance" column exactly at both palettes).
    """
    if palette < 1:
        raise ValueError(f"palette must be >= 1, got {palette!r}")
    return 1.0 - 1.0 / palette


@dataclass
class RecoveredBitResult:
    """One HYPOTHETICAL scenario for `effective_cost_per_recovered_bit`: what cost/recovered-bit
    WOULD be, IF a post-merger reader achieved `assumed_symbol_error`. See
    `format_effective_cost_per_recovered_bit`'s mandatory headline caveat: the stock, frozen
    Qwen2.5-VL tower's OWN measured post-merger symbol error is at/near chance (probe_report.md /
    docs/FINDINGS.md sec 3), i.e. it realizes NONE of this in practice -- every row this
    dataclass produces is a conditional projection, never a realized benefit."""

    payload_bits: int
    tokens: int
    bits_per_symbol: int
    nsym: int
    nsize: int
    assumed_symbol_error: float
    assumption_label: str
    rs_budget_fraction: float
    within_rs_budget: bool
    recovered_bit_fraction: float
    recovered_bits: int
    cost_per_recovered_bit: Optional[float]
    note: str


def effective_cost_per_recovered_bit(
    payload_bits: int,
    tokens: int,
    assumed_symbol_error: Optional[float],
    bits_per_symbol: int,
    nsym: int = 32,
    nsize: int = RS_NSIZE,
    assumption_label: str = "",
) -> RecoveredBitResult:
    """HONESTY-CRITICAL (Task 4, part 2): cost/recovered-bit = tokens / recovered_bits, where
    `recovered_bits` is computed from an EXPLICIT, CALLER-SUPPLIED `assumed_symbol_error` -- NOT
    a default that implies success. `assumed_symbol_error=None` (or omitted) raises ValueError:
    this function will not silently assume any reader recovers anything. This mirrors the reason
    this whole report exists: the stock, frozen Qwen2.5-VL tower's own measured post-merger
    symbol error is AT/NEAR CHANCE (probe_report.md / docs/FINDINGS.md sec 3) -- i.e. the real,
    measured number for "what does the stock tower achieve" is "effectively none of this", and
    asserting a default recovery rate here would silently launder that negative result away.

    Recovery model ("the code's own model", per the Task 4 spec): Reed-Solomon's guarantee
    (`rs_error_correction_capacity`) is a step function, not a graded one -- within its
    correction budget (`floor(nsym/2)` bytes per `nsize`-byte chunk) decode succeeds EXACTLY,
    deterministically; beyond it, decode detects-or-fails with overwhelming probability (see
    `rs_error_correction_capacity`'s docstring -- `decode_pixels` never returns a partial-credit
    payload either way). So this function treats `assumed_symbol_error` as a proxy for the
    fraction of RS-layer bytes it corrupts (ASSUMPTION: one misclassified symbol implies at
    least one corrupted packed byte, so a symbol-error rate is a lower bound on / stand-in for
    the byte-error rate RS actually sees -- not an exact mapping, but it does not require
    inventing any additional error-propagation model here) and compares it directly against the
    RS correction-budget fraction (`rs_budget_fraction = floor(nsym/2) / nsize`, ~6.27% at the
    codec's default nsym=32 -- the same figure docs/FINDINGS.md/README.md already cite):

      - `assumed_symbol_error <= rs_budget_fraction`: WITHIN budget -> the RS guarantee's exact
        half applies -- `recovered_bit_fraction = 1.0`, `recovered_bits = payload_bits`.
      - `assumed_symbol_error >  rs_budget_fraction`: beyond budget -> decode detects-or-fails
        with overwhelming probability and `decode_pixels` never returns a partial payload ->
        `recovered_bit_fraction = 0.0`, `recovered_bits = 0`, and `cost_per_recovered_bit` is
        `None` (undefined/infinite: tokens spent divided by zero bits recovered is not a number
        -- see `format_effective_cost_per_recovered_bit`, which prints this as "undefined
        (infinite) -- 0 bits recovered").

    `bits_per_symbol` is not used in the threshold arithmetic above (the RS budget is already
    stated in bytes, independent of palette) -- it is carried through purely for the report's own
    context line (translating `assumed_symbol_error` back to "at palette=2**bits_per_symbol,
    chance-level error would be ...", see `chance_level_symbol_error`), so a reader of the report
    can sanity-check an assumed error against the chance-level anchor for the SAME palette
    without re-deriving it.
    """
    if assumed_symbol_error is None:
        raise ValueError(
            "assumed_symbol_error must be given explicitly -- this function refuses to assume "
            "any recovery rate by default. The stock, frozen Qwen2.5-VL tower's own measured "
            "post-merger symbol error is at/near CHANCE (see probe_report.md / "
            "docs/FINDINGS.md sec 3), i.e. it realizes approximately NONE of the hypothetical "
            "benefit this function computes -- pass an explicit, labeled assumption (e.g. the "
            "RS correction budget as an optimistic 'if it worked' anchor via "
            "rs_error_correction_capacity(), or chance_level_symbol_error(palette) to see cost "
            "correctly blow up to undefined/infinite) instead of relying on a default."
        )
    if not (0.0 <= assumed_symbol_error <= 1.0):
        raise ValueError(
            f"assumed_symbol_error must be in [0.0, 1.0], got {assumed_symbol_error!r}"
        )
    if payload_bits < 0:
        raise ValueError(f"payload_bits must be >= 0, got {payload_bits!r}")
    if tokens <= 0:
        raise ValueError(f"tokens must be > 0, got {tokens!r}")
    if bits_per_symbol < 1:
        raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol!r}")

    rs = rs_error_correction_capacity(nsym=nsym, nsize=nsize)
    rs_budget_fraction = rs.max_correctable_byte_errors_per_chunk / rs.nsize
    within_budget = assumed_symbol_error <= rs_budget_fraction
    recovered_bit_fraction = 1.0 if within_budget else 0.0
    recovered_bits = payload_bits if within_budget else 0
    cost_per_recovered_bit = (tokens / recovered_bits) if recovered_bits > 0 else None

    note = (
        f"assumed_symbol_error={assumed_symbol_error:.4f} vs. rs_budget_fraction="
        f"{rs_budget_fraction:.4f} (floor({nsym}/2)/{nsize}) -> "
        f"{'WITHIN' if within_budget else 'BEYOND'} the RS correction budget -> "
        f"recovered_bit_fraction={recovered_bit_fraction:.1f}. "
        + (
            "cost_per_recovered_bit is a real, finite number under this ASSUMPTION only -- it "
            "is not observed on any real reader."
            if cost_per_recovered_bit is not None
            else "cost_per_recovered_bit is undefined/infinite: tokens were spent but the "
            "assumed error rate is beyond what this RS configuration can correct, so "
            "decode_pixels-style logic detects-or-fails and 0 bits are recovered."
        )
    )
    return RecoveredBitResult(
        payload_bits=payload_bits,
        tokens=tokens,
        bits_per_symbol=bits_per_symbol,
        nsym=nsym,
        nsize=nsize,
        assumed_symbol_error=assumed_symbol_error,
        assumption_label=assumption_label or "(unlabeled assumption)",
        rs_budget_fraction=rs_budget_fraction,
        within_rs_budget=within_budget,
        recovered_bit_fraction=recovered_bit_fraction,
        recovered_bits=recovered_bits,
        cost_per_recovered_bit=cost_per_recovered_bit,
        note=note,
    )


def format_effective_cost_per_recovered_bit(results: List[RecoveredBitResult]) -> str:
    """Pretty-print one or more `RecoveredBitResult` scenarios as plain text. The headline is the
    mandatory caveat (Task 4, part 2): this whole report is a CONDITIONAL projection under an
    ASSUMED recovery rate the stock tower does not achieve, never a realized benefit."""
    lines = [
        "EFFECTIVE COST PER RECOVERED BIT -- HYPOTHETICAL, ASSUMPTION-GATED (not a realized "
        "benefit)",
        "=" * 78,
        "",
        "This is a CONDITIONAL projection under an ASSUMED recovery rate that the stock frozen "
        "Qwen2.5-VL tower does NOT achieve -- the probe measured post-merger symbol error at/near "
        "chance (see probe_report.md / docs/FINDINGS.md §3). It is an upper bound on a "
        "hypothetical, not a realized benefit.",
        "",
    ]
    for r in results:
        palette_note = f"palette=2**{r.bits_per_symbol}={2 ** r.bits_per_symbol}"
        cost_str = (
            f"{r.cost_per_recovered_bit:.6f} tokens/bit"
            if r.cost_per_recovered_bit is not None
            else "undefined (infinite) -- 0 bits recovered"
        )
        lines += [
            f"scenario: {r.assumption_label}",
            f"  ASSUMPTION: assumed_symbol_error = {r.assumed_symbol_error:.4f} ({palette_note})",
            f"  RS correction budget: {r.rs_budget_fraction:.4f} (floor({r.nsym}/2)/{r.nsize})",
            f"  within RS budget?: {r.within_rs_budget}",
            f"  recovered_bit_fraction: {r.recovered_bit_fraction:.1f}  "
            f"(recovered_bits={r.recovered_bits} of payload_bits={r.payload_bits})",
            f"  tokens spent: {r.tokens}",
            f"  cost per recovered bit: {cost_str}",
            f"  {r.note}",
            "",
        ]
    lines.append(
        "Reminder: nothing above is measured on a real reader. The stock, frozen Qwen2.5-VL "
        "tower's measured post-merger symbol error is at/near chance for every palette tested "
        "(probe_report.md, probe_report_7b.md; see docs/FINDINGS.md §3) -- i.e. its OWN row "
        "in this table would show recovered_bit_fraction=0.0 and cost_per_recovered_bit "
        "undefined/infinite, same as the chance-level scenario above."
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

    # Two payload shapes, deliberately, per decision C3: a text-like JSON payload (where
    # heliogram is shown LOSING to plain text in context) and a genuinely binary payload (where
    # no raw-text baseline exists at all -- heliogram's candidate niche). Printing both, back to
    # back, is the point: this demo no longer only shows the comparison that flatters heliogram.
    json_payload = sample_structured_payload(seed=args.seed, target_bytes=args.target_bytes)
    json_result = token_savings_demo(
        json_payload, palette=args.palette, patch_size=args.patch_size, nsym=args.nsym
    )
    print("### Payload 1/2: text-like (JSON) ###")
    print(format_token_savings_report(json_result))

    binary_payload = sample_binary_payload(seed=args.seed, target_bytes=args.target_bytes)
    binary_result = token_savings_demo(
        binary_payload, palette=args.palette, patch_size=args.patch_size, nsym=args.nsym
    )
    print()
    print("### Payload 2/2: genuinely binary (deterministic random bytes) ###")
    print(format_token_savings_report(binary_result))

    print()
    print(format_exactness_argument())

    print()
    print(format_cost_asymmetry())

    # Two assumed-error scenarios, both computed fresh, right now, from real arithmetic on this
    # run's own binary-payload numbers (payload_bits/tokens) -- never invented: (1) the RS
    # correction budget itself as an optimistic "if it worked" anchor (assumed_symbol_error
    # exactly AT the budget -> just barely within it -> full hypothetical recovery), and (2) the
    # chance-level anchor for this run's palette (assumed_symbol_error = chance_level_symbol_error
    # (args.palette), the same "at/near chance" regime the probe actually measured -> cost blows
    # up to undefined/infinite). See format_effective_cost_per_recovered_bit's mandatory caveat.
    rs_guarantee = rs_error_correction_capacity(nsym=args.nsym)
    rs_budget_fraction = rs_guarantee.max_correctable_byte_errors_per_chunk / rs_guarantee.nsize
    bps = args.palette.bit_length() - 1  # log2(palette); palette is always a power of two here

    optimistic = effective_cost_per_recovered_bit(
        payload_bits=binary_result.payload_len * 8,
        tokens=binary_result.total_patches,
        assumed_symbol_error=rs_budget_fraction,
        bits_per_symbol=bps,
        nsym=args.nsym,
        assumption_label=(
            f"OPTIMISTIC anchor: assumed_symbol_error == the RS correction budget itself "
            f"({rs_budget_fraction:.4f}, i.e. 'if it worked exactly at the edge of the budget')"
        ),
    )
    chance = effective_cost_per_recovered_bit(
        payload_bits=binary_result.payload_len * 8,
        tokens=binary_result.total_patches,
        assumed_symbol_error=chance_level_symbol_error(args.palette),
        bits_per_symbol=bps,
        nsym=args.nsym,
        assumption_label=(
            f"CHANCE-LEVEL anchor: assumed_symbol_error == chance_level_symbol_error(palette="
            f"{args.palette}) -- the regime the probe actually measured (see probe_report.md)"
        ),
    )
    print()
    print(format_effective_cost_per_recovered_bit([optimistic, chance]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
