"""heliogram.baselines -- reference points to compare the codec's bits/patch against.

(a) base64_bits_per_token: the "cost" of shipping bytes as base64 text tokens (~6 bits/token
    analytically; accepts a real tokenizer for a measured value).
(b) MeasuredBase64Baseline / measure_base64_baseline / load_measured_baseline: a REAL,
    tokenizer-measured version of (a), persisted to heliogram/data/base64_baseline.json so the
    rest of this project (heliogram.harness, heliogram.benefit) can use a measured number instead
    of the ~6.0 bits/token analytic guess -- see measure_base64_baseline's docstring for why this
    exists: the council's #1 finding on this project was that the headline Bar A/C economic claim
    was a ratio of two UNMEASURED constants, and that the ~6.0 bits/token analytic figure errs in
    the direction ADVERSE to heliogram's claim (BPE merges commonly give base64 text MORE than 1
    char/token, which makes the true bits/token LOWER than 6.0, not higher -- i.e. base64 is
    probably cheaper in real tokenizers than the analytic estimate assumes, which cuts against
    heliogram's "we beat base64" argument, not for it).
(c) rendered_text_density: a model-free, geometric estimate of how densely typeset text packs
    into the SAME patch grid unit the codec uses. The *true* bits/patch for rendered text needs
    actual OCR by an un-fine-tuned VLM -- that is Phase 2 work and is not done here.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "Base64Baseline",
    "base64_bits_per_token",
    "MeasuredBase64Baseline",
    "MEASURED_BASELINE_PATH",
    "load_measured_baseline",
    "measure_base64_baseline",
    "RenderedTextDensity",
    "rendered_text_density",
]


@dataclass
class Base64Baseline:
    bits_per_token: float
    note: str


def base64_bits_per_token(
    tokenizer: object = None,
    sample_bytes: Optional[bytes] = None,
    seed: int = 0,
) -> Base64Baseline:
    """~6 bits/token analytic baseline for base64 text, or a real measurement if given a tokenizer.

    base64 uses a 64-symbol alphabet (log2(64) = 6 bits/char), and common tokenizers emit
    roughly one token per base64 character, so ~6 bits/token is the standard analytic estimate.

    If `tokenizer` is given (any object exposing `.encode(str) -> Sequence[int]`, e.g. a
    HuggingFace tokenizer, or a plain callable `str -> Sequence[int]`), this instead *measures*
    bits/token directly: it base64-encodes `sample_bytes` (4096 deterministic pseudo-random bytes
    by default, seeded by `seed`) and divides the original bit count by the resulting token count.

    This function is kept EXACTLY as the original analytic-default/single-tokenizer-measurement
    helper -- callers that want a real, persisted, multi-size/multi-seed measurement against a
    named tokenizer (with provenance) should use `measure_base64_baseline`/
    `load_measured_baseline` instead; this function remains the harness's CPU-only fallback when
    no measurement has been taken (see heliogram.harness.BASE64_BITS_PER_TOKEN).
    """
    if sample_bytes is None:
        rng = random.Random(seed)
        sample_bytes = bytes(rng.getrandbits(8) for _ in range(4096))

    if tokenizer is None:
        return Base64Baseline(
            bits_per_token=6.0,
            note=(
                "analytic: base64 alphabet size 64 -> log2(64)=6 bits/char; ~1 char/token for "
                "typical BPE tokenizers on base64 streams. Pass a real tokenizer for a measured "
                "value."
            ),
        )

    b64_text = base64.b64encode(sample_bytes).decode("ascii")
    token_ids = tokenizer.encode(b64_text) if hasattr(tokenizer, "encode") else tokenizer(b64_text)
    n_tokens = len(token_ids)
    if n_tokens == 0:
        raise ValueError("tokenizer produced zero tokens for the base64 sample")
    bits = len(sample_bytes) * 8
    return Base64Baseline(
        bits_per_token=bits / n_tokens,
        note=(
            f"measured: {n_tokens} tokens for {len(sample_bytes)} bytes ({bits} bits) of "
            "base64 text via the provided tokenizer"
        ),
    )


# --------------------------------------------------------------------------------------------
# Measured (persisted, multi-size/seed, provenance-tracked) base64 tokenizer baseline
# --------------------------------------------------------------------------------------------

MEASURED_BASELINE_PATH = Path(__file__).parent / "data" / "base64_baseline.json"


@dataclass
class MeasuredBase64Baseline:
    """A REAL, tokenizer-measured base64-bits/token baseline, pooled across multiple payload
    sizes and RNG seeds, with full provenance -- the artifact `measure_base64_baseline` writes
    and `load_measured_baseline` reads back.

    This is the fix for the council's #1 finding on this project: the headline Bar A/C economic
    claim ("heliogram beats base64-in-text-context") was previously a ratio of two UNMEASURED
    constants -- an analytic ~6 bits/patch codec estimate divided by an analytic ~6.0 bits/token
    base64 guess. This dataclass is the measured replacement for the second constant.

    Fields:
      bits_per_token    pooled bits/token across every (size, seed) sample measured (total
                         payload bits / total tokens emitted for their base64 encoding) -- THE
                         number other modules should read as "the measured base64 baseline".
      chars_per_token   pooled base64 CHARACTERS per token (not bytes) -- the direct measurement
                         of how many base64 alphabet characters the tokenizer's BPE merges pack
                         into one token. >1.0 here is exactly the adverse-to-heliogram direction
                         the council flagged: it means the analytic 6.0 bits/token (which assumes
                         ~1 char/token) OVERSTATES base64's real token cost.
      tokens_per_kb     pooled tokens per 1024 bytes of original (pre-base64) payload -- a
                         convenience figure for back-of-envelope context-budget math.
      tokenizer_id      the HuggingFace model/tokenizer id actually loaded (e.g.
                         "Qwen/Qwen2.5-VL-7B-Instruct").
      tokenizer_package  the installed package + version string that produced this measurement
                         (e.g. "transformers==4.51.0"), so a stale measurement is auditable
                         against a library upgrade that might change tokenization.
      sample_sizes      the payload sizes (in bytes) actually measured, e.g. [1024, 4096, 16384].
      per_size          {size (bytes): pooled bits/token at that size across all seeds} -- lets a
                         caller see whether bits/token is stable across payload size (it should
                         be, roughly, for a stationary random-byte source) or check for a specific
                         size's measurement rather than trusting only the pooled top-level figure.
      measured_note     a human-readable summary of how this number was produced, echoed by the
                         `.note` property below.
    """

    bits_per_token: float
    chars_per_token: float
    tokens_per_kb: float
    tokenizer_id: str
    tokenizer_package: str
    sample_sizes: List[int] = field(default_factory=list)
    per_size: Dict[int, float] = field(default_factory=dict)
    measured_note: str = ""

    @property
    def note(self) -> str:
        """Alias for `measured_note`. Fixed cross-group contract: callers elsewhere in this repo
        (heliogram.harness in particular) read `.note`, `.bits_per_token`, and `.tokenizer_id` off
        whatever baseline object they are handed (measured or analytic) without needing to know
        which dataclass they got -- `Base64Baseline` already has `.note`/`.bits_per_token`
        natively; this property gives `MeasuredBase64Baseline` the same two-attribute surface
        (tokenizer_id is a plain field on both this class and needed by name, not aliased)."""
        return self.measured_note


def load_measured_baseline() -> Optional[MeasuredBase64Baseline]:
    """Load a previously-measured base64 baseline from `MEASURED_BASELINE_PATH`
    (heliogram/data/base64_baseline.json), written by `measure_base64_baseline(write=True)`.

    Returns `None` -- NEVER raises -- if the file is missing, unreadable, not valid JSON, or does
    not contain the expected keys. This is a soft/optional read: every caller in this repo
    (heliogram.harness, heliogram.benefit) must keep working with the 6.0-bits/token analytic
    default (`base64_bits_per_token()`) when no measurement has ever been taken in this checkout,
    or transformers/network access was unavailable when Group C's measurement was attempted (see
    this module's `__main__` docstring and NOTES.md for whether that measurement actually
    succeeded in this repo's history) -- absence of the file is an expected, ordinary state, not
    an error condition.
    """
    try:
        raw = MEASURED_BASELINE_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
        per_size = {int(k): float(v) for k, v in data["per_size"].items()}
        return MeasuredBase64Baseline(
            bits_per_token=float(data["bits_per_token"]),
            chars_per_token=float(data["chars_per_token"]),
            tokens_per_kb=float(data["tokens_per_kb"]),
            tokenizer_id=str(data["tokenizer_id"]),
            tokenizer_package=str(data["tokenizer_package"]),
            sample_sizes=[int(s) for s in data["sample_sizes"]],
            per_size=per_size,
            measured_note=str(data["measured_note"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        # Malformed/hand-edited/partial file -- never fabricate a partial measurement out of it,
        # just report "no measurement available", identically to the file being absent.
        return None


def measure_base64_baseline(
    tokenizer_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    sizes: Sequence[int] = (1024, 4096, 16384),
    seeds: Sequence[int] = (0, 1, 2),
    write: bool = True,
) -> MeasuredBase64Baseline:
    """ACTUALLY measure base64 bits/token against a real BPE tokenizer, replacing the ~6.0
    bits/token analytic guess with a number this repo really computed.

    Loads `tokenizer_id` via `transformers.AutoTokenizer.from_pretrained` -- imported INSIDE this
    function, not at module scope, so `import heliogram.baselines` never requires transformers
    (same CPU-only-by-default import boundary as heliogram.vlm/heliogram.patchsize). If
    transformers is not installed, raises `RuntimeError` telling the caller to
    `pip install heliogram[baseline]` (or `pip install transformers`) -- this deliberately never
    falls back to the analytic 6.0 estimate under a "measured" label; that would defeat the whole
    point of this function (the council's #1 finding: the headline economic claim needs this
    number MEASURED, not assumed).

    For every `size` in `sizes` and `seed` in `seeds`: base64-encodes `size` deterministic
    pseudorandom bytes using the SAME sampling convention `base64_bits_per_token`'s default uses
    (`random.Random(seed).getrandbits(8)` for `size` bytes -- reused here, not redefined, so the
    two functions' "random bytes" are byte-identical for a shared (size, seed)), tokenizes the
    resulting base64 text with the real tokenizer, and records the token/char/bit counts.

    `per_size[size]` is bits/token POOLED across all seeds at that size (total bits / total
    tokens for that size, not a naive mean-of-per-seed-ratios, which would over-weight small
    samples). The top-level `bits_per_token`/`chars_per_token`/`tokens_per_kb` are pooled across
    EVERY (size, seed) sample together -- the single number other modules should read as "the
    measured base64 baseline".

    When `write=True` (the default), writes the full result plus provenance (tokenizer_id, the
    installed transformers version, sample_sizes, seeds, and a human-readable measured_note) to
    `MEASURED_BASELINE_PATH` as JSON, creating `heliogram/data/` if it does not exist yet.

    Needs real network access to HuggingFace Hub the first time `tokenizer_id` is loaded (its
    tokenizer.json/vocab/merges files, not model weights -- small, and cached locally after the
    first call). If that access is unavailable in a given environment, this raises whatever
    `transformers`/`huggingface_hub` raises (typically `OSError`) -- it is NOT caught and
    downgraded to a fabricated value here; callers/CLI users should see the real failure.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "measure_base64_baseline requires the 'transformers' package to load a real "
            f"tokenizer ({tokenizer_id!r}). Install it with `pip install heliogram[baseline]` "
            "(or `pip install transformers`) and retry. This function deliberately does NOT "
            "fall back to the analytic 6.0 bits/token estimate under a 'measured' label -- see "
            "this function's docstring for why that would defeat its whole purpose."
        ) from exc

    import transformers as _transformers_pkg

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    per_size_bits_per_token: Dict[int, float] = {}
    total_bits = 0
    total_tokens = 0
    total_chars = 0

    for size in sizes:
        size_bits = 0
        size_tokens = 0
        for seed in seeds:
            rng = random.Random(seed)
            sample_bytes = bytes(rng.getrandbits(8) for _ in range(size))
            b64_text = base64.b64encode(sample_bytes).decode("ascii")
            token_ids = tokenizer.encode(b64_text)
            n_tokens = len(token_ids)
            if n_tokens == 0:
                raise ValueError(
                    f"tokenizer {tokenizer_id!r} produced zero tokens for a {size}-byte "
                    f"base64 sample (seed={seed}) -- refusing to divide by zero"
                )
            bits = size * 8
            size_bits += bits
            size_tokens += n_tokens
            total_chars += len(b64_text)
        per_size_bits_per_token[size] = size_bits / size_tokens
        total_bits += size_bits
        total_tokens += size_tokens

    bits_per_token = total_bits / total_tokens
    chars_per_token = total_chars / total_tokens
    tokens_per_kb = total_tokens / (total_bits / 8 / 1024)
    tokenizer_package = f"transformers=={getattr(_transformers_pkg, '__version__', 'unknown')}"

    result = MeasuredBase64Baseline(
        bits_per_token=bits_per_token,
        chars_per_token=chars_per_token,
        tokens_per_kb=tokens_per_kb,
        tokenizer_id=tokenizer_id,
        tokenizer_package=tokenizer_package,
        sample_sizes=list(sizes),
        per_size=per_size_bits_per_token,
        measured_note=(
            f"measured: {tokenizer_id} tokenizer ({tokenizer_package}), {len(sizes)} payload "
            f"sizes x {len(seeds)} seeds each = {len(sizes) * len(seeds)} base64 samples "
            f"({list(sizes)} bytes, seeds {list(seeds)}), {total_tokens} tokens total for "
            f"{total_bits} bits of original payload -> {bits_per_token:.4f} bits/token "
            f"({chars_per_token:.4f} base64 chars/token, {tokens_per_kb:.2f} tokens/KB). "
            f"Compare to the {6.0:.1f} bits/token analytic default (base64_bits_per_token()): "
            f"{'LOWER' if bits_per_token < 6.0 else 'HIGHER'} than 6.0 means the analytic "
            f"estimate was {'adverse to' if bits_per_token < 6.0 else 'favorable to'} "
            "heliogram's economic claim -- BPE merges commonly give base64 text MORE than 1 "
            "char/token, which pushes bits/token below the naive log2(64)=6 estimate."
        ),
    )

    if write:
        MEASURED_BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bits_per_token": result.bits_per_token,
            "chars_per_token": result.chars_per_token,
            "tokens_per_kb": result.tokens_per_kb,
            "tokenizer_id": result.tokenizer_id,
            "tokenizer_package": result.tokenizer_package,
            "sample_sizes": result.sample_sizes,
            "seeds": list(seeds),
            "per_size": {str(k): v for k, v in result.per_size.items()},
            "measured_note": result.measured_note,
        }
        MEASURED_BASELINE_PATH.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return result


@dataclass
class RenderedTextDensity:
    image: Image.Image
    patches_used: int
    chars_per_patch: float
    bits_per_patch: float
    text_len: int
    note: str


def rendered_text_density(payload: bytes, patch_size: int = 14) -> RenderedTextDensity:
    """Typeset `payload` (base64-encoded, monospace) into an image using the SAME patch-size
    unit the codec uses, and measure a purely GEOMETRIC (model-free) density: how many patches
    of typeset text it takes to hold the payload, and the bits/patch that implies if every
    character were perfectly legible.

    This does NOT run OCR and is not a measurement of what an actual VLM can read off the image
    -- that requires the un-fine-tuned VLM's OCR accuracy, which is Phase 2 work (out of scope
    here, no GPU). Treat bits_per_patch here as an upper-bound / packing-density baseline only.
    """
    text = base64.b64encode(payload).decode("ascii")
    font = ImageFont.load_default()
    bbox = font.getbbox("M")
    char_w = max(1, bbox[2] - bbox[0])
    char_h = max(1, bbox[3] - bbox[1])

    # square-ish canvas, sized in whole patches, that roughly fits len(text) characters
    target_w_px = max(
        patch_size,
        math.ceil(math.sqrt(len(text)) * char_w / patch_size) * patch_size,
    )
    chars_per_line = max(1, target_w_px // char_w)
    n_lines = max(1, math.ceil(len(text) / chars_per_line))
    target_h_px = max(patch_size, math.ceil(n_lines * char_h / patch_size) * patch_size)

    img = Image.new("RGB", (target_w_px, target_h_px), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for line_idx in range(n_lines):
        line = text[line_idx * chars_per_line : (line_idx + 1) * chars_per_line]
        draw.text((0, line_idx * char_h), line, fill=(0, 0, 0), font=font)

    patches_w = target_w_px // patch_size
    patches_h = target_h_px // patch_size
    patches_used = patches_w * patches_h
    chars_per_patch = len(text) / patches_used
    bits_per_patch = chars_per_patch * 6.0  # base64: 6 bits/char

    return RenderedTextDensity(
        image=img,
        patches_used=patches_used,
        chars_per_patch=chars_per_patch,
        bits_per_patch=bits_per_patch,
        text_len=len(text),
        note=(
            "geometric/model-free: measures typeset packing density only, assumes perfect "
            "legibility. Real bits/patch for rendered text needs OCR accuracy from an "
            "un-fine-tuned VLM (Phase 2, out of scope here)."
        ),
    )


# --------------------------------------------------------------------------------------------
# CLI: `python -m heliogram.baselines --measure`
# --------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure (or inspect) heliogram's real, tokenizer-measured base64 bits/token "
            "baseline. Without --measure, prints whatever measurement (if any) is already "
            "persisted at heliogram/data/base64_baseline.json."
        )
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help=(
            "run measure_base64_baseline() for real (needs `pip install transformers` / "
            "`pip install heliogram[baseline]` and network access to HuggingFace Hub the first "
            "time the tokenizer is fetched) and write heliogram/data/base64_baseline.json"
        ),
    )
    parser.add_argument(
        "--tokenizer-id",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace tokenizer id to measure (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1024, 4096, 16384],
        help="payload sizes in bytes to sample (default: 1024 4096 16384)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="RNG seeds to sample per size (default: 0 1 2)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.measure:
        result = measure_base64_baseline(
            tokenizer_id=args.tokenizer_id, sizes=tuple(args.sizes), seeds=tuple(args.seeds)
        )
        print(f"measured bits/token: {result.bits_per_token:.4f}")
        print(f"measured chars/token: {result.chars_per_token:.4f}")
        print(f"tokens/KB: {result.tokens_per_kb:.2f}")
        print(f"tokenizer: {result.tokenizer_id} ({result.tokenizer_package})")
        print(f"wrote: {MEASURED_BASELINE_PATH}")
        return 0

    existing = load_measured_baseline()
    if existing is None:
        print(
            f"no measurement found at {MEASURED_BASELINE_PATH}; using the analytic 6.0 "
            "bits/token default. Run with --measure to produce a real one."
        )
        return 1
    print(existing.note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
