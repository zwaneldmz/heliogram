"""heliogram.vlm -- Phase-2 VLM decoder plug point (GPU work, no GPU in this repo's CPU env).

DATA HONESTY (read this first): nothing in this module has been run against a real model in
this repository -- there is no GPU here. `QwenVLDecoder.__call__` and `zero_shot_symbol_error`
both REQUIRE a real, already-loaded `model`/`processor` (e.g. from `transformers`); if either is
missing they raise immediately rather than fabricating a number or falling back to
`decode_pixels`. Every torch/transformers import in this file is local to the one method that
actually needs it (`QwenVLDecoder._generate`) -- this module's own top-level imports are exactly
as light as `heliogram.codec`'s (pillow/numpy/reedsolo + stdlib), so `import heliogram.vlm` (and
`import heliogram`, which re-exports it) never requires torch/transformers/peft/bitsandbytes.

What's actually implemented vs. what's untested:

- Parsing a model's text response into a symbol list, and feeding that symbol list through the
  exact Reed-Solomon/framing layer `heliogram.codec.decode_pixels` uses (`_payload_from_symbols`
  below) -- this is plain code, testable and tested on CPU (see tests/), no model required.
- The actual prompt wording (now the SAME canonical `heliogram.dataset.build_prompt` used by
  `scripts/train_qlora.py`'s training-target construction -- see "PROMPT UNIFICATION" below) and
  the `processor(...)`/`model.generate(...)` call sequence in `_generate` -- this follows the
  documented Hugging Face Qwen2-VL/Qwen2.5-VL chat-template pattern, but has never been run
  against the real model/processor. Treat it as a documented starting point to adjust once a GPU
  and the real `Qwen/Qwen2.5-VL-7B-Instruct` processor are on hand, not as a verified integration.

See `scripts/train_qlora.py` for how a real `model`/`processor` pair would be produced, and the
README's "Phase 2 (GPU)" section for the end-to-end flow.

THE BET this decoder exists to test (Slice C retarget -- see `heliogram/dataset.py`'s module
docstring for the full argument): whether a fine-tuned VLM can classify a BIG color palette
(`palette` in `{64, 128, 256}`) correctly at `subpatch=1` through the same realistic corruption
where `heliogram.codec.decode_pixels`'s nearest-neighbor classifier is MEASURED to fail (JPEG
q70, and at larger payloads JPEG q85 -- see RESULTS.md). That is why `QwenVLDecoder`'s own
default `palette` below is `256`, not a small palette: this class's whole reason to exist is
the regime the pixel decoder cannot handle, not the regime it already can. `subpatch` stays 1
throughout -- sub-patch geometry (`subpatch>1`) is a separate, documented, pixel-decoder-only
axis this decoder is not aimed at (see `codec.py`'s DATA HONESTY note).

PROMPT UNIFICATION (D5(b) of the Phase-2 scaffold review): `QwenVLDecoder._build_prompt` now
delegates entirely to `heliogram.dataset.build_prompt`, the SAME function
`scripts/train_qlora.py` calls to build training targets. Before this, the two were independently
-written strings that happened to describe the same task -- a real gap, since a fine-tuned model
is prompt-brittle (its weights were updated against ONE specific wording) and any drift between
training-time and inference-time wording is a silent capability tax paid at inference for no
reason. See `heliogram/dataset.py`'s module docstring for the full "PROMPT/OUTPUT-CONTRACT
UNIFICATION" note, including the row-per-line (not fenced-code-block) output contract change and
why (`SYMBOL_ALPHABET` includes the backtick character).

KNOWN, OUT-OF-SCOPE-FOR-THIS-MODULE GAP (see `heliogram/dataset.py`'s "PROCESSOR RESIZE HAZARD"
note): `heliogram.dataset.generate_examples`/`write_dataset` pad every TRAINING image to an even
patch-grid width/height so Qwen's `smart_resize` is the identity transform on it. `_generate`
below does NOT apply that same padding to `img` before calling `processor(...)` -- images reaching
`QwenVLDecoder` at INFERENCE time come straight from `heliogram.codec.encode()` (real deployment,
not this repo's training-data generator), which has no guarantee of even patch-grid dimensions.
A production `encode()` output with an odd patch-count dimension would still hit the same resize
hazard at inference time. Fixing this is out of scope for this group's assignment (scoped to
`scripts/train_qlora.py` + `heliogram/dataset.py` + `scripts/gen_dataset.py`); flagged here so it
is not mistaken for "fixed everywhere" -- a real deployment should pad (or otherwise guarantee
28px-alignment for) images before they reach `QwenVLDecoder`, mirroring
`heliogram.dataset.pad_to_even_patch_grid`.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from PIL import Image
from reedsolo import RSCodec

from .codec import (
    CODEC_VERSION,
    PATCH_SIZE,
    VALID_SUBPATCHES,
    HeliogramDecodeError,
    bits_per_symbol,
    encode,
    extract_symbols,
    rs_encoded_length,
)
from .dataset import (
    SYMBOL_ALPHABET,
    build_prompt,
    format_output_text,
    n_data_cells,
    parse_output_text,
    random_payload,
    target_to_symbols,
)

__all__ = [
    "QwenVLDecoder",
    "ZeroShotResult",
    "zero_shot_symbol_error",
    "expected_max_new_tokens",
    "teacher_forced_symbol_accuracy",
]


def _check_subpatch(patch_size: int, subpatch: int) -> None:
    if subpatch not in VALID_SUBPATCHES:
        raise ValueError(f"subpatch must be one of {VALID_SUBPATCHES}, got {subpatch!r}")
    if patch_size % subpatch != 0:
        raise ValueError(
            f"patch_size ({patch_size}) must be evenly divisible by subpatch ({subpatch})"
        )


def _payload_from_symbols(symbols: Sequence[int], palette: int, nsym: int) -> bytes:
    """Feed a symbol list through the exact same RS/framing layer `decode_pixels` uses, without
    re-deriving those symbols from pixels -- this is the seam that lets a VLM-produced symbol
    list (instead of `extract_symbols`' pixel classification) reach the same Reed-Solomon decode
    and frame-header parsing `heliogram/codec.py` already implements.

    Reuses `heliogram.codec`'s bit-packing helper (`_symbols_to_bytes`, imported here even
    though it is a private name) rather than re-deriving the MSB-first symbol-bits-to-bytes
    scheme independently: `heliogram/codec.py` is out of scope to edit in this slice and does
    not expose a public "symbols -> payload" entry point separate from the full
    pixel-sampling `decode_pixels` (which starts from an image, not an arbitrary symbol list).
    Duplicating that bit-packing arithmetic here would be exactly the kind of RS/framing
    reimplementation this module is meant to avoid.
    """
    from .codec import _symbols_to_bytes  # local: see docstring for why this private import

    bps = bits_per_symbol(palette)
    stream = _symbols_to_bytes(list(symbols), bps)

    if len(stream) < 5:
        raise HeliogramDecodeError("recovered stream shorter than the 5-byte framing header")

    payload_len = struct.unpack(">I", stream[1:5])[0]
    message_len = 5 + payload_len
    ecc_len = rs_encoded_length(message_len, nsym)
    if len(stream) < ecc_len:
        raise HeliogramDecodeError(
            f"recovered stream too short ({len(stream)}B) for the framed message "
            f"({ecc_len}B expected) -- VLM transcription likely incomplete or malformed"
        )

    ecc_bytes = bytes(stream[:ecc_len])
    try:
        decoded_message, _, _ = RSCodec(nsym).decode(ecc_bytes)
    except Exception as exc:  # reedsolo raises ReedSolomonError on uncorrectable input
        raise HeliogramDecodeError(f"Reed-Solomon decode failed: {exc}") from exc

    decoded_message = bytes(decoded_message)
    if len(decoded_message) < message_len:
        raise HeliogramDecodeError("decoded message shorter than the expected framing")
    if decoded_message[0] != CODEC_VERSION:
        raise HeliogramDecodeError(
            f"unsupported/corrupted codec version byte {decoded_message[0]!r} "
            f"(expected {CODEC_VERSION})"
        )
    return decoded_message[5 : 5 + payload_len]


def _extract_symbol_string(text: str) -> str:
    """Extract the transcribed symbol string from a model's raw text response: delegates to
    `heliogram.dataset.parse_output_text` (strip ALL whitespace, nothing else) -- see that
    function's docstring for why. Kept as a distinct name in this module purely for backward
    compatibility with existing internal callers/tests that reference
    `heliogram.vlm._extract_symbol_string` directly, not because the logic differs.

    HISTORY (D5(d) of the Phase-2 scaffold review): an earlier version of this function preferred
    the contents of a fenced code block (```...```) when present, because the earlier prompt
    wording (also since unified away, see `heliogram.dataset.build_prompt`) asked the model to
    wrap its answer in one. That heuristic was REMOVED, not merely deprioritized: `SYMBOL_ALPHABET`
    deliberately includes the backtick character (index 89, reachable once `palette` > 64), so a
    correct transcription of large-palette DATA can legitimately contain a run of three
    consecutive backtick symbols -- indistinguishable, by a fence-detecting regex, from the
    model's own closing fence, which silently truncated everything after it. The row-per-line,
    fence-free output contract `build_prompt` now specifies sidesteps the ambiguity entirely
    instead of trying to resolve it (newlines double as resync anchors a fence never provided).
    UNTESTED against real model output beyond this (see module docstring): instruction-following
    quirks (extra prose, stray punctuation) remain something only a real model run can surface.
    """
    return parse_output_text(text)


class QwenVLDecoder:
    """Phase-2 decoder plug point: wraps a (ideally QLoRA fine-tuned, see
    `scripts/train_qlora.py`) Qwen2.5-VL-style model to transcribe a heliogram symbol grid
    directly from pixels, in place of `decode_pixels`' nearest-neighbor pixel classifier.

    Callable with `(img, palette=, patch_size=, nsym=, subpatch=)` -- the same keywords
    `heliogram.codec.decode()` forwards to any `decoder=` callable -- so
    `decode(img, palette=P, subpatch=k, decoder=QwenVLDecoder(model=..., processor=...,
    palette=P, subpatch=k))` runs end to end: build a prompt describing the grid, call the
    model, parse its text response back into a symbol list (via
    `heliogram.dataset.target_to_symbols` -- the exact inverse of the compact target-string
    encoding `heliogram.dataset`/`scripts/train_qlora.py` use for ground truth/training
    targets), then feed that symbol list through `_payload_from_symbols` above (the same
    RS/framing layer `decode_pixels` uses -- no separate RS implementation for the model path).

    The constructor's `palette`/`subpatch`/`patch_size` are this decoder's fixed configuration
    (what the wrapped model was fine-tuned/prompted for); `__call__` validates that whatever
    `decode()` passes at call time matches, raising ValueError on a mismatch rather than
    silently using one value or the other. `nsym` is NOT required to match between construction
    and call time: it only affects the final RS-decode step (after transcription), not the
    prompt or parsing, so callers may freely vary it per call, exactly like `decode_pixels`.

    `palette` defaults to `256` (not `decode_pixels`' small default of `8`) -- see the module
    docstring's "THE BET" paragraph: this class exists specifically to test the large-palette
    regime the pixel decoder is measured to fail under corruption, so its own default should
    point at that regime, not an easy one `decode_pixels` already handles. Always pass the
    `palette`/`subpatch` your actual fine-tuned checkpoint was trained for, though -- these
    defaults are a documentation choice, not a substitute for matching your checkpoint.

    DATA HONESTY: calling an instance with `model=None` (the default) raises RuntimeError
    pointing at `scripts/train_qlora.py` -- there is no bundled model, and no code path here
    invents a result. See the module docstring for what is/isn't tested.
    """

    def __init__(
        self,
        model: object = None,
        processor: object = None,
        palette: int = 256,
        subpatch: int = 1,
        patch_size: int = PATCH_SIZE,
        nsym: int = 32,
        max_new_tokens: Optional[int] = None,
        token_margin: float = 3.0,
        device: Optional[str] = None,
    ) -> None:
        """`max_new_tokens` defaults to `None`, meaning "compute it per-image from the actual
        expected symbol count" -- see `expected_max_new_tokens` (D5(a) of the Phase-2 scaffold
        review). A previous fixed default of 2048 badly undersized generation for anything past a
        small payload (a P=256, 4096-byte payload needs ~4800+ output characters once
        Reed-Solomon framing overhead is included, guaranteeing truncated, undecodable output
        long before the model finished transcribing) -- pass an explicit int here only to
        override the dynamic sizing with a fixed budget (e.g. to cap GPU time for a known-small
        deployment), not as the normal path. `token_margin` (default 3.0x `n_data_cells`) is
        forwarded to `expected_max_new_tokens` when `max_new_tokens` is `None`; see that
        function's docstring for why 3x, not 1x, is the safety margin."""
        bits_per_symbol(palette)  # validates palette is in VALID_PALETTES
        _check_subpatch(patch_size, subpatch)
        self.model = model
        self.processor = processor
        self.palette = palette
        self.subpatch = subpatch
        self.patch_size = patch_size
        self.nsym = nsym
        self.max_new_tokens = max_new_tokens
        self.token_margin = token_margin
        self.device = device

    def _require_model(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "QwenVLDecoder has no fine-tuned model/processor loaded (model=None or "
                "processor=None). There is no model shipped in this repo -- fine-tune one with "
                "scripts/train_qlora.py (GPU required; see that script's top-of-file docstring "
                "for hardware expectations), then load the resulting checkpoint and pass it as "
                "model=/processor= here. To measure a STOCK (not fine-tuned) model instead, use "
                "zero_shot_symbol_error() with a real loaded model/processor."
            )

    def _build_prompt(self, width: int, height: int) -> str:
        """Instruction prompt asking the model to transcribe the data-cell grid. `width`/
        `height` are PATCH grid dimensions (as returned by `extract_symbols`), so the model is
        told exactly how many characters/lines to produce.

        Delegates entirely to `heliogram.dataset.build_prompt` -- see the module docstring's
        "PROMPT/OUTPUT-CONTRACT UNIFICATION" note (D5(b)/D5(d) of the Phase-2 scaffold review):
        this is THE canonical prompt wording, shared byte-for-byte with
        `scripts/train_qlora.py`'s training-target construction, not an independently-written
        string that merely happens to describe the same task. UNTESTED wording against a real
        model -- see module docstring."""
        return build_prompt(self.palette, width, height, self.subpatch)

    def _generate(self, img: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Run one VLM generation call. ALL torch/transformers imports are local to this method
        -- it is the only place in this module that can require those packages, and it is only
        ever reached after `_require_model` has confirmed a real model/processor were supplied.
        `max_new_tokens` is an explicit, required argument (not read off `self.max_new_tokens`
        directly) because the caller (`__call__`, `zero_shot_symbol_error`) is responsible for
        resolving `self.max_new_tokens`'s `None`-means-"size it dynamically" contract via
        `expected_max_new_tokens` first -- see `QwenVLDecoder.__init__`'s docstring.
        UNTESTED (see module docstring): this follows the documented HF Qwen2-VL/Qwen2.5-VL
        chat-template + processor + generate pattern, but has never been run against the real
        processor in this environment (no GPU here). `qwen_vl_utils.process_vision_info` may be
        worth using instead of the raw `images=[img]` call once this runs for real -- see
        requirements-gpu.txt.
        """
        import torch  # lazy: heavy GPU dep, see module docstring

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[chat_text], images=[img], return_tensors="pt")

        target_device = self.device or getattr(self.model, "device", None)
        if target_device is not None:
            inputs = {
                k: (v.to(target_device) if hasattr(v, "to") else v) for k, v in inputs.items()
            }

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[:, input_len:]
        decoded = self.processor.batch_decode(
            new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded[0]

    def _parse_symbols(self, text: str) -> List[int]:
        """Parse the model's raw text response into a symbol list, reusing
        `heliogram.dataset.target_to_symbols`. Raises HeliogramDecodeError (not a bare ValueError)
        on any failure, consistent with `decode_pixels`' error contract."""
        candidate = _extract_symbol_string(text)
        try:
            return target_to_symbols(candidate, self.palette)
        except ValueError as exc:
            raise HeliogramDecodeError(
                f"could not parse the model's response into a valid symbol sequence: {exc}"
            ) from exc

    def __call__(
        self,
        img: Image.Image,
        palette: int = 256,
        patch_size: int = PATCH_SIZE,
        nsym: int = 32,
        subpatch: int = 1,
    ) -> bytes:
        self._require_model()
        if palette != self.palette or subpatch != self.subpatch or patch_size != self.patch_size:
            raise ValueError(
                f"QwenVLDecoder was constructed for palette={self.palette}, subpatch="
                f"{self.subpatch}, patch_size={self.patch_size}, but was called with "
                f"palette={palette}, subpatch={subpatch}, patch_size={patch_size}. Construct a "
                "decoder matching the image's actual encoding, or call decode()/decode_pixels "
                "with matching palette=/subpatch=/patch_size= arguments."
            )
        width = img.width // patch_size
        height = img.height // patch_size
        prompt = self._build_prompt(width, height)
        max_new_tokens = self.max_new_tokens or expected_max_new_tokens(
            width, height, subpatch, margin=self.token_margin
        )
        text = self._generate(img, prompt, max_new_tokens=max_new_tokens)
        symbols = self._parse_symbols(text)
        return _payload_from_symbols(symbols, palette=palette, nsym=nsym)


def expected_max_new_tokens(
    width: int, height: int, subpatch: int = 1, margin: float = 3.0
) -> int:
    """Size a generation token budget from the ACTUAL expected transcription length, instead of
    a fixed guess (D5(a) of the Phase-2 scaffold review). `width`/`height` are PATCH grid
    dimensions (as returned by `extract_symbols`/passed to `_build_prompt`); the number of
    characters a correct transcription contains is `heliogram.dataset.n_data_cells(width, height,
    subpatch)` -- the SAME formula `build_prompt` uses to tell the model how much to output, so
    the two can never silently drift apart (see `n_data_cells`'s own docstring).

    `margin` (default 3.0x `n_data_cells`, plus a small fixed constant for the row-per-line
    output's `height - 2` newline separators -- see below): a correct transcription of a P<=64
    (base64-range) grid is one token per character for most BPE tokenizers, but
    `SYMBOL_ALPHABET`'s characters past index 63 (used once `palette` > 64) are NOT guaranteed
    single-token in an arbitrary vocabulary -- punctuation/Latin-Extended-A code points are less
    likely to have a dedicated single-token vocabulary entry than ASCII letters/digits. A flat
    3x margin is a documented, UNVERIFIED (no GPU/real tokenizer here to measure actual token
    counts for `SYMBOL_ALPHABET`, see module docstring) safety factor -- generous enough to
    survive some worst-case multi-token characters without being so large it wastes GPU time
    generating far past a correct response. Revisit with real measurements
    (`processor.tokenizer.encode(ch)` over every `SYMBOL_ALPHABET` character) once a GPU and the
    real Qwen tokenizer are on hand.

    A previous fixed default (`QwenVLDecoder(max_new_tokens=2048)`) badly undersized this for
    anything past a small payload: a P=256, 4096-byte payload needs roughly 4800+ output
    characters (`n_data_cells` scales with payload size plus Reed-Solomon parity overhead),
    guaranteeing truncated, undecodable output well before the model finished transcribing --
    exactly the bug this function exists to fix. See `QwenVLDecoder.__init__`'s docstring for how
    this is wired in as the `max_new_tokens=None` default.
    """
    cells = n_data_cells(width, height, subpatch)
    newlines = max(0, height - 2)  # format_output_text inserts height-2 '\n' separators
    return max(1, math.ceil((cells + newlines) * margin))


@dataclass
class ZeroShotResult:
    """One config's worth of `zero_shot_symbol_error` results, averaged over `trials`
    payloads. `raw_responses` keeps every raw model response for manual inspection -- there is
    no other record of what the stock model actually said."""

    palette: int
    subpatch: int
    patch_size: int
    payload_size: int
    trials: int
    symbol_error_rate: float
    decode_success_rate: float
    raw_responses: List[str] = field(default_factory=list)


def _resolve_zero_shot_config(cfg: Dict[str, object]) -> Dict[str, int]:
    """Resolve one `zero_shot_symbol_error` config dict's defaults (`"palette"` required;
    `"subpatch"`, `"patch_size"`, `"nsym"`, `"payload_size"` optional). Factored out as a small
    pure function -- no model/processor involved -- specifically so its defaults (in particular
    `payload_size`'s fallback of 1024, D5(a) of the Phase-2 scaffold review, changed from an
    earlier 4096) are unit-testable on CPU without a model; see `zero_shot_symbol_error`'s
    docstring for why 1024 is the current fallback."""
    return {
        "palette": int(cfg["palette"]),  # type: ignore[arg-type]
        "subpatch": int(cfg.get("subpatch", 1)),  # type: ignore[arg-type]
        "patch_size": int(cfg.get("patch_size", PATCH_SIZE)),  # type: ignore[arg-type]
        "nsym": int(cfg.get("nsym", 32)),  # type: ignore[arg-type]
        "payload_size": int(cfg.get("payload_size", 1024)),  # type: ignore[arg-type]
    }


def zero_shot_symbol_error(
    model: object,
    processor: object,
    configs: Sequence[Dict[str, object]],
    n_trials: int = 3,
    seed: int = 0,
    max_new_tokens: Optional[int] = None,
) -> List[ZeroShotResult]:
    """Phase-2 "Step 8": run a STOCK (not fine-tuned) model directly over heliogram-encoded
    images and measure raw per-symbol transcription error against ground truth, the same way
    `heliogram.harness` measures `symbol_error_rate` for `decode_pixels` -- ground truth is read
    straight off the clean image at encode time via `extract_symbols` (exact by construction).

    DATA HONESTY (the entire point of this function): `model`/`processor` MUST be real, already
    -loaded objects (e.g. a stock `Qwen/Qwen2.5-VL-7B-Instruct` loaded via `transformers`,
    exactly as `scripts/train_qlora.py` would load it, just without the LoRA fine-tune step).
    Passing `model=None` or `processor=None` raises ValueError immediately -- there is no
    fallback path, default value, or cached result that could return an invented number. Every
    number in the returned `ZeroShotResult`s comes from an actual call to `model.generate(...)`
    (via `QwenVLDecoder._generate`, reused here so the exact same prompt/parsing code measured
    is the exact same code `QwenVLDecoder` uses for real decoding -- not a separate, untested
    measurement path). This repo's CPU pytest environment has no GPU, so this function itself is
    only ever unit-tested here with the `model=None`/`processor=None` guard, never with a real
    forward pass -- see tests/test_phase2_scaffold.py.

    `configs` is a sequence of dicts, each with at least a `"palette"` key (subset of
    `heliogram.codec.VALID_PALETTES`) and optionally `"subpatch"`, `"patch_size"`, `"nsym"`,
    `"payload_size"` (defaults: 1, `PATCH_SIZE`, 32, 1024 -- see below). Returns one
    `ZeroShotResult` per config, in the same order, each averaged over `n_trials` random payloads
    (seeded from `seed`, deterministic in the *inputs* generated -- not in the model's output,
    which is outside this function's control). `max_new_tokens` (default `None`) is forwarded to
    each `QwenVLDecoder`, which -- per D5(a) of the Phase-2 scaffold review -- means "size it
    per-image from the actual expected symbol count" (`expected_max_new_tokens`) rather than a
    fixed guess; see `QwenVLDecoder.__init__`'s docstring.

    Per the Slice C retarget (see `heliogram.dataset`'s module docstring), the recommended
    `configs` for this project's actual open research question are `palette` in `{64, 128, 256}`
    at a range of payload sizes spanning the low-KB regime (e.g. `[{"palette": p, "payload_size":
    s} for p in (64, 128, 256) for s in (1024, 4096, 16384)]`) -- mirroring exactly the
    (palette, payload_size) cells `heliogram.harness`'s own sweep measures `decode_pixels` to
    clean-decode but fail under JPEG q70/q85 on (see RESULTS.md's "Token crossover" section), so
    a zero-shot (and later fine-tuned) VLM's numbers land in the same table as the pixel
    decoder's for a direct before/after comparison. USE THAT EXPLICIT LIST for the actual claim
    this project cares about -- the `payload_size` FALLBACK below (used only when a config omits
    `"payload_size"` entirely) is 1024, not 4096: `max_new_tokens` is now sized correctly for any
    payload (see above), but ZERO-SHOT (no fine-tuning) evaluation of a STOCK model is exactly
    the setting where free-running generation drift is most likely across a LONG output (a
    4096-byte, P=256 payload needs ~4800+ correctly-transcribed characters merely to feed a
    well-formed stream into RS decode -- see `teacher_forced_symbol_accuracy` for a metric that
    does not have this length dependency at all). Keeping the bare/no-override default bounded at
    a length a stock model has a realistic chance of getting through raises the odds a zero-shot
    number reflects perception quality rather than "ran out of budget/drifted partway through a
    long document" -- this is a change to the CHEAPEST-default fallback only, not a claim that
    1024B is the interesting regime (it is not; see the explicit `configs` list above for that).
    """
    if model is None or processor is None:
        raise ValueError(
            "zero_shot_symbol_error requires a real, already-loaded model and processor (got "
            "model=None or processor=None) -- it never fabricates results. Load a stock "
            "Qwen2.5-VL-7B-Instruct (or similar) via transformers first; see "
            "scripts/train_qlora.py's docstring for the expected package versions."
        )

    results: List[ZeroShotResult] = []
    for cfg in configs:
        resolved = _resolve_zero_shot_config(cfg)
        palette = resolved["palette"]
        subpatch = resolved["subpatch"]
        patch_size = resolved["patch_size"]
        nsym = resolved["nsym"]
        payload_size = resolved["payload_size"]

        decoder = QwenVLDecoder(
            model=model,
            processor=processor,
            palette=palette,
            subpatch=subpatch,
            patch_size=patch_size,
            nsym=nsym,
            max_new_tokens=max_new_tokens,
        )

        symbol_errors = 0
        symbol_total = 0
        successes = 0
        responses: List[str] = []
        for trial in range(n_trials):
            payload = random_payload(seed + trial, payload_size)
            img = encode(
                payload,
                palette=palette,
                patch_size=patch_size,
                nsym=nsym,
                seed=0,
                subpatch=subpatch,
            )
            _, _, truth = extract_symbols(
                img, palette=palette, patch_size=patch_size, subpatch=subpatch
            )

            width = img.width // patch_size
            height = img.height // patch_size
            prompt = decoder._build_prompt(width, height)
            call_max_new_tokens = max_new_tokens or expected_max_new_tokens(
                width, height, subpatch
            )
            text = decoder._generate(
                img, prompt, max_new_tokens=call_max_new_tokens
            )  # real forward pass -- see docstring
            responses.append(text)

            try:
                observed = decoder._parse_symbols(text)
            except HeliogramDecodeError:
                observed = []

            n = min(len(truth), len(observed))
            symbol_errors += sum(1 for i in range(n) if truth[i] != observed[i])
            symbol_errors += abs(len(truth) - len(observed))  # length mismatch counts as error
            symbol_total += len(truth)

            try:
                if _payload_from_symbols(observed, palette=palette, nsym=nsym) == payload:
                    successes += 1
            except HeliogramDecodeError:
                pass

        results.append(
            ZeroShotResult(
                palette=palette,
                subpatch=subpatch,
                patch_size=patch_size,
                payload_size=payload_size,
                trials=n_trials,
                symbol_error_rate=(symbol_errors / symbol_total) if symbol_total else 0.0,
                decode_success_rate=(successes / n_trials) if n_trials else 0.0,
                raw_responses=responses,
            )
        )
    return results


def teacher_forced_symbol_accuracy(
    model: object,
    processor: object,
    img: Image.Image,
    target: str,
    palette: int,
    patch_size: int = PATCH_SIZE,
    subpatch: int = 1,
) -> float:
    """PRIMARY Phase-2 perception metric (D5(c) of the Phase-2 scaffold review -- see the module
    docstring's "THE BET" paragraph): TEACHER-FORCED per-symbol accuracy, deliberately NOT
    free-running generation accuracy.

    WHY THIS METRIC EXISTS, AND WHY IT IS THE RECOMMENDED PRIMARY ONE: `QwenVLDecoder.__call__`
    and `zero_shot_symbol_error` both measure accuracy via FREE-RUNNING generation -- the model
    produces its own tokens autoregressively, conditioning each next-symbol prediction on its OWN
    previous output rather than ground truth. A single early misclassification can therefore
    cascade into position drift that has nothing to do with whether the model's visual
    PERCEPTION of any individual cell's color is correct. Free-running exact-transcription
    accuracy CONFLATES two different failure modes: (1) "the model misjudged this cell's color"
    (a perception failure -- what this project's actual bet, per the module docstring, is about)
    and (2) "the model's own prior mistake threw off its position tracking / it lost its place in
    the grid" (a sequence-modeling/robustness failure, orthogonal to color perception). Reed-
    Solomon (the codec's own error-correction layer) makes this worse to reason about from a
    free-running number alone: RS corrects SUBSTITUTION errors (wrong symbol value at a KNOWN
    position) but not INSERTION/DELETION (extra or missing symbols shifting every subsequent
    position) -- exactly the failure mode generation drift produces, and exactly why
    `zero_shot_symbol_error` has to fall back on a crude "length mismatch counts as error"
    bookkeeping (see that function's body) for anything past the first misalignment.

    Teacher forcing sidesteps both problems: it feeds the model the GROUND-TRUTH target as
    context in ONE forward pass, then reads off the model's OWN predicted-token distribution at
    each target position independently -- an error at position i never contaminates position
    i+1's score, and there is no length-mismatch/resync question at all (there is exactly one
    prediction per ground-truth position, by construction). The result is a clean, position-
    independent per-symbol accuracy that isolates PERCEPTION specifically -- this is why
    `scripts/train_qlora.py`'s per-stage held-out evaluation calls this function, not a
    generate-and-compare loop. Free-running metrics remain necessary too (end-to-end deployment
    IS free-running), but should be read as "perception AND sequence robustness combined", not
    perception alone.

    HOW: builds the canonical prompt (`heliogram.dataset.build_prompt`) and the canonical
    row-per-line response text (`heliogram.dataset.format_output_text`) for `target`, forms ONE
    teacher-forced forward pass over prompt+response, then at each of the `n_data_cells` response
    positions restricts the model's next-token logits to the `palette` candidate token ids for
    `SYMBOL_ALPHABET[:palette]` (i.e. "which of the P possible color symbols did the model
    consider most likely here", not "which of the model's ~150K-token vocabulary" -- exactly the
    valid output set `build_prompt` already tells the model) and takes the argmax among just
    those, comparing against the true symbol at that position. Returns the fraction correct
    (0.0-1.0); returns 0.0 for an image with zero data cells (should be unreachable given
    `build_prompt`'s own `height >= 2` requirement).

    STRONG, EXPLICIT ASSUMPTIONS (UNTESTED -- no GPU/real tokenizer here to verify, see module
    docstring) -- this function raises ValueError IMMEDIATELY, loudly, and specifically (never
    silently misaligns and returns a wrong number) if any of these do not hold for the real
    tokenizer/processor in use:
      (a) every character in `SYMBOL_ALPHABET[:palette]` tokenizes to EXACTLY one token;
      (b) the row-per-line response text (all `palette`-alphabet characters plus '\\n' row
          separators) tokenizes to exactly one token per character, with NO cross-character
          merging -- checked via `len(tokenize(response_text)) == len(response_text)`, since (a)
          alone does not rule out a BPE merge across two ADJACENT single-token characters;
      (c) the prompt/response token boundary is PREFIX-CONSISTENT -- tokenizing "prompt-only"
          text is a token-for-token prefix of tokenizing "prompt+response" text. This is the same
          simplifying assumption `scripts/train_qlora.py`'s label-masking (`_mask_prompt_tokens`)
          makes; verified here via an explicit equality check against the response's own
          standalone tokenization.
    """
    if model is None or processor is None:
        raise ValueError(
            "teacher_forced_symbol_accuracy requires a real, already-loaded model and processor "
            "(got model=None or processor=None) -- it never fabricates results, same contract "
            "as zero_shot_symbol_error; see this module's docstring."
        )

    width = img.width // patch_size
    height = img.height // patch_size
    if width < 1 or height < 2:
        raise ValueError(f"image too small for patch_size={patch_size}: {img.size}")

    expected_len = n_data_cells(width, height, subpatch)
    if len(target) != expected_len:
        raise ValueError(
            f"target has {len(target)} characters but the image's grid implies {expected_len} "
            f"data cells (width={width}, height={height}, subpatch={subpatch}) -- target must be "
            "the exact ground-truth transcription for THIS image (e.g. from generate_examples)"
        )

    alphabet_prefix = SYMBOL_ALPHABET[:palette]
    tokenizer = getattr(processor, "tokenizer", processor)

    char_token_ids: List[int] = []
    for ch in alphabet_prefix:
        ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"alphabet character {ch!r} (palette={palette}) does not tokenize to exactly one "
                f"token with this tokenizer (got {ids!r}) -- teacher_forced_symbol_accuracy "
                "requires a 1-char-1-token alphabet subset to align target positions with logit "
                "positions; see this function's docstring's assumption (a)."
            )
        char_token_ids.append(ids[0])

    prompt = build_prompt(palette, width, height, subpatch)
    response_text = format_output_text(target, width, subpatch)

    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if len(response_ids) != len(response_text):
        raise ValueError(
            f"row-per-line response text tokenized to {len(response_ids)} tokens but has "
            f"{len(response_text)} characters -- teacher_forced_symbol_accuracy requires exact "
            "1-char-1-token alignment (no cross-character merging) for this tokenizer/text; see "
            "this function's docstring's assumption (b)."
        )

    prompt_messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}],
        }
    ]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
    ]
    prompt_chat_text = processor.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    full_chat_text = processor.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_inputs = processor(text=[prompt_chat_text], images=[img], return_tensors="pt")
    full_inputs = processor(text=[full_chat_text], images=[img], return_tensors="pt")

    prompt_len = int(prompt_inputs["input_ids"].shape[1])
    full_input_ids = full_inputs["input_ids"][0]

    actual_response_ids = full_input_ids[prompt_len : prompt_len + len(response_ids)].tolist()
    if actual_response_ids != response_ids:
        raise ValueError(
            "the full prompt+response tokenization's suffix does not match the response text's "
            "own standalone tokenization -- the prefix-consistency assumption this function "
            "relies on (see docstring's assumption (c)) does not hold for this tokenizer/prompt/"
            "response combination; teacher-forced per-symbol scoring cannot safely align "
            "positions here."
        )

    import torch  # lazy: heavy GPU dep, see module docstring -- deferred until every pure-Python
    # guard clause above (model/processor presence, target-length, tokenizer-alignment checks)
    # has already had a chance to raise without requiring torch to be installed.

    with torch.no_grad():
        logits = model(**full_inputs).logits[0]  # (seq_len, vocab)

    true_symbols = target_to_symbols(target, palette)
    row_len = width * subpatch * subpatch
    correct = 0
    total = 0
    char_pos = 0  # offset within response_text/response_ids
    for i, true_symbol in enumerate(true_symbols):
        if i > 0 and i % row_len == 0:
            char_pos += 1  # skip the '\n' row separator format_output_text inserted here
        predict_pos = prompt_len + char_pos - 1  # logits[t] predicts the token at t+1
        candidate_logits = logits[predict_pos, char_token_ids]
        predicted_symbol = int(torch.argmax(candidate_logits).item())
        if predicted_symbol == true_symbol:
            correct += 1
        total += 1
        char_pos += 1

    return (correct / total) if total else 0.0
