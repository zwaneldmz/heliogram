"""heliogram.patchsize -- audit heliogram.codec.PATCH_SIZE against a real model's ViT patch size,
instead of trusting a memorized constant.

WHY THIS MODULE EXISTS (handoff M0 guardrail #3: "verify the patch size empirically for the
chosen model; do NOT hardcode 14 from memory"). `heliogram.codec.PATCH_SIZE` (14px) is described
throughout this repo as "~1 ViT patch/token" for Qwen2-VL/Qwen2.5-VL-style models. That number is
easy to get subtly wrong from memory -- ViT patch sizes vary across model families (14, 16, and
32 all exist in the wild) -- and easy to let silently drift from what a real checkpoint's own
config actually says. This module gives that constant an auditable trail instead of a bare
assertion:

  - KNOWN_PATCH_SIZES is a small table of DOCUMENTED patch sizes, read off each model's public
    config/technical report. These are NOT values this repo has empirically re-derived --
    there is no GPU here to load a real checkpoint and check its actual vision tower.
  - verify_patch_size() upgrades a report from source="documented" to source="measured" the
    moment a caller hands it a real, already-loaded `processor` and/or `config` object (e.g.
    from `transformers`, loaded exactly as `scripts/train_qlora.py` would) -- it reads the patch
    size straight off that object's known attribute paths. This module itself never imports
    torch/transformers to perform that load; it only *reads attributes* off whatever object the
    caller already constructed and handed in, mirroring heliogram/vlm.py's boundary (model/
    processor are opaque objects supplied by the caller, never constructed here). That is the
    "lazy-GPU empirical path" in this module's one-line summary: the GPU/heavy-dependency work
    (actually loading a model/processor) happens lazily, in caller code, outside this module;
    this module stays exactly as import-light as heliogram.codec (no torch/transformers at
    module scope, in fact no torch/transformers ANYWHERE in this file).

DATA HONESTY: source="measured" is returned ONLY when a real processor/config object was
supplied AND a patch_size attribute was actually found on it -- never fabricated, never silently
defaulted to a documented value while claiming "measured". With no processor/config (the only
mode reachable in this CPU-only repo today), verify_patch_size() returns the documented value
with source="documented" and a `note` saying empirical verification needs a real model/processor.
If a processor/config WAS supplied but no patch_size attribute could be found anywhere expected,
this raises ValueError rather than quietly falling back to the documented table -- silently
downgrading an explicit empirical request to a canned number would be exactly the kind of
fabrication this project's data-honesty rule forbids.

CLI: `python3 -m heliogram.patchsize --model Qwen/Qwen2.5-VL-7B-Instruct` prints the documented
report for that model id and flags whether it matches heliogram.codec.PATCH_SIZE.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from .codec import PATCH_SIZE

__all__ = [
    "KNOWN_PATCH_SIZES",
    "PatchSizeReport",
    "known_patch_size",
    "verify_patch_size",
    "format_report",
    "build_parser",
    "main",
]

# DOCUMENTED patch sizes, read off each model's public vision-tower config (e.g. the
# `vision_config.patch_size` field of the model's `config.json` / technical report) -- NOT
# values measured against a real, loaded model in this repo (no GPU here; see module docstring).
# Both Qwen2-VL and Qwen2.5-VL use the same 14px ViT patch size in their published configs, which
# is exactly why heliogram.codec.PATCH_SIZE defaults to 14 -- this table is what that default is
# actually accountable to, instead of a bare "trust me" comment.
KNOWN_PATCH_SIZES: Dict[str, int] = {
    "Qwen/Qwen2.5-VL-7B-Instruct": 14,
    "Qwen/Qwen2-VL-7B-Instruct": 14,
}


@dataclass
class PatchSizeReport:
    """One model's patch-size finding. `source` is "documented" (read from KNOWN_PATCH_SIZES,
    no real model involved) or "measured" (read off a real, caller-supplied processor/config
    object's attributes). `matches_codec_default` compares against heliogram.codec.PATCH_SIZE,
    the constant this whole audit exists to keep honest."""

    model_id: str
    patch_size: int
    source: str
    matches_codec_default: bool
    note: str


def known_patch_size(model_id: str) -> int:
    """Look up the DOCUMENTED patch size for `model_id`. Raises KeyError (listing every known
    model id) if `model_id` has no entry -- callers who need an unlisted model must either add a
    documented entry here or pass a real `processor`/`config` to verify_patch_size() for
    empirical measurement instead."""
    try:
        return KNOWN_PATCH_SIZES[model_id]
    except KeyError:
        raise KeyError(
            f"no documented patch size known for model_id={model_id!r}. Known model ids: "
            f"{sorted(KNOWN_PATCH_SIZES)}. Add a documented (config-sourced) entry to "
            "KNOWN_PATCH_SIZES, or pass a real processor=/config= to verify_patch_size() for "
            "empirical measurement instead of a documented lookup."
        ) from None


def _guarded_attr_chain(obj: Any, *attrs: str) -> Optional[Any]:
    """Walk a dotted attribute chain (e.g. "image_processor", "patch_size"), returning None the
    moment any hop is missing or `obj` itself is None -- never raises AttributeError. This is the
    "guard attribute access" step verify_patch_size uses so a caller-supplied processor/config
    that simply doesn't have the expected shape degrades to "not found" rather than crashing."""
    current = obj
    for attr in attrs:
        if current is None:
            return None
        current = getattr(current, attr, None)
    return current


def verify_patch_size(
    model_id: str, processor: object = None, config: object = None
) -> PatchSizeReport:
    """Report heliogram's patch-size assumption for `model_id`, upgrading to an empirical
    ("measured") finding whenever a real `processor` and/or `config` object is supplied.

    - `processor`/`config` BOTH None (the default, and the only mode reachable in this repo's
      CPU-only environment): returns the DOCUMENTED value from KNOWN_PATCH_SIZES, source=
      "documented", with a note explaining empirical verification needs a real model/processor.
      Raises KeyError (via known_patch_size) if `model_id` is not in KNOWN_PATCH_SIZES.
    - Either `processor` or `config` given: tries, in order, `processor.image_processor.
      patch_size`, then `config.vision_config.patch_size`, then `config.vision_config.
      spatial_patch_size` (all attribute accesses guarded -- a missing hop just means "try the
      next path", never a crash). The first one found wins, source="measured". If NONE of these
      paths yield a value, raises ValueError -- this function never silently falls back to the
      documented table when the caller explicitly asked for an empirical check; a document-table
      fallback dressed up as a failed measurement would be exactly the fabrication this project's
      data-honesty rule forbids.

    `matches_codec_default` is `patch_size == heliogram.codec.PATCH_SIZE` either way.
    """
    empirical_requested = processor is not None or config is not None

    if not empirical_requested:
        patch_size = known_patch_size(model_id)
        source = "documented"
        note = (
            "no processor=/config= supplied -- this is the documented value from "
            "KNOWN_PATCH_SIZES (read off the model's published config, not measured here). "
            "Empirical verification requires a real, already-loaded model/processor (this "
            "repo has no GPU) -- load one lazily in caller code and pass it as processor=/"
            "config= to upgrade this report to source='measured'."
        )
    else:
        measured = _guarded_attr_chain(processor, "image_processor", "patch_size")
        path = "processor.image_processor.patch_size"
        if measured is None:
            measured = _guarded_attr_chain(config, "vision_config", "patch_size")
            path = "config.vision_config.patch_size"
        if measured is None:
            measured = _guarded_attr_chain(config, "vision_config", "spatial_patch_size")
            path = "config.vision_config.spatial_patch_size"
        if measured is None:
            raise ValueError(
                f"processor/config was supplied for model_id={model_id!r} but no patch_size "
                "attribute was found at any of: processor.image_processor.patch_size, "
                "config.vision_config.patch_size, config.vision_config.spatial_patch_size. "
                "Refusing to silently fall back to the documented table -- pass an object that "
                "actually exposes one of these attributes, or call verify_patch_size(model_id) "
                "with no processor/config for the documented value instead."
            )
        patch_size = int(measured)
        source = "measured"
        note = f"read empirically off the supplied object at `{path}`."

    return PatchSizeReport(
        model_id=model_id,
        patch_size=patch_size,
        source=source,
        matches_codec_default=(patch_size == PATCH_SIZE),
        note=note,
    )


def format_report(report: PatchSizeReport) -> str:
    """Plain-text pretty-printer for a PatchSizeReport, used by both the CLI and anyone
    debugging interactively."""
    match_str = "yes" if report.matches_codec_default else "NO -- MISMATCH"
    lines = [
        f"model_id:    {report.model_id}",
        f"patch_size:  {report.patch_size}px  (source: {report.source})",
        f"codec.PATCH_SIZE == {PATCH_SIZE}px -> matches: {match_str}",
        f"note:        {report.note}",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help="model id to look up, e.g. Qwen/Qwen2.5-VL-7B-Instruct (see KNOWN_PATCH_SIZES for "
        "the full documented list)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = verify_patch_size(args.model)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
