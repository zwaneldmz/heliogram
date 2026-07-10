"""heliogram demo: encode a small JSON doc, decode it back, and compare footprint to base64 tokens.

Run: python demo.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

from heliogram import decode_pixels, encode
from heliogram.codec import PATCH_SIZE

PALETTE = 8
NSYM = 32
OUTPUT_PATH = Path(__file__).with_name("demo_output.png")


def main() -> None:
    doc = {
        "id": 4271,
        "kind": "sensor_reading",
        "location": {"lat": -33.9249, "lon": 18.4241},
        "tags": ["heliogram", "vlm", "context-compression"],
        "readings": [21.4, 21.6, 21.9, 22.0, 21.8],
        "active": True,
    }
    payload = json.dumps(doc, separators=(",", ":")).encode("utf-8")

    img = encode(payload, palette=PALETTE, patch_size=PATCH_SIZE, nsym=NSYM, seed=0)
    img.save(OUTPUT_PATH)

    decoded = decode_pixels(img, palette=PALETTE, patch_size=PATCH_SIZE, nsym=NSYM)
    assert decoded == payload, "roundtrip failed!"

    width_patches = img.width // PATCH_SIZE
    height_patches = img.height // PATCH_SIZE
    total_patches = width_patches * height_patches

    b64_len = len(base64.b64encode(payload))
    payload_bits = len(payload) * 8
    effective_bits_per_patch = payload_bits / total_patches
    base64_bits_per_token = 6.0  # analytic baseline, see heliogram.baselines

    print(f"payload:              {len(payload)} bytes of JSON")
    print(
        f"saved image:          {OUTPUT_PATH.name} ({img.width}x{img.height}px, "
        f"{width_patches}x{height_patches} patches)"
    )
    print("roundtrip:            OK (decoded payload == original payload)")
    print()
    print(f"heliogram patches:    {total_patches}   (~1 ViT token/patch)")
    print(f"base64 tokens (est):  {b64_len}   (~1 token/char analytic baseline, 6 bits/char)")
    print()
    print(
        f"effective bits/patch: {effective_bits_per_patch:.2f}  "
        f"(payload bits / total patches, includes calibration-row + Reed-Solomon overhead)"
    )
    print(f"base64 bits/token:    {base64_bits_per_token:.2f}  (analytic baseline)")
    verdict = "beats" if effective_bits_per_patch > base64_bits_per_token else "is below"
    print(
        f"-> heliogram {verdict} the base64 baseline for this {len(payload)}-byte payload "
        f"({effective_bits_per_patch / base64_bits_per_token:.2f}x). Fixed overhead (32B RS "
        "parity, calibration row) dominates small payloads -- see heliogram.harness for how "
        "this scales with payload size and corruption."
    )


if __name__ == "__main__":
    main()
