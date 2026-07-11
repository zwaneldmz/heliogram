# Pinned CPU-repro path for heliogram. Reproduces the toolchain versions RESULTS.md's Provenance
# line was measured under (Python 3.12.3; numpy 2.1.2; Pillow 11.0.0; reedsolo 1.7.0) as closely
# as a base image allows, via requirements-cpu.lock.txt. This image is CPU-only by design -- see
# requirements-gpu.lock.txt / requirements-gpu.txt / RUNBOOK-GPU.md for the separate, untested
# (no GPU here) Phase-2 GPU stack, which is deliberately NOT installed here.
FROM python:3.12-slim

WORKDIR /app

# fonts-dejavu-core: a real monospace TrueType font for heliogram.typography's geometric packing
# check (heliogram/typography.py's _load_monospace_font refuses to fall back to a proportional
# font -- see that function's docstring -- so a real monospace font must be present).
# make: scripts/smoke.sh is invoked via `make smoke` (this image's default CMD).
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core make \
    && rm -rf /var/lib/apt/lists/*

# Install the pinned CPU lockfile first (better layer caching -- source changes below don't
# invalidate the dependency layer).
COPY requirements-cpu.lock.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.lock.txt

# Now bring in the source tree and install heliogram itself in editable mode.
COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# The model-free, version-robust end-to-end check (pytest suite + deterministic CLI invariants
# vs. tests/expected/*.txt fixtures) -- see Makefile / scripts/smoke.sh.
CMD ["make", "smoke"]
