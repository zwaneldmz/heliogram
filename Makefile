.PHONY: smoke

# The model-free, version-robust end-to-end repro check: the pytest suite plus a handful of
# deterministic CLI invariants (heliogram.benefit's RS budget/token counts,
# heliogram.instruments.foreign_tile's TPR/FPR, heliogram.typography's geometric verdict) diffed
# against tests/expected/*.txt fixtures. See scripts/smoke.sh for the full check list and the
# Dockerfile for the pinned-CPU-lockfile environment this is designed to pass in.
smoke:
	bash scripts/smoke.sh
