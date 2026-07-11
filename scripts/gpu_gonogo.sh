#!/usr/bin/env bash
# scripts/gpu_gonogo.sh -- one-shot Phase-0 + Design-A go/no-go for the merger-adapter question.
#
# *** GPU REQUIRED *** (loads a real Qwen2.5-VL tower). Runs, in order, writing all outputs under
# $OUTDIR (default ./gpu_gonogo_out):
#   [1/3] pre-merger  palette=16 LINEAR probe  -> the ALIGNMENT GATE (must reproduce ~0.1344 clean;
#                                                 this is what every downstream number is trusted
#                                                 against -- see probe_report_premerger.md).
#   [2/3] post-merger palette=16 LINEAR probe  -> confirms the negative the gate is measured
#                                                 against (expect at/near chance, ~0.66-0.74).
#   [3/3] Design A (frozen-feature NONLINEAR readout via scripts/drive_merger_adapter.py) -> the
#         cheap, decisive test. Design A itself re-checks the 0.1344 alignment and aborts on drift
#         or over-budget, so this script is a convenience sequencer, not a second source of truth.
#
# It deliberately does NOT run Design B: read Design A first (see the printed decision rule at the
# end). Nothing here fabricates a number -- every value comes from the scripts it calls, which
# refuse without a real model.
#
# Usage:   bash scripts/gpu_gonogo.sh [MODEL_ID]
#   MODEL_ID  default Qwen/Qwen2.5-VL-3B-Instruct (pass .../Qwen2.5-VL-7B-Instruct to scale up)
#   OUTDIR    env var, default ./gpu_gonogo_out
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTDIR="${OUTDIR:-gpu_gonogo_out}"
# DEVICE/DTYPE overrides: forward-only probe + Design A run fine on CPU, which is the escape
# hatch when the installed torch has no kernels for the GPU (e.g. torch 2.5.1 on an RTX 5090 /
# Blackwell sm_120). CPU is slower but gets the decisive go/no-go. Example:
#   DEVICE=cpu DTYPE=float32 bash scripts/gpu_gonogo.sh
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
cd "$(dirname "$0")/.."
mkdir -p "$OUTDIR"

echo "### model: $MODEL_ID   device: $DEVICE   dtype: $DTYPE"
echo "### outputs -> $OUTDIR/"

echo
echo "### [1/3] pre-merger palette=16 LINEAR probe  (ALIGNMENT GATE -- expect clean ~0.1344)"
python scripts/run_probe.py --model-id "$MODEL_ID" --device "$DEVICE" --dtype "$DTYPE" \
    --probe-stage pre_merger --palettes 16 --corruptions clean,jpeg_q70 \
    --out "$OUTDIR/probe_pre_p16.md" --json "$OUTDIR/probe_pre_p16.json"

echo
echo "### [2/3] post-merger palette=16 LINEAR probe  (expect at/near chance ~0.66-0.74)"
python scripts/run_probe.py --model-id "$MODEL_ID" --device "$DEVICE" --dtype "$DTYPE" \
    --probe-stage merged --palettes 16 --corruptions clean \
    --out "$OUTDIR/probe_post_p16.md" --json "$OUTDIR/probe_post_p16.json"

echo
echo "### [3/3] Design A: frozen-feature nonlinear readout  (the cheap go/no-go)"
python scripts/drive_merger_adapter.py --design a --device "$DEVICE" --dtype "$DTYPE" \
    --model-id "$MODEL_ID" --palette 16 --corruptions clean,jpeg_q70 \
    --out "$OUTDIR/design_a.json"

echo
echo "############################################################"
echo "### DONE. Decision rule -- read $OUTDIR/design_a.json:"
echo "###   * Design A at/near chance      -> the per-patch info is already gone at the merger"
echo "###                                     INPUT -> a trainable merger cannot recover it ->"
echo "###                                     STRONG NEGATIVE, stop here."
echo "###   * Design A recovers the 4 syms  -> info is present at the merger input; run Design B"
echo "###     (toward/below the ~6.27% RS     (the actual trainable-merger gate):"
echo "###      budget)"
echo "###"
echo "###     python scripts/drive_merger_adapter.py --design b --lora-variant B1 \\"
echo "###       --model-id $MODEL_ID --palette 16 --corruptions clean,jpeg_q70 \\"
echo "###       --epochs 20 --lora-rank 8 --out $OUTDIR/design_b.json"
echo "###"
echo "###   Reminder: a trained readout head is NOT the LM using the symbols -- Design B crossing"
echo "###   the budget REOPENS the fine-tune question, it does not by itself prove the economics."
echo "############################################################"
