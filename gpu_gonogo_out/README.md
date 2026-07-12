# gpu_gonogo_out — Session-3 GPU artifacts (2026-07-12)

Raw outputs of the merger-adapter go/no-go (`RUNBOOK-GPU.md` section 7), run on a rented
RunPod box (**RTX 4090, CUDA, bfloat16**, Qwen/Qwen2.5-VL-3B-Instruct, fresh clone of this
branch). The pod had no push access, so these files were transcribed **verbatim** from the
session's terminal output (`drive_merger_adapter.py` prints the exact JSON it writes;
`run_probe.py` prints the exact markdown report it writes). The two probe `.json` files
were not captured in the transcript and are omitted — the `.md` reports carry the same
numbers.

Every run below independently re-measured the pre-merger palette=16/clean alignment gate
and reproduced the committed number **exactly** (0.13442 vs target 0.1344, tolerance
±0.02) — the first reproduction of `probe_report_premerger.md`'s headline number on a real
GPU (the original was CPU/float32; this run was CUDA/bfloat16).

| file | what | key numbers (test symbol error) |
|---|---|---|
| `probe_pre_p16.md` | pre-merger linear probe, gate + jpeg | clean **0.1344**, jpeg_q70 0.1902 |
| `probe_post_p16.md` | post-merger linear probe | clean **0.7358** (chance 0.9375) |
| `design_a.json` | Design A, defaults (6 train / 3 test imgs) | clean 0.4916 / jpeg 0.4871, train ~0 (overfit) |
| `design_a_bigdata.json` | Design A, 24 train / 8 test imgs | clean **0.2464** / jpeg 0.2680, train ~0 (still data-limited) |
| `design_b.json` | Design B **B1** (LoRA r8, 20 ep, 24/8 imgs) | clean 0.5547 / jpeg 0.8849 (under-optimized) |
| `design_b_strong.json` | Design B **B1** (LoRA r32/α64, 80 ep, 48/12 imgs) | clean **0.3919** / jpeg **0.4100** |
| `design_b2_strong.json` | Design B **B2** (parallel adapter, 80 ep, 48/12 imgs) | clean **0.3752** / jpeg 0.9075 (jpeg diverged) |

RS budget (nsym=32): **0.0627**. Stock frozen-merger post-merger baseline: 0.6551–0.7358.

**Verdict: NO-GO** — see `RUNBOOK-GPU.md` "Session-3 verdict" for the full decision-rule
reading (including the honest note that Design B *did* clear the weaker
"clearly below the stock baseline" clause while decisively failing the budget and the
frozen pre-merger linear bar).
