# Frozen-encoder linear-probe report (Phase-2 Step 0)

Model (frozen vision tower): `Qwen/Qwen2.5-VL-7B-Instruct`. Probe: linear softmax readout over merged-token embeddings, oracle labels, train-statistics standardization only.

**Scope honesty:** a PASS here means the frozen embeddings linearly carry the symbols -- it de-risks, but does not replace, the fine-tune (the LM still has to learn to read them). A FAIL on clean images means the information was discarded by the tower and no fine-tune of its consumers can recover it.

| palette | corruption | probe symbol error | train error | RS budget | chance | verdict |
|---|---|---|---|---|---|---|
| 16 | clean | 0.6551 | 0.1110 | 0.0627 | 0.9375 | at/near chance -- embeddings do not linearly separate the symbols here (if this happens on CLEAN images, check the token-order assumption first, then treat the LM-token branch as unsupported for this tower) |
| 256 | clean | 0.8979 | 0.0021 | 0.0627 | 0.9961 | at/near chance -- embeddings do not linearly separate the symbols here (if this happens on CLEAN images, check the token-order assumption first, then treat the LM-token branch as unsupported for this tower) |
