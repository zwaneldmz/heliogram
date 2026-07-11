# Frozen-encoder linear-probe report (Phase-2 Step 0)

Model (frozen vision tower): `Qwen/Qwen2.5-VL-3B-Instruct`. Probe: linear softmax readout over merged-token embeddings, oracle labels, train-statistics standardization only.

**Scope honesty:** a PASS here means the frozen embeddings linearly carry the symbols -- it de-risks, but does not replace, the fine-tune (the LM still has to learn to read them). A FAIL on clean images means the information was discarded by the tower and no fine-tune of its consumers can recover it.

| palette | corruption | probe symbol error | train error | RS budget | chance | verdict |
|---|---|---|---|---|---|---|
| 2 | clean | 0.1831 | 0.1493 | 0.0627 | 0.5000 | above RS budget but far below chance -- partial signal; not decodable end-to-end at this operating point via a linear readout |
| 4 | clean | 0.3329 | 0.2481 | 0.0627 | 0.7500 | above RS budget but far below chance -- partial signal; not decodable end-to-end at this operating point via a linear readout |
