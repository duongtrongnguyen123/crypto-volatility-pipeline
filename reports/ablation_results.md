# TRR meta-learner — ablations & rigor

## 1. Within-source fairness (time-split inside each news source)

| Era | n_test | crashes | LLM zero-shot | GBM technical | GBM full |
|---|---|---|---|---|---|
| 2016_2020 | 388 | 21 | 0.653 | 0.665 | 0.695 |
| 2021_2023 | 235 | 2 | 0.548 | 0.552 | 0.629 |

## 2. Calibration (walk-forward OOF)

Brier: raw=0.1394  isotonic=0.0725

| prob bin | n | predicted | observed |
|---|---|---|---|
| 0.0-0.2 | 899 | 0.032 | 0.060 |
| 0.2-0.4 | 103 | 0.291 | 0.058 |
| 0.4-0.6 | 47 | 0.480 | 0.085 |
| 0.6-0.8 | 40 | 0.685 | 0.050 |
| 0.8-1.0 | 21 | 0.894 | 0.143 |

## 3. Precision@K (highest-risk days)

base rate = 0.062
- P@10 = 0.20  (3.2x base)
- P@20 = 0.15  (2.4x base)
- P@30 = 0.13  (2.1x base)
