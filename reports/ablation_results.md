# TRR meta-learner — ablations & rigor

## 1. Within-source fairness (time-split inside each news source)

| Era | n_test | crashes | LLM zero-shot | GBM technical | GBM full |
|---|---|---|---|---|---|
| 2016_2020 | 388 | 21 | 0.653 | 0.665 | 0.695 |
| 2021_2023 | 235 | 2 | 0.548 | 0.552 | 0.588 |

## 2. Calibration (walk-forward OOF)

Brier: raw=0.1397  isotonic=0.0704

| prob bin | n | predicted | observed |
|---|---|---|---|
| 0.0-0.2 | 899 | 0.033 | 0.059 |
| 0.2-0.4 | 99 | 0.289 | 0.071 |
| 0.4-0.6 | 53 | 0.487 | 0.057 |
| 0.6-0.8 | 40 | 0.688 | 0.100 |
| 0.8-1.0 | 19 | 0.892 | 0.105 |

## 3. Precision@K (highest-risk days)

base rate = 0.062
- P@10 = 0.20  (3.2x base)
- P@20 = 0.10  (1.6x base)
- P@30 = 0.10  (1.6x base)
