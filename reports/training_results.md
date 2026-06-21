# TRR meta-learner — does training help?

Dataset: 1555 days 2016-02-05..2023-12-15, 72 crashes (4.6%).

Out-of-time split: train 2016-2020 (n=969, 31 crashes, analyst news) -> test 2021-2023 (n=586, 41 crashes, FNSPID news).


## Out-of-time / cross-source test (2021-2023)

| Model | AUROC | PR-AUC |
|---|---|---|
| GBM_technical_only | 0.698 | 0.126 |
| GBM_full_ensemble | 0.670 | 0.136 |
| Logistic_stack | 0.654 | 0.109 |
| LLM_zeroshot | 0.557 | 0.081 |
| news_volume | 0.356 | 0.051 |

(test base rate 0.070)


## Walk-forward CV (pooled out-of-fold, 5 splits)

| Model | AUROC | PR-AUC |
|---|---|---|
| ensemble_walkforward | 0.604 | 0.102 |
| LLM_zeroshot_samefolds | 0.577 | 0.082 |

(n_eval=1036)


## Full-ensemble permutation importance (test AUROC drop)

- downside_5d: +0.0533
- pf_range: +0.0372
- log_news: +0.0266
- crash_prob: +0.0141
- vol_20d: +0.0102
- ret_1d: +0.0101
- log_edges: +0.0079
- dd_from_high_20d: +0.0034
- vol_10d: +0.0023
- ret_5d: +0.0003
- ret_10d: -0.0111
