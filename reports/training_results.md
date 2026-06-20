# TRR meta-learner — does training help?

Dataset: 1555 days 2016-02-05..2023-12-15, 72 crashes (4.6%).

Out-of-time split: train 2016-2020 (n=969, 31 crashes, analyst news) -> test 2021-2023 (n=586, 41 crashes, FNSPID news).


## Out-of-time / cross-source test (2021-2023)

| Model | AUROC | PR-AUC |
|---|---|---|
| GBM_technical_only | 0.682 | 0.127 |
| GBM_full_ensemble | 0.667 | 0.139 |
| Logistic_stack | 0.661 | 0.118 |
| LLM_zeroshot | 0.557 | 0.081 |
| news_volume | 0.356 | 0.051 |

(test base rate 0.070)


## Walk-forward CV (pooled out-of-fold, 5 splits)

| Model | AUROC | PR-AUC |
|---|---|---|
| ensemble_walkforward | 0.615 | 0.109 |
| LLM_zeroshot_samefolds | 0.577 | 0.082 |

(n_eval=1036)


## Full-ensemble permutation importance (test AUROC drop)

- downside_5d: +0.0817
- vol_20d: +0.0310
- ret_1d: +0.0197
- log_news: +0.0157
- crash_prob: +0.0128
- dd_from_high_20d: +0.0071
- log_edges: +0.0051
- ret_5d: -0.0025
- vol_10d: -0.0052
- ret_10d: -0.0121
