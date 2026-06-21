# RAG vs baseline — paired bootstrap significance (2000 resamples)

| Window | days | crashes | ΔAUROC | 95% CI | p(Δ≤0) | sig? |
|---|---|---|---|---|---|---|
| stock_COVID_2019_20 | 343 | 14 | +0.063 | [-0.012, +0.128] | 0.046 | no |
| stock_broad_2018_20 | 761 | 31 | +0.074 | [+0.014, +0.136] | 0.009 | **yes** |
| FNSPID_bear_2021_23 | 836 | 41 | +0.065 | [+0.016, +0.115] | 0.004 | **yes** |

2/3 windows show a statistically significant RAG gain (95% CI excludes 0). Few crash events widen the CIs — read accordingly.

