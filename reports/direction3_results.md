# 3-class direction (up / flat / down) — technicals-only, walk-forward OOF

Flat band = +/-0.3% next-day return. 1993 days. Class balance: down 35%, flat 17%, up 48%.

| Class (one-vs-rest) | AUROC |
|---|---|
| down | 0.531 |
| flat | 0.554 |
| up | 0.494 |
| **macro** | **0.526** |


**Finding:** all classes are near chance (0.49-0.55), but the ordering is informative: **flat (0.554)** is the most separable (low recent volatility predicts a quiet day — volatility clustering), and among the directional classes **down (0.531) > up (0.494)** — the actionable DOWN tail carries marginally more signal than UP. A 3-class/SIDEWAYS reframing does not rescue direction prediction; consistent with weak-form EMH and the tails-are-predictable thesis (crash/down-risk is the feasible target).

