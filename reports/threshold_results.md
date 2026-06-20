# Operating-threshold selection (walk-forward OOF)

1110 days, 69 crashes (6.2% base rate). At this base rate with a moderate-AUROC detector, absolute precision is structurally limited; the decision-relevant view is the **alert rate** (how many days you flag) vs recall/precision lift.

## Operating points by alert rate (top-k% riskiest days)

### Alert on riskiest 5% of days (threshold = 0.612)
precision=0.09  recall=0.07  F1=0.08  F2=0.08  lift=1.4x

| | pred crash | pred calm |
|---|---|---|
| **actual crash** | 5 (TP) | 64 (FN) |
| **actual calm**  | 51 (FP) | 990 (TN) |

### Alert on riskiest 10% of days (threshold = 0.391)
precision=0.08  recall=0.13  F1=0.10  F2=0.12  lift=1.3x

| | pred crash | pred calm |
|---|---|---|
| **actual crash** | 9 (TP) | 60 (FN) |
| **actual calm**  | 102 (FP) | 939 (TN) |

### Alert on riskiest 15% of days (threshold = 0.269)
precision=0.07  recall=0.17  F1=0.10  F2=0.14  lift=1.2x

| | pred crash | pred calm |
|---|---|---|
| **actual crash** | 12 (TP) | 57 (FN) |
| **actual calm**  | 155 (FP) | 886 (TN) |

### Alert on riskiest 20% of days (threshold = 0.184)
precision=0.07  recall=0.22  F1=0.10  F2=0.15  lift=1.1x

| | pred crash | pred calm |
|---|---|---|
| **actual crash** | 15 (TP) | 54 (FN) |
| **actual calm**  | 207 (FP) | 834 (TN) |

### Max-F1 operating point (threshold = 0.018)
precision=0.09  recall=0.68  F1=0.15  F2=0.29  lift=1.4x

| | pred crash | pred calm |
|---|---|---|
| **actual crash** | 47 (TP) | 22 (FN) |
| **actual calm**  | 501 (FP) | 540 (TN) |

