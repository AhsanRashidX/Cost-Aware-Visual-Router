# Phase 0 verdict (generated from results/phase0/*.json)

> **Note (added later):** in this exploratory report, the "match hybrid" operating points were chosen as the cheapest setting by *test* GPU cost among those meeting the validation target. The selection key has since been changed to validation cost in the code. These numbers are exploratory and are not used in the paper; the paper's operating point comes from evaluate.py, which selects on validation data only.


Go/no-go criteria were fixed before running. Curves = test ANLS at a GPU-s budget, mean over 5 seeds.

## Probe 2 - cheaper text path (hypothetical s = cost factor, r = share of correct answers kept)

| Scenario | 0.20 | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|---|---|---|
| s=1.0,r=1.0 router | - | - | - | 0.433 | 0.532 | 0.549 | 0.559 | 0.559 |
| s=1.0,r=1.0 oracle | - | - | - | 0.630 | 0.630 | 0.630 | 0.630 | 0.630 |
| s=0.33,r=0.9 router | 0.340 | 0.435 | 0.489 | 0.519 | 0.541 | 0.553 | 0.558 | 0.558 |
| s=0.33,r=0.9 oracle | 0.472 | 0.625 | 0.628 | 0.628 | 0.628 | 0.628 | 0.628 | 0.628 |
| s=0.2,r=0.9 router | 0.395 | 0.457 | 0.493 | 0.522 | 0.542 | 0.554 | 0.558 | 0.558 |
| s=0.2,r=0.9 oracle | 0.560 | 0.628 | 0.628 | 0.628 | 0.628 | 0.628 | 0.628 | 0.628 |
| s=0.2,r=0.8 router | 0.361 | 0.432 | 0.481 | 0.515 | 0.537 | 0.552 | 0.558 | 0.558 |
| s=0.2,r=0.8 oracle | 0.527 | 0.625 | 0.625 | 0.625 | 0.625 | 0.625 | 0.625 | 0.625 |

## Probe 4 - gold page in every path's context (44.1% of queries)

| Policy | ANLS | GPU-s |
|---|---|---|
| always text | 0.623 | 0.431 |
| always visual | 0.858 | 0.597 |
| always hybrid | 0.850 | 1.008 |
| oracle | 0.915 | 0.490 |
| router at budgets | 0.55: 0.807, 0.65: 0.844, 0.75: 0.853, 0.85: 0.855, 0.95: 0.855 | |

## Probe 5 - router predictions blended toward the truth (required prediction quality)

| alpha | AUC text / visual / hybrid | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|
| 0.0 | 0.829 / 0.850 / 0.838 | - | 0.513 | 0.542 | 0.556 | 0.559 | 0.559 |
| 0.1 | 0.889 / 0.903 / 0.895 | - | 0.559 | 0.586 | 0.593 | 0.593 | 0.593 |
| 0.2 | 0.937 / 0.944 / 0.941 | - | 0.594 | 0.613 | 0.614 | 0.614 | 0.614 |
| 0.3 | 0.973 / 0.974 / 0.974 | - | 0.614 | 0.624 | 0.625 | 0.625 | 0.625 |
| 0.4 | 0.992 / 0.992 / 0.992 | - | 0.623 | 0.627 | 0.628 | 0.628 | 0.628 |
| 0.5 | 0.999 / 0.999 / 0.999 | - | 0.628 | 0.630 | 0.630 | 0.630 | 0.630 |
| 0.7 | 1.000 / 1.000 / 1.000 | - | 0.630 | 0.630 | 0.630 | 0.630 | 0.630 |
