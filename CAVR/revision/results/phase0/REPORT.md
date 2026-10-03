# Phase 0 probes (from cache; lambda/tau picked on val, reported on test, mean over seeds)

Reference: always-hybrid ANLS 0.566 at 1.009 GPU-s; always-visual ANLS 0.542 at 0.621 GPU-s.
'Match hybrid' = cheapest setting whose val ANLS >= val(hybrid) - 0.005. n = seeds where such a setting exists.

## Probe 3 - router features (option B)

| Features | AUC text ok | AUC visual ok | AUC hybrid ok | Match hybrid: ANLS / GPU-s (saving) | n | Match visual: ANLS / GPU-s | n |
|---|---|---|---|---|---|---|---|
| Q (question only) | 0.724 | 0.756 | 0.769 | nan / nan (nan%) | 0 | 0.540 / 0.729 | 5 |
| Q+BM25-content | 0.791 | 0.783 | 0.790 | 0.550 / 0.800 (21%) | 1 | 0.537 / 0.687 | 5 |
| Q+retrieval+content | 0.829 | 0.850 | 0.838 | 0.564 / 0.768 (24%) | 1 | 0.542 / 0.644 | 5 |

## Probe 1 - cascades with the current shared 7B generator (option C)

| Cascade [signal] | Match hybrid: ANLS / GPU-s (saving) | n | Match visual: ANLS / GPU-s | n |
|---|---|---|---|---|
| text->visual [answer-only] | nan / nan (nan%) | 0 | 0.535 / 0.959 | 5 |
| text->visual [answer+retrieval+content] | nan / nan (nan%) | 0 | 0.534 / 1.001 | 5 |
| text->hybrid [answer-only] | 0.562 / 1.396 (-38%) | 5 | 0.544 / 1.203 | 5 |
| text->hybrid [answer+retrieval+content] | 0.561 / 1.385 (-37%) | 5 | 0.535 / 1.180 | 5 |
| visual->hybrid [answer-only] | 0.560 / 0.961 (5%) | 5 | 0.544 / 0.634 | 5 |
| visual->hybrid [answer+retrieval+content] | 0.563 / 0.953 (6%) | 5 | 0.547 / 0.679 | 5 |

## Probe 2 - hypothetical cheap text path (option A): text GPU cost x s, keeps fraction r of its correct answers

Router = Q+retrieval+content features. HYPOTHETICAL: no smaller model was run.

| s (cost factor) | r (answers kept) | Always-text ANLS / GPU-s | Router match hybrid: ANLS / GPU-s (saving) | n | Oracle ANLS / GPU-s |
|---|---|---|---|---|---|
| 1.0 | 1.0 | 0.364 / 0.434 | 0.564 / 0.768 (24%) | 1 | 0.630 / 0.486 |
| 1.0 | 0.9 | 0.327 / 0.434 | 0.565 / 0.810 (20%) | 1 | 0.628 / 0.494 |
| 1.0 | 0.8 | 0.293 / 0.434 | 0.555 / 0.787 (22%) | 2 | 0.626 / 0.500 |
| 1.0 | 0.7 | 0.252 / 0.434 | 0.558 / 0.801 (21%) | 2 | 0.621 / 0.509 |
| 0.5 | 1.0 | 0.364 / 0.217 | 0.566 / 0.753 (25%) | 1 | 0.630 / 0.344 |
| 0.5 | 0.9 | 0.326 / 0.217 | nan / nan (nan%) | 0 | 0.628 / 0.359 |
| 0.5 | 0.8 | 0.294 / 0.217 | 0.556 / 0.787 (22%) | 2 | 0.624 / 0.370 |
| 0.5 | 0.7 | 0.255 / 0.217 | 0.556 / 0.768 (24%) | 2 | 0.621 / 0.386 |
| 0.33 | 1.0 | 0.364 / 0.143 | 0.567 / 0.758 (25%) | 1 | 0.630 / 0.293 |
| 0.33 | 0.9 | 0.327 / 0.143 | 0.567 / 0.787 (22%) | 1 | 0.627 / 0.310 |
| 0.33 | 0.8 | 0.296 / 0.143 | 0.567 / 0.791 (22%) | 1 | 0.625 / 0.324 |
| 0.33 | 0.7 | 0.253 / 0.143 | 0.568 / 0.804 (20%) | 1 | 0.621 / 0.343 |
| 0.2 | 1.0 | 0.364 / 0.087 | 0.567 / 0.751 (26%) | 1 | 0.630 / 0.254 |
| 0.2 | 0.9 | 0.325 / 0.087 | 0.568 / 0.773 (23%) | 1 | 0.627 / 0.274 |
| 0.2 | 0.8 | 0.289 / 0.087 | 0.570 / 0.792 (21%) | 1 | 0.623 / 0.291 |
| 0.2 | 0.7 | 0.256 / 0.087 | nan / nan (nan%) | 0 | 0.621 / 0.308 |

OCR window index for content features: 17.7 s of GPU time, once, at index time.
