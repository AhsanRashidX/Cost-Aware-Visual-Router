# Phase 1: content-aware router (generated from results/router_ce/summary.json)

Always-hybrid ANLS 0.566 at 1.009 GPU-s; always-visual 0.542 at 0.621. Router overhead charged: cross-encoder latency + tokenization; ColPali charged to text-routed queries.

Go criteria (fixed before running): mean test success AUC >= 0.87 AND hybrid-matching operating point (lambda chosen on val) at >= 30% less GPU than always-hybrid.

| System | AUC text / visual / hybrid (mean) | ANLS@0.45 | ANLS@0.55 | ANLS@0.65 | ANLS@0.75 | ANLS@0.85 | ANLS@0.95 | Match hybrid (val-chosen λ): test ANLS / GPU-s / saving / n | ΔANLS vs hybrid [95% CI], McNemar p (seed 1) |
|---|---|---|---|---|---|---|---|---|---|
| ce_full | 0.824 / 0.859 / 0.843 (0.842) | - | 0.505 | 0.555 | 0.561 | 0.565 | 0.565 | 0.562 / 0.698 / 31% / 4 | -0.012 [-0.023, +0.000], p=0.12 |
| ce_no_tok | 0.822 / 0.851 / 0.835 (0.836) | - | 0.494 | 0.552 | 0.561 | 0.563 | 0.563 | 0.563 / 0.667 / 34% / 3 | -0.011 [-0.022, +0.001], p=0.13 |
| gbm_tab_tok | 0.831 / 0.856 / 0.840 (0.842) | - | 0.517 | 0.547 | 0.557 | 0.559 | 0.559 | none / 0 | - |
| gbm_tab | 0.829 / 0.850 / 0.838 (0.839) | - | 0.513 | 0.542 | 0.556 | 0.559 | 0.559 | 0.564 / 0.768 / 24% / 1 | -0.014 [-0.023, -0.004], p=0.014 |

Cross-encoder latency (batch 1, GPU): 12.6 ms; tokenization 0.33 ms; inputs truncated at 512 tokens: 47%.
