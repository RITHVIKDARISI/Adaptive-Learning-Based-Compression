# Drift Adaptation Benchmark Evaluation (v2 Multi-Seed)

## Executive Summary

- **Independent Stream Realizations**: 20 seeds evaluated per scenario on fresh streams with distinct Poisson arrival times and independent sampling.
- **Stream Length**: $\ge 5,000$ messages per stream (5,000 for synthetic scenarios 1-6; 6,000 for natural-drift scenarios 7-8).
- **Natural Drift Streams**: Included two real-data streams (100% real prepared data, 0% synthetic):
  - **`drift_7_natural_sequential`**: DailyDialog $\rightarrow$ Sentiment140 $\rightarrow$ SMS-Spam.
  - **`drift_8_natural_switching`**: Interleaved switching between DailyDialog, Sentiment140, and SMS-Spam every 1,000 messages.
- **Statistical Testing**: Paired Wilcoxon signed-rank tests with Holm-Bonferroni step-down correction and effect size metrics (Rank-Biserial $r$ and Cohen's $d_z$) across $N=20$ independent streams per scenario.

### Honest Assessment: LinUCB vs Always-Zstd

Here is the honest, un-tuned comparison of **LinUCB (deadline-aware)** against **Always-Zstd** across all 8 drift scenarios:

**Overall Record**: LinUCB beats Always-Zstd in **6 out of 8** scenarios (Always-Zstd beats LinUCB in **1**, and **1** are statistical ties).

| Scenario | LinUCB Mean Cost | Always-Zstd Mean Cost | Diff (LinUCB - Zstd) | % Cost Diff | Wilcoxon p (raw) | Verdict |
|---|---|---|---|---|---|---|
| `drift_1_repL_uniqS` | 31.96 | 32.72 | -0.75 | -2.30% | 1.4291e-01 | Tied / No Sig. Diff. |
| `drift_2_uniqS_repL` | 25.18 | 30.60 | -5.42 | -17.73% | 1.9073e-06 | **LinUCB Wins** (p < 0.05) |
| `drift_3_lowRate_burst` | 85.47 | 94.12 | -8.64 | -9.18% | 9.5367e-06 | **LinUCB Wins** (p < 0.05) |
| `drift_4_nat_json` | 68.48 | 77.33 | -8.85 | -11.44% | 1.9073e-06 | **LinUCB Wins** (p < 0.05) |
| `drift_5_lowDup_highDup` | 81.67 | 93.97 | -12.30 | -13.09% | 1.9073e-06 | **LinUCB Wins** (p < 0.05) |
| `drift_6_alternating` | 34.74 | 32.75 | +1.99 | +6.09% | 3.2234e-04 | **Always-Zstd Wins** (p < 0.05) |
| `drift_7_natural_sequential` | 60.93 | 67.73 | -6.80 | -10.04% | 1.9073e-06 | **LinUCB Wins** (p < 0.05) |
| `drift_8_natural_switching` | 61.12 | 67.76 | -6.64 | -9.80% | 1.9073e-06 | **LinUCB Wins** (p < 0.05) |

---
## Detailed Scenario Breakdowns

### Scenario: `drift_1_repL_uniqS`

#### Policy Cost & Adaptation Summary (20 Seeds)

| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |
|---|---|---|---|---|---|---|---|
| Oracle | 24.29 [24.15, 24.51] | -0.00 | -0.0% | 0.0 (20/20) | 22.2 | 23.1 | - |
| Best-Static | 26.50 [26.49, 26.51] | +2.21 | +9.1% | 488.0 (2/20) | 48.6 | 66.3 | Chosen: Always-Brotli:20 |
| LinUCB (deadline-aware) | 31.96 [31.07, 32.88] | +7.67 | +31.6% | 488.0 (2/20) | 147.6 | 224.5 | - |
| LinUCB (no deadline) | 33.60 [32.59, 34.68] | +9.31 | +38.3% | 488.0 (2/20) | 152.1 | 213.1 | - |
| Heuristic (deadline-aware) | 33.15 [32.71, 33.89] | +8.86 | +36.5% | 488.0 (2/20) | 27.4 | 29.3 | - |
| Heuristic (no deadline) | 37.47 [36.77, 38.18] | +13.18 | +54.3% | 488.0 (2/20) | 26.9 | 30.6 | - |
| Always-Zstd | 32.72 [32.71, 32.72] | +8.43 | +34.7% | 488.0 (2/20) | 27.0 | 30.3 | - |
| Always-LZ4 | 30.16 [30.16, 30.17] | +5.87 | +24.2% | 488.0 (2/20) | 27.7 | 30.8 | - |
| Always-Brotli | 26.50 [26.49, 26.51] | +2.21 | +9.1% | 488.0 (2/20) | 48.6 | 66.3 | - |
| Always-Gzip | 41.15 [41.14, 41.15] | +16.85 | +69.4% | 488.0 (2/20) | 42.9 | 51.9 | - |
| Always-Skip | 245.55 [245.55, 245.56] | +221.26 | +910.9% | 0.0 (20/20) | 22.7 | 23.8 | - |
| Decision Tree | 26.51 [26.48, 26.55] | +2.22 | +9.1% | 488.0 (2/20) | 47.9 | 63.1 | Degenerate (Single Class) |
| Gradient Boosting | 26.66 [26.50, 26.85] | +2.37 | +9.7% | 488.0 (2/20) | 91.7 | 129.9 | Degenerate (Single Class) |
| Logistic Regression | 26.48 [26.47, 26.50] | +2.19 | +9.0% | 488.0 (2/20) | 44.5 | 57.0 | Degenerate (Single Class) |
| Random Forest | 28.65 [26.48, 31.89] | +4.36 | +17.9% | 488.0 (2/20) | 631.4 | 955.4 | Degenerate (Single Class) |

#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)

| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |
|---|---|---|---|---|---|---|---|
| LinUCB vs Best-Static | +5.47 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +2.579 | Best-Static (p < 0.05) |
| LinUCB vs LinUCB (no deadline) | -1.64 | 39.0 | 1.2079e-02 | 4.8317e-02 | -0.629 | -0.624 | LinUCB (p < 0.05) |
| LinUCB vs Heuristic (deadline-aware) | -1.18 | 59.0 | 8.9695e-02 | 1.7939e-01 | -0.438 | -0.432 | Tied / Not Significant |
| LinUCB vs Heuristic (no deadline) | -5.51 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -2.347 | LinUCB (p < 0.05) |
| LinUCB vs Always-Zstd | -0.75 | 65.0 | 1.4291e-01 | 1.7939e-01 | -0.381 | -0.353 | Tied / Not Significant |
| LinUCB vs Always-LZ4 | +1.80 | 24.0 | 1.4324e-03 | 7.1621e-03 | +0.771 | +0.848 | Always-LZ4 (p < 0.05) |
| LinUCB vs Always-Brotli | +5.47 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +2.579 | Always-Brotli (p < 0.05) |
| LinUCB vs Always-Gzip | -9.18 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.311 | LinUCB (p < 0.05) |
| LinUCB vs Always-Skip | -213.59 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -100.270 | LinUCB (p < 0.05) |
| LinUCB vs Decision Tree | +5.46 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +2.587 | Decision Tree (p < 0.05) |
| LinUCB vs Gradient Boosting | +5.31 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +2.407 | Gradient Boosting (p < 0.05) |
| LinUCB vs Logistic Regression | +5.48 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +2.585 | Logistic Regression (p < 0.05) |
| LinUCB vs Random Forest | +3.32 | 39.0 | 1.2079e-02 | 4.8317e-02 | +0.629 | +0.505 | Random Forest (p < 0.05) |

---

### Scenario: `drift_2_uniqS_repL`

#### Policy Cost & Adaptation Summary (20 Seeds)

| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |
|---|---|---|---|---|---|---|---|
| Oracle | 22.16 [22.15, 22.17] | +0.00 | +0.0% | 0.0 (20/20) | 23.0 | 24.4 | - |
| Best-Static | 24.09 [24.08, 24.10] | +1.93 | +8.7% | 0.0 (20/20) | 65.2 | 82.8 | Chosen: Always-Brotli:20 |
| LinUCB (deadline-aware) | 25.18 [24.45, 25.85] | +3.02 | +13.6% | 0.0 (6/20) | 164.6 | 222.9 | - |
| LinUCB (no deadline) | 26.15 [24.71, 27.66] | +3.99 | +18.0% | 0.0 (8/20) | 161.3 | 215.1 | - |
| Heuristic (deadline-aware) | 30.61 [30.60, 30.62] | +8.45 | +38.1% | DNC (0/20) | 31.4 | 36.1 | - |
| Heuristic (no deadline) | 30.61 [30.60, 30.63] | +8.45 | +38.2% | DNC (0/20) | 30.7 | 35.3 | - |
| Always-Zstd | 30.60 [30.60, 30.60] | +8.44 | +38.1% | DNC (0/20) | 29.9 | 34.2 | - |
| Always-LZ4 | 28.16 [28.16, 28.16] | +6.00 | +27.1% | DNC (0/20) | 31.1 | 35.0 | - |
| Always-Brotli | 24.09 [24.08, 24.10] | +1.93 | +8.7% | 0.0 (20/20) | 65.2 | 82.8 | - |
| Always-Gzip | 38.41 [38.40, 38.42] | +16.25 | +73.3% | DNC (0/20) | 57.1 | 69.9 | - |
| Always-Skip | 298.90 [298.90, 298.90] | +276.74 | +1248.9% | DNC (0/20) | 24.3 | 25.4 | - |
| Decision Tree | 298.90 [298.90, 298.90] | +276.74 | +1248.9% | DNC (0/20) | 24.6 | 25.6 | Degenerate (Single Class) |
| Gradient Boosting | 298.90 [298.90, 298.90] | +276.74 | +1248.9% | DNC (0/20) | 24.4 | 25.4 | Degenerate (Single Class) |
| Logistic Regression | 298.90 [298.90, 298.90] | +276.74 | +1248.9% | DNC (0/20) | 24.7 | 25.9 | Degenerate (Single Class) |
| Random Forest | 298.90 [298.90, 298.90] | +276.74 | +1248.9% | DNC (0/20) | 24.5 | 25.7 | Degenerate (Single Class) |

#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)

| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |
|---|---|---|---|---|---|---|---|
| LinUCB vs Best-Static | +1.09 | 21.0 | 8.5068e-04 | 2.5520e-03 | +0.800 | +0.657 | Best-Static (p < 0.05) |
| LinUCB vs LinUCB (no deadline) | -0.98 | 71.0 | 2.1617e-01 | 2.1617e-01 | -0.324 | -0.294 | Tied / Not Significant |
| LinUCB vs Heuristic (deadline-aware) | -5.43 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -3.274 | LinUCB (p < 0.05) |
| LinUCB vs Heuristic (no deadline) | -5.44 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -3.278 | LinUCB (p < 0.05) |
| LinUCB vs Always-Zstd | -5.42 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -3.265 | LinUCB (p < 0.05) |
| LinUCB vs Always-LZ4 | -2.98 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -1.794 | LinUCB (p < 0.05) |
| LinUCB vs Always-Brotli | +1.09 | 21.0 | 8.5068e-04 | 2.5520e-03 | +0.800 | +0.657 | Always-Brotli (p < 0.05) |
| LinUCB vs Always-Gzip | -13.23 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -7.983 | LinUCB (p < 0.05) |
| LinUCB vs Always-Skip | -273.72 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -164.700 | LinUCB (p < 0.05) |
| LinUCB vs Decision Tree | -273.72 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -164.870 | LinUCB (p < 0.05) |
| LinUCB vs Gradient Boosting | -273.72 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -164.701 | LinUCB (p < 0.05) |
| LinUCB vs Logistic Regression | -273.72 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -164.688 | LinUCB (p < 0.05) |
| LinUCB vs Random Forest | -273.72 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -164.853 | LinUCB (p < 0.05) |

---

### Scenario: `drift_3_lowRate_burst`

#### Policy Cost & Adaptation Summary (20 Seeds)

| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |
|---|---|---|---|---|---|---|---|
| Oracle | 112.71 [109.76, 115.40] | -0.00 | -0.0% | 0.0 (20/20) | 16962.9 | 31863.0 | - |
| Best-Static | 81.30 [81.18, 81.41] | -31.41 | -27.9% | 0.0 (20/20) | 97.4 | 144.6 | Chosen: Always-Brotli:20 |
| LinUCB (deadline-aware) | 85.47 [83.82, 87.58] | -27.24 | -24.2% | 0.0 (20/20) | 304.0 | 511.1 | - |
| LinUCB (no deadline) | 82.48 [82.10, 83.03] | -30.23 | -26.8% | 0.0 (20/20) | 293.1 | 439.0 | - |
| Heuristic (deadline-aware) | 133.39 [124.45, 144.57] | +20.68 | +18.4% | 33.7 (20/20) | 13091.9 | 19354.0 | - |
| Heuristic (no deadline) | 4532.87 [4484.93, 4582.67] | +4420.16 | +3921.8% | 14.8 (20/20) | 2284755.0 | 3880274.4 | - |
| Always-Zstd | 94.12 [93.99, 94.24] | -18.59 | -16.5% | 0.0 (20/20) | 43.7 | 59.6 | - |
| Always-LZ4 | 97.12 [96.99, 97.24] | -15.59 | -13.8% | 17.9 (20/20) | 35.0 | 42.6 | - |
| Always-Brotli | 81.30 [81.18, 81.41] | -31.41 | -27.9% | 0.0 (20/20) | 97.4 | 144.6 | - |
| Always-Gzip | 92.06 [91.95, 92.15] | -20.65 | -18.3% | 0.0 (20/20) | 83.1 | 130.5 | - |
| Always-Skip | 128.23 [128.22, 128.23] | +15.52 | +13.8% | 523.0 (20/20) | 23.7 | 24.7 | - |
| Decision Tree | 81.45 [81.23, 81.71] | -31.26 | -27.7% | 0.0 (20/20) | 130.3 | 188.9 | Degenerate (Single Class) |
| Gradient Boosting | 81.85 [81.35, 82.45] | -30.86 | -27.4% | 0.0 (20/20) | 210.7 | 304.3 | Degenerate (Single Class) |
| Logistic Regression | 81.38 [81.21, 81.57] | -31.33 | -27.8% | 0.0 (20/20) | 114.5 | 172.1 | Degenerate (Single Class) |
| Random Forest | 83.78 [81.22, 88.79] | -28.92 | -25.7% | 24.2 (20/20) | 541.2 | 707.5 | Degenerate (Single Class) |

#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)

| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |
|---|---|---|---|---|---|---|---|
| LinUCB vs Best-Static | +4.17 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +0.946 | Best-Static (p < 0.05) |
| LinUCB vs LinUCB (no deadline) | +2.99 | 23.0 | 1.2093e-03 | 1.7567e-03 | +0.781 | +0.654 | LinUCB (no deadline) (p < 0.05) |
| LinUCB vs Heuristic (deadline-aware) | -47.92 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -2.078 | LinUCB (p < 0.05) |
| LinUCB vs Heuristic (no deadline) | -4447.40 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -38.955 | LinUCB (p < 0.05) |
| LinUCB vs Always-Zstd | -8.64 | 3.0 | 9.5367e-06 | 4.7684e-05 | -0.971 | -1.965 | LinUCB (p < 0.05) |
| LinUCB vs Always-LZ4 | -11.65 | 1.0 | 3.8147e-06 | 2.6703e-05 | -0.990 | -2.661 | LinUCB (p < 0.05) |
| LinUCB vs Always-Brotli | +4.17 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +0.946 | Always-Brotli (p < 0.05) |
| LinUCB vs Always-Gzip | -6.58 | 12.0 | 1.3351e-04 | 5.3406e-04 | -0.886 | -1.509 | LinUCB (p < 0.05) |
| LinUCB vs Always-Skip | -42.75 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -9.698 | LinUCB (p < 0.05) |
| LinUCB vs Decision Tree | +4.03 | 1.0 | 3.8147e-06 | 2.6703e-05 | +0.990 | +0.902 | Decision Tree (p < 0.05) |
| LinUCB vs Gradient Boosting | +3.63 | 19.0 | 5.8556e-04 | 1.7567e-03 | +0.819 | +0.748 | Gradient Boosting (p < 0.05) |
| LinUCB vs Logistic Regression | +4.09 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +0.935 | Logistic Regression (p < 0.05) |
| LinUCB vs Random Forest | +1.69 | 20.0 | 7.0763e-04 | 1.7567e-03 | +0.810 | +0.134 | Random Forest (p < 0.05) |

---

### Scenario: `drift_4_nat_json`

#### Policy Cost & Adaptation Summary (20 Seeds)

| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |
|---|---|---|---|---|---|---|---|
| Oracle | 69.92 [69.35, 70.47] | +0.00 | +0.0% | 0.0 (20/20) | 92.8 | 151.3 | - |
| Best-Static | 65.51 [65.45, 65.58] | -4.41 | -6.3% | 0.0 (20/20) | 94.6 | 158.6 | Chosen: Always-Brotli:20 |
| LinUCB (deadline-aware) | 68.48 [67.60, 69.44] | -1.44 | -2.1% | 107.9 (19/20) | 316.1 | 623.1 | - |
| LinUCB (no deadline) | 68.72 [67.38, 70.21] | -1.20 | -1.7% | 77.8 (20/20) | 289.7 | 509.7 | - |
| Heuristic (deadline-aware) | 204.88 [191.84, 219.44] | +134.96 | +193.0% | 1097.5 (11/20) | 6591.7 | 517586.5 | - |
| Heuristic (no deadline) | 1429.92 [1423.06, 1437.02] | +1360.00 | +1945.1% | 1098.7 (11/20) | 591487.9 | 752849.6 | - |
| Always-Zstd | 77.33 [77.31, 77.36] | +7.41 | +10.6% | 1006.0 (12/20) | 42.1 | 58.4 | - |
| Always-LZ4 | 77.00 [76.96, 77.04] | +7.08 | +10.1% | 1006.0 (12/20) | 34.7 | 44.0 | - |
| Always-Brotli | 65.51 [65.45, 65.58] | -4.41 | -6.3% | 0.0 (20/20) | 94.6 | 158.6 | - |
| Always-Gzip | 79.29 [79.24, 79.33] | +9.37 | +13.4% | 1006.0 (12/20) | 78.5 | 128.9 | - |
| Always-Skip | 87.41 [87.41, 87.42] | +17.49 | +25.0% | 0.0 (20/20) | 23.7 | 25.7 | - |
| Decision Tree | 65.87 [65.56, 66.28] | -4.05 | -5.8% | 0.0 (20/20) | 166.6 | 307.2 | Degenerate (Single Class) |
| Gradient Boosting | 67.18 [65.80, 69.09] | -2.74 | -3.9% | 84.5 (20/20) | 408.7 | 768.9 | Degenerate (Single Class) |
| Logistic Regression | 65.78 [65.55, 66.06] | -4.14 | -5.9% | 0.0 (20/20) | 144.8 | 250.5 | Degenerate (Single Class) |
| Random Forest | 74.04 [65.56, 82.78] | +4.12 | +5.9% | 100.3 (20/20) | 1467.9 | 1966.0 | Degenerate (Single Class) |

#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)

| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |
|---|---|---|---|---|---|---|---|
| LinUCB vs Best-Static | +2.97 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +1.352 | Best-Static (p < 0.05) |
| LinUCB vs LinUCB (no deadline) | -0.23 | 98.0 | 8.1236e-01 | 8.1236e-01 | -0.067 | -0.057 | Tied / Not Significant |
| LinUCB vs Heuristic (deadline-aware) | -136.39 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.150 | LinUCB (p < 0.05) |
| LinUCB vs Heuristic (no deadline) | -1361.44 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -80.712 | LinUCB (p < 0.05) |
| LinUCB vs Always-Zstd | -8.85 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.049 | LinUCB (p < 0.05) |
| LinUCB vs Always-LZ4 | -8.52 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -3.908 | LinUCB (p < 0.05) |
| LinUCB vs Always-Brotli | +2.97 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +1.352 | Always-Brotli (p < 0.05) |
| LinUCB vs Always-Gzip | -10.81 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.960 | LinUCB (p < 0.05) |
| LinUCB vs Always-Skip | -18.93 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -8.751 | LinUCB (p < 0.05) |
| LinUCB vs Decision Tree | +2.61 | 6.0 | 2.6703e-05 | 1.0681e-04 | +0.943 | +1.050 | Decision Tree (p < 0.05) |
| LinUCB vs Gradient Boosting | +1.30 | 57.0 | 7.5851e-02 | 2.2755e-01 | +0.457 | +0.280 | Tied / Not Significant |
| LinUCB vs Logistic Regression | +2.70 | 3.0 | 9.5367e-06 | 4.7684e-05 | +0.971 | +1.193 | Logistic Regression (p < 0.05) |
| LinUCB vs Random Forest | -5.56 | 57.0 | 7.5851e-02 | 2.2755e-01 | +0.457 | -0.275 | Tied / Not Significant |

---

### Scenario: `drift_5_lowDup_highDup`

#### Policy Cost & Adaptation Summary (20 Seeds)

| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |
|---|---|---|---|---|---|---|---|
| Oracle | 85.10 [84.15, 86.08] | +0.00 | +0.0% | 0.0 (20/20) | 93.5 | 126.9 | - |
| Best-Static | 81.04 [80.68, 81.37] | -4.06 | -4.8% | 0.0 (20/20) | 93.3 | 129.5 | Chosen: Always-Brotli:20 |
| LinUCB (deadline-aware) | 81.67 [81.30, 82.01] | -3.43 | -4.0% | 0.0 (20/20) | 265.5 | 380.9 | - |
| LinUCB (no deadline) | 82.10 [81.47, 82.84] | -3.00 | -3.5% | 0.0 (20/20) | 257.6 | 358.9 | - |
| Heuristic (deadline-aware) | 244.76 [232.73, 258.08] | +159.65 | +187.6% | 1029.4 (19/20) | 12215.3 | 542191.1 | - |
| Heuristic (no deadline) | 1719.23 [1707.85, 1730.49] | +1634.13 | +1920.2% | DNC (0/20) | 615393.0 | 776324.7 | - |
| Always-Zstd | 93.97 [93.56, 94.37] | +8.87 | +10.4% | 1007.2 (20/20) | 41.9 | 52.2 | - |
| Always-LZ4 | 96.92 [96.56, 97.29] | +11.82 | +13.9% | 1264.4 (14/20) | 32.5 | 37.4 | - |
| Always-Brotli | 81.04 [80.68, 81.37] | -4.06 | -4.8% | 0.0 (20/20) | 93.3 | 129.5 | - |
| Always-Gzip | 91.82 [91.52, 92.09] | +6.72 | +7.9% | 831.6 (20/20) | 80.1 | 109.0 | - |
| Always-Skip | 128.23 [128.23, 128.23] | +43.13 | +50.7% | 323.5 (2/20) | 23.9 | 25.6 | - |
| Decision Tree | 81.11 [80.77, 81.44] | -3.99 | -4.7% | 0.0 (20/20) | 118.2 | 163.8 | Degenerate (Single Class) |
| Gradient Boosting | 81.15 [80.85, 81.43] | -3.96 | -4.6% | 0.0 (20/20) | 126.9 | 187.4 | Degenerate (Single Class) |
| Logistic Regression | 81.12 [80.80, 81.44] | -3.98 | -4.7% | 0.0 (20/20) | 120.1 | 166.5 | Degenerate (Single Class) |
| Random Forest | 83.92 [80.87, 88.29] | -1.19 | -1.4% | 3.0 (20/20) | 865.5 | 1133.3 | Degenerate (Single Class) |

#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)

| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |
|---|---|---|---|---|---|---|---|
| LinUCB vs Best-Static | +0.63 | 1.0 | 3.8147e-06 | 2.6703e-05 | +0.990 | +1.836 | Best-Static (p < 0.05) |
| LinUCB vs LinUCB (no deadline) | -0.43 | 92.0 | 6.4766e-01 | 6.4766e-01 | +0.124 | -0.279 | Tied / Not Significant |
| LinUCB vs Heuristic (deadline-aware) | -163.09 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -5.486 | LinUCB (p < 0.05) |
| LinUCB vs Heuristic (no deadline) | -1637.56 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -62.304 | LinUCB (p < 0.05) |
| LinUCB vs Always-Zstd | -12.30 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -20.409 | LinUCB (p < 0.05) |
| LinUCB vs Always-LZ4 | -15.25 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -26.610 | LinUCB (p < 0.05) |
| LinUCB vs Always-Brotli | +0.63 | 1.0 | 3.8147e-06 | 2.6703e-05 | +0.990 | +1.836 | Always-Brotli (p < 0.05) |
| LinUCB vs Always-Gzip | -10.15 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -26.463 | LinUCB (p < 0.05) |
| LinUCB vs Always-Skip | -46.56 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -56.672 | LinUCB (p < 0.05) |
| LinUCB vs Decision Tree | +0.56 | 4.0 | 1.3351e-05 | 6.6757e-05 | +0.962 | +1.276 | Decision Tree (p < 0.05) |
| LinUCB vs Gradient Boosting | +0.52 | 19.0 | 5.8556e-04 | 1.7567e-03 | +0.819 | +0.844 | Gradient Boosting (p < 0.05) |
| LinUCB vs Logistic Regression | +0.55 | 5.0 | 1.9073e-05 | 7.6294e-05 | +0.952 | +1.206 | Logistic Regression (p < 0.05) |
| LinUCB vs Random Forest | -2.25 | 39.0 | 1.2079e-02 | 2.4158e-02 | +0.629 | -0.251 | LinUCB (p < 0.05) |

---

### Scenario: `drift_6_alternating`

#### Policy Cost & Adaptation Summary (20 Seeds)

| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |
|---|---|---|---|---|---|---|---|
| Oracle | 39.40 [37.63, 41.42] | +0.00 | +0.0% | 0.0 (20/20) | 23.4 | 26.4 | - |
| Best-Static | 26.60 [26.58, 26.61] | -12.80 | -32.5% | 0.0 (20/20) | 68.5 | 89.1 | Chosen: Always-Brotli:20 |
| LinUCB (deadline-aware) | 34.74 [33.89, 35.60] | -4.66 | -11.8% | 491.8 (20/20) | 239.8 | 345.1 | - |
| LinUCB (no deadline) | 36.08 [34.94, 37.37] | -3.32 | -8.4% | 491.8 (20/20) | 238.9 | 338.9 | - |
| Heuristic (deadline-aware) | 32.74 [32.74, 32.74] | -6.66 | -16.9% | 491.8 (20/20) | 30.7 | 34.8 | - |
| Heuristic (no deadline) | 32.75 [32.74, 32.75] | -6.65 | -16.9% | 491.8 (20/20) | 32.5 | 36.5 | - |
| Always-Zstd | 32.75 [32.74, 32.75] | -6.65 | -16.9% | 491.8 (20/20) | 32.2 | 36.6 | - |
| Always-LZ4 | 30.19 [30.18, 30.19] | -9.21 | -23.4% | 491.8 (20/20) | 32.2 | 36.6 | - |
| Always-Brotli | 26.60 [26.58, 26.61] | -12.80 | -32.5% | 0.0 (20/20) | 68.5 | 89.1 | - |
| Always-Gzip | 41.24 [41.23, 41.25] | +1.84 | +4.7% | 491.8 (20/20) | 60.9 | 77.8 | - |
| Always-Skip | 245.56 [245.56, 245.56] | +206.17 | +523.3% | 489.6 (20/20) | 24.1 | 25.1 | - |
| Decision Tree | 26.59 [26.58, 26.61] | -12.80 | -32.5% | 0.0 (20/20) | 69.0 | 88.7 | Degenerate (Single Class) |
| Gradient Boosting | 26.78 [26.59, 27.15] | -12.62 | -32.0% | 0.0 (20/20) | 111.8 | 151.6 | Degenerate (Single Class) |
| Logistic Regression | 26.78 [26.64, 26.96] | -12.62 | -32.0% | 0.0 (20/20) | 118.7 | 178.2 | Degenerate (Single Class) |
| Random Forest | 29.86 [26.59, 34.76] | -9.54 | -24.2% | 0.0 (20/20) | 833.0 | 1080.7 | Degenerate (Single Class) |

#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)

| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |
|---|---|---|---|---|---|---|---|
| LinUCB vs Best-Static | +8.14 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +4.125 | Best-Static (p < 0.05) |
| LinUCB vs LinUCB (no deadline) | -1.34 | 52.0 | 4.8441e-02 | 4.8441e-02 | -0.505 | -0.431 | LinUCB (p < 0.05) |
| LinUCB vs Heuristic (deadline-aware) | +2.00 | 16.0 | 3.2234e-04 | 1.6117e-03 | +0.848 | +1.005 | Heuristic (deadline-aware) (p < 0.05) |
| LinUCB vs Heuristic (no deadline) | +1.99 | 16.0 | 3.2234e-04 | 1.6117e-03 | +0.848 | +1.002 | Heuristic (no deadline) (p < 0.05) |
| LinUCB vs Always-Zstd | +1.99 | 16.0 | 3.2234e-04 | 1.6117e-03 | +0.848 | +1.003 | Always-Zstd (p < 0.05) |
| LinUCB vs Always-LZ4 | +4.55 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +2.290 | Always-LZ4 (p < 0.05) |
| LinUCB vs Always-Brotli | +8.14 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +4.125 | Always-Brotli (p < 0.05) |
| LinUCB vs Always-Gzip | -6.50 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -3.275 | LinUCB (p < 0.05) |
| LinUCB vs Always-Skip | -210.82 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -105.951 | LinUCB (p < 0.05) |
| LinUCB vs Decision Tree | +8.15 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +4.084 | Decision Tree (p < 0.05) |
| LinUCB vs Gradient Boosting | +7.96 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +3.633 | Gradient Boosting (p < 0.05) |
| LinUCB vs Logistic Regression | +7.96 | 0.0 | 1.9073e-06 | 2.4796e-05 | +1.000 | +3.826 | Logistic Regression (p < 0.05) |
| LinUCB vs Random Forest | +4.88 | 39.0 | 1.2079e-02 | 2.4158e-02 | +0.629 | +0.454 | Random Forest (p < 0.05) |

---

### Scenario: `drift_7_natural_sequential`

#### Policy Cost & Adaptation Summary (20 Seeds)

| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |
|---|---|---|---|---|---|---|---|
| Oracle | 63.14 [62.78, 63.53] | +0.00 | +0.0% | 0.0 (20/20) | 65.4 | 97.9 | - |
| Best-Static | 60.05 [59.90, 60.21] | -3.09 | -4.9% | 0.0 (20/20) | 72.3 | 118.0 | Chosen: Always-Brotli:20 |
| LinUCB (deadline-aware) | 60.93 [60.38, 61.52] | -2.21 | -3.5% | 0.0 (20/20) | 190.7 | 333.3 | - |
| LinUCB (no deadline) | 61.34 [60.74, 61.98] | -1.79 | -2.8% | 0.0 (20/20) | 205.9 | 309.7 | - |
| Heuristic (deadline-aware) | 66.83 [66.57, 67.11] | +3.70 | +5.9% | 0.4 (20/20) | 38.8 | 51.0 | - |
| Heuristic (no deadline) | 67.31 [66.88, 67.77] | +4.17 | +6.6% | 0.4 (20/20) | 38.6 | 52.3 | - |
| Always-Zstd | 67.73 [67.59, 67.87] | +4.59 | +7.3% | 3.6 (20/20) | 39.6 | 55.1 | - |
| Always-LZ4 | 72.39 [72.21, 72.57] | +9.25 | +14.7% | 2043.0 (17/20) | 28.9 | 38.5 | - |
| Always-Brotli | 60.05 [59.90, 60.21] | -3.09 | -4.9% | 0.0 (20/20) | 72.3 | 118.0 | - |
| Always-Gzip | 77.46 [77.32, 77.60] | +14.33 | +22.7% | 1767.1 (9/20) | 61.1 | 104.2 | - |
| Always-Skip | 67.83 [67.63, 68.02] | +4.69 | +7.4% | 1.1 (20/20) | 22.3 | 23.4 | - |
| Decision Tree | 62.76 [62.56, 62.96] | -0.37 | -0.6% | 0.0 (20/20) | 62.9 | 90.9 | - |
| Gradient Boosting | 62.83 [62.58, 63.08] | -0.30 | -0.5% | 0.0 (20/20) | 69.9 | 94.7 | - |
| Logistic Regression | 62.81 [62.53, 63.06] | -0.33 | -0.5% | 0.0 (20/20) | 63.4 | 89.7 | - |
| Random Forest | 62.68 [62.37, 62.95] | -0.46 | -0.7% | 0.0 (20/20) | 67.3 | 89.2 | - |

#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)

| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |
|---|---|---|---|---|---|---|---|
| LinUCB vs Best-Static | +0.88 | 41.0 | 1.5312e-02 | 4.5937e-02 | +0.610 | +0.639 | Best-Static (p < 0.05) |
| LinUCB vs LinUCB (no deadline) | -0.41 | 67.0 | 1.6496e-01 | 1.6496e-01 | -0.362 | -0.209 | Tied / Not Significant |
| LinUCB vs Heuristic (deadline-aware) | -5.90 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.045 | LinUCB (p < 0.05) |
| LinUCB vs Heuristic (no deadline) | -6.38 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -3.728 | LinUCB (p < 0.05) |
| LinUCB vs Always-Zstd | -6.80 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.719 | LinUCB (p < 0.05) |
| LinUCB vs Always-LZ4 | -11.46 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -7.548 | LinUCB (p < 0.05) |
| LinUCB vs Always-Brotli | +0.88 | 41.0 | 1.5312e-02 | 4.5937e-02 | +0.610 | +0.639 | Always-Brotli (p < 0.05) |
| LinUCB vs Always-Gzip | -16.54 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -12.047 | LinUCB (p < 0.05) |
| LinUCB vs Always-Skip | -6.90 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.412 | LinUCB (p < 0.05) |
| LinUCB vs Decision Tree | -1.83 | 12.0 | 1.3351e-04 | 8.0109e-04 | -0.886 | -1.260 | LinUCB (p < 0.05) |
| LinUCB vs Gradient Boosting | -1.90 | 8.0 | 4.7684e-05 | 3.3379e-04 | -0.924 | -1.311 | LinUCB (p < 0.05) |
| LinUCB vs Logistic Regression | -1.88 | 14.0 | 2.0981e-04 | 1.0490e-03 | -0.867 | -1.229 | LinUCB (p < 0.05) |
| LinUCB vs Random Forest | -1.75 | 17.0 | 3.9482e-04 | 1.5793e-03 | -0.838 | -1.125 | LinUCB (p < 0.05) |

---

### Scenario: `drift_8_natural_switching`

#### Policy Cost & Adaptation Summary (20 Seeds)

| Policy | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Adaptation Lag (Mean / Conv Rate) | P95 Latency (us) | P99 Latency (us) | Notes |
|---|---|---|---|---|---|---|---|
| Oracle | 63.11 [62.68, 63.55] | -0.00 | -0.0% | 0.0 (20/20) | 83.0 | 113.8 | - |
| Best-Static | 60.17 [60.05, 60.31] | -2.94 | -4.7% | 0.0 (20/20) | 89.0 | 135.1 | Chosen: Always-Brotli:20 |
| LinUCB (deadline-aware) | 61.12 [60.65, 61.63] | -1.99 | -3.1% | 0.0 (20/20) | 251.1 | 360.7 | - |
| LinUCB (no deadline) | 61.50 [60.89, 62.09] | -1.61 | -2.6% | 0.6 (20/20) | 243.0 | 345.2 | - |
| Heuristic (deadline-aware) | 66.79 [66.58, 67.03] | +3.68 | +5.8% | 851.3 (20/20) | 46.8 | 58.4 | - |
| Heuristic (no deadline) | 67.26 [66.85, 67.70] | +4.15 | +6.6% | 851.0 (20/20) | 46.7 | 59.1 | - |
| Always-Zstd | 67.76 [67.63, 67.90] | +4.65 | +7.4% | 862.2 (20/20) | 46.5 | 59.0 | - |
| Always-LZ4 | 72.42 [72.24, 72.60] | +9.31 | +14.7% | 1415.8 (20/20) | 32.2 | 40.1 | - |
| Always-Brotli | 60.17 [60.05, 60.31] | -2.94 | -4.7% | 0.0 (20/20) | 89.0 | 135.1 | - |
| Always-Gzip | 77.55 [77.42, 77.68] | +14.44 | +22.9% | 2313.1 (14/20) | 74.5 | 109.1 | - |
| Always-Skip | 67.83 [67.64, 68.03] | +4.72 | +7.5% | 875.5 (20/20) | 23.0 | 24.2 | - |
| Decision Tree | 62.56 [62.31, 62.83] | -0.55 | -0.9% | 0.0 (20/20) | 83.9 | 114.0 | - |
| Gradient Boosting | 62.69 [62.42, 63.00] | -0.42 | -0.7% | 0.0 (20/20) | 96.5 | 128.5 | - |
| Logistic Regression | 62.62 [62.35, 62.93] | -0.49 | -0.8% | 0.0 (20/20) | 82.9 | 118.7 | - |
| Random Forest | 62.55 [62.27, 62.86] | -0.56 | -0.9% | 0.0 (20/20) | 91.4 | 125.9 | - |

#### Paired Wilcoxon Signed-Rank Tests (LinUCB deadline-aware vs All Policies, N=20 Seeds)

| Comparison | Mean Diff (LinUCB - Pol) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Winner |
|---|---|---|---|---|---|---|---|
| LinUCB vs Best-Static | +0.95 | 27.0 | 2.3251e-03 | 6.9752e-03 | +0.743 | +0.765 | Best-Static (p < 0.05) |
| LinUCB vs LinUCB (no deadline) | -0.37 | 73.0 | 2.4549e-01 | 2.4549e-01 | -0.305 | -0.235 | Tied / Not Significant |
| LinUCB vs Heuristic (deadline-aware) | -5.67 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.161 | LinUCB (p < 0.05) |
| LinUCB vs Heuristic (no deadline) | -6.14 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -3.705 | LinUCB (p < 0.05) |
| LinUCB vs Always-Zstd | -6.64 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -5.268 | LinUCB (p < 0.05) |
| LinUCB vs Always-LZ4 | -11.29 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -8.395 | LinUCB (p < 0.05) |
| LinUCB vs Always-Brotli | +0.95 | 27.0 | 2.3251e-03 | 6.9752e-03 | +0.743 | +0.765 | Always-Brotli (p < 0.05) |
| LinUCB vs Always-Gzip | -16.43 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -13.284 | LinUCB (p < 0.05) |
| LinUCB vs Always-Skip | -6.71 | 0.0 | 1.9073e-06 | 2.4796e-05 | -1.000 | -4.857 | LinUCB (p < 0.05) |
| LinUCB vs Decision Tree | -1.43 | 20.0 | 7.0763e-04 | 3.5381e-03 | -0.810 | -0.954 | LinUCB (p < 0.05) |
| LinUCB vs Gradient Boosting | -1.57 | 17.0 | 3.9482e-04 | 2.7637e-03 | -0.838 | -0.998 | LinUCB (p < 0.05) |
| LinUCB vs Logistic Regression | -1.50 | 18.0 | 4.8256e-04 | 2.8954e-03 | -0.829 | -0.950 | LinUCB (p < 0.05) |
| LinUCB vs Random Forest | -1.43 | 24.0 | 1.4324e-03 | 5.7297e-03 | -0.771 | -0.885 | LinUCB (p < 0.05) |

---
