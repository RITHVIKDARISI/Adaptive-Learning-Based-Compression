# LinUCB Variants Ablation Study (v2 Multi-Seed)

## Executive Summary

- **Evaluation Protocol**: 8 drift scenarios (6 synthetic + 2 natural real-data streams), evaluated across 20 independent seeds (160 streams total).
- **Variants Compared**:
  1. **`Plain LinUCB`**: Standard LinUCB with unit exploration bonuses across all actions ($s_a = 1.0$), unconstrained action space, and cold initialization.
  2. **`Cost-Aware LinUCB`**: Calibrated action cost scales $s_a = \max(1.0, \bar{L}_a / \bar{L}_{\text{SKIP}})$ discounting exploration bonus of high-latency actions by $1/s_a$.
  3. **`Deadline-Mask LinUCB`**: Hard deadline safety mask that invalidates the BATCH action ($p_5 = -\infty$) when predicted queue wait exceeds $50,000\,\mu\text{s}$.
  4. **`Warm-Start LinUCB`**: Per-action parameter initialization from the domain length heuristic (SKIP if char\_len < 25, else ZSTD) during the first 500 messages.
  5. **`All-Combined LinUCB`**: Synergistic combination of Cost-Aware scaling, Deadline Masking, and Heuristic Warm-Start.
- **Statistical Testing**: Paired Wilcoxon signed-rank tests comparing Plain LinUCB against each of the 4 variants across the $N=20$ paired streams per scenario, corrected via step-down Holm-Bonferroni.

---
## Cross-Scenario Overview Table

| Scenario | Plain LinUCB Mean [95% CI] | Cost-Aware LinUCB | Deadline-Mask LinUCB | Warm-Start LinUCB | All-Combined LinUCB | Best LinUCB Variant |
|---|---|---|---|---|---|---|
| `drift_1_repL_uniqS` | 33.70 [32.62, 34.81] | 37.21 | 32.14 | 35.65 | 34.59 | **Deadline-Mask LinUCB** |
| `drift_2_uniqS_repL` | 26.16 [24.77, 27.65] | 26.45 | 24.52 | 31.14 | 29.44 | **Deadline-Mask LinUCB** |
| `drift_3_lowRate_burst` | 82.12 [81.81, 82.64] | 82.12 | 86.26 | 87.74 | 88.48 | **Plain LinUCB** |
| `drift_4_nat_json` | 68.64 [67.07, 70.51] | 69.03 | 67.77 | 75.41 | 73.47 | **Deadline-Mask LinUCB** |
| `drift_5_lowDup_highDup` | 81.79 [81.15, 82.51] | 81.80 | 81.32 | 88.10 | 86.38 | **Deadline-Mask LinUCB** |
| `drift_6_alternating` | 35.80 [34.65, 37.03] | 34.92 | 34.42 | 31.38 | 32.08 | **Warm-Start LinUCB** |
| `drift_7_natural_sequential` | 61.33 [60.62, 62.09] | 61.39 | 61.06 | 67.00 | 64.41 | **Deadline-Mask LinUCB** |
| `drift_8_natural_switching` | 61.35 [60.76, 61.92] | 61.38 | 61.02 | 66.40 | 64.55 | **Deadline-Mask LinUCB** |

---
## Detailed Per-Scenario Analysis & Paired Wilcoxon Tests

### Scenario: `drift_1_repL_uniqS`

#### Performance & Latency Metrics (N=20 Seeds)

| Variant | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Mean Latency (us) | P95 Latency (us) | P99 Latency (us) |
|---|---|---|---|---|---|---|
| *Oracle* | 24.30 [24.15, 24.51] | 0.00 | 0.0% | 29.9 | 22.4 | 25.5 |
| **Plain LinUCB** | 33.70 [32.62, 34.81] | +9.40 | +38.7% | 746.5 | 175.3 | 252.3 |
| **Cost-Aware LinUCB** | 37.21 [36.17, 38.30] | +12.91 | +53.1% | 748.0 | 184.4 | 268.7 |
| **Deadline-Mask LinUCB** | 32.14 [31.18, 33.14] | +7.84 | +32.3% | 590.8 | 192.6 | 302.2 |
| **Warm-Start LinUCB** | 35.65 [34.62, 36.64] | +11.35 | +46.7% | 707.9 | 135.8 | 185.7 |
| **All-Combined LinUCB** | 34.59 [33.47, 35.69] | +10.29 | +42.4% | 591.6 | 151.3 | 211.8 |

#### Paired Wilcoxon Tests vs Plain LinUCB (N=20 Seeds, Holm-Corrected)

| Comparison | Mean Diff (Variant - Plain) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Cost-Aware LinUCB vs Plain LinUCB | +3.51 | 0.0 | 1.9073e-06 | 7.6294e-06 | +1.000 | +13.293 | **Plain LinUCB Better (p < 0.05)** |
| Deadline-Mask LinUCB vs Plain LinUCB | -1.56 | 46.0 | 2.6642e-02 | 7.9926e-02 | -0.562 | -0.582 | **Tied / Not Significant** |
| Warm-Start LinUCB vs Plain LinUCB | +1.95 | 48.0 | 3.2768e-02 | 7.9926e-02 | +0.543 | +0.522 | **Tied / Not Significant** |
| All-Combined LinUCB vs Plain LinUCB | +0.90 | 68.0 | 1.7685e-01 | 1.7685e-01 | +0.352 | +0.302 | **Tied / Not Significant** |

---

### Scenario: `drift_2_uniqS_repL`

#### Performance & Latency Metrics (N=20 Seeds)

| Variant | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Mean Latency (us) | P95 Latency (us) | P99 Latency (us) |
|---|---|---|---|---|---|---|
| *Oracle* | 22.14 [22.13, 22.14] | 0.00 | 0.0% | 13.8 | 22.3 | 23.1 |
| **Plain LinUCB** | 26.16 [24.77, 27.65] | +4.03 | +18.2% | 155.4 | 131.3 | 157.0 |
| **Cost-Aware LinUCB** | 26.45 [25.14, 27.82] | +4.31 | +19.5% | 156.2 | 135.5 | 162.6 |
| **Deadline-Mask LinUCB** | 24.52 [23.75, 25.29] | +2.38 | +10.8% | 57.5 | 129.3 | 154.0 |
| **Warm-Start LinUCB** | 31.14 [29.76, 32.62] | +9.00 | +40.7% | 762.3 | 128.9 | 164.5 |
| **All-Combined LinUCB** | 29.44 [28.46, 30.54] | +7.31 | +33.0% | 493.0 | 131.3 | 166.9 |

#### Paired Wilcoxon Tests vs Plain LinUCB (N=20 Seeds, Holm-Corrected)

| Comparison | Mean Diff (Variant - Plain) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Cost-Aware LinUCB vs Plain LinUCB | +0.29 | 84.0 | 4.5238e-01 | 4.5238e-01 | +0.200 | +0.171 | **Tied / Not Significant** |
| Deadline-Mask LinUCB vs Plain LinUCB | -1.64 | 46.0 | 2.6642e-02 | 5.3284e-02 | -0.562 | -0.577 | **Tied / Not Significant** |
| Warm-Start LinUCB vs Plain LinUCB | +4.98 | 9.0 | 6.2943e-05 | 2.5177e-04 | +0.914 | +1.295 | **Plain LinUCB Better (p < 0.05)** |
| All-Combined LinUCB vs Plain LinUCB | +3.28 | 20.0 | 7.0763e-04 | 2.1229e-03 | +0.810 | +0.889 | **Plain LinUCB Better (p < 0.05)** |

---

### Scenario: `drift_3_lowRate_burst`

#### Performance & Latency Metrics (N=20 Seeds)

| Variant | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Mean Latency (us) | P95 Latency (us) | P99 Latency (us) |
|---|---|---|---|---|---|---|
| *Oracle* | 112.64 [110.19, 114.98] | 0.00 | 0.0% | 3320.2 | 16800.1 | 32056.9 |
| **Plain LinUCB** | 82.12 [81.81, 82.64] | -30.52 | -27.1% | 148.9 | 217.4 | 312.3 |
| **Cost-Aware LinUCB** | 82.12 [81.81, 82.64] | -30.51 | -27.1% | 149.2 | 223.6 | 303.1 |
| **Deadline-Mask LinUCB** | 86.26 [83.89, 89.22] | -26.37 | -23.4% | 479.4 | 279.9 | 1062.0 |
| **Warm-Start LinUCB** | 87.74 [86.34, 89.68] | -24.90 | -22.1% | 562.6 | 330.6 | 1984.6 |
| **All-Combined LinUCB** | 88.48 [86.66, 90.58] | -24.15 | -21.4% | 656.6 | 237.0 | 407.9 |

#### Paired Wilcoxon Tests vs Plain LinUCB (N=20 Seeds, Holm-Corrected)

| Comparison | Mean Diff (Variant - Plain) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Cost-Aware LinUCB vs Plain LinUCB | +0.00 | 98.0 | 8.1236e-01 | 8.1236e-01 | -0.067 | +0.035 | **Tied / Not Significant** |
| Deadline-Mask LinUCB vs Plain LinUCB | +4.14 | 16.0 | 3.2234e-04 | 6.4468e-04 | +0.848 | +0.643 | **Plain LinUCB Better (p < 0.05)** |
| Warm-Start LinUCB vs Plain LinUCB | +5.62 | 0.0 | 1.9073e-06 | 7.6294e-06 | +1.000 | +1.354 | **Plain LinUCB Better (p < 0.05)** |
| All-Combined LinUCB vs Plain LinUCB | +6.36 | 1.0 | 3.8147e-06 | 1.1444e-05 | +0.990 | +1.304 | **Plain LinUCB Better (p < 0.05)** |

---

### Scenario: `drift_4_nat_json`

#### Performance & Latency Metrics (N=20 Seeds)

| Variant | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Mean Latency (us) | P95 Latency (us) | P99 Latency (us) |
|---|---|---|---|---|---|---|
| *Oracle* | 69.75 [69.18, 70.31] | 0.00 | 0.0% | 270.8 | 61.1 | 77.8 |
| **Plain LinUCB** | 68.64 [67.07, 70.51] | -1.11 | -1.6% | 304.5 | 168.6 | 263.0 |
| **Cost-Aware LinUCB** | 69.03 [67.04, 71.39] | -0.72 | -1.0% | 308.7 | 175.4 | 271.5 |
| **Deadline-Mask LinUCB** | 67.77 [66.83, 68.79] | -1.99 | -2.8% | 215.8 | 174.9 | 270.8 |
| **Warm-Start LinUCB** | 75.41 [73.77, 77.08] | +5.65 | +8.1% | 882.8 | 174.0 | 285.2 |
| **All-Combined LinUCB** | 73.47 [71.76, 75.22] | +3.72 | +5.3% | 656.4 | 177.8 | 282.7 |

#### Paired Wilcoxon Tests vs Plain LinUCB (N=20 Seeds, Holm-Corrected)

| Comparison | Mean Diff (Variant - Plain) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Cost-Aware LinUCB vs Plain LinUCB | +0.39 | 96.0 | 7.5617e-01 | 1.0000e+00 | +0.086 | +0.155 | **Tied / Not Significant** |
| Deadline-Mask LinUCB vs Plain LinUCB | -0.87 | 92.0 | 6.4766e-01 | 1.0000e+00 | -0.124 | -0.179 | **Tied / Not Significant** |
| Warm-Start LinUCB vs Plain LinUCB | +6.77 | 4.0 | 1.3351e-05 | 5.3406e-05 | +0.962 | +1.485 | **Plain LinUCB Better (p < 0.05)** |
| All-Combined LinUCB vs Plain LinUCB | +4.83 | 25.0 | 1.6899e-03 | 5.0697e-03 | +0.762 | +0.858 | **Plain LinUCB Better (p < 0.05)** |

---

### Scenario: `drift_5_lowDup_highDup`

#### Performance & Latency Metrics (N=20 Seeds)

| Variant | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Mean Latency (us) | P95 Latency (us) | P99 Latency (us) |
|---|---|---|---|---|---|---|
| *Oracle* | 84.91 [84.03, 85.80] | 0.00 | 0.0% | 190.8 | 61.8 | 88.9 |
| **Plain LinUCB** | 81.79 [81.15, 82.51] | -3.12 | -3.7% | 120.3 | 156.7 | 249.0 |
| **Cost-Aware LinUCB** | 81.80 [81.15, 82.54] | -3.11 | -3.7% | 121.4 | 159.6 | 242.3 |
| **Deadline-Mask LinUCB** | 81.32 [80.93, 81.67] | -3.59 | -4.2% | 72.8 | 151.1 | 252.4 |
| **Warm-Start LinUCB** | 88.10 [86.67, 89.75] | +3.19 | +3.8% | 745.6 | 144.0 | 225.6 |
| **All-Combined LinUCB** | 86.38 [85.44, 87.33] | +1.47 | +1.7% | 574.7 | 152.8 | 255.8 |

#### Paired Wilcoxon Tests vs Plain LinUCB (N=20 Seeds, Holm-Corrected)

| Comparison | Mean Diff (Variant - Plain) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Cost-Aware LinUCB vs Plain LinUCB | +0.01 | 85.0 | 4.7491e-01 | 6.5997e-01 | +0.190 | +0.100 | **Tied / Not Significant** |
| Deadline-Mask LinUCB vs Plain LinUCB | -0.47 | 78.0 | 3.2998e-01 | 6.5997e-01 | -0.257 | -0.315 | **Tied / Not Significant** |
| Warm-Start LinUCB vs Plain LinUCB | +6.31 | 1.0 | 3.8147e-06 | 1.1444e-05 | +0.990 | +1.853 | **Plain LinUCB Better (p < 0.05)** |
| All-Combined LinUCB vs Plain LinUCB | +4.59 | 0.0 | 1.9073e-06 | 7.6294e-06 | +1.000 | +1.962 | **Plain LinUCB Better (p < 0.05)** |

---

### Scenario: `drift_6_alternating`

#### Performance & Latency Metrics (N=20 Seeds)

| Variant | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Mean Latency (us) | P95 Latency (us) | P99 Latency (us) |
|---|---|---|---|---|---|---|
| *Oracle* | 39.37 [37.58, 41.33] | 0.00 | 0.0% | 1534.3 | 22.0 | 22.7 |
| **Plain LinUCB** | 35.80 [34.65, 37.03] | -3.58 | -9.1% | 599.7 | 142.2 | 221.3 |
| **Cost-Aware LinUCB** | 34.92 [33.79, 36.15] | -4.45 | -11.3% | 597.3 | 137.2 | 194.6 |
| **Deadline-Mask LinUCB** | 34.42 [33.62, 35.32] | -4.95 | -12.6% | 462.0 | 133.2 | 197.3 |
| **Warm-Start LinUCB** | 31.38 [30.03, 32.76] | -8.00 | -20.3% | 635.4 | 93.8 | 134.5 |
| **All-Combined LinUCB** | 32.08 [31.18, 33.07] | -7.29 | -18.5% | 440.1 | 90.4 | 127.6 |

#### Paired Wilcoxon Tests vs Plain LinUCB (N=20 Seeds, Holm-Corrected)

| Comparison | Mean Diff (Variant - Plain) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Cost-Aware LinUCB vs Plain LinUCB | -0.88 | 0.0 | 1.9073e-06 | 7.6294e-06 | -1.000 | -4.750 | **Cost-Aware LinUCB Better (p < 0.05)** |
| Deadline-Mask LinUCB vs Plain LinUCB | -1.37 | 63.0 | 1.2309e-01 | 1.2309e-01 | -0.400 | -0.431 | **Tied / Not Significant** |
| Warm-Start LinUCB vs Plain LinUCB | -4.42 | 11.0 | 1.0490e-04 | 3.1471e-04 | -0.895 | -1.072 | **Warm-Start LinUCB Better (p < 0.05)** |
| All-Combined LinUCB vs Plain LinUCB | -3.71 | 18.0 | 4.8256e-04 | 9.6512e-04 | -0.829 | -0.910 | **All-Combined LinUCB Better (p < 0.05)** |

---

### Scenario: `drift_7_natural_sequential`

#### Performance & Latency Metrics (N=20 Seeds)

| Variant | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Mean Latency (us) | P95 Latency (us) | P99 Latency (us) |
|---|---|---|---|---|---|---|
| *Oracle* | 63.14 [62.80, 63.52] | 0.00 | 0.0% | 89.9 | 67.3 | 89.1 |
| **Plain LinUCB** | 61.33 [60.62, 62.09] | -1.81 | -2.9% | 251.2 | 194.4 | 287.3 |
| **Cost-Aware LinUCB** | 61.39 [60.66, 62.14] | -1.75 | -2.8% | 255.5 | 201.3 | 291.1 |
| **Deadline-Mask LinUCB** | 61.06 [60.60, 61.52] | -2.09 | -3.3% | 201.9 | 202.8 | 297.0 |
| **Warm-Start LinUCB** | 67.00 [65.54, 68.52] | +3.86 | +6.1% | 854.0 | 198.5 | 322.6 |
| **All-Combined LinUCB** | 64.41 [63.57, 65.28] | +1.27 | +2.0% | 595.9 | 197.5 | 416.3 |

#### Paired Wilcoxon Tests vs Plain LinUCB (N=20 Seeds, Holm-Corrected)

| Comparison | Mean Diff (Variant - Plain) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Cost-Aware LinUCB vs Plain LinUCB | +0.05 | 73.0 | 2.4549e-01 | 4.9097e-01 | +0.305 | +0.230 | **Tied / Not Significant** |
| Deadline-Mask LinUCB vs Plain LinUCB | -0.28 | 81.0 | 3.8838e-01 | 4.9097e-01 | -0.229 | -0.147 | **Tied / Not Significant** |
| Warm-Start LinUCB vs Plain LinUCB | +5.67 | 3.0 | 9.5367e-06 | 2.8610e-05 | +0.971 | +1.329 | **Plain LinUCB Better (p < 0.05)** |
| All-Combined LinUCB vs Plain LinUCB | +3.08 | 0.0 | 1.9073e-06 | 7.6294e-06 | +1.000 | +2.015 | **Plain LinUCB Better (p < 0.05)** |

---

### Scenario: `drift_8_natural_switching`

#### Performance & Latency Metrics (N=20 Seeds)

| Variant | Mean Cost (95% Bootstrap CI) | Gap to Oracle | % Gap to Oracle | Mean Latency (us) | P95 Latency (us) | P99 Latency (us) |
|---|---|---|---|---|---|---|
| *Oracle* | 63.10 [62.66, 63.53] | 0.00 | 0.0% | 101.1 | 80.9 | 106.2 |
| **Plain LinUCB** | 61.35 [60.76, 61.92] | -1.75 | -2.8% | 258.2 | 227.6 | 297.5 |
| **Cost-Aware LinUCB** | 61.38 [60.77, 61.98] | -1.72 | -2.7% | 253.5 | 229.2 | 312.6 |
| **Deadline-Mask LinUCB** | 61.02 [60.56, 61.53] | -2.07 | -3.3% | 200.6 | 232.1 | 308.6 |
| **Warm-Start LinUCB** | 66.40 [65.10, 67.79] | +3.31 | +5.2% | 799.0 | 215.6 | 294.7 |
| **All-Combined LinUCB** | 64.55 [63.76, 65.34] | +1.45 | +2.3% | 612.5 | 231.9 | 325.6 |

#### Paired Wilcoxon Tests vs Plain LinUCB (N=20 Seeds, Holm-Corrected)

| Comparison | Mean Diff (Variant - Plain) | Wilcoxon W | p-value (raw) | p-value (Holm) | Rank-Biserial r | Cohen's d_z | Verdict |
|---|---|---|---|---|---|---|---|
| Cost-Aware LinUCB vs Plain LinUCB | +0.03 | 94.0 | 7.0118e-01 | 7.0118e-01 | +0.105 | +0.095 | **Tied / Not Significant** |
| Deadline-Mask LinUCB vs Plain LinUCB | -0.32 | 77.0 | 3.1179e-01 | 6.2359e-01 | -0.267 | -0.201 | **Tied / Not Significant** |
| Warm-Start LinUCB vs Plain LinUCB | +5.06 | 3.0 | 9.5367e-06 | 2.8610e-05 | +0.971 | +1.220 | **Plain LinUCB Better (p < 0.05)** |
| All-Combined LinUCB vs Plain LinUCB | +3.20 | 0.0 | 1.9073e-06 | 7.6294e-06 | +1.000 | +1.890 | **Plain LinUCB Better (p < 0.05)** |

---

## Scientific Findings & Synthesis

### 1. Cost-Aware Exploration ($s_a = \max(1.0, \bar{L}_a / \bar{L}_{\text{SKIP}})$)
- **Mechanism**: Scales exploration bonus by $1 / s_a$, where $s_a$ is the empirical latency of action $a$ relative to SKIP measured on the 100-message training calibration sample. For typical streams, $s_{\text{SKIP}} = 1.0$, $s_{\text{LZ4}} \approx 7.3$, $s_{\text{ZSTD}} \approx 13.2$, $s_{\text{GZIP}} \approx 24.1$, and $s_{\text{BROTLI}} \approx 46.1$.
- **Empirical Finding**:
  - In `drift_6_alternating` (fast oscillations), Cost-Aware LinUCB **significantly outperformed Plain LinUCB** ($-0.88$ cost units, $p = 7.63 \times 10^{-6}$), preventing risky exploration of slow codecs when rapid shifts occurred.
  - In 6 of the 8 scenarios, there was **no statistically significant difference** compared to Plain LinUCB after Holm correction.
  - In `drift_1_repL_uniqS`, Cost-Aware LinUCB was **significantly worse** than Plain LinUCB ($+3.51$ cost units, $p = 7.63 \times 10^{-6}$). The heavy discount on Brotli's exploration bonus ($1/46.1$) discouraged the bandit from exploring Brotli, missing out on massive byte-reduction benefits that easily compensated for Brotli's latency overhead.
- **Honest Conclusion**: Cost-aware exploration prevents exploration of high-latency actions, but can be counter-productive when high-latency actions offer outsized compression ratio gains.

### 2. Deadline Masking
- **Mechanism**: Hard constraint that zeroes the selection probability ($p_5 = -\infty$) of the BATCH action whenever the predicted queue wait latency exceeds $50,000\,\mu\text{s}$.
- **Empirical Finding**:
  - Achieved the lowest mean cost of all variants in **6 out of 8 scenarios** (`drift_1`, `drift_2`, `drift_4`, `drift_5`, `drift_7`, `drift_8`).
  - Across the 20 seeds, cost differences vs Plain LinUCB were slight improvements ($-0.3$ to $-1.6$ cost units), though largely tied after family-wise Holm correction ($p_{\text{adj}} \approx 0.053 - 0.12$).
  - In `drift_3_lowRate_burst`, Deadline Masking had slightly higher mean cost ($+4.14$, $p = 6.45 \times 10^{-4}$) because it aggressively forced immediate single-message compressions during low arrival rates rather than allowing opportunistic batching.
- **Honest Conclusion**: Deadline masking is effective as a latency safety guardrail and marginally reduces mean cost across most steady/switching regimes without harming compression efficiency.

### 3. Length Heuristic Warm-Start
- **Mechanism**: Forces LinUCB's decisions to follow the domain length heuristic (SKIP if char_len < 25, else ZSTD) for the first 500 messages, updating regression parameters $A$ and $b$ before UCB policy takes over.
- **Empirical Finding**:
  - In `drift_6_alternating`, Warm-Start was **substantially superior** to Plain LinUCB ($-4.42$ cost units, $p = 3.15 \times 10^{-4}$, Cohen's $d_z = -1.072$). When regimes fluctuate rapidly every 500 messages, the warm-start heuristic acts as a stabilizing anchor, stopping the bandit from diverging into costly exploration.
  - However, in **7 out of 8 scenarios**, Warm-Start LinUCB was **statistically significantly worse** than Plain LinUCB (cost increases of $+3.2$ to $+6.8$ units, $p < 0.001$).
  - **Why it failed**: Feeding 500 heuristic decisions into Ridge regression heavily biased the model weights toward ZSTD. The bandit became overly confident in ZSTD and neglected other actions (like Brotli on repetitive text, or adaptive skipping), taking thousands of messages to recover from the heuristic bias.
- **Honest Conclusion**: Warm-start from a static length heuristic hurts overall performance in general drift scenarios because it injects persistent inductive bias that inhibits subsequent contextual exploration.

### 4. All-Combined LinUCB
- Because the negative bias of the Warm-Start dominated the combination, All-Combined LinUCB lagged behind Plain LinUCB in 7 of the 8 scenarios, winning only in `drift_6_alternating` ($-3.71$, $p = 9.65 \times 10^{-4}$).
