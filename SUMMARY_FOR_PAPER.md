# SUMMARY FOR PAPER: Adaptive Learning-Based Compression Benchmarks

**Generated**: 2026-10-14T16:07:40.424367
**System**: Windows 11 (10.0.26300), Intel64 Family 6 Model 186 Stepping 3, GenuineIntel, 15.72 GB RAM
**Python**: 3.13.5 | **Git Commit SHA**: `4edaa08e4fced1444609fa091d3c5c9e6181e496`

---

## 1. Verified Real Dataset Manifest (100% Real, Zero Synthetic Rows)

| Dataset | Clean CSV File | Hugging Face Source | Rows | Unique Texts | Duplicate % | Avg Bytes/Row | Source Breakdown |
|:---|:---|:---|---:|---:|---:|---:|:---|
| DailyDialog | `dailydialog_clean.csv` | `daily_dialog` | 5000 | 4822 | 3.56% | 60.65 | real: 5000 |
| Sentiment140 | `sentiment140_clean.csv` | `bdanko/sentiment140` | 5000 | 4981 | 0.38% | 68.77 | real: 5000 |
| SMS Spam Collection (ham only) | `sms_spam_ham_clean.csv` | `ucirvine/sms_spam` | 4827 | 4518 | 6.40% | 71.51 | real: 4827 |

---

## 2. Multi-Seed Benchmark Results (Seeds: 42, 123, 456, 789, 2026; Lambda = 0.01)
Latency accounting strictly satisfies: `e2e_lat_us = feat_lat + cache_lookup + sched_infer + comp + decomp + queue_wait + cache_store + cache_fill_comp`. Throughput is computed from a single `time.perf_counter()` timer around the stream loop.

### DailyDialog

| Scheduler | Compression Ratio | Space Saving % | Mean E2E Latency (us) | Mean Old Latency (us) | Latency Delta (us) | % Increase | Wall-Clock Throughput (msg/s) | Wall-Clock Throughput (MB/s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always-Zstd | 0.9776 | -2.29% | 755.68 | 17.16 | +738.52 | +4303.8% | 1651.6 | 0.0958 |
| Always-LZ4 | 0.9304 | -7.48% | 815.78 | 9.39 | +806.39 | +8587.6% | 1208.5 | 0.0701 |
| Heuristic | 1.0030 | 0.30% | 820.36 | 17.46 | +802.91 | +4599.6% | 1205.2 | 0.0699 |
| Length-Threshold | 1.0030 | 0.30% | 884.91 | 18.39 | +866.52 | +4711.4% | 1119.8 | 0.0649 |
| Always-Brotli | 1.1123 | 10.10% | 898.96 | 59.68 | +839.28 | +1406.2% | 1101.3 | 0.0639 |
| Always-Gzip | 0.8471 | -18.04% | 902.49 | 48.12 | +854.37 | +1775.5% | 1094.2 | 0.0634 |
| Logistic Regression | 1.1418 | 12.42% | 1071.96 | 217.84 | +854.12 | +392.1% | 921.1 | 0.0534 |
| Decision Tree | 1.1479 | 12.88% | 1084.65 | 239.38 | +845.27 | +353.1% | 909.3 | 0.0527 |
| Always-Skip | 1.0000 | 0.00% | 1211.74 | 8.39 | +1203.35 | +14340.0% | 818.0 | 0.0474 |
| LinUCB | 1.1402 | 12.30% | 1305.11 | 514.76 | +790.35 | +153.5% | 1025.5 | 0.0595 |
| Gradient Boosting | 1.1464 | 12.77% | 2217.99 | 1294.32 | +923.67 | +71.4% | 447.0 | 0.0259 |
| Random Forest | 1.1486 | 12.94% | 5743.56 | 4779.09 | +964.47 | +20.2% | 173.7 | 0.0101 |
| Random-Action | 0.9660 | -3.52% | 155648.66 | 154888.70 | +759.96 | +0.5% | 1220.6 | 0.0708 |
| Fixed-Batch | 1.1045 | 9.46% | 838797.84 | 838004.94 | +792.90 | +0.1% | 1202.7 | 0.0697 |

### Sentiment140

| Scheduler | Compression Ratio | Space Saving % | Mean E2E Latency (us) | Mean Old Latency (us) | Latency Delta (us) | % Increase | Wall-Clock Throughput (msg/s) | Wall-Clock Throughput (MB/s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always-LZ4 | 0.9323 | -7.26% | 791.32 | 8.71 | +782.61 | +8982.2% | 1255.2 | 0.0825 |
| Always-Skip | 1.0000 | 0.00% | 805.87 | 4.97 | +800.89 | +16111.3% | 1228.4 | 0.0807 |
| Always-Zstd | 0.9958 | -0.42% | 841.88 | 18.49 | +823.39 | +4452.6% | 1171.0 | 0.0769 |
| Always-Gzip | 0.8754 | -14.23% | 865.95 | 46.33 | +819.61 | +1769.0% | 1146.1 | 0.0753 |
| Always-Brotli | 1.1343 | 11.84% | 891.58 | 59.02 | +832.56 | +1410.6% | 1107.6 | 0.0728 |
| Decision Tree | 1.1524 | 13.23% | 975.41 | 216.04 | +759.36 | +351.5% | 1024.2 | 0.0673 |
| Heuristic | 1.0092 | 0.91% | 1142.51 | 26.73 | +1115.78 | +4174.5% | 875.5 | 0.0575 |
| Logistic Regression | 1.1519 | 13.18% | 1190.23 | 248.50 | +941.73 | +379.0% | 844.4 | 0.0555 |
| LinUCB | 1.1320 | 11.66% | 1266.99 | 409.09 | +857.91 | +209.7% | 925.0 | 0.0608 |
| Length-Threshold | 1.0092 | 0.91% | 1360.86 | 31.35 | +1329.51 | +4241.3% | 725.5 | 0.0477 |
| Gradient Boosting | 1.1527 | 13.25% | 2184.49 | 1258.34 | +926.16 | +73.6% | 454.0 | 0.0298 |
| Random Forest | 1.1533 | 13.29% | 5902.38 | 4877.01 | +1025.38 | +21.0% | 169.0 | 0.0111 |
| Random-Action | 0.9907 | -0.93% | 106383.54 | 105013.00 | +1370.54 | +1.3% | 679.9 | 0.0447 |
| Fixed-Batch | 1.2594 | 20.60% | 520314.96 | 519092.90 | +1222.06 | +0.2% | 835.1 | 0.0549 |

### SMS-Spam-Ham

| Scheduler | Compression Ratio | Space Saving % | Mean E2E Latency (us) | Mean Old Latency (us) | Latency Delta (us) | % Increase | Wall-Clock Throughput (msg/s) | Wall-Clock Throughput (MB/s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always-Skip | 1.0000 | 0.00% | 806.93 | 5.11 | +801.82 | +15683.6% | 1227.2 | 0.0832 |
| Always-LZ4 | 0.9454 | -5.77% | 809.76 | 8.52 | +801.24 | +9403.3% | 1217.3 | 0.0825 |
| Always-Zstd | 1.0212 | 2.08% | 840.40 | 17.63 | +822.76 | +4665.6% | 1173.7 | 0.0796 |
| Always-Gzip | 0.8968 | -11.51% | 872.40 | 47.54 | +824.85 | +1735.0% | 1134.7 | 0.0769 |
| Always-Brotli | 1.1471 | 12.82% | 940.86 | 61.72 | +879.14 | +1424.4% | 1051.4 | 0.0713 |
| Length-Threshold | 1.0346 | 3.34% | 1028.39 | 21.83 | +1006.55 | +4610.5% | 1003.4 | 0.0680 |
| Heuristic | 1.0346 | 3.34% | 1340.68 | 30.02 | +1310.66 | +4366.3% | 736.6 | 0.0499 |
| LinUCB | 1.1350 | 11.90% | 1349.72 | 464.18 | +885.54 | +190.8% | 907.4 | 0.0615 |
| Decision Tree | 1.1744 | 14.85% | 1682.77 | 358.65 | +1324.11 | +369.2% | 588.4 | 0.0399 |
| Logistic Regression | 1.1662 | 14.25% | 1708.48 | 347.63 | +1360.85 | +391.5% | 577.7 | 0.0392 |
| Gradient Boosting | 1.1732 | 14.76% | 2396.35 | 1393.94 | +1002.42 | +71.9% | 414.3 | 0.0281 |
| Random Forest | 1.1752 | 14.91% | 64608.92 | 63420.51 | +1188.41 | +1.9% | 123.4 | 0.0084 |
| Random-Action | 0.9922 | -0.79% | 320652.31 | 319823.23 | +829.08 | +0.3% | 1122.6 | 0.0761 |
| Fixed-Batch | 1.0412 | 3.96% | 1879693.32 | 1878864.83 | +828.48 | +0.0% | 1149.6 | 0.0779 |

---

## 3. Overall Cross-Dataset Latency Shift (Old Definition vs. New 8-Stage Definition)

| Scheduler | Mean Old Latency (us) | Mean New Latency (us) | Absolute Overhead (us) | % Increase | Mean Throughput (msg/s) |
|:---|---:|---:|---:|---:|---:|
| Always-LZ4 | 8.87 | 805.62 | +796.75 | +8977.8% | 1227.0 |
| Always-Zstd | 17.76 | 812.65 | +794.89 | +4475.2% | 1332.1 |
| Always-Gzip | 47.33 | 880.28 | +832.95 | +1759.8% | 1125.0 |
| Always-Brotli | 60.14 | 910.47 | +850.32 | +1413.9% | 1086.8 |
| Always-Skip | 6.16 | 941.51 | +935.36 | +15188.4% | 1091.2 |
| Length-Threshold | 23.86 | 1091.39 | +1067.53 | +4474.7% | 949.6 |
| Heuristic | 24.73 | 1101.18 | +1076.45 | +4352.1% | 939.1 |
| Decision Tree | 271.36 | 1247.61 | +976.25 | +359.8% | 840.6 |
| LinUCB | 462.68 | 1307.28 | +844.60 | +182.5% | 952.6 |
| Logistic Regression | 271.32 | 1323.56 | +1052.24 | +387.8% | 781.1 |
| Gradient Boosting | 1315.53 | 2266.28 | +950.75 | +72.3% | 438.4 |
| Random Forest | 24358.87 | 25418.29 | +1059.42 | +4.3% | 155.4 |
| Random-Action | 193241.64 | 194228.17 | +986.53 | +0.5% | 1007.7 |
| Fixed-Batch | 1078654.23 | 1079602.04 | +947.82 | +0.1% | 1062.4 |

---
*Artifacts generated in `benchmark_results/v2/timing/` with raw records, environment specs, and per-message Parquet trace.*
