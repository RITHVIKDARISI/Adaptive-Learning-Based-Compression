# SUMMARY FOR PAPER

**Generated**: 2026-10-14T14:25:40.260782

**System**: Windows 11, Intel64 Family 6 Model 186 Stepping 3, GenuineIntel, 15.72 GB RAM

**Python**: 3.13.5


## Dataset Statistics

| Dataset      | File                   |   Rows |   Mean_Len_Bytes |   Median_Len_Bytes |   Min_Len_Bytes |   Max_Len_Bytes |   Std_Len_Bytes |   Unique_Texts |   Duplicate_Rate |
|:-------------|:-----------------------|-------:|-----------------:|-------------------:|----------------:|----------------:|----------------:|---------------:|-----------------:|
| DailyDialog  | dailydialog_clean.csv  |   5000 |          12.8702 |            13.0000 |               2 |              32 |          5.5506 |            388 |           0.9224 |
| Sentiment140 | sentiment140_clean.csv |   5000 |          67.9720 |            63.0000 |               3 |             149 |         35.5048 |           4981 |           0.0038 |
| NUS-SMS      | nussms_clean.csv       |   5000 |          69.8016 |            51.0000 |               2 |             910 |         58.0390 |           4527 |           0.0946 |

## Main Results (lambda = 0.01)


### DailyDialog

| Scheduler           |   Compression_Ratio |   Space_Saving_Pct |   Mean_E2E_Latency_us |   P99_E2E_Latency_us |   Cache_Hit_Rate |   Total_Cost |
|:--------------------|--------------------:|-------------------:|----------------------:|---------------------:|-----------------:|-------------:|
| Logistic-Regression |              1.0000 |             0.0000 |                9.3954 |              37.6030 |           0.0000 |   32224.8840 |
| Decision-Tree       |              1.0000 |             0.0000 |                9.6497 |              29.8060 |           0.0000 |   32231.2420 |
| Random-Forest       |              1.0000 |             0.0000 |               10.1629 |              33.2140 |           0.0000 |   32244.0730 |
| Always-Skip         |              1.0000 |             0.0000 |               10.6100 |              36.0310 |           0.0000 |   32255.2500 |
| Gradient-Boosting   |              1.0000 |             0.0000 |               11.3469 |              36.6520 |           0.0000 |   32273.6730 |
| Heuristic           |              0.9829 |            -1.7443 |               12.4952 |              47.1130 |           0.0220 |   32855.1820 |
| LinUCB-Bandit       |              0.8774 |           -13.9731 |             1467.4892 |             470.5270 |           0.2068 |   39603.9490 |
| Always-Brotli       |              0.7618 |           -31.2598 |               23.8252 |             120.6059 |           0.9100 |   42427.3130 |
| Always-LZ4          |              0.7056 |           -41.7255 |               12.1100 |              39.4151 |           0.9100 |   45583.1320 |
| Always-Zstd         |              0.5871 |           -70.3345 |               10.6592 |              50.5090 |           0.9100 |   54682.5050 |
| Fixed-Batch         |              0.5804 |           -72.3038 |            87546.0621 |         1470977.7710 |           0.9096 |   55319.5150 |
| Always-Gzip         |              0.3913 |          -155.5424 |               34.0823 |             181.2720 |           0.9100 |   82101.2720 |


### Sentiment140

| Scheduler           |   Compression_Ratio |   Space_Saving_Pct |   Mean_E2E_Latency_us |   P99_E2E_Latency_us |   Cache_Hit_Rate |   Total_Cost |
|:--------------------|--------------------:|-------------------:|----------------------:|---------------------:|-----------------:|-------------:|
| Fixed-Batch         |              1.2524 |            20.1525 |           515857.5671 |         1295326.7479 |           0.0008 |  135829.3590 |
| Always-Brotli       |              1.1301 |            11.5153 |               89.9892 |             324.3620 |           0.0008 |  151858.4310 |
| Logistic-Regression |              1.1499 |            13.0393 |              251.7598 |             668.4091 |           0.0004 |  153470.9330 |
| Decision-Tree       |              1.1497 |            13.0186 |              281.8415 |             730.3990 |           0.0004 |  154259.9270 |
| LinUCB-Bandit       |              1.1284 |            11.3820 |              552.9271 |             580.0760 |           0.0008 |  154547.3390 |
| Heuristic           |              1.0051 |             0.5072 |               27.7568 |              78.4031 |           0.0004 |  169235.4710 |
| Always-Skip         |              1.0000 |             0.0000 |               10.8531 |              34.8030 |           0.0000 |  169828.3270 |
| Always-Zstd         |              0.9906 |            -0.9525 |               37.6792 |             159.9311 |           0.0008 |  171899.7770 |
| Gradient-Boosting   |              1.1496 |            13.0157 |             1370.0055 |            2681.7569 |           0.0004 |  181434.6190 |
| Always-LZ4          |              0.9309 |            -7.4252 |               14.3066 |              41.1020 |           0.0008 |  182461.6470 |
| Always-Gzip         |              0.8691 |           -15.0657 |               67.8534 |             171.8120 |           0.0008 |  196180.4100 |
| Random-Forest       |              1.1500 |            13.0410 |             5039.4688 |            8790.6700 |           0.0004 |  273094.9840 |


### NUS-SMS

| Scheduler           |   Compression_Ratio |   Space_Saving_Pct |   Mean_E2E_Latency_us |   P99_E2E_Latency_us |   Cache_Hit_Rate |   Total_Cost |
|:--------------------|--------------------:|-------------------:|----------------------:|---------------------:|-----------------:|-------------:|
| LinUCB-Bandit       |              1.1601 |            13.7987 |              512.2968 |             426.5130 |           0.0160 |  148501.8130 |
| Always-Brotli       |              1.1367 |            12.0267 |               64.6714 |             195.0060 |           0.0856 |  149631.4180 |
| Logistic-Regression |              1.1693 |            14.4774 |              379.1332 |             829.1420 |           0.0116 |  153327.4610 |
| Decision-Tree       |              1.1685 |            14.4205 |              412.7585 |             932.5390 |           0.0116 |  154266.4480 |
| Heuristic           |              1.0284 |             2.7597 |               23.9772 |              67.4279 |           0.0400 |  164385.9960 |
| Fixed-Batch         |              1.0266 |             2.5942 |          1735871.1885 |         7033867.4602 |           0.0852 |  164547.7520 |
| Always-Zstd         |              1.0096 |             0.9551 |               21.6733 |              64.5110 |           0.0856 |  167369.9790 |
| Always-Skip         |              1.0000 |             0.0000 |                6.2466 |              19.5000 |           0.0000 |  168722.1660 |
| Gradient-Boosting   |              1.1687 |            14.4329 |             1378.5334 |            2807.6860 |           0.0116 |  178474.6700 |
| Always-LZ4          |              0.9403 |            -6.3536 |               11.4598 |              30.7010 |           0.0856 |  179524.4030 |
| Always-Gzip         |              0.8812 |           -13.4879 |               53.6644 |             171.1870 |           0.0856 |  192145.7590 |
| Random-Forest       |              1.1692 |            14.4679 |             6738.1233 |           13264.9430 |           0.0112 |  312310.5920 |

## Best Scheduler by Dataset (lowest total cost at lambda=0.01)

- **DailyDialog**: Logistic-Regression (cost=32224.88, ratio=1.0000, latency=9.4 us)
- **Sentiment140**: Fixed-Batch (cost=135829.36, ratio=1.2524, latency=515857.6 us)
- **NUS-SMS**: LinUCB-Bandit (cost=148501.81, ratio=1.1601, latency=512.3 us)

## Decision Tree Feature Importances (lambda=0.01)

### Sentiment140

- char_len: 88.17%
- uppercase_ratio: 6.35%
- entropy: 4.29%
- punctuation_ratio: 0.68%
- unique_word_ratio: 0.26%
- word_len: 0.25%
- repetition_score: 0.00%
- arrival_rate: 0.00%
- emoji_ratio: 0.00%

## Lambda Sweep Summary

| Dataset | Lambda | Best Scheduler | Cost | Ratio |
|---------|--------|---------------|------|-------|
| DailyDialog | 0.001 | Decision-Tree | 32012.45 | 1.0000 |
| DailyDialog | 0.01 | Logistic-Regression | 32224.88 | 1.0000 |
| DailyDialog | 0.1 | Logistic-Regression | 34079.64 | 1.0000 |
| DailyDialog | 1.0 | Decision-Tree | 51487.40 | 1.0000 |
| Sentiment140 | 0.001 | Logistic-Regression | 142484.76 | 1.1945 |
| Sentiment140 | 0.01 | Logistic-Regression | 153470.93 | 1.1499 |
| Sentiment140 | 0.1 | LinUCB-Bandit | 195556.79 | 1.1269 |
| Sentiment140 | 1.0 | LinUCB-Bandit | 484659.20 | 1.0001 |
| NUS-SMS | 0.001 | LinUCB-Bandit | 145573.15 | 1.1630 |
| NUS-SMS | 0.01 | LinUCB-Bandit | 148501.81 | 1.1601 |
| NUS-SMS | 0.1 | LinUCB-Bandit | 176696.89 | 1.1219 |
| NUS-SMS | 1.0 | LinUCB-Bandit | 396417.80 | 1.0033 |

## Figures

1. `benchmark_results/v2/pareto_frontier.png` - Pareto frontier across 3 datasets
2. `benchmark_results/v2/feature_importance.png` - Decision Tree feature importance breakdown

## Artifacts

- `benchmark_results/v2/experiment_results.json` - Full raw results
- `benchmark_results/v2/final_comparison_table.csv` - Main comparison table
- `benchmark_results/v2/dataset_statistics.csv` - Dataset statistics
- `benchmark_results/v2/environment.json` - Environment specs
- `data/MANIFEST.json` - Dataset provenance and checksums