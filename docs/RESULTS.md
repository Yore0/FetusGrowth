# Archived experiment results

## Scope

These are aggregate results from the private Huaxi/Shenzhen experiment. No source data or patient-level outputs are included. The headline robustness experiment used continuous-v2 checkpoint-1667 and the three raw numeric heads.

No ground-truth-based post-hoc correction is included in the numbers below.

## Full clean continuous-prefix split

| target | n eligible | MAE | RMSE | bias | Pearson r |
|---|---:|---:|---:|---:|---:|
| delivery day | 24,561 | 5.891 d | 9.102 d | +0.791 d | 0.303 |
| birth weight | 24,479 | 276.322 g | 360.388 g | +4.593 g | 0.482 |
| birth length | 20,543 | 0.841 cm | 1.330 cm | -0.012 cm | 0.424 |

## Delivery-day MAE by continuous cutoff bin

| cutoff bin | ours | DeepSeek-V4-Flash | Qwen3.5-9B base | Hulu-Med-7B |
|---|---:|---:|---:|---:|
| 70–111 | 6.336 | 7.698 | 7.643 | 11.352 |
| 112–181 | 6.240 | 7.712 | 10.020 | 14.847 |
| 182–215 | 5.840 | 7.374 | 8.961 | 26.368 |
| 216–244 | 5.657 | 7.213 | 10.243 | 27.659 |
| 245–258 | 5.372 | 6.852 | 8.842 | 21.253 |

Ours is the only compared model with a strictly decreasing clean delivery MAE across all five bins in this evaluation.

## Paired robustness

Mean normalized composite absolute-error increase versus the identical clean parent:

| perturbation | delta |
|---|---:|
| content mask 15 / 30 / 50% | +0.003 / +0.008 / +0.013 |
| visit dropout 20 / 40% | +0.007 / +0.013 |
| latest-visit dropout | +0.008 |
| ultrasound / lab removal | +0.039 / +0.005 |
| local window 28 / 56 / 84 d | +0.020 / +0.015 / +0.012 |
| compound realistic | +0.023 |

Ultrasound removal produces the largest expected degradation, especially for birth weight. The model remains responsive: paired predictions move when content is removed rather than collapsing to a fully invariant constant prior.

## Canonical cohort evaluation

An earlier checkpoint-selection report on canonical stage views found:

| split | delivery MAE | weight MAE | length MAE |
|---|---:|---:|---:|
| Shenzhen test | 5.720 d | 281.34 g | 0.675 cm |
| internal validation | 6.113 d | 287.03 g | 0.805 cm |
| Huaxi seen-test | 5.837 d | 270.39 g | 0.915 cm |

The normalized population composite decreased across all five stages in all three datasets. Huaxi seen-test is train-seen and is reported only as the project-specific exception.

## Limitations

- Prediction variance is still lower than target variance, indicating regression toward the population mean.
- MAE gains are stronger than correlation/R² gains.
- The headline checkpoint and the population-monotonic selector need not choose the same step; users should declare their selection rule before evaluating the test benchmark.
- Exact replication requires access to the private data and the matching base/adapter weights.
