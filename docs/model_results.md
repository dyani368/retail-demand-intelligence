# Forecast evaluation

Run `python scripts/train_tree_model.py`. The fitted artifact and JSON metrics
are saved under `artifacts/tree_model/`.

The target is sales for the next day per store/product family. Validation uses
June 21–July 18, 2017. The final test uses July 19–August 15, 2017. Each split
contains 28 daily origins, with 1,782 series per date. We make one-day rolling
forecasts; actual sales from earlier dates are available to features on later
dates. The test labels are kept out of model fitting and early-stopping choices.

| Model | Split | RMSLE | WAPE | MAE |
|---|---|---:|---:|---:|
| Same weekday last week | Validation | 0.5142 | 17.46% | 83.29 |
| LightGBM | Validation | 0.3675 | 11.42% | 54.46 |
| Same weekday last week | Final test | 0.5468 | 18.51% | 86.93 |
| LightGBM | Final test | 0.3798 | 12.83% | 60.28 |

Final-test RMSLE decreased by **30.54%** relative to the seasonal baseline. The
calculation is `(0.54683 - 0.37984) / 0.54683 * 100`. Both models are scored on
the same 49,896 rows. The separate Kaggle test file has no sales labels, so it
is used only to generate future forecasts and a submission-shaped CSV.

## Model choices

We train against `log1p(sales)`, then apply `expm1` to predictions and clip at
zero. LightGBM uses 31 leaves, learning rate 0.05, up to 5,000 trees, and
categorical store/family IDs. Validation monitored squared log error, which is
the same per-row squared difference used by RMSLE before its final square root.
The selected iteration and whether the 5,000-tree ceiling was reached are
recorded in the generated metrics. We do not tune against the final test.

These metrics are evidence from this particular dataset, forecast protocol and
holdout period. They do not prove the same accuracy for another retailer or
future dates.
