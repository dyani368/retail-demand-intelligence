# First baseline result

Run with `python scripts/evaluate_baseline.py`. Results are saved in
`artifacts/baseline/metrics.json`; test predictions are in
`artifacts/baseline/test_predictions.csv`.

| Split | Dates | RMSLE | WAPE | MAE | Rows |
|---|---|---:|---:|---:|---:|
| Validation | 2017-06-21 to 2017-07-18 | 0.5142 | 17.46% | 83.29 | 49,896 |
| Final test | 2017-07-19 to 2017-08-15 | 0.5468 | 18.51% | 86.93 | 49,896 |

The rule predicts each store-family's sales from the same weekday one week
earlier. It uses zero for a missing match. That fallback affects 0.65% of all
historical rows, mostly at the start of the dataset or around absent dates; no
test-period rows lack a baseline match.

Here “test” means our final labeled 28-day holdout, not Kaggle's separate
unlabeled test.csv.
