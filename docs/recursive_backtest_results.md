# Fixed-origin recursive backtest

The project contains a 16-day fixed-origin backtest in
`scripts/evaluate_recursive_horizon.py`. At each origin, it selects a tree
count using the prior 28 one-day-ahead validation dates, refits through the
origin, then forecasts the next 16 days recursively. Later days use earlier
model predictions as sales history; held-out actuals are used only for scoring.
The weekly baseline follows the same recursive protocol.

The latest run was interrupted after two of three folds while the user checked
on the long-running process. It did not write a complete metrics file, and no
multi-day score is reported. The dashboard hides any older saved result unless
the current run status is explicitly `complete`. Run
`python scripts/evaluate_recursive_horizon.py` to finish and publish all three
folds after the updated feature definition.

The current one-day-ahead holdout metrics are separate and complete; see
`docs/model_results.md`. Do not mix one-day-ahead scores with recursive
multi-day scores.
