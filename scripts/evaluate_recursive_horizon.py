"""Backtest fixed-origin recursive forecasts at the competition horizon."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
from lightgbm import early_stopping, log_evaluation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.forecasting import recursive_forecast
from train_tree_model import DATA, DATE, FEATURES, make_features, make_model, score

OUT = ROOT / 'artifacts' / 'horizon_backtest'
STATUS = OUT / 'status.json'
HORIZON_DAYS = 16
# All forecast windows finish before the later June validation and July/August
# holdouts used by the existing one-day-ahead evaluation.
ORIGINS = [pd.Timestamp('2017-04-30'), pd.Timestamp('2017-05-31'), pd.Timestamp('2017-06-20')]


def recursive_weekly_baseline(history, future):
    """Repeat the same weekday last week, feeding predictions into later days."""
    known = {}
    for row in history.itertuples(index=False):
        known[(int(row.store_nbr), row.family, pd.Timestamp(row.date).date())] = float(row.sales)
    outputs = []
    for day in sorted(pd.to_datetime(future[DATE].unique())):
        day_rows = future.loc[future[DATE].eq(day)]
        predictions = []
        for row in day_rows.itertuples(index=False):
            prior_date = (pd.Timestamp(day) - pd.Timedelta(days=7)).date()
            predictions.append(max(0.0, known.get((int(row.store_nbr), row.family, prior_date), 0.0)))
        result = day_rows[[DATE, 'store_nbr', 'family', 'sales']].copy()
        result['prediction'] = predictions
        outputs.append(result)
        for row, prediction in zip(day_rows.itertuples(index=False), predictions):
            known[(int(row.store_nbr), row.family, pd.Timestamp(day).date())] = prediction
    return pd.concat(outputs, ignore_index=True)


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({'status': 'running', 'started_at': pd.Timestamp.now().isoformat(),
                                  'horizon_days': HORIZON_DAYS, 'origins': [str(d.date()) for d in ORIGINS]}, indent=2),
                      encoding='utf-8')
    raw = pd.read_csv(DATA, parse_dates=[DATE], usecols=[DATE, 'store_nbr', 'family', 'sales', 'onpromotion'])
    featured = make_features(raw)
    all_predictions, fold_results, horizon_results = [], [], []
    families = sorted(raw.family.unique().tolist())

    for fold_no, origin in enumerate(ORIGINS, start=1):
        val_start = origin - pd.Timedelta(days=27)
        val = featured[featured[DATE].between(val_start, origin)]
        train = featured[featured[DATE] < val_start]
        selector = make_model(5000)
        selector.fit(train[FEATURES], np.log1p(train.sales),
                     categorical_feature=['store_nbr', 'family_code'],
                     eval_X=val[FEATURES], eval_y=np.log1p(val.sales), eval_metric='l2',
                     callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        selected_trees = int(selector.best_iteration_ or selector.n_estimators)

        through_origin = featured[featured[DATE] <= origin]
        model = make_model(selected_trees)
        model.fit(through_origin[FEATURES], np.log1p(through_origin.sales),
                  categorical_feature=['store_nbr', 'family_code'])
        end = origin + pd.Timedelta(days=HORIZON_DAYS)
        future = raw[raw[DATE].between(origin + pd.Timedelta(days=1), end)].copy()
        history = raw[raw[DATE] <= origin][[DATE, 'store_nbr', 'family', 'sales']]
        # The data has a known ID only in Kaggle's test file. This synthetic key
        # is internal to the shared recursive forecasting function.
        future['id'] = (future[DATE].dt.strftime('%Y%m%d') + '_' + future.store_nbr.astype(str)
                        + '_' + future.family.astype(str))
        pred = recursive_forecast(model, history, future[[
            'id', DATE, 'store_nbr', 'family', 'onpromotion']], families)
        actual = future[[DATE, 'store_nbr', 'family', 'sales']]
        pred = pred.merge(actual, on=[DATE, 'store_nbr', 'family'], validate='one_to_one')
        pred['origin'] = origin
        pred['horizon_day'] = (pred[DATE] - origin).dt.days
        baseline = recursive_weekly_baseline(history, future)
        pred = pred.merge(baseline[[DATE, 'store_nbr', 'family', 'prediction']].rename(
            columns={'prediction': 'baseline_prediction'}), on=[DATE, 'store_nbr', 'family'], validate='one_to_one')
        pred = pred.rename(columns={'prediction': 'lightgbm_prediction'})
        all_predictions.append(pred)

        fold_results.append({
            'fold': fold_no, 'origin': str(origin.date()),
            'forecast_start': str(future[DATE].min().date()), 'forecast_end': str(future[DATE].max().date()),
            'horizon_days': int(future[DATE].nunique()), 'selected_trees': selected_trees,
            'lightgbm': score(pred.sales, pred.lightgbm_prediction),
            'recursive_weekly_baseline': score(pred.sales, pred.baseline_prediction),
        })
        for label, subset in [('days_1_to_7', pred[pred.horizon_day <= 7]),
                              ('days_8_to_16', pred[pred.horizon_day > 7])]:
            horizon_results.append({
                'fold': fold_no, 'origin': str(origin.date()), 'horizon_group': label,
                'horizon_days': int(subset.horizon_day.nunique()),
                'lightgbm': score(subset.sales, subset.lightgbm_prediction),
                'recursive_weekly_baseline': score(subset.sales, subset.baseline_prediction),
            })
        print(f"Fold {fold_no}/{len(ORIGINS)} complete: origin={origin.date()}, trees={selected_trees}, "
              f"LightGBM RMSLE={fold_results[-1]['lightgbm']['rmsle']:.4f}", flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(OUT / 'predictions.csv', index=False)
    combined = {
        'protocol': ('At each origin, select tree count on the immediately preceding 28 one-day-ahead dates, '
                    'refit through the origin, then forecast the next 16 calendar days recursively. '
                    'Later days use earlier model predictions, not held-out actual sales.'),
        'horizon_days': HORIZON_DAYS,
        'folds': fold_results,
        'horizon_groups': horizon_results,
        'aggregate': {
            'lightgbm': score(predictions.sales, predictions.lightgbm_prediction),
            'recursive_weekly_baseline': score(predictions.sales, predictions.baseline_prediction),
        },
        'runtime_seconds': round(time.perf_counter() - started, 2),
    }
    (OUT / 'metrics.json').write_text(json.dumps(combined, indent=2), encoding='utf-8')
    STATUS.write_text(json.dumps({'status': 'complete', 'finished_at': pd.Timestamp.now().isoformat(),
                                  'horizon_days': HORIZON_DAYS, 'origins': [str(d.date()) for d in ORIGINS]}, indent=2),
                      encoding='utf-8')
    print(json.dumps(combined, indent=2))


if __name__ == '__main__':
    main()
