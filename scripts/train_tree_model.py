"""Train a small tree model and compare it with the seasonal baseline."""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'store-sales-time-series-forecasting' / 'train.csv'
OUT = ROOT / 'artifacts' / 'tree_model'
GROUP = ['store_nbr', 'family']
DATE = 'date'
FEATURES = ['sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
            'sales_mean_prev_7_days', 'sales_mean_prev_28_days',
            'onpromotion', 'day_of_week', 'month', 'store_nbr', 'family_code']


def score(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.maximum(np.asarray(pred, dtype=float), 0)
    return {'rmsle': float(np.sqrt(np.mean((np.log1p(y) - np.log1p(pred)) ** 2))),
            'wape_pct': float(100 * np.abs(y-pred).sum() / np.abs(y).sum()),
            'mae': float(np.abs(y-pred).mean())}


def make_features(df):
    df = df.sort_values(GROUP + [DATE]).copy()
    grouped = df.groupby(GROUP, sort=False).sales
    dates = df[DATE]
    for lag in (1, 7, 14, 28):
        prior_sales = grouped.shift(lag)
        prior_date = df.groupby(GROUP, sort=False)[DATE].shift(lag)
        # A shift of seven records is only accepted when it is exactly seven
        # calendar days earlier. This protects lags across missing Christmases.
        df[f'sales_lag_{lag}'] = prior_sales.where((dates - prior_date).dt.days.eq(lag))
    # Time-based windows exclude the target date and include only observations
    # within the prior calendar interval. A missing Christmas row cannot stretch
    # a nominal 7-day feature into an 8-day interval.
    group_date_index = pd.MultiIndex.from_frame(df[GROUP + [DATE]])
    for window in (7, 28):
        rolling = (df.set_index(DATE).groupby(GROUP, sort=False).sales
                   .rolling(f'{window}D', closed='left', min_periods=max(3, window // 2)).mean())
        df[f'sales_mean_prev_{window}_days'] = rolling.reindex(group_date_index).to_numpy()
    df['day_of_week'] = df[DATE].dt.dayofweek.astype('int8')
    df['month'] = df[DATE].dt.month.astype('int8')
    df['family_code'] = df.family.astype('category').cat.codes.astype('int8')
    return df


def make_model(n_estimators=5000):
    return LGBMRegressor(objective='regression', n_estimators=n_estimators,
        learning_rate=0.05, num_leaves=31, max_depth=-1,
        min_child_samples=100, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=1.0, max_bin=127,
        random_state=42, n_jobs=-1, verbosity=-1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=[DATE], usecols=[DATE, 'store_nbr', 'family', 'sales', 'onpromotion'])
    df = make_features(df)
    val_start, val_end = pd.Timestamp('2017-06-21'), pd.Timestamp('2017-07-18')
    test_start = pd.Timestamp('2017-07-19')
    val = df[df[DATE].between(val_start, val_end)]
    train_val = df[df[DATE] < val_start]
    # Fit with the training segment only. Stop when the later validation period
    # no longer improves, rather than selecting tree count on the final test.
    start = time.perf_counter()
    model = make_model(5000)
    model.fit(train_val[FEATURES], np.log1p(train_val.sales),
        categorical_feature=['store_nbr', 'family_code'],
        eval_X=val[FEATURES], eval_y=np.log1p(val.sales), eval_metric='l2',
        callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    best_iteration = model.best_iteration_ or model.n_estimators
    val_pred = np.expm1(model.predict(val[FEATURES])).clip(0)
    validation_seconds = time.perf_counter() - start
    # Freeze choices after validation; refit through validation, then evaluate
    # the final period once. Test labels are not used to fit the model.
    final_train = df[df[DATE] < test_start]
    test = df[df[DATE] >= test_start]
    start = time.perf_counter()
    final_model = make_model(best_iteration)
    final_model.fit(final_train[FEATURES], np.log1p(final_train.sales),
        categorical_feature=['store_nbr', 'family_code'])
    test_pred = np.expm1(final_model.predict(test[FEATURES])).clip(0)
    test_seconds = time.perf_counter() - start
    baseline_test = np.maximum(test.sales_lag_7.fillna(0).to_numpy(), 0)
    baseline_val = np.maximum(val.sales_lag_7.fillna(0).to_numpy(), 0)
    results = {
        'model': 'LightGBM LGBMRegressor, trained on log1p(sales)',
        'best_iteration_selected_on_validation': int(best_iteration),
        'hyperparameters': {'max_estimators': 5000, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 100,
                            'subsample': 0.8, 'colsample_bytree': 0.85, 'max_bin': 127,
                            'early_stopping_rounds': 100},
        'features': FEATURES,
        'feature_notes': ['sales lags use exact calendar offsets',
                          'rolling means use only earlier observations inside exact 7- and 28-calendar-day windows',
                          'promotion is assumed to be known for the target day'],
        'validation_dates': ['2017-06-21', '2017-07-18'],
        'test_dates': [str(test[DATE].min().date()), str(test[DATE].max().date())],
        'validation': {'lightgbm': score(val.sales, val_pred), 'seasonal_baseline': score(val.sales, baseline_val)},
        'test': {'lightgbm': score(test.sales, test_pred), 'seasonal_baseline': score(test.sales, baseline_test)},
        'fit_seconds': {'validation_model': round(validation_seconds, 2), 'final_model': round(test_seconds, 2)},
    }
    # After the untouched evaluation, fit a deployment model on every labeled
    # historical date. This artifact is for future forecasts, not test scoring.
    deployment_model = make_model(best_iteration)
    deployment_model.fit(df[FEATURES], np.log1p(df.sales),
                         categorical_feature=['store_nbr', 'family_code'])
    joblib.dump(deployment_model, OUT / 'forecast_model.joblib')
    (OUT / 'model_metadata.json').write_text(json.dumps({'features': FEATURES,
        'families': sorted(df.family.unique().tolist()), 'training_end': str(df[DATE].max().date()),
        'model': results['model']}, indent=2), encoding='utf-8')
    test[['date', 'store_nbr', 'family', 'sales']].assign(prediction=test_pred).to_csv(OUT / 'test_predictions.csv', index=False)
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
