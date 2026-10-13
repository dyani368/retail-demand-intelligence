"""Recursive daily forecasts with only past observed or predicted sales."""
from collections import defaultdict
from datetime import timedelta
import math
import pandas as pd

FEATURES = ['sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_28',
            'sales_mean_prev_7_days', 'sales_mean_prev_28_days',
            'onpromotion', 'day_of_week', 'month', 'store_nbr', 'family_code']


def recursive_forecast(model, history: pd.DataFrame, future: pd.DataFrame,
                       families: list[str]) -> pd.DataFrame:
    """Forecast future rows one date at a time; later days use earlier forecasts."""
    family_codes = {name: i for i, name in enumerate(sorted(families))}
    history = history.copy()
    history['date'] = pd.to_datetime(history.date)
    future = future.copy()
    future['date'] = pd.to_datetime(future.date)
    sales_by_series = defaultdict(dict)
    for row in history.itertuples(index=False):
        sales_by_series[(int(row.store_nbr), row.family)][row.date.date()] = float(row.sales)

    outputs = []
    for day in sorted(future.date.unique()):
        day_rows = future.loc[future.date.eq(day)]
        feature_rows = []
        for row in day_rows.itertuples(index=False):
            store, family = int(row.store_nbr), row.family
            known = sales_by_series[(store, family)]
            current = pd.Timestamp(day)
            feat = {}
            for lag in (1, 7, 14, 28):
                feat[f'sales_lag_{lag}'] = known.get((current - pd.Timedelta(days=lag)).date(), float('nan'))
            for window in (7, 28):
                window_start = (current - pd.Timedelta(days=window)).date()
                vals = [value for date, value in known.items()
                        if window_start <= date < current.date()]
                feat[f'sales_mean_prev_{window}_days'] = sum(vals) / len(vals) if len(vals) >= max(3, window // 2) else float('nan')
            feat.update({'onpromotion': float(row.onpromotion),
                         'day_of_week': current.dayofweek, 'month': current.month,
                         'store_nbr': store, 'family_code': family_codes[family]})
            feature_rows.append(feat)
        X = pd.DataFrame(feature_rows, columns=FEATURES)
        predictions = model.predict(X).clip(0)
        predictions = [max(0.0, math.expm1(float(v))) for v in predictions]
        result = day_rows[['id', 'date', 'store_nbr', 'family', 'onpromotion']].copy()
        result['prediction'] = predictions
        outputs.append(result)
        # These become the available history for the next forecast date.
        for row, prediction in zip(day_rows.itertuples(index=False), predictions):
            sales_by_series[(int(row.store_nbr), row.family)][pd.Timestamp(day).date()] = prediction
    return pd.concat(outputs, ignore_index=True)
