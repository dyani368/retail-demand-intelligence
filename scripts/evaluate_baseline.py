"""Evaluate a simple same-weekday-last-week forecast on recent dates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'store-sales-time-series-forecasting' / 'train.csv'
OUT = ROOT / 'artifacts' / 'baseline'
KEYS = ['date', 'store_nbr', 'family']


def score(actual: pd.Series, predicted: pd.Series) -> dict:
    """Return metrics with explicit definitions and safe zero handling."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 0)
    error = predicted - actual
    return {
        'rmsle': float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))),
        'wape_pct': float(100 * np.abs(error).sum() / np.abs(actual).sum()),
        'mae': float(np.abs(error).mean()),
        'rows': int(len(actual)),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=['date'], usecols=KEYS + ['sales'])
    # Match on the actual calendar date seven days earlier. This avoids treating
    # seven rows as seven days when Christmas dates are absent from the source.
    lag = df.rename(columns={'date': 'source_date', 'sales': 'sales_7_calendar_days_ago'})
    df['source_date'] = df['date'] - pd.Timedelta(days=7)
    df = df.merge(lag, on=['source_date', 'store_nbr', 'family'], how='left', validate='one_to_one')
    df['prediction'] = df['sales_7_calendar_days_ago'].fillna(0).clip(lower=0)

    # A rolling-origin evaluation: each target day uses observed sales from the
    # prior week. For later days, earlier test-period actuals are available.
    dates = sorted(df.date.unique())
    final_test = pd.Timestamp('2017-07-19')
    validation_start = pd.Timestamp('2017-06-21')
    validation_end = pd.Timestamp('2017-07-18')
    val = df[df.date.between(validation_start, validation_end)]
    test = df[df.date >= final_test]
    results = {
        'baseline': 'Predict sales from the same store and family exactly 7 calendar days earlier; use zero when unavailable.',
        'evaluation': 'Rolling next-day forecasts. Earlier observed actuals are available for each later day.',
        'validation_dates': [str(validation_start.date()), str(validation_end.date())],
        'test_dates': [str(pd.Timestamp(test.date.min()).date()), str(pd.Timestamp(test.date.max()).date())],
        'validation': score(val.sales, val.prediction),
        'test': score(test.sales, test.prediction),
        'limitations': [
            'This is a deliberately simple reference rule, not a trained model.',
            'Missing source dates have no prior-week value; zero is a simple fallback and will be reported.',
            'The Kaggle competition test file is separate and has no labels; local metrics use the last 28 labeled dates in train.csv.',
        ],
    }
    results['missing_baseline_prediction_rows'] = int(df.sales_7_calendar_days_ago.isna().sum())
    results['missing_baseline_prediction_pct'] = float(df.sales_7_calendar_days_ago.isna().mean() * 100)
    (OUT / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    test[['date', 'store_nbr', 'family', 'sales', 'prediction']].to_csv(OUT / 'test_predictions.csv', index=False)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
