"""Create recursive forecasts for Kaggle's unlabeled future dates."""
from pathlib import Path
import json
import sys
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.forecasting import recursive_forecast
DATA_DIR = ROOT / 'store-sales-time-series-forecasting'
MODEL_DIR = ROOT / 'artifacts' / 'tree_model'
OUT = ROOT / 'artifacts' / 'forecasts'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    history = pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['date'],
                          usecols=['date', 'store_nbr', 'family', 'sales'])
    future = pd.read_csv(DATA_DIR / 'test.csv', parse_dates=['date'])
    metadata = json.loads((MODEL_DIR / 'model_metadata.json').read_text(encoding='utf-8'))
    model = joblib.load(MODEL_DIR / 'forecast_model.joblib')
    forecasts = recursive_forecast(model, history, future, metadata['families'])
    if len(forecasts) != len(future) or forecasts.id.duplicated().any():
        raise ValueError('Forecast IDs are missing or duplicated.')
    if set(forecasts.id) != set(future.id):
        raise ValueError('Forecast IDs do not match the competition test rows.')
    forecasts.to_csv(OUT / 'forecasts.csv', index=False)
    # Match the competition input's row order exactly.
    submission = (forecasts.set_index('id').loc[future.id.to_list(), ['prediction']]
                  .rename(columns={'prediction': 'sales'}).reset_index())
    submission.to_csv(OUT / 'submission.csv', index=False)
    (OUT / 'metadata.json').write_text(json.dumps({
        'features': metadata['features'],
        'model': metadata['model'],
        'training_end': metadata['training_end'],
        'forecast_start': str(forecasts.date.min().date()),
        'forecast_end': str(forecasts.date.max().date()),
        'rows': int(len(forecasts)),
    }, indent=2), encoding='utf-8')
    print(f'Wrote {len(forecasts):,} recursive forecasts for {forecasts.date.nunique()} dates.')


if __name__ == '__main__':
    main()
