"""Reproducible first-stage audit. Run: python scripts/audit_data.py"""
from pathlib import Path
import argparse
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def audit(data_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {p.stem: pd.read_csv(p, parse_dates=['date'] if p.stem != 'stores' else None)
              for p in sorted(data_dir.glob('*.csv')) if p.stem != 'sample_submission'}
    required = {'train', 'test', 'stores', 'oil', 'transactions', 'holidays_events'}
    if required - tables.keys():
        raise ValueError(f'Missing tables: {sorted(required - tables.keys())}')
    train = tables['train']
    keys = ['date', 'store_nbr', 'family']
    pairs = train[['store_nbr', 'family']].drop_duplicates()
    dates = pd.date_range(train.date.min(), train.date.max())
    missing_dates = dates.difference(train.date.unique())
    counts = train.groupby('date').size()
    # A complete observed-date panel still need not have every calendar date.
    pair_counts = train.groupby(['store_nbr', 'family']).size()
    store_join = train.merge(tables['stores'], on='store_nbr', how='left', validate='many_to_one', indicator=True)
    result = {
        'files': {k: {'rows': len(v), 'columns': list(v.columns),
                      'nulls': {c: int(n) for c, n in v.isna().sum().items()}}
                  for k, v in tables.items()},
        'train': {
            'rows': len(train), 'stores': int(train.store_nbr.nunique()),
            'families': int(train.family.nunique()), 'series': len(pairs),
            'start': str(train.date.min().date()), 'end': str(train.date.max().date()),
            'duplicate_keys': int(train.duplicated(keys).sum()),
            'duplicate_ids': int(train.id.duplicated().sum()),
            'negative_sales': int((train.sales < 0).sum()),
            'negative_promotions': int((train.onpromotion < 0).sum()),
            'zero_sales_pct': float(train.sales.eq(0).mean() * 100),
            'fractional_sales_rows': int((train.sales % 1 != 0).sum()),
            'missing_calendar_dates': [str(d.date()) for d in missing_dates],
            'observed_dates': int(train.date.nunique()),
            'rows_per_observed_date_min': int(counts.min()),
            'rows_per_observed_date_max': int(counts.max()),
            'observations_per_series_min': int(pair_counts.min()),
            'observations_per_series_max': int(pair_counts.max()),
            'sales_quantiles': {str(q): float(v) for q, v in train.sales.quantile([0, .5, .9, .99, 1]).items()},
            'store_join_rows': len(store_join),
            'unmatched_store_rows': int(store_join['_merge'].eq('left_only').sum()),
        },
        'auxiliary_duplicate_keys': {
            'stores': int(tables['stores'].duplicated(['store_nbr']).sum()),
            'oil': int(tables['oil'].duplicated(['date']).sum()),
            'transactions': int(tables['transactions'].duplicated(['date', 'store_nbr']).sum()),
            'holiday_dates': int(tables['holidays_events'].duplicated(['date']).sum()),
        },
        'test': {'start': str(tables['test'].date.min().date()),
                 'end': str(tables['test'].date.max().date()),
                 'duplicate_keys': int(tables['test'].duplicated(keys).sum()),
                 'contains_sales': 'sales' in tables['test'].columns},
    }
    train.groupby('date', as_index=False).agg(sales=('sales', 'sum'), promotions=('onpromotion', 'sum')).to_csv(output_dir / 'daily_sales.csv', index=False)
    train.groupby('family', as_index=False).agg(total_sales=('sales', 'sum'), mean_sales=('sales', 'mean'), zero_share=('sales', lambda s: s.eq(0).mean())).to_csv(output_dir / 'family_summary.csv', index=False)
    train.groupby(train.date.dt.dayofweek).sales.agg(['mean', 'median']).to_csv(output_dir / 'weekday_summary.csv')
    (output_dir / 'data_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    critical = (result['train']['duplicate_keys'], result['train']['negative_sales'], result['train']['unmatched_store_rows'])
    if any(critical):
        raise ValueError('Critical audit failure; see data_audit.json')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'store-sales-time-series-forecasting')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts' / 'audit')
    args = parser.parse_args()
    result = audit(args.data_dir, args.output_dir)
    print(json.dumps(result, indent=2))
