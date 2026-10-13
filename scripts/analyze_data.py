"""Explore audited summaries without reloading three million rows."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'artifacts' / 'audit'
OUT = ROOT / 'artifacts' / 'analysis'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    daily = pd.read_csv(AUDIT / 'daily_sales.csv', parse_dates=['date']).set_index('date')
    family = pd.read_csv(AUDIT / 'family_summary.csv').sort_values('total_sales', ascending=False)
    weekday = pd.read_csv(AUDIT / 'weekday_summary.csv', index_col=0)
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    axes[0].plot(daily.index, daily.sales, alpha=.35, linewidth=.6, label='Observed daily sales')
    # Calendar reindexing makes absent dates visibly unknown, not observed zeros.
    calendar = daily.reindex(pd.date_range(daily.index.min(), daily.index.max()))
    axes[0].plot(calendar.index, calendar.sales.rolling(28, min_periods=21).mean(), label='Trailing 28-calendar-day mean')
    axes[0].set(title='How does aggregate sales volume change over time?', ylabel='Sales (mixed family quantities)')
    axes[0].legend()
    top = family.head(10).sort_values('total_sales')
    axes[1].barh(top.family, top.total_sales)
    axes[1].set(title='Which families contribute the largest recorded quantities?', xlabel='Recorded sales quantity (not revenue)')
    axes[2].bar(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], weekday['mean'])
    axes[2].set(title='Does the mean store-family observation vary by weekday?', ylabel='Mean sales per observed row')
    fig.savefig(OUT / 'overview.png', dpi=140)
    plt.close(fig)
    summary = {
        'top_families_by_quantity': family.head(5)[['family', 'total_sales']].to_dict(orient='records'),
        'highest_mean_weekday': int(weekday['mean'].idxmax()),
        'lowest_mean_weekday': int(weekday['mean'].idxmin()),
        'weekday_high_low_ratio': float(weekday['mean'].max() / weekday['mean'].min()),
        'warning': 'Descriptive associations only; mixed family quantities are not revenue or SKU inventory.'
    }
    import json
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
