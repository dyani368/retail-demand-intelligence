"""Fit the validation model and export global and local TreeSHAP explanations."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from train_tree_model import DATA, DATE, FEATURES, make_features, make_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts' / 'explainability'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA, parse_dates=[DATE], usecols=[DATE, 'store_nbr', 'family', 'sales', 'onpromotion'])
    data = make_features(raw)
    validation_start = pd.Timestamp('2017-06-21')
    validation_end = pd.Timestamp('2017-07-18')
    train = data[data[DATE] < validation_start]
    validation = data[data[DATE].between(validation_start, validation_end)]

    model_metrics = json.loads((ROOT / 'artifacts' / 'tree_model' / 'metrics.json').read_text(encoding='utf-8'))
    tree_count = int(model_metrics['best_iteration_selected_on_validation'])
    model = make_model(tree_count)
    model.fit(train[FEATURES], np.log1p(train.sales), categorical_feature=['store_nbr', 'family_code'])

    sample = validation.sample(n=min(5000, len(validation)), random_state=42).copy()
    # LightGBM's native pred_contrib uses exact TreeSHAP for this tree model.
    contributions = model.booster_.predict(sample[FEATURES], pred_contrib=True)
    shap_values = contributions[:, :-1]
    predictions = np.expm1(model.predict(sample[FEATURES])).clip(0)

    summary = pd.DataFrame({
        'feature': FEATURES,
        'mean_absolute_shap_log_sales': np.abs(shap_values).mean(axis=0),
        'mean_shap_log_sales': shap_values.mean(axis=0),
    }).sort_values('mean_absolute_shap_log_sales', ascending=False)
    summary.to_csv(OUT / 'shap_summary.csv', index=False)

    # Keep the sampled inputs and per-row contributions for inspection in the UI.
    examples = sample[[DATE, 'store_nbr', 'family', 'sales', *FEATURES]].copy()
    examples['predicted_sales'] = predictions
    examples['base_value_log1p'] = contributions[:, -1]
    for index, feature in enumerate(FEATURES):
        examples[f'shap__{feature}'] = shap_values[:, index]
    examples.to_csv(OUT / 'shap_examples.csv', index=False)

    # Choose a representative validation prediction near the center of absolute
    # error, avoiding a cherry-picked best or worst example.
    errors = np.abs(np.log1p(sample.sales.to_numpy()) - np.log1p(predictions))
    local_i = int(np.argsort(errors)[len(errors) // 2])
    local = {
        'date': str(sample.iloc[local_i][DATE].date()),
        'store_nbr': int(sample.iloc[local_i].store_nbr),
        'family': sample.iloc[local_i].family,
        'actual_sales': float(sample.iloc[local_i].sales),
        'predicted_sales': float(predictions[local_i]),
        'features': {name: float(sample.iloc[local_i][name]) for name in FEATURES},
        'contributions_on_log1p_scale': {name: float(value) for name, value in zip(FEATURES, shap_values[local_i])},
        'base_value': float(contributions[local_i, -1]),
        'note': 'Feature contributions add to the model prediction on log1p(sales) scale; they explain prediction, not cause.',
    }
    (OUT / 'local_explanation.json').write_text(json.dumps(local, indent=2), encoding='utf-8')

    metadata = {
        'features': FEATURES,
        'tree_count': tree_count,
        'validation_dates': [str(validation_start.date()), str(validation_end.date())],
        'shap_method': 'LightGBM native TreeSHAP contributions',
        'sample_rows': int(len(sample)),
        'local_case_selection': 'median absolute log-scale validation error in the reproducible sample',
    }
    (OUT / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps({
        'metadata': metadata,
        'top_features': summary.head(5).to_dict(orient='records'),
        'local_case': {key: local[key] for key in ('date', 'store_nbr', 'family', 'actual_sales', 'predicted_sales')},
    }, indent=2))


if __name__ == '__main__':
    main()
