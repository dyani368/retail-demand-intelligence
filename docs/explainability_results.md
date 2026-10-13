# TreeSHAP explanations

Run `python scripts/explain_model.py` to fit the validation model and calculate
LightGBM's native TreeSHAP contributions for 5,000 reproducibly sampled rows
from June 21–July 18, 2017. The sample is held out from that model's training.
The dashboard shows the global summary and lets you select a sampled local
example for the currently selected store and product family.

## Global feature contributions

Mean absolute SHAP measures how far, on average, a feature moves predictions
from the model baseline. Values are on the log1p(sales) scale.

| Feature | Mean absolute SHAP |
|---|---:|
| Previous 7-calendar-day sales average | 1.228 |
| Yesterday's sales | 0.638 |
| Promotion count | 0.326 |
| Product family | 0.122 |
| Sales 7 days earlier | 0.083 |

These numbers describe the fitted model's behavior on the sampled validation
rows. They do not establish that a feature caused sales to change.

## Local example

For store 50, DELI, on July 12, 2017, actual sales were 489.7 and the model
predicted 419.4. TreeSHAP attributes the prediction to the model's baseline
plus the contribution of each feature. The largest upward contributions in
this example came from the previous 7-day average, yesterday's sales and the
promotion count. This is one validation example, not a claim about all stores.

The local contributions add to the model output on the log1p scale (up to
floating-point rounding). Explanations are predictive, not causal.
