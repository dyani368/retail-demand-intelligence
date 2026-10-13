# Learning guide: data to deployment

## 1. Define the problem before writing model code

The target is sales tomorrow for one store and one product family. A forecast
needs an origin (when issued), a horizon (how far ahead), and a grain (what one
prediction represents). Our initial horizon is one day and our grain is a
store-family-day. A 28-day test window means 28 successive next-day forecasts;
it does not mean one forecast issued 28 days ahead.

Why this matters: a lag of one day is available for a next-day forecast, but actual
sales from tomorrow are unavailable when forecasting the entire coming week.
Multi-day replenishment therefore requires a separate forecast protocol.

## 2. Understand each table

| Table | Grain | Role and caution |
|---|---|---|
| train | Store-family-date | Sales target and promotion count |
| stores | Store | City, state, type and cluster; many sales rows map to one store |
| holidays_events | Event/locale/date | Multiple events can share dates; local events apply only to relevant locations |
| oil | Date | Some observations missing; future prices unknown |
| transactions | Store-date | Historical context; target-day counts unavailable at forecast origin |
| test | Store-family-date | Future inputs without labels; cannot directly calculate local error |
| sample_submission | Prediction ID | Kaggle output template, not historical evidence |

## 3. Why the audit comes first

Duplicate store-family-date keys would count the same observation more than once.
Missing calendar days make a row-based shift different from a true day-based lag.
Missing cells and missing rows are separate problems: a CSV can have no empty
cells while still omitting dates. We will align each series to a daily calendar
before constructing features and will not automatically turn unknown sales into
zero. Zero means observed no sales; missing means no observation.

Joining stores should preserve the number of sales rows. The script uses a
many-to-one join check and counts unmatched records. Holidays are different:
joining raw events on date alone can multiply rows and sales totals. They require
locale-aware aggregation before a sales join. Missing oil observations should
not be filled using future prices.

## 4. Interview checks for this stage

1. What does one row represent, and what is the target?
2. Why is a 28-day test window different from a 28-day forecast horizon?
3. Why does zero sales not necessarily mean zero demand?
4. How could a holiday join accidentally inflate total sales?
5. Why must we check missing dates even when no cells are empty?

Answer in plain language using an example from our data. Later stages will add
their implementation choices, measured findings and interview questions here.

## 5. First forecast: same weekday last week

Before training a machine-learning model, we use a clear reference rule: for
tomorrow, predict the sales from the same store and family seven calendar days
earlier. This is the seasonal baseline. It is useful because grocery sales often
follow weekly patterns. A complicated model must improve on this rule to justify
its extra complexity.

The script matches actual dates, rather than shifting each series by seven rows.
That distinction matters because Christmas is missing from the source. When the
matching date is unavailable, this first baseline uses zero; the script counts
those cases so we can inspect the limitation.

We used the 28 dates from 2017-06-21 to 2017-07-18 for validation, and the last
28 available dates, 2017-07-19 to 2017-08-15, as the final test. Each date
contains 1,782 store-family observations. The model predicts one day at a time;
when assessing a later date, the previous day's observed sales are available.

### What the metrics mean

- **RMSLE 0.547:** typical error measured after compressing large sales values
  with a logarithm. Lower is better; it is not an error of 0.547 sales units.
- **WAPE 18.5%:** the sum of absolute errors was 18.5% of total actual sales
  across all rows in the test window. It gives more influence to high-volume
  observations and remains defined when some individual actuals are zero.
- **MAE 86.9:** average absolute miss is 86.9 recorded sales units per
  store-family-day. Family sizes differ, so this is less comparable across
  families than a scale-adjusted metric.

These are baseline results, not the final model results. The validation score
can guide model choices. We keep the final test period for one final comparison
after decisions are made.

### Interview check

Why compare LightGBM with a seasonal baseline? A good answer: "The baseline
shows what a simple weekly rule can do. It gives me a reference point, so I can
tell whether the added complexity improves forecasts on future dates."

## 6. Add model features and train a tree model

A feature is an input clue for a prediction. We added yesterday's sales, sales
from one, two and four weeks earlier, recent averages, target weekday and month,
promotion count, store and family. A tree model learns rules such as "when this
family's recent sales are high and tomorrow is Sunday, predict more."

The lag values are checked against calendar dates, so a missing Christmas does
not silently shift the meaning of a one-week lag. Rolling averages use exact
prior 7- and 28-calendar-day windows and exclude the target date. Promotion is
used as a feature under the assumption that tomorrow's
promotion plan is known when tomorrow's forecast is made.

We train on log1p(sales), which reduces the influence of very large sales rows.
Predictions are converted back with expm1 before scoring. We first used
scikit-learn's HistGradientBoostingRegressor as a prototype, then reran the
comparison with LightGBM after it was installed. LightGBM is the final model.

On the final 28 dates, LightGBM scored 0.3798 RMSLE, 12.83% WAPE and 60.28 MAE.
The seasonal baseline scored 0.5468 RMSLE, 18.51% WAPE and 86.93 MAE on the same
rows. That is a 30.5% reduction in RMSLE versus the baseline. This result is
evidence from this particular held-out period, not a promise about other
retailers or future dates.

### What to explain in an interview

"I created sales lags and shifted rolling averages because recent history is
useful for next-day demand. I made each feature use earlier dates only to avoid
showing the model the answer. LightGBM builds decision trees sequentially, with
later trees correcting earlier errors. I compared it with a same-weekday
baseline on identical future dates; RMSLE fell from 0.547 to 0.380, about 30.5%."

The original 1,000-tree ceiling was too low: validation kept improving when it
was reached. The ceiling is now 5,000 with a 100-round early-stopping patience;
the refreshed run selected 3,479 trees before that ceiling. We do not tune
against the final test period.

## 7. Explain model predictions with TreeSHAP

LightGBM's native TreeSHAP gives each input a contribution to a prediction in
log1p(sales) space. The contributions add to the model output, starting from a
baseline value. The project summarizes absolute contribution size over 5,000
reproducibly sampled validation examples and shows a local explanation for one
validation prediction.

SHAP explains how the fitted model combined its inputs for that example. It
does not show that changing a promotion or another feature would cause sales
to change.

### Interview check

"How did you explain a prediction?" Describe TreeSHAP: it breaks the model's
log-scale prediction into a baseline plus contributions from each input. Then
explain that this is a model explanation, not a causal claim.

## 8. Turn a forecast into a business scenario

The source data contains no current stock, supplier lead times, ordering costs or
stockout history. So the app cannot calculate a proven optimal order. It shows
an adjustable scenario instead:

`target stock = forecast demand over lead time + chosen safety buffer`

`suggested order = max(0, target stock - current stock)`

The safety buffer is a user-supplied percentage. For example, if the forecast
for the next three days totals 100 units, the buffer is 20%, and current stock
is 90, target stock is 120 and the scenario suggests ordering 30. This result
depends on the assumptions and is not a measured reduction in stockouts or cost.

### Interview check

"What does the replenishment recommendation optimize?" It does not optimize a
validated business objective. It calculates a transparent scenario from
forecast demand, user-entered stock, lead time and buffer. To optimize orders,
we would need costs, service-level targets, lead-time variability and inventory
records.

## 9. Serve the forecasts

The competition's separate test file contains 16 future dates but no actual
sales labels. We generate a 16-day recursive forecast: the first day uses
historical sales, then later days use earlier model predictions as their sales
history. This is different from the historical evaluation, where previous
actual sales are available each day.

The Streamlit dashboard shows a selected store-family's history and forecasts,
model evaluation metrics, SHAP summary and a replenishment scenario. FastAPI
serves the precomputed future forecast through `GET /forecast` and the scenario
calculation through `POST /replenishment`. Docker packages the API. To serve
new forecast dates, we must generate them from known promotion inputs and rerun
the pipeline.

The pipeline command is `python scripts/run_pipeline.py`. It runs the audit,
analysis, baseline, model, explanations and future forecast generation in
order. See the root README for starting Streamlit, FastAPI and Docker.
