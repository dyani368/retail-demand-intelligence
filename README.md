# Retail Demand Intelligence & Replenishment Engine

An end-to-end forecasting project built with the Kaggle Store Sales dataset. It predicts daily sales for store/product-family pairs, compares LightGBM with a weekly baseline, explains model behavior, and turns forecasts into a transparent replenishment scenario.

## Current measured result

On the final 28 labeled dates (**2017-07-19 to 2017-08-15**), LightGBM achieved **0.380 RMSLE and 12.83% WAPE**, compared with **0.547 RMSLE and 18.51% WAPE** for the same-weekday-last-week baseline. That is a **30.5% RMSLE reduction** on this holdout. This is a rolling one-day-ahead score and describes this dataset and period.

That result is a rolling **one-day-ahead** evaluation. A 16-day recursive backtest is implemented separately, but its refreshed run was interrupted before all folds completed; no current multi-day score is reported. See [recursive backtest status](docs/recursive_backtest_results.md).

## Setup

Download the files from [Kaggle Store Sales – Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), then extract them into `store-sales-time-series-forecasting/`. Keep the original dataset out of Git. Python 3.11+ is recommended.

```powershell
python -m pip install -r requirements.txt
python scripts/run_pipeline.py
```

The pipeline audits the CSVs, creates descriptive summaries, evaluates the seasonal baseline, trains/evaluates LightGBM, generates TreeSHAP explanations, and makes recursive forecasts for the competition's unlabeled dates. It writes outputs under `artifacts/`.

## Run the app

```powershell
streamlit run app/dashboard.py
```

The dashboard shows the historical one-day-ahead evaluation, forecasts, and global and local TreeSHAP views. You can select a store and product family, inspect future forecasts, and adjust stock, lead time and safety buffer. The dataset has no inventory, purchase-order or cost records, so the replenishment section is an assumption-based scenario.

## Run the API

```powershell
uvicorn app.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API page. `GET /forecast` returns future predictions. `POST /replenishment` calculates a scenario order quantity from the forecast and supplied assumptions.

## Reproduce with Docker

Generate artifacts first, then:

```powershell
docker compose up --build
```

The API is available at `http://127.0.0.1:8000/docs`. The compose file mounts only the generated forecast CSV into the container.

Local deployment was verified on October 13, 2026: the Docker image built and the container started successfully; `/health`, `/forecast`, and `/replenishment` responded correctly. The running Streamlit dashboard was checked across all four views, including family selection, stock changes, and SHAP rendering. To stop the Docker API, run `docker compose down`.

## Project flow

1. **Data:** inspect row meaning, missing dates, duplicate keys and joins.
2. **Analysis:** identify sales patterns by date, family and weekday.
3. **Features:** derive lagged history, rolling averages and known calendar/promotion values.
4. **Model:** compare a same-weekday baseline with LightGBM.
5. **Validation:** compare the model with a weekly seasonal baseline on later dates, using a chronological holdout.
6. **Explainability:** use LightGBM's native TreeSHAP contributions on held-out validation examples.
7. **Business decision:** calculate a configurable replenishment scenario.
8. **Deployment:** expose forecasts in Streamlit and FastAPI; package the API with Docker.

