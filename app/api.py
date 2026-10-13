"""Small read-only API for forecasts and replenishment scenarios."""
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from app.decisions import replenishment_scenario

ROOT = Path(__file__).resolve().parents[1]
FORECAST_FILE = ROOT / 'artifacts' / 'forecasts' / 'forecasts.csv'
app = FastAPI(title='Retail Demand Forecast API', version='1.0.0')


def forecasts():
    if not FORECAST_FILE.exists():
        raise HTTPException(503, 'Generate forecasts first: python scripts/run_pipeline.py')
    return pd.read_csv(FORECAST_FILE, parse_dates=['date'])


@app.get('/health')
def health():
    return {'status': 'ok', 'forecasts_ready': FORECAST_FILE.exists()}


@app.get('/forecast')
def get_forecast(family: str, store_nbr: int = Query(ge=1), days: int = Query(default=7, ge=1, le=16)):
    df = forecasts()
    rows = df[(df.store_nbr == store_nbr) & (df.family == family)].head(days)
    if rows.empty:
        raise HTTPException(404, 'No forecast found for that store and family.')
    return {'store_nbr': store_nbr, 'family': family,
            'forecasts': [{'date': row.date.date().isoformat(), 'sales': float(row.prediction)}
                          for row in rows.itertuples()]}


class ReplenishmentRequest(BaseModel):
    store_nbr: int = Field(ge=1)
    family: str
    current_stock: float = Field(ge=0)
    lead_days: int = Field(ge=1, le=16)
    safety_pct: float = Field(default=20.0, ge=0)


@app.post('/replenishment')
def replenish(request: ReplenishmentRequest):
    df = forecasts()
    rows = df[(df.store_nbr == request.store_nbr) & (df.family == request.family)].head(request.lead_days)
    if len(rows) < request.lead_days:
        raise HTTPException(404, 'Not enough forecast dates for that store and family.')
    result = replenishment_scenario(rows.prediction, request.current_stock,
                                    request.lead_days, request.safety_pct)
    return {'store_nbr': request.store_nbr, 'family': request.family, **result}
