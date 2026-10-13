"""Interactive Streamlit dashboard for forecasts, validation, and explanations."""
from pathlib import Path
import json
import math
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import altair as alt
import pandas as pd
import streamlit as st

from app.decisions import replenishment_scenario

ROOT = PROJECT_ROOT
DATA_DIR = ROOT / 'store-sales-time-series-forecasting'
ARTIFACTS = ROOT / 'artifacts'
FORECASTS = ARTIFACTS / 'forecasts' / 'forecasts.csv'
FORECAST_METADATA = ARTIFACTS / 'forecasts' / 'metadata.json'
METRICS = ARTIFACTS / 'tree_model' / 'metrics.json'
SHAP_SUMMARY = ARTIFACTS / 'explainability' / 'shap_summary.csv'
SHAP_EXAMPLES = ARTIFACTS / 'explainability' / 'shap_examples.csv'
LOCAL_SHAP = ARTIFACTS / 'explainability' / 'local_explanation.json'
EXPLAINABILITY_METADATA = ARTIFACTS / 'explainability' / 'metadata.json'
FEATURE_LABELS = {
    'sales_mean_prev_7_days': 'Previous 7-day sales average',
    'sales_mean_prev_28_days': 'Previous 28-day sales average',
    'sales_lag_1': "Yesterday's sales", 'sales_lag_7': 'Sales 7 days earlier',
    'sales_lag_14': 'Sales 14 days earlier', 'sales_lag_28': 'Sales 28 days earlier',
    'onpromotion': 'Promotion count', 'day_of_week': 'Day of week',
    'month': 'Month', 'store_nbr': 'Store', 'family_code': 'Product family',
}

st.set_page_config(page_title='Retail demand intelligence', page_icon=':material/monitoring:', layout='wide')
st.title('Retail demand intelligence', icon=':material/monitoring:')
st.caption('Store-family demand forecasts, chronological model checks, and transparent replenishment scenarios.')


@st.cache_data
def load_history():
    return pd.read_csv(DATA_DIR / 'train.csv', parse_dates=['date'],
                       usecols=['date', 'store_nbr', 'family', 'sales'])


@st.cache_data
def load_forecasts():
    return pd.read_csv(FORECASTS, parse_dates=['date'])


@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


required = {'forecasts': FORECASTS, 'forecast metadata': FORECAST_METADATA, 'metrics': METRICS}
missing = [label for label, path in required.items() if not path.exists()]
if missing:
    st.error(f"Missing project output: {', '.join(missing)}. Run `python scripts/run_pipeline.py` first.")
    st.stop()

history = load_history()
future = load_forecasts()
metrics = json.loads(METRICS.read_text(encoding='utf-8'))
forecast_meta = json.loads(FORECAST_METADATA.read_text(encoding='utf-8'))
if forecast_meta.get('features') != metrics.get('features'):
    st.error('Forecasts were generated with an older feature definition. Run `python scripts/generate_forecasts.py` to refresh them.')
    st.stop()
overview_tab, forecast_tab, validation_tab, explanation_tab = st.tabs(
    ['Overview', 'Forecast & planning', 'Model validation', 'Explainability'])

with overview_tab:
    test = metrics['test']
    baseline_rmsle = test['seasonal_baseline']['rmsle']
    model_rmsle = test['lightgbm']['rmsle']
    reduction = 100 * (baseline_rmsle - model_rmsle) / baseline_rmsle
    st.subheader('Historical one-day-ahead holdout')
    st.caption('Daily forecasts evaluated on a chronological holdout against a weekly seasonal baseline.')
    with st.container(horizontal=True):
        st.metric('LightGBM RMSLE', f'{model_rmsle:.3f}', border=True)
        st.metric('LightGBM WAPE', f"{test['lightgbm']['wape_pct']:.2f}%", border=True)
        st.metric('Weekly baseline RMSLE', f'{baseline_rmsle:.3f}', border=True)
        st.metric('RMSLE reduction', f'{reduction:.1f}%', 'vs. weekly baseline', border=True)
    st.write(f"Holdout: **{metrics['test_dates'][0]} to {metrics['test_dates'][1]}** · {len(future):,} future competition rows are shown in the planning view.")

with forecast_tab:
    stores = sorted(future.store_nbr.unique())
    families = sorted(future.family.unique())
    with st.container(horizontal=True):
        store = st.selectbox('Store', stores)
        family = st.selectbox('Product family', families)
    hist_series = history[(history.store_nbr == store) & (history.family == family)].tail(90)
    forecast_series = future[(future.store_nbr == store) & (future.family == family)].sort_values('date').copy()
    history_plot = hist_series[['date', 'sales']].rename(columns={'sales': 'Units'}).assign(Series='Observed history')
    future_plot = forecast_series[['date', 'prediction']].rename(columns={'prediction': 'Units'}).assign(Series='Forecast')
    chart_data = pd.concat([history_plot, future_plot], ignore_index=True)
    st.subheader(f'Store {store} · {family}')
    st.line_chart(chart_data, x='date', y='Units', color='Series', alt='Observed sales followed by recursively forecast daily sales')
    st.caption(f"Forecast dates: {forecast_series.date.min().date()} to {forecast_series.date.max().date()}. From day 2 onward, prior model predictions become sales-history inputs.")

    st.subheader('Replenishment scenario')
    st.caption('Change the assumptions to see how an example order quantity responds.')
    with st.container(horizontal=True):
        stock = st.number_input('Current stock (assumed units)', min_value=0.0, value=500.0, step=50.0)
        lead = st.slider('Lead time (days)', min_value=1, max_value=len(forecast_series), value=min(7, len(forecast_series)))
        safety = st.slider('Safety buffer (%)', min_value=0, max_value=100, value=20)
    scenario = replenishment_scenario(forecast_series.prediction, stock, lead, safety)
    with st.container(horizontal=True):
        st.metric('Forecast demand over lead time', f"{scenario['forecast_lead_time_demand']:,.1f}", border=True)
        st.metric('Target stock with buffer', f"{scenario['target_stock']:,.1f}", border=True)
        st.metric('Scenario order quantity', f"{scenario['suggested_order_quantity']:,}", border=True)
    with st.expander('Replenishment assumptions'):
        st.write('Target stock combines forecast demand over the lead time with the selected safety buffer. The order quantity subtracts current stock and rounds upward to whole units.')
        st.caption('Stock, lead time and safety buffer are scenario inputs. Inventory balances, supplier history, costs and service-level targets are not supplied by this dataset.')
    display_forecast = forecast_series[['date', 'onpromotion', 'prediction']].rename(
        columns={'date': 'Date', 'onpromotion': 'Promotion count', 'prediction': 'Forecast sales'})
    st.dataframe(display_forecast, hide_index=True, alt='Daily forecasts and known promotion counts for the selected store and family')

with validation_tab:
    st.subheader('Chronological model evaluation')
    st.write('Each historical date is predicted one day ahead. Actual sales from earlier dates are available for later dates in this evaluation.')
    cols = st.columns(2)
    for col, label, key in [(cols[0], 'Rolling one-day-ahead holdout', 'test'),
                            (cols[1], 'Rolling validation period', 'validation')]:
        result = metrics[key]
        with col.container(border=True):
            st.markdown(f'**{label}**')
            st.write(f"Dates: {metrics['test_dates'][0] if key == 'test' else metrics['validation_dates'][0]} to {metrics['test_dates'][1] if key == 'test' else metrics['validation_dates'][1]}")
            comparison = pd.DataFrame([
                {'Model': 'LightGBM', **result['lightgbm']},
                {'Model': 'Same-weekday seasonal baseline', **result['seasonal_baseline']},
            ])
            st.dataframe(comparison, hide_index=True, alt=f'{label} LightGBM and seasonal baseline metrics')

    with st.expander('Evaluation methodology'):
        st.write('The historical scores measure one-day-ahead forecasts. Each forecast uses sales observed before its target date. Model choices use the validation period, and the later holdout reports performance against the weekly baseline.')
        st.write('The planning view uses a 16-day recursive forecast: the first day uses observed history and later days use earlier predictions. The future competition file has no actual sales labels, so no accuracy score is reported for those dates.')

with explanation_tab:
    st.subheader('Global TreeSHAP summary')
    st.caption('Average feature contributions across 5,000 sampled validation predictions.')
    explanation_is_current = False
    if EXPLAINABILITY_METADATA.exists():
        explain_meta = json.loads(EXPLAINABILITY_METADATA.read_text(encoding='utf-8'))
        explanation_is_current = explain_meta.get('features') == metrics.get('features')
    if not explanation_is_current:
        st.warning('The saved SHAP files were generated for an earlier feature definition. They are hidden until explanations are regenerated with `python scripts/explain_model.py`.')
    if explanation_is_current and SHAP_SUMMARY.exists():
        summary = load_csv(SHAP_SUMMARY).sort_values('mean_absolute_shap_log_sales', ascending=True).tail(11)
        summary['Feature'] = summary.feature.map(FEATURE_LABELS)
        global_chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('mean_absolute_shap_log_sales:Q', title='Mean absolute TreeSHAP contribution (log1p sales)', axis=alt.Axis(tickCount=6)),
            y=alt.Y('Feature:N', sort='-x', title=None, axis=alt.Axis(labelLimit=240)),
            tooltip=['Feature', alt.Tooltip('mean_absolute_shap_log_sales:Q', format='.3f')],
        )
        st.altair_chart(global_chart, alt='Features ranked by their average absolute TreeSHAP contribution')
    if explanation_is_current and LOCAL_SHAP.exists():
        local = json.loads(LOCAL_SHAP.read_text(encoding='utf-8'))
        if SHAP_EXAMPLES.exists():
            examples = load_csv(SHAP_EXAMPLES)
            examples['date'] = pd.to_datetime(examples['date'])
            matching = examples[(examples.store_nbr == store) & (examples.family == family)]
            if not matching.empty:
                date_options = sorted(matching.date.dt.strftime('%Y-%m-%d').unique().tolist())
                selected_date = st.selectbox('Validation date for local explanation', date_options)
                selected = matching[matching.date.dt.strftime('%Y-%m-%d') == selected_date].iloc[0]
                local = {
                    'date': selected.date.date().isoformat(), 'store_nbr': int(selected.store_nbr),
                    'family': selected.family, 'actual_sales': float(selected.sales),
                    'predicted_sales': float(selected.predicted_sales),
                    'base_value': float(selected.base_value_log1p),
                    'contributions_on_log1p_scale': {
                        feature: float(selected[f'shap__{feature}']) for feature in metrics['features']
                    },
                }
        st.subheader('One prediction, explained')
        st.write(f"Validation example: **{local['date']}**, store **{local['store_nbr']}**, family **{local['family']}** · actual **{local['actual_sales']:,.1f}** · predicted **{local['predicted_sales']:,.1f}** sales units.")
        contributions = pd.DataFrame([
            {'Feature': FEATURE_LABELS.get(name, name), 'Contribution': value}
            for name, value in local['contributions_on_log1p_scale'].items()
        ]).sort_values('Contribution')
        local_chart = alt.Chart(contributions).mark_bar().encode(
            x=alt.X('Contribution:Q', title='TreeSHAP contribution (log1p sales)'),
            y=alt.Y('Feature:N', sort='-x', title=None, axis=alt.Axis(labelLimit=240)),
            color=alt.condition(alt.datum.Contribution >= 0, alt.value('#168C78'), alt.value('#C45B54')),
            tooltip=['Feature', alt.Tooltip('Contribution:Q', format='.3f')],
        )
        st.altair_chart(local_chart, alt='Local TreeSHAP contributions for one held-out LightGBM prediction')
        reconstructed = local['base_value'] + sum(local['contributions_on_log1p_scale'].values())
        with st.expander('SHAP methodology'):
            st.write('LightGBM native TreeSHAP breaks a prediction into a baseline and individual feature contributions on the log1p(sales) scale. These contributions explain model behavior and do not establish causal effects.')
            st.caption(f"Base value + contributions = {reconstructed:.4f}; model prediction on the same scale = {math.log1p(local['predicted_sales']):.4f}.")
