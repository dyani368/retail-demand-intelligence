"""Transparent replenishment scenario calculations."""
import math


def replenishment_scenario(daily_forecasts, current_stock, lead_days, safety_pct=20.0):
    """Estimate target stock as forecast lead-time demand plus a buffer.

    This is a user-configured scenario, not an optimized inventory policy.
    """
    daily_forecasts = list(daily_forecasts)
    if not 1 <= lead_days <= len(daily_forecasts):
        raise ValueError(f'lead_days must be from 1 to {len(daily_forecasts)}')
    if current_stock < 0 or safety_pct < 0:
        raise ValueError('stock and safety percentage must be non-negative')
    demand = sum(max(0.0, float(x)) for x in daily_forecasts[:lead_days])
    buffer = demand * safety_pct / 100.0
    target = demand + buffer
    order = max(0.0, target - current_stock)
    return {'forecast_lead_time_demand': demand,
            'assumed_safety_buffer': buffer,
            'target_stock': target,
            'suggested_order_quantity': math.ceil(order),
            'scenario_note': 'Assumes the supplied forecasts and safety percentage; no real inventory, lead-time, cost, or service-level data was provided.'}
