"""Metric and cost calculations for solved schedules."""

from __future__ import annotations

import pandas as pd

from src.config import ModelConfig


def _sum_column_mwh(hourly_results: pd.DataFrame, column: str, config: ModelConfig) -> float:
    if column not in hourly_results.columns:
        return 0.0
    return float((hourly_results[column] * config.delta_t).sum())


def compute_cost_breakdown(hourly_results: pd.DataFrame, config: ModelConfig) -> dict[str, float]:
    """Compute the monetary objective components from hourly results.

    Returns separate renewable, grid, energy, peak, and total costs using the
    tariff parameters stored in `config`.
    """

    renewable_cost = (
        hourly_results["renewable_consumption"] * config.renewable_price * config.delta_t
    ).sum()
    grid_cost = (
        hourly_results["grid_consumption"] * hourly_results["grid_price"] * config.delta_t
    ).sum()
    peak_load = hourly_results["total_load"].max()
    peak_grid_import = hourly_results["grid_consumption"].max()
    peak_over_contracted = max(0.0, peak_grid_import - config.contracted_power)
    peak_cost = peak_over_contracted * config.peak_price

    return {
        "renewable_cost": float(renewable_cost),
        "grid_cost": float(grid_cost),
        "energy_cost": float(renewable_cost + grid_cost),
        "peak_load": float(peak_load),
        "peak_grid_import": float(peak_grid_import),
        "contracted_power": float(config.contracted_power),
        "peak_over_contracted": float(peak_over_contracted),
        "peak_cost": float(peak_cost),
        "total_cost": float(renewable_cost + grid_cost + peak_cost),
    }


def compute_summary_metrics(hourly_results: pd.DataFrame, config: ModelConfig) -> dict[str, float]:
    """Compute the full set of summary metrics for a solved schedule.

    The result extends the cost breakdown with total, renewable, and grid
    energy totals so notebooks and tests can assert both objective values and
    aggregate consumption levels.
    """

    cost_breakdown = compute_cost_breakdown(hourly_results, config)
    pue_series = hourly_results["pue"] if "pue" in hourly_results.columns else pd.Series([config.pue])
    return {
        **cost_breakdown,
        "pue": float(pue_series.mean()),
        "pue_min": float(pue_series.min()),
        "pue_max": float(pue_series.max()),
        "baseline_energy_mwh": _sum_column_mwh(hourly_results, "baseline_load", config),
        "flexible_energy_mwh": _sum_column_mwh(hourly_results, "flexible_load", config)
        if "flexible_load" in hourly_results.columns
        else _sum_column_mwh(hourly_results, "total_load", config),
        "it_energy_mwh": _sum_column_mwh(hourly_results, "it_load", config)
        if "it_load" in hourly_results.columns
        else _sum_column_mwh(hourly_results, "total_load", config),
        "total_energy_mwh": float((hourly_results["total_load"] * config.delta_t).sum()),
        "renewable_available_mwh": float((hourly_results["renewable_available"] * config.delta_t).sum()),
        "renewable_energy_mwh": float((hourly_results["renewable_consumption"] * config.delta_t).sum()),
        "renewable_curtailment_mwh": _sum_column_mwh(hourly_results, "renewable_curtailment", config),
        "grid_energy_mwh": float((hourly_results["grid_consumption"] * config.delta_t).sum()),
        "battery_charge_mwh": _sum_column_mwh(hourly_results, "battery_charge", config),
        "battery_discharge_mwh": _sum_column_mwh(hourly_results, "battery_discharge", config),
    }
