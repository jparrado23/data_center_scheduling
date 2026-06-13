"""Metric and cost calculations for solved schedules."""

from __future__ import annotations

import pandas as pd

from src.config import ModelConfig


def compute_cost_breakdown(hourly_results: pd.DataFrame, config: ModelConfig) -> dict[str, float]:
    """Compute the monetary objective components from hourly results.

    Returns separate renewable, grid, energy, peak, and total costs using the
    tariff parameters stored in `config`.
    """

    renewable_contract_cost = (
        hourly_results["renewable_available"] * config.renewable_price * config.delta_t
    ).sum()
    grid_cost = (
        hourly_results["grid_consumption"] * hourly_results["grid_price"] * config.delta_t
    ).sum()
    peak_load = hourly_results["total_load"].max()
    peak_over_contracted = max(0.0, peak_load - config.contracted_power)
    peak_cost = peak_load * config.peak_price

    return {
        "renewable_contract_cost": float(renewable_contract_cost),
        "renewable_cost": float(renewable_contract_cost),
        "grid_cost": float(grid_cost),
        "energy_cost": float(renewable_contract_cost + grid_cost),
        "peak_load": float(peak_load),
        "contracted_power": float(config.contracted_power),
        "peak_over_contracted": float(peak_over_contracted),
        "peak_cost": float(peak_cost),
        "total_cost": float(renewable_contract_cost + grid_cost + peak_cost),
    }


def compute_summary_metrics(hourly_results: pd.DataFrame, config: ModelConfig) -> dict[str, float]:
    """Compute the full set of summary metrics for a solved schedule.

    The result extends the cost breakdown with total, renewable, and grid
    energy totals so notebooks and tests can assert both objective values and
    aggregate consumption levels.
    """

    cost_breakdown = compute_cost_breakdown(hourly_results, config)
    return {
        **cost_breakdown,
        "baseline_energy_mwh": float((hourly_results.get("baseline_load", 0.0) * config.delta_t).sum()),
        "flexible_energy_mwh": float((hourly_results.get("flexible_load", hourly_results["total_load"]) * config.delta_t).sum()),
        "total_energy_mwh": float((hourly_results["total_load"] * config.delta_t).sum()),
        "renewable_available_mwh": float((hourly_results["renewable_available"] * config.delta_t).sum()),
        "renewable_energy_mwh": float((hourly_results["renewable_consumption"] * config.delta_t).sum()),
        "renewable_curtailment_mwh": float((hourly_results.get("renewable_curtailment", 0.0) * config.delta_t).sum()),
        "grid_energy_mwh": float((hourly_results["grid_consumption"] * config.delta_t).sum()),
    }
