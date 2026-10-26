"""Shared evaluation helpers for reduced QUBO scheduling experiments.

The QUBO uses an effective-price approximation because it has no explicit
renewable/grid dispatch variables.  These helpers keep that approximation
separate from the physical post-evaluation applied to schedules from either
QUBO or MILP.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def make_effective_price_qubo_inputs(energy_df: pd.DataFrame) -> pd.DataFrame:
    """Convert an energy profile with ``effective_price`` into QUBO inputs.

    The builder's energy term treats a positive renewable availability as a
    binary signal.  To encode a continuous, instance-specific effective price
    faithfully, this function sets renewable availability to zero and uses the
    effective price as the grid-price coefficient.  This is an intentional
    reduced-model input, not a physical dispatch profile.
    """

    required = {"hour", "effective_price", "pue"}
    missing = required.difference(energy_df.columns)
    if missing:
        raise ValueError(f"energy profile missing QUBO price columns: {sorted(missing)}")
    return pd.DataFrame(
        {
            "hour": energy_df["hour"].astype(int),
            "renewable_available": 0.0,
            "grid_price": energy_df["effective_price"].astype(float),
            "pue": energy_df["pue"].astype(float),
            "baseline_load": 0.0,
        }
    )


def evaluate_physical_schedule(
    schedule_df: pd.DataFrame,
    energy_df: pd.DataFrame,
    *,
    delta_t: float,
    contracted_power_mw: float | None = None,
    peak_price: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    """Evaluate a fixed schedule under the physical renewable/grid model.

    Renewable generation is used first only when it is no more expensive than
    the grid.  The reported peak is the maximum grid import, matching the MILP
    contracted-power constraint.  No battery is modeled in this evaluator.
    """

    if delta_t <= 0:
        raise ValueError("delta_t must be positive")
    required = {"hour", "grid_price", "renewable_available", "pue"}
    missing = required.difference(energy_df.columns)
    if missing:
        raise ValueError(f"energy profile missing physical columns: {sorted(missing)}")

    rows: list[dict[str, float]] = []
    has_schedule = not schedule_df.empty
    for energy in energy_df.sort_values("hour").itertuples(index=False):
        hour = int(energy.hour)
        active = schedule_df.iloc[0:0]
        if has_schedule:
            active = schedule_df[
                (schedule_df["start_hour"] <= hour)
                & (hour < schedule_df["start_hour"] + schedule_df["duration"])
            ]
        it_load = float(active["power"].sum()) if not active.empty else 0.0
        facility_load = float(energy.pue) * it_load + float(getattr(energy, "baseline_load", 0.0))
        renewable_available = float(energy.renewable_available)
        renewable_price = float(getattr(energy, "renewable_price", energy.grid_price))
        grid_price = float(energy.grid_price)
        renewable_used = min(facility_load, renewable_available) if renewable_price <= grid_price else 0.0
        grid_used = max(facility_load - renewable_used, 0.0)
        rows.append(
            {
                "hour": hour,
                "it_load_mw": it_load,
                "facility_load_mw": facility_load,
                "renewable_available_mw": renewable_available,
                "renewable_used_mw": renewable_used,
                "grid_used_mw": grid_used,
                "grid_price": grid_price,
                "renewable_price": renewable_price,
                "energy_cost": delta_t * (renewable_used * renewable_price + grid_used * grid_price),
            }
        )

    profile = pd.DataFrame(rows)
    peak_grid_import = float(profile["grid_used_mw"].max()) if not profile.empty else 0.0
    peak_excess = max(0.0, peak_grid_import - float(contracted_power_mw or 0.0))
    energy_cost = float(profile["energy_cost"].sum())
    peak_cost = float(peak_price) * peak_excess
    return profile, {
        "physical_energy_cost": energy_cost,
        "physical_peak_grid_import_mw": peak_grid_import,
        "physical_peak_excess_mw": peak_excess,
        "physical_peak_cost": peak_cost,
        "physical_total_cost": energy_cost + peak_cost,
        "facility_energy_mwh": float((profile["facility_load_mw"] * delta_t).sum()),
        "renewable_energy_mwh": float((profile["renewable_used_mw"] * delta_t).sum()),
        "grid_energy_mwh": float((profile["grid_used_mw"] * delta_t).sum()),
    }
