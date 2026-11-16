"""Tests for the one-cluster synthetic stress-instance generator."""

import numpy as np
import pandas as pd

from src.instance_generator.single_cluster_stress import (
    SingleClusterStressConfig,
    expand_hourly_energy_to_slots,
    potential_active_load,
)


def _hourly_energy() -> pd.DataFrame:
    """Return a price and solar profile with a pronounced cheap midday period."""

    hours = np.arange(24)
    # A central low-price hour and alternating prices give every window contrast.
    grid_price = np.where(hours % 2 == 0, 80.0, 160.0)
    grid_price[10] = 40.0
    renewable = np.zeros_like(hours, dtype=float)
    return pd.DataFrame(
        {
            "hour": hours,
            "grid_price": grid_price,
            "renewable_available": renewable,
            "renewable_price": 40.0,
            "pue": 1.25,
            "baseline_load": 0.0,
        }
    )


def test_expand_hourly_energy_repeats_power_inputs_across_slots() -> None:
    """Sub-hour expansion preserves hourly MW availability and price values."""

    expanded = expand_hourly_energy_to_slots(_hourly_energy().head(2), 15)

    assert expanded["hour"].tolist() == list(range(8))
    assert expanded.loc[:3, "renewable_available"].tolist() == [0.0] * 4
    assert expanded.loc[4:, "grid_price"].tolist() == [160.0] * 4


def test_potential_active_load_counts_each_job_when_any_start_can_cover_slot() -> None:
    """Potential load includes a job once even when it has several valid starts."""

    jobs = pd.DataFrame(
        {
            "node_units_required": [16, 32],
            "duration": [2, 1],
            "earliest_start": [0, 2],
            "latest_start": [1, 2],
        }
    )

    assert potential_active_load(jobs, horizon_slots=4).tolist() == [16, 16, 48, 0]
