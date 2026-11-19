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


# ---------------------------------------------------------------------------
# Hard-frontier extensions: default-behaviour regression and new knobs.
# ---------------------------------------------------------------------------

import hashlib
from dataclasses import replace

import pytest

from src.instance_generator.price_shapes import build_energy_for_day
from src.instance_generator.single_cluster_stress import (
    generate_single_cluster_stress_instance,
    jobs_for_target_utilization,
)

# SHA-256 of ``DataFrame.to_csv(index=False)`` for long_wide_window seed 7602,
# captured from the generator before the frontier-search knobs were added.
REFERENCE_7602_JOBS_SHA256 = "f501fc5f5a8dbf7202d44681f3d9d20085cc5dc9c56851b1ffe951ca69af7d7f"
REFERENCE_7602_HIDDEN_SHA256 = "f5201d92a9281e2a7a448be1cf93979567edb72a4d7736dfd01c6b62290de3cb"


def _long_wide_window_config() -> SingleClusterStressConfig:
    """Return the ``long_wide_window`` profile from the focused sweep script."""

    return SingleClusterStressConfig(
        num_jobs=42,
        target_utilization=0.82,
        utilization_tolerance=0.06,
        node_demand_values=(16, 32, 64, 96),
        node_demand_probabilities=(0.15, 0.35, 0.35, 0.15),
        duration_values=(4, 8, 12, 16),
        duration_probabilities=(0.15, 0.35, 0.35, 0.15),
        min_start_options=12,
        max_start_options=20,
        min_price_spread_ratio=1.15,
        min_peak_potential_load_ratio=1.50,
        max_attempts=100,
    )


def _sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()


@pytest.fixture(scope="module")
def base_energy() -> pd.DataFrame:
    return build_energy_for_day("base")


def _generate(config: SingleClusterStressConfig, energy: pd.DataFrame, seed: int = 7602):
    return generate_single_cluster_stress_instance(energy, config, seed, f"test_seed_{seed}")


def _assert_certificate_feasible(jobs: pd.DataFrame, hidden: pd.DataFrame, capacity: int) -> None:
    merged = jobs.merge(hidden[["job_id", "hidden_start"]], on="job_id", validate="one_to_one")
    assert (merged["hidden_start"] >= merged["earliest_start"]).all()
    assert (merged["hidden_start"] <= merged["latest_start"]).all()
    load = np.zeros(96, dtype=int)
    for job in merged.itertuples(index=False):
        load[job.hidden_start:job.hidden_start + job.duration] += job.node_units_required
    assert load.max() <= capacity


def test_default_generation_is_byte_identical_to_reference(base_energy: pd.DataFrame) -> None:
    """New knobs at their defaults must reproduce long_wide_window seed 7602."""

    jobs, _, _, hidden, _, _ = generate_single_cluster_stress_instance(
        base_energy, _long_wide_window_config(), 7602, "long_wide_window_seed_7602"
    )

    assert _sha256(jobs) == REFERENCE_7602_JOBS_SHA256
    assert _sha256(hidden) == REFERENCE_7602_HIDDEN_SHA256


def test_power_intensity_stays_within_range_and_keeps_node_demand(base_energy: pd.DataFrame) -> None:
    config = replace(_long_wide_window_config(), power_intensity_range=(0.5, 1.0))
    jobs, _, _, _, _, metadata = _generate(config, base_energy)

    nominal = jobs["node_units_required"] * config.power_per_node_mw
    ratio = jobs["power"] / nominal
    assert ratio.between(0.5, 1.0).all()
    assert ratio.std() > 0
    assert np.allclose(jobs["power_kw"], jobs["power"] * 1000.0)
    assert jobs["node_units_required"].equals(jobs["gpu_count_required"])
    assert np.allclose(ratio, metadata["power_intensity"])


def test_unsteered_generation_yields_feasible_certificate(base_energy: pd.DataFrame) -> None:
    config = replace(_long_wide_window_config(), steer_to_cheap_slot=False, min_peak_potential_load_ratio=0.0)
    jobs, clusters, _, hidden, _, metadata = _generate(config, base_energy)

    _assert_certificate_feasible(jobs, hidden, int(clusters["node_capacity"].iloc[0]))
    assert metadata["targeted_cheap_potential_nodes"] == 0


def test_random_feasible_planting_respects_capacity(base_energy: pd.DataFrame) -> None:
    config = replace(_long_wide_window_config(), planting="random_feasible", max_attempts=300)
    jobs, clusters, _, hidden, _, _ = _generate(config, base_energy)

    _assert_certificate_feasible(jobs, hidden, int(clusters["node_capacity"].iloc[0]))


def test_invalid_frontier_knobs_are_rejected() -> None:
    with pytest.raises(ValueError):
        SingleClusterStressConfig(planting="unknown")
    with pytest.raises(ValueError):
        SingleClusterStressConfig(power_intensity_range=(1.0, 0.5))
    with pytest.raises(ValueError):
        SingleClusterStressConfig(min_peak_potential_load_ratio=-0.1)


def test_jobs_for_target_utilization_matches_expected_load() -> None:
    config = _long_wide_window_config()
    # E[d] = 50.4, E[p] = 10 -> 0.82 * 256 * 96 / 504 = 39.98
    assert jobs_for_target_utilization(config, 0.82) == 40
