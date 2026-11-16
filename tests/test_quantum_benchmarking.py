"""Tests for shared physical-vs-QUBO benchmark helpers."""

import pandas as pd
import pytest

from src.quantum.benchmarking import (
    add_effective_price_for_qubo,
    evaluate_physical_schedule,
    make_effective_price_qubo_inputs,
)


def _energy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": [0, 1],
            "grid_price": [100.0, 200.0],
            "renewable_price": [50.0, 50.0],
            "renewable_available": [0.10, 0.0],
            "pue": [1.25, 1.25],
            "baseline_load": [0.0, 0.0],
            "effective_price": [60.0, 200.0],
        }
    )


def test_make_effective_price_qubo_inputs_uses_effective_price_directly():
    inputs = make_effective_price_qubo_inputs(_energy())

    assert inputs["grid_price"].tolist() == [60.0, 200.0]
    assert inputs["renewable_available"].tolist() == [0.0, 0.0]


def test_effective_price_depends_on_instance_load_scale():
    jobs = pd.DataFrame([{"power": 0.10, "duration": 1}])
    clusters = pd.DataFrame([{"capacity": 1.0}])

    priced = add_effective_price_for_qubo(_energy(), jobs, clusters, renewable_price=50.0)

    assert priced["expected_load_scale_mw"].iloc[0] == pytest.approx(0.125)
    assert priced["renewable_share"].iloc[0] == pytest.approx(0.8)
    assert priced["effective_price"].iloc[0] == pytest.approx(60.0)


def test_evaluate_physical_schedule_uses_renewable_then_grid_and_grid_peak():
    schedule = pd.DataFrame(
        [{"job_id": "j1", "start_hour": 0, "duration": 2, "power": 0.10}]
    )

    profile, summary = evaluate_physical_schedule(
        schedule, _energy(), delta_t=1.0, contracted_power_mw=0.10, peak_price=1_000.0
    )

    assert profile["facility_load_mw"].tolist() == [0.125, 0.125]
    assert profile["renewable_used_mw"].tolist() == [0.10, 0.0]
    assert profile["grid_used_mw"].tolist() == pytest.approx([0.025, 0.125])
    assert summary["physical_energy_cost"] == pytest.approx(32.5)
    assert summary["physical_peak_grid_import_mw"] == pytest.approx(0.125)
    assert summary["physical_peak_cost"] == pytest.approx(25.0)
