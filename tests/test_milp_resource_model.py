from dataclasses import replace

import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy")

from src.config import ModelConfig
from src.evaluation.results import extract_cluster_hourly_results, extract_hourly_results, extract_schedule
from src.milp.gurobi_model import build_milp_model
from src.milp.solve import solve_model


def _hourly(hours: int = 2, grid_price: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": list(range(hours)),
            "renewable_available": [0.0] * hours,
            "grid_price": grid_price or [100.0] * hours,
        }
    )


def _clusters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cluster_id": "cluster_t4",
                "capacity": 0.100,
                "compatible_categories": "training",
                "cluster_role": "low_or_inference_gpu_pool",
                "gpu_type": "T4",
                "gpu_count": 8,
                "gpu_capacity": 8,
                "cpu_capacity": 8,
                "memory_capacity_gb": 64,
            },
            {
                "cluster_id": "cluster_g2",
                "capacity": 0.100,
                "compatible_categories": "training",
                "cluster_role": "mainstream_gpu_pool",
                "gpu_type": "G2",
                "gpu_count": 8,
                "gpu_capacity": 8,
                "cpu_capacity": 16,
                "memory_capacity_gb": 128,
            },
        ]
    )


def _job(job_id: str, **overrides) -> dict:
    row = {
        "job_id": job_id,
        "category": "training",
        "workload_family": "training",
        "duration": 1,
        "power": 0.010,
        "earliest_start": 0,
        "latest_start": 1,
        "gpu_type_required": "T4",
        "gpu_count_required": 1,
        "cpu_required": 8,
        "memory_required_gb": 32,
    }
    row.update(overrides)
    return row


def _solve(model):
    model.Params.OutputFlag = 0
    try:
        solve_model(model)
    except gp.GurobiError as exc:
        pytest.skip(f"Gurobi is installed but not usable in this environment: {exc}")


def test_gpu_type_requirement_filters_assignment_variables():
    jobs_df = pd.DataFrame([_job("j1", gpu_type_required="G2")])
    model, variables = build_milp_model(jobs_df, _hourly(), _clusters(), ModelConfig())
    model.update()

    variable_clusters = {cluster for _, cluster, _ in variables["x"]}

    assert variable_clusters == {"cluster_g2"}


def test_pipe_separated_gpu_types_allow_multiple_partitions():
    jobs_df = pd.DataFrame([_job("j1", gpu_type_required="T4|G2")])
    model, variables = build_milp_model(jobs_df, _hourly(), _clusters(), ModelConfig())
    model.update()

    variable_clusters = {cluster for _, cluster, _ in variables["x"]}

    assert variable_clusters == {"cluster_t4", "cluster_g2"}


def test_cpu_capacity_can_force_two_jobs_to_different_hours():
    jobs_df = pd.DataFrame([_job("j1"), _job("j2")])
    clusters_df = _clusters().iloc[[0]].copy()
    config = replace(ModelConfig(), contracted_power=1.0)
    model, variables = build_milp_model(jobs_df, _hourly(), clusters_df, config)

    _solve(model)

    schedule = extract_schedule(jobs_df, variables)
    cluster_results = extract_cluster_hourly_results(variables)

    assert sorted(schedule["start_hour"].tolist()) == [0, 1]
    assert cluster_results["cluster_cpu_load"].max() <= 8


def test_pue_scales_it_load_into_facility_load():
    jobs_df = pd.DataFrame([_job("j1", earliest_start=0, latest_start=0)])
    config = replace(ModelConfig(), pue=1.5, contracted_power=1.0)
    model, variables = build_milp_model(jobs_df, _hourly(), _clusters().iloc[[0]], config)

    _solve(model)

    hourly_results = extract_hourly_results(_hourly(), variables)

    assert hourly_results.loc[0, "it_load"] == pytest.approx(0.010)
    assert hourly_results.loc[0, "total_load"] == pytest.approx(0.015)
    assert hourly_results.loc[0, "grid_consumption"] == pytest.approx(0.015)


def test_hourly_pue_overrides_constant_config_pue():
    jobs_df = pd.DataFrame([_job("j1", earliest_start=0, latest_start=1)])
    hourly_df = _hourly()
    hourly_df["pue"] = [2.0, 1.0]
    config = replace(ModelConfig(), pue=1.5, contracted_power=1.0)
    model, variables = build_milp_model(jobs_df, hourly_df, _clusters().iloc[[0]], config)

    _solve(model)

    hourly_results = extract_hourly_results(hourly_df, variables)

    assert set(hourly_results["pue"]) == {1.0, 2.0}
    scheduled_hour = int(hourly_results.loc[hourly_results["flexible_load"] > 0, "hour"].iloc[0])
    assert hourly_results.loc[scheduled_hour, "total_load"] == pytest.approx(
        hourly_results.loc[scheduled_hour, "it_load"] * hourly_results.loc[scheduled_hour, "pue"]
    )


def test_battery_can_shift_grid_energy_from_cheap_hour_to_expensive_hour():
    jobs_df = pd.DataFrame([_job("j1", earliest_start=1, latest_start=1)])
    config = replace(
        ModelConfig(),
        contracted_power=1.0,
        battery_power_capacity=0.010,
        battery_energy_capacity=0.010,
        battery_charge_efficiency=1.0,
        battery_discharge_efficiency=1.0,
    )
    model, variables = build_milp_model(jobs_df, _hourly(grid_price=[10.0, 1000.0]), _clusters().iloc[[0]], config)

    _solve(model)

    hourly_results = extract_hourly_results(_hourly(grid_price=[10.0, 1000.0]), variables)

    assert hourly_results.loc[0, "battery_charge"] == pytest.approx(0.010)
    assert hourly_results.loc[1, "battery_discharge"] == pytest.approx(0.010)
    assert hourly_results.loc[1, "grid_consumption"] == pytest.approx(0.0)


def test_resource_metadata_missing_capacity_fails_fast():
    jobs_df = pd.DataFrame([_job("j1")])
    clusters_df = pd.DataFrame(
        [
            {
                "cluster_id": "cluster_incomplete",
                "capacity": 0.100,
                "compatible_categories": "training",
                "cluster_role": "incomplete_gpu_pool",
                "gpu_type": "T4",
                "gpu_count": 8,
                "gpu_capacity": 8,
            }
        ]
    )

    with pytest.raises(ValueError, match="CPU capacity"):
        build_milp_model(jobs_df, _hourly(), clusters_df, ModelConfig())


def test_cpu_and_memory_constraints_can_be_disabled_for_ablation():
    jobs_df = pd.DataFrame([_job("j1")])
    clusters_df = pd.DataFrame(
        [
            {
                "cluster_id": "cluster_incomplete",
                "capacity": 0.100,
                "compatible_categories": "training",
                "cluster_role": "incomplete_gpu_pool",
                "gpu_type": "T4",
                "gpu_count": 8,
                "gpu_capacity": 8,
            }
        ]
    )

    model, variables = build_milp_model(
        jobs_df,
        _hourly(),
        clusters_df,
        ModelConfig(),
        enforce_cpu_constraints=False,
        enforce_memory_constraints=False,
    )
    model.update()

    assert variables["enforce_cpu_capacity"] is False
    assert variables["enforce_memory_capacity"] is False
    assert len(variables["x"]) > 0


def test_cpu_over_capacity_is_allowed_when_cpu_constraints_are_disabled():
    jobs_df = pd.DataFrame([_job("j1", cpu_required=32)])
    clusters_df = _clusters().iloc[[0]].copy()
    clusters_df["cpu_capacity"] = [8]

    model, variables = build_milp_model(
        jobs_df,
        _hourly(),
        clusters_df,
        ModelConfig(),
        enforce_cpu_constraints=False,
    )
    model.update()

    assert variables["enforce_cpu_capacity"] is False
    assert len(variables["x"]) > 0


def test_memory_over_capacity_is_allowed_when_memory_constraints_are_disabled():
    jobs_df = pd.DataFrame([_job("j1", memory_required_gb=256)])
    clusters_df = _clusters().iloc[[0]].copy()
    clusters_df["memory_capacity_gb"] = [64]

    model, variables = build_milp_model(
        jobs_df,
        _hourly(),
        clusters_df,
        ModelConfig(),
        enforce_memory_constraints=False,
    )
    model.update()

    assert variables["enforce_memory_capacity"] is False
    assert len(variables["x"]) > 0


def test_resource_profile_ignores_category_allowlist():
    jobs_df = pd.DataFrame([_job("j1", category="inference", workload_family="inference")])
    clusters_df = _clusters().iloc[[0]].copy()
    clusters_df["compatible_categories"] = ["training"]

    model, variables = build_milp_model(jobs_df, _hourly(), clusters_df, ModelConfig())
    model.update()

    assert len(variables["x"]) > 0


def test_inference_reservation_treats_missing_values_as_false():
    jobs_df = pd.DataFrame([_job("j1", workload_family="training", gpu_type_required="G2")])
    clusters_df = _clusters()
    clusters_df["reserved_for_online_inference"] = [None, None]

    model, variables = build_milp_model(jobs_df, _hourly(), clusters_df, ModelConfig())
    model.update()

    variable_clusters = {cluster for _, cluster, _ in variables["x"]}

    assert variable_clusters == {"cluster_g2"}


def test_inference_reservation_flag_is_ignored_by_resource_compatibility():
    jobs_df = pd.DataFrame(
        [
            _job(
                "j1",
                category="training",
                workload_family="training",
                gpu_type_required="T4",
            )
        ]
    )
    clusters_df = _clusters().iloc[[0]].copy()
    clusters_df["reserved_for_online_inference"] = [True]

    model, variables = build_milp_model(jobs_df, _hourly(), clusters_df, ModelConfig())
    model.update()

    assert len(variables["x"]) > 0


def test_model_config_rejects_invalid_pue_and_battery_values():
    with pytest.raises(ValueError, match="pue"):
        ModelConfig(pue=0)
    with pytest.raises(ValueError, match="battery_discharge_efficiency"):
        ModelConfig(battery_discharge_efficiency=0)
    with pytest.raises(ValueError, match="battery_charge_efficiency"):
        ModelConfig(battery_charge_efficiency=1.5)
