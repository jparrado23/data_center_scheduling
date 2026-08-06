import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.quantum.qubo_builder import (
    QuboPenaltyWeights,
    build_scheduling_qubo,
    decode_qubo_sample,
    qubo_energy,
    validate_decoded_schedule,
)


def _jobs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "job_id": "j1",
                "category": "gpu",
                "duration": 1,
                "power": 0.01,
                "earliest_start": 0,
                "latest_start": 1,
                "gpu_type_required": "T4",
                "gpu_count_required": 1,
                "cpu_required": 2.0,
                "memory_required_gb": 8.0,
            },
            {
                "job_id": "j2",
                "category": "gpu",
                "duration": 1,
                "power": 0.01,
                "earliest_start": 0,
                "latest_start": 1,
                "gpu_type_required": "T4",
                "gpu_count_required": 1,
                "cpu_required": 2.0,
                "memory_required_gb": 8.0,
            },
        ]
    )


def _hourly() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hour": [0, 1],
            "renewable_available": [0.02, 0.0],
            "grid_price": [20.0, 100.0],
        }
    )


def _clusters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cluster_id": "cluster_t4",
                "capacity": 0.02,
                "gpu_type": "T4",
                "gpu_count": 2,
                "gpu_capacity": 2,
                "cpu_capacity": 4.0,
                "memory_capacity_gb": 16.0,
            }
        ]
    )


def _two_clusters() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cluster_id": "cluster_t4",
                "capacity": 0.02,
                "gpu_type": "T4",
                "gpu_count": 2,
                "gpu_capacity": 2,
                "cpu_capacity": 4.0,
                "memory_capacity_gb": 16.0,
            },
            {
                "cluster_id": "cluster_v100",
                "capacity": 0.02,
                "gpu_type": "V100",
                "gpu_count": 2,
                "gpu_capacity": 2,
                "cpu_capacity": 4.0,
                "memory_capacity_gb": 16.0,
            },
        ]
    )


def test_build_scheduling_qubo_creates_assignment_variables():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))

    assert qubo.metadata["num_assignment_variables"] == 4
    assert qubo.metadata["num_slack_variables"] == 4
    assert qubo.num_variables == 8
    assert len(qubo.linear) == 8
    assert qubo.metadata["num_quadratic_terms"] > 0


def test_explicit_compatible_clusters_restrict_assignment_variables():
    jobs = _jobs().copy()
    jobs["gpu_type_required"] = ""
    jobs["compatible_clusters"] = [["cluster_t4"], ["cluster_v100"]]

    qubo = build_scheduling_qubo(jobs, _hourly(), _two_clusters(), ModelConfig(delta_t=1.0))
    assignment_variables = [var for var in qubo.variables if var.get("variable_type") == "assignment"]

    assert qubo.metadata["num_assignment_variables"] == 4
    assert {var["cluster"] for var in assignment_variables if var["job_id"] == "j1"} == {"cluster_t4"}
    assert {var["cluster"] for var in assignment_variables if var["job_id"] == "j2"} == {"cluster_v100"}


def test_explicit_compatible_clusters_override_stale_gpu_type_metadata():
    jobs = _jobs().copy()
    jobs["gpu_type_required"] = ["V100", "T4"]
    jobs["compatible_clusters"] = [["cluster_t4"], ["cluster_v100"]]

    qubo = build_scheduling_qubo(jobs, _hourly(), _two_clusters(), ModelConfig(delta_t=1.0))
    assignment_variables = [var for var in qubo.variables if var.get("variable_type") == "assignment"]

    assert qubo.metadata["num_assignment_variables"] == 4
    assert {var["cluster"] for var in assignment_variables if var["job_id"] == "j1"} == {"cluster_t4"}
    assert {var["cluster"] for var in assignment_variables if var["job_id"] == "j2"} == {"cluster_v100"}


def test_notebook_tiny_fixed_pue_qubo_structure_matches_builder():
    jobs = pd.DataFrame(
        [
            {
                "job_id": "j1",
                "category": "gpu",
                "duration": 1,
                "power": 0.010,
                "earliest_start": 0,
                "latest_start": 1,
                "gpu_count_required": 1,
                "cpu_required": 0.0,
                "memory_required_gb": 0.0,
                "compatible_clusters": ["c_t4"],
            },
            {
                "job_id": "j2",
                "category": "gpu",
                "duration": 1,
                "power": 0.015,
                "earliest_start": 0,
                "latest_start": 2,
                "gpu_count_required": 1,
                "cpu_required": 0.0,
                "memory_required_gb": 0.0,
                "compatible_clusters": ["c_t4"],
            },
            {
                "job_id": "j3",
                "category": "gpu",
                "duration": 1,
                "power": 0.008,
                "earliest_start": 0,
                "latest_start": 2,
                "gpu_count_required": 1,
                "cpu_required": 0.0,
                "memory_required_gb": 0.0,
                "compatible_clusters": ["c_v100"],
            },
        ]
    )
    clusters = pd.DataFrame(
        [
            {"cluster_id": "c_t4", "capacity": 1.0, "gpu_capacity": 1},
            {"cluster_id": "c_v100", "capacity": 1.0, "gpu_capacity": 1},
        ]
    )
    hourly = pd.DataFrame(
        {
            "hour": [0, 1, 2],
            "renewable_available": [0.0, 0.0, 0.0],
            "grid_price": [30.0, 100.0, 40.0],
            "pue": [1.20, 1.20, 1.20],
        }
    )

    qubo = build_scheduling_qubo(
        jobs,
        hourly,
        clusters,
        ModelConfig(delta_t=1.0, renewable_price=10_000.0, pue=1.20),
        QuboPenaltyWeights(assignment=200.0, gpu_capacity=80.0, peak_smoothing=5000.0),
    )

    assert qubo.metadata["num_assignment_variables"] == 8
    assert qubo.metadata["num_slack_variables"] == 6
    assert qubo.num_variables == 14
    assert qubo.metadata["num_quadratic_terms"] == 22


def test_decode_and_validate_feasible_qubo_sample():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))
    sample = np.zeros(qubo.num_variables, dtype=int)
    for index, variable in enumerate(qubo.variables):
        if variable.get("variable_type") == "assignment" and (variable["job_id"], variable["start"]) in {
            ("j1", 0),
            ("j2", 1),
        }:
            sample[index] = 1

    schedule = decode_qubo_sample(qubo, sample)
    report = validate_decoded_schedule(schedule, _jobs(), _clusters(), [0, 1])

    assert report["feasible"] is True
    assert len(schedule) == 2


def test_gpu_slack_bits_reduce_capacity_penalty():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))
    sample = np.zeros(qubo.num_variables, dtype=int)
    for index, variable in enumerate(qubo.variables):
        if variable.get("variable_type") == "assignment" and (variable["job_id"], variable["start"]) in {
            ("j1", 0),
            ("j2", 1),
        }:
            sample[index] = 1

    missing_slack_energy = qubo_energy(qubo, sample)
    for index, variable in enumerate(qubo.variables):
        if variable.get("variable_type") == "gpu_slack" and variable["bit"] == 0:
            sample[index] = 1

    assert qubo_energy(qubo, sample) < missing_slack_energy


def test_validate_decoded_schedule_catches_missing_assignment():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))
    sample = np.zeros(qubo.num_variables, dtype=int)
    schedule = decode_qubo_sample(qubo, sample)
    report = validate_decoded_schedule(schedule, _jobs(), _clusters(), [0, 1])

    assert report["feasible"] is False
    assert report["violations"]
