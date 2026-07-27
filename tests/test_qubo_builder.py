import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.quantum.qubo_builder import build_scheduling_qubo, decode_qubo_sample, validate_decoded_schedule


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


def test_build_scheduling_qubo_creates_assignment_variables():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))

    assert qubo.num_variables == 4
    assert len(qubo.linear) == 4
    assert qubo.metadata["num_quadratic_terms"] > 0


def test_decode_and_validate_feasible_qubo_sample():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))
    sample = np.zeros(qubo.num_variables, dtype=int)
    for index, variable in enumerate(qubo.variables):
        if (variable["job_id"], variable["start"]) in {("j1", 0), ("j2", 1)}:
            sample[index] = 1

    schedule = decode_qubo_sample(qubo, sample)
    report = validate_decoded_schedule(schedule, _jobs(), _clusters(), [0, 1])

    assert report["feasible"] is True
    assert len(schedule) == 2


def test_validate_decoded_schedule_catches_missing_assignment():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))
    sample = np.zeros(qubo.num_variables, dtype=int)
    schedule = decode_qubo_sample(qubo, sample)
    report = validate_decoded_schedule(schedule, _jobs(), _clusters(), [0, 1])

    assert report["feasible"] is False
    assert report["violations"]
