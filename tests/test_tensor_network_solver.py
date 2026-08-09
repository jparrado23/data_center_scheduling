from itertools import product

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.quantum.qubo_builder import build_scheduling_qubo, decode_qubo_sample, qubo_energy, validate_decoded_schedule
from src.quantum.tensor_network import benchmark_tensor_network_solver, greedy_min_degree_order, solve_qubo_tensor_network


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


def _solve_bruteforce(qubo):
    best_energy = float("inf")
    best_sample = None
    for bits in product([0, 1], repeat=qubo.num_variables):
        sample = np.array(bits, dtype=int)
        energy = qubo_energy(qubo, sample)
        if energy < best_energy:
            best_energy = energy
            best_sample = sample
    return best_energy, best_sample


def test_tensor_network_solver_matches_bruteforce_energy():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))

    brute_energy, _ = _solve_bruteforce(qubo)
    result = solve_qubo_tensor_network(qubo)

    assert result.energy == brute_energy
    assert qubo_energy(qubo, result.sample) == brute_energy
    assert result.max_factor_rank >= 2
    assert result.max_factor_size >= 4


def test_tensor_network_solution_decodes_to_feasible_schedule():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))

    result = solve_qubo_tensor_network(qubo)
    schedule = decode_qubo_sample(qubo, result.sample)
    report = validate_decoded_schedule(schedule, _jobs(), _clusters(), [0, 1])

    assert report["feasible"] is True
    assert len(schedule) == len(_jobs())


def test_tensor_network_respects_intermediate_size_budget():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))
    order = greedy_min_degree_order([], qubo.num_variables)

    try:
        solve_qubo_tensor_network(qubo, contraction_order=order, max_intermediate_entries=1)
    except ValueError as exc:
        assert "max_intermediate_entries" in str(exc)
    else:
        raise AssertionError("expected contraction budget to fail")


def test_tensor_network_benchmark_includes_bruteforce_reference_for_small_qubo():
    qubo = build_scheduling_qubo(_jobs(), _hourly(), _clusters(), ModelConfig(delta_t=1.0))

    metrics = benchmark_tensor_network_solver(qubo)

    assert metrics["bruteforce_status"] == "completed"
    assert metrics["matches_bruteforce"] is True
    assert metrics["energy"] == metrics["bruteforce_energy"]
