"""Tests for the reusable simulated-annealing QUBO runner."""

from __future__ import annotations

import pandas as pd

from src.quantum.annealing import _energies_are_consistent, run_simulated_annealing
from src.quantum.qubo_builder import SchedulingQubo


def test_energy_consistency_allows_large_qubo_roundoff() -> None:
    """Different summation orders may produce harmless micro-scale drift."""

    assert _energies_are_consistent(2189.7090151999146, 2189.7090075755477)
    assert not _energies_are_consistent(2189.7190151999146, 2189.7090075755477)


def test_simulated_annealing_returns_feasible_lowest_energy_schedule() -> None:
    """A one-variable assignment QUBO decodes to the only feasible schedule."""

    qubo = SchedulingQubo(
        linear={0: -1.0},
        variables=[
            {
                "variable_type": "assignment",
                "job_id": "job_0",
                "cluster": "cluster_0",
                "start": 0,
                "duration": 1,
                "power": 1.0,
                "gpu_count_required": 1.0,
                "cpu_required": 0.0,
                "memory_required_gb": 0.0,
            }
        ],
    )
    jobs = pd.DataFrame(
        [{
            "job_id": "job_0", "earliest_start": 0, "latest_start": 0,
            "duration": 1, "power": 1.0, "gpu_count_required": 1.0,
            "cpu_required": 0.0, "memory_required_gb": 0.0,
        }]
    )
    clusters = pd.DataFrame(
        [{"cluster_id": "cluster_0", "gpu_count": 1, "cpu_capacity": 1.0, "memory_capacity_gb": 1.0}]
    )

    result = run_simulated_annealing(
        qubo, jobs, clusters, [0], num_reads=10, seed=7
    )

    assert result.validation["feasible"]
    assert result.sample == {0: 1}
    assert result.feasible_sample_count >= 1
