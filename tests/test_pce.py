"""Tests for backend-independent PCE scheduling helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.quantum.pce import (
    FixedAlphaConfig,
    ProgressiveBinarizationConfig,
    evaluate_candidate,
    homogeneous_pauli_labels,
    minimum_pce_qubits,
    pauli_labels_for_qubo,
    qubo_to_ising,
    run_fixed_alpha_grid,
    run_progressive_binarization,
)
from src.quantum.qubo_builder import SchedulingQubo


def _single_job_inputs() -> tuple[SchedulingQubo, pd.DataFrame, pd.DataFrame]:
    """Create the smallest decoded scheduling example for PCE tests."""

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
        metadata={"hours": [0]},
    )
    jobs = pd.DataFrame(
        [{
            "job_id": "job_0",
            "earliest_start": 0,
            "latest_start": 0,
            "duration": 1,
            "power": 1.0,
            "gpu_count_required": 1.0,
            "cpu_required": 0.0,
            "memory_required_gb": 0.0,
        }]
    )
    clusters = pd.DataFrame(
        [{
            "cluster_id": "cluster_0",
            "gpu_count": 1,
            "cpu_capacity": 1.0,
            "memory_capacity_gb": 1.0,
        }]
    )
    return qubo, jobs, clusters


def test_pauli_allocation_and_ising_conversion() -> None:
    """Order-two allocation and Ising conversion preserve basic QUBO structure."""

    qubo, _, _ = _single_job_inputs()
    qubits, labels = pauli_labels_for_qubo(qubo)
    model = qubo_to_ising(qubo)

    assert minimum_pce_qubits(3) == 2
    assert homogeneous_pauli_labels(2) == ["XX", "YY", "ZZ"]
    assert qubits == 2
    assert labels == ["XX"]
    assert model.fields.tolist() == [-0.5]
    assert model.constant == -0.5


def test_fixed_alpha_and_progressive_binarization_keep_feasible_candidate() -> None:
    """Both PCE runners retain a feasible decoded schedule from correlations."""

    qubo, jobs, clusters = _single_job_inputs()

    def correlations(_: np.ndarray) -> np.ndarray:
        return np.array([1.0])

    candidate = evaluate_candidate(
        np.array([0.0]), 3.0, correlations, qubo, jobs, clusters, [0]
    )
    assert candidate.validation["feasible"]
    assert candidate.bits.tolist() == [1]

    fixed_config = FixedAlphaConfig(maxiter=3)
    runs, selected = run_fixed_alpha_grid(
        [1.0, 3.0], np.array([0.0]), correlations, qubo, jobs, clusters,
        [0], fixed_config,
    )
    assert len(runs) == 2
    assert selected.selected_candidate.validation["feasible"]

    pb_result = run_progressive_binarization(
        np.array([0.0]), correlations, qubo, jobs, clusters, [0],
        ProgressiveBinarizationConfig(
            initial_alpha=3.0,
            max_stages=2,
            fixed_alpha=fixed_config,
        ),
    )
    assert pb_result.selected_candidate.validation["feasible"]
    assert len(pb_result.stages) == 1
