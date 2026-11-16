"""Tests for term-level PCE penalty diagnostics."""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

import src.quantum.pce as pce_module
from src.config import ModelConfig
from src.quantum.pce import FixedAlphaConfig
from src.quantum.pce_diagnostics import (
    run_fixed_alpha_penalty_trace,
    split_scheduling_qubo,
)
from src.quantum.qubo_builder import (
    QuboPenaltyWeights,
    build_scheduling_qubo,
    qubo_energy,
)


def _inputs():
    """Return two competing jobs and one two-GPU cluster."""

    jobs = pd.DataFrame([
        {
            "job_id": job_id,
            "category": "gpu",
            "duration": 1,
            "power": 0.01,
            "earliest_start": 0,
            "latest_start": 1,
            "gpu_count_required": 1,
            "compatible_clusters": "c0",
            "cpu_required": 0.0,
            "memory_required_gb": 0.0,
        }
        for job_id in ("j0", "j1")
    ])
    hourly = pd.DataFrame({
        "hour": [0, 1],
        "grid_price": [20.0, 80.0],
        "renewable_available": [0.0, 0.0],
        "pue": [1.25, 1.25],
    })
    clusters = pd.DataFrame([{
        "cluster_id": "c0",
        "capacity": 0.02,
        "gpu_count": 2,
        "gpu_capacity": 2,
        "cpu_capacity": 1.0,
        "memory_capacity_gb": 1.0,
    }])
    return jobs, hourly, clusters, ModelConfig(delta_t=1.0, pue=1.25)


def test_penalty_components_reconstruct_binary_and_relaxed_objectives():
    """The trace must use the exact coefficients and variable order of the QUBO."""

    jobs, hourly, clusters, config = _inputs()
    weights = QuboPenaltyWeights(assignment=1000.0, gpu_capacity=1.0, peak_smoothing=0.0)
    qubo = build_scheduling_qubo(jobs, hourly, clusters, config, weights)
    components = split_scheduling_qubo(qubo, hourly, clusters, config, weights)

    bits = np.array([1, 0, 0, 1] + [0] * qubo.metadata["num_slack_variables"])
    assert qubo_energy(qubo, bits) == pytest.approx(
        sum(qubo_energy(term, bits) for term in components.values())
    )

    evaluator_calls = []

    def correlations(theta):
        """Return deterministic correlations and track the terminal re-evaluation."""

        evaluator_calls.append(np.asarray(theta, dtype=float).copy())
        return np.full(qubo.num_variables, np.tanh(theta[0] + 0.2))

    result = run_fixed_alpha_penalty_trace(
        0.5,
        np.array([0.0]),
        correlations,
        qubo,
        jobs,
        clusters,
        [0, 1],
        components,
        FixedAlphaConfig(maxiter=4),
    )
    assert len(result.evaluations) == len(result.run.history) + 1
    assert len(result.candidates) == len(result.run.history) + 1
    assert len(evaluator_calls) == len(result.run.history) + 1
    assert result.evaluations.iloc[-1]["evaluation_type"] == "post_optimizer_final"
    assert result.candidates[-1].theta == pytest.approx(result.run.optimization.x)
    assert np.allclose(
        result.evaluations["relaxed_total"],
        result.evaluations[["relaxed_energy", "relaxed_assignment", "relaxed_gpu"]].sum(axis=1),
    )
    assert np.allclose(
        result.run.history["qubo_energy"],
        result.evaluations.iloc[:-1][
            ["decoded_energy", "decoded_assignment", "decoded_gpu"]
        ].sum(axis=1),
    )


def test_penalty_components_require_disabled_peak_proxy():
    """A peak term cannot silently disappear from the diagnostic decomposition."""

    jobs, hourly, clusters, config = _inputs()
    weights = QuboPenaltyWeights(assignment=1000.0, gpu_capacity=1.0, peak_smoothing=1.0)
    qubo = build_scheduling_qubo(jobs, hourly, clusters, config, weights)
    with pytest.raises(ValueError, match="peak smoothing"):
        split_scheduling_qubo(qubo, hourly, clusters, config, weights)


def test_trace_retains_a_feasible_terminal_sample_missing_from_optimizer_history(monkeypatch):
    """The post-optimizer measurement must count even if objective samples were infeasible."""

    jobs, hourly, clusters, config = _inputs()
    weights = QuboPenaltyWeights(assignment=1000.0, gpu_capacity=0.0, peak_smoothing=0.0)
    qubo = build_scheduling_qubo(jobs, hourly, clusters, config, weights)
    components = split_scheduling_qubo(qubo, hourly, clusters, config, weights)
    calls = 0

    def evaluator(_theta):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.ones(qubo.num_variables)
        return np.array([1.0, -1.0, -1.0, 1.0])

    def one_objective_evaluation(objective, initial_theta, **_kwargs):
        objective(initial_theta)
        return SimpleNamespace(x=np.asarray(initial_theta), success=True, nfev=1)

    monkeypatch.setattr(pce_module, "minimize", one_objective_evaluation)
    trace = run_fixed_alpha_penalty_trace(
        1.0,
        np.array([0.0]),
        evaluator,
        qubo,
        jobs,
        clusters,
        [0, 1],
        components,
        FixedAlphaConfig(maxiter=1),
    )

    assert len(trace.run.history) == 1
    assert len(trace.candidates) == 2
    assert not trace.candidates[0].validation["feasible"]
    assert trace.candidates[-1].validation["feasible"]
    assert trace.run.selected_candidate.validation["feasible"]
    assert trace.evaluations.iloc[-1]["evaluation_type"] == "post_optimizer_final"
    assert bool(trace.evaluations.iloc[-1]["schedule_feasible"])
