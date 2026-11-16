"""Tests for physical GPU validation and duplicate-only PCE repair."""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.quantum.pce_repair import (
    physical_gpu_summary,
    repair_duplicate_only_schedule,
    select_direct_or_repaired_schedule,
    trace_physical_diagnostics,
)


def _raw_inputs():
    jobs = pd.DataFrame([
        {
            "job_id": "j0",
            "earliest_start": 0,
            "latest_start": 1,
            "duration": 1,
            "raw_gpu_num": 80,
            "gpu_count_required": 1,
            "compatible_clusters": "c0",
            "cpu_required": 0.0,
            "memory_required_gb": 0.0,
        },
        {
            "job_id": "j1",
            "earliest_start": 0,
            "latest_start": 1,
            "duration": 1,
            "raw_gpu_num": 80,
            "gpu_count_required": 1,
            "compatible_clusters": "c0",
            "cpu_required": 0.0,
            "memory_required_gb": 0.0,
        },
    ])
    clusters = pd.DataFrame([{
        "cluster_id": "c0",
        "raw_gpu_count": 128,
        "gpu_count": 2,
        "gpu_capacity": 2,
        "cpu_capacity": 0.0,
        "memory_capacity_gb": 0.0,
    }])
    energy = pd.DataFrame({
        "hour": [0, 1],
        "effective_price": [80.0, 20.0],
        "grid_price": [80.0, 20.0],
        "renewable_available": [0.0, 0.0],
        "renewable_price": [10.0, 10.0],
        "pue": [1.25, 1.25],
    })
    return jobs, clusters, energy


def _row(job_id: str, start: int, variable: int) -> dict:
    return {
        "job_id": job_id,
        "assigned_cluster": "c0",
        "start_hour": start,
        "duration": 1,
        "power": 0.032,
        "gpu_count_required": 1,
        "cpu_required": 0.0,
        "memory_required_gb": 0.0,
        "qubo_variable": variable,
    }


def test_physical_validation_rejects_capacity_hidden_by_blocks():
    """Raw GPUs, not rounded block counts, determine final feasibility."""

    jobs, clusters, _ = _raw_inputs()
    schedule = pd.DataFrame([_row("j0", 0, 0), _row("j1", 0, 1)])

    summary = physical_gpu_summary(schedule, jobs, clusters, [0, 1])

    assert summary["missing_job_count"] == 0
    assert summary["duplicate_assignments"] == 0
    assert not summary["physical_gpu_feasible"]
    assert summary["physical_gpu_max_excess"] == pytest.approx(32.0)


def test_duplicate_only_repair_keeps_cheapest_start_and_is_feasible():
    """Removing duplicates must yield one raw-GPU-feasible assignment per job."""

    jobs, clusters, energy = _raw_inputs()
    clusters.loc[0, "raw_gpu_count"] = 200
    schedule = pd.DataFrame([
        _row("j0", 0, 0),
        _row("j0", 1, 1),
        _row("j1", 0, 2),
        _row("j1", 1, 3),
    ])

    repaired = repair_duplicate_only_schedule(
        schedule, jobs, clusters, energy, [0, 1], delta_t=1.0
    )
    summary = physical_gpu_summary(repaired, jobs, clusters, [0, 1])

    assert repaired.groupby("job_id").size().eq(1).all()
    assert repaired["start_hour"].tolist() == [1, 1]
    assert summary["fully_feasible"]


def test_selection_prefers_direct_feasibility_before_repair():
    """A repairable candidate cannot displace an already feasible PCE schedule."""

    jobs, clusters, energy = _raw_inputs()
    clusters.loc[0, "raw_gpu_count"] = 200
    repairable = pd.DataFrame([
        _row("j0", 0, 0), _row("j0", 1, 1),
        _row("j1", 0, 2), _row("j1", 1, 3),
    ])
    direct = pd.DataFrame([_row("j0", 0, 0), _row("j1", 1, 3)])
    trace = SimpleNamespace(candidates=[
        SimpleNamespace(schedule=repairable),
        SimpleNamespace(schedule=direct),
    ])

    selected = select_direct_or_repaired_schedule(
        trace, jobs, clusters, energy, [0, 1], delta_t=1.0
    )

    assert selected.status == "direct"
    assert selected.source_evaluation == 2
    assert selected.direct_candidate_count == 1
    assert selected.repairable_candidate_count == 1
    assert selected.removed_assignments == 0


def test_trace_diagnostics_separates_missing_and_repairable_candidates():
    """Trace diagnostics must expose the reason each theta point is rejected."""

    jobs, clusters, _ = _raw_inputs()
    clusters.loc[0, "raw_gpu_count"] = 200
    missing = pd.DataFrame([_row("j0", 0, 0)])
    repairable = pd.DataFrame([
        _row("j0", 0, 0), _row("j0", 1, 1), _row("j1", 1, 3)
    ])
    candidates = [
        SimpleNamespace(schedule=missing, pce_loss=2.0, qubo_energy=3.0),
        SimpleNamespace(schedule=repairable, pce_loss=1.0, qubo_energy=2.0),
    ]
    evaluations = pd.DataFrame({
        "relaxed_energy": [1.0, 1.0],
        "relaxed_assignment": [2.0, 1.0],
        "relaxed_gpu": [3.0, 2.0],
    })
    trace = SimpleNamespace(candidates=candidates, evaluations=evaluations)

    diagnostics = trace_physical_diagnostics(trace, jobs, clusters, [0, 1])

    assert diagnostics["category"].tolist() == [
        "missing_jobs", "duplicate_only_repairable"
    ]
    assert diagnostics["running_min_missing_jobs"].tolist() == [1, 0]
