"""Annealing runners for reduced scheduling QUBOs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.quantum.qubo_builder import (
    SchedulingQubo,
    decode_qubo_sample,
    qubo_energy,
    to_dimod_bqm,
    validate_decoded_schedule,
)


@dataclass
class AnnealingResult:
    """Best decoded sample returned by a simulated annealing run."""

    qubo_energy: float
    sample: dict[int, int]
    schedule: pd.DataFrame
    validation: dict[str, Any]
    runtime_s: float
    num_reads: int
    feasible_sample_count: int
    examined_sample_count: int


def run_simulated_annealing(
    qubo: SchedulingQubo,
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    hours: Sequence[int],
    *,
    num_reads: int = 1_000,
    seed: int | None = None,
) -> AnnealingResult:
    """Sample a QUBO and return the lowest-energy feasible decoded schedule.

    If no feasible sample is found, the lowest-energy returned sample is kept
    for diagnosis. Its validation result must not be treated as a valid
    scheduling outcome.
    """

    if num_reads < 1:
        raise ValueError("num_reads must be positive")
    try:
        from dwave.samplers import SimulatedAnnealingSampler
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'quantum' extra to use simulated annealing") from error

    started = perf_counter()
    sampleset = SimulatedAnnealingSampler().sample(
        to_dimod_bqm(qubo), num_reads=num_reads, seed=seed
    )
    runtime_s = perf_counter() - started
    best_raw: tuple[float, dict[int, int], pd.DataFrame, dict[str, Any]] | None = None
    best_feasible: tuple[float, dict[int, int], pd.DataFrame, dict[str, Any]] | None = None
    feasible_count = 0
    examined = 0
    for sample, energy in sampleset.data(["sample", "energy"]):
        examined += 1
        sample_dict = {int(index): int(value) for index, value in sample.items()}
        schedule = decode_qubo_sample(qubo, sample_dict)
        validation = validate_decoded_schedule(schedule, jobs_df, clusters_df, list(hours))
        record = (float(energy), sample_dict, schedule, validation)
        if best_raw is None:
            best_raw = record
        if validation["feasible"]:
            feasible_count += 1
            if best_feasible is None or float(energy) < best_feasible[0]:
                best_feasible = record

    if best_raw is None:
        raise RuntimeError("simulated annealing returned no samples")
    energy, sample, schedule, validation = best_feasible or best_raw
    direct_energy = qubo_energy(qubo, sample)
    if not np.isclose(energy, direct_energy, rtol=1e-9, atol=1e-7):
        raise RuntimeError(
            f"annealer energy {energy} disagrees with direct QUBO energy {direct_energy}"
        )
    return AnnealingResult(
        qubo_energy=direct_energy,
        sample=sample,
        schedule=schedule,
        validation=validation,
        runtime_s=runtime_s,
        num_reads=num_reads,
        feasible_sample_count=feasible_count,
        examined_sample_count=examined,
    )
