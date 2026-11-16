"""Inspect how PCE's full loss responds to scheduling penalty weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.config import ModelConfig
from src.quantum.pce import (
    CorrelationEvaluator,
    FixedAlphaConfig,
    FixedAlphaRun,
    PceCandidate,
    decode_soft_spins,
    ising_energy,
    pce_loss_from_correlations,
    qubo_to_ising,
    run_fixed_alpha,
)
from src.quantum.qubo_builder import (
    QuboPenaltyWeights,
    SchedulingQubo,
    add_assignment_constraint_penalty,
    add_fixed_pue_energy_cost,
    add_gpu_capacity_constraint_penalty,
    build_hourly_lookup,
    decode_qubo_sample,
    qubo_energy,
    validate_decoded_schedule,
)


@dataclass
class PcePenaltyTrace:
    """A fixed-alpha run, all objective samples, and its terminal sample."""

    run: FixedAlphaRun
    evaluations: pd.DataFrame
    components: dict[str, SchedulingQubo]
    candidates: list[PceCandidate]
    includes_terminal_sample: bool = False


def split_scheduling_qubo(
    qubo: SchedulingQubo,
    hourly_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    config: ModelConfig,
    weights: QuboPenaltyWeights,
) -> dict[str, SchedulingQubo]:
    """Rebuild separate QUBOs with exactly the same variable indices as ``qubo``."""

    if weights.peak_smoothing != 0:
        raise ValueError("This diagnostic requires peak smoothing to be disabled")
    assignment_count = int(qubo.metadata["num_assignment_variables"])
    energy = SchedulingQubo(variables=list(qubo.variables))
    assignment = SchedulingQubo(variables=list(qubo.variables))
    gpu = SchedulingQubo(variables=list(qubo.variables[:assignment_count]))

    add_fixed_pue_energy_cost(energy, build_hourly_lookup(hourly_df, config), config)
    add_assignment_constraint_penalty(assignment, weights.assignment)
    add_gpu_capacity_constraint_penalty(
        gpu, clusters_df, list(qubo.metadata["hours"]), weights.gpu_capacity
    )
    if gpu.variables != qubo.variables:
        raise ValueError("GPU term changed the QUBO variable order")

    components = {"energy": energy, "assignment": assignment, "gpu": gpu}
    if not np.isclose(sum(term.offset for term in components.values()), qubo.offset):
        raise ValueError("Separated offsets do not reconstruct the full QUBO")
    for attribute in ("linear", "quadratic"):
        source = getattr(qubo, attribute)
        keys = set(source).union(*(getattr(term, attribute) for term in components.values()))
        for key in keys:
            coefficient = sum(getattr(term, attribute).get(key, 0.0) for term in components.values())
            if not np.isclose(coefficient, source.get(key, 0.0), rtol=1e-10, atol=1e-6):
                raise ValueError(f"Separated {attribute} coefficients do not reconstruct the QUBO")
    return components


def run_fixed_alpha_penalty_trace(
    alpha: float,
    initial_theta: np.ndarray,
    evaluator: CorrelationEvaluator,
    qubo: SchedulingQubo,
    jobs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    hours: Sequence[int],
    components: dict[str, SchedulingQubo],
    config: FixedAlphaConfig,
) -> PcePenaltyTrace:
    """Record each loss component at every optimizer evaluation."""

    correlations: list[np.ndarray] = []
    parameter_points: list[np.ndarray] = []

    def tracked_evaluator(theta: np.ndarray) -> np.ndarray:
        """Capture correlations without re-evaluating a statevector later."""

        values = np.asarray(evaluator(theta), dtype=float)
        parameter_points.append(np.asarray(theta, dtype=float).copy())
        correlations.append(values.copy())
        return values

    run = run_fixed_alpha(
        alpha, initial_theta, tracked_evaluator, qubo, jobs_df, clusters_df, hours, config
    )
    models = {name: qubo_to_ising(term) for name, term in components.items()}
    full_model = qubo_to_ising(qubo)
    job_options: dict[str, list[int]] = {str(job_id): [] for job_id in jobs_df["job_id"]}
    for index, variable in enumerate(qubo.variables):
        if variable["variable_type"] == "assignment":
            job_options[variable["job_id"]].append(index)

    records: list[dict] = []
    candidates: list[PceCandidate] = []

    def record_candidate(
        evaluation: int,
        evaluation_type: str,
        theta: np.ndarray,
        observed_correlations: np.ndarray,
        expected_qubo_energy: float | None = None,
    ) -> None:
        """Decode and retain one measured point, including the terminal check."""

        soft_spins, full_loss = pce_loss_from_correlations(
            observed_correlations, alpha, full_model
        )
        bits = decode_soft_spins(soft_spins)
        schedule = decode_qubo_sample(qubo, bits)
        validation = validate_decoded_schedule(
            schedule, jobs_df, clusters_df, list(hours)
        )
        binary_energy = float(qubo_energy(qubo, bits))
        if expected_qubo_energy is not None and not np.isclose(
            binary_energy, expected_qubo_energy, rtol=1e-10, atol=1e-6
        ):
            raise ValueError("decoded QUBO energy does not match the optimizer trace")
        candidates.append(
            PceCandidate(
                theta=theta,
                alpha=float(alpha),
                correlations=observed_correlations,
                soft_spins=soft_spins,
                bits=bits,
                pce_loss=float(full_loss),
                qubo_energy=binary_energy,
                schedule=schedule,
                validation=validation,
            )
        )
        relaxed = {name: float(ising_energy(soft_spins, model)) for name, model in models.items()}
        decoded = {name: qubo_energy(term, bits) for name, term in components.items()}
        counts = [int(bits[options].sum()) for options in job_options.values()]
        if not np.isclose(sum(relaxed.values()), full_loss, rtol=1e-10, atol=1e-6):
            raise ValueError("Relaxed components do not reconstruct the PCE loss")
        if not np.isclose(sum(decoded.values()), binary_energy, rtol=1e-10, atol=1e-6):
            raise ValueError("Decoded components do not reconstruct the QUBO energy")
        records.append(
            {
                "evaluation": evaluation,
                "evaluation_type": evaluation_type,
                "relaxed_total": full_loss,
                **{f"relaxed_{name}": value for name, value in relaxed.items()},
                **{f"decoded_{name}": value for name, value in decoded.items()},
                "schedule_feasible": bool(validation["feasible"]),
                "missing_jobs": sum(count == 0 for count in counts),
                "duplicate_jobs": sum(max(0, count - 1) for count in counts),
            }
        )

    for index, previous in run.history.iterrows():
        record_candidate(
            evaluation=int(previous.evaluation),
            evaluation_type="optimizer",
            theta=parameter_points[index],
            observed_correlations=correlations[index],
            expected_qubo_energy=float(previous.qubo_energy),
        )

    # run_fixed_alpha evaluates optimization.x once more after COBYLA terminates.
    # With finite-shot estimators this is a new sample and can decode differently
    # from the optimizer's last objective call, so retain it in feasibility checks.
    terminal_index = len(run.history)
    if len(correlations) != terminal_index + 1:
        raise ValueError(
            "expected one post-optimizer evaluator call in addition to optimizer history"
        )
    record_candidate(
        evaluation=terminal_index + 1,
        evaluation_type="post_optimizer_final",
        theta=parameter_points[terminal_index],
        observed_correlations=correlations[terminal_index],
    )

    return PcePenaltyTrace(
        run, pd.DataFrame.from_records(records), components, candidates,
        includes_terminal_sample=True,
    )
