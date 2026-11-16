"""Tensor-network solvers for scheduling QUBOs.

The QUBO is represented as an energy factor graph with one binary tensor index
per QUBO variable. Exact contraction is implemented with variable elimination:
factors in each bucket are combined, then the selected index is minimized out.
This is a zero-temperature tensor-network contraction for the ground state.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any

import numpy as np

from src.quantum.qubo_builder import SchedulingQubo, qubo_energy


@dataclass(frozen=True)
class TensorFactor:
    """Dense energy tensor over a small ordered set of binary variables."""

    scope: tuple[int, ...]
    table: np.ndarray


@dataclass(frozen=True)
class TensorNetworkSolveResult:
    """Result returned by exact tensor-network QUBO contraction."""

    energy: float
    sample: np.ndarray
    elapsed_seconds: float
    contraction_order: tuple[int, ...]
    max_factor_rank: int
    max_factor_size: int
    method: str = "exact_min_sum_variable_elimination"


@dataclass(frozen=True)
class _EliminationRecord:
    variable: int
    remaining_scope: tuple[int, ...]
    argmin_table: np.ndarray


def qubo_to_energy_factors(qubo: SchedulingQubo) -> list[TensorFactor]:
    """Convert linear and quadratic QUBO terms into local energy tensors."""

    factors: list[TensorFactor] = []
    for index, coefficient in sorted(qubo.linear.items()):
        table = np.array([0.0, float(coefficient)], dtype=float)
        factors.append(TensorFactor(scope=(int(index),), table=table))

    for (left, right), coefficient in sorted(qubo.quadratic.items()):
        table = np.zeros((2, 2), dtype=float)
        table[1, 1] = float(coefficient)
        factors.append(TensorFactor(scope=(int(left), int(right)), table=table))

    return factors


def greedy_min_degree_order(factors: list[TensorFactor], num_variables: int) -> tuple[int, ...]:
    """Build a simple contraction order from the current factor graph."""

    neighbors: dict[int, set[int]] = {index: set() for index in range(num_variables)}
    for factor in factors:
        for left_position, left in enumerate(factor.scope):
            for right in factor.scope[left_position + 1 :]:
                neighbors[left].add(right)
                neighbors[right].add(left)

    order: list[int] = []
    remaining = set(range(num_variables))
    while remaining:
        variable = min(remaining, key=lambda item: (len(neighbors[item] & remaining), item))
        order.append(variable)
        active_neighbors = neighbors[variable] & remaining
        for left in active_neighbors:
            for right in active_neighbors:
                if left != right:
                    neighbors[left].add(right)
        remaining.remove(variable)
    return tuple(order)


def solve_qubo_tensor_network(
    qubo: SchedulingQubo,
    contraction_order: list[int] | tuple[int, ...] | None = None,
    max_intermediate_entries: int | None = None,
) -> TensorNetworkSolveResult:
    """Solve a QUBO exactly with min-sum tensor-network contraction.

    The method is exact but exponential in the induced width of the chosen
    contraction order. ``max_intermediate_entries`` can be set to fail fast when
    a contraction would create a dense tensor larger than the desired budget.
    """

    started = perf_counter()
    factors = qubo_to_energy_factors(qubo)
    order = tuple(contraction_order) if contraction_order is not None else greedy_min_degree_order(factors, qubo.num_variables)
    _validate_order(order, qubo.num_variables)

    records: list[_EliminationRecord] = []
    max_factor_rank = max((len(factor.scope) for factor in factors), default=0)
    max_factor_size = max((int(factor.table.size) for factor in factors), default=1)

    for variable in order:
        bucket = [factor for factor in factors if variable in factor.scope]
        factors = [factor for factor in factors if variable not in factor.scope]
        if not bucket:
            records.append(_EliminationRecord(variable=variable, remaining_scope=(), argmin_table=np.array(0, dtype=np.int8)))
            continue

        combined_scope = _combined_scope(bucket)
        combined = _sum_aligned_factors(bucket, combined_scope)
        max_factor_rank = max(max_factor_rank, len(combined_scope))
        max_factor_size = max(max_factor_size, int(combined.size))
        if max_intermediate_entries is not None and combined.size > max_intermediate_entries:
            raise ValueError(
                f"contraction for variable {variable} would create {combined.size} entries, "
                f"exceeding max_intermediate_entries={max_intermediate_entries}"
            )

        axis = combined_scope.index(variable)
        remaining_scope = tuple(item for item in combined_scope if item != variable)
        minimized = np.min(combined, axis=axis)
        argmin_table = np.argmin(combined, axis=axis).astype(np.int8)
        records.append(_EliminationRecord(variable=variable, remaining_scope=remaining_scope, argmin_table=argmin_table))
        if remaining_scope:
            factors.append(TensorFactor(scope=remaining_scope, table=np.asarray(minimized, dtype=float)))
        else:
            factors.append(TensorFactor(scope=(), table=np.asarray(minimized, dtype=float)))

    residual_energy = float(sum(float(np.asarray(factor.table)) for factor in factors))
    sample = _reconstruct_sample(records, qubo.num_variables)
    energy = float(qubo.offset + residual_energy)

    # Numerical contraction and direct QUBO evaluation should agree; use the
    # direct value as the reported energy to keep downstream comparisons stable.
    checked_energy = qubo_energy(qubo, sample)
    if not np.isclose(energy, checked_energy, rtol=1e-9, atol=1e-7):
        raise RuntimeError(f"tensor contraction energy {energy} disagrees with QUBO energy {checked_energy}")

    return TensorNetworkSolveResult(
        energy=checked_energy,
        sample=sample,
        elapsed_seconds=perf_counter() - started,
        contraction_order=order,
        max_factor_rank=max_factor_rank,
        max_factor_size=max_factor_size,
    )


def benchmark_tensor_network_solver(
    qubo: SchedulingQubo,
    brute_force_max_variables: int = 24,
    max_intermediate_entries: int | None = None,
) -> dict[str, Any]:
    """Run the tensor solver and return performance metrics.

    For small QUBOs, brute force is also run as an exact correctness reference.
    Larger QUBOs skip brute force automatically because enumeration scales as
    ``2 ** num_variables``.
    """

    result = solve_qubo_tensor_network(qubo, max_intermediate_entries=max_intermediate_entries)
    metrics: dict[str, Any] = {
        "method": result.method,
        "num_variables": qubo.num_variables,
        "num_linear_terms": len(qubo.linear),
        "num_quadratic_terms": len(qubo.quadratic),
        "energy": result.energy,
        "elapsed_seconds": result.elapsed_seconds,
        "max_factor_rank": result.max_factor_rank,
        "max_factor_size": result.max_factor_size,
        "contraction_order": result.contraction_order,
        "sample": result.sample,
    }
    if qubo.num_variables > brute_force_max_variables:
        metrics["bruteforce_status"] = "skipped"
        metrics["bruteforce_reason"] = f"{qubo.num_variables} variables exceeds limit {brute_force_max_variables}"
        return metrics

    started = perf_counter()
    brute_energy, brute_sample = _solve_bruteforce(qubo)
    metrics["bruteforce_status"] = "completed"
    metrics["bruteforce_energy"] = brute_energy
    metrics["bruteforce_elapsed_seconds"] = perf_counter() - started
    metrics["matches_bruteforce"] = bool(np.isclose(result.energy, brute_energy, rtol=1e-9, atol=1e-7))
    metrics["bruteforce_sample"] = brute_sample
    return metrics


def _validate_order(order: tuple[int, ...], num_variables: int) -> None:
    expected = set(range(num_variables))
    actual = set(order)
    if actual != expected or len(order) != num_variables:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"contraction_order must contain every variable exactly once; missing={missing}, extra={extra}")


def _solve_bruteforce(qubo: SchedulingQubo) -> tuple[float, np.ndarray]:
    best_energy = float("inf")
    best_sample = np.zeros(qubo.num_variables, dtype=int)
    for bits in product([0, 1], repeat=qubo.num_variables):
        sample = np.array(bits, dtype=int)
        energy = qubo_energy(qubo, sample)
        if energy < best_energy:
            best_energy = energy
            best_sample = sample
    return float(best_energy), best_sample


def _combined_scope(factors: list[TensorFactor]) -> tuple[int, ...]:
    return tuple(sorted({variable for factor in factors for variable in factor.scope}))


def _sum_aligned_factors(factors: list[TensorFactor], target_scope: tuple[int, ...]) -> np.ndarray:
    combined = np.zeros((2,) * len(target_scope), dtype=float)
    for factor in factors:
        combined = combined + _align_factor(factor, target_scope)
    return combined


def _align_factor(factor: TensorFactor, target_scope: tuple[int, ...]) -> np.ndarray:
    shape = [1] * len(target_scope)
    for axis, variable in enumerate(factor.scope):
        shape[target_scope.index(variable)] = factor.table.shape[axis]
    return factor.table.reshape(shape)


def _reconstruct_sample(records: list[_EliminationRecord], num_variables: int) -> np.ndarray:
    assignment: dict[int, int] = {}
    for record in reversed(records):
        if record.remaining_scope:
            key: Any = tuple(assignment[variable] for variable in record.remaining_scope)
            value = int(record.argmin_table[key])
        else:
            value = int(np.asarray(record.argmin_table))
        assignment[record.variable] = value
    return np.array([assignment.get(index, 0) for index in range(num_variables)], dtype=int)
